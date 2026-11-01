#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Do the pick and place with Gemini Robotics-ER 2 instead of a learned policy.

Each attempt runs four phases in order:
    1. Recognize: locate the cube in the ceiling camera's image.
    2. Decide: from the images, pick up the cube, finish or give up.
    3. Plan: Gemini calls the motions of scripted_pick_place.py one by one.
    4. Judge: from the images, did the task succeed?

Gemini gets only what a real robot has: camera images and the arm's own joints. No
decision reads the simulator's true state: it is only recorded, and scores the result
after everything has finished.

Run from the training directory with its virtual environment:
    .venv/Scripts/python.exe ../scripts/gemini_orchestrate.py --seed 3000 \
        --output outputs/orchestrate/try
"""

import argparse
import enum
import json
import pathlib

import cv2
import imageio_ffmpeg
import numpy as np
from evaluate_act import FPS, JUDGE_CAMERAS, Scene
from gemini_judge import GeminiJudge
from google.genai import types
from pydantic import BaseModel
from scripted_pick_place import ScriptedArm

MAX_PLAN_STEPS = 24
DECIDE_PROMPT = """These images show a robot table, one image per camera: {cameras}.
The right arm has to pick up the orange cube and place it inside the black square tray.
The cube was located at x={x:.3f} y={y:.3f} in meters. The right hand can pick it up
when x is between 0.42 and 0.66 and y is between -0.24 and 0.20.

Choose one action:
- pick: the cube is outside the tray and the right hand can pick it up
- done: the cube is already resting inside the tray
- stop: the cube can't be picked up, for example if it is out of reach or not visible"""
PLAN_PROMPT = """You control the right arm of a robot through the given functions.
Task: pick up the orange cube and place it inside the black square tray.
The cube is at x={x:.3f} y={y:.3f} in meters, as located in the ceiling camera's image.

Work out the order of the motions yourself and call one function at a time. Each result
gives the arm's state and is followed by the ceiling camera's image, so check that a step
worked before the next one. The gripper joint stops near -0.2 when holding the cube and
closes to about -0.05 when empty; -0.785 is fully open.
When the arm has let go of the cube and moved clear, or if you can't go on, stop calling
functions and say what happened in one sentence."""
PUSH_PROMPT = """
Close to the tray, its walls can block the hand so that the fingers don't close properly.
If the cube is there or a grasp has failed, don't repeat the same grasp: push the cube with
push_cube, then call locate_cube for its new position before grasping. Grasps work best
near x=0.53 y=-0.025."""


class Action(enum.Enum):
    PICK = "pick"
    DONE = "done"
    STOP = "stop"


class Decision(BaseModel):
    observation: str
    action: Action


def position_parameters():
    return types.Schema(
        type="OBJECT",
        properties={
            "x": types.Schema(type="NUMBER", description="the cube's x in meters"),
            "y": types.Schema(type="NUMBER", description="the cube's y in meters"),
        },
        required=["x", "y"],
    )


def declare(name, description, with_position=False):
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=position_parameters() if with_position else None,
    )


FUNCTIONS = [
    declare(
        "move_above_cube",
        "Open the hand and go to a point above and behind the cube. Also moves the left arm "
        "away from the tray.",
        True,
    ),
    declare("descend", "Come down to the table's height behind the cube, without touching it.", True),
    declare("approach", "Slide the open fingers forward so that they are on both sides of the cube.", True),
    declare("close_gripper", "Close the fingers."),
    declare("lift", "Raise the hand 15 cm straight up."),
    declare("move_above_tray", "Carry what the hand holds to above the middle of the tray."),
    declare("open_gripper", "Open the fingers, letting go of what the hand holds."),
    declare("retreat", "Pull the hand back and return both arms to their home pose."),
]
PUSH_FUNCTIONS = [
    types.FunctionDeclaration(
        name="push_cube",
        description="Push the cube along the table with the closed gripper. It may stop short of "
        "the target or go at an angle.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "x": types.Schema(type="NUMBER", description="the cube's x in meters"),
                "y": types.Schema(type="NUMBER", description="the cube's y in meters"),
                "target_x": types.Schema(type="NUMBER", description="where to push it, x in meters"),
                "target_y": types.Schema(type="NUMBER", description="where to push it, y in meters"),
            },
            required=["x", "y", "target_x", "target_y"],
        ),
    ),
    declare(
        "locate_cube",
        "Move the hand aside so that the ceiling camera sees the table, and return the cube's "
        "position in meters.",
    ),
]


def jpeg_part(image):
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode an image")
    return types.Part.from_bytes(data=jpeg.tobytes(), mime_type="image/jpeg")


class Orchestrator:
    """Runs the four phases of an attempt."""

    def __init__(self, scene, arm, judge, can_push=True):
        self.scene = scene
        self.arm = arm
        self.judge = judge
        self.can_push = can_push

    def recognize(self):
        pixel = self.judge.locate_cube(self.scene.images()["ceiling"])
        if self.judge.locate_error is not None:
            # A failed request says nothing about the cube: don't score this run.
            raise SystemExit(f"Gemini request failed: {self.judge.locate_error}")
        return None if pixel is None else self.scene.ceiling_pixel_to_table(*pixel)

    def decide(self, cube_xy):
        images = self.scene.images()
        parts = [
            DECIDE_PROMPT.format(cameras=", ".join(JUDGE_CAMERAS), x=cube_xy[0], y=cube_xy[1]),
            *[jpeg_part(images[name]) for name in JUDGE_CAMERAS],
        ]
        return self.judge._generate(parts, Decision).parsed

    def _call(self, name, args):
        """Run a motion and return what the robot itself can report."""
        extra = {}
        method = getattr(self.arm, name, None)
        try:
            if name == "locate_cube":
                self.arm.step_aside()
                cube_xy = self.recognize()
                extra = {"cube": None if cube_xy is None else np.round(cube_xy, 3).tolist()}
            elif name == "push_cube":
                method(
                    np.array([float(args["x"]), float(args["y"])]),
                    np.array([float(args["target_x"]), float(args["target_y"])]),
                )
            elif "x" in args:
                method(np.array([float(args["x"]), float(args["y"])]))
            else:
                method()
        except (RuntimeError, ValueError, KeyError) as error:
            return {"status": "error", "message": str(error)}
        return {
            "status": "ok",
            "grip_point": np.round(self.arm._grip_position(), 3).tolist(),
            "gripper_joint": round(self.arm.gripper(), 3),
            **extra,
        }

    def plan(self, cube_xy):
        """Let Gemini call the motions until it stops. Return its calls and last words."""
        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=FUNCTIONS + (PUSH_FUNCTIONS if self.can_push else [])
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
        )
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=PLAN_PROMPT.format(x=cube_xy[0], y=cube_xy[1])
                        + (PUSH_PROMPT if self.can_push else "")
                    ),
                    jpeg_part(self.scene.images()["ceiling"]),
                ],
            )
        ]
        calls = []
        for _ in range(MAX_PLAN_STEPS):
            response = self.judge.request(contents, config)
            content = response.candidates[0].content
            contents.append(content)
            function_calls = [p.function_call for p in content.parts if p.function_call]
            if not function_calls:
                return calls, response.text
            parts = []
            for call in function_calls:
                args = dict(call.args or {})
                result = self._call(call.name, args)
                calls.append({"function": call.name, "args": args, "result": result})
                print(f"    {call.name}({args}) -> {result}", flush=True)
                parts.append(types.Part.from_function_response(name=call.name, response=result))
            parts.append(types.Part.from_text(text="The ceiling camera now:"))
            parts.append(jpeg_part(self.scene.images()["ceiling"]))
            contents.append(types.Content(role="user", parts=parts))
        return calls, "stopped at the step limit"

    def judge_result(self):
        images = self.scene.images()
        return self.judge.judge({name: images[name] for name in JUDGE_CAMERAS})

    def clear_arm(self):
        """Whatever the plan did, let go and go home so that the cameras see the table."""
        self.arm.release()
        self.arm.go_home()
        self.arm.grasped_xy = None
        self.arm.held_offset = None
        self.arm.target = None

    def attempt(self):
        record = {}
        print("  1. recognize", flush=True)
        cube_xy = self.recognize()
        if cube_xy is None:
            record["recognize"] = None
            record["decide"] = {"action": "stop", "observation": "the cube wasn't found"}
            print("    the cube wasn't found", flush=True)
            return record, True
        # The true position is for the record only: nothing below reads it.
        true_xy = self.scene.cube_position()[:2]
        record["recognize"] = {
            "cube": np.round(cube_xy, 3).tolist(),
            "true_cube": np.round(true_xy, 3).tolist(),
            "error_m": round(float(np.linalg.norm(cube_xy - true_xy)), 3),
        }
        print(f"    {record['recognize']}", flush=True)

        print("  2. decide", flush=True)
        decision = self.decide(cube_xy)
        record["decide"] = {"action": decision.action.value, "observation": decision.observation}
        print(f"    {record['decide']}", flush=True)
        if decision.action is not Action.PICK:
            return record, True

        print("  3. plan", flush=True)
        calls, last_words = self.plan(cube_xy)
        record["plan"] = {"calls": calls, "last_words": last_words}
        print(f"    {last_words}", flush=True)
        self.clear_arm()

        print("  4. judge", flush=True)
        record["judge"] = self.judge_result()
        print(f"    {record['judge']}", flush=True)
        return record, record["judge"].get("gemini_success", False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--cube-y", type=float, default=0.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--no-push", action="store_true", help="don't offer push_cube to Gemini")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    scene = Scene(args.cube_y)
    true_start = scene.reset(None, np.random.default_rng(args.seed))
    writer = None
    ceiling = scene.renderers["ceiling"]
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(args.output / f"seed_{args.seed}.mp4"),
            (ceiling.width, ceiling.height),
            fps=FPS,
            macro_block_size=1,
            quality=6,
        )
        writer.send(None)

    def on_frame():
        if writer is not None:
            ceiling.update_scene(scene.data, camera="camera_ceiling")
            writer.send(ceiling.render().tobytes())

    scene.on_frame = on_frame
    arm = ScriptedArm(scene)
    orchestrator = Orchestrator(scene, arm, GeminiJudge(), not args.no_push)
    attempts = []
    for index in range(args.attempts):
        print(f"attempt {index + 1}", flush=True)
        record, finished = orchestrator.attempt()
        attempts.append(record)
        if finished:
            break
    if writer is not None:
        writer.close()

    # Scoring after everything has finished: the only verdict from the simulator's state.
    result = {
        "seed": args.seed,
        "true_cube_start": np.round(true_start, 3).tolist(),
        "cube_end": np.round(scene.cube_position(), 3).tolist(),
        "placed": scene.placed(),
        "sim_seconds": round(scene.data.time, 1),
        "attempts": attempts,
        "motions": arm.log,
    }
    print(json.dumps({key: result[key] for key in ("seed", "cube_end", "placed", "sim_seconds")}))
    scene.close()
    if args.output is not None:
        (args.output / f"seed_{args.seed}.json").write_text(
            json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
