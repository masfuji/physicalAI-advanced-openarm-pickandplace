#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Evaluate ACT after Gemini Robotics-ER 2 has moved the cube out of its weak area.

Each episode runs once, without retries:
    1. Recognize: Gemini locates the cube in the ceiling camera's image.
    2. Decide: is the located cube in the area where the policy is weak?
    3. Slide: if so, push it to where the policy is good at.
    4. Confirm: Gemini locates it again, and it is pushed again if needed.
    5. Run: the policy runs until the cube is placed or time is up.
    6. Judge: Gemini judges the result, for the record.

Only the simulator's state ends step 5 early and scores the result: it doesn't decide
anything else. The true position is recorded, but nothing reads it.

Run from the training directory with its virtual environment:
    .venv/Scripts/python.exe ../scripts/evaluate_planned.py \
        outputs/act_pick_place_v1/checkpoints/100000/pretrained_model \
        --episodes 100 --seed 3100 --output outputs/eval_planned/plan
"""

import argparse
import json
import pathlib

import imageio_ffmpeg
import numpy as np
from evaluate_act import FPS, JUDGE_CAMERAS, MAX_PUSHES, Policy, Scene, run_attempt
from gemini_judge import GeminiJudge
from recovery import CLOSE_ENOUGH, RightArm

# From 300 runs of the policy alone (15, 30 and 45 seconds), counting earlier successes more.
SLIDE_TARGET = (0.515, -0.01)
# Pushing helped only here: closer to the robot it didn't raise the success rate.
WEAK_X_FROM = 0.555
HORIZON_SECONDS = 45.0


def locate(scene, judge):
    """Return the located cube's xy, or None if Gemini doesn't see it."""
    pixel = judge.locate_cube(scene.images()["ceiling"])
    if judge.locate_error is not None:
        # A failed request says nothing about the cube: don't score this run.
        raise SystemExit(f"Gemini request failed: {judge.locate_error}")
    return None if pixel is None else scene.ceiling_pixel_to_table(*pixel)


def is_weak(cube):
    return bool(cube[0] >= WEAK_X_FROM)


def slide(scene, arm, judge, cube, target):
    """Push the cube toward the target and check where it went. Return what happened."""
    record = {"pushes": 0, "located": [], "stop": None}
    distance = np.linalg.norm(cube - target)
    while record["pushes"] < MAX_PUSHES:
        if not arm.push_cube(cube, target):
            record["stop"] = f"not pushed: {arm.failure}"
            break
        record["pushes"] += 1
        arm.step_aside()
        cube = locate(scene, judge)
        if cube is None:
            record["stop"] = "not found after the push"
            break
        record["located"].append(np.round(cube, 3).tolist())
        new_distance = np.linalg.norm(cube - target)
        if new_distance < CLOSE_ENOUGH:
            record["stop"] = "reached the target"
            break
        if new_distance >= distance:
            # Pushing again could send it further: the policy gets it from here.
            record["stop"] = "moved no closer"
            break
        distance = new_distance
    else:
        record["stop"] = "out of pushes"
    return record


def run_episode(scene, policy, judge, seed, horizon, target, video_path, trajectory_path):
    true_start = scene.reset(None, np.random.default_rng(seed))
    ceiling = scene.renderers["ceiling"]
    writer = None
    if video_path is not None:
        writer = imageio_ffmpeg.write_frames(
            str(video_path), (ceiling.width, ceiling.height), fps=FPS, macro_block_size=1, quality=6
        )
        writer.send(None)

    def on_frame():
        if writer is not None:
            ceiling.update_scene(scene.data, camera="camera_ceiling")
            writer.send(ceiling.render().tobytes())

    scene.on_frame = on_frame
    arm = RightArm(scene)
    arm.left_target = scene.state()[8:].copy()
    try:
        cube = locate(scene, judge)
        weak = None if cube is None else is_weak(cube)
        sliding = None
        if weak:
            sliding = slide(scene, arm, judge, cube, target)
            arm.go_home()
        started_policy_at = scene.data.time
        # For the record only: nothing reads it.
        cube_at_policy = scene.cube_position().copy()
        attempt = run_attempt(scene, policy, horizon, writer, None, trajectory_path)
        images = scene.images()
        verdict = judge.judge({name: images[name] for name in JUDGE_CAMERAS})
    finally:
        if writer is not None:
            writer.close()
    position = scene.cube_position()
    return {
        "seed": seed,
        "located": None if cube is None else np.round(cube, 3).tolist(),
        "true_cube_start": np.round(true_start, 3).tolist(),
        "location_error_m": None
        if cube is None
        else round(float(np.linalg.norm(cube - true_start)), 3),
        "weak": weak,
        "slide": sliding,
        "true_cube_at_policy_start": np.round(cube_at_policy, 3).tolist(),
        "cube_end": np.round(position, 3).tolist(),
        "lifted": attempt["lifted"],
        "lifted_at_s": attempt["lifted_at_s"],
        "max_cube_z": attempt["max_cube_z"],
        "placed": attempt["placed"],
        "policy_seconds": attempt["seconds"],
        "seconds_before_policy": round(started_policy_at, 1),
        "total_sim_seconds": round(scene.data.time, 1),
        "inference_ms_per_frame": attempt["inference_ms_per_frame"],
        **verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=pathlib.Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--cube-y", type=float, default=0.0)
    parser.add_argument("--horizon-seconds", type=float, default=HORIZON_SECONDS)
    parser.add_argument("--slide-target", type=float, nargs=2, default=SLIDE_TARGET, metavar=("X", "Y"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    judge = GeminiJudge()
    scene = Scene(args.cube_y)
    policy = Policy(args.policy, None, None)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(args.episodes):
        result = run_episode(
            scene,
            policy,
            judge,
            args.seed + index,
            int(args.horizon_seconds * FPS),
            np.array(args.slide_target),
            args.output / f"episode_{index:02d}.mp4",
            args.output / f"episode_{index:02d}.npz",
        )
        results.append(result)
        print(f"{index:2d} {json.dumps(result, ensure_ascii=False)}", flush=True)
        # Written every time, so that what ran is kept if the API credit runs out.
        summary = {
            "episodes": len(results),
            "placed": sum(r["placed"] for r in results),
            "lifted": sum(r["lifted"] for r in results),
            "weak": sum(bool(r["weak"]) for r in results),
            "judge_errors": sum("gemini_error" in r for r in results),
            "judge_disagreements": sum(
                "gemini_success" in r and r["gemini_success"] != r["placed"] for r in results
            ),
            "usage": judge.usage,
        }
        (args.output / "results.json").write_text(
            json.dumps(
                {"args": str(vars(args)), "summary": summary, "episodes": results},
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary))
    scene.close()


if __name__ == "__main__":
    main()
