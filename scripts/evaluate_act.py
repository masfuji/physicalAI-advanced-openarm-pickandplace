#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Evaluate a trained ACT policy on the pick and place scene in MuJoCo.

The simulator and the policy run in one process in lockstep, so no policy server
is needed and slow inference can't stall the arms. The scene is set up the same
way as dora-openarm-mujoco does for data collection.

Run from the training directory with its virtual environment:
    .venv/Scripts/python.exe ../scripts/evaluate_act.py \
        outputs/act_pick_place_v1/checkpoints/100000/pretrained_model \
        --dataset data/pick_place_cube_lerobot_half --start dataset \
        --output outputs/eval/dataset_start
"""

import argparse
import contextlib
import glob
import json
import math
import pathlib
import time

import cv2
import imageio_ffmpeg
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import torch
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from openarm_mujoco.v2 import JointResolver
from recovery import RightArm

SCENE = pathlib.Path(__file__).parent.parent / "config/scenes/pick_place.xml"
CAMERAS = ("wrist_right", "wrist_left", "head_left", "head_right", "ceiling")
JUDGE_CAMERAS = ("ceiling", "head_left")
FPS = 30
CUBE_RANGE = 0.05
SETTLE_SECONDS = 1.0
# The inside of the black frame and the height of a cube resting on its floor.
FRAME_INNER_HALF = 0.06
RESTING_Z = 1.035
RESTING_Z_TOLERANCE = 0.01
LIFTED_ABOVE_TABLE = 0.03
TABLE_RESTING_Z = 1.025
# Seen for this long in a row to rule out a cube passing through.
SUCCESS_HOLD_FRAMES = FPS
# Where the policy succeeded most often in earlier evaluations.
DEFAULT_SLIDE_TARGET = (0.53, -0.025)
# Beyond this area the right hand can't do anything for the cube.
WORKABLE_MIN_XY = np.array([0.42, -0.24])
WORKABLE_MAX_XY = np.array([0.66, 0.20])
# One push may stop short or go at an angle: locate the cube again and go on.
MAX_PUSHES = 3
# Height of the visible part of the cube, for locating it in an image.
CUBE_TOP_Z = 1.045


class Scene:
    """The simulation of one evaluation run."""

    def __init__(self, cube_y=None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.mapper = JointResolver(self.model)
        # dora-openarm-mujoco makes the cell translucent. The policy has only
        # seen the cameras with it.
        cell = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cell_vis")
        if cell >= 0:
            self.model.geom_rgba[cell, 3] = 0.2
        self.home = self._id(mujoco.mjtObj.mjOBJ_KEY, "home")
        self.cube_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
        joint = self.model.body_jntadr[self.cube_body]
        self.cube_qpos = int(self.model.jnt_qposadr[joint])
        self.cube_qvel = int(self.model.jnt_dofadr[joint])
        if cube_y is not None:
            self.model.key_qpos[self.home, self.cube_qpos + 1] = cube_y
        self.frame_xy = self.model.body_pos[
            self._id(mujoco.mjtObj.mjOBJ_BODY, "black_frame")
        ][:2].copy()
        self.addresses = {
            side: [
                self.model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in [f"openarm_{side}_joint{i}" for i in range(1, 8)]
                + [f"openarm_{side}_finger_joint1"]
            ]
            for side in ("right", "left")
        }
        self.home_arms = np.concatenate(
            [self.model.key_qpos[self.home, self.addresses[s]] for s in ("right", "left")]
        )
        self.on_frame = None
        self.renderers = {}
        for name in CAMERAS:
            width, height = self.model.cam_resolution[
                self._id(mujoco.mjtObj.mjOBJ_CAMERA, f"camera_{name}")
            ]
            self.model.vis.global_.offwidth = max(
                self.model.vis.global_.offwidth, int(width)
            )
            self.model.vis.global_.offheight = max(
                self.model.vis.global_.offheight, int(height)
            )
            self.renderers[name] = mujoco.Renderer(
                self.model, height=int(height), width=int(width)
            )

    def _id(self, kind, name):
        return mujoco.mj_name2id(self.model, kind, name)

    def reset(self, start, rng):
        """Put the arms at start, randomize the cube and let everything settle."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home)
        if start is not None:
            self.set_arms(start, self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.set_arms(self.state(), self.data.ctrl)

        pose = self.data.qpos[self.cube_qpos : self.cube_qpos + 7]
        pose[:2] += rng.uniform(-CUBE_RANGE, CUBE_RANGE, size=2)
        yaw = rng.uniform(-np.pi, np.pi)
        rotated = np.empty(4)
        mujoco.mju_mulQuat(
            rotated, [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], pose[3:7].copy()
        )
        pose[3:7] = rotated
        self.step(int(SETTLE_SECONDS / self.model.opt.timestep))
        return self.cube_position()[:2].copy()

    def set_arms(self, values, target):
        """Write the 16 values, right arm first, to data.qpos or data.ctrl."""
        setter = (
            self.mapper.set_qpos if target is self.data.qpos else self.mapper.set_ctrl
        )
        setter(target, values[:8], "right")
        setter(target, values[8:], "left")

    def step(self, count):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    def state(self):
        return np.concatenate(
            [self.data.qpos[self.addresses[s]] for s in ("right", "left")]
        ).astype(np.float32)

    def images(self):
        images = {}
        for name, renderer in self.renderers.items():
            renderer.update_scene(self.data, camera=f"camera_{name}")
            images[name] = renderer.render().copy()
        return images

    def cube_position(self):
        return self.data.qpos[self.cube_qpos : self.cube_qpos + 3]

    def cube_touches_arm(self):
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if self.cube_body in bodies:
                bodies.discard(self.cube_body)
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, bodies.pop()
                )
                if name and name.startswith("openarm_"):
                    return True
        return False

    def placed(self):
        """Return whether the cube has been released on the floor of the frame."""
        position = self.cube_position()
        speed = np.linalg.norm(self.data.qvel[self.cube_qvel : self.cube_qvel + 3])
        return bool(
            np.all(np.abs(position[:2] - self.frame_xy) < FRAME_INNER_HALF)
            and abs(position[2] - RESTING_Z) < RESTING_Z_TOLERANCE
            and speed < 0.02
            and not self.cube_touches_arm()
        )

    def ceiling_pixel_to_table(self, u, v):
        """Return the xy where a ceiling camera pixel's ray meets the cube's height."""
        camera = self._id(mujoco.mjtObj.mjOBJ_CAMERA, "camera_ceiling")
        renderer = self.renderers["ceiling"]
        focal = 0.5 * renderer.height / math.tan(math.radians(self.model.cam_fovy[camera]) / 2)
        ray = np.array([u - renderer.width / 2, renderer.height / 2 - v, -focal])
        direction = self.data.cam_xmat[camera].reshape(3, 3) @ ray
        origin = self.data.cam_xpos[camera]
        return (origin + direction * (CUBE_TOP_Z - origin[2]) / direction[2])[:2]

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()


class Policy:
    """A pretrained policy that maps simulator observations to arm targets."""

    def __init__(self, path, n_action_steps, temporal_ensemble_coeff):
        self.policy = ACTPolicy.from_pretrained(path)
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        if temporal_ensemble_coeff is not None:
            self.policy.config.n_action_steps = 1
            self.policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
            self.policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff, self.policy.config.chunk_size
            )
        self.policy.eval()
        self.device = self.policy.config.device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=path
        )
        self.sizes = {
            name: self.policy.config.input_features[
                f"observation.images.{name}"
            ].shape[1:]
            for name in CAMERAS
        }

    def reset(self):
        self.policy.reset()

    @torch.inference_mode()
    def act(self, state, images):
        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0)}
        for name, image in images.items():
            height, width = self.sizes[name]
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            batch[f"observation.images.{name}"] = (
                torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255
            )
        batch = self.preprocessor(batch)
        autocast = (
            torch.autocast("cuda")
            if self.policy.config.use_amp
            else contextlib.nullcontext()
        )
        with autocast:
            action = self.policy.select_action(batch)
        return self.postprocessor(action).squeeze(0).cpu().numpy()


def load_start_poses(dataset):
    """Return the arm positions at the start of each dataset episode."""
    frames = pd.concat(
        pd.read_parquet(path, columns=["frame_index", "observation.state"])
        for path in glob.glob(f"{dataset}/data/**/*.parquet", recursive=True)
    )
    return np.stack(frames[frames["frame_index"] == 0]["observation.state"])


def run_attempt(scene, policy, horizon, video, viewer, trajectory_path, stop_when_placed=True):
    """Run the policy until time is up, or until the task is done if stop_when_placed."""
    policy.reset()
    steps_per_frame = round(1 / FPS / scene.model.opt.timestep)
    lifted_at = None
    placed_for = 0
    max_z = float(scene.cube_position()[2])
    frame = 0
    trajectory = {"state": [], "action": [], "cube": []}
    inference_seconds = 0.0
    for frame in range(horizon):
        started = time.perf_counter()
        images = scene.images()
        state = scene.state()
        inference_started = time.perf_counter()
        action = policy.act(state, images)
        inference_seconds += time.perf_counter() - inference_started
        trajectory["state"].append(state)
        trajectory["action"].append(action)
        trajectory["cube"].append(scene.cube_position().copy())
        scene.set_arms(action, scene.data.ctrl)
        scene.step(steps_per_frame)
        if video is not None:
            video.send(images["ceiling"].tobytes())

        z = float(scene.cube_position()[2])
        max_z = max(max_z, z)
        if lifted_at is None and z > TABLE_RESTING_Z + LIFTED_ABOVE_TABLE:
            lifted_at = frame / FPS
        placed_for = placed_for + 1 if scene.placed() else 0
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0, 1 / FPS - (time.perf_counter() - started)))
        if stop_when_placed and placed_for >= SUCCESS_HOLD_FRAMES:
            break
    if trajectory_path is not None:
        np.savez_compressed(
            trajectory_path,
            n_action_steps=policy.policy.config.n_action_steps,
            **{key: np.stack(values) for key, values in trajectory.items()},
        )
    return {
        "lifted": lifted_at is not None,
        "lifted_at_s": lifted_at,
        "max_cube_z": round(max_z, 3),
        "placed": placed_for >= SUCCESS_HOLD_FRAMES,
        "seconds": round((frame + 1) / FPS, 1),
        "inference_ms_per_frame": round(inference_seconds / (frame + 1) * 1000, 1),
    }


def locate_cube(scene, judge, retry_with):
    """Return the cube's xy, from Gemini or from the simulator's state."""
    if retry_with == "sim":
        return scene.cube_position()[:2].copy()
    pixel = judge.locate_cube(scene.images()["ceiling"])
    return None if pixel is None else scene.ceiling_pixel_to_table(*pixel)


def locate_or_clear_view(scene, arm, judge, retry_with):
    """Locate the cube with the hand out of the way, from home if it's still hidden."""
    arm.step_aside()
    cube = locate_cube(scene, judge, retry_with)
    if cube is not None:
        return cube, False
    arm.go_home()
    return locate_cube(scene, judge, retry_with), True


def recover(scene, arm, judge, retry_with, slide_target):
    """Move the cube for another attempt, without stopping at home. Return what happened."""
    arm.release()
    if retry_with == "sim" and scene.placed():
        # Letting go above the tray can finish the task. With Gemini, the check of
        # the located cube below finds this without the simulator's state.
        arm.go_home()
        return {"action": "placed while releasing", "stop": True}
    cube, went_home = locate_or_clear_view(scene, arm, judge, retry_with)
    views_cleared = int(went_home)
    if cube is None:
        return {"action": "cube not located", "stop": False}
    located = {"located_cube": np.round(cube, 3).tolist()}
    truth = scene.cube_position()[:2].copy()
    located["location_error_m"] = round(float(np.linalg.norm(cube - truth)), 3)
    if np.all(np.abs(cube - scene.frame_xy) < FRAME_INNER_HALF):
        arm.go_home()
        return {"action": "cube is in the tray", "stop": True, **located}
    if np.any(cube < WORKABLE_MIN_XY) or np.any(cube > WORKABLE_MAX_XY):
        arm.go_home()
        return {"action": "cube is out of reach", "stop": True, **located}
    if slide_target is None:
        arm.go_home()
        return {"action": "retry without sliding", "stop": False, **located}
    before = truth
    pushes = 0
    while pushes < MAX_PUSHES and arm.push_cube(cube, slide_target):
        pushes += 1
        cube, went_home = locate_or_clear_view(scene, arm, judge, retry_with)
        views_cleared += int(went_home)
        if cube is None:
            break
    after = scene.cube_position()[:2]
    # The policy always starts from home with the gripper open.
    arm.go_home()
    if pushes == 0:
        return {"action": f"not slid: {arm.failure}", "stop": False, **located}
    return {
        "action": "slid",
        "stop": False,
        **located,
        "pushes": pushes,
        "views_cleared": views_cleared,
        "cube_before": before.round(3).tolist(),
        "cube_after": after.round(3).tolist(),
        "target_error_m": round(float(np.linalg.norm(after - slide_target)), 3),
    }


def run_episode(
    scene,
    policy,
    start,
    seed,
    horizon,
    video_path,
    trajectory_path,
    viewer,
    judge,
    max_retries=0,
    retry_with="sim",
    slide_target=None,
    stop_when_placed=True,
):
    rng = np.random.default_rng(seed)
    cube_start = scene.reset(start, rng)
    writer = None
    ceiling = scene.renderers["ceiling"]
    if video_path is not None:
        writer = imageio_ffmpeg.write_frames(
            str(video_path), (ceiling.width, ceiling.height), fps=FPS, macro_block_size=1, quality=6
        )
        writer.send(None)

    def on_frame():
        if writer is not None:
            ceiling.update_scene(scene.data, camera="camera_ceiling")
            writer.send(ceiling.render().tobytes())
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(1 / FPS)

    scene.on_frame = on_frame
    arm = RightArm(scene)
    started_at = scene.data.time
    attempts = []
    for attempt_index in range(max_retries + 1):
        path = trajectory_path
        if path is not None and attempt_index > 0:
            path = path.with_name(f"{path.stem}_attempt{attempt_index + 1}.npz")
        attempt = run_attempt(scene, policy, horizon, writer, viewer, path, stop_when_placed)
        attempts.append(attempt)
        if judge is not None:
            images = scene.images()
            attempt.update(judge.judge({name: images[name] for name in JUDGE_CAMERAS}))
        if retry_with == "gemini":
            # A failed request says nothing about the task: don't act on it.
            done = attempt.get("gemini_success", True)
        else:
            done = attempt["placed"]
        if done or attempt_index == max_retries:
            break
        if viewer is not None and not viewer.is_running():
            break
        recovery = recover(scene, arm, judge, retry_with, slide_target)
        attempt["recovery"] = recovery
        if recovery["stop"]:
            break
    if writer is not None:
        writer.close()

    position = scene.cube_position()
    # The attempt's flag misses a cube released while retreating.
    placed = attempts[-1]["placed"] or scene.placed()
    return {
        **{
            key: attempts[-1][key]
            for key in ("gemini_success", "gemini_observation", "gemini_error")
            if key in attempts[-1]
        },
        "seed": seed,
        "cube_start": cube_start.round(3).tolist(),
        "cube_end": position.round(3).tolist(),
        "lifted": any(a["lifted"] for a in attempts),
        "lifted_at_s": attempts[0]["lifted_at_s"],
        "max_cube_z": max(a["max_cube_z"] for a in attempts),
        "distance_to_frame_m": round(
            float(np.linalg.norm(position[:2] - scene.frame_xy)), 3
        ),
        "placed": placed,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "seconds": round(sum(a["seconds"] for a in attempts), 1),
        "total_sim_seconds": round(scene.data.time - started_at, 1),
        "inference_ms_per_frame": attempts[0]["inference_ms_per_frame"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=pathlib.Path)
    parser.add_argument("--start", choices=["home", "dataset"], default="dataset")
    parser.add_argument("--dataset", type=pathlib.Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--horizon-seconds", type=float, default=30.0)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument(
        "--cube-y", type=float, help="center of the cube's random area instead of the scene's"
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        help="infer on every frame and average the overlapping plans; "
        "positive favors older plans, negative favors newer ones",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--viewer", action="store_true", help="watch in real time")
    parser.add_argument(
        "--gemini-judge",
        action="store_true",
        help="also ask Gemini Robotics-ER 2 for the result at the end of each attempt",
    )
    parser.add_argument(
        "--max-retries", type=int, default=0, help="attempts to add after a failed one"
    )
    parser.add_argument(
        "--retry-with",
        choices=["sim", "gemini"],
        default="sim",
        help="what decides that an attempt failed and locates the cube: the simulator's "
        "state or Gemini Robotics-ER 2",
    )
    parser.add_argument(
        "--slide-target",
        type=float,
        nargs=2,
        default=DEFAULT_SLIDE_TARGET,
        metavar=("X", "Y"),
        help="where to push the cube before a retry",
    )
    parser.add_argument("--no-slide", action="store_true", help="retry without pushing the cube")
    parser.add_argument(
        "--no-sim-stop",
        action="store_true",
        help="run every attempt for the whole horizon, as on a real robot where "
        "the simulator's state can't end an attempt",
    )
    args = parser.parse_args()
    if args.start == "dataset" and args.dataset is None:
        parser.error("--start dataset needs --dataset")

    judge = None
    if args.gemini_judge or args.retry_with == "gemini":
        # Imported here because only this option needs the Gemini packages.
        import gemini_judge

        judge = gemini_judge.GeminiJudge()

    starts = load_start_poses(args.dataset) if args.start == "dataset" else [None]
    scene = Scene(args.cube_y)
    policy = Policy(args.policy, args.n_action_steps, args.temporal_ensemble_coeff)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    results = []
    viewer = mujoco.viewer.launch_passive(scene.model, scene.data) if args.viewer else None
    for index in range(args.episodes):
        video = args.output / f"episode_{index:02d}.mp4" if args.output else None
        result = run_episode(
            scene,
            policy,
            starts[index % len(starts)],
            args.seed + index,
            int(args.horizon_seconds * FPS),
            video,
            args.output / f"episode_{index:02d}.npz" if args.output else None,
            viewer,
            judge,
            args.max_retries,
            args.retry_with,
            None if args.no_slide else np.array(args.slide_target),
            not args.no_sim_stop,
        )
        results.append(result)
        print(f"{index:2d} {json.dumps(result)}", flush=True)
        if viewer is not None and not viewer.is_running():
            break
    if viewer is not None:
        viewer.close()
    scene.close()

    count = len(results)
    summary = {
        "episodes": count,
        "lifted": sum(r["lifted"] for r in results),
        "placed": sum(r["placed"] for r in results),
        "median_distance_to_frame_m": float(
            np.median([r["distance_to_frame_m"] for r in results])
        ),
    }
    summary["attempts"] = sum(r["attempt_count"] for r in results)
    summary["slid"] = sum(
        a.get("recovery", {}).get("action") == "slid" for r in results for a in r["attempts"]
    )
    if judge is not None:
        summary["gemini"] = gemini_judge.summarize([a for r in results for a in r["attempts"]])
    print(json.dumps(summary))
    if args.output is not None:
        (args.output / "results.json").write_text(
            json.dumps({"args": str(vars(args)), "summary": summary, "episodes": results}, indent=1),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
