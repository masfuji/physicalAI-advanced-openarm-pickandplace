#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Pick and place the cube with scripted motions instead of a policy.

Each motion is one method, so that a planner can call them one by one. The grasp is
the one in the teleoperated data: the fingers come in nearly level and diagonally
across the table, and closing them turns the cube into line, so its yaw isn't needed.

Run from the training directory with its virtual environment:
    .venv/Scripts/python.exe ../scripts/scripted_pick_place.py --output outputs/scripted/try
"""

import argparse
import json
import pathlib

import imageio_ffmpeg
import mujoco
import numpy as np
from evaluate_act import CAMERAS, FPS, Scene
from recovery import FRAME_SECONDS, GRIPPER_CLOSED, GRIPPER_OPEN, RightArm

# Measured at the moment of lifting in successful episodes of the policy.
GRASP_TOOL = np.array([0.70, 0.71, -0.06])
GRASP_SEED = np.array([0.08, 0.08, -0.27, 1.22, 0.04, -0.27, 0.54])
GRASP_OFFSET_ALONG_TOOL = 0.015
GRASP_HEIGHT_ABOVE_CUBE = 0.01
CUBE_REST_Z = 1.025
# Backing away along the fingers keeps the open hand from touching the cube.
APPROACH_DISTANCE = 0.12
APPROACH_HEIGHT = 0.06
LIFT_HEIGHT = 0.15
# The arm's pose while carrying the cube above the tray, from the same episodes.
PLACE_SEED = np.array([0.39, -0.15, -0.54, 1.31, -0.20, 0.06, 0.45])
# At home the left hand is above the tray: it would stop the right hand there.
LEFT_ASIDE = np.array([0.02, -0.21, -0.06, 1.29, -0.11, 0.77, -0.19, 0.79])
TRAY_XY = np.array([0.55, 0.15])
RELEASE_CUBE_Z = 1.10
LINE_SPEED = 0.08
CLOSE_SECONDS = 1.0
SETTLE_SECONDS = 0.5
RELEASE_WAIT_SECONDS = 1.5
ORIENTATION_WEIGHT = 0.3


def grasp_rotation():
    """Return the site's rotation at the grasp: its x axis up, its z axis against the tool."""
    tool = GRASP_TOOL / np.linalg.norm(GRASP_TOOL)
    z = -tool
    x = np.array([0.0, 0.0, 1.0]) - z[2] * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class ScriptedArm(RightArm):
    """The right arm with one method per step of the task."""

    def __init__(self, scene):
        super().__init__(scene)
        self.log = []
        self.rotation = grasp_rotation()
        self.grasped_xy = None
        self.held_offset = None
        self.target = None
        self.place_rotation = self._rotation_at(PLACE_SEED)
        self.left_target = scene.home_arms[8:].copy()

    def _rotation_at(self, joints):
        data = self.ik_data
        data.qpos[self.qpos_addresses] = joints
        mujoco.mj_kinematics(self.model, data)
        return data.site_xmat[self.site].reshape(3, 3).copy()

    def solve_pose(self, position, rotation, seed, iterations=200):
        """Return the joints reaching a full pose of the grip point, and the errors."""
        data = self.ik_data
        data.qpos[self.qpos_addresses] = seed
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            point, _ = self._grip_point()
            current = data.site_xmat[self.site].reshape(3, 3)
            position_error = position - point
            rotation_error = 0.5 * sum(np.cross(current[:, i], rotation[:, i]) for i in range(3))
            if np.linalg.norm(position_error) < 5e-4 and np.linalg.norm(rotation_error) < 5e-3:
                break
            jacobian_position = np.zeros((3, self.model.nv))
            jacobian_rotation = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jacobian_position, jacobian_rotation, self.site)
            offset = point - data.site_xpos[self.site]
            jacobian_position = jacobian_position + np.cross(jacobian_rotation.T, offset).T
            jacobian = np.vstack([jacobian_position, ORIENTATION_WEIGHT * jacobian_rotation])[
                :, self.dof_addresses
            ]
            error = np.concatenate([position_error, ORIENTATION_WEIGHT * rotation_error])
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3 * np.eye(6), error)
            data.qpos[self.qpos_addresses] = np.clip(
                data.qpos[self.qpos_addresses] + 0.5 * step, self.limits[:, 0], self.limits[:, 1]
            )
        mujoco.mj_kinematics(self.model, data)
        point, _ = self._grip_point()
        current = data.site_xmat[self.site].reshape(3, 3)
        rotation_error = 0.5 * sum(np.cross(current[:, i], rotation[:, i]) for i in range(3))
        return (
            data.qpos[self.qpos_addresses].copy(),
            float(np.linalg.norm(position - point)),
            float(np.linalg.norm(rotation_error)),
        )

    def _grip_position(self):
        data = self.ik_data
        data.qpos[self.qpos_addresses] = self.joints()
        mujoco.mj_kinematics(self.model, data)
        return self._grip_point()[0]

    def _reach(self, position, rotation, seed, gripper, seconds=2.0):
        joints, position_error, rotation_error = self.solve_pose(position, rotation, seed)
        if position_error > 0.005:
            raise RuntimeError(f"out of reach: {position_error * 100:.1f} cm from {position}")
        self.move_joints(joints, gripper, seconds=seconds)
        self.hold(joints, gripper, SETTLE_SECONDS)
        self.target = joints
        return rotation_error

    def _line(self, end, rotation, gripper):
        """Move the grip point along a straight line, planning it all before moving."""
        start = self._grip_position()
        frames = max(1, round(np.linalg.norm(end - start) / LINE_SPEED / FRAME_SECONDS))
        joints = self.joints()
        path = []
        for frame in range(1, frames + 1):
            joints, position_error, _ = self.solve_pose(
                start + (end - start) * frame / frames, rotation, joints, iterations=50
            )
            if position_error > 0.003:
                raise RuntimeError(f"line out of reach: {position_error * 100:.1f} cm")
            path.append(joints)
        for joints in path:
            self._command(joints, gripper)
        self.hold(path[-1], gripper, SETTLE_SECONDS)
        self.target = path[-1]

    def _move_gripper(self, gripper, wait_seconds):
        if self.target is None:
            raise RuntimeError("the arm hasn't been moved to a target yet")
        # Commanding the measured joints would let the arm sag little by little.
        grip_start = self.gripper()
        frames = round(CLOSE_SECONDS / FRAME_SECONDS)
        for frame in range(1, frames + 1):
            self._command(self.target, grip_start + (gripper - grip_start) * frame / frames)
        self.hold(self.target, gripper, wait_seconds)

    def _record(self, name, **values):
        self.log.append(
            {
                "step": name,
                # For the record only: no decision reads this.
                "true_cube": np.round(self.scene.cube_position(), 3).tolist(),
                "grip_point": np.round(self._grip_position(), 3).tolist(),
                "gripper": round(self.gripper(), 3),
                **values,
            }
        )

    def _grasp_position(self, cube_xy):
        tool = GRASP_TOOL / np.linalg.norm(GRASP_TOOL)
        center = np.array([*cube_xy, CUBE_REST_Z + GRASP_HEIGHT_ABOVE_CUBE])
        return center + tool * GRASP_OFFSET_ALONG_TOOL

    def push_cube(self, cube_xy, target_xy):
        """Push the cube across the table with the closed gripper."""
        if not super().push_cube(cube_xy, target_xy):
            raise RuntimeError(f"not pushed: {self.failure}")
        self.target = self.joints()
        self.grasped_xy = None
        self.held_offset = None
        self._record("push_cube")
        return True

    def move_above_cube(self, cube_xy):
        """Go above the point the fingers will come in from, with the hand open."""
        tool = GRASP_TOOL / np.linalg.norm(GRASP_TOOL)
        back = self._grasp_position(cube_xy) - tool * APPROACH_DISTANCE
        above = back + [0, 0, APPROACH_HEIGHT]
        joints, position_error, _ = self.solve_pose(above, self.rotation, GRASP_SEED)
        if position_error > 0.005:
            raise RuntimeError(f"out of reach: {position_error * 100:.1f} cm from {above}")
        self.move_joints(joints, GRIPPER_OPEN, left=LEFT_ASIDE)
        self.hold(joints, GRIPPER_OPEN, SETTLE_SECONDS)
        self.target = joints
        self._record("move_above_cube")

    def descend(self, cube_xy):
        """Come down beside the cube, still clear of it."""
        tool = GRASP_TOOL / np.linalg.norm(GRASP_TOOL)
        self._line(self._grasp_position(cube_xy) - tool * APPROACH_DISTANCE, self.rotation, GRIPPER_OPEN)
        self._record("descend")

    def approach(self, cube_xy):
        """Slide the open fingers around the cube."""
        self._line(self._grasp_position(cube_xy), self.rotation, GRIPPER_OPEN)
        self.grasped_xy = np.asarray(cube_xy, dtype=float)
        self._record("approach")

    def close_gripper(self):
        self._move_gripper(GRIPPER_CLOSED, SETTLE_SECONDS)
        self._record("close_gripper")

    def lift(self):
        if self.grasped_xy is None:
            raise RuntimeError("lift needs approach first")
        start = self._grip_position()
        # From where the cube was seen: a robot can't read the cube's true position.
        self.held_offset = start - np.array([*self.grasped_xy, CUBE_REST_Z])
        self._line(start + [0, 0, LIFT_HEIGHT], self.rotation, GRIPPER_CLOSED)
        self._record("lift")

    def move_above_tray(self):
        if self.held_offset is None:
            raise RuntimeError("move_above_tray needs lift first")
        target = np.array([*TRAY_XY, RELEASE_CUBE_Z]) + self.held_offset
        self._reach(target, self.place_rotation, PLACE_SEED, GRIPPER_CLOSED, seconds=3.0)
        self._record("move_above_tray")

    def open_gripper(self):
        self._move_gripper(GRIPPER_OPEN, RELEASE_WAIT_SECONDS)
        self._record("open_gripper")

    def retreat(self):
        tool = -self.place_rotation[:, 2]
        self._line(self._grip_position() - tool * 0.10, self.place_rotation, GRIPPER_OPEN)
        self.go_home()
        self._record("retreat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--cube-y", type=float, default=0.0)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    scene = Scene(args.cube_y)
    cube_start = scene.reset(None, np.random.default_rng(args.seed))
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
    error = None
    try:
        arm.move_above_cube(cube_start)
        arm.descend(cube_start)
        arm.approach(cube_start)
        arm.close_gripper()
        arm.lift()
        arm.move_above_tray()
        arm.open_gripper()
        arm.retreat()
    except RuntimeError as failure:
        error = str(failure)
    if writer is not None:
        writer.close()

    result = {
        "seed": args.seed,
        "cube_start": np.round(cube_start, 3).tolist(),
        "cube_end": np.round(scene.cube_position(), 3).tolist(),
        "placed": scene.placed(),
        "sim_seconds": round(scene.data.time, 1),
        "error": error,
        "steps": arm.log,
    }
    print(json.dumps(result, indent=1))
    scene.close()
    if args.output is not None:
        (args.output / f"seed_{args.seed}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
