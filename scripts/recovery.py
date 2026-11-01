#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Scripted motions for recovering from a failed attempt in the MuJoCo scene.

After a failure the arm retreats, then pushes the cube to a position that the
policy is good at, so that the next attempt doesn't repeat the same failure.
"""

import mujoco
import numpy as np

SITE = "right_ee_control_point"
# From the end-effector control point to the ends of the closed fingers.
TIP_POINT = np.array([0.0, 0.0, -0.17])
TABLE_Z = 1.005
CUBE_HALF = 0.02
# Height of the fingertips while pushing and while moving above.
PUSH_Z = TABLE_Z + 0.015
CLEAR_Z = TABLE_Z + 0.12
# How far the fingers lean toward the push direction, from straight down. The hand
# must stay behind the cube, or it lands on the cube before the fingertips get low.
# Negative is leaning away from it: the joints don't allow more toward the robot.
PUSH_LEANS = np.radians([35, 20, 5, -5, -15, -25])
TOOL_WEIGHT = 0.2
MAX_TOOL_TILT = np.radians(15)
SHOULDER_XY = np.array([0.185, -0.094])
REST_TOOL_DOWN = 0.8
PUSHER_RADIUS = 0.01
PUSH_START_GAPS = (0.03, 0.02, 0.01, 0.005)
# The fingertips must not come down on the black tray.
TRAY_MIN_XY = np.array([0.46, 0.07])
TRAY_MAX_XY = np.array([0.64, 0.24])
# How far a push may turn away from the target, tried in order. Pushing at an angle
# makes room when the tray is behind the cube, and the next push corrects the rest.
PUSH_ANGLES = np.radians([0, 25, -25, 50, -50, 70, -70])
# Not worth pushing when already this close.
CLOSE_ENOUGH = 0.02
SETTLE_SECONDS = 0.5
# Where the hand waits so that the ceiling camera sees the cube, short of going home.
ASIDE_POSITION = np.array([0.50, -0.22, TABLE_Z + 0.15])
ASIDE_SECONDS = 1.0
GRIPPER_CLOSED = -0.05
GRIPPER_OPEN = -0.785
PUSH_SPEED = 0.05
MOVE_SECONDS = 2.0
FRAME_SECONDS = 1 / 30


def rest_tool_direction(position):
    """Return a comfortable direction of the fingers at a position."""
    away = position[:2] - SHOULDER_XY
    away = away / np.linalg.norm(away) * np.sqrt(1 - REST_TOOL_DOWN**2)
    return np.array([away[0], away[1], -REST_TOOL_DOWN])


def push_tool_direction(direction, lean):
    return np.array([*(direction * np.sin(lean)), -np.cos(lean)])


class RightArm:
    """Moves the right arm of a evaluate_act.Scene with position targets."""

    def __init__(self, scene):
        self.scene = scene
        self.model = scene.model
        self.ik_data = mujoco.MjData(scene.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, SITE)
        joints = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_joint{i}")
            for i in range(1, 8)
        ]
        self.limits = self.model.jnt_range[joints]
        self.qpos_addresses = [self.model.jnt_qposadr[j] for j in joints]
        self.dof_addresses = [self.model.jnt_dofadr[j] for j in joints]
        self.rng = np.random.default_rng(0)
        self.failure = None
        self.left_target = None

    def _grip_point(self):
        rotation = self.ik_data.site_xmat[self.site].reshape(3, 3)
        return self.ik_data.site_xpos[self.site] + rotation @ TIP_POINT, -rotation[:, 2]

    def solve(self, position, seed, preferred=None, iterations=200):
        """Return the joints reaching position from seed and the remaining errors."""
        if preferred is None:
            preferred = rest_tool_direction(position)
        data = self.ik_data
        data.qpos[self.qpos_addresses] = seed
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            point, tool = self._grip_point()
            position_error = position - point
            tool_error = np.cross(tool, preferred)
            if np.linalg.norm(position_error) < 5e-4 and np.linalg.norm(tool_error) < 5e-3:
                break
            jacobian_position = np.zeros((3, self.model.nv))
            jacobian_rotation = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(
                self.model, data, jacobian_position, jacobian_rotation, self.site
            )
            offset = point - data.site_xpos[self.site]
            jacobian_position = jacobian_position + np.cross(
                jacobian_rotation.T, offset
            ).T
            jacobian = np.vstack(
                [jacobian_position, TOOL_WEIGHT * jacobian_rotation]
            )[:, self.dof_addresses]
            error = np.concatenate([position_error, TOOL_WEIGHT * tool_error])
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-3 * np.eye(6), error
            )
            data.qpos[self.qpos_addresses] = np.clip(
                data.qpos[self.qpos_addresses] + 0.5 * step, self.limits[:, 0], self.limits[:, 1]
            )
        mujoco.mj_kinematics(self.model, data)
        point, tool = self._grip_point()
        return (
            data.qpos[self.qpos_addresses].copy(),
            float(np.linalg.norm(position - point)),
            float(np.arccos(np.clip(tool @ preferred, -1, 1))),
        )

    def joints(self):
        return self.scene.state()[:7].astype(np.float64)

    def gripper(self):
        return float(self.scene.state()[7])

    def _command(self, joints, gripper):
        values = self.scene.state().copy()
        values[8:] = self.left_target
        values[:7] = joints
        values[7] = gripper
        self.scene.set_arms(values, self.scene.data.ctrl)
        self.scene.step(round(FRAME_SECONDS / self.model.opt.timestep))
        if self.scene.on_frame is not None:
            self.scene.on_frame()

    def move_joints(self, target, gripper=None, seconds=MOVE_SECONDS, left=None):
        """Move to the joints smoothly, optionally changing the gripper and left arm."""
        left_start = self.left_target
        left_end = left_start if left is None else left
        start = self.joints()
        grip_start = self.gripper()
        grip_end = grip_start if gripper is None else gripper
        frames = max(1, round(seconds / FRAME_SECONDS))
        for frame in range(1, frames + 1):
            ratio = 0.5 - 0.5 * np.cos(np.pi * frame / frames)
            self.left_target = left_start + (left_end - left_start) * ratio
            self._command(
                start + (target - start) * ratio, grip_start + (grip_end - grip_start) * ratio
            )

    def plan_line(self, joints, start, end, speed, preferred=None):
        """Return the joints along a line, or None if a point of it is out of reach."""
        frames = max(1, round(np.linalg.norm(end - start) / speed / FRAME_SECONDS))
        path = []
        for frame in range(1, frames + 1):
            joints, position_error, _ = self.solve(
                start + (end - start) * frame / frames, joints, preferred, iterations=50
            )
            if position_error > 0.003:
                return None
            path.append(joints)
        return path

    def release(self):
        """Release whatever is held and rise, with the left arm going home meanwhile."""
        self.left_target = self.scene.state()[8:].copy()
        self.lift_and_open()

    def lift_and_open(self):
        """Release whatever is held and rise away from the table."""
        data = self.ik_data
        data.qpos[self.qpos_addresses] = self.joints()
        mujoco.mj_kinematics(self.model, data)
        point, _ = self._grip_point()
        left_home = self.scene.home_arms[8:]
        self.move_joints(self.joints(), GRIPPER_OPEN, seconds=1.0)
        target = point + [0, 0, max(0.0, CLEAR_Z - point[2])]
        joints, position_error, _ = self.solve(target, self.joints())
        if position_error < 0.01:
            self.move_joints(joints, GRIPPER_OPEN, seconds=1.0, left=left_home)
        else:
            self.move_joints(self.joints(), GRIPPER_OPEN, seconds=1.0, left=left_home)

    def step_aside(self):
        """Clear the ceiling camera's view of the table without going home."""
        joints, position_error, _ = self.solve(ASIDE_POSITION, self.joints())
        if position_error < 0.01:
            self.move_joints(joints, seconds=ASIDE_SECONDS)
        else:
            self.go_home()

    def go_home(self):
        self.move_joints(
            self.scene.home_arms[:7], self.scene.home_arms[7], left=self.scene.home_arms[8:]
        )
        self.hold(self.scene.home_arms[:7], self.scene.home_arms[7], SETTLE_SECONDS)

    def push_cube(self, cube_xy, target_xy):
        """Push the cube with the closed gripper. Return False if it wasn't pushed."""
        cube_xy = np.asarray(cube_xy, dtype=float)
        target_xy = np.asarray(target_xy, dtype=float)
        path = target_xy - cube_xy
        distance = np.linalg.norm(path)
        if distance < CLOSE_ENOUGH:
            self.failure = "already close"
            return False
        heading = np.arctan2(path[1], path[0])
        contact = CUBE_HALF + PUSHER_RADIUS
        self.failure = "the tray is in the way"
        for angle in PUSH_ANGLES:
            direction = np.array([np.cos(heading + angle), np.sin(heading + angle)])
            start_xy = self._clear_start(cube_xy, direction, contact)
            if start_xy is None:
                continue
            # Stop at the point of this line that is closest to the target.
            end_xy = cube_xy + direction * (distance * np.cos(angle) - contact)
            above = np.array([*start_xy, CLEAR_Z])
            low = np.array([*start_xy, PUSH_Z])
            end = np.array([*end_xy, PUSH_Z])
            plan = self.plan_push(above, low, end, direction)
            if plan is None:
                self.failure = "no reachable path"
                continue
            self._execute_push(plan, end, end_xy)
            return True
        return False

    @staticmethod
    def _clear_start(cube_xy, direction, contact):
        """Return where to come down for a push, away from the tray, if there is a place."""
        for gap in PUSH_START_GAPS:
            start_xy = cube_xy - direction * (contact + gap)
            if not (np.all(start_xy > TRAY_MIN_XY) and np.all(start_xy < TRAY_MAX_XY)):
                return start_xy
        return None

    def _execute_push(self, plan, end, end_xy):
        first, descent, push, preferred = plan
        self.move_joints(first, GRIPPER_CLOSED)
        self.hold(first, GRIPPER_CLOSED, SETTLE_SECONDS)
        for joints in descent:
            self._command(joints, GRIPPER_CLOSED)
        self.hold(descent[-1], GRIPPER_CLOSED, SETTLE_SECONDS)
        for joints in push:
            self._command(joints, GRIPPER_CLOSED)
        rise = self.plan_line(
            push[-1], end, np.array([*end_xy, CLEAR_Z]), PUSH_SPEED * 2, preferred
        )
        for joints in rise or []:
            self._command(joints, GRIPPER_CLOSED)

    def hold(self, joints, gripper, seconds):
        for _ in range(round(seconds / FRAME_SECONDS)):
            self._command(joints, gripper)

    def plan_push(self, above, low, end, direction):
        """Plan the whole push before moving, leaning as far as the joints allow."""
        near = self.joints()
        seeds = [near] + list(self.rng.uniform(self.limits[:, 0], self.limits[:, 1], (60, 7)))
        candidates = [push_tool_direction(direction, lean) for lean in PUSH_LEANS]
        # Fine for pushing away from the robot and sideways.
        candidates.append(rest_tool_direction(low))
        for preferred in candidates:
            best = None
            for seed in seeds:
                first, position_error, tilt = self.solve(above, seed, preferred)
                if position_error > 0.002 or tilt > MAX_TOOL_TILT:
                    continue
                distance = float(np.linalg.norm(first - near))
                if best is not None and distance >= best[0]:
                    continue
                descent = self.plan_line(first, above, low, PUSH_SPEED * 2, preferred)
                push = descent and self.plan_line(descent[-1], low, end, PUSH_SPEED, preferred)
                if push:
                    best = (distance, first, descent, push, preferred)
            if best is not None:
                return best[1:]
        return None
