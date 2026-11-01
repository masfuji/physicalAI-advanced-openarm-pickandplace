#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Write an interactive 3D HTML plot of the gripper trajectories in a dataset.

The observed joint positions are converted to the position between the fingers
with MuJoCo forward kinematics.

Usage:
    (from the training directory)
    uv run --no-sync python ../scripts/plot_trajectories.py \
        ../config/data/pick_place_cube/dataset
"""

import argparse
import json
import pathlib

import mujoco
import numpy as np
import openarm_dataset
from plotly.offline import get_plotlyjs

ARMS = ("right", "left")
# Distance from the end-effector control point to between the fingertips.
GRIP_POINT = np.array([0.0, 0.0, -0.14])
# |gripper angle| below this is drawn as closed. Open is 0.785 rad, closed on
# the 4 cm cube is about 0.2 rad.
CLOSED_BELOW = 0.4
# The observations are about 38 Hz. This keeps the page responsive.
STRIDE = 2


def forward_kinematics(model, data, arm, qpos):
    """Return the grip point positions in the world frame."""
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_ee_control_point")
    addresses = [
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm}_joint{i}")
        ]
        for i in range(1, 8)
    ]
    positions = np.empty((len(qpos), 3))
    for index, joints in enumerate(qpos):
        data.qpos[addresses] = joints
        mujoco.mj_kinematics(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        positions[index] = data.site_xpos[site] + rotation @ GRIP_POINT
    return positions


def load_episodes(dataset, model, data):
    """Return the plot data of all episodes."""
    episodes = []
    for episode in dataset.meta.episodes:
        obs = dataset.load_obs(episode)
        arms = {}
        for arm in ARMS:
            frame = obs[f"arms/{arm}/qpos"].iloc[::STRIDE]
            joints = frame.iloc[:, :7].to_numpy()
            gripper = frame["gripper"].to_numpy()
            time = (frame.index - frame.index[0]).total_seconds().to_numpy()
            position = forward_kinematics(model, data, arm, joints)
            arms[arm] = {
                "x": position[:, 0].round(4).tolist(),
                "y": position[:, 1].round(4).tolist(),
                "z": position[:, 2].round(4).tolist(),
                "t": time.round(2).tolist(),
                "closed": (np.abs(gripper) < CLOSED_BELOW).tolist(),
                "path": float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
                "duration": float(time[-1]),
                "closed_seconds": float(
                    (np.abs(gripper) < CLOSED_BELOW).mean() * time[-1]
                ),
            }
        episodes.append(
            {"id": episode["id"], "success": bool(episode["success"]), "arms": arms}
        )
    return episodes


def body_position(model, data, name):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body].round(4).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument(
        "--scene",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent / "config/scenes/pick_place.xml",
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    scene = {
        "cube": body_position(model, data, "orange_cube"),
        "frame": body_position(model, data, "black_frame"),
    }

    dataset = openarm_dataset.Dataset(str(args.dataset))
    payload = {"episodes": load_episodes(dataset, model, data), "scene": scene}

    template = (pathlib.Path(__file__).parent / "plot_trajectories.html").read_text(
        encoding="utf-8"
    )
    html = template.replace("/*PLOTLY_JS*/", get_plotlyjs()).replace(
        "/*DATA*/null", json.dumps(payload, separators=(",", ":"))
    )
    output = args.output or args.dataset.parent / "trajectories.html"
    output.write_text(html, encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
