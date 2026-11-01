#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Write an interactive plot of evaluate_act.py trajectories, split by action chunk.

Usage (from the training directory):
    uv run --no-sync python ../scripts/plot_eval_trajectories.py \
        outputs/eval_trajectories/home_start outputs/eval_trajectories/dataset_start \
        --output outputs/eval_trajectories/chunks.html
"""

import argparse
import json
import pathlib

import mujoco
import numpy as np
from plot_trajectories import ARMS, body_position, forward_kinematics
from plotly.offline import get_plotlyjs

SCENE = pathlib.Path(__file__).parent.parent / "config/scenes/pick_place.xml"
FPS = 30


def load_arm(model, data, side_index, arm, trajectory, chunk_length):
    """Return the plot data of one arm: per-frame series and per-chunk numbers."""
    offset = side_index * 8
    state = trajectory["state"][:, offset : offset + 8]
    action = trajectory["action"][:, offset : offset + 8]
    actual = forward_kinematics(model, data, arm, state[:, :7])
    command = forward_kinematics(model, data, arm, action[:, :7])

    chunks = []
    for start in range(0, len(state), chunk_length):
        end = min(start + chunk_length, len(state))
        path = actual[start:end]
        chunks.append(
            {
                "seconds": [round(start / FPS, 2), round(end / FPS, 2)],
                "path_m": round(
                    float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 3
                ),
                # How far the new plan's first target is from where the arm is.
                "start_gap_m": round(float(np.linalg.norm(command[start] - actual[start])), 3),
                # How far the plan's target leaps from the end of the previous plan.
                "command_jump_m": round(
                    float(np.linalg.norm(command[start] - command[start - 1])), 3
                )
                if start
                else None,
            }
        )
    return {
        "actual": actual.round(4).T.tolist(),
        "command": command.round(4).T.tolist(),
        "gripper": np.abs(state[:, 7]).round(3).tolist(),
        "gripper_command": np.abs(action[:, 7]).round(3).tolist(),
        "chunks": chunks,
    }


def load_condition(directory, model, data):
    """Return the plot data of all episodes of one evaluation."""
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    episodes = []
    for index, result in enumerate(results["episodes"]):
        trajectory = np.load(directory / f"episode_{index:02d}.npz")
        chunk_length = int(trajectory["n_action_steps"])
        episodes.append(
            {
                "id": index,
                "placed": result["placed"],
                "lifted": result["lifted"],
                "chunk_length": chunk_length,
                "cube": trajectory["cube"].round(4).T.tolist(),
                "arms": {
                    arm: load_arm(model, data, i, arm, trajectory, chunk_length)
                    for i, arm in enumerate(ARMS)
                },
            }
        )
    return {"name": directory.name, "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=pathlib.Path, nargs="+")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    payload = {
        "fps": FPS,
        "scene": {
            "cube": body_position(model, data, "orange_cube"),
            "frame": body_position(model, data, "black_frame"),
        },
        "conditions": [load_condition(d, model, data) for d in args.directories],
    }

    template = (pathlib.Path(__file__).parent / "plot_eval_trajectories.html").read_text(
        encoding="utf-8"
    )
    html = template.replace("/*PLOTLY_JS*/", get_plotlyjs()).replace(
        "/*DATA*/null", json.dumps(payload, separators=(",", ":"))
    )
    args.output.write_text(html, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
