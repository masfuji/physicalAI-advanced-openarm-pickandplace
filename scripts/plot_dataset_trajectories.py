#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Write an interactive 3D plot of the training episodes with the cube's random area."""
import json
import pathlib
import sys

import mujoco
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA = pathlib.Path("data/pick_place_cube_lerobot_half")
POSITIONS = pathlib.Path(sys.argv[1])
OUTPUT = pathlib.Path(sys.argv[2])
SCENE = "../config/scenes/pick_place.xml"
GRIP_POINT = np.array([0.0, 0.0, -0.14])  # plot_trajectories.py
CLOSED_BELOW = 0.4
STRIDE = 2
TABLE_Z = 1.005
CUBE_REST_Z = 1.025
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.48, 0.58, -0.05, 0.05
TRAY = (0.55, 0.15, 0.07)
TARGET = (0.53, -0.025)
INK = "#1E2A38"
OPEN_COLOR = "#2A6496"
CLOSED_COLOR = "#B4500A"
LEFT_COLOR = "#8A95A1"

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)


def kinematics(arm, joints):
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_ee_control_point")
    addresses = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm}_joint{i}")]
        for i in range(1, 8)
    ]
    out = np.empty((len(joints), 3))
    for index, values in enumerate(joints):
        data.qpos[addresses] = values
        mujoco.mj_kinematics(model, data)
        out[index] = data.site_xpos[site] + data.site_xmat[site].reshape(3, 3) @ GRIP_POINT
    return out


frames = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"]) for p in sorted((DATA / "data").rglob("*.parquet")))
starts = {r["episode"]: r for r in json.loads(POSITIONS.read_text())}
figure = go.Figure()

# The table's plane in the area around the task, the random area and the tray.
square = lambda x0, x1, y0, y1, z: ([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5)
x, y, z = square(X_MIN, X_MAX, Y_MIN, Y_MAX, CUBE_REST_Z)
figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#0B7FAB", width=7), name="Cube の出現範囲（±5 cm）", hoverinfo="name"))
figure.add_trace(go.Mesh3d(x=[X_MIN, X_MAX, X_MAX, X_MIN], y=[Y_MIN, Y_MIN, Y_MAX, Y_MAX], z=[CUBE_REST_Z] * 4, i=[0, 0], j=[1, 2], k=[2, 3], color="#0B7FAB", opacity=0.18, hoverinfo="skip", showlegend=False))
x, y, z = square(TRAY[0] - TRAY[2], TRAY[0] + TRAY[2], TRAY[1] - TRAY[2], TRAY[1] + TRAY[2], TABLE_Z + 0.04)
figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=INK, width=7), name="枠", hoverinfo="name"))

xy = np.array([[starts[e]["x"], starts[e]["y"]] for e in sorted(starts)])
figure.add_trace(go.Scatter3d(
    x=xy[:, 0], y=xy[:, 1], z=np.full(len(xy), CUBE_REST_Z), mode="markers",
    marker=dict(size=4, color=CLOSED_COLOR, line=dict(color="#F7F6F2", width=1)),
    name="学習データの Cube 初期位置（26 本）",
    text=[f"episode {e}<br>x {starts[e]['x']:.3f}, y {starts[e]['y']:.3f}" for e in sorted(starts)], hoverinfo="text",
))
figure.add_trace(go.Scatter3d(x=[TARGET[0]], y=[TARGET[1]], z=[CUBE_REST_Z], mode="markers", marker=dict(size=6, symbol="diamond", color=INK), name="スライドの目標位置", hovertext="x 0.53, y -0.025", hoverinfo="text"))

gaps = []
for episode, frame in frames.groupby("episode_index"):
    frame = frame.sort_values("frame_index")
    state = np.stack(frame["observation.state"].to_numpy())[::STRIDE]
    time = frame["frame_index"].to_numpy()[::STRIDE] / 30
    right = kinematics("right", state[:, :7])
    closed = np.abs(state[:, 7]) < CLOSED_BELOW
    text = [f"episode {episode}<br>{t:.1f} 秒" for t in time]
    # Sanity check: where the fingers first close should be at the detected cube.
    first = int(np.argmax(closed))
    gaps.append(np.linalg.norm(right[first, :2] - xy[list(sorted(starts)).index(int(episode))]))
    group = f"episode {episode}"
    figure.add_trace(go.Scatter3d(x=right[:, 0], y=right[:, 1], z=right[:, 2], mode="lines", line=dict(color=OPEN_COLOR, width=2), opacity=0.55, name=group, legendgroup=group, text=text, hoverinfo="text"))
    held = right.copy()
    held[~closed] = np.nan
    figure.add_trace(go.Scatter3d(x=held[:, 0], y=held[:, 1], z=held[:, 2], mode="lines", line=dict(color=CLOSED_COLOR, width=4), opacity=0.85, name=group + " 閉", legendgroup=group, showlegend=False, text=text, hoverinfo="text"))
    left = kinematics("left", state[:, 8:15])
    figure.add_trace(go.Scatter3d(x=left[:, 0], y=left[:, 1], z=left[:, 2], mode="lines", line=dict(color=LEFT_COLOR, width=2), opacity=0.4, name="左手（全エピソード）", legendgroup="left", showlegend=int(episode) == 0, visible="legendonly", text=text, hoverinfo="text"))

print("episodes", len(gaps), "fingers-first-close to detected cube cm: median %.2f max %.2f" % (np.median(gaps) * 100, np.max(gaps) * 100))
print("right grip point ranges", {k: (round(float(np.nanmin(v)), 3), round(float(np.nanmax(v)), 3)) for k, v in dict(x=right[:, 0], y=right[:, 1], z=right[:, 2]).items()})

figure.update_layout(
    title=dict(text="学習データの右手の軌道（青：指が開く、オレンジ：指が閉じる）", font=dict(color=INK, size=18)),
    font=dict(family="Noto Sans JP, Meiryo, sans-serif", color=INK),
    paper_bgcolor="#F7F6F2",
    legend=dict(itemsizing="constant", font=dict(size=11)),
    margin=dict(l=0, r=0, t=50, b=0),
    scene=dict(
        aspectmode="data",
        xaxis=dict(title="x [m]（奥）", backgroundcolor="#ECEAE3", gridcolor="#D8D5CC"),
        yaxis=dict(title="y [m]（枠側が +）", backgroundcolor="#ECEAE3", gridcolor="#D8D5CC", autorange="reversed"),
        zaxis=dict(title="z [m]", backgroundcolor="#ECEAE3", gridcolor="#D8D5CC"),
        camera=dict(eye=dict(x=-1.3, y=-0.9, z=1.1)),
    ),
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
figure.write_html(OUTPUT, include_plotlyjs=True)
print(OUTPUT.resolve(), round(OUTPUT.stat().st_size / 1e6, 1), "MB", "traces", len(figure.data))
