#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Overlay the ACT-only success rate per cube start position on the ceiling camera."""
import json
import math
import pathlib
import sys

import cv2
import mujoco
import numpy as np

SCENE = pathlib.Path("../config/scenes/pick_place.xml")
RESULTS = pathlib.Path(sys.argv[1])
OUTPUT = pathlib.Path(sys.argv[2])
BINS = int(sys.argv[3])
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.48, 0.58, -0.05, 0.05
TABLE_Z = 1.005
WIDTH, HEIGHT = 3840, 2160

episodes = json.loads(RESULTS.read_text(encoding="utf-8"))["episodes"]
xs = np.linspace(X_MIN, X_MAX, BINS + 1)
ys = np.linspace(Y_MIN, Y_MAX, BINS + 1)
total = np.zeros((BINS, BINS), int)
placed = np.zeros((BINS, BINS), int)
for episode in episodes:
    x, y = episode["cube_start"]
    i = min(max(np.searchsorted(xs, x, side="right") - 1, 0), BINS - 1)
    j = min(max(np.searchsorted(ys, y, side="right") - 1, 0), BINS - 1)
    total[i, j] += 1
    placed[i, j] += episode["placed"]
print("episodes", len(episodes), "placed", int(placed.sum()))
for i in reversed(range(BINS)):
    print(
        f"x {xs[i]:.3f}-{xs[i + 1]:.3f}: "
        + "  ".join(f"{placed[i, j]}/{total[i, j]}" for j in reversed(range(BINS)))
    )
print("columns: y from", Y_MAX, "to", Y_MIN)
for name, (x0, x1, y0, y1) in {
    "report good x.51-.55 y-.05-0": (0.51, 0.55, -0.05, 0.0),
    "report weak x.55-.59 y0-.055": (0.55, 0.59, 0.0, 0.055),
}.items():
    chosen = [e for e in episodes if x0 <= e["cube_start"][0] < x1 and y0 <= e["cube_start"][1] < y1]
    print(name, sum(e["placed"] for e in chosen), "/", len(chosen))

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
cell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cell_vis")
if cell >= 0:
    model.geom_rgba[cell, 3] = 0.2
home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
for geom in range(model.ngeom):
    if model.geom_bodyid[geom] == body:
        model.geom_rgba[geom, 3] = 0.0
mujoco.mj_resetDataKeyframe(model, data, home)
mujoco.mj_forward(model, data)
model.vis.global_.offwidth = WIDTH
model.vis.global_.offheight = HEIGHT
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
renderer.update_scene(data, camera="camera_ceiling")
image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
renderer.close()

camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_ceiling")
focal = 0.5 * HEIGHT / math.tan(math.radians(model.cam_fovy[camera]) / 2)
rotation = data.cam_xmat[camera].reshape(3, 3)
origin = data.cam_xpos[camera]


def pixel(x, y):
    local = rotation.T @ (np.array([x, y, TABLE_Z]) - origin)
    return (
        int(round(WIDTH / 2 + focal * local[0] / -local[2])),
        int(round(HEIGHT / 2 - focal * local[1] / -local[2])),
    )


# Light to dark blue: darker is a higher success rate.
LIGHT = np.array([247, 232, 214], float)  # BGR
DARK = np.array([115, 60, 20], float)
overlay = image.copy()
for i in range(BINS):
    for j in range(BINS):
        corners = np.array(
            [pixel(xs[i], ys[j]), pixel(xs[i + 1], ys[j]), pixel(xs[i + 1], ys[j + 1]), pixel(xs[i], ys[j + 1])]
        )
        if total[i, j]:
            rate = placed[i, j] / total[i, j]
            cv2.fillConvexPoly(overlay, corners, (LIGHT + (DARK - LIGHT) * rate).tolist(), cv2.LINE_AA)
image = cv2.addWeighted(overlay, 0.9, image, 0.1, 0)
for i in range(BINS):
    for j in range(BINS):
        corners = np.array(
            [pixel(xs[i], ys[j]), pixel(xs[i + 1], ys[j]), pixel(xs[i + 1], ys[j + 1]), pixel(xs[i], ys[j + 1])]
        )
        cv2.polylines(image, [corners], True, (255, 255, 255), 2, cv2.LINE_AA)
left, top = pixel(X_MAX + 0.11, Y_MAX + 0.25)
right, bottom = pixel(X_MIN - 0.11, Y_MIN - 0.25)
scene_panel = image[max(top, 0) : bottom, max(left, 0) : right]

# The enlarged map, oriented like the camera: +x is up and +y is left.
INK = (56, 42, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX
height = scene_panel.shape[0]
size = (height - 170) // BINS
panel = np.full((height, size * BINS + 210, 3), 255, np.uint8)
x0, y0 = 150, 50
for i in range(BINS):
    for j in range(BINS):
        column = BINS - 1 - j
        row = BINS - 1 - i
        top_left = (x0 + column * size, y0 + row * size)
        bottom_right = (top_left[0] + size, top_left[1] + size)
        rate = placed[i, j] / total[i, j] if total[i, j] else 0
        cv2.rectangle(panel, top_left, bottom_right, (LIGHT + (DARK - LIGHT) * rate).tolist(), -1)
        cv2.rectangle(panel, top_left, bottom_right, (255, 255, 255), 3)
        text = f"{placed[i, j]}/{total[i, j]}"
        (w, h), _ = cv2.getTextSize(text, FONT, 1.1, 2)
        color = (255, 255, 255) if rate >= 0.5 else INK
        cv2.putText(
            panel, text, (top_left[0] + (size - w) // 2, top_left[1] + (size + h) // 2),
            FONT, 1.1, color, 2, cv2.LINE_AA,
        )
for index in range(BINS + 1):
    label = f"{xs[BINS - index]:.3f}"
    cv2.putText(panel, label, (40, y0 + index * size + 10), FONT, 0.7, INK, 1, cv2.LINE_AA)
    label = f"{ys[BINS - index]:+.3f}"
    (w, _), _ = cv2.getTextSize(label, FONT, 0.7, 1)
    cv2.putText(panel, label, (x0 + index * size - w // 2, y0 + BINS * size + 35), FONT, 0.7, INK, 1, cv2.LINE_AA)
cv2.putText(panel, "x [m]", (40, 30), FONT, 0.75, INK, 2, cv2.LINE_AA)
cv2.putText(panel, "y [m]", (x0 + BINS * size // 2 - 35, y0 + BINS * size + 75), FONT, 0.75, INK, 2, cv2.LINE_AA)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT), np.hstack([scene_panel, panel]))
