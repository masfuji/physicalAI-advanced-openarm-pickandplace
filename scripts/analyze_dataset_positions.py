#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Estimate the cube's start in each training episode from the ceiling video.

The detector is first validated on evaluation videos, whose true positions are known.
"""
import json
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "../scripts")
from evaluate_act import Scene

DATA = pathlib.Path("data/pick_place_cube_lerobot_half")
EVAL = pathlib.Path("outputs/eval_abc")
OUT = pathlib.Path(sys.argv[1])
scene = Scene(0.0)
scene.reset(None, np.random.default_rng(0))
full = scene.renderers["ceiling"]
FULL_W, FULL_H = full.width, full.height


def detect(frame_bgr):
    """Return the centre of the cube's bright top face in full-size pixels, or None."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (8, 150, 170), (32, 255, 255))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 10:
        return None
    u, v = centroids[best]
    return (u + 0.5) * FULL_W / w, (v + 0.5) * FULL_H / h, int(stats[best, cv2.CC_STAT_AREA])


def first_frame(path, seconds=0.0):
    capture = cv2.VideoCapture(str(path))
    if seconds:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


# Validation on evaluation runs, also at half size like the dataset.
results = json.loads((EVAL / "a_act_only/results.json").read_text(encoding="utf-8"))["episodes"]
errors, errors_half, offsets = [], [], []
for index, episode in enumerate(results):
    frame = first_frame(EVAL / f"a_act_only/episode_{index:02d}.mp4")
    found = detect(frame)
    if found is None:
        continue
    xy = scene.ceiling_pixel_to_table(found[0], found[1])
    errors.append(np.linalg.norm(xy - episode["cube_start"]))
    offsets.append(xy - np.array(episode["cube_start"]))
    half = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    found = detect(half)
    if found is not None:
        errors_half.append(np.linalg.norm(scene.ceiling_pixel_to_table(found[0], found[1]) - episode["cube_start"]))
offset = np.mean(offsets, axis=0)
print(f"validation n={len(errors)} error cm median={np.median(errors) * 100:.2f} max={np.max(errors) * 100:.2f}; "
      f"half n={len(errors_half)} median={np.median(errors_half) * 100:.2f} max={np.max(errors_half) * 100:.2f}; mean offset {np.round(offset, 4)}")

meta = pd.concat(pd.read_parquet(p) for p in sorted((DATA / "meta/episodes").rglob("*.parquet")))
print("meta columns", [c for c in meta.columns if "ceiling" in c or c in ("episode_index", "length")])
key = "videos/observation.images.ceiling"
rows = []
for _, m in meta.iterrows():
    path = DATA / f"videos/observation.images.ceiling/chunk-{int(m[key + '/chunk_index']):03d}/file-{int(m[key + '/file_index']):03d}.mp4"
    frame = first_frame(path, float(m[key + "/from_timestamp"]) + 0.02)
    found = detect(frame) if frame is not None else None
    if found is None:
        rows.append(dict(episode=int(m["episode_index"]), length=int(m["length"]), x=None, y=None, area=None))
        continue
    xy = scene.ceiling_pixel_to_table(found[0], found[1]) - offset
    rows.append(dict(episode=int(m["episode_index"]), length=int(m["length"]), x=round(float(xy[0]), 3), y=round(float(xy[1]), 3), area=found[2], size=f"{frame.shape[1]}x{frame.shape[0]}"))
OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
frame_df = pd.DataFrame(rows)
print(frame_df.to_string())
print(frame_df[["x", "y", "length"]].describe().round(3).to_string())

xs = np.linspace(0.48, 0.58, 5)
ys = np.linspace(-0.05, 0.05, 5)
counts = np.zeros((4, 4), int)
outside = 0
for _, r in frame_df.dropna(subset=["x"]).iterrows():
    if not (xs[0] <= r.x <= xs[-1] and ys[0] <= r.y <= ys[-1]):
        outside += 1
        continue
    i = min(np.searchsorted(xs, r.x, side="right") - 1, 3)
    j = min(np.searchsorted(ys, r.y, side="right") - 1, 3)
    counts[i, j] += 1
print("outside eval area", outside)
for i in reversed(range(4)):
    print(f"x {xs[i]:.3f}-{xs[i + 1]:.3f}: " + "  ".join(str(counts[i, j]) for j in reversed(range(4))))
print("columns: y from +0.05 to -0.05")
scene.close()
