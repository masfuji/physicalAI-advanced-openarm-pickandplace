#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Rank targets by the weighted score of the cube starts around them, as a push lands about 2 cm off."""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path("outputs/eval_abc")
RUNS = {15: "a_act_only", 30: "a3_act_only_30s", 45: "a2_act_only_45s"}
WEIGHTS = {15: 3, 30: 2, 45: 1}
eps = {k: json.loads((ROOT / v / "results.json").read_text(encoding="utf-8"))["episodes"] for k, v in RUNS.items()}
xy = np.array([e["cube_start"] for e in eps[15]])
score = sum(WEIGHTS[k] * np.array([e["placed"] for e in eps[k]], float) for k in RUNS) / sum(WEIGHTS.values())
rows = []
for x in np.arange(0.49, 0.5701, 0.005):
    for y in np.arange(-0.04, 0.0401, 0.005):
        d = np.linalg.norm(xy - [x, y], axis=1)
        a, b = d < 0.02, d < 0.03
        if a.sum() < 8:
            continue
        rows.append((round(score[b].mean(), 3), round(score[a].mean(), 3), int(a.sum()), int(b.sum()), round(float(x), 3), round(float(y), 3), round(float(score[d < 0.04].mean()), 3)))
rows.sort(reverse=True)
print("score_3cm score_2cm n2 n3 x y score_4cm")
for row in rows[:12]:
    print(row)
for target in ((0.53, -0.025), (0.52, -0.01), (0.49, 0.005)):
    d = np.linalg.norm(xy - target, axis=1)
    print(target, {r: (round(float(score[d < r].mean()), 3), int((d < r).sum())) for r in (0.02, 0.03, 0.04)})
