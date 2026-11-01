#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Copy a LeRobot v3.0 dataset with its videos scaled by a factor.

Frame counts and timestamps don't change. The image statistics are kept because
scaling barely changes the per-channel mean and standard deviation.
"""

import argparse
import json
import pathlib
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--factor", type=float, default=0.5)
    args = parser.parse_args()

    shutil.copytree(args.input, args.output, ignore=shutil.ignore_patterns("*.mp4"))
    for video in sorted(args.input.rglob("*.mp4")):
        output = args.output / video.relative_to(args.input)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                "-vf", f"scale=trunc(iw*{args.factor}/2)*2:trunc(ih*{args.factor}/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "20",
                str(output),
            ],
            check=True,
        )

    info_path = args.output / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for feature in info["features"].values():
        if feature.get("dtype") != "video":
            continue
        height, width, channels = feature["shape"]
        height = int(height * args.factor / 2) * 2
        width = int(width * args.factor / 2) * 2
        feature["shape"] = [height, width, channels]
        feature["info"]["video.height"] = height
        feature["info"]["video.width"] = width
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
