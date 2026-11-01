# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Dora node that shows a JPEG camera stream in a desktop window.

Inputs
------
image : uint8[N]
    JPEG-encoded frame.
"""

import argparse

import cv2
import dora
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="Camera", help="Window title")
    parser.add_argument("--width", type=int, default=960, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    args = parser.parse_args()

    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.title, args.width, args.height)
    window_open = True

    for event in dora.Node():
        if event["type"] != "INPUT" or event["id"] != "image":
            continue
        # Keep draining inputs after the window is closed so that the other
        # nodes in the dataflow are not affected.
        if not window_open:
            continue
        frame = cv2.imdecode(np.asarray(event["value"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imshow(args.title, frame)
        cv2.waitKey(1)
        window_open = cv2.getWindowProperty(args.title, cv2.WND_PROP_VISIBLE) >= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
