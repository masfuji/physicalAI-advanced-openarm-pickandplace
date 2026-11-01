# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Dora node that merges the controller's X button and a UI request.

Inputs
------
button_x : bool[1]
    X button state from the VR controller.
randomize : bool[1]
    Request from the UI.

Outputs
-------
button_x : bool[1]
    X button state, sent only when it changes: a slow subscriber would drop
    presses from a continuous stream. A UI request is sent as a press followed
    by a release.
"""

import dora
import pyarrow as pa


def main():
    node = dora.Node()
    pressed = False
    for event in node:
        if event["type"] != "INPUT":
            continue
        if event["id"] == "button_x":
            value = bool(event["value"][0].as_py())
            if value != pressed:
                pressed = value
                node.send_output("button_x", pa.array([value]), event["metadata"])
        elif event["id"] == "randomize":
            node.send_output("button_x", pa.array([True]))
            node.send_output("button_x", pa.array([False]))


if __name__ == "__main__":
    main()
