#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Run lerobot-train on Windows without the privilege to create symlinks.

Takes the same arguments as lerobot-train. The checkpoints/last link is not
created, so resume from a checkpoint directory by its step number.
"""

import lerobot.scripts.lerobot_train as train

train.update_last_checkpoint = lambda checkpoint_dir: None

if __name__ == "__main__":
    train.main()
