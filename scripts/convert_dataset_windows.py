#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Run openarm-dataset-convert on Windows.

openarm_dataset opens a NamedTemporaryFile and writes to it by its name. By
default, Windows refuses to open such a file again while it is open. Takes the
same arguments as openarm-dataset-convert.
"""

import functools
import tempfile

from openarm_dataset.convert import main

tempfile.NamedTemporaryFile = functools.partial(
    tempfile.NamedTemporaryFile, delete_on_close=False
)

if __name__ == "__main__":
    main()
