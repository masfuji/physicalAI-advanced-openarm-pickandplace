# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""dora-openarm-data-collection-ui with a button to randomize the scene objects.

Outputs (in addition to the ones of dora-openarm-data-collection-ui)
--------------------------------------------------------------------
randomize : bool[1]
    Sent when the button is pressed. Ignored while an episode is being recorded
    so that the objects don't jump in the recorded data.
"""

import dora_openarm_data_collection_ui.main as ui
import jinja2
import pyarrow as pa
from fastapi import Request
from fastapi.responses import RedirectResponse

_TEMPLATE = "root.html"
# The button is put on the panel that is only shown while not recording.
_INSERT_BEFORE = '<form id="skip-form"'
_BUTTON = """<form id="randomize-form" action="{{ url_for("_randomize") }}" method="post">
        <input class="big-button start" type="submit" value="Randomize Cube">
      </form>
      """


@ui.app.post("/randomize")
def _randomize(request: Request):
    """Ask the simulator to randomize the scene objects."""
    if not ui.state.collecting:
        ui.node.send_output("randomize", pa.array([True]))
    return RedirectResponse(request.url_for("_root"), 303)


def _load_template(loader, environment):
    """Load the upstream template with the button inserted."""
    source, filename, uptodate = loader.get_source(environment, _TEMPLATE)
    if source.count(_INSERT_BEFORE) != 1:
        raise RuntimeError(f"{_INSERT_BEFORE!r} not found once in {filename}")
    return source.replace(_INSERT_BEFORE, _BUTTON + _INSERT_BEFORE), filename, uptodate


def main():
    """Run the data collection UI with the additional button."""
    environment = ui.templates.env
    upstream = environment.loader
    environment.loader = jinja2.FunctionLoader(
        lambda name: _load_template(upstream, environment)
        if name == _TEMPLATE
        else upstream.get_source(environment, name)
    )
    # Fail on startup, not on the first page view, if upstream has changed.
    environment.get_template(_TEMPLATE)
    ui.main()


if __name__ == "__main__":
    main()
