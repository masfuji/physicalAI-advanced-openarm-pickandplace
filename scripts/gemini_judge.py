#!/usr/bin/env python3
# Copyright (c) 2026 masfuji
# SPDX-License-Identifier: MIT

"""Judge whether the pick and place task succeeded with Gemini Robotics-ER 2.

GEMINI_API_KEY is read from the environment or from .env at the repository root.

Used by evaluate_act.py --gemini-judge. This also judges the last frames of an
evaluate_act.py output directory and compares them with the simulator's results:
    .venv/Scripts/python.exe ../scripts/gemini_judge.py \
        outputs/eval_cube_shift/y0_home
"""

import argparse
import json
import os
import pathlib
import time

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

ENV_FILE = pathlib.Path(__file__).resolve().parents[1] / ".env"
MODEL = "gemini-robotics-er-2-preview"
REQUEST_TIMEOUT_MS = 60_000
# A request that fails ends the retries of an episode: wait out rate limits instead.
REQUEST_ATTEMPTS = 4
RETRY_WAIT_SECONDS = 15
PROMPT = """These images show the end of a robot episode, one image per camera: {cameras}.
The task was: pick up the orange cube and place it inside the black square tray.

The task succeeded only if the orange cube is resting inside the black tray and
no gripper is holding it. A cube that is outside the tray, on top of a tray wall,
in the air, or still gripped is a failure.

Describe where the cube is, then give the verdict."""
LOCATE_PROMPT = """Point to the center of the orange cube in this image.
Give y and x normalized to 0-1000, with the origin at the top left of the image.
Set found to false if no orange cube is visible."""


class Verdict(BaseModel):
    """The structured answer. The observation comes first to ground the verdict."""

    observation: str
    success: bool


class CubePoint(BaseModel):
    """Where the cube is, in 0-1000 normalized image coordinates."""

    found: bool
    y: int
    x: int


class GeminiJudge:
    """Asks Gemini whether camera images show a completed task."""

    def __init__(self):
        load_dotenv(ENV_FILE)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(f"GEMINI_API_KEY is not set: add it to {ENV_FILE}")
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        )
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

    def _generate(self, contents, schema):
        return self.request(
            contents,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
            ),
        )

    def request(self, contents, config):
        """Send a request, trying again if it fails."""
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            try:
                response = self.client.models.generate_content(
                    model=MODEL, contents=contents, config=config
                )
                metadata = response.usage_metadata
                self.usage["requests"] += 1
                self.usage["input_tokens"] += metadata.prompt_token_count or 0
                self.usage["output_tokens"] += metadata.candidates_token_count or 0
                self.usage["thinking_tokens"] += metadata.thoughts_token_count or 0
                return response
            except Exception as error:
                if attempt == REQUEST_ATTEMPTS:
                    raise
                print(f"request failed, trying again: {type(error).__name__}", flush=True)
                time.sleep(RETRY_WAIT_SECONDS * attempt)

    def judge(self, images):
        """Return the verdict for {camera name: RGB image}.

        A failed request is returned as gemini_error instead of being raised: it
        must neither end an evaluation nor be counted as a failed task.
        """
        parts = [PROMPT.format(cameras=", ".join(images))]
        for image in images.values():
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("failed to encode an image")
            parts.append(types.Part.from_bytes(data=jpeg.tobytes(), mime_type="image/jpeg"))
        try:
            verdict = self._generate(parts, Verdict).parsed
            if verdict is None:
                return {"gemini_error": f"unparsable response: {response.text!r}"}
        except Exception as error:
            return {"gemini_error": f"{type(error).__name__}: {error}"}
        return {
            "gemini_success": verdict.success,
            "gemini_observation": verdict.observation,
        }

    def locate_cube(self, image):
        """Return the cube's pixel (u, v) in an RGB image, or None if not found."""
        self.locate_error = None
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("failed to encode an image")
        try:
            point = self._generate(
                [
                    LOCATE_PROMPT,
                    types.Part.from_bytes(data=jpeg.tobytes(), mime_type="image/jpeg"),
                ],
                CubePoint,
            ).parsed
        except Exception as error:
            print(f"locate_cube failed: {type(error).__name__}: {error}", flush=True)
            self.locate_error = error
            return None
        if point is None or not point.found:
            return None
        height, width = image.shape[:2]
        return point.x / 1000 * width, point.y / 1000 * height


def summarize(results):
    """Compare the verdicts with the simulator's placed flag at the same moment."""
    judged = [r for r in results if "gemini_success" in r]
    return {
        "judged": len(judged),
        "errors": len(results) - len(judged),
        "agreed": sum(r["gemini_success"] == r["placed"] for r in judged),
        "said_success_but_failed": sum(r["gemini_success"] and not r["placed"] for r in judged),
        "said_failure_but_placed": sum(r["placed"] and not r["gemini_success"] for r in judged),
    }


def last_frame(path):
    """Return the last readable frame of a video as an RGB image."""
    capture = cv2.VideoCapture(str(path))
    frame = None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame = image
    capture.release()
    if frame is None:
        raise RuntimeError(f"no frames in {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--episodes", type=int, help="judge only the first ones")
    args = parser.parse_args()

    judge = GeminiJudge()
    episodes = json.loads((args.directory / "results.json").read_text(encoding="utf-8"))[
        "episodes"
    ]
    results = []
    for index, episode in enumerate(episodes[: args.episodes]):
        image = last_frame(args.directory / f"episode_{index:02d}.mp4")
        result = {"episode": index, "placed": episode["placed"], **judge.judge({"ceiling": image})}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = summarize(results)
    print(json.dumps(summary))
    (args.directory / "gemini_judgments.json").write_text(
        json.dumps({"summary": summary, "episodes": results}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
