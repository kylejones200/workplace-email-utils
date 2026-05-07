from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
from pathlib import Path

import jsonschema

from agent_email_extract_non_nlp.agent import run

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_request_example_validates_against_schema() -> None:
    jsonschema.validate(instance=_read_json(ROOT / "examples" / "request.json"), schema=_read_json(ROOT / "schemas" / "request.schema.json"))


def test_response_example_validates_against_schema() -> None:
    jsonschema.validate(instance=_read_json(ROOT / "examples" / "response.json"), schema=_read_json(ROOT / "schemas" / "response.schema.json"))


def test_live_response_validates_against_response_schema() -> None:
    jsonschema.validate(instance=run(_read_json(ROOT / "examples" / "request.json")), schema=_read_json(ROOT / "schemas" / "response.schema.json"))
