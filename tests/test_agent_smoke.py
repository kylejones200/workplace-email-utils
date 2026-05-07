from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
from pathlib import Path

from agent_email_extract_non_nlp.agent import run

ROOT = Path(__file__).resolve().parents[1]


def test_example_request_smoke() -> None:
    request_payload = json.loads((ROOT / "examples" / "request.json").read_text(encoding="utf-8"))
    response = run(request_payload)
    assert response["status"] in {"success", "partial"}
    assert isinstance(response["data"], dict)


def test_response_matches_example_shape_not_exact_values() -> None:
    response_example = json.loads((ROOT / "examples" / "response.json").read_text(encoding="utf-8"))
    request_payload = json.loads((ROOT / "examples" / "request.json").read_text(encoding="utf-8"))
    response = run(request_payload)
    assert set(response.keys()) == set(response_example.keys())
    assert set(response["metadata"].keys()) == set(response_example["metadata"].keys())
