from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_email_extract_non_nlp.agent import run

ALLOWED_STATUS = {"success", "partial", "error"}


def _assert_contract_shape(response: dict) -> None:
    assert isinstance(response, dict)
    for key in ("status", "message", "data", "metadata"):
        assert key in response
    assert response["status"] in ALLOWED_STATUS
    metadata = response["metadata"]
    for key in ("agent_slug", "agent_version", "run_id"):
        assert key in metadata


def test_entrypoint_callable_contract() -> None:
    response = run({"input": {"hello": "world"}})
    _assert_contract_shape(response)


def test_empty_request_returns_structured_error() -> None:
    response = run({})
    _assert_contract_shape(response)
    assert response["status"] == "error"


def test_malformed_request_returns_structured_error() -> None:
    response = run("bad")  # type: ignore[arg-type]
    _assert_contract_shape(response)
    assert response["status"] == "error"


def test_unknown_fields_return_structured_error() -> None:
    response = run({"input": {}, "unexpected": True})
    _assert_contract_shape(response)
    assert response["status"] == "error"
