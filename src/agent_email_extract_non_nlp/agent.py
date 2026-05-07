"""Shared shell contract entrypoint for agent_email_extract_non-nlp."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4

AGENT_SLUG = "agent_email_extract_non_nlp"

try:
    AGENT_VERSION = version("agent_email_extract_non_nlp")
except PackageNotFoundError:  # pragma: no cover
    AGENT_VERSION = "0.1.0"


def _response(status: str, message: str, *, data: dict | None = None) -> dict:
    return {
        "status": status,
        "message": message,
        "data": data or {},
        "metadata": {
            "agent_slug": AGENT_SLUG,
            "agent_version": AGENT_VERSION,
            "run_id": str(uuid4()),
        },
    }


def run(request: dict) -> dict:
    if not isinstance(request, dict):
        return _response("error", "Request must be a dictionary.")
    if "input" not in request:
        return _response("error", "Field `input` is required.")
    unknown = set(request) - {"input", "strict"}
    if unknown:
        return _response("error", f"Unknown request fields: {sorted(unknown)}")
    return _response("success", "Agent executed.", data={"echo": request["input"]})
