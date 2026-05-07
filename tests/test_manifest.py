from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_exists_and_has_required_fields() -> None:
    required = {"name", "slug", "version", "description", "entrypoint", "capabilities", "pages", "navigation", "schemas", "examples"}
    manifest = _load_manifest()
    assert required.issubset(manifest.keys())


def test_slug_and_version_format() -> None:
    manifest = _load_manifest()
    assert re.fullmatch(r"[a-z0-9_]+", manifest["slug"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"])


def test_entrypoint_is_importable_callable() -> None:
    module_name, symbol_name = _load_manifest()["entrypoint"].split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    assert callable(getattr(module, symbol_name))


def test_pages_and_navigation_are_consistent() -> None:
    manifest = _load_manifest()
    page_ids = set()
    for page in manifest["pages"]:
        for field in ("id", "title", "route", "component", "description"):
            assert field in page
        page_ids.add(page["id"])
    for nav in manifest["navigation"]:
        assert nav["page_id"] in page_ids


def test_schema_and_example_paths_exist() -> None:
    manifest = _load_manifest()
    for rel in manifest["schemas"].values():
        assert (ROOT / rel).exists()
    for rel in manifest["examples"].values():
        assert (ROOT / rel).exists()
