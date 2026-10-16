"""No JSON file under tests/llm (cassettes, goldens) holds anything shaped like a secret."""

from pathlib import Path

import pytest

from yian.llm.testing import find_secrets

ROOT = Path(__file__).parent
JSON_FILES = sorted(ROOT.rglob("*.json"))


def test_there_are_json_files_to_check():
    assert JSON_FILES, "no JSON files under tests/llm, so the secret check would pass vacuously"


@pytest.mark.parametrize("path", JSON_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_json_file_has_no_secrets(path):
    assert find_secrets(path.read_text(encoding="utf-8")) == []
