"""Every shipped chat flow loads (tools must be registered first)."""
from pathlib import Path

import pytest

from yian.cli.chat import tools  # noqa: F401 - registers the flow-op tools
from yian.compose.loader import load_flow


@pytest.mark.parametrize(
    "path",
    [
        "src/yian/cli/chat/chat.yaml",
        "tests/cli/chat_example.yaml",
    ],
)
def test_chat_flow_loads(path):
    p = Path(path)
    loaded = load_flow(p.read_text())
    assert loaded.name
    assert loaded.compiled.nodes["chat"].max_iters is None
