"""Every shipped chat flow loads (tools must be registered first)."""
from pathlib import Path

import pytest

from agent_composer.cli.chat import tools  # noqa: F401 - registers the flow-op tools
from agent_composer.compose.loader import load_flow


@pytest.mark.parametrize(
    "path",
    [
        "src/agent_composer/cli/chat/chat.yaml",
        "examples/chat.yaml",
        ".claude/skills/composing-agents/templates/chat.yaml",
    ],
)
def test_chat_flow_loads(path):
    p = Path(path)
    loaded = load_flow(p.read_text())
    assert loaded.name
