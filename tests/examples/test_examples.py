"""Every shipped example must load and compile (no live LLM).

These guard the `examples/` gallery against drift: a flow that no longer parses,
references an unknown node, or breaks a binding fails here. AGENT examples need a
provider to *run*, so we only assert they LOAD — the engine boundary is exercised
by the CODE-flow CLI tests and the engine suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yian.compose.loader import load_flow

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples").glob("*.yaml"))
# TS-only examples: they use TS-only built-ins and syntax (chat.yaml also renders its history
# with the `prompt:` list style; chat.yaml and compact_history.yaml use list comprehensions,
# len() and slices). tests/cli/chat_example.yaml is Python yian's chat example.
TS_ONLY = {"approve-plan.yaml", "chat.yaml", "compact_history.yaml", "human-approval.yaml"}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads_and_compiles(path: Path):
    if path.name in TS_ONLY:
        pytest.skip("uses TS-only built-ins (#1025)")
    loaded = load_flow(path.read_text(), search_paths=[path.parent])
    assert loaded.compiled is not None
    assert loaded.input is not None


def test_examples_present():
    names = {p.name for p in EXAMPLES}
    assert {"hello.yaml", "summarize.yaml", "classify.yaml"} <= names


def test_debate_agents_are_tool_free():
    path = next(path for path in EXAMPLES if path.name == "debate.yaml")
    loaded = load_flow(path.read_text(), search_paths=[path.parent])
    for node_id in ("frame", "for_case", "against_case", "verdict"):
        node = loaded.compiled.nodes[node_id]
        assert node.tools == []
