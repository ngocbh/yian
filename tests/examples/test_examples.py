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
# TS-only examples: chat.yaml renders its history with the `prompt:` list style (#1002).
TS_ONLY = {"chat.yaml"}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads_and_compiles(path: Path):
    if path.name in TS_ONLY:
        pytest.skip("TS-only example (#1002)")
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
