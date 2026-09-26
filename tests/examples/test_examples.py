"""Every shipped example must load and compile (no live LLM).

These guard the `examples/` gallery against drift: a flow that no longer parses,
references an unknown node, or breaks a binding fails here. AGENT examples need a
provider to *run*, so we only assert they LOAD — the engine boundary is exercised
by the CODE-flow CLI tests and the engine suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_composer.compose.loader import load_flow

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples").glob("*.yaml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads_and_compiles(path: Path):
    loaded = load_flow(path.read_text(), search_paths=[path.parent])
    assert loaded.compiled is not None
    assert loaded.input is not None


def test_examples_present():
    names = {p.name for p in EXAMPLES}
    assert {"hello.yaml", "summarize.yaml", "classify.yaml"} <= names


def test_debate_agents_use_shared_plain_pipeline():
    path = next(path for path in EXAMPLES if path.name == "debate.yaml")
    loaded = load_flow(path.read_text(), search_paths=[path.parent])
    for node_id in ("frame", "for_case", "against_case", "verdict"):
        node = loaded.compiled.nodes[node_id]
        assert node.mode == "plain"
        assert node.tools == []
