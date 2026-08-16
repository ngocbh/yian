from pathlib import Path

import pytest

from agent_composer.compose.loader import load_flow
from agent_composer.events import SourceRange
from agent_composer.nodes.start.node import StartNode
from agent_composer.run import resume_flow, run_flow
from agent_composer.tools import install_ask_user

_SEEDS = Path(__file__).resolve().parents[1] / "seeds"


def test_node_range_defaults_none():
    n = StartNode("s")
    assert n.range is None and n.field_ranges == {}


@pytest.mark.parametrize(
    "seed, node_id, want_field",
    [
        ("25-nested-suspension.yaml", "gate", "call"),  # call
        ("05-call-map.yaml", "research_each", "over"),  # map
        ("28-refine-loop.yaml", "polish", "call"),  # loop
    ],
)
def test_call_map_loop_nodes_carry_range(seed, node_id, want_field):
    """A loader-built call/map/loop node must be stamped with `.range`/`.field_ranges`.

    Regression guard: `_assemble` once called `build_call_node`/`build_loop_node` directly,
    bypassing the `kinds/*.py::build` wrappers that stamp source metadata — so the funnel's
    concrete-range resolution silently fell back to the node header for a binding failure on a
    map's `over:` or a call-arg. Routing through `KINDS[kind].build` restores parity with leaves.
    """
    text = (_SEEDS / seed).read_text()
    loaded = load_flow(text, search_paths=[_SEEDS])
    node = loaded.compiled.nodes[node_id]
    assert isinstance(node.range, SourceRange)
    assert node.field_ranges  # non-empty
    assert isinstance(node.field_ranges[("field", want_field)], SourceRange)


# A blocking flow whose successful records must exclude compatibility node metadata.
_RANGE_FLOW = """
id: f
name: f
input:
  settle_at: date
nodes:
  settle:
    kind: human_input
    prompt: "continue?"
    output: Optional[str]
  done:
    kind: code
    depends_on: [settle]
    output: str
    code: tests.seeds.fns:confirm_action
output: ${done.output}
"""


def test_node_range_does_not_ride_run_log(tmp_path: Path):
    """Stamped compatibility metadata must not ride the authoritative run log.

    Recovery rebuilds topology from the supplied authored flow, so node ranges, field maps,
    and synthetic wiring labels are neither execution identity nor durable state. This stamps
    all three on the reusable source graph, runs to a durable terminal, verifies the JSONL,
    then silently reconstructs the same result from a clean flow.
    """
    path = tmp_path / "range-free.jsonl"
    loaded = load_flow(_RANGE_FLOW)
    # Stamp compatibility-only source metadata before RunStarted fingerprints and serializes.
    node = loaded.compiled.nodes["done"]
    node.range = SourceRange(4, 1, 6, 1)
    node.field_ranges[("field", "code")] = SourceRange(6, 5, 6, 40)
    node.wiring_labels = {"__r0": "cases[0].when"}  # a synth-param label — also render-only

    with install_ask_user(lambda **request: None):
        res = run_flow(
            loaded,
            {"settle_at": "2026-07-01"},
            log=path,
        )
    assert res.status == "succeeded"
    assert res.output == "cancel"

    blob = path.read_text()
    # The invariant: no compatibility-only Node metadata leaked into authoritative records.
    assert "field_ranges" not in blob
    assert "start_line" not in blob  # a serialized SourceRange would carry its fields
    assert "\"range\"" not in blob
    assert "wiring_labels" not in blob
    assert "cases[0].when" not in blob  # the label value must not ride the blob either

    # Terminal recovery is read-only and reconstructs the same result from clean topology.
    before = path.read_bytes()
    res2 = resume_flow(load_flow(_RANGE_FLOW), log=path)
    assert res2.status == "succeeded"
    assert res2.output == "cancel"
    assert res2.events == []
    assert path.read_bytes() == before
