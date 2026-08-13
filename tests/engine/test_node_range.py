from pathlib import Path

import pytest

from agent_composer.compose.loader import load_flow
from agent_composer.events import SourceRange
from agent_composer.nodes.start.node import StartNode
from tests.engine._fakes import replace_human_inputs_with_artificial_pauses

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


# An artificial Pause flow so ``run_flow`` produces a compatibility checkpoint.
_PAUSE_FLOW = """
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


def test_node_range_does_not_ride_checkpoint():
    """A stamped `.range`/`.field_ranges` must NOT ride the serialized checkpoint.

    The checkpoint stores no topology (it is rebuilt from the flow spec on restore), so
    render-only Node metadata must never appear in the blob. This stamps a range on a live
    node, round-trips the checkpoint through `dumps()`/`loads()`, and asserts (a) the blob
    carries no range/field_ranges keys and (b) the restored run resumes successfully.
    """
    from agent_composer.compose.loader import load_flow
    from agent_composer.run import run_flow, resume_flow
    from agent_composer.suspension.checkpoint import RunCheckpoint
    from agent_composer.suspension.commands import DeliverAnswerCommand
    from agent_composer.events import SourceRange

    loaded = load_flow(_PAUSE_FLOW)
    replace_human_inputs_with_artificial_pauses(loaded.compiled)
    # Stamp render-only source metadata on a live node BEFORE the run serializes.
    node = loaded.compiled.nodes["done"]
    node.range = SourceRange(4, 1, 6, 1)
    node.field_ranges = {("field", "code"): SourceRange(6, 5, 6, 40)}
    node.wiring_labels = {"__r0": "cases[0].when"}  # a synth-param label — also render-only

    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert res.status == "paused"
    assert res.checkpoint is not None

    blob = res.checkpoint.dumps()
    # The invariant: no render-only Node metadata leaked into the serialized checkpoint.
    assert "field_ranges" not in blob
    assert "start_line" not in blob  # a serialized SourceRange would carry its fields
    assert "\"range\"" not in blob
    assert "wiring_labels" not in blob
    assert "cases[0].when" not in blob  # the label value must not ride the blob either

    # And the restored run reconstructs + resumes to completion.
    ckpt = RunCheckpoint.loads(blob)
    release = DeliverAnswerCommand(node_id="settle", value=None)
    res2 = resume_flow(loaded, checkpoint=ckpt, commands=[release])
    assert res2.status == "succeeded"
    assert res2.output == "cancel"  # confirm_action(rec={}) -> "cancel"
