from agent_composer.nodes.start.node import StartNode


def test_node_range_defaults_none():
    n = StartNode("s")
    assert n.range is None and n.field_ranges == {}


# A flow that WAITs (pauses) so `run_flow` produces a checkpoint we can round-trip.
_WAIT_FLOW = """
id: f
name: f
input:
  settle_at: date
nodes:
  settle:
    kind: wait
    until: ${input.settle_at}
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
    from agent_composer.compose.run import run_flow, resume_flow
    from agent_composer.suspension.checkpoint import RunCheckpoint
    from agent_composer.suspension.commands import DeliverAnswerCommand
    from agent_composer.events import SourceRange

    loaded = load_flow(_WAIT_FLOW)
    # Stamp render-only source metadata on a live node BEFORE the run serializes.
    node = loaded.compiled.nodes["done"]
    node.range = SourceRange(4, 1, 6, 1)
    node.field_ranges = {"code": SourceRange(6, 5, 6, 40)}

    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert res.status == "paused"
    assert res.checkpoint is not None

    blob = res.checkpoint.dumps()
    # The invariant: no render-only Node metadata leaked into the serialized checkpoint.
    assert "field_ranges" not in blob
    assert "start_line" not in blob  # a serialized SourceRange would carry its fields
    assert "\"range\"" not in blob

    # And the restored run reconstructs + resumes to completion.
    ckpt = RunCheckpoint.loads(blob)
    release = DeliverAnswerCommand(node_id="settle", value=None)  # WAIT release = Output(None)
    res2 = resume_flow(loaded, checkpoint=ckpt, commands=[release])
    assert res2.status == "succeeded"
    assert res2.output == "cancel"  # confirm_action(rec={}) -> "cancel"

