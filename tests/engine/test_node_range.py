from pathlib import Path

import pytest

from yian.compile.manifest import flow_manifest
from yian.compose.loader import load_flow
from yian.nodes.start.node import StartNode
from yian.run import resume_flow, run_flow

_REMOVED_FIELDS = {
    "range",
    "field_ranges",
    "pre_assert_ranges",
    "post_assert_ranges",
}


def test_node_has_no_compose_range_metadata():
    n = StartNode("s")
    assert _REMOVED_FIELDS.isdisjoint(vars(n))


@pytest.mark.parametrize("field", sorted(_REMOVED_FIELDS))
def test_manifest_rejects_retired_node_range_fields(field: str):
    """A dynamic legacy field cannot silently disappear from program identity."""
    loaded = load_flow(_RANGE_FLOW)
    setattr(loaded.compiled.nodes["done"], field, object())

    with pytest.raises(ValueError, match=field):
        flow_manifest(loaded.compiled)


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


def test_render_only_node_metadata_does_not_ride_run_log(tmp_path: Path):
    """Render-only wiring labels must not ride the authoritative run log.

    Recovery rebuilds topology from the supplied authored flow, so synthetic wiring labels
    are neither execution identity nor durable state. The semantic node also remains free of
    Compose-owned range metadata before the run and after clean reconstruction.
    """
    path = tmp_path / "range-free.jsonl"
    loaded = load_flow(_RANGE_FLOW)
    node = loaded.compiled.nodes["done"]
    assert _REMOVED_FIELDS.isdisjoint(vars(node))
    node.wiring_labels = {"__r0": "cases[0].when"}  # a synth-param label — also render-only

    res = run_flow(
        loaded,
        {"settle_at": "2026-07-01"},
        log=path,
        tools={"ask_user": lambda **request: None},
    )
    assert res.status == "succeeded"
    assert res.output == "cancel"

    blob = path.read_text()
    # The invariant: no render-only Node metadata leaked into authoritative records.
    assert "wiring_labels" not in blob
    assert "cases[0].when" not in blob  # the label value must not ride the blob either

    # Terminal recovery is read-only and reconstructs the same result from clean topology.
    before = path.read_bytes()
    res2 = resume_flow(load_flow(_RANGE_FLOW), log=path)
    assert res2.status == "succeeded"
    assert res2.output == "cancel"
    assert res2.events == []
    assert path.read_bytes() == before
