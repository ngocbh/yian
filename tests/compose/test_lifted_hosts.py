"""The host each desugar pass records on a node it lifts out of an authored node.

A lifted node's `host` is the node whose text it was written in. A later edge step gives the
lifted node its host's CASE routes, `depends_on`, and `runs_after`. A node that is itself a route
target (a branch `call(...)`), a value entry, or a node that comes from the flow
`output:`/`asserts:` has no host.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.desugar.adaptive_questions import desugar_adaptive_questions
from yian.compose.desugar.case_targets import desugar_case_call_targets
from yian.compose.desugar.inline_calls import _synth_id_gen, desugar_inline_calls
from yian.compose.desugar.value_entries import read_value_entry
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument
from yian.compose.source.catalog import _SourceCatalogBuilder


def _descriptors(nodes_yaml: str) -> dict:
    """Parse a flow whose `nodes:` section is ``nodes_yaml`` into descriptors."""
    text = "id: f\nname: f\nnodes:\n" + nodes_yaml
    return parse_nodes(SourceDocument.parse(text).root.child("nodes"))


def _synth(descriptors: dict) -> dict:
    """The lifted `__call_N` descriptors, keyed by id."""
    return {nid: d for nid, d in descriptors.items() if nid.startswith("__call_")}


def test_authored_descriptors_have_no_host():
    descriptors = _descriptors(
        "  a:\n    kind: code\n    code: m:f\n    output: str\n"
    )
    assert descriptors["a"].host is None


def test_host_is_not_an_authored_key():
    text = """
id: f
name: f
nodes:
  a:
    kind: code
    code: "return 1"
    host: b
    output: int
output: ${a.output}
"""
    with pytest.raises(LoadError, match="host"):
        load_flow(text)


def test_inline_calls_in_a_node_input_record_the_node_as_host():
    descriptors = _descriptors(
        "  publish:\n    kind: code\n    code: m:f\n"
        "    input:\n      draft: call(f, x=call(g, y=1))\n    output: str\n"
    )
    new, _outputs, _asserts = desugar_inline_calls(
        descriptors, None, next_id=_synth_id_gen()
    )
    synth = _synth(new)
    assert sorted(synth) == ["__call_0", "__call_1"]
    assert all(d.host == "publish" for d in synth.values())
    assert new["publish"].host is None


def test_inline_calls_in_map_over_and_loop_seed_record_their_node():
    descriptors = _descriptors(
        "  each:\n    kind: map\n    call: work\n    over: call(items)\n"
        "  turn:\n    kind: loop\n    call: body\n    times: 2\n"
        "    input:\n      s: call(seed)\n"
    )
    new, _outputs, _asserts = desugar_inline_calls(
        descriptors, None, next_id=_synth_id_gen()
    )
    hosts = {d.call: d.host for d in _synth(new).values()}
    assert hosts == {"items": "each", "seed": "turn"}


def test_inline_calls_in_flow_output_and_asserts_have_no_host():
    descriptors = _descriptors(
        "  a:\n    kind: code\n    code: m:f\n    output: int\n"
    )
    new, _outputs, _asserts = desugar_inline_calls(
        descriptors,
        {"r": "call(f)"},
        asserts_section=["call(g)"],
        next_id=_synth_id_gen(),
    )
    synth = _synth(new)
    assert sorted(d.call for d in synth.values()) == ["f", "g"]
    assert all(d.host is None for d in synth.values())


def test_nested_branch_call_records_the_outer_call_as_host():
    descriptors = _descriptors(
        "  gate:\n    kind: case\n    cases:\n"
        "      - when: ${input.go}\n"
        '        then: call(outer, x=call(inner, y="hi"))\n'
        "    else: call(take)\n"
    )
    new = desugar_case_call_targets(descriptors, _synth_id_gen())
    by_callee = {d.call: d for d in _synth(new).values()}
    outer = by_callee["outer"]
    assert by_callee["inner"].host == outer.id
    assert outer.host is None
    assert by_callee["take"].host is None
    assert new["gate"].cases[0]["then"] == outer.id


def test_adaptive_composer_records_the_gate_as_host():
    descriptors = _descriptors(
        "  ask:\n    kind: human_input\n"
        "    adaptive_questions:\n      prompt: Ask about ${topic}\n"
        "    input:\n      topic: ${input.topic}\n"
    )
    new = desugar_adaptive_questions(descriptors)
    assert new["ask__compose"].host == "ask"
    assert new["ask"].host is None


def test_value_entry_has_no_host():
    nodes = SourceDocument.parse(
        "id: f\nname: f\nnodes:\n  verdict: !str approve\n"
    ).root.child("nodes")
    builder = _SourceCatalogBuilder()

    def read_value(node_id, cursor):
        return read_value_entry(
            node_id,
            cursor,
            registry={},
            owner=("f",),
            catalog_builder=builder,
            diagnostics=LoadDiagnostics.from_cursor(cursor),
        )

    descriptors = parse_nodes(nodes, read_value=read_value)
    assert descriptors["verdict"].host is None
