"""6.7 — the `case` kind handle: parse records the `on`/`cases`/`else` field ranges and
survives YAML 1.1's `on:` -> True key coercion; build LOWERS the case to a strict `CaseNode`
plus its data + control edges (the case's lowering is its own `.build(ctx)`, no leaf Node)."""

import pytest

from agent_composer.nodes.case import CaseNode
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.nodes.case import CaseDescriptor
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.events import SourceRange


_FLOW = """\
nodes:
  route:
    kind: case
    on: ${score.output}
    cases:
      - when: high
        then: hot
    else: cold
"""


def _reader_for(text, node_id):
    value, spans, root = load_with_spans(text, return_root=True)
    root_reader = NodeReader(value, spans, root)
    return root_reader.child("nodes").child(node_id)


def test_case_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "route")
    desc = CaseDescriptor.parse("route", reader)

    # The bare `on:` key is coerced to `True` by YAML 1.1; parse maps it back to the
    # `on` field (a SpanLoader win — the range still resolves via the raw key text).
    assert desc == CaseDescriptor(
        id="route",
        on="${score.output}",
        cases=[{"when": "high", "then": "hot"}],
        else_="cold",
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "on")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "cases")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "else")], SourceRange)


def test_case_parse_requires_cases():
    flow = "nodes:\n  route:\n    kind: case\n    on: ${x.output}\n"
    reader = _reader_for(flow, "route")
    with pytest.raises(LoadError, match="missing required field 'cases'"):
        CaseDescriptor.parse("route", reader)


def test_case_build_lowers_to_case_node_with_edges():
    reader = _reader_for(_FLOW, "route")
    desc = CaseDescriptor.parse("route", reader)
    result = desc.build(BuildContext({}))
    # the case's lowering IS its build: a strict CaseNode + a flow-owned {__on -> source} wiring.
    assert isinstance(result.node, CaseNode)
    assert result.node.id == "route"
    assert result.wiring == {"__on": "${score.output}"}
    # the flat edges tuple is DATA (score->route on __on) then CONTROL (route->hot / route->cold).
    data = [e for e in result.edges if e.source_handle is None]
    control = [e for e in result.edges if e.source_handle is not None]
    assert ("score", "route", "__on") in {(e.from_, e.to, e.input_group) for e in data}
    ctl = {(e.from_, e.to, e.source_handle) for e in control}
    assert ("route", "hot", "hot") in ctl
    assert ("route", "cold", "default") in ctl
    # `.range`/`.field_ranges` are stamped onto the built node via `_finish`.
    assert isinstance(result.node.range, SourceRange)
