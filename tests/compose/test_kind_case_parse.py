"""6.7 — the `case` kind handle: parse records the `on`/`cases`/`else` field ranges and
survives YAML 1.1's `on:` -> True key coercion; build raises (a case is desugared, never
built as a leaf node)."""

import pytest

from agent_composer.compose.errors import LoadError
from agent_composer.compose.kinds.case import CaseDescriptor, build, parse
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.spanloader import load_with_spans
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
    desc = parse("route", reader)

    # The bare `on:` key is coerced to `True` by YAML 1.1; parse maps it back to the
    # `on` field (a SpanLoader win — the range still resolves via the raw key text).
    assert desc == CaseDescriptor(
        id="route",
        on="${score.output}",
        cases=[{"when": "high", "then": "hot"}],
        else_="cold",
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges["on"], SourceRange)
    assert isinstance(desc.field_ranges["cases"], SourceRange)
    assert isinstance(desc.field_ranges["else"], SourceRange)


def test_case_parse_requires_cases():
    flow = "nodes:\n  route:\n    kind: case\n    on: ${x.output}\n"
    reader = _reader_for(flow, "route")
    with pytest.raises(LoadError, match="missing required field 'cases'"):
        parse("route", reader)


def test_case_build_raises_because_case_is_desugared():
    reader = _reader_for(_FLOW, "route")
    desc = parse("route", reader)
    with pytest.raises(LoadError, match="not a leaf node"):
        build(desc, registry={}, resolver=None)
