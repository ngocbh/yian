"""6.7 — the `case` kind handle: parse records the `on`/`cases`/`else` field ranges and
survives YAML 1.1's `on:` -> True key coercion; build LOWERS the case to a strict `CaseNode`
plus its data + control edges (the case's lowering is its own `.build(ctx)`, no leaf Node).
"""

import pytest

from agent_composer.compose import load_flow
from agent_composer.nodes.case import CaseNode
from agent_composer.nodes.contracts import Binding
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.nodes.case import CaseDescriptor
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import OriginRef
from agent_composer.events import SourceRange
from agent_composer.run import run_flow


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

_SEARCHED_ORIGIN_FLOW = """\
id: searched-case-origins
name: searched-case-origins
nodes:
  zero:
    kind: code
    code: "return 0"
    output: int
  gate:
    kind: case
    cases:
      - when: "${zero.output} > 0"
        then: positive
      - when: "1 / ${zero.output} > 0"
        then: broken
      - when: "${zero.output} > 0"
        then: duplicate
    else: negative
  positive:
    kind: code
    code: 'return "positive"'
    output: str
  broken:
    kind: code
    code: 'return "broken"'
    output: str
  duplicate:
    kind: code
    code: 'return "duplicate"'
    output: str
  negative:
    kind: code
    code: 'return "negative"'
    output: str
output: ${positive.output | broken.output | duplicate.output | negative.output}
"""

_ON_ORIGIN_FLOW = """\
id: on-case-origins
name: on-case-origins
nodes:
  label:
    kind: code
    code: 'return "high"'
    output: str
  route:
    kind: case
    on: ${label.output}
    cases:
      - when: high
        then: hot
      - when: low
        then: cold
    else: cold
  hot:
    kind: code
    code: 'return "hot"'
    output: str
  cold:
    kind: code
    code: 'return "cold"'
    output: str
output: ${hot.output | cold.output}
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_case_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "route")
    desc = CaseDescriptor.parse("route", reader)

    # The bare `on:` key is coerced to `True` by YAML 1.1; parse maps it back to the
    # `on` field (the source cursor still resolves the range via the raw key text).
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


def test_searched_case_origins_are_positional_and_select_the_failing_entry():
    """Duplicate text stays distinct while runtime selects the active ordered arm."""
    label = "searched-case.yaml"
    document = SourceDocument.parse(_SEARCHED_ORIGIN_FLOW, label=label)
    cases_cursor = document.root.child("nodes").child("gate").child("cases")
    expected_ranges = tuple(
        cases_cursor.child(index).child("when").range for index in range(3)
    )
    flow = load_flow(_SEARCHED_ORIGIN_FLOW, source_label=label)
    node = flow.compiled.nodes["gate"]

    origins = tuple(case.origin for case in node.cases)
    assert all(isinstance(origin, OriginRef) for origin in origins)
    assert len({origin.key for origin in origins if origin is not None}) == 3
    assert origins[0] != origins[2]
    assert flow.origin_resolver is not None
    assert (
        tuple(flow.origin_resolver.resolve(origin).range for origin in origins)
        == expected_ranges
    )

    binding = flow.compiled.wiring["gate"]["__r0"]
    assert isinstance(binding, Binding)
    assert binding.origin == origins[0]

    result = run_flow(flow, {})

    assert result.failure is not None
    assert result.failure.trace[-1] == origins[1]
    assert flow.diagnose(result.failure).frames[-1].range == expected_ranges[1]


def test_on_case_preserves_predicate_and_generated_binding_origins():
    """The synthetic `__on` binding belongs to `on:`, not to a case predicate."""
    label = "on-case.yaml"
    document = SourceDocument.parse(_ON_ORIGIN_FLOW, label=label)
    route_cursor = CaseDescriptor._normalize_cursor(
        document.root.child("nodes").child("route")
    )
    flow = load_flow(_ON_ORIGIN_FLOW, source_label=label)
    node = flow.compiled.nodes["route"]
    resolver = flow.origin_resolver
    assert resolver is not None

    predicate_origins = tuple(case.origin for case in node.cases)
    assert all(isinstance(origin, OriginRef) for origin in predicate_origins)
    assert tuple(
        resolver.resolve(origin).range for origin in predicate_origins
    ) == tuple(
        route_cursor.child("cases").child(index).child("when").range
        for index in range(2)
    )

    on_binding = flow.compiled.wiring["route"]["__on"]
    assert isinstance(on_binding, Binding)
    assert on_binding.origin not in predicate_origins
    assert resolver.resolve(on_binding.origin).range == route_cursor.child("on").range


def test_source_free_case_build_keeps_predicates_and_wiring_origin_free():
    """Direct descriptor builds retain their historical raw wiring shape."""
    descriptor = CaseDescriptor(
        id="gate",
        cases=[{"when": "${score.output} > 0", "then": "positive"}],
    )

    result = descriptor.build(BuildContext({}))

    assert result.node.cases[0].origin is None
    assert result.wiring == {"__r0": "${score.output}"}


def test_case_descriptor_rejects_misaligned_or_nonopaque_origins():
    """Programmatic descriptors cannot detach provenance from ordered cases."""
    with pytest.raises(ValueError, match="must align"):
        CaseDescriptor(
            id="gate",
            cases=[{"when": "true", "then": "positive"}],
            when_origins=(OriginRef("first"), OriginRef("second")),
        )
    with pytest.raises(TypeError, match="predicate origin"):
        CaseDescriptor(
            id="gate",
            cases=[{"when": "true", "then": "positive"}],
            when_origins=("flow.yaml:12",),  # type: ignore[arg-type]
        )
