"""The `case` kind handle: parsing records the `on`/`cases`/`else` field ranges and
survives YAML 1.1's `on:` -> True key coercion; build LOWERS the case to a strict `CaseNode`
plus its data + control edges (the case's lowering is its own `.build(ctx)`, no leaf Node).
"""

import textwrap

import pytest

from yian.compile.expand import clone_child
from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.case import CaseDescriptor
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument
from yian.compose.source.catalog import _scope_origin
from yian.diagnostics import OriginRef
from yian.diagnostics import SourceRange
from yian.nodes.case import CaseNode
from yian.nodes.contracts import Binding
from yian.run import run_flow


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

_CASE_CHILD_BODY = """\
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
    else: negative
  positive:
    kind: code
    code: 'return "positive"'
    output: str
  broken:
    kind: code
    code: 'return "broken"'
    output: str
  negative:
    kind: code
    code: 'return "negative"'
    output: str
output: ${positive.output | broken.output | negative.output}
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


def test_case_with_both_on_spellings_is_a_located_error():
    # Plain `on:` (YAML 1.1 True) and quoted `"on":` are the same field; keep neither silently.
    flow = (
        "id: c\nname: c\nnodes:\n  route:\n    kind: case\n    on: ${x.output}\n"
        '    "on": ${y.output}\n    cases:\n      - when: a\n        then: hot\n'
    )
    with pytest.raises(LoadError, match='key `on` and key `"on"` both mean') as exc:
        load_flow(flow)

    assert exc.value.diagnostic.frames[-1].range == SourceRange(6, 5, 6, 7)


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
    # YAML coordinates remain on the Compose-only build product.
    assert isinstance(result.node_range, SourceRange)


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

    descriptor = CaseDescriptor(
        id="gate",
        cases=[{"when": "true", "then": "positive"}],
        when_origins=(OriginRef("first"),),
    )
    descriptor.cases.append({"when": "false", "then": "negative"})
    with pytest.raises(ValueError, match="must align"):
        descriptor.build(BuildContext({}))


@pytest.mark.parametrize("malformed_cases", ["null", "7"])
def test_source_attachment_does_not_preempt_malformed_case_validation(
    malformed_cases,
):
    """The origin hook stays inert for malformed shapes until the normal phases."""
    text = f"""\
id: malformed-case
name: malformed-case
uses:
  child: absent
nodes:
  gate:
    kind: case
    cases: {malformed_cases}
output: ${{gate.output}}
    """

    def fail_resolver(_path, _version=None):
        raise RuntimeError("resolver reached")

    with pytest.raises(RuntimeError, match="resolver reached"):
        load_flow(text, child_resolver=fail_resolver)


def test_named_definition_case_failure_keeps_child_predicate_origin():
    """Named child assembly and CALL growth retain the exact failing arm."""
    text = (
        "id: named-case-origin\n"
        "name: named-case-origin\n"
        "defs:\n"
        "  child:\n" + textwrap.indent(_CASE_CHILD_BODY, "    ") + "nodes:\n"
        "  invoke:\n"
        "    kind: call\n"
        "    call: child\n"
        "output: ${invoke.output}\n"
    )
    document = SourceDocument.parse(text, label="named-case.yaml")
    expected = (
        document.root.child("defs")
        .child("child")
        .child("nodes")
        .child("gate")
        .child("cases")
        .child(1)
        .child("when")
    )
    flow = load_flow(text, source_label="named-case.yaml")
    case_node = flow.compiled.nodes["invoke"].child.nodes["gate"]

    result = run_flow(flow, {})

    assert result.failure is not None
    assert result.failure.trace[-1] == case_node.cases[1].origin
    assert flow.diagnose(result.failure).frames[-1].range == expected.range


def test_inline_definition_case_failure_keeps_child_predicate_origin():
    """Inline child assembly and CALL growth retain the exact failing arm."""
    text = (
        "id: inline-case-origin\n"
        "name: inline-case-origin\n"
        "nodes:\n"
        "  invoke:\n"
        "    kind: call\n"
        + textwrap.indent(_CASE_CHILD_BODY, "    ")
        + "output: ${invoke.output}\n"
    )
    document = SourceDocument.parse(text, label="inline-case.yaml")
    expected = (
        document.root.child("nodes")
        .child("invoke")
        .child("nodes")
        .child("gate")
        .child("cases")
        .child(1)
        .child("when")
    )
    flow = load_flow(text, source_label="inline-case.yaml")
    case_node = flow.compiled.nodes["invoke"].child.nodes["gate"]

    result = run_flow(flow, {})

    assert result.failure is not None
    assert result.failure.trace[-1] == case_node.cases[1].origin
    assert flow.diagnose(result.failure).frames[-1].range == expected.range


def test_external_alias_rebases_case_and_generated_binding_origins():
    """An external view scopes CASE carriers without mutating its canonical child."""
    child = load_flow(_SEARCHED_ORIGIN_FLOW, source_label="child-case.yaml")
    parent = """\
id: parent-case-origin
name: parent-case-origin
nodes:
  invoke:
    kind: call
    call: child_view
output: ${invoke.output}
uses:
  child_view: child
"""
    loaded = load_flow(
        parent,
        child_resolver=lambda _path, _version=None: child,
    )
    canonical = child.compiled.nodes["gate"]
    embedded = loaded.compiled.nodes["invoke"].child.nodes["gate"]

    assert tuple(case.origin for case in embedded.cases) == tuple(
        _scope_origin("child_view", case.origin) for case in canonical.cases
    )
    canonical_binding = child.compiled.wiring["gate"]["__r0"]
    embedded_binding = loaded.compiled.nodes["invoke"].child.wiring["gate"]["__r0"]
    assert embedded_binding.origin == _scope_origin(
        "child_view", canonical_binding.origin
    )
    assert loaded.origin_resolver is not None
    assert child.origin_resolver is not None
    assert tuple(
        loaded.origin_resolver.resolve(case.origin) for case in embedded.cases
    ) == tuple(child.origin_resolver.resolve(case.origin) for case in canonical.cases)


def test_generated_child_clone_preserves_case_predicate_origins():
    """CALL cloning keeps ordered CASE sites and generated binding provenance."""
    loaded = load_flow(_SEARCHED_ORIGIN_FLOW)
    original = loaded.compiled.nodes["gate"]

    cloned = clone_child(loaded.compiled, callsite="invoke", record={})
    generated = cloned.nodes["invoke/gate"]

    assert generated is not original
    assert tuple(case.origin for case in generated.cases) == tuple(
        case.origin for case in original.cases
    )
    assert cloned.wiring["invoke/gate"]["__r0"].origin == (
        loaded.compiled.wiring["gate"]["__r0"].origin
    )


def test_formatting_moves_case_ranges_without_changing_keys_or_fingerprint():
    """Semantic CASE identities survive comments while resolution follows the text."""
    moved = _SEARCHED_ORIGIN_FLOW.replace(
        "  gate:\n    kind: case\n",
        "  # Formatting-only comments move the CASE.\n  gate:\n    kind: case\n",
    ).replace(
        "    cases:\n",
        "    # The ordered predicates keep their semantic indices.\n    cases:\n",
    )
    baseline = load_flow(_SEARCHED_ORIGIN_FLOW)
    relocated = load_flow(moved)
    baseline_node = baseline.compiled.nodes["gate"]
    relocated_node = relocated.compiled.nodes["gate"]
    baseline_origins = tuple(case.origin for case in baseline_node.cases)
    relocated_origins = tuple(case.origin for case in relocated_node.cases)

    assert relocated_origins == baseline_origins
    assert flow_fingerprint(relocated.compiled) == flow_fingerprint(baseline.compiled)
    assert baseline.origin_resolver is not None
    assert relocated.origin_resolver is not None
    assert tuple(
        baseline.origin_resolver.resolve(origin).range for origin in baseline_origins
    ) != tuple(
        relocated.origin_resolver.resolve(origin).range for origin in relocated_origins
    )


def _parse_case(body):
    # `parse_nodes` runs the shared `output:` -> `outputs` key projection.
    text = "nodes:\n  g:\n" + textwrap.indent(textwrap.dedent(body), "    ")
    return parse_nodes(SourceDocument.parse(text).root.child("nodes"))["g"]


def test_case_parse_accepts_output_type():
    desc = _parse_case(
        """\
        kind: case
        output: str
        cases:
          - when: "${a.output}"
            then: b
        """
    )
    assert desc.outputs == "str"


def test_case_parse_accepts_output_mapping():
    desc = _parse_case(
        """\
        kind: case
        output:
          a: int
        cases:
          - when: "${a.output}"
            then: b
        """
    )
    assert desc.outputs == {"a": "int"}


def test_case_parse_keeps_then_styles_aligned_with_cases():
    desc = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output} == 1"
            then: reply
          - when: "${a.output} == 2"
            then: "reply"
          - when: "${a.output} == 3"
            then: 'r'
          - when: "${a.output} == 4"
            then: |
              some text
          - when: "${a.output} == 5"
            then: 3
          - when: "${a.output} == 6"
            then:
              a: 1
        """
    )
    assert desc.then_styles == ["plain", "double", "single", "literal", "plain", None]


def test_case_parse_keeps_else_style():
    plain = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output}"
            then: b
        else: reply
        """
    )
    double = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output}"
            then: b
        else: "approve"
        """
    )
    missing = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output}"
            then: b
        """
    )
    assert plain.else_style == "plain"
    assert double.else_style == "double"
    assert missing.else_style is None


def test_case_parse_tells_else_null_from_no_else():
    null_else = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output}"
            then: b
        else: null
        """
    )
    no_else = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output}"
            then: b
        """
    )
    named_else = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output}"
            then: b
        else: reply
        """
    )
    assert null_else.else_ is None and null_else.has_else is True
    assert no_else.has_else is False
    assert named_else.has_else is True


def test_case_descriptor_constructed_with_else_has_else():
    desc = CaseDescriptor(id="g", cases=[{"when": "a", "then": "b"}], else_="b")
    assert desc.has_else is True


def test_case_parse_keeps_then_null_key():
    desc = _parse_case(
        """\
        kind: case
        cases:
          - when: "${a.output}"
            then: null
        """
    )
    assert "then" in desc.cases[0]
    assert desc.cases[0]["then"] is None
    assert desc.then_styles == ["plain"]


def test_case_parse_on_form_keeps_output_and_styles():
    desc = _parse_case(
        """\
        kind: case
        on: ${a.output}
        output: str
        cases:
          - when: x
            then: "b"
        else: 'c'
        """
    )
    assert desc.on == "${a.output}"
    assert desc.outputs == "str"
    assert desc.then_styles == ["double"]
    assert desc.else_style == "single"
