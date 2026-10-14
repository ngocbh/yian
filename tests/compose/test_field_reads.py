"""Load-time checks for reading a field a value's known type can't have."""

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.run import run_flow


def _flow(step_output: str, rest: str, *, preamble: str = "") -> str:
    """A flow with one CODE node `step` typed `step_output`, plus `rest` (YAML tail).

    `preamble` holds top-level blocks placed before `input:`, such as `typedefs:` or `defs:`.
    """
    return (
        "id: p\nname: p\n"
        + preamble
        + "input:\n  s: str\n"
        "nodes:\n"
        "  step:\n"
        "    kind: code\n"
        "    input:\n"
        "      x: ${input.s}\n"
        f"    output: {step_output}\n"
        "    code: return None\n"
        + rest
    )


def _load_error(text: str) -> str:
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    return str(exc.value)


def test_optional_record_field_reads_null_when_the_whole_value_is_null():
    text = _flow(
        "Optional[Rec]",
        "output:\n  a: ${step.output.a}\n  note: 'a=${step.output.a}'\n",
        preamble="typedefs:\n  Rec: {a: str}\n",
    )
    result = run_flow(load_flow(text), {"s": "x"})
    assert result.status == "succeeded"
    assert result.output == {"a": None, "note": "a="}


def test_case_when_on_a_missing_optional_record_input_is_falsy():
    """A `when:` reading a field of an omitted `Optional[Rec]` input takes the else branch."""
    text = (
        "id: p\nname: p\n"
        "typedefs:\n  Rec: {a: str}\n"
        "input:\n  s: str\n  maybe: Optional[Rec]\n"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        "    cases:\n"
        "      - when: ${input.maybe.a == 'x'}\n"
        "        then: hit\n"
        "    else: miss\n"
        "  hit:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: str\n"
        "    code: return 'hit'\n"
        "  miss:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: str\n"
        "    code: return 'miss'\n"
        "output: ${hit.output | miss.output}\n"
    )
    flow = load_flow(text)
    assert run_flow(flow, {"s": "x"}).output == "miss"
    assert run_flow(flow, {"s": "x", "maybe": {"a": "x"}}).output == "hit"


@pytest.mark.parametrize(
    "step_output, label",
    [
        ("str", "str"),
        ("int", "int"),
        ("float", "float"),
        ("bool", "bool"),
        ("date", "date"),
        ("datetime", "datetime"),
        ("int[0, 10]", "int[0,10]"),
        ("Literal[a, b]", 'Literal["a","b"]'),
        ("list[str]", "list[str]"),
        ("Optional[str]", "Optional[str]"),
    ],
)
def test_field_on_a_fieldless_output_in_a_flow_output_is_a_load_error(step_output, label):
    msg = _load_error(_flow(step_output, "output:\n  o: ${step.output.order}\n"))
    assert (
        f"flow output from: reference ${{step.output.order}} reads field 'order' "
        f"on type {label}, which has no fields"
    ) in msg


def test_field_on_str_in_a_node_binding_is_a_load_error():
    rest = (
        "  sink:\n"
        "    kind: code\n"
        "    input:\n"
        "      v: ${step.output.order}\n"
        "    output: str\n"
        "    code: return 'x'\n"
        "output: ${sink.output}\n"
    )
    msg = _load_error(_flow("str", rest))
    assert "node 'sink' input 'v' from: reference ${step.output.order} reads field 'order'" in msg


_SINK = (
    "  a:\n"
    "    kind: code\n"
    "    input:\n"
    "      v: ${input.s}\n"
    "    output: str\n"
    "    code: return 'x'\n"
)


def test_field_on_str_in_a_case_when_is_a_load_error():
    rest = _SINK + (
        "  route:\n"
        "    kind: case\n"
        "    cases:\n"
        "      - when: ${step.output.order == 'x'}\n"
        "        then: a\n"
        "    else: a\n"
        "output: ${a.output}\n"
    )
    msg = _load_error(_flow("str", rest))
    assert (
        "node 'route' cases[0].when from: reference ${step.output.order} reads field 'order' "
        "on type str"
    ) in msg


def test_field_on_str_in_a_case_on_is_a_load_error():
    rest = _SINK + (
        "  route:\n"
        "    kind: case\n"
        "    on: ${step.output.kind}\n"
        "    cases:\n"
        "      - when: x\n"
        "        then: a\n"
        "    else: a\n"
        "output: ${a.output}\n"
    )
    msg = _load_error(_flow("str", rest))
    assert (
        "node 'route' on: from: reference ${step.output.kind} reads field 'kind' on type str"
    ) in msg


@pytest.mark.parametrize(
    "binding",
    [
        "'note: ${step.output.order}'",          # embedded in a template
        "${step.output.order | input.s}",         # coalesce
        "${input.s :- step.output.order}",       # default RHS
    ],
)
def test_field_on_str_in_any_reference_form_is_a_load_error(binding):
    msg = _load_error(_flow("str", f"output:\n  o: {binding}\n"))
    assert "reference ${step.output.order} reads field 'order' on type str" in msg


def test_nested_field_on_a_record_field_of_type_str_is_a_load_error():
    msg = _load_error(_flow("{a: str}", "output:\n  o: ${step.output.a.b}\n"))
    assert "reference ${step.output.a.b} reads field 'b' on type str, which has no fields" in msg


def test_undeclared_record_field_keeps_its_message():
    msg = _load_error(_flow("{a: str}", "output:\n  o: ${step.output.b}\n"))
    assert "reference ${step.output.b} reads unknown field 'b' on record" in msg


_EACH_DEFS = (
    "defs:\n"
    "  each:\n"
    "    input:\n      v: str\n"
    "    nodes:\n"
    "      inner:\n"
    "        kind: code\n"
    "        input:\n          v: ${input.v}\n"
    "        output: str\n"
    "        code: return 'x'\n"
    "    output: ${inner.output}\n"
)


def test_field_on_str_in_map_over_is_a_load_error():
    """MAP `over:` is a node binding too."""
    text = _flow(
        "str",
        "  m:\n"
        "    kind: map\n"
        "    call: each\n"
        "    over: ${step.output.items}\n"
        "    input:\n      v: ${item}\n"
        "output: ${m.output}\n",
        preamble=_EACH_DEFS,
    )
    msg = _load_error(text)
    assert (
        "node 'm' map over: reference ${step.output.items} reads field 'items' on type str"
    ) in msg


def test_field_on_str_in_wait_until_is_a_load_error():
    """WAIT `until:` is a node binding too."""
    text = _flow(
        "str",
        "  settle:\n"
        "    kind: wait\n"
        "    until: ${step.output.at}\n"
        "output: ${step.output}\n",
    )
    msg = _load_error(text)
    assert (
        "node 'settle' wait until: reference ${step.output.at} reads field 'at' on type str"
    ) in msg


_BARE_CHILD = (
    "id: p\nname: p\n"
    "input:\n  s: str\n  xs: list[str]\n"
    "defs:\n"
    "  child:\n"
    "    input:\n      s: str\n"
    "    nodes:\n"
    "      step:\n"
    "        kind: code\n"
    "        input:\n          v: ${input.s}\n"
    "        output: str\n"
    "        code: return 'x'\n"
    "    output: ${step.output}\n"
)


def test_field_on_a_bare_call_output_is_a_load_error():
    """The issue's reproduction, with a bare child output (not one-field unwrapping)."""
    text = _BARE_CHILD + (
        "nodes:\n"
        "  c:\n"
        "    kind: call\n"
        "    call: child\n"
        "    input:\n      s: ${input.s}\n"
        "output: ${c.output.order}\n"
    )
    assert (
        "flow output from: reference ${c.output.order} reads field 'order' on type str, "
        "which has no fields"
    ) in _load_error(text)


def test_field_on_a_map_output_is_a_load_error():
    text = _BARE_CHILD + (
        "nodes:\n"
        "  m:\n"
        "    kind: map\n"
        "    call: child\n"
        "    over: ${input.xs}\n"
        "    input:\n      s: ${item}\n"
        "output: ${m.output.first}\n"
    )
    assert "reference ${m.output.first} reads field 'first' on type list[str]" in _load_error(text)


def test_field_on_str_in_a_flow_assert_is_a_load_error():
    text = _flow("str", "asserts:\n  - ${step.output.order == 'x'}\noutput: ${step.output}\n")
    assert (
        "assert \"${step.output.order == 'x'}\": reference ${step.output.order} reads field "
        "'order' on type str"
    ) in _load_error(text)


# A child whose output is a partial record: `a` is typed, `b` comes from an untyped CODE node.
_PARTIAL_CHILD = (
    "id: p\nname: p\n"
    "input:\n  s: str\n"
    "defs:\n"
    "  child:\n"
    "    input:\n      s: str\n"
    "    nodes:\n"
    "      typed:\n"
    "        kind: code\n"
    "        input:\n          v: ${input.s}\n"
    "        output: str\n"
    "        code: return 'x'\n"
    "      untyped:\n"
    "        kind: code\n"
    "        input:\n          v: ${input.s}\n"
    "        code: \"return {'k': 1}\"\n"
    "    output:\n      a: ${typed.output}\n      b: ${untyped.output}\n"
    "nodes:\n"
    "  c:\n"
    "    kind: call\n"
    "    call: child\n"
    "    input:\n      s: ${input.s}\n"
)


def test_partial_record_output_checks_names_and_typed_fields():
    assert "reads unknown field 'zzz' on record" in _load_error(
        _PARTIAL_CHILD + "output: ${c.output.zzz}\n"
    )
    assert "reference ${c.output.a.x} reads field 'x' on type str" in _load_error(
        _PARTIAL_CHILD + "output: ${c.output.a.x}\n"
    )
    load_flow(_PARTIAL_CHILD + "output: ${c.output.b.anything}\n")  # opaque field


def test_node_assert_on_a_partial_record_output_checks_typed_fields():
    text = _PARTIAL_CHILD.replace(
        "    input:\n      s: ${input.s}\n",
        "    input:\n      s: ${input.s}\n    asserts:\n      - output.a.x == 1\n",
    ) + "output: ${c.output.a}\n"
    assert "node 'c' assert: reference ${output.a.x} reads field 'x' on type str" in _load_error(
        text
    )


def test_node_assert_on_an_exact_record_output_checks_typed_fields():
    text = _flow("{a: str}", "output: ${step.output.a}\n").replace(
        "    code: return None\n",
        "    code: return None\n    asserts:\n      - output.a.b == 1\n",
    )
    assert (
        "node 'step' assert: reference ${output.a.b} reads field 'b' on type str, "
        "which has no fields"
    ) in _load_error(text)


@pytest.mark.parametrize(
    "step_output, ref, preamble",
    [
        ("object", "${step.output.anything.deeper}", ""),          # opaque
        ("Optional[Rec]", "${step.output.a}", "typedefs:\n  Rec: {a: str}\n"),
        ("{a: str, b: object}", "${step.output.b.anything}", ""),  # object-typed field
        ("list[str]", "${step.output}", ""),                     # whole value
    ],
)
def test_reads_the_type_allows_still_load(step_output, ref, preamble):
    load_flow(_flow(step_output, f"output:\n  o: {ref}\n", preamble=preamble))


def test_untyped_output_reads_still_load():
    text = (
        "id: p\nname: p\ninput:\n  s: str\n"
        "nodes:\n"
        "  step:\n"
        "    kind: code\n"
        "    input:\n      x: ${input.s}\n"
        "    code: \"return {'a': 1}\"\n"
        "output:\n  o: ${step.output.anything.deeper}\n"
    )
    load_flow(text)


_INPUTS = (
    "input:\n  s: str\n  r:\n    a: str\n  tags: list[str]\n  o: object\n  d: file\n"
)


def _input_flow(ref: str) -> str:
    return (
        "id: p\nname: p\n" + _INPUTS + "nodes:\n"
        "  step:\n"
        "    kind: code\n"
        "    input:\n"
        f"      v: {ref}\n"
        "    output: str\n"
        "    code: return 'x'\n"
        "output: ${step.output}\n"
    )


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("${input.s.first}", "reads field 'first' on type str, which has no fields"),
        ("${input.tags.first}", "reads field 'first' on type list[str], which has no fields"),
        ("${input.r.b}", "reads unknown field 'b' on record"),
        ("${input.r.a.b}", "reads field 'b' on type str, which has no fields"),
    ],
)
def test_flow_input_field_reads_are_checked(ref, expected):
    msg = _load_error(_input_flow(ref))
    assert f"node 'step' input 'v' from: reference {ref} {expected}" in msg


@pytest.mark.parametrize(
    "ref", ["${input.r.a}", "${input.o.any.depth}", "${input.s}", "${input.d.uri}"]
)
def test_valid_flow_input_reads_still_load(ref):
    load_flow(_input_flow(ref))


def test_defs_body_input_field_reads_are_checked():
    """A named `defs:` body's own typed inputs are checked."""
    text = (
        "id: p\nname: p\ninput:\n  s: str\n"
        "defs:\n"
        "  child:\n"
        "    input:\n      s: str\n"
        "    nodes:\n"
        "      step:\n"
        "        kind: code\n"
        "        input:\n          v: ${input.s.first}\n"
        "        output: str\n"
        "        code: return 'x'\n"
        "    output: ${step.output}\n"
        "nodes:\n"
        "  c:\n"
        "    kind: call\n"
        "    call: child\n"
        "    input:\n      s: ${input.s}\n"
        "output: ${c.output}\n"
    )
    assert "reference ${input.s.first} reads field 'first' on type str" in _load_error(text)


def test_inline_body_input_field_reads_are_checked():
    """An inline body's inputs, inferred from the call site, are checked."""
    text = (
        "id: p\nname: p\ninput:\n  s: str\n"
        "nodes:\n"
        "  c:\n"
        "    kind: call\n"
        "    input:\n      v: ${input.s}\n"
        "    nodes:\n"
        "      inner:\n"
        "        kind: code\n"
        "        input:\n          w: ${input.v.z}\n"
        "        output: str\n"
        "        code: return 'x'\n"
        "    output: ${inner.output}\n"
        "output: ${c.output}\n"
    )
    assert "reference ${input.v.z} reads field 'z' on type str" in _load_error(text)


def _map_flow(over_input: str, item_ref: str) -> str:
    return (
        "id: p\nname: p\n"
        "typedefs:\n  Rec: {a: str}\n"
        f"input:\n  xs: {over_input}\n"
        + _EACH_DEFS
        + "nodes:\n"
        "  m:\n"
        "    kind: map\n"
        "    call: each\n"
        "    over: ${input.xs}\n"
        f"    input:\n      v: {item_ref}\n"
        "output: ${m.output}\n"
    )


@pytest.mark.parametrize(
    "over_input, item_ref, expected",
    [
        ("list[str]", "${item.a}", "reads field 'a' on type str, which has no fields"),
        ("list[Rec]", "${item.b}", "reads unknown field 'b' on record"),
        ("Optional[list[Rec]]", "${item.b}", "reads unknown field 'b' on record"),
    ],
)
def test_map_item_field_reads_are_checked(over_input, item_ref, expected):
    msg = _load_error(_map_flow(over_input, item_ref))
    assert f"node 'm' map input 'v' from: reference {item_ref} {expected}" in msg


@pytest.mark.parametrize(
    "over_input, item_ref",
    [
        ("list[Rec]", "${item.a}"),
        ("list[str]", "${item}"),
        ("list[object]", "${item.anything}"),
        ("list[Any]", "${item.anything}"),
        ("object", "${item.anything}"),  # over: not a proven list
    ],
)
def test_map_item_reads_the_type_allows_still_load(over_input, item_ref):
    load_flow(_map_flow(over_input, item_ref))
