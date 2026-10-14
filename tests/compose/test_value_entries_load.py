"""Load checks for value entries in `nodes:` (`total: !int ${a.output + b.output}`).

A tagged entry lowers at load to a CODE node with the entry's id and one typed input
named after it, so the tag is checked like a typed binding's: a span-free literal is
checked while the entry is read, a span's inferred type must not be disjoint from the
tag, and a nested tag checks its own value. A tag never fills in a missing field. These
tests cover the located errors (a tag without a value, a mistyped literal, a spaced or
unknown tag, a nested mismatch, a number that is not finite, a disjoint span or flow
call, a span reading the entry itself, a flow call inside `${...}`) and the flow
fingerprint.

A bare `!object` over a written-out mapping keeps the record the mapping spells: reading
a field it lacks is a load error, a `${...}` field has its producer's type, a nested
tagged `!object` stays opaque, and an empty key in the record is a load error, since no
record field can be named "". `!object ${x.output}` stays `object`.
"""

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compose import LoadError, load_flow
from yian.nodes.base import NodeKind
from yian.run import run_flow

_BUY = "typedefs:\n  Buy:\n    action: Literal(buy)\n    conviction: float\n"

_F_DEF = (
    "defs:\n"
    "  f:\n"
    "    input:\n"
    "      x: !str\n"
    "    nodes:\n"
    "      up:\n"
    "        kind: code\n"
    "        input:\n"
    "          x: ${input.x}\n"
    "        output: str\n"
    "        code: |\n"
    "          return inputs['x'].upper()\n"
    "    output: ${up.output}\n"
)


def _flow(entries: str, *, output: str = "${x.output}", head: str = "") -> str:
    """Return a flow with a `t: str` input, a float producer `s`, and `entries`."""
    return (
        "id: f\n"
        "name: f\n"
        f"{head}"
        "input:\n"
        "  t: !str\n"
        "nodes:\n"
        "  s:\n"
        "    kind: code\n"
        "    output: float\n"
        "    code: |\n"
        "      return 0.5\n"
        f"{entries}"
        f"output: {output}\n"
    )


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _load_error(text: str) -> LoadError:
    """Load `text`, expecting a load error, and return it."""
    with pytest.raises(LoadError) as info:
        load_flow(text)
    return info.value


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def test_a_value_entry_builds_a_code_node_with_the_entry_id():
    flow = load_flow(_flow("  x: !int 3\n"))

    node = flow.compiled.nodes["x"]
    assert node.kind is NodeKind.CODE
    assert node.output_contract.type.kind.value == "int"


def test_named_type_entry_reads_a_float_producer():
    text = _flow("  x: !Buy\n    action: buy\n    conviction: ${s.output}\n", head=_BUY)

    load_flow(text)


def test_named_type_entry_rejects_a_disjoint_producer_at_the_entry():
    text = _flow(
        "  x: !Buy\n    action: buy\n    conviction: ${s.output}\n", head=_BUY
    ).replace("output: float", "output: str")

    error = _load_error(text)

    assert str(error) == (
        "value entry 'x': declared Buy, but the value is "
        '{"action":"buy","conviction":str}'
    )
    assert _error_line(error) == _line_of(text, "  x: !Buy")


def test_a_tag_does_not_fill_in_a_missing_field():
    text = _flow("  x: !Buy {conviction: 1}\n", head=_BUY)

    error = _load_error(text)

    assert str(error) == (
        "'x': the value is not a valid Buy: record missing required fields: "
        "['action'] (at `x.action`)"
    )
    assert _error_line(error) == _line_of(text, "  x: !Buy")


def test_a_named_field_read_into_a_disjoint_binding_is_rejected():
    text = _flow(
        "  x: !Buy {action: buy, conviction: 1}\n"
        "  use:\n"
        "    kind: code\n"
        "    input:\n"
        "      c: !str ${x.output.conviction}\n"
        "    code: |\n"
        "      return inputs['c']\n",
        head=_BUY,
    )

    error = _load_error(text)

    assert str(error) == "node 'use' input 'c': declared str, but the value is float"
    assert _error_line(error) == _line_of(text, "c: !str")


def test_a_nested_tag_is_checked_at_its_own_line():
    good = _flow("  x: !object\n    d: !Buy {action: buy, conviction: 1}\n", head=_BUY)
    load_flow(good)

    bad = good.replace("conviction: 1}", "conviction: x}")
    error = _load_error(bad)

    assert str(error) == (
        "'x.d': the value is not a valid Buy: 'x' does not match declared type float "
        "(at `x.d.conviction`)"
    )
    assert _error_line(error) == _line_of(bad, "    d: !Buy")


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("  x: !int\n", "value entry 'x' needs a value after its tag, e.g. `x: !int ...`"),
        (
            "  x: !str 30\n",
            "'x': `30` is not a valid str; quote it to keep it as text: `x: !str \"30\"`",
        ),
        (
            "  x: !int (1..3) 2\n",
            "'x': a YAML tag ends at the first space; write `!int(1..3)` without spaces",
        ),
        (
            "  x: !Strr hi\n",
            "'x': `!Strr` is not a type (unknown type 'Strr'). Did you mean `str`?",
        ),
        (
            "  x: !object [1]\n",
            "'x': the value is not a valid object: [1] does not match declared type object",
        ),
    ],
)
def test_entry_errors_are_located_at_the_entry(entry, message):
    text = _flow(entry)

    error = _load_error(text)

    assert str(error) == message
    assert _error_line(error) == _line_of(text, "  x: !")


def test_a_disjoint_span_names_the_value_entry():
    text = _flow("  x: !str ${s.output}\n")

    error = _load_error(text)

    assert str(error) == "value entry 'x': declared str, but the value is float"
    assert _error_line(error) == _line_of(text, "  x: !str")


def test_a_flow_call_of_a_disjoint_type_is_rejected_at_the_entry():
    text = _flow("  x: !int call(f, x=${input.t})\n", head=_F_DEF)

    error = _load_error(text)

    assert str(error) == "value entry 'x': declared int, but the value is str"
    assert _error_line(error) == _line_of(text, "  x: !int call")


def test_a_flow_call_inside_a_span_keeps_its_error():
    text = _flow("  x: !str ${ f(input.t) }\n", head=_F_DEF)

    error = _load_error(text)

    assert str(error) == (
        "x: a flow call 'f' inside ${...} is not supported — write it as the whole "
        "value with the call(...) directive: call(f, ...)"
    )
    assert _error_line(error) == _line_of(text, "  x: !str ${ f(")


def test_entries_reading_each_other_are_rejected():
    text = _flow(
        "  x: !object\n    a: ${y.output}\n  y: !object\n    b: ${x.output}\n"
    )

    error = _load_error(text)

    assert str(error) == "node build dependency cycle: x -> y -> x"
    assert _error_line(error) == _line_of(text, "  y: !object")


def test_an_entry_reading_itself_is_rejected():
    text = _flow("  x: !object\n    a: ${x.output}\n")

    error = _load_error(text)

    assert str(error) == "node build dependency cycle: x -> x"
    assert _error_line(error) == _line_of(text, "  x: !object")


def test_an_entry_under_a_credential_name_loads():
    load_flow(_flow("  api_key: !str sk-live-123\n", output="${api_key.output}"))
    load_flow(
        _flow(
            "  cfg: !object\n    access_token: abc\n    x: ${s.output}\n",
            output="${cfg.output}",
        )
    )


def test_a_mistyped_literal_under_a_credential_name_gets_the_ordinary_error():
    plain = _load_error(_flow("  x: !int sk-live-123\n"))
    text = _flow("  api_key: !int sk-live-123\n", output="${api_key.output}")

    error = _load_error(text)

    assert str(error) == str(plain).replace("'x'", "'api_key'").replace(
        "`x:", "`api_key:"
    )
    assert _error_line(error) == _line_of(text, "api_key:")


def test_a_nested_binding_loads():
    load_flow(
        _flow("  cfg: !object\n    access_token: ${input.t}\n", output="${cfg.output}")
    )


def test_fingerprint_is_stable_and_tracks_the_literal():
    text = _flow("  x: !object {exited: true, reply: ''}\n")

    first = flow_fingerprint(load_flow(text).compiled)
    second = flow_fingerprint(load_flow(text).compiled)
    changed = flow_fingerprint(load_flow(text.replace("true", "false")).compiled)

    assert first == second
    assert changed != first


def _record_flow(entries: str, reader: str) -> str:
    """Return a flow with `entries` and a CODE node `use` whose input `c` is `reader`."""
    return (
        "id: f\n"
        "name: f\n"
        "input:\n"
        "  n: !int\n"
        "nodes:\n"
        f"{entries}"
        "  use:\n"
        "    kind: code\n"
        "    input:\n"
        f"      c: {reader}\n"
        "    code: |\n"
        "      return inputs['c']\n"
        "output: ${use.output}\n"
    )


_EXIT_RESULT = '  exit_result: !object {exited: true, reply: ""}\n'


def test_a_written_out_object_entry_keeps_its_fields():
    load_flow(_record_flow(_EXIT_RESULT, "${exit_result.output.reply}"))


def test_a_written_out_object_entry_rejects_an_unknown_field_at_the_reader():
    text = _record_flow(_EXIT_RESULT, "${exit_result.output.zz}")

    error = _load_error(text)

    assert str(error) == (
        "flow has unresolved references:\n"
        "  node 'use' input 'c' from: reference ${exit_result.output.zz} reads unknown "
        "field 'zz' on record"
    )
    assert _error_line(error) == _line_of(text, "c: ${exit_result")


def _producer(output: str) -> str:
    """Return a CODE node `p` returning text, declaring `output` unless it is empty."""
    declared = f"    output: {output}\n" if output else ""
    return f"  p:\n    kind: code\n{declared}    code: |\n      return 'hi'\n"


_SPAN_RECORD = "  r: !object\n    a: ${p.output}\n    b: 1\n"


def test_a_span_field_takes_its_producer_type():
    text = _record_flow(_producer("str") + _SPAN_RECORD, "!int ${r.output.a}")

    error = _load_error(text)

    assert str(error) == "node 'use' input 'c': declared int, but the value is str"
    assert _error_line(error) == _line_of(text, "c: !int")


@pytest.mark.parametrize("output", ["str", ""])
def test_a_span_record_rejects_an_unknown_field(output):
    text = _record_flow(_producer(output) + _SPAN_RECORD, "${r.output.c}")

    error = _load_error(text)

    assert str(error) == (
        "flow has unresolved references:\n"
        "  node 'use' input 'c' from: reference ${r.output.c} reads unknown field 'c' "
        "on record"
    )
    assert _error_line(error) == _line_of(text, "c: ${r.output.c}")


def test_a_span_from_an_untyped_producer_is_opaque():
    load_flow(_record_flow(_producer("") + _SPAN_RECORD, "!int ${r.output.a}"))


@pytest.mark.parametrize(
    "entry",
    [
        "  r: !object\n    a: ${p.output}\n    p: {q: 1}\n",
        "  r: !object\n    p: {q: 1}\n",
    ],
)
def test_a_nested_untagged_mapping_keeps_its_fields(entry):
    text = _record_flow(_producer("str") + entry, "${r.output.p.zz}")

    error = _load_error(text)

    assert str(error) == (
        "flow has unresolved references:\n"
        "  node 'use' input 'c' from: reference ${r.output.p.zz} reads unknown field "
        "'zz' on record"
    )
    assert _error_line(error) == _line_of(text, "c: ${r.output.p.zz}")


@pytest.mark.parametrize(
    "entry",
    [
        "  r: !object\n    a: ${p.output}\n    p: !object {q: 1}\n",
        "  r: !object\n    p: !object {q: 1}\n",
    ],
)
def test_a_nested_tagged_object_is_opaque(entry):
    load_flow(_record_flow(_producer("str") + entry, "${r.output.p.zz}"))


def test_an_object_entry_over_an_outside_value_stays_object():
    entries = (
        "  x:\n"
        "    kind: code\n"
        "    output: {q: int}\n"
        "    code: |\n"
        "      return {'q': 1}\n"
        "  o: !object ${x.output}\n"
    )

    load_flow(_record_flow(entries, "${o.output.zz}"))


def test_an_escaped_span_in_an_object_entry_is_text():
    text = _record_flow('  r: !object {a: "$${input.n}"}\n', "!int ${r.output.a}")

    error = _load_error(text)

    assert str(error) == "node 'use' input 'c': declared int, but the value is str"
    assert _error_line(error) == _line_of(text, "c: !int")


@pytest.mark.parametrize(("field", "typ"), [("[1, 2]", "list(int)"), ("true", "bool")])
def test_a_literal_field_takes_its_literal_type(field, typ):
    text = _record_flow(f"  r: !object\n    b: {field}\n", "!str ${r.output.b}")

    error = _load_error(text)

    assert str(error) == f"node 'use' input 'c': declared str, but the value is {typ}"
    assert _error_line(error) == _line_of(text, "c: !str")


_G_DEF = (
    "defs:\n"
    "  g:\n"
    "    input:\n"
    "      n: !int\n"
    "    nodes:\n"
    "      v:\n"
    "        kind: code\n"
    "        output: {q: int}\n"
    "        code: |\n"
    "          return {'q': 1}\n"
    "    output: ${v.output}\n"
)


@pytest.mark.parametrize(
    "entry",
    [
        "  o: !object call(g, n=${input.n})\n",
        "  o: !Optional(object) {a: 1}\n",
        "  o: !Optional(object)\n    a: ${input.n}\n",
    ],
)
def test_an_object_entry_not_over_a_bare_object_mapping_keeps_its_tag(entry):
    text = _record_flow(entry, "${o.output.zz}").replace("input:\n", _G_DEF + "input:\n", 1)

    load_flow(text)


_EMPTY_KEY = (
    "a record key can't be empty; a written-out `!object` keeps its keys as record "
    "fields"
)


@pytest.mark.parametrize(
    ("entry", "label", "needle"),
    [
        ('  r: !object {"": 1, a: 2}\n', "r", "  r: !object"),
        ('  r: !object {"": "${input.n}"}\n', "r", "  r: !object"),
        ('  r: !object {a: {"": 1}}\n', "r.a", "  r: !object"),
        ('  r: !object {a: [1, {"": 1}]}\n', "r.a[1]", "  r: !object"),
        ('  r: !object\n    a:\n      "": 1\n', "r.a", '      "": 1'),
    ],
)
def test_an_object_entry_with_an_empty_key_is_rejected(entry, label, needle):
    # No record field can be named "", and the read must not quietly go unchecked.
    text = _record_flow(entry, "${r.output.b}")

    error = _load_error(text)

    assert str(error) == f"{label!r}: {_EMPTY_KEY}"
    assert _error_line(error) == _line_of(text, needle)


@pytest.mark.parametrize(
    ("entry", "value"),
    [
        ('  r: !Optional(object) {"": 1}\n', {"": 1}),
        ('  r: !list(object) [{"": 1}]\n', [{"": 1}]),
        ('  r: !object {a: !object {"": 1}}\n', {"a": {"": 1}}),
    ],
)
def test_an_empty_key_outside_a_written_out_record_loads(entry, value):
    flow = load_flow(_record_flow(entry, "${r.output}"))

    assert run_flow(flow, {"n": 3}).output == value


@pytest.mark.parametrize(
    ("entry", "label", "shown", "needle"),
    [
        ("  x: !float .inf\n", "x", ".inf", "  x: !float"),
        ("  x: !float -.inf\n", "x", "-.inf", "  x: !float"),
        ("  x: !object {a: .nan}\n", "x.a", ".nan", "  x: !object"),
        ("  x: !object\n    a: .inf\n", "x.a", ".inf", "    a: .inf"),
        ("  x: !list(float) [1.0, .nan]\n", "x[1]", ".nan", "  x: !list"),
        ("  x: !object {a: !float .inf}\n", "x.a", ".inf", "  x: !object"),
        ('  x: !object {a: "${input.t}", b: .inf}\n', "x.b", ".inf", "  x: !object"),
    ],
)
def test_a_number_that_is_not_finite_is_rejected_at_the_entry(
    entry, label, shown, needle
):
    text = _flow(entry)

    error = _load_error(text)

    assert str(error) == f"{label!r}: numbers must be finite (got {shown})"
    assert _error_line(error) == _line_of(text, needle)
