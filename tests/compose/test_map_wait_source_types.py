"""A MAP `over:` or WAIT `until:` of the wrong type is a load error at that value.

Literals are checked where the node is parsed; references and expressions by their
inferred type once the graph is built.
"""

from __future__ import annotations

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.compose.nodes.base import value_type_error

OVER_FIX = "write a list, or a reference such as `over: ${input.items}`"
UNTIL_FIX = (
    "write one such as `until: 2026-05-21`, or a reference such as `until: ${input.as_of}`"
)


def _over_message(read_as: str, node: str = "m") -> str:
    return f"node {node!r} (kind=map): `over:` must be a list, but {read_as}; {OVER_FIX}"


def _until_message(read_as: str, node: str = "w") -> str:
    return (
        f"node {node!r} (kind=wait): `until:` must be a date or datetime, but {read_as}; "
        f"{UNTIL_FIX}"
    )


def _error(text: str) -> LoadError:
    """Load `text`, expecting a `LoadError` (not a raw `TypeError`/`KeyError`)."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line(text: str, marker: str) -> int:
    """The 1-based line of `text` holding `marker` (the first match)."""
    for number, line in enumerate(text.splitlines(), start=1):
        if marker in line:
            return number
    raise AssertionError(f"{marker!r} not in text")


def _start_line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


_ITEMS_DEF = """\
  items:
    nodes:
      c:
        kind: code
        output: list(str)
        code: |
          return ["a", "b"]
    output: ${c.output}
"""


def _map(value: str) -> str:
    """A flow whose MAP node `m` has `over: <value>` and calls `child`."""
    return (
        """\
id: f
name: f
defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: ${input.x}
        code: |
          return inputs["x"]
    output: ${echo.output}
"""
        + _ITEMS_DEF
        + """\
nodes:
  m:
    kind: map
    over: @V
    call: child
    input:
      x: 1
output: ${m.output}
"""
    ).replace("@V", value)


def _wait(value: str) -> str:
    """A flow whose WAIT node `w` has `until: <value>`, followed by a code node."""
    return (
        """\
id: f
name: f
input:
  d: !date
defs:
"""
        + _ITEMS_DEF
        + """\
nodes:
  w:
    kind: wait
    until: @V
  after:
    kind: code
    depends_on: [w]
    output: int
    code: |
      return 1
output: ${after.output}
"""
    ).replace("@V", value)


def test_value_type_error_wording():
    assert value_type_error("over", "a list", 5, fix="F") == (
        "`over:` must be a list, but YAML read it as an integer; F"
    )
    assert value_type_error("over", "a list", None, fix="F") == (
        "`over:` must be a list, but it is empty; F"
    )


@pytest.mark.parametrize(
    ("value", "phrase"),
    [
        ("5", "an integer"),
        ("0", "an integer"),
        ("true", "a boolean"),
        ("false", "a boolean"),
        ("1.5", "a float"),
        ("{x: 1}", "a mapping"),
        ("{}", "a mapping"),
        ("2026-01-01", "a date"),
        ("2026-01-01T10:00:00", "a date and time"),
    ],
)
def test_non_list_over_literal_is_a_load_error(value, phrase):
    text = _map(value)
    error = _error(text)
    assert error.diagnostic.message == _over_message(f"YAML read it as {phrase}")
    assert _start_line(error) == _line(text, "over: ")


def test_null_over_is_empty():
    text = _map("null")
    error = _error(text)
    assert error.diagnostic.message == _over_message("it is empty")
    assert _start_line(error) == _line(text, "over: ")


@pytest.mark.parametrize(
    "value",
    [
        "ok",
        "'5'",
        "'2026-01-01'",
        "'$${x}'",
        "'call(items)'",
        "' call(items)'",
        "|\n      call(items)",
    ],
    ids=["word", "int", "date", "escape", "quoted-call", "spaced-call", "block-call"],
)
def test_text_over_is_a_load_error(value):
    text = _map(value)
    error = _error(text)
    assert error.diagnostic.message == _over_message("YAML read it as text")
    assert _start_line(error) == _line(text, "over: ")


@pytest.mark.parametrize("value", ["[1, 2]", "[]"])
def test_list_over_literals_load(value):
    load_flow(_map(value))


def test_call_directive_over_is_left_to_the_graph_stage():
    load_flow(_map("call(items)"))


def test_call_text_under_until_is_not_a_directive():
    text = _wait("call(items)")
    error = _error(text)
    assert error.diagnostic.message == _until_message(
        "YAML read it as text that is not a date or datetime"
    )
    assert _start_line(error) == _line(text, "until: ")


@pytest.mark.parametrize(
    ("value", "phrase"),
    [
        ("5", "an integer"),
        ("0", "an integer"),
        ("true", "a boolean"),
        ("false", "a boolean"),
        ("1.5", "a float"),
        ("[a]", "a list"),
        ("[]", "a list"),
        ("{x: 1}", "a mapping"),
        ("{}", "a mapping"),
    ],
)
def test_non_date_until_literal_is_a_load_error(value, phrase):
    text = _wait(value)
    error = _error(text)
    assert error.diagnostic.message == _until_message(f"YAML read it as {phrase}")
    assert _start_line(error) == _line(text, "until: ")


def test_null_until_is_empty():
    text = _wait("null")
    error = _error(text)
    assert error.diagnostic.message == _until_message("it is empty")
    assert _start_line(error) == _line(text, "until: ")


@pytest.mark.parametrize("value", ["ok", "'2026-13-01'", "'$${x}'"])
def test_text_until_must_parse_as_a_date(value):
    text = _wait(value)
    error = _error(text)
    assert error.diagnostic.message == _until_message(
        "YAML read it as text that is not a date or datetime"
    )
    assert _start_line(error) == _line(text, "until: ")


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01",
        "2026-01-01T10:00:00",
        "'2026-01-01'",
        "'2026-01-01T10:00'",
        "'2026-01-01T10:00:00+07:00'",
    ],
)
def test_date_until_literals_load(value):
    load_flow(_wait(value))


_OVER_IN_DEF = """\
id: f
name: f
defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: ${input.x}
        code: |
          return inputs["x"]
    output: ${echo.output}
  outer:
    nodes:
      inner:
        kind: map
        over: 5
        call: child
        input:
          x: 1
    output: ${inner.output}
nodes:
  a:
    kind: call
    call: outer
output: ${a.output}
"""

_UNTIL_IN_INLINE_BODY = """\
id: f
name: f
nodes:
  a:
    kind: call
    nodes:
      pause:
        kind: wait
        until: 5
      worker:
        kind: code
        depends_on: [pause]
        output: int
        code: |
          return 1
    output: ${worker.output}
output: ${a.output}
"""


@pytest.mark.parametrize(
    ("text", "message", "marker"),
    [
        (_OVER_IN_DEF, _over_message("YAML read it as an integer", "inner"), "over: 5"),
        (
            _UNTIL_IN_INLINE_BODY,
            _until_message("YAML read it as an integer", "pause"),
            "until: 5",
        ),
    ],
    ids=["defs", "inline"],
)
def test_literal_check_reaches_defs_and_inline_bodies(text, message, marker):
    error = _error(text)
    assert message in str(error)
    assert _start_line(error) == _line(text, marker)


def test_tagged_wrong_literal_reports_the_type_first():
    error = _error(_map("!int 5"))
    assert error.diagnostic.message == _over_message("YAML read it as an integer")

    error = _error(_wait("!int 5"))
    assert error.diagnostic.message == _until_message("YAML read it as an integer")

    text = _wait("!date ${input.d}")
    error = _error(text)
    assert error.diagnostic.message == (
        "a tag types a value — a flow input, a binding, a flow output, an agent "
        "variable, an `llm_config` value, a value entry in `nodes:`, or a value inside "
        "one; `until:` is not one"
    )
    assert _start_line(error) == _line(text, "until: ")


OVER_REF_FIX = "write a reference to a list"
_MODE = "typedefs:\n  Mode: Literal(ok, no)\n"
UNTIL_REF_FIX = "write a reference to a date, datetime, or text"


def _over_ref_message(label: str, node: str = "m") -> str:
    return (
        f"node {node!r} (kind=map): `over:` must be a list, but the value is {label}; "
        f"{OVER_REF_FIX}"
    )


def _until_ref_message(label: str, node: str = "w") -> str:
    return (
        f"node {node!r} (kind=wait): `until:` must be a date or datetime, but the value is "
        f"{label}; {UNTIL_REF_FIX}"
    )


def _map_ref(over: str, inputs: str = "  s: !str\n", typedefs: str = "") -> str:
    """A flow with `inputs` whose MAP node `m` has `over: <over>` and calls `child`.

    A code node `c` with `output: int` and one `u` with no declared output are
    available as producers.
    """
    return (
        "id: f\nname: f\n"
        + typedefs
        + "input:\n"
        + inputs
        + """\
defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: ${input.x}
        code: |
          return inputs["x"]
    output: ${echo.output}
  words:
    nodes:
      c:
        kind: code
        output: str
        code: |
          return "a"
    output: ${c.output}
nodes:
  c:
    kind: code
    output: int
    code: |
      return 1
  u:
    kind: code
    code: |
      return [1]
  m:
    kind: map
    over: @V
    call: child
    input:
      x: 1
output: ${m.output}
"""
    ).replace("@V", over)


def _wait_ref(until: str, inputs: str = "  s: !str\n", typedefs: str = "") -> str:
    """A flow with `inputs` whose WAIT node `w` has `until: <until>`, then a code node.

    A code node `c` with `output: int` and one `u` with no declared output are
    available as producers.
    """
    return (
        "id: f\nname: f\n"
        + typedefs
        + "input:\n"
        + inputs
        + """\
nodes:
  c:
    kind: code
    output: int
    code: |
      return 1
  u:
    kind: code
    code: |
      return "2026-01-01"
  w:
    kind: wait
    until: @V
  after:
    kind: code
    depends_on: [w]
    output: int
    code: |
      return 1
output: ${after.output}
"""
    ).replace("@V", until)


@pytest.mark.parametrize(
    ("over", "inputs", "label"),
    [
        ("${input.a}", "  a: !str\n", "str"),
        ("${input.a}", "  a: !int\n", "int"),
        ("${input.a}", "  a: !bool\n", "bool"),
        ("${input.a}", "  a: !object\n", "object"),
        ("${input.a}", "  a: !date\n", "date"),
        ("${input.a}", "  a: !Optional(int)\n", "Optional(int)"),
        ("${input.a}", "  a: !Optional(str)\n", "Optional(str)"),
        ("${input.a}", "  a: !Mode\n", "Literal(no, ok)"),
        ("'a ${input.s}'", "  s: !str\n", "str"),
        ("${c.output}", "  s: !str\n", "int"),
    ],
    ids=[
        "str",
        "int",
        "bool",
        "object",
        "date",
        "optional-int",
        "optional-str",
        "literal",
        "template",
        "typed-producer",
    ],
)
def test_disjoint_over_reference_is_a_load_error(over, inputs, label):
    text = _map_ref(over, inputs, typedefs=_MODE)
    error = _error(text)
    assert error.diagnostic.message == _over_ref_message(label)
    assert _start_line(error) == _line(text, "over: ")


_ISSUE_FLOW = """\
id: f
name: f
input:
  n: !int
nodes:
  w:
    kind: wait
    until: ${input.n}
  b:
    kind: code
    depends_on: [w]
    output: str
    code: return 'x'
output: ${b.output}
"""


@pytest.mark.parametrize(
    ("until", "inputs", "label"),
    [
        ("${input.a}", "  a: !int\n", "int"),
        ("${input.a}", "  a: !bool\n", "bool"),
        ("${input.a}", "  a: !float\n", "float"),
        ("${input.a}", "  a: !object\n", "object"),
        ("${input.a}", "  a: !list(str)\n", "list(str)"),
        ("${input.a}", "  a: !Optional(int)\n", "Optional(int)"),
        ("${input.a}", "  a: !Mode\n", "Literal(no, ok)"),
        ("${c.output}", "  s: !str\n", "int"),
    ],
    ids=[
        "int",
        "bool",
        "float",
        "object",
        "list",
        "optional-int",
        "literal",
        "typed-producer",
    ],
)
def test_disjoint_until_reference_is_a_load_error(until, inputs, label):
    text = _wait_ref(until, inputs, typedefs=_MODE)
    error = _error(text)
    assert error.diagnostic.message == _until_ref_message(label)
    assert _start_line(error) == _line(text, "until: ")


def test_the_issue_flow_is_a_load_error_at_until():
    error = _error(_ISSUE_FLOW)
    assert error.diagnostic.message == _until_ref_message("int")
    assert _start_line(error) == 8


@pytest.mark.parametrize(
    "text",
    [
        _map_ref("${input.a}", "  a: !list(str)\n"),
        _map_ref("${input.a}", "  a: !Optional(list(str))\n"),
        _map_ref("${input.a}", "  a: !Any\n"),
        _map_ref("${[input.s]}"),
        _map_ref("${input.n + 1}", "  n: !int\n"),
        _map_ref("${u.output}"),
        _wait_ref("${input.a}", "  a: !str\n"),
        _wait_ref("${input.a}", "  a: !date\n"),
        _wait_ref("${input.a}", "  a: !datetime\n"),
        _wait_ref("${input.a}", "  a: !Optional(date)\n"),
        _wait_ref("${input.a}", "  a: !Any\n"),
        _wait_ref("${u.output}"),
        _wait_ref(
            "${input.a}",
            "  a: !Day\n",
            typedefs="typedefs:\n  Day: Literal(ok, '2026-01-01')\n",
        ),
    ],
    ids=[
        "over-list",
        "over-optional-list",
        "over-any",
        "over-list-expression",
        "over-arithmetic",
        "over-untyped-producer",
        "until-str",
        "until-date",
        "until-datetime",
        "until-optional-date",
        "until-any",
        "until-untyped-producer",
        "until-overlapping-literal",
    ],
)
def test_fitting_or_unproven_sources_load(text):
    load_flow(text)


def test_call_directive_over_with_a_str_output_is_a_load_error():
    text = _map_ref("call(words)")
    error = _error(text)
    assert error.diagnostic.message == _over_ref_message("str")
    assert _start_line(error) == _line(text, "over: ")


def test_isinstance_keeps_its_own_error():
    text = _map_ref("${isinstance(input.a, int)}", "  a: !int\n")
    error = _error(text)
    assert "isinstance is not supported in a map node's over:" in str(error)
    assert _start_line(error) == _line(text, "over: ")

    text = _wait_ref("${isinstance(input.a, int)}", "  a: !int\n")
    error = _error(text)
    assert "isinstance is not supported in a wait node's until:" in str(error)
    assert _start_line(error) == _line(text, "until: ")


def test_tagged_wrong_reference_reports_the_type_first():
    text = _wait_ref("!list(str) ${input.n}", "  n: !int\n")
    error = _error(text)
    assert error.diagnostic.message == _until_ref_message("int")
    assert _start_line(error) == _line(text, "until: ")

    text = _wait_ref("!date ${input.d}", "  d: !date\n")
    error = _error(text)
    assert error.diagnostic.message.startswith("a tag types a value")
    assert _start_line(error) == _line(text, "until: ")


def test_reference_check_reaches_defs():
    text = """\
id: f
name: f
defs:
  pause:
    input:
      n: !int
    nodes:
      w:
        kind: wait
        until: ${input.n}
      after:
        kind: code
        depends_on: [w]
        output: int
        code: |
          return 1
    output: ${after.output}
nodes:
  a:
    kind: call
    call: pause
    input:
      n: 1
output: ${a.output}
"""
    error = _error(text)
    assert _until_ref_message("int") in str(error)
    assert _start_line(error) == _line(text, "until: ")


def test_dangling_reference_keeps_its_error():
    text = _map_ref("${input.nope}")
    error = _error(text)
    assert error.diagnostic.message == (
        "flow has unresolved references:\n"
        "  node 'm' map over: reference ${input.nope} points to unknown flow input 'nope'"
    )


def test_record_typedef_is_named():
    text = _map_ref(
        "${input.r}", "  r: !Buy\n", typedefs="typedefs:\n  Buy: {action: str}\n"
    )
    error = _error(text)
    assert error.diagnostic.message == _over_ref_message("Buy")
    assert _start_line(error) == _line(text, "over: ")


def test_none_typed_source_is_a_load_error():
    assert _error(_map_ref("${input.z}", "  z: !None\n")).diagnostic.message == (
        _over_ref_message("None")
    )
    assert _error(_wait_ref("${input.z}", "  z: !None\n")).diagnostic.message == (
        _until_ref_message("None")
    )


def test_type_error_comes_before_the_cycle_error():
    # `after` waits on `w`, so reading it in `until:` is also a cycle.
    text = _wait_ref("${after.output}")
    assert _error(text).diagnostic.message == _until_ref_message("int")


def _case_routed(branch: str, field: str) -> str:
    """A flow whose CASE routes to `m` only in union member `branch` of `d`'s output."""
    node = (
        "  m:\n    kind: map\n    over: ${d.output.items}\n    call: child\n"
        "    input:\n      x: 1\n"
        if field == "over"
        else "  m:\n    kind: wait\n    until: ${d.output.when}\n"
    )
    return f"""\
id: p
name: p
typedefs:
  A: {{tag: 'Literal(a)', items: list(str), when: date}}
  B: {{tag: 'Literal(b)', items: int, when: int}}
  U: Union(A, B)
nodes:
  d:
    kind: code
    output: U
    code: return None
  r:
    kind: case
    on: ${{d.output.tag}}
    cases:
      - when: {branch}
        then: m
    else: z
{node}  z:
    kind: code
    code: return 1
output: 1
defs:
  child:
    input:
      x: !Any
    nodes:
      e:
        kind: code
        input:
          x: ${{input.x}}
        code: return 1
    output: ${{e.output}}
"""


@pytest.mark.parametrize("field", ["over", "until"])
def test_a_case_branch_narrows_the_source(field):
    load_flow(_case_routed("a", field))
    message = _error(_case_routed("b", field)).diagnostic.message
    expected = (
        _over_ref_message("int") if field == "over" else _until_ref_message("int", node="m")
    )
    assert message == expected
