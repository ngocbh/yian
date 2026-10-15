"""A literal in an untyped flow output that isn't a JSON value is a load error.

An untyped output must hold JSON values, so an unquoted YAML date or datetime written
in a flow's `output:` (root, a `defs:` body, or an inline CALL/MAP/LOOP body) fails at
load, located at the value, with the quoted form to write. A YAML `!!set` or `!!binary`
literal fails the same way without a hint, and a number that is not finite (`.nan`,
`1e400`) names the spelling YAML reads. A quoted or tagged value keeps its behavior.
"""

from __future__ import annotations

import re

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow

_HEAD = "id: p\nname: p\nnodes:\n  b:\n    kind: code\n    code: return 1\n"
_DATE = "`2026-01-01` is not a JSON value (YAML reads it as a date)"
_HINT = "quote it to keep it as text: "


def _flow(output: str) -> str:
    """Return a flow with producer `b` and the block `output:` lines `output`."""
    return _HEAD + "output:\n" + "".join(f"  {line}\n" for line in output.splitlines())


def _ok(text: str):
    """Load and run `text`, returning its output and asserting it succeeded."""
    result = run_flow(load_flow(text), {})
    assert result.status == "succeeded", result.failure
    return result.output


def test_untagged_date_in_an_untyped_output_field_is_a_load_error():
    """An unquoted date in a record field fails at load, at the field's line."""
    text = _flow("as_of: 2026-01-01\nh: ${b.output}")
    message = f"'as_of': {_DATE}; {_HINT}`as_of: \"2026-01-01\"`"
    with pytest.raises(LoadError, match=re.escape(message)) as excinfo:
        load_flow(text)
    line = text.splitlines().index("  as_of: 2026-01-01") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


@pytest.mark.parametrize(
    ("g", "label"),
    [
        ("{d: 2026-01-01}", "g.d"),
        ("[1, 2026-01-01]", "g[1]"),
        ("{x: [{d: 2026-01-01}]}", "g.x[0].d"),
    ],
)
def test_nested_untagged_date_names_its_path(g, label):
    """A nested unquoted date names its full path; the hint omits the key."""
    message = f"'{label}': {_DATE}; {_HINT}`\"2026-01-01\"`"
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_flow(f"g: {g}\nh: ${{b.output}}"))


def test_a_nested_date_is_located_at_the_value():
    """A nested unquoted date's error range starts at the date, not the field."""
    text = _flow("g:\n  x: 1\n  d: 2026-01-01\nh: ${b.output}")
    with pytest.raises(LoadError, match=re.escape(f"'g.d': {_DATE}")) as excinfo:
        load_flow(text)
    line = text.splitlines().index("    d: 2026-01-01") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


@pytest.mark.parametrize(
    ("as_of", "message"),
    [
        (
            "2026-01-01T10:00:00",
            "'as_of': `2026-01-01T10:00:00` is not a JSON value (YAML reads it as a "
            f"datetime); {_HINT}`as_of: \"2026-01-01T10:00:00\"`",
        ),
        (
            "!!timestamp '2026-01-01'",
            f"'as_of': {_DATE}; {_HINT}`as_of: \"2026-01-01\"`",
        ),
    ],
)
def test_a_datetime_or_timestamp_in_an_untyped_output_is_a_load_error(as_of, message):
    """An unquoted datetime, or a `!!timestamp` scalar, fails at load."""
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_flow(f"as_of: {as_of}\nh: ${{b.output}}"))


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("2026-01-01", f"'output': {_DATE}; {_HINT}`output: \"2026-01-01\"`"),
        ("[2026-01-01]", f"'output[0]': {_DATE}; {_HINT}`\"2026-01-01\"`"),
    ],
)
def test_untagged_date_as_the_whole_output_is_a_load_error(output, message):
    """An unquoted date as the whole `output:`, or an item of it, fails at load."""
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_HEAD + f"output: {output}\n")


_INLINE_CALL = """id: p
name: p
nodes:
  c:
    kind: call
    nodes:
      b:
        kind: code
        code: return 1
    output:
      d: 2026-01-01
      h: ${b.output}
output: ${c.output}
"""

_INLINE_MAP = """id: p
name: p
nodes:
  m:
    kind: map
    over: [1, 2]
    nodes:
      b:
        kind: code
        code: return 1
    output:
      d: 2026-01-01
      h: ${b.output}
output: ${m.output}
"""

_INLINE_LOOP = """id: p
name: p
nodes:
  l:
    kind: loop
    times: 1
    input:
      n: 0
      d: !Any "x"
    nodes:
      b:
        kind: code
        input:
          n: ${input.n}
        code: return inputs['n'] + 1
    output:
      n: ${b.output}
      d: 2026-01-01
output: ${l.output}
"""

_NAMED_CALL = """id: p
name: p
defs:
  body:
    nodes:
      b:
        kind: code
        code: return 1
    output:
      d: 2026-01-01
      h: ${b.output}
nodes:
  c:
    kind: call
    call: body
output: ${c.output}
"""

_NAMED_LOOP = """id: p
name: p
defs:
  step:
    input:
      n: !int
      d: !Any
    nodes:
      b:
        kind: code
        input:
          n: ${input.n}
        code: return inputs['n'] + 1
    output:
      n: ${b.output}
      d: 2026-01-01
nodes:
  l:
    kind: loop
    times: 1
    call: step
    input:
      n: 0
      d: !Any "x"
output: ${l.output}
"""


@pytest.mark.parametrize(
    "text",
    [_INLINE_CALL, _INLINE_MAP, _INLINE_LOOP, _NAMED_CALL, _NAMED_LOOP],
    ids=["inline-call", "inline-map", "inline-loop", "named-call", "named-loop"],
)
def test_untagged_date_in_a_body_output_is_a_load_error(text):
    """An unquoted date in a body's untyped output fails at load."""
    with pytest.raises(LoadError, match=re.escape(f"'d': {_DATE}")):
        load_flow(text)


def test_a_body_output_date_is_located_at_the_value():
    """An unquoted date in a `defs:` body's output is located at the date."""
    with pytest.raises(LoadError, match=re.escape(f"'d': {_DATE}")) as excinfo:
        load_flow(_NAMED_CALL)
    line = _NAMED_CALL.splitlines().index("      d: 2026-01-01") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


@pytest.mark.parametrize(
    ("g", "message"),
    [
        ("!!set {a: null}", "'g': {'a'} is not a JSON value"),
        ("!!binary aGVsbG8=", "'g': b'hello' is not a JSON value"),
    ],
)
def test_set_or_binary_literal_in_an_untyped_output_is_a_load_error(g, message):
    """A YAML set or binary literal fails at load, without a quoting hint."""
    with pytest.raises(LoadError, match=re.escape(message) + "$"):
        load_flow(_flow(f"g: {g}\nh: ${{b.output}}"))


def test_an_aliased_date_names_the_output_field():
    """An alias to an unquoted date names the output field, located at the anchor."""
    text = (
        "id: p\nname: p\nnodes:\n  b:\n    kind: code\n"
        "    input:\n      d: &d 2026-01-01\n    code: return 1\n"
        "output:\n  g: *d\n  h: ${b.output}\n"
    )
    with pytest.raises(LoadError, match=re.escape(f"'g': {_DATE}")) as excinfo:
        load_flow(text)
    line = text.splitlines().index("      d: &d 2026-01-01") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


@pytest.mark.parametrize(
    ("g", "message"),
    [
        (
            "!!omap [a: 2026-01-01]",
            f"'g[0][1]': {_DATE}; {_HINT}`\"2026-01-01\"`",
        ),
        ("!!pairs [a: !!set {x: null}]", "'g[0][1]': {'x'} is not a JSON value"),
    ],
)
def test_a_non_json_value_in_an_omap_pair_is_a_load_error(g, message):
    """A date or set as an `!!omap` or `!!pairs` value fails at load, at the pair."""
    text = _flow(f"g: {g}\nh: ${{b.output}}")
    with pytest.raises(LoadError, match=re.escape(message) + "$") as excinfo:
        load_flow(text)
    lines = text.splitlines()
    line = lines.index(f"  g: {g}") + 1
    frame_range = excinfo.value.diagnostic.frames[-1].range
    assert frame_range.start_line == line
    assert frame_range.start_col == lines[line - 1].index("[") + 2


def test_quoted_or_tagged_dates_keep_their_behavior():
    """A quoted date stays text; a tagged date keeps its tag's load behavior."""
    quoted = _flow('as_of: "2026-01-01"\nh: ${b.output}')
    assert _ok(quoted) == {"as_of": "2026-01-01", "h": 1}
    with pytest.raises(
        LoadError, match=re.escape("is not a valid Any; quote it to keep it as text")
    ):
        load_flow(_flow("as_of: !Any 2026-01-01\nh: ${b.output}"))
    load_flow(_flow('as_of: !date "2026-01-01"\nh: ${b.output}'))
    load_flow(_flow('as_of: "${b.output}-2026-01-01"\nh: ${b.output}'))
    load_flow(_flow("as_of: ${b.output}-2026-01-01\nh: ${b.output}"))


def test_other_untagged_literals_in_untyped_outputs_still_load():
    """JSON literals and a YAML `!!omap` in an untyped output load and run."""
    text = _flow("n: 1\ns: x\nl: [1, null]\nm: {k: true}\nh: ${b.output}")
    assert _ok(text) == {"n": 1, "s": "x", "l": [1, None], "m": {"k": True}, "h": 1}
    omap = _ok(_flow("g: !!omap [a: 1]\nh: ${b.output}"))
    assert omap["g"] == [["a", 1]]


_FINITE = "numbers must be finite"


@pytest.mark.parametrize(
    ("g", "got"),
    [
        (".nan", ".nan"),
        (".inf", ".inf"),
        ("-.inf", "-.inf"),
        (".NaN", ".NaN, which YAML reads as .nan"),
        (".Inf", ".Inf, which YAML reads as .inf"),
        ("+.inf", "+.inf, which YAML reads as .inf"),
        ("-.Inf", "-.Inf, which YAML reads as -.inf"),
        ("1e400", "1e400, which YAML reads as .inf"),
        ("1.0e+400", "1.0e+400, which YAML reads as .inf"),
    ],
)
def test_a_non_finite_number_in_an_untyped_output_is_a_load_error(g, got):
    """A number that is not finite fails at load, naming the spelling YAML reads."""
    text = _flow(f"g: {g}\nh: ${{b.output}}")
    message = f"'g': {_FINITE} (got {got})"
    with pytest.raises(LoadError, match=f"^{re.escape(message)}$") as excinfo:
        load_flow(text)
    line = text.splitlines().index(f"  g: {g}") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


@pytest.mark.parametrize(
    ("output", "label"),
    [
        ("g: [1, .nan]\nh: ${b.output}", "g[1]"),
        ("g: {x: .nan}\nh: ${b.output}", "g.x"),
        ("g: !!float .nan\nh: ${b.output}", "g"),
        ("g: !!omap [{a: .nan}]\nh: ${b.output}", "g[0][1]"),
    ],
)
def test_a_nested_or_core_tagged_non_finite_number_is_a_load_error(output, label):
    """A nested, `!!float`, or `!!omap` pair `.nan` fails at load, on its line."""
    text = _flow(output)
    message = f"'{label}': {_FINITE} (got .nan)"
    with pytest.raises(LoadError, match=f"^{re.escape(message)}$") as excinfo:
        load_flow(text)
    line = text.splitlines().index(f"  {output.splitlines()[0]}") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


@pytest.mark.parametrize(
    ("output", "label"),
    [(".nan", "output"), ("[1, .nan]", "output[1]")],
)
def test_a_non_finite_number_as_the_whole_output_is_a_load_error(output, label):
    """A `.nan` as the whole `output:`, or an item of it, fails at load."""
    text = _HEAD + f"output: {output}\n"
    message = f"'{label}': {_FINITE} (got .nan)"
    with pytest.raises(LoadError, match=f"^{re.escape(message)}$") as excinfo:
        load_flow(text)
    line = text.splitlines().index(f"output: {output}") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


_NAN_BODY = """id: p
name: p
defs:
  sub:
    nodes:
      a:
        kind: code
        code: return 1
      b:
        kind: code
        code: return 2
    output:
      g: .nan
      h: ${b.output}
nodes:
  c:
    kind: call
    call: sub
output: ${c.output}
"""


def test_a_non_finite_number_in_a_body_output_is_a_load_error():
    """A `.nan` in a `defs:` body's untyped output fails at load, at the value."""
    message = f"defs entry 'sub': 'g': {_FINITE} (got .nan)"
    with pytest.raises(LoadError, match=f"^{re.escape(message)}$") as excinfo:
        load_flow(_NAN_BODY)
    line = _NAN_BODY.splitlines().index("      g: .nan") + 1
    assert excinfo.value.diagnostic.frames[-1].range.start_line == line


def test_quoted_tagged_or_finite_numbers_keep_their_behavior():
    """Quoted or `!!str` `.nan` stays text; `!float .nan` fails at run; finite loads."""
    assert _ok(_flow('g: ".nan"\nh: ${b.output}')) == {"g": ".nan", "h": 1}
    assert _ok(_flow("g: !!str .nan\nh: ${b.output}")) == {"g": ".nan", "h": 1}
    tagged = run_flow(load_flow(_flow("g: !float .nan\nh: ${b.output}")), {})
    assert tagged.status == "failed"
    assert tagged.failure.code == "binding.evaluate"
    finite = _ok(_flow("g: [1.5, -2e10, 1e300]\nh: ${b.output}"))
    assert finite == {"g": [1.5, -2e10, 1e300], "h": 1}
