"""Load checks for value entries in `nodes:` (`total: !int ${a.output + b.output}`).

A tagged entry lowers at load to a CODE node with the entry's id and one typed input
named after it, so the tag is checked like a typed binding's: a span-free literal is
checked while the entry is read, a span's inferred type must not be disjoint from the
tag, and a nested tag checks its own value. A tag never fills in a missing field. These
tests cover the located errors (a tag without a value, a mistyped literal, a spaced or
unknown tag, a nested mismatch, a disjoint span or flow call, a span reading the entry
itself, a flow call inside `${...}`), secret literals, which are rejected before any
check could echo them, and the flow fingerprint.
"""

import re

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compose import LoadError, load_flow
from yian.nodes.base import NodeKind

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

    assert "a flow call 'f' inside ${...} is not supported" in str(error)
    assert _error_line(error) == _line_of(text, "  x: !str ${ f(")


def test_an_entry_reading_itself_is_rejected():
    text = _flow("  x: !object\n    a: ${x.output}\n")

    error = _load_error(text)

    assert str(error) == "flow has a cycle involving ['x']; flows must be acyclic"
    assert _error_line(error) == _line_of(text, "  x: !object")


@pytest.mark.parametrize(
    "entry", ["  api_key: !str sk-live-123\n", "  api_key: !int sk-live-123\n"]
)
def test_a_secret_named_entry_literal_is_rejected_without_echoing_it(entry):
    text = _flow(entry, output="${api_key.output}")

    error = _load_error(text)

    assert str(error) == (
        "value entry 'api_key' is a secret name, so its value can't be written in the "
        "flow file; pass it as an input (${input.api_key}) or from the host"
    )
    assert _error_line(error) == _line_of(text, "api_key:")


def test_a_nested_secret_literal_is_rejected_before_the_value_is_read():
    text = _flow(
        "  cfg: !object\n    access_token: abc\n    x: ${s.output}\n",
        output="${cfg.output}",
    )

    error = _load_error(text)

    assert str(error) == (
        "value entry 'cfg' has key 'access_token' (a secret name) with a literal value; "
        "pass "
        "the value as an input (${input.cfg}) or from the host"
    )
    assert "abc" not in str(error)


def test_a_nested_secret_binding_loads():
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
