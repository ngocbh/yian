"""Untyped values that cross a JSON boundary are checked as JSON values."""

from datetime import date
from enum import Enum
import json
from pathlib import Path
import re
import textwrap
from typing import Any

import pytest
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.diagnostics import OriginRef
from yian.events import NodeStarted
from yian.nodes.contracts import PartialRecordContract, ValueContract
from yian.run import resume_flow, run_flow
from yian.runtime.log import canonical_typed, decode_record
from yian.runtime.state import _canonical_output_value
from yian.typesys.values import (
    FileRef,
    Type,
    TypeCheckError,
    ValueKind,
    build_value,
    reject_non_json,
    value_repr,
)


def _cyclic() -> list[Any]:
    """Return a list that contains itself."""
    value: list[Any] = []
    value.append(value)
    return value


@pytest.mark.parametrize(
    ("value", "message", "path"),
    [
        (
            {"d": {"x": [1, date(2026, 1, 1)]}},
            "output at d.x[1]: datetime.date(2026, 1, 1) is not a JSON value",
            ("d", "x", 1),
        ),
        ({1: "a"}, "output: object keys must be text, got 1", ()),
        ({"a": {1: "a"}}, "output at a: object keys must be text, got 1", ("a",)),
        ({"s": {1, 2}}, "output at s: {1, 2} is not a JSON value", ("s",)),
        ({1, 2}, "output: {1, 2} is not a JSON value", ()),
        (_cyclic(), "output at [0]: value contains itself", (0,)),
    ],
)
def test_reject_non_json_locates_the_bad_part(value, message, path):
    with pytest.raises(TypeCheckError) as caught:
        reject_non_json(value, "output")
    assert str(caught.value) == message
    assert caught.value.path == path


def test_reject_non_json_rejects_a_top_level_object():
    with pytest.raises(TypeCheckError) as caught:
        reject_non_json(object(), "output")
    assert re.fullmatch(r"output: <object object at .*> is not a JSON value", str(caught.value))
    assert caught.value.path == ()


@pytest.mark.parametrize(
    "value",
    [FileRef(uri="f"), [FileRef(uri="f")], {"f": FileRef(uri="f")}],
    ids=["top", "item", "field"],
)
def test_reject_non_json_accepts_files_only_when_asked(value):
    with pytest.raises(TypeCheckError, match=re.escape("is not a JSON value")):
        reject_non_json(value, "output")
    assert reject_non_json(value, "output", files=True) is None


def test_reject_non_json_with_files_still_rejects_other_values():
    with pytest.raises(TypeCheckError) as caught:
        reject_non_json({"f": FileRef(uri="f"), "d": date(2026, 1, 1)}, "output", files=True)
    assert str(caught.value) == "output at d: datetime.date(2026, 1, 1) is not a JSON value"
    assert caught.value.path == ("d",)


@pytest.mark.parametrize(
    "value",
    [None, True, 2**80, 1.5, "x", {"t": (1, 2)}, [None, {"a": []}]],
)
def test_reject_non_json_accepts_json_values(value):
    assert reject_non_json(value, "output") is None


@pytest.mark.parametrize(
    ("value", "message", "path"),
    [
        (
            {"d": date(2026, 1, 1)},
            "output at d: datetime.date(2026, 1, 1) is not a JSON value",
            ("d",),
        ),
        ({"s": {1, 2}}, "output at s: {1, 2} is not a JSON value", ("s",)),
        ({1: "a"}, "output: object keys must be text, got 1", ()),
        ({"a": {1: "a"}}, "output at a: object keys must be text, got 1", ("a",)),
    ],
)
def test_canonical_typed_rejects_untyped_non_json_values(value, message, path):
    with pytest.raises(TypeCheckError) as caught:
        canonical_typed(value, None)
    assert str(caught.value) == message
    assert caught.value.path == path


_FILE_WIRE = {"uri": "f", "mime": None, "name": None}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"f": FileRef(uri="f")}, {"kind": "object", "value": {"f": _FILE_WIRE}}),
        ([FileRef(uri="f")], {"kind": "list[Any]", "value": [_FILE_WIRE]}),
    ],
    ids=["field", "item"],
)
def test_canonical_typed_accepts_untyped_files(value, expected):
    wire, _ = canonical_typed(value, None)
    # A nested file is recorded as a plain object.
    assert wire == expected


class _Key(str, Enum):
    """A text enum, used as a dict key."""

    A = "a"


def test_canonical_typed_records_a_text_subclass_key_as_plain_text():
    wire, _ = canonical_typed({_Key.A: 1}, None)
    assert wire == {"kind": "object", "value": {"a": 1}}
    assert [type(key) for key in wire["value"]] == [str]


def test_canonical_typed_does_not_walk_a_built_value(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a built value was walked")

    monkeypatch.setattr("yian.runtime.log.reject_non_json", fail)
    wire, _ = canonical_typed(build_value({"a": 1}), None)
    assert wire["value"] == {"a": 1}


@pytest.mark.parametrize(
    ("value", "declared", "message"),
    [
        ({1: "a"}, ValueKind.OBJECT, "object keys must be text, got 1"),
        ({"s": {1, 2}}, ValueKind.OBJECT, "{1, 2} is not a JSON value"),
        ({1: "a"}, ValueKind.STRING, "does not match declared type str"),
    ],
)
def test_canonical_typed_reports_the_type_check_for_typed_values(value, declared, message):
    with pytest.raises(TypeCheckError, match=re.escape(message)):
        canonical_typed(value, declared)


@pytest.mark.parametrize(
    ("value", "declared"),
    [("\ud800", ValueKind.STRING), ({"k": "\ud800"}, ValueKind.OBJECT), ("\ud800", None)],
    ids=["str", "object", "untyped"],
)
def test_canonical_typed_still_rejects_lone_surrogates(value, declared):
    with pytest.raises(ValueError, match="^Unicode surrogate is not valid canonical JSON$"):
        canonical_typed(value, declared)


# --------------------------------------------------------------------------- #
# Flows: an untyped output crosses the same boundary
# --------------------------------------------------------------------------- #

_FILE = "__import__('yian.typesys.values', fromlist=['FileRef']).FileRef(uri='f')"


def _flow(code: str, *, output: str | None = None) -> str:
    """Return a flow whose CODE node `mk` returns `code`, typed `output`, as the flow output.

    Args:
        code (`str`): The Python expression `mk` returns.
        output (`str | None`, *optional*, defaults to `None`): The `output:` type of `mk`,
            or `None` for an untyped node.
    """
    typed = "" if output is None else f"    output: {output}\n"
    return (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  mk:\n"
        "    kind: code\n"
        "    input:\n"
        "      s: ${input.s}\n"
        + typed
        + f"    code: |\n      return {code}\n"
        "output: ${mk.output}\n"
    )


def _run(text: str, **kwargs: Any):
    """Load and run `text` with `s: "x"`."""
    return run_flow(load_flow(text), {"s": "x"}, **kwargs)


def _ok(text: str) -> Any:
    """Run `text` and return its output, asserting it succeeded."""
    result = _run(text)
    assert result.status == "succeeded", result.failure
    return result.output


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            '{"d": __import__("datetime").date(2026, 1, 1)}',
            "output at d: datetime.date(2026, 1, 1) is not a JSON value",
        ),
        (
            '{"d": __import__("datetime").datetime(2026, 1, 1)}',
            "output at d: datetime.datetime(2026, 1, 1, 0, 0) is not a JSON value",
        ),
        (
            '[1, __import__("datetime").date(2026, 1, 1)]',
            "output at [1]: datetime.date(2026, 1, 1) is not a JSON value",
        ),
        ('{"s": {1, 2}}', "output at s: {1, 2} is not a JSON value"),
        ('{"a": {1: "a"}}', "output at a: object keys must be text, got 1"),
        (
            '{"x": __import__("decimal").Decimal("1")}',
            "output at x: Decimal('1') is not a JSON value",
        ),
    ],
)
def test_an_untyped_output_rejects_a_non_json_part_with_its_path(code, message):
    result = _run(_flow(code))
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"
    assert result.failure.message == message


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ('{1: "a"}', r"output: object keys must be text, got 1"),
        ("{1, 2}", r"output: \{1, 2\} is not a JSON value"),
        ("object()", r"output: <object object at .*> is not a JSON value"),
        (
            '__import__("datetime").date(2026, 1, 1)',
            r"output: datetime\.date\(2026, 1, 1\) is not a JSON value",
        ),
        ('{"o": object()}', r"output at o: <object object at .*> is not a JSON value"),
    ],
)
def test_an_untyped_inline_return_fails_at_the_output(code, message):
    result = _run(_flow(code))
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"
    assert re.fullmatch(message, result.failure.message), result.failure.message
    # An untyped output has no `output:` contract, so the failure traces to the node.
    assert result.failure.trace == (OriginRef("/s:root/s:node/s:mk"),)


class _NoRepr:
    """A value whose repr raises."""

    def __repr__(self) -> str:
        raise RuntimeError("no repr")


def test_value_repr_names_the_type_when_repr_raises():
    assert value_repr(_NoRepr()) == "<_NoRepr object>"


@pytest.mark.parametrize("output", [None, "object"], ids=["untyped", "object"])
def test_an_inline_return_whose_repr_raises_fails_at_the_output(output):
    typed = "" if output is None else f"    output: {output}\n"
    text = (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  mk:\n"
        "    kind: code\n"
        "    input:\n"
        "      s: ${input.s}\n"
        + typed
        + "    code: |\n"
        "      class R:\n"
        "          def __repr__(self):\n"
        "              raise RuntimeError('no repr')\n"
        "      return {'r': R()}\n"
        "output: ${mk.output}\n"
    )
    result = _run(text)
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"
    assert "<R object>" in result.failure.message, result.failure.message


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("object", "object keys must be text, got 1"),
        ("Any", "object keys must be text, got 1"),
        ("str", "{1: 'a'} does not match declared type str"),
    ],
)
def test_a_typed_inline_return_with_a_non_text_key_fails_at_the_output(output, message):
    result = _run(_flow('{1: "a"}', output=output))
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"
    assert result.failure.message == message


def _body_flow(body: str) -> str:
    """Return a flow whose untyped CODE node `mk` runs the inline `body`."""
    return (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  mk:\n"
        "    kind: code\n"
        "    input:\n"
        "      s: ${input.s}\n"
        "    code: |\n"
        + textwrap.indent(body, "      ")
        + "output: ${mk.output}\n"
    )


@pytest.mark.parametrize(
    "body",
    [
        "import dataclasses\n"
        "@dataclasses.dataclass\n"
        "class D:\n"
        "    x: int = dataclasses.field(init=False)\n"
        "return {'d': D()}\n",
        "class L(list):\n"
        "    def __iter__(self):\n"
        "        raise RuntimeError('cannot iterate')\n"
        "return L([1])\n",
        "class M(dict):\n"
        "    def items(self):\n"
        "        raise KeyError('items')\n"
        "return M(a=1)\n",
    ],
    ids=["unset-dataclass-field", "list-iter-raises", "dict-items-raises"],
)
def test_an_inline_return_that_breaks_the_output_check_fails_the_node(body):
    result = _run(_body_flow(body))
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"


def test_a_reference_return_that_breaks_the_output_check_fails_the_node():
    text = (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  mk:\n"
        "    kind: code\n"
        "    input:\n"
        "      s: ${input.s}\n"
        "    code: tests.engine._compose_codefns:unreadable_list\n"
        "output: ${mk.output}\n"
    )
    result = _run(text)
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"
    assert result.failure.message == "cannot iterate"


@pytest.mark.parametrize(
    ("code", "value"),
    [
        ('{"t": (1, 2)}', {"t": [1, 2]}),
        ("(1, 2)", [1, 2]),
        ("2**80", 2**80),
        ("None", None),
        ('"x"', "x"),
        ('[1, "a", {"k": 2.5}]', [1, "a", {"k": 2.5}]),
    ],
)
def test_an_untyped_output_accepts_json_values(code, value):
    assert _ok(_flow(code)) == value


def test_an_untyped_output_accepts_a_file():
    assert _ok(_flow(_FILE)) == FileRef(uri="f")


def _partial_record_nodes(indent: str = "  ") -> str:
    """Return nodes `a` (a typed `file`) and `b` (untyped), indented under `nodes:`."""
    return textwrap.indent(
        "a:\n"
        "  kind: code\n"
        "  input:\n    s: ${input.s}\n"
        "  output: file\n"
        f"  code: |\n    return {_FILE}\n"
        "b:\n"
        "  kind: code\n"
        "  input:\n    s: ${input.s}\n"
        "  code: |\n    return [1]\n",
        indent,
    )


_PARTIAL_RECORD_FLOW = (
    "id: p\nname: p\n"
    "input:\n  s: !str\n"
    "nodes:\n"
    + _partial_record_nodes()
    + "output:\n  f: ${a.output}\n  g: ${b.output}\n"
)

_PARTIAL_RECORD_BODY = (
    "defs:\n"
    "  body:\n"
    "    input:\n      s: !str\n"
    "    nodes:\n"
    + _partial_record_nodes("      ")
    + "    output:\n      f: ${a.output}\n      g: ${b.output}\n"
)


def test_a_partial_record_output_holds_a_file():
    assert _ok(_PARTIAL_RECORD_FLOW) == {"f": _FILE_WIRE, "g": [1]}


def test_a_call_body_returns_a_partial_record_holding_a_file():
    text = (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        + _PARTIAL_RECORD_BODY
        + "nodes:\n"
        "  c:\n    kind: call\n    call: body\n    input:\n      s: ${input.s}\n"
        "output: ${c.output}\n"
    )
    assert _ok(text) == {"f": _FILE_WIRE, "g": [1]}


def test_a_map_body_returns_a_partial_record_holding_a_file():
    text = (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        + _PARTIAL_RECORD_BODY
        + "nodes:\n"
        "  m:\n    kind: map\n    over: ['x', 'y']\n    call: body\n"
        "    input:\n      s: ${item}\n"
        "output: ${m.output}\n"
    )
    assert _ok(text) == [{"f": _FILE_WIRE, "g": [1]}] * 2


def test_a_partial_record_output_holds_an_untyped_file():
    text = (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  u:\n    kind: code\n    input:\n      s: ${input.s}\n"
        f"    code: |\n      return {_FILE}\n"
        "  b:\n    kind: code\n    input:\n      s: ${input.s}\n"
        "    code: |\n      return 1\n"
        "output:\n  f: ${u.output}\n  g: ${b.output}\n"
    )
    assert _ok(text) == {"f": _FILE_WIRE, "g": 1}


def test_a_map_body_returns_an_untyped_file():
    text = (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "defs:\n"
        "  body:\n"
        "    input:\n      s: !str\n"
        "    nodes:\n"
        "      u:\n        kind: code\n        input:\n          s: ${input.s}\n"
        f"        code: |\n          return {_FILE}\n"
        "    output: ${u.output}\n"
        "nodes:\n"
        "  m:\n    kind: map\n    over: ['x', 'y']\n    call: body\n"
        "    input:\n      s: ${item}\n"
        "output: ${m.output}\n"
    )
    assert _ok(text) == [_FILE_WIRE] * 2


def _literal_flow(g: str) -> str:
    """Return a flow whose untyped output record is `{g: <g>, h: ${b.output}}`."""
    return (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  b:\n    kind: code\n    input:\n      s: ${input.s}\n"
        "    code: |\n      return 1\n"
        f"output:\n  g: {g}\n  h: ${{b.output}}\n"
    )


@pytest.mark.parametrize(
    ("g", "message"),
    [
        (
            "2026-01-01",
            "'g': `2026-01-01` is not a JSON value (YAML reads it as a date); "
            'quote it to keep it as text: `g: "2026-01-01"`',
        ),
        (
            "{d: 2026-01-01}",
            "'g.d': `2026-01-01` is not a JSON value (YAML reads it as a date); "
            'quote it to keep it as text: `"2026-01-01"`',
        ),
    ],
)
def test_an_unquoted_date_literal_in_an_untyped_output_fails_at_load(g, message):
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_literal_flow(g))


def test_a_quoted_date_literal_in_an_untyped_output_stays_text():
    assert _ok(_literal_flow('"2026-01-01"')) == {"g": "2026-01-01", "h": 1}


def _partial_file_contract() -> PartialRecordContract:
    """Return a partial record with a typed `file` field `f` and an opaque field `g`."""
    return PartialRecordContract(
        fields=(("f", ValueContract(type=Type.scalar(ValueKind.FILE))), ("g", None))
    )


def test_a_partial_record_checks_its_opaque_field_as_json():
    contract = _partial_file_contract()
    wire, _ = _canonical_output_value({"f": FileRef(uri="f"), "g": [1]}, contract, None)
    assert wire["value"] == {"f": _FILE_WIRE, "g": [1]}

    with pytest.raises(TypeCheckError) as caught:
        _canonical_output_value({"f": FileRef(uri="f"), "g": {1, 2}}, contract, None)
    assert str(caught.value) == "output at g: {1, 2} is not a JSON value"
    assert caught.value.path == ("g",)


# --------------------------------------------------------------------------- #
# Replay: a recorded untyped value re-crosses the boundary unchanged
# --------------------------------------------------------------------------- #


def _committed(path: Path) -> dict[str, dict]:
    """Return each node's committed wire value from the log at `path`."""
    values: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        decode_record(line)  # every record still validates
        raw = json.loads(line)
        if raw["kind"] == "node_succeeded":
            values[raw["payload"]["node_id"]] = raw["payload"]["value"]
    return values


@pytest.mark.parametrize(
    ("text", "expected", "wires"),
    [
        (
            _flow('{"a": [1, None], "t": (1, 2)}'),
            {"a": [1, None], "t": [1, 2]},
            {"mk": {"kind": "object", "value": {"a": [1, None], "t": [1, 2]}}},
        ),
        (
            _PARTIAL_RECORD_FLOW,
            {"f": _FILE_WIRE, "g": [1]},
            {
                "a": {"kind": "file", "value": _FILE_WIRE},
                "b": {"kind": "list[int]", "value": [1]},
                "__end__": {"kind": "object", "value": {"f": _FILE_WIRE, "g": [1]}},
            },
        ),
    ],
    ids=["untyped", "partial-record"],
)
def test_resuming_a_finished_run_replays_untyped_values(tmp_path: Path, text, expected, wires):
    path = tmp_path / "run.jsonl"
    first = _run(text, log=path)
    assert first.status == "succeeded", first.failure
    assert first.output == expected
    committed = _committed(path)
    assert {node: committed[node] for node in wires} == wires
    before = path.read_bytes()

    resumed = resume_flow(load_flow(text), log=path)

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == expected
    assert not [event for event in resumed.events if isinstance(event, NodeStarted)]
    assert path.read_bytes() == before
