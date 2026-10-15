"""Load-time checks and types for dotted reads of a file's uri, mime and name."""

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.compose.helpers.output_contracts import project_output_contract
from yian.expr import infer_expr_type
from yian.expr.grammar import parse_expr
from yian.nodes.code import CodeNode
from yian.nodes.contracts import ValueContract
from yian.run import run_flow
from yian.runtime.state import GrowthRejected, _validate_growth_wiring_refs
from yian.typesys.values import Type, ValueKind, file_field_type

F = "__import__('yian.typesys.values', fromlist=['FileRef']).FileRef(uri='f')"
NAMED_F = (
    "__import__('yian.typesys.values', fromlist=['FileRef'])"
    ".FileRef(uri='f', mime='text/plain', name='a.txt')"
)
URII = "reads field 'urii' on a file, which has only uri, mime, and name"
STR = Type.scalar(ValueKind.STRING)
OPTIONAL_STR = Type(kind=ValueKind.STRING, nullable=True)
FILE = Type.scalar(ValueKind.FILE)

# `a` holds a file, an absent `Optional(file)`, a record with an `Optional(str)` field
# to compare against, and the opaque `object` and `Any` slots.
_PRODUCER = f"""\
  a:
    kind: code
    output:
      f: file
      g: Optional(file)
      r: {{opt: Optional(str), s: str}}
      o: object
      y: Any
    code: |
      return {{'f': {NAMED_F}, 'g': None, 'r': {{'opt': None, 's': 'x'}}, 'o': {{'k': 1}}, 'y': 1}}
"""

_CHILD = """\
defs:
  child:
    input:
      s: !str
    nodes:
      k: {kind: code, input: {s: "${input.s}"}, code: "return inputs['s']", output: str}
    output: ${k.output}
  child_int:
    input:
      s: !int
    nodes:
      k: {kind: code, input: {s: "${input.s}"}, code: "return 1", output: int}
    output: ${k.output}
"""


def _flow(nodes: str, output: str = "${b.output}", head: str = "", tail: str = "") -> str:
    """A flow whose nodes are the producer `a`, then `nodes`."""
    return f"id: p\nname: p\n{head}nodes:\n{_PRODUCER}{nodes}output: {output}\n{tail}"


def _reader(binding: str) -> str:
    """A CODE node `b` binding `x: <binding>` and returning it."""
    return f"  b:\n    kind: code\n    input:\n      x: {binding}\n    code: return inputs['x']\n"


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line_of(text: str, marker: str) -> int:
    for number, line in enumerate(text.splitlines(), start=1):
        if marker in line:
            return number
    raise AssertionError(f"{marker!r} not in text")


def _assert_located_error(text: str, message: str, marker: str) -> None:
    """Assert loading `text` fails with `message`, located at the line holding `marker`."""
    error = _load_error(text)
    assert message in str(error)
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, marker)


def _ok(text: str, inputs: dict | None = None) -> object:
    result = run_flow(load_flow(text), inputs or {})
    assert result.status == "succeeded", result.failure
    return result.output


# --- file_field_type --- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [("uri", STR), ("mime", OPTIONAL_STR), ("name", OPTIONAL_STR), ("urii", None),
     ("__class__", None), ("", None)],
)
def test_file_field_type_gives_the_type_of_each_file_field(name, expected):
    assert file_field_type(name) == expected


# --- other names are located load errors at every reader --- #

_CASE = """\
  gate:
    kind: case
{cases}
    else: c
  b: {{kind: code, code: "return 1", output: int}}
  c: {{kind: code, code: "return 2", output: int}}
"""


@pytest.mark.parametrize(
    ("nodes", "output", "tail", "marker"),
    [
        (_reader("${a.output.f.urii}"), "${b.output}", "", "x: ${a.output.f.urii}"),
        (_reader('"u=${a.output.f.urii}"'), "${b.output}", "", "u=${a.output.f.urii}"),
        (
            _CASE.format(cases="    cases:\n      - when: \"a.output.f.urii == 'f'\"\n"
                         "        then: b"),
            "${b.output | c.output}", "", "when: \"a.output.f.urii",
        ),
        (
            _CASE.format(cases="    on: ${a.output.f.urii}\n    cases:\n      - when: f\n"
                         "        then: b"),
            "${b.output | c.output}", "", "on: ${a.output.f.urii}",
        ),
        (_reader("1"), "${a.output.f.urii}", "", "output: ${a.output.f.urii}"),
        (_reader("1"), "${b.output}", "asserts:\n  - \"a.output.f.urii == 'f'\"\n",
         "a.output.f.urii == 'f'"),
    ],
    ids=["input", "template", "case-when", "case-on", "flow-output", "flow-assert"],
)
def test_another_field_of_a_file_is_a_located_load_error(nodes, output, tail, marker):
    _assert_located_error(_flow(nodes, output, tail=tail), URII, marker)


def test_another_field_of_a_top_level_file_output_is_a_load_error():
    text = (
        f"id: p\nname: p\nnodes:\n  a:\n    kind: code\n    output: file\n"
        f"    code: return {F}\n{_reader('${a.output.urii}')}output: ${{b.output}}\n"
    )
    _assert_located_error(text, URII, "x: ${a.output.urii}")


def test_another_field_of_an_optional_file_is_a_load_error():
    _assert_located_error(
        _flow(_reader("${a.output.g.urii}")), URII, "x: ${a.output.g.urii}"
    )


def test_a_field_of_a_file_uri_is_a_load_error():
    _assert_located_error(
        _flow(_reader("${a.output.f.uri.x}")),
        "reads field 'x' on type str, which has no fields",
        "x: ${a.output.f.uri.x}",
    )


def test_another_field_of_a_file_flow_input_is_a_load_error():
    text = (
        "id: p\nname: p\ninput:\n  f: !file\nnodes:\n"
        f"{_reader('${input.f.urii}')}output: ${{b.output}}\n"
    )
    _assert_located_error(text, URII, "x: ${input.f.urii}")


def test_another_field_of_a_carried_file_in_a_loop_predicate_is_a_load_error():
    text = _flow(
        "  repeat:\n    kind: loop\n    until: \"f.urii == 'f'\"\n    max: 3\n"
        "    input:\n      f: ${a.output.f}\n"
        "    nodes:\n      bump:\n        kind: code\n        input:\n          f: ${input.f}\n"
        "        output:\n          f: file\n        code: \"return {'f': inputs['f']}\"\n"
        "    output: ${bump.output}\n",
        "${repeat.output.f.uri}",
    )
    error = _load_error(text)
    assert f"`until:` reference ${{f.urii}} {URII}" in str(error)


def test_a_carried_file_uri_in_a_loop_predicate_loads():
    text = _flow(
        "  repeat:\n    kind: loop\n    until: \"f.uri == 'f'\"\n    max: 3\n"
        "    input:\n      f: ${a.output.f}\n"
        "    nodes:\n      bump:\n        kind: code\n        input:\n          f: ${input.f}\n"
        "        output:\n          f: file\n        code: \"return {'f': inputs['f']}\"\n"
        "    output: ${bump.output}\n",
        "${repeat.output.f.uri}",
    )
    assert _ok(text) == "f"


# --- MAP items, unions and constructors --- #


def test_another_field_of_a_file_item_in_a_named_map_is_a_load_error():
    text = (
        f"id: p\nname: p\n{_CHILD}input:\n  fs: !list(file)\nnodes:\n"
        "  b:\n    kind: map\n    over: ${input.fs}\n    call: child\n"
        "    input:\n      s: ${item.urii}\noutput: ${b.output}\n"
    )
    _assert_located_error(text, URII, "s: ${item.urii}")


def test_another_field_of_a_record_item_file_in_a_named_map_is_a_load_error():
    text = (
        f"id: p\nname: p\ntypedefs:\n  Item: {{f: file}}\n{_CHILD}input:\n  rs: !list(Item)\nnodes:\n"
        "  b:\n    kind: map\n    over: ${input.rs}\n    call: child\n"
        "    input:\n      s: ${item.f.urii}\noutput: ${b.output}\n"
    )
    _assert_located_error(text, URII, "s: ${item.f.urii}")


def test_the_uri_of_a_file_item_in_a_named_map_feeds_a_str_input():
    text = (
        f"id: p\nname: p\n{_CHILD}input:\n  fs: !list(file)\nnodes:\n"
        "  b:\n    kind: map\n    over: ${input.fs}\n    call: child\n"
        "    input:\n      s: ${item.uri}\noutput: ${b.output}\n"
    )
    assert _ok(text, {"fs": [{"uri": "x", "mime": None, "name": None}]}) == ["x"]


_UNION_HEAD = (
    "typedefs:\n  A: {f: file, k: int}\n  B: {f: file, j: str}\n  AB: Union(A, B)\n"
)
_UNION_PRODUCER = (
    f"  u:\n    kind: code\n    output: AB\n    code: |\n      return {{'f': {F}, 'k': 1}}\n"
)


def test_the_uri_of_a_file_every_union_member_has_loads_and_reads():
    text = _flow(_UNION_PRODUCER + _reader("${u.output.f.uri}"), head=_UNION_HEAD)
    assert _ok(text) == "f"


def test_another_field_of_a_file_every_union_member_has_is_a_load_error():
    text = _flow(_UNION_PRODUCER + _reader("${u.output.f.urii}"), head=_UNION_HEAD)
    _assert_located_error(text, URII, "x: ${u.output.f.urii}")


_REC_HEAD = "typedefs:\n  Rec: {f: file}\n"


def test_another_field_of_a_file_after_a_constructor_is_a_load_error():
    text = _flow(_reader("${Rec(f=a.output.f).f.urii}"), head=_REC_HEAD)
    assert URII in str(_load_error(text))


def test_the_uri_of_a_file_after_a_constructor_is_typed_str():
    assert _ok(_flow(_reader("${Rec(f=a.output.f).f.uri}"), head=_REC_HEAD)) == "f"
    error = _load_error(_flow(_reader("!int ${Rec(f=a.output.f).f.uri}"), head=_REC_HEAD))
    assert "declared int, but the value is str" in str(error)


def test_a_constructor_trailer_on_an_optional_file_is_nullable():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"f": Type(kind=ValueKind.FILE, nullable=True)},
        required=frozenset({"f"}),
    )
    types = {"R": record}.get

    def infer(text: str) -> Type | None:
        return infer_expr_type(parse_expr(text), lambda _path: None, types)

    assert infer("R(x).f.uri") == OPTIONAL_STR
    assert infer("R(x).f.mime") == OPTIONAL_STR
    assert infer("R(x).f.urii") is None


# --- inline bodies --- #


def _inline_call(binding: str) -> str:
    return (
        "id: p\nname: p\ninput:\n  f: !file\nnodes:\n"
        f"  b:\n    kind: call\n    input:\n      s: {binding}\n    nodes:\n"
        "      k: {kind: code, input: {s: \"${input.s}\"}, code: \"return inputs['s']\"}\n"
        "    output: ${k.output}\noutput: ${b.output}\n"
    )


def _inline_map(over: str, binding: str) -> str:
    return (
        f"id: p\nname: p\ntypedefs:\n  Item: {{f: file}}\ninput:\n  xs: !{over}\nnodes:\n"
        f"  b:\n    kind: map\n    over: ${{input.xs}}\n    input:\n      s: {binding}\n"
        "    nodes:\n"
        "      k: {kind: code, input: {s: \"${input.s}\"}, code: \"return inputs['s']\"}\n"
        "    output: ${k.output}\noutput: ${b.output}\n"
    )


_FORM = {"uri": "f", "mime": "text/plain", "name": None}


def test_an_inline_call_body_reads_the_uri_of_a_file_input():
    assert _ok(_inline_call("${input.f.uri}"), {"f": _FORM}) == "f"


def test_an_inline_map_body_reads_the_fields_of_a_file_item():
    assert _ok(_inline_map("list(Item)", "${item.f.mime}"), {"xs": [{"f": _FORM}]}) == [
        "text/plain"
    ]
    assert _ok(_inline_map("list(file)", "${item.uri}"), {"xs": [_FORM]}) == ["f"]


@pytest.mark.parametrize(
    "text",
    [
        _inline_call("${input.f.urii}"),
        _inline_map("list(Item)", "${item.f.urii}"),
        _inline_map("list(file)", "${item.urii}"),
    ],
    ids=["call-input", "map-record-item", "map-file-item"],
)
def test_another_field_of_a_file_in_an_inline_body_is_a_located_load_error(text):
    _assert_located_error(text, URII, "s: ${")


def test_an_inline_call_body_input_from_a_file_uri_is_typed_str():
    text = _flow(
        "  b:\n    kind: call\n    input:\n      s: ${a.output.f.uri}\n    nodes:\n"
        "      k:\n        kind: code\n        input: {s: \"${input.s}\"}\n"
        "        code: \"return inputs['s']\"\n        output: str\n"
        "    output: ${k.output}\n"
    )
    assert _ok(text) == "f"


# --- the types of uri, mime and name --- #


def test_the_uri_of_a_file_into_an_int_typed_binding_is_a_load_error():
    error = _load_error(_flow(_reader("!int ${a.output.f.uri}")))
    assert "node 'b' input 'x': declared int, but the value is str" in str(error)


@pytest.mark.parametrize("path", ["a.output.f.uri", "a.output.f.mime", "a.output.r.opt"])
def test_a_str_typed_binding_of_a_file_field_loads_like_a_record_field(path):
    load_flow(_flow(_reader(f"!str ${{{path}}}")))


def test_the_uri_of_an_optional_file_is_optional_str():
    error = _load_error(_flow(_reader("!int ${a.output.g.uri}")))
    assert "declared int, but the value is Optional(str)" in str(error)


def test_project_output_contract_types_the_fields_of_a_file():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"f": FILE, "g": Type(kind=ValueKind.FILE, nullable=True)},
        required=frozenset({"f", "g"}),
    )
    contract = ValueContract(record)

    def projected(*fields: str) -> Type | None:
        result = project_output_contract(contract, list(fields))
        return None if result is None else result.type

    assert projected("f", "uri") == STR
    assert projected("f", "mime") == OPTIONAL_STR
    assert projected("g", "uri") == OPTIONAL_STR
    assert projected("f", "urii") is None


@pytest.mark.parametrize(
    ("child", "path", "message"),
    [
        ("child", "a.output.f.mime", "child expects 'str', source is 'Optional(str)'"),
        ("child", "a.output.f.name", "child expects 'str', source is 'Optional(str)'"),
        ("child", "a.output.r.opt", "child expects 'str', source is 'Optional(str)'"),
        ("child_int", "a.output.f.uri", "child expects 'int', source is 'str'"),
    ],
)
def test_a_file_field_into_a_child_input_is_checked_at_load(child, path, message):
    text = _flow(
        f"  b:\n    kind: call\n    call: {child}\n    input:\n      s: ${{{path}}}\n",
        head=_CHILD,
    )
    assert message in str(_load_error(text))


def test_the_uri_of_a_file_into_a_str_child_input_runs():
    text = _flow(
        "  b:\n    kind: call\n    call: child\n    input:\n      s: ${a.output.f.uri}\n",
        head=_CHILD,
    )
    assert _ok(text) == "f"


def test_a_case_on_the_uri_of_a_file_with_an_int_when_is_a_load_error():
    text = _flow(
        _CASE.format(cases="    on: ${a.output.f.uri}\n    cases:\n      - when: 1\n"
                     "        then: b"),
        "${b.output | c.output}",
    )
    assert "`when: 1` is an int but `on: ${a.output.f.uri}` is `str`" in str(
        _load_error(text)
    )


def test_a_growth_fragment_reading_another_field_of_a_file_is_rejected():
    node = CodeNode("g", code="return None")
    record = Type(kind=ValueKind.OBJECT, fields={"f": FILE}, required=frozenset({"f"}))
    producers = {"a": ValueContract(record)}
    _validate_growth_wiring_refs(
        "g", node, {"v": "${a.output.f.uri}"}, {"a", "g"}, producers.get, set()
    )
    with pytest.raises(GrowthRejected, match=URII):
        _validate_growth_wiring_refs(
            "g", node, {"v": "${a.output.f.urii}"}, {"a", "g"}, producers.get, set()
        )


# --- guards: the three fields load and read; opaque slots stay unchecked --- #


def test_the_fields_of_a_file_load_and_read_at_run():
    text = _flow(
        "  b:\n    kind: code\n    input:\n"
        "      u: ${a.output.f.uri}\n      m: ${a.output.f.mime}\n"
        "      k: ${a.output.f.name}\n      g: ${a.output.g.uri}\n"
        "      t: \"u=${a.output.f.uri}\"\n"
        "    code: return [inputs['u'], inputs['m'], inputs['k'], inputs['g'], inputs['t']]\n"
    )
    assert _ok(text) == ["f", "text/plain", "a.txt", None, "u=f"]


@pytest.mark.parametrize("path", ["a.output.o.zzz", "a.output.y.zzz", "a.output.o.f.urii"])
def test_reads_on_object_and_any_stay_unchecked(path):
    load_flow(_flow(_reader(f"${{{path}}}")))
