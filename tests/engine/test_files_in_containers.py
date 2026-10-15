"""Files inside typed containers: a `file` slot reads a file's JSON form, an `object`
slot reads a nested file as its mapping, and an untagged union whose members then
share a value is a load error."""

from collections import defaultdict
import copy
from pathlib import Path
import textwrap
from typing import Any

import pytest
from yian.compile.manifest import _canonical_value, _ManifestContext
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.expr.expressions import eval_expr
from yian.expr.grammar import parse_expr
from yian.expr.lineage import ResolvedValue, SourceRef
from yian.nodes.base import _HostStop
from yian.nodes.binding import bind_params
from yian.nodes.contracts import Binding, Parameter, ValueContract
from yian.nodes.loop import LoopNode
from yian.run import resume_flow, run_flow
from yian.runtime.trace import SourceEntry
from yian.typesys.pool import VariablePool
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import (
    FileRef,
    FileValue,
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
    field_of,
    normalize_files,
    types_overlap,
)

from tests.engine._graph_builder import _engine
from tests.engine.test_expand import _child_flow

_REGISTRY = read_typedefs(
    {
        "Rec": {"f": "file", "n": "int"},
        "Form": {"uri": "str", "mime": "Optional(str)", "name": "Optional(str)"},
        "IntUri": {"uri": "int"},
        "Other": {"uri": "str", "other": "str"},
        "XFile": {"x": "file"},
        "XObject": {"x": "object"},
        "Box": {"x": "object"},
        "Num": {"n": "float"},
        "TFile": {"kind": "Literal(file)", "x": "file"},
        "TObject": {"kind": "Literal(object)", "x": "object"},
    }
)

FILE = Type.scalar(ValueKind.FILE)
OBJECT = Type.scalar(ValueKind.OBJECT)
ANY = Type.scalar(ValueKind.ANY)
REF = FileRef(uri="f")
FORM = {"uri": "f", "mime": None, "name": None}
F = "__import__('yian.typesys.values', fromlist=['FileRef']).FileRef(uri='f')"


def _t(text: str) -> Type:
    """Resolve a type expression against this module's typedefs."""
    return type_for(text, _REGISTRY)


def _run(text: str, **kwargs: Any):
    """Load and run a dedented flow with no inputs."""
    return run_flow(load_flow(textwrap.dedent(text)), {}, **kwargs)


def _ok(text: str) -> Any:
    """Run a flow and return its output, asserting it succeeded."""
    result = _run(text)
    assert result.status == "succeeded", result.failure
    return result.output


# --- a file slot reads the JSON form --- #


def test_a_file_slot_reads_the_json_form():
    assert build_value_as(FILE, dict(FORM)) == FileValue(value=REF)


def test_a_file_slot_keeps_a_file_ref():
    assert build_value_as(FILE, REF) == FileValue(value=REF)


def test_the_json_form_keeps_its_mime_and_name():
    value = {"uri": "f", "mime": "text/plain", "name": "a.txt"}

    assert build_value_as(FILE, value).value == FileRef(**value)


@pytest.mark.parametrize(
    ("declared", "value", "expected"),
    [
        ("list(file)", [FORM, REF], [REF, REF]),
        ("Rec", {"f": FORM, "n": 1}, {"f": REF, "n": 1}),
        ("Optional(file)", None, None),
        ("list(Optional(file))", [FORM, None], [REF, None]),
        ("list(Rec)", [{"f": FORM, "n": 1}], [{"f": REF, "n": 1}]),
    ],
)
def test_a_nested_file_slot_reads_the_json_form(declared, value, expected):
    assert build_value_as(_t(declared), value).to_object() == expected


@pytest.mark.parametrize(
    "value",
    [
        {"uri": "f", "name": None},
        {"uri": "f", "mime": None, "name": None, "size": 1},
        {"uri": 1, "mime": None, "name": None},
        {"uri": "f", "mime": 2, "name": None},
    ],
)
def test_a_file_slot_rejects_a_dict_that_is_not_the_json_form(value):
    with pytest.raises(TypeCheckError, match="does not match declared type file"):
        build_value_as(FILE, value)
    with pytest.raises(TypeCheckError, match="does not match declared type file") as caught:
        build_value_as(_t("list(file)"), [value])
    assert caught.value.path == (0,)


# --- an object slot reads a nested file as its mapping --- #


def test_an_object_reads_a_nested_file_as_its_mapping():
    assert build_value_as(OBJECT, {"f": REF}).value == {"f": FORM}


def test_a_list_of_objects_reads_a_nested_file_as_its_mapping():
    flat = build_value_as(ValueKind.LIST_OBJECT, [{"f": REF}, {"g": [REF]}])
    refined = build_value_as(_t("list(object)"), [{"f": REF}])

    assert flat.value == [{"f": FORM}, {"g": [FORM]}]
    assert refined.value == [{"f": FORM}]


def test_an_object_field_reads_a_nested_file_as_its_mapping():
    assert build_value_as(_t("Box"), {"x": {"f": REF}}).value == {"x": {"f": FORM}}


def test_an_object_without_files_is_stored_as_given():
    value = {"items": (1, 2), "nested": {"a": [3]}}

    assert build_value_as(OBJECT, value).value["items"] is value["items"]


def test_reading_nested_files_leaves_the_given_object_unchanged():
    value = {"f": REF, "keep": {"a": 1}}

    stored = build_value_as(OBJECT, value).value

    assert value == {"f": REF, "keep": {"a": 1}}
    assert stored["keep"] is value["keep"]


@pytest.mark.parametrize(
    ("declared", "value"),
    [
        (OBJECT, REF),
        (_t("list(object)"), [REF]),
        (ANY, REF),
        (ANY, {"f": REF}),
        (_t("list(Any)"), [REF]),
        (_t("list(Any)"), [{"f": REF}]),
    ],
)
def test_a_file_where_no_file_position_reads_it_still_fails(declared, value):
    with pytest.raises(TypeCheckError):
        build_value_as(declared, value)


def test_an_object_holding_itself_still_fails():
    cyclic: dict = {}
    cyclic["self"] = cyclic
    cyclic["f"] = REF

    with pytest.raises(TypeCheckError, match="value contains itself"):
        build_value_as(OBJECT, cyclic)


# --- normalize_files --- #


def test_normalize_files_reads_the_json_form_at_a_file_position():
    assert normalize_files(_t("Rec"), {"f": dict(FORM), "n": 1}) == {"f": REF, "n": 1}


def test_normalize_files_reads_a_nested_file_as_a_mapping_at_an_object_position():
    assert normalize_files(_t("Box"), {"x": {"f": REF}}) == {"x": {"f": FORM}}


def test_normalize_files_keeps_an_int_at_a_float_position():
    normalized = normalize_files(_t("Num"), {"n": 3})

    assert type(normalized["n"]) is int


@pytest.mark.parametrize(
    ("declared", "value"),
    [
        ("int", 3),
        ("Num", {"n": 3}),
        ("list(str)", ["a"]),
        ("Rec", {"f": REF, "n": 1}),
        ("list(file)", [REF]),
        ("object", {"a": [1]}),
        ("Any", {"uri": "f", "mime": None, "name": None}),
    ],
)
def test_normalize_files_returns_the_same_value_when_nothing_changes(declared, value):
    assert normalize_files(_t(declared), value) is value


@pytest.mark.parametrize(
    ("members", "expected"),
    [(("XFile", "XObject"), {"x": REF}), (("XObject", "XFile"), {"x": FORM})],
)
def test_normalize_files_picks_an_untagged_union_member_like_the_write_check(
    members, expected
):
    # Both members admit `{"x": FORM}`, so the first one decides how `x` reads.
    union = Type(kind=ValueKind.OBJECT, variants=tuple(_t(name) for name in members))
    value = {"x": dict(FORM)}

    assert normalize_files(union, value) == expected
    assert build_value_as(union, value).to_object() == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"kind": "file", "x": FORM}, {"kind": "file", "x": REF}),
        ({"kind": "object", "x": {"f": REF}}, {"kind": "object", "x": {"f": FORM}}),
    ],
)
def test_normalize_files_picks_a_tagged_union_member_by_its_tag(value, expected):
    assert normalize_files(_t("Union(TFile, TObject)"), value) == expected


def test_normalize_files_reads_an_optional_union():
    declared = _t("Optional(Union(TFile, TObject))")

    assert normalize_files(declared, None) is None
    assert normalize_files(declared, {"kind": "file", "x": FORM}) == {"kind": "file", "x": REF}


def test_normalize_files_reads_a_tuple_as_a_list_only_when_it_changes():
    unchanged = (REF, REF)

    assert normalize_files(_t("list(file)"), (FORM, REF)) == [REF, REF]
    assert normalize_files(_t("list(file)"), unchanged) is unchanged


@pytest.mark.parametrize(
    ("declared", "value"),
    [
        ("Rec", "text"),
        ("list(file)", {"f": FORM}),
        ("Union(TFile, TObject)", {"kind": "other", "x": FORM}),
        ("Union(TFile, TObject)", {"x": FORM}),
        ("Union(XFile, IntUri)", {"y": FORM}),
        ("Union(XFile, IntUri)", [FORM]),
    ],
)
def test_normalize_files_keeps_a_value_that_fails_the_type(declared, value):
    assert normalize_files(_t(declared), value) is value


@pytest.mark.parametrize(
    ("declared", "value"),
    [
        ("file", FORM),
        ("Rec", {"f": FORM, "n": 1}),
        ("list(Rec)", [{"f": REF, "n": 1}, {"f": FORM, "n": 2}]),
        ("Optional(list(Optional(file)))", [FORM, None]),
        ("object", {"f": REF, "g": [REF, 1]}),
        ("list(object)", [{"f": REF}, {"a": 1}]),
        ("Box", {"x": {"f": [REF]}}),
        ("Union(TFile, TObject)", {"kind": "file", "x": FORM}),
        ("Union(XFile, IntUri)", {"x": FORM}),
    ],
)
def test_normalize_files_agrees_with_the_write_check(declared, value):
    typ = _t(declared)

    assert normalize_files(typ, value) == build_value_as(typ, value).to_object()


# --- untagged unions whose members can share a file --- #


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (FILE, OBJECT, True),
        (FILE, ANY, True),
        (FILE, _t("Form"), True),
        (_t("XFile"), _t("XObject"), True),
        (FILE, _t("int"), False),
        (FILE, _t("IntUri"), False),
        (FILE, _t("Other"), False),
    ],
)
def test_a_file_overlaps_the_types_that_admit_its_json_form(a, b, expected):
    assert types_overlap(a, b) is expected
    assert types_overlap(b, a) is expected


_UNION_FLOW = """\
id: u
name: u
typedefs:
  XFile:
    {tag}x: file
  XObject:
    {other}x: object
  U: Union(XFile, XObject)
nodes:
  a:
    kind: code
    output: U
    code: |
      return dict({value}x={F})
output: ${{a.output}}
"""


def test_an_untagged_union_of_a_file_and_an_object_member_is_a_load_error():
    text = _UNION_FLOW.format(tag="", other="", value="", F=F)

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    message = str(caught.value)
    assert "members 'XFile' and 'XObject' can match the same value" in message
    assert "add a tag field" in message


def test_a_tagged_union_of_a_file_and_an_object_member_loads():
    text = _UNION_FLOW.format(
        tag="kind: Literal(file)\n    ",
        other="kind: Literal(object)\n    ",
        value="kind='file', ",
        F=F,
    )

    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure


# --- flows that carry files in typed containers --- #


def test_a_map_over_a_list_of_files_into_a_file_input_succeeds():
    output = _ok(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              f: !file
            nodes:
              c:
                kind: code
                input:
                  f: ${{input.f}}
                code: |
                  return 'ok'
            output: ${{c.output}}
        nodes:
          a:
            kind: code
            output: list(file)
            code: |
              return [{F}, {F}]
          m:
            kind: map
            over: ${{a.output}}
            call: body
            input:
              f: ${{item}}
        output: ${{m.output}}
        """)

    assert output == ["ok", "ok"]


def test_a_loop_carrying_a_file_succeeds():
    output = _ok(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              f: !file
              n: !int
            nodes:
              c:
                kind: code
                input:
                  f: ${{input.f}}
                  n: ${{input.n}}
                output:
                  f: file
                  n: int
                code: |
                  return dict(f=inputs['f'], n=inputs['n'] + 1)
            output: ${{c.output}}
        nodes:
          a:
            kind: code
            output: file
            code: |
              return {F}
          l:
            kind: loop
            times: 3
            call: body
            input:
              f: ${{a.output}}
              n: 0
        output: ${{l.output}}
        """)

    assert output["n"] == 3


def test_a_call_into_a_list_of_files_input_succeeds():
    output = _ok(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              fs: !list(file)
            nodes:
              c:
                kind: code
                input:
                  fs: ${{input.fs}}
                code: |
                  return len(inputs['fs'])
            output: ${{c.output}}
        nodes:
          a:
            kind: code
            output: list(file)
            code: |
              return [{F}]
          k:
            kind: call
            call: body
            input:
              fs: ${{a.output}}
        output: ${{k.output}}
        """)

    assert output == 1


def test_a_call_body_returning_a_record_with_a_file_succeeds():
    _ok(f"""\
        id: p
        name: p
        defs:
          body:
            nodes:
              c:
                kind: code
                output:
                  f: file
                code: |
                  return dict(f={F})
            output: ${{c.output}}
        nodes:
          k:
            kind: call
            call: body
          d:
            kind: code
            input:
              f: ${{k.output.f}}
            code: |
              return 'ok'
        output: ${{d.output}}
        """)


_LIST_FILE_FLOW = textwrap.dedent(f"""\
    id: p
    name: p
    nodes:
      a:
        kind: code
        output: list(file)
        code: |
          return [{F}, {F}]
    output:
      a: ${{a.output}}
    """)


def test_resuming_a_run_with_a_typed_list_of_files_succeeds(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    first = run_flow(load_flow(_LIST_FILE_FLOW), {}, log=path)
    assert first.status == "succeeded", first.failure

    resumed = resume_flow(load_flow(_LIST_FILE_FLOW), log=path)

    assert resumed.status == "succeeded", resumed.failure


# --- typed bindings and constructors deliver the slot's file representation --- #


def _bind(declared: Type, output: Any) -> Any:
    """Bind one parameter to `!declared ${a.output}` over a pool holding `output`."""
    pool = VariablePool()
    pool.set("a", output)
    record = bind_params(
        [Parameter("x")], {"x": Binding("${a.output}", contract=ValueContract(declared))}, pool
    )
    return record["x"]


def test_an_object_binding_reads_a_nested_file_as_its_mapping():
    assert _bind(OBJECT, {"f": REF}) == {"f": FORM}


def test_a_file_binding_reads_the_json_form_as_a_file():
    assert _bind(_t("Rec"), {"f": dict(FORM), "n": 1}) == {"f": REF, "n": 1}


_RECORD_PRODUCER = f"""\
  a:
    kind: code
    output:
      f: file
    code: |
      return dict(f={F})
"""


def test_an_object_binding_delivers_a_nested_file_as_its_mapping():
    output = _ok(
        "id: p\nname: p\nnodes:\n" + _RECORD_PRODUCER
        + "  b:\n    kind: code\n    input:\n      x: !object ${a.output}\n"
        + "    code: |\n      return type(inputs['x']['f']).__name__\n"
        + "output: ${b.output}\n"
    )

    assert output == "dict"


def test_an_object_output_binding_over_a_record_with_a_file_succeeds():
    output = _ok("id: p\nname: p\nnodes:\n" + _RECORD_PRODUCER + "output: !object ${a.output}\n")

    assert output == {"f": FORM}


def test_a_file_binding_delivers_the_json_form_as_a_file():
    output = _ok("""\
        id: p
        name: p
        nodes:
          u:
            kind: code
            code: |
              return dict(uri='f', mime=None, name=None)
          b:
            kind: code
            input:
              f: !file ${u.output}
            code: |
              return type(inputs['f']).__name__
        output: ${b.output}
        """)

    assert output == "FileRef"


def test_a_float_binding_still_delivers_an_int():
    output = _ok("""\
        id: p
        name: p
        nodes:
          i:
            kind: code
            output: int
            code: |
              return 3
          b:
            kind: code
            input:
              n: !float ${i.output}
            code: |
              return type(inputs['n']).__name__
        output: ${b.output}
        """)

    assert output == "int"


def test_a_constructor_delivers_a_nested_file_as_its_mapping_at_an_object_field():
    value = eval_expr(
        parse_expr("Box(x=a)"), lambda path: {"f": REF} if path == "a" else None,
        types={"Box": _t("Box")},
    )

    assert value == {"x": {"f": FORM}}


def test_a_constructor_in_a_flow_delivers_a_nested_file_as_its_mapping():
    output = _ok(
        "id: p\nname: p\ntypedefs:\n  Box:\n    x: object\nnodes:\n" + _RECORD_PRODUCER
        + "  b:\n    kind: code\n    input:\n      t: ${Box(x=a.output)}\n"
        + "    code: |\n      return type(inputs['t']['x']['f']).__name__\n"
        + "output: ${b.output}\n"
    )

    assert output == "dict"


# --- dotted reads of a file's uri, mime and name --- #

_NAMED_F = (
    "__import__('yian.typesys.values', fromlist=['FileRef'])"
    ".FileRef(uri='f', mime='text/plain', name='a.txt')"
)
_PRODUCERS = f"""\
  a:
    kind: code
    output: file
    code: |
      return {F}
  n:
    kind: code
    output: file
    code: |
      return {_NAMED_F}
"""


def _file_flow(nodes: str, output: str = "${b.output}", extra: str = "") -> str:
    """A flow whose nodes are the file producers `a` and `n`, then `nodes`."""
    return f"id: p\nname: p\nnodes:\n{_PRODUCERS}{nodes}output: {output}\n{extra}"


def test_a_binding_reads_the_fields_of_a_file():
    output = _ok(_file_flow(
        "  b:\n    kind: code\n    input:\n"
        "      u: ${a.output.uri}\n      m: ${n.output.mime}\n      k: ${n.output.name}\n"
        "    code: |\n      return [inputs['u'], inputs['m'], inputs['k']]\n"
    ))

    assert output == ["f", "text/plain", "a.txt"]


def test_a_template_binding_reads_the_uri_of_a_file():
    output = _ok(_file_flow(
        '  b:\n    kind: code\n    input:\n      u: "u=${a.output.uri}"\n'
        "    code: |\n      return inputs['u']\n"
    ))

    assert output == "u=f"


def test_a_prompt_reads_the_name_of_a_file():
    flow = load_flow(_file_flow(
        "  b:\n    kind: human_input\n    prompt: \"u=${f.uri} k=${f.name}\"\n"
        "    input:\n      f: ${n.output}\n    output: str\n"
    ))

    result = run_flow(flow, {}, tools={"ask_user": lambda **request: request["question"]})

    assert result.status == "succeeded", result.failure
    assert result.output == "u=f k=a.txt"


_CASE_NODES = """\
  gate:
    kind: case
{cases}
    else: other
  b: {{kind: code, code: "return 'hit'", output: str}}
  other: {{kind: code, code: "return 'else'", output: str}}
"""


@pytest.mark.parametrize(
    "cases",
    [
        "    on: ${a.output.uri}\n    cases:\n      - when: f\n        then: b",
        "    cases:\n      - when: \"a.output.uri == 'f'\"\n        then: b",
    ],
    ids=["on", "when"],
)
def test_a_case_routes_on_the_uri_of_a_file(cases):
    output = _ok(_file_flow(_CASE_NODES.format(cases=cases), "${b.output | other.output}"))

    assert output == "hit"


def test_a_flow_assert_reads_the_uri_of_a_file():
    output = _ok(_file_flow(
        "  b: {kind: code, code: \"return 'ok'\", output: str}\n",
        extra="asserts:\n  - \"a.output.uri == 'f' and n.output.name == 'a.txt'\"\n",
    ))

    assert output == "ok"


def test_a_node_pre_assert_reads_the_uri_of_a_file():
    output = _ok(_file_flow(
        "  b:\n    kind: code\n    input:\n      f: ${a.output}\n"
        "    code: |\n      return 'ok'\n    output: str\n"
        "    asserts:\n      - \"f.uri == 'f'\"\n"
    ))

    assert output == "ok"


def test_a_node_post_assert_reads_the_uri_of_a_file_in_its_output():
    output = _ok(_file_flow(
        "  b:\n    kind: code\n    output:\n      f: file\n      n: int\n"
        f"    code: |\n      return dict(f={F}, n=1)\n"
        "    asserts:\n      - \"output.f.uri == 'f'\"\n",
        "${b.output.n}",
    ))

    assert output == 1


def _missing_outcome(producer: str, path: str) -> tuple[str, Any]:
    """Run a node binding `${<path>}` after `producer`; return its status and output."""
    result = _run(
        f"id: p\nname: p\nnodes:\n{producer}"
        f"  b:\n    kind: code\n    input:\n      x: ${{{path}}}\n"
        "    code: |\n      return repr(inputs['x'])\n"
        "output: ${b.output}\n"
    )
    return result.status, result.output


_OBJECT_PRODUCER = """\
  a:
    kind: code
    output: object
    code: |
      return {'uri': 'f'}
"""


@pytest.mark.parametrize("field", ["other", "__class__"])
def test_another_field_of_a_file_is_missing_like_a_missing_key(field):
    expected = _missing_outcome(_OBJECT_PRODUCER, f"a.output.{field}")

    assert expected == ("succeeded", "None")
    assert _missing_outcome(_PRODUCERS, f"a.output.{field}") == expected


def test_render_as_json_writes_a_file_as_its_mapping():
    output = _ok(_file_flow(
        "  b:\n    kind: code\n    input:\n      j: ${render_as_json(a.output)}\n"
        "    code: |\n      return inputs['j']\n"
    ))

    assert output == '{\n  "uri": "f",\n  "mime": null,\n  "name": null\n}'


def test_a_traced_item_read_of_a_file_field_routes_to_its_source():
    item = ResolvedValue(REF, True, SourceRef("run", 1, ("xs", 0)))

    traced = eval_expr(
        parse_expr("item.uri"), lambda path: ResolvedValue(None, False), item=item, trace=True
    )

    assert traced.value == "f"
    assert traced.route.source == SourceRef("run", 1, ("xs", 0, "uri"))


def test_a_traced_item_read_of_another_file_field_has_no_source_route():
    item = ResolvedValue(REF, True, SourceRef("run", 1, ("xs", 0)))

    traced = eval_expr(
        parse_expr("item.other"), lambda path: ResolvedValue(None, False), item=item, trace=True
    )

    assert traced.value is None
    assert traced.route.source is None


def test_a_learnable_run_reads_the_fields_of_a_file_through_its_trace():
    # A learnable run resolves bindings through the run trace, not the pool.
    result = run_flow(load_flow(_file_flow(
        "  b:\n    kind: code\n    input:\n"
        "      u: ${a.output.uri}\n      k: ${n.output.name}\n      x: ${a.output.other}\n"
        "    code: |\n      return [inputs['u'], inputs['k'], inputs['x']]\n"
    )), {}, learnable=True)

    assert result.status == "succeeded", result.failure
    assert result.output == ["f", "a.txt", None]


@pytest.mark.parametrize(
    ("value", "name", "expected"),
    [
        ({"a": 1}, "a", (True, 1)),
        ({"a": None}, "a", (True, None)),
        ({"a": 1}, "b", (False, None)),
        (FileRef(uri="f", mime="m", name="n"), "uri", (True, "f")),
        (FileRef(uri="f", mime="m", name="n"), "mime", (True, "m")),
        (FileRef(uri="f", mime="m", name="n"), "name", (True, "n")),
        (FileRef(uri="f"), "name", (True, None)),
        (FileRef(uri="f"), "other", (False, None)),
        (FileRef(uri="f"), "__class__", (False, None)),
        (FileRef(uri="f"), "model_dump", (False, None)),
        ("text", "upper", (False, None)),
        ([1], "0", (False, None)),
        (None, "a", (False, None)),
    ],
)
def test_field_of_reads_a_dict_key_or_a_file_field(value, name, expected):
    assert field_of(value, name) == expected


class _OverridingDict(dict):
    def get(self, key, default=None):
        return "get"

    def __getitem__(self, key):
        return "getitem"

    def __contains__(self, key):
        return False


def test_field_of_reads_the_stored_value_of_a_dict_subclass_that_overrides_lookup():
    assert field_of(_OverridingDict(a=1), "a") == (True, 1)


def test_field_of_misses_on_a_defaultdict_without_adding_the_key():
    value = defaultdict(list)

    assert field_of(value, "a") == (False, None)
    assert "a" not in value


class _TaggedFileRef(FileRef):
    tag: str = "t"


def test_field_of_reads_only_the_file_fields_of_a_file_subclass():
    value = _TaggedFileRef(uri="f")

    assert field_of(value, "uri") == (True, "f")
    assert field_of(value, "tag") == (False, None)


# --- committed values and trace reads decode a file through the target's type --- #

def _ok_run(text: str, inputs: dict | None = None, **kwargs: Any) -> Any:
    """Run a dedented flow and return its output, asserting it succeeded."""
    result = run_flow(load_flow(textwrap.dedent(text)), inputs or {}, **kwargs)
    assert result.status == "succeeded", result.failure
    return result.output


_READER_FLOW = f"""\
    id: p
    name: p
    nodes:
      a:
        kind: code
        output: list(file)
        code: |
          return [{F}, {F}]
      r:
        kind: code
        output:
          f: file
          n: int
        code: |
          return dict(f={F}, n=1)
      b:
        kind: code
        input:
          xs: ${{a.output}}
          f: ${{r.output.f}}
          u: ${{r.output.f.uri}}
        code: |
          return [[type(x).__name__ for x in inputs['xs']], type(inputs['f']).__name__,
                  inputs['u']]
    output: ${{b.output}}
    """


@pytest.mark.parametrize("learnable", [False, True])
def test_a_reader_sees_the_files_of_a_typed_output(learnable):
    output = _ok_run(_READER_FLOW, learnable=learnable)

    assert output == [["FileRef", "FileRef"], "FileRef", "f"]


@pytest.mark.parametrize("learnable", [False, True])
def test_the_success_event_holds_the_files_of_a_typed_output(learnable):
    events: list[Any] = []
    _ok_run(_READER_FLOW, learnable=learnable, on_event=events.append)

    (output,) = [
        event.output for event in events
        if type(event).__name__ == "NodeSucceeded" and event.node_id == "a"
    ]
    assert output == [REF, REF]
    assert all(type(item) is FileRef for item in output)


@pytest.mark.parametrize("learnable", [False, True])
def test_an_untyped_output_holding_files_is_read_as_mappings(learnable):
    output = _ok_run(f"""\
        id: p
        name: p
        nodes:
          a:
            kind: code
            code: |
              return [{F}]
          b:
            kind: code
            input:
              xs: ${{a.output}}
            code: |
              return [type(x).__name__ for x in inputs['xs']]
        output: ${{b.output}}
        """, learnable=learnable)

    assert output == ["dict"]


@pytest.mark.parametrize("learnable", [False, True])
def test_a_map_body_reads_its_file_input_as_a_file(learnable):
    output = _ok_run(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              f: !file
            nodes:
              c:
                kind: code
                input:
                  x: ${{input.f}}
                code: |
                  return type(inputs['x']).__name__
            output: ${{c.output}}
        nodes:
          a:
            kind: code
            output: list(file)
            code: |
              return [{F}, {F}]
          m:
            kind: map
            over: ${{a.output}}
            call: body
            input:
              f: ${{item}}
        output: ${{m.output}}
        """, learnable=learnable)

    assert output == ["FileRef", "FileRef"]


@pytest.mark.parametrize("learnable", [False, True])
def test_a_call_body_reads_its_list_of_files_input_as_files(learnable):
    output = _ok_run(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              fs: !list(file)
            nodes:
              c:
                kind: code
                input:
                  fs: ${{input.fs}}
                code: |
                  return [type(x).__name__ for x in inputs['fs']]
            output: ${{c.output}}
        nodes:
          a:
            kind: code
            output: list(file)
            code: |
              return [{F}]
          k:
            kind: call
            call: body
            input:
              fs: ${{a.output}}
        output: ${{k.output}}
        """, learnable=learnable)

    assert output == ["FileRef"]


@pytest.mark.parametrize("learnable", [False, True])
def test_the_parent_reads_a_file_a_call_body_returns(learnable):
    output = _ok_run(f"""\
        id: p
        name: p
        defs:
          body:
            nodes:
              c:
                kind: code
                output:
                  f: file
                code: |
                  return dict(f={F})
            output: ${{c.output}}
        nodes:
          k:
            kind: call
            call: body
          d:
            kind: code
            input:
              x: ${{k.output.f}}
            code: |
              return type(inputs['x']).__name__
        output: ${{d.output}}
        """, learnable=learnable)

    assert output == "FileRef"


@pytest.mark.parametrize("learnable", [False, True])
def test_the_parent_reads_the_typed_file_of_a_partial_record_body_output(learnable):
    output = _ok_run(f"""\
        id: p
        name: p
        defs:
          body:
            nodes:
              c:
                kind: code
                code: |
                  return dict(f={F}, g={F})
            output:
              f: !file ${{c.output.f}}
              g: ${{c.output.g}}
        nodes:
          k:
            kind: call
            call: body
          d:
            kind: code
            input:
              f: ${{k.output.f}}
              g: ${{k.output.g}}
            code: |
              return [type(inputs['f']).__name__, inputs['g']]
        output: ${{d.output}}
        """, learnable=learnable)

    assert output == ["FileRef", FORM]


_LOOP_FLOW = f"""\
    id: p
    name: p
    defs:
      body:
        input:
          f: !file
          n: !int
        nodes:
          c:
            kind: code
            input:
              f: ${{input.f}}
              n: ${{input.n}}
            output:
              f: file
              n: int
            code: |
              return dict(f=inputs['f'], n=inputs['n'] + 1)
        output: ${{c.output}}
    nodes:
      a:
        kind: code
        output: file
        code: |
          return {F}
      l:
        kind: loop
        times: 2
        call: body
        input:
          f: ${{a.output}}
          n: 0
      d:
        kind: code
        input:
          x: ${{l.output.f}}
          n: ${{l.output.n}}
        code: |
          return [type(inputs['x']).__name__, inputs['n']]
    output: ${{d.output}}
    """


@pytest.mark.parametrize("learnable", [False, True])
def test_the_parent_reads_a_file_a_loop_carries(learnable):
    assert _ok_run(_LOOP_FLOW, learnable=learnable) == ["FileRef", 2]


def _partial_file_loop() -> LoopNode:
    node = LoopNode(
        "lp", flow_id="f", child=_child_flow(), predicate_kind="times", times=1, max_iters=1,
        returned_fields=("n",),
        child_inputs=[
            Parameter("f", contract=ValueContract(FILE), required=True),
            Parameter("n", contract=ValueContract(Type.scalar(ValueKind.INTEGER)), required=True),
        ],
    )
    node.params = [Parameter("f"), Parameter("n")]
    return node


def test_a_partial_loop_keeps_an_omitted_file_as_a_file():
    clone = _partial_file_loop().respawn(1, {"f": dict(FORM), "n": 1})

    by_name = {param.name: param for param in clone.params}
    assert by_name["f"].default == REF
    assert type(by_name["f"].default) is FileRef
    assert by_name["n"].has_default is False


_BODY_OUTPUTS = {
    "whole": "${c.output}",
    "typed": "\n                g: !file ${c.output.g}\n                n: !int ${c.output.n}",
}


@pytest.mark.parametrize("body_output", sorted(_BODY_OUTPUTS))
@pytest.mark.parametrize("learnable", [False, True])
def test_a_partial_loop_compares_an_omitted_file_with_a_returned_one(learnable, body_output):
    # The body returns `g` and `n` but not `f`, so `f` is carried from the seed.
    output = _ok_run(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              f: !file
              n: !int
            nodes:
              c:
                kind: code
                input:
                  f: ${{input.f}}
                  n: ${{input.n}}
                output:
                  g: file
                  n: int
                code: |
                  return dict(g=inputs['f'], n=inputs['n'] + 1)
            output: {_BODY_OUTPUTS[body_output]}
        nodes:
          a:
            kind: code
            output: file
            code: |
              return {F}
          l:
            kind: loop
            until: f == g
            max: 3
            call: body
            input:
              f: ${{a.output}}
              g: !file ${{a.output}}
              n: 0
        output: ${{l.output.n}}
        """, learnable=learnable)

    assert output == 1


@pytest.mark.parametrize("body_output", sorted(_BODY_OUTPUTS))
@pytest.mark.parametrize("learnable", [False, True])
def test_a_partial_loop_compares_a_carried_file_the_body_does_not_take(learnable, body_output):
    # The untyped `h` makes the loop's contract a partial record. The body takes only `n`,
    # so `f` is carried from the seed with no body input to read it through.
    output = _ok_run(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              n: !int
            nodes:
              c:
                kind: code
                input:
                  n: ${{input.n}}
                output:
                  g: file
                  n: int
                code: |
                  return dict(g={F}, n=inputs['n'] + 1)
            output: {_BODY_OUTPUTS[body_output]}
        nodes:
          a:
            kind: code
            output: file
            code: |
              return {F}
          u:
            kind: code
            code: |
              return {F}
          l:
            kind: loop
            until: f == g
            max: 3
            call: body
            input:
              f: ${{a.output}}
              g: !file ${{a.output}}
              h: ${{u.output}}
              n: 0
        output: ${{l.output.n}}
        """, learnable=learnable)

    assert output == 1



@pytest.mark.parametrize("learnable", [False, True])
def test_a_loop_reads_a_file_in_a_carried_object_as_a_mapping_on_every_iteration(learnable):
    # The seed binds `f` from `r`'s live output, where `f.x` is a file; from the second
    # iteration `f` comes back through the body's `object` output. Both must read alike.
    output = _ok_run(f"""\
        id: p
        name: p
        defs:
          body:
            input:
              f: !object
              n: !int
            nodes:
              c:
                kind: code
                input:
                  n: ${{input.n}}
                output: int
                code: |
                  return inputs['n'] + 1
            output:
              f: ${{input.f}}
              n: ${{c.output}}
        nodes:
          r:
            kind: code
            output:
              x: file
            code: |
              return dict(x={F})
          l:
            kind: loop
            while: isinstance(f.x, object) and n < 2
            max: 3
            call: body
            input:
              f: ${{r.output}}
              n: 0
        output: ${{l.output.n}}
        """, learnable=learnable)

    assert output == 2

_GATED_READER_FLOW = f"""\
    id: p
    name: p
    nodes:
      a:
        kind: code
        output: list(file)
        code: |
          return [{F}, {F}]
      ask:
        kind: human_input
        prompt: "Go?"
      b:
        kind: code
        input:
          xs: ${{a.output}}
          go: ${{ask.output}}
        code: |
          return [type(x).__name__ for x in inputs['xs']]
    output: ${{b.output}}
    """


@pytest.mark.parametrize("flow", [_READER_FLOW, _LOOP_FLOW])
def test_resuming_a_finished_run_with_files_replays_its_output(tmp_path: Path, flow):
    path = tmp_path / "run.jsonl"
    first = run_flow(load_flow(textwrap.dedent(flow)), {}, log=path)
    assert first.status == "succeeded", first.failure

    events: list[Any] = []
    resumed = resume_flow(load_flow(textwrap.dedent(flow)), log=path, on_event=events.append)

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output
    assert [type(event).__name__ for event in events if "Node" in type(event).__name__] == []


@pytest.mark.parametrize("learnable", [False, True])
def test_a_reader_after_resume_sees_the_files_of_a_replayed_output(tmp_path: Path, learnable):
    path = tmp_path / "run.jsonl"
    flow = textwrap.dedent(_GATED_READER_FLOW)

    def interrupt(**request):
        raise _HostStop("stop before the answer")

    with pytest.raises(_HostStop):
        run_flow(load_flow(flow), {}, log=path, tools={"ask_user": interrupt},
                 learnable=learnable)
    resumed = resume_flow(load_flow(flow), log=path, tools={"ask_user": lambda **request: "y"})

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == ["FileRef", "FileRef"]


@pytest.mark.parametrize("learnable", [False, True])
def test_resuming_an_unfinished_run_returns_the_files_of_a_typed_output(tmp_path: Path, learnable):
    # `a` is replayed from the log; the run then finishes live, so the result reads
    # through the contract the replayed state carries.
    path = tmp_path / "run.jsonl"
    flow = textwrap.dedent(_GATED_READER_FLOW).replace(
        "output: ${b.output}\n", "output: ${a.output}\n")

    def interrupt(**request):
        raise _HostStop("stop before the answer")

    with pytest.raises(_HostStop):
        run_flow(load_flow(flow), {}, log=path, tools={"ask_user": interrupt},
                 learnable=learnable)
    events: list[Any] = []
    resumed = resume_flow(load_flow(flow), log=path, on_event=events.append,
                          tools={"ask_user": lambda **request: "y"})

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == [REF, REF]
    assert all(type(item) is FileRef for item in resumed.output)
    (succeeded,) = [event for event in events if type(event).__name__ == "RunSucceeded"]
    assert succeeded.output == resumed.output
    assert all(type(item) is FileRef for item in succeeded.output)


@pytest.mark.parametrize("learnable", [False, True])
def test_a_flow_input_file_reaches_code_as_a_file(learnable):
    output = _ok_run("""\
        id: p
        name: p
        input:
          f: !file
          fs: !list(file)
        nodes:
          c:
            kind: code
            input:
              f: ${input.f}
              fs: ${input.fs}
            code: |
              return [type(inputs['f']).__name__, [type(x).__name__ for x in inputs['fs']]]
        output: ${c.output}
        """, {"f": REF, "fs": [REF]}, learnable=learnable)

    assert output == ["FileRef", ["FileRef"]]


def test_backward_context_holds_the_files_of_a_typed_output_as_json():
    # Feedback shapes and source lookups stay on the recorded JSON value.
    flow = copy.deepcopy(load_flow(textwrap.dedent(_READER_FLOW)).compiled)
    engine = _engine(flow, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    (occurrence,) = [item for item in trace.occurrences if item.node_id == "a"]

    assert trace.backward_context(occurrence.source).output == [FORM, FORM]
    assert trace.lookup_source(occurrence.source) == [FORM, FORM]
    assert trace.current_writers["a"].value == [REF, REF]


def test_a_source_entry_with_a_record_reading_type_is_hashable():
    files = Type(kind=ValueKind.OBJECT, fields={"f": FILE}, required=frozenset())
    entry = SourceEntry(None, 0, b"null", files)

    assert hash(entry) == hash(SourceEntry(None, 0, b"null"))
    assert entry != SourceEntry(None, 0, b"null")


def test_a_file_default_is_fingerprinted_apart_from_its_mapping():
    assert _canonical_value(REF, _ManifestContext(), "default") != _canonical_value(
        dict(FORM), _ManifestContext(), "default"
    )


# --- the run result reads its files through the flow's output type --- #

_TWO_FILES = f"""\
    id: p
    name: p
    nodes:
      a:
        kind: code
        output: list(file)
        code: |
          return [{F}, {F}]
      b:
        kind: code
        code: |
          return 1
    output: {{output}}
    """

_RESULT_OUTPUTS = {
    "whole": ("${a.output}", lambda output: output),
    "tagged": ("!list(file) ${a.output}", lambda output: output),
    "record": ("\n      a: ${a.output}\n      n: ${b.output}", lambda output: output["a"]),
}


def _two_files_flow(output: str) -> str:
    return _TWO_FILES.replace("{output}", output)


@pytest.mark.parametrize("shape", sorted(_RESULT_OUTPUTS))
@pytest.mark.parametrize("learnable", [False, True])
def test_the_run_result_holds_the_files_of_a_typed_output(learnable, shape):
    output, files_of = _RESULT_OUTPUTS[shape]
    events: list[Any] = []
    result = run_flow(load_flow(textwrap.dedent(_two_files_flow(output))), {},
                      learnable=learnable, on_event=events.append)

    assert result.status == "succeeded", result.failure
    assert files_of(result.output) == [REF, REF]
    assert all(type(item) is FileRef for item in files_of(result.output))
    (succeeded,) = [event for event in events if type(event).__name__ == "RunSucceeded"]
    assert succeeded.output == result.events[-1].output == result.output
    assert all(type(item) is FileRef for item in files_of(succeeded.output))


@pytest.mark.parametrize("shape", sorted(_RESULT_OUTPUTS))
@pytest.mark.parametrize("learnable", [False, True])
def test_resuming_a_finished_run_returns_the_files_of_a_typed_output(tmp_path: Path, learnable, shape):
    path = tmp_path / "run.jsonl"
    flow = textwrap.dedent(_two_files_flow(_RESULT_OUTPUTS[shape][0]))
    first = run_flow(load_flow(flow), {}, log=path, learnable=learnable)
    assert first.status == "succeeded", first.failure

    resumed = resume_flow(load_flow(flow), log=path)

    assert resumed.output == first.output
    files = _RESULT_OUTPUTS[shape][1](resumed.output)
    assert all(type(item) is FileRef for item in files)


@pytest.mark.parametrize("learnable", [False, True])
def test_the_run_result_holds_the_files_of_an_untyped_output_as_mappings(learnable):
    result = run_flow(load_flow(textwrap.dedent(f"""\
        id: p
        name: p
        nodes:
          a:
            kind: code
            code: |
              return [{F}]
        output: ${{a.output}}
        """)), {}, learnable=learnable)

    assert result.status == "succeeded", result.failure
    assert result.output == [FORM]
    assert type(result.output[0]) is dict
