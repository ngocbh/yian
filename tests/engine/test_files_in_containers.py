"""Files inside typed containers: a `file` slot reads a file's JSON form, an `object`
slot reads a nested file as its mapping, and an untagged union whose members then
share a value is a load error."""

from pathlib import Path
import textwrap
from typing import Any

import pytest
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.expr.expressions import eval_expr
from yian.expr.grammar import parse_expr
from yian.nodes.binding import bind_params
from yian.nodes.contracts import Binding, Parameter, ValueContract
from yian.run import resume_flow, run_flow
from yian.typesys.pool import VariablePool
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import (
    FileRef,
    FileValue,
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
    normalize_files,
    types_overlap,
)

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


def test_normalize_files_picks_a_union_member_like_the_write_check():
    union = Type(kind=ValueKind.OBJECT, variants=(_t("Num"), _t("XFile")))

    assert normalize_files(union, {"x": dict(FORM)}) == {"x": REF}


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
                  return inputs['f']['uri']
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

    assert output == ["f", "f"]


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
