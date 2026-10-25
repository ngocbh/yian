"""A text or expression field YAML reads as non-text is a load error at that value."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.compose.nodes.base import not_text_detail, not_text_error

NON_TEXT = [
    ("5", "an integer"),
    ("0", "an integer"),
    ("true", "a boolean"),
    ("false", "a boolean"),
    ("1.5", "a float"),
    ("2026-10-14", "a date"),
    ("[a]", "a list"),
    ("[]", "a list"),
    ("{x: 1}", "a mapping"),
    ("{}", "a mapping"),
]


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


def test_not_text_error_names_the_field():
    assert not_text_error("prompt", 5) == (
        "`prompt:` must be text, but YAML read it as an integer; put it in quotes"
    )
    assert not_text_error("questions[0].question", [1]) == (
        "`questions[0].question:` must be text, but YAML read it as a list; put it in quotes"
    )
    assert not_text_error("questions[0].header", None) == (
        "`questions[0].header:` must be text, but it is empty; write the text"
    )


def test_not_text_error_takes_a_fix():
    assert not_text_error("kind", [1], fix="write one of a, b") == (
        "`kind:` must be text, but YAML read it as a list; write one of a, b"
    )
    assert not_text_error("kind", None, fix="write one of a, b") == (
        "`kind:` must be text, but it is empty; write one of a, b"
    )


def _code(v: str) -> str:
    """A flow whose CODE node `a` has `code: <v>`."""
    return """\
id: f
name: f
nodes:
  a:
    kind: code
    output: str
    code: @V
output: ${a.output}
""".replace("@V", v)


def _tool(v: str) -> str:
    """A flow whose TOOL node `a` has `tool_id: <v>`."""
    return """\
id: f
name: f
nodes:
  a:
    kind: tool
    tool_id: @V
    input: {symbol: x}
output: ${a.output}
""".replace("@V", v)


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_code_is_a_load_error(value, phrase):
    text = _code(value)
    error = _error(text)
    assert (
        f"node 'a' (kind=code): `code:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "code: ")


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_tool_id_is_a_load_error(value, phrase):
    text = _tool(value)
    error = _error(text)
    assert (
        f"node 'a' (kind=tool): `tool_id:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "tool_id: ")


@pytest.mark.parametrize(
    ("text", "kind", "key"),
    [(_code("null"), "code", "code"), (_tool("null"), "tool", "tool_id")],
    ids=["code", "tool_id"],
)
def test_null_code_and_tool_id_are_empty(text, kind, key):
    error = _error(text)
    assert (
        f"node 'a' (kind={kind}): `{key}:` must be text, but it is empty; write the text"
    ) in str(error)
    assert _start_line(error) == _line(text, f"{key}: ")


def test_tagged_non_text_tool_id():
    # `!str 5` still reads as an integer here; the tag's own error comes once it is quoted.
    error = _error(_tool("!str 5"))
    assert "`tool_id:` must be text, but YAML read it as an integer; put it in quotes" in str(
        error
    )


_CODE_IN_DEF = """\
id: f
name: f
defs:
  body:
    nodes:
      one:
        kind: code
        output: str
        code: "return 'x'"
      two:
        kind: code
        output: str
        code: 5
    output: ${two.output}
nodes:
  a:
    kind: call
    call: body
output: ${a.output}
"""

_CODE_COMPACT = """\
id: f
name: f
output: str
kind: code
code: 5
"""

_CODE_MERGED = """\
id: f
name: f
x-base: &base {code: 5}
nodes:
  a:
    <<: *base
    kind: code
    output: str
output: ${a.output}
"""

_CODE_INLINE_BODY = """\
id: f
name: f
nodes:
  a:
    kind: call
    nodes:
      echo:
        kind: code
        output: str
        code: 5
    output: ${echo.output}
output: ${a.output}
"""


@pytest.mark.parametrize(
    ("text", "node", "line"),
    [
        (_CODE_IN_DEF, "two", _line(_CODE_IN_DEF, "code: 5")),
        (_CODE_COMPACT, "f", _line(_CODE_COMPACT, "code: 5")),
        (_CODE_MERGED, "a", _line(_CODE_MERGED, "x-base:")),
        (_CODE_INLINE_BODY, "echo", _line(_CODE_INLINE_BODY, "code: 5")),
    ],
    ids=["def", "compact", "merge", "inline-body"],
)
def test_code_in_def_compact_merge_and_inline_body(text, node, line):
    error = _error(text)
    assert f"node {node!r} (kind=code): `code:` must be text" in str(error)
    assert _start_line(error) == line


@pytest.mark.parametrize("text", [_code("\"return 'x'\""), _tool("'5'")])
def test_text_code_and_tool_id_still_load(text):
    load_flow(text)


_CALL_HEADS = {
    "call": "",
    "map": "    over: ${input.xs}\n",
    "loop": "    times: 2\n",
}


def _call(kind: str, v: str) -> str:
    """A flow whose `kind` node `a` has `call: <v>`, beside a `defs:` entry `body`."""
    return (
        """\
id: f
name: f
input:
  xs: !list(int)
defs:
  body:
    input:
      v: !int
    nodes:
      one:
        kind: code
        output: {v: int}
        code: "return {'v': 1}"
    output: ${one.output}
nodes:
  a:
    kind: @KIND
@HEAD    call: @V
    input: {v: 1}
output: ${a.output}
"""
        .replace("@KIND", kind)
        .replace("@HEAD", _CALL_HEADS[kind])
        .replace("@V", v)
    )


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
@pytest.mark.parametrize("kind", list(_CALL_HEADS))
def test_non_text_call_is_a_load_error(kind, value, phrase):
    text = _call(kind, value)
    error = _error(text)
    assert (
        f"node 'a' (kind={kind}): `call:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "call: ")


@pytest.mark.parametrize("kind", list(_CALL_HEADS))
def test_null_call_is_empty(kind):
    text = _call(kind, "null")
    error = _error(text)
    assert (
        f"node 'a' (kind={kind}): `call:` must be text, but it is empty; write the text"
    ) in str(error)
    assert _start_line(error) == _line(text, "call: ")


@pytest.mark.parametrize("kind", list(_CALL_HEADS))
def test_text_call_still_loads(kind):
    load_flow(_call(kind, "body"))


def _case_on(v: str) -> str:
    """A flow whose CASE node `a` has `on: <v>`, routing to CODE nodes `b` and `c`."""
    return """\
id: f
name: f
input:
  proj: !str
nodes:
  a:
    kind: case
    on: @V
    cases:
      - when: x
        then: b
    else: c
  b:
    kind: code
    output: str
    code: "return 'b'"
  c:
    kind: code
    output: str
    code: "return 'c'"
output: ${b.output | c.output}
""".replace("@V", v)


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_case_on_is_a_load_error(value, phrase):
    text = _case_on(value)
    error = _error(text)
    assert (
        f"node 'a' (kind=case): `on:` must be text, but YAML read it as {phrase}; "
        "write a reference such as `on: ${input.x}`"
    ) in str(error)
    assert _start_line(error) == _line(text, "on: ")


@pytest.mark.parametrize("value", ["input.proj", "hello", "''"])
def test_case_on_text_without_a_reference_is_a_load_error(value):
    text = _case_on(value)
    error = _error(text)
    assert (
        "node 'a' (kind=case): `on:` must be a reference, but has no `${...}`; "
        "write a reference such as `on: ${input.x}`"
    ) in str(error)
    assert _start_line(error) == _line(text, "on: ")


def test_null_case_on_is_absent():
    # `on: null` is the searched form, where `when: x` names no input.
    assert "unresolved references" in str(_error(_case_on("null")))


def test_text_case_on_still_loads():
    load_flow(_case_on("${input.proj}"))


def _node_name(v: str) -> str:
    """A flow whose CODE node `a` has `node_name: <v>`."""
    return """\
id: f
name: f
nodes:
  a:
    kind: code
    node_name: @V
    output: str
    code: "return 'x'"
output: ${a.output}
""".replace("@V", v)


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_node_name_is_a_load_error(value, phrase):
    text = _node_name(value)
    error = _error(text)
    assert (
        f"node 'a' (kind=code): `node_name:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "node_name: ")


@pytest.mark.parametrize("value", ["null", "'5'"])
def test_null_and_text_node_name_still_load(value):
    load_flow(_node_name(value))


_TEXT_CASE_ON = _case_on("${input.proj}")
_SEARCHED_CASE = _case_on("null").replace("    on: null\n", "")


@pytest.mark.parametrize(
    ("text", "marker", "fragment"),
    [
        (
            _call("loop", "body").replace("    times: 2\n", "    times: 2\n    while: 5\n"),
            "while: 5",
            "must be an expression string over the carried record",
        ),
        (
            _call("loop", "body").replace("    times: 2\n", "    times: 2\n    until: 5\n"),
            "until: 5",
            "must be an expression string over the carried record",
        ),
        (
            _SEARCHED_CASE.replace("when: x", "when: 5"),
            "when: 5",
            "each searched case needs a string `when:` expression",
        ),
        (
            _TEXT_CASE_ON.replace("when: x", "when: [a]"),
            "when: [a]",
            "takes text, a number, or a bool",
        ),
        (
            _TEXT_CASE_ON.replace("then: b", "then: 5"),
            "then: 5",
            "a branch names a node or calls a flow",
        ),
        (
            _TEXT_CASE_ON.replace("else: c", "else: 5"),
            "else: 5",
            "a branch names a node or calls a flow",
        ),
        (
            _code("\"return 'x'\"").replace(
                "    output: str\n", '    output: str\n    asserts: ["true", 5]\n'
            ),
            "asserts: ",
            "`asserts:` must be a list of strings",
        ),
    ],
    ids=["while", "until", "searched-when", "on-when", "then", "else", "asserts"],
)
def test_already_located_fields_are_unchanged(text, marker, fragment):
    error = _error(text)
    assert fragment in str(error)
    assert _start_line(error) == _line(text, marker)


def _kind(v: str) -> str:
    """A flow whose `nodes:` entry `a` has `kind: <v>`."""
    return """\
id: f
name: f
nodes:
  a:
    kind: @V
    prompt: hi
output: ${a.output}
""".replace("@V", v)


def _compact_kind(v: str) -> str:
    """A compact (single-node) flow with `kind: <v>`."""
    return """\
id: f
name: f
output: int
kind: @V
code: return 1
""".replace("@V", v)


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_kind_is_a_load_error(value, phrase):
    text = _kind(value)
    error = _error(text)
    assert (
        f"node 'a': `kind:` must be text, but YAML read it as {phrase}; "
        "write one of agent, call, case, code, human_input, loop, map, tool, wait"
    ) in str(error)
    assert _start_line(error) == _line(text, "kind: ")


@pytest.mark.parametrize(
    ("value", "phrase"),
    [(value, f"YAML read it as {phrase}") for value, phrase in NON_TEXT]
    + [("null", "it is empty")],
)
def test_non_text_compact_kind_is_a_load_error(value, phrase):
    text = _compact_kind(value)
    error = _error(text)
    assert (
        f"compact (single-node) flow: `kind:` must be text, but {phrase}; "
        "write one of agent, code, human_input, tool"
    ) in str(error)
    assert _start_line(error) == _line(text, "kind: ")


def _compact_id(v: str | None) -> str:
    """A compact (single-node) flow with `id: <v>`, or no `id:` when `v` is None."""
    head = "" if v is None else f"id: {v}\n"
    return head + "name: f\noutput: int\nkind: code\ncode: return 1\n"


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_compact_id_is_a_load_error(value, phrase):
    text = _compact_id(value)
    error = _error(text)
    assert (
        f"compact (single-node) flow: `id:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "id: ")


@pytest.mark.parametrize("value", ["null", "''", None], ids=["null", "empty", "absent"])
def test_null_empty_and_absent_compact_id_are_required(value):
    error = _error(_compact_id(value))
    assert "compact (single-node) flow: a top-level `id:` is required" in str(error)


_KIND_IN_DEF = """\
id: f
name: f
defs:
  sub:
    kind: [a]
    output: str
    code: "return 'x'"
nodes:
  a:
    kind: call
    call: sub
output: ${a.output}
"""

_KIND_IN_INLINE_BODY = """\
id: f
name: f
nodes:
  a:
    kind: call
    nodes:
      echo:
        kind: [a]
        output: str
        code: "return 'x'"
    output: ${echo.output}
output: ${a.output}
"""


@pytest.mark.parametrize("text", [_KIND_IN_DEF, _KIND_IN_INLINE_BODY], ids=["def", "inline-body"])
def test_non_text_kind_in_a_def_and_inline_body(text):
    error = _error(text)
    assert "`kind:` must be text" in str(error)
    assert _start_line(error) == _line(text, "kind: [a]")


@pytest.mark.parametrize(
    ("value", "fragment"),
    [("null", "add `kind:` for a node"), ("llm", "unknown kind"), ("''", "unknown kind")],
)
def test_null_and_text_kinds_are_unchanged(value, fragment):
    assert fragment in str(_error(_kind(value)))


class _Model(BaseModel):
    name: str
    uses: dict[str, str] = {}
    stop: str | list[str] | None = None


def _detail(data: dict) -> tuple | None:
    with pytest.raises(ValidationError) as caught:
        _Model.model_validate(data)
    return not_text_detail(caught.value)


def test_not_text_detail_reads_the_first_string_error():
    assert _detail({"name": 5}) == ("name", 5)
    assert _detail({"name": "n", "uses": {"lib": [1]}}) == ("uses.lib", [1])
    assert _detail({"name": "n", "uses": {"str": 5}}) == ("uses.str", 5)
    assert _detail({"name": None}) == ("name", None)
    assert _detail({}) is None
    # A union field takes more than text, so its error is not a text-field error.
    assert _detail({"name": "n", "stop": 5}) is None
    assert _detail({"name": "n", "stop": [5]}) is None


def _flow_field(field: str, v: str) -> str:
    """A flow whose top-level `<field>:` is `<v>`."""
    text = """\
id: f
name: f
description: d
version: '1'
nodes:
  a:
    kind: code
    output: str
    code: "return 'x'"
output: ${a.output}
"""
    lines = [
        f"{field}: @V" if line.startswith(f"{field}: ") else line for line in text.splitlines()
    ]
    return "\n".join(lines).replace("@V", v) + "\n"


@pytest.mark.parametrize(
    ("value", "phrase"),
    [("5", "an integer"), ("true", "a boolean"), ("1.5", "a float"), ("[a]", "a list")],
)
@pytest.mark.parametrize("field", ["id", "name", "description", "version"])
def test_non_text_flow_field_is_a_load_error(field, value, phrase):
    text = _flow_field(field, value)
    error = _error(text)
    assert str(error).startswith(
        f"`{field}:` must be text, but YAML read it as {phrase}; put it in quotes"
    )
    assert "validation error for ComposeFile" not in str(error)
    assert _start_line(error) == _line(text, f"{field}: ")


@pytest.mark.parametrize("field", ["id", "name"])
def test_null_flow_field(field):
    error = _error(_flow_field(field, "null"))
    assert f"`{field}:` must be text, but it is empty; write the text" in str(error)


@pytest.mark.parametrize("field", ["description", "version"])
def test_null_optional_flow_field_loads(field):
    load_flow(_flow_field(field, "null"))


_FLOW = """\
id: f
name: f
nodes:
  a:
    kind: code
    output: str
    code: "return 'x'"
output: ${a.output}
"""


@pytest.mark.parametrize(
    ("value", "phrase"),
    [(value, f"YAML read it as {phrase}") for value, phrase in NON_TEXT]
    + [("null", "it is empty")],
)
def test_non_text_uses_ref_is_a_load_error(value, phrase):
    text = _FLOW + "uses:\n  lib: @V\n".replace("@V", value)
    error = _error(text)
    assert f"`uses.lib:` must be text, but {phrase}" in str(error)
    assert _start_line(error) == _line(text, "lib: ")


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_flow_assert_is_a_load_error(value, phrase):
    text = _FLOW + 'asserts:\n  - "true"\n  - @V\n'.replace("@V", value)
    error = _error(text)
    assert f"`asserts[1]:` must be text, but YAML read it as {phrase}" in str(error)
    assert _start_line(error) == len(text.splitlines())


def test_other_flow_schema_errors_are_unchanged():
    error = _error("id: f\nname: f\noutput: x\n")
    assert "validation error for ComposeFile" in str(error)
    assert "nodes" in str(error)


_LLM_ON_NODE = """\
id: f
name: f
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: @V
output: ${a.output}
"""

_LLM_ON_FLOW = """\
id: f
name: f
llm_config: @V
nodes:
  a:
    kind: agent
    prompt: hi
output: ${a.output}
"""

_LLM_ON_DEF = """\
id: f
name: f
defs:
  sub:
    llm_config: @V
    nodes:
      b:
        kind: agent
        prompt: hi
    output: ${b.output}
nodes:
  a:
    kind: call
    call: sub
output: ${a.output}
"""

_LLM_ON_GATE = """\
id: f
name: f
input:
  proj: !str
nodes:
  ask:
    kind: human_input
    input: {proj: '${input.proj}'}
    adaptive_questions:
      prompt: "Q ${proj}"
      llm_config: @V
output: ${ask.output}
"""

_LLM_SITES = [
    (_LLM_ON_NODE, "node 'a': llm_config: "),
    (_LLM_ON_FLOW, "flow: llm_config: "),
    (_LLM_ON_DEF, "defs entry 'sub': llm_config: "),
    (_LLM_ON_GATE, "adaptive_questions.llm_config: "),
]
_LLM_SITE_IDS = ["node", "flow", "def", "adaptive"]


@pytest.mark.parametrize(("builder", "prefix"), _LLM_SITES, ids=_LLM_SITE_IDS)
def test_non_text_llm_config_field_is_a_load_error(builder, prefix):
    text = builder.replace("@V", "{model: 5}")
    error = _error(text)
    assert (
        prefix + "`model:` must be text, but YAML read it as an integer; put it in quotes"
    ) in str(error)
    assert "validation error for LLMConfig" not in str(error)
    assert _start_line(error) == _line(text, "llm_config: ")


@pytest.mark.parametrize(("builder", "prefix"), _LLM_SITES, ids=_LLM_SITE_IDS)
def test_nested_llm_config_text_field(builder, prefix):
    error = _error(builder.replace("@V", "{provider: openai, openai: {user: 5}}"))
    assert prefix + "`openai.user:` must be text" in str(error)


@pytest.mark.parametrize(
    ("value", "count"),
    [
        ("{provider: 5}", "1 validation error"),
        ("{provider: azure, azure: {stop: 5}}", "2 validation errors"),
        ("{provider: azure, azure: {stop: [5]}}", "2 validation errors"),
    ],
)
@pytest.mark.parametrize(("builder", "prefix"), _LLM_SITES, ids=_LLM_SITE_IDS)
def test_other_llm_config_errors_are_unchanged(builder, prefix, value, count):
    message = str(_error(builder.replace("@V", value)))
    assert prefix + count + " for LLMConfig" in message


def _def_asserts(v: str) -> str:
    """A flow whose multi-node `defs:` entry `b` has `asserts: <v>`."""
    return """\
id: f
name: f
defs:
  b:
    nodes:
      c:
        kind: code
        output: int
        code: return 1
    output: ${c.output}
    asserts: @V
nodes:
  a:
    kind: call
    call: b
output: ${a.output}
""".replace("@V", v)


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_def_assert_is_a_load_error(value, phrase):
    text = _def_asserts('\n      - "true"\n      - ' + value)
    error = _error(text)
    assert (
        f"defs entry 'b': `asserts[1]:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "      - " + value)


@pytest.mark.parametrize("value", ["5", "false", "0", "''", "{}", "{x: 1}", "output > 0"])
def test_non_list_def_asserts_is_a_load_error(value):
    text = _def_asserts(value)
    error = _error(text)
    assert "defs entry 'b': `asserts:` must be a list of strings" in str(error)
    assert _start_line(error) == _line(text, "asserts: ")


def test_null_def_asserts_loads():
    load_flow(_def_asserts("null"))
