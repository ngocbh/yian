"""Tags outside typed slots: every tag a document carries must type a value.

A tag types a flow input, a binding, a flow output, an agent variable, or a value
inside one. A tag anywhere else (on `kind:`, a node's `output:` declaration, a
typedef, a mapping key, ...) would be silently ignored, so it is a located load
error. Tags under top-level `x-*` keys are exempt unless used in a slot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import _load_flow, _LoadCtx, load_flow
from yian.compose.source import SourceDocument


def _generic(key: str) -> str:
    """Return the message for a tag at `key:`, which is not a typed slot."""
    return (
        "a tag types a value — a flow input, a binding, a flow output, an agent "
        f"variable, a value entry in `nodes:`, or a value inside one; `{key}:` is not one"
    )


_TYPEDEF = "typedef 'Buy': a type definition is written as a type, without a tag"


def _load_error(text: str, **kwargs) -> LoadError:
    """Load `text`, expecting a load error, and return it."""
    with pytest.raises(LoadError) as info:
        load_flow(text, **kwargs)
    return info.value


def _error_line(error: LoadError) -> int:
    """Return the 1-based line of the error's last diagnostic frame."""
    frames = error.diagnostic.frames
    assert frames, error.diagnostic.message
    return frames[-1].range.start_line


def _line_of(text: str, line: str) -> int:
    """Return the 1-based line number of the line equal to `line` in `text`."""
    return text.splitlines().index(line) + 1


def _assert_error(text: str, message: str, line: str, **kwargs) -> None:
    """Assert `text` fails to load with `message` located at `line`."""
    error = _load_error(text, **kwargs)
    assert str(error.diagnostic.message).endswith(message), error.diagnostic.message
    assert _error_line(error) == _line_of(text, line)


_CODE = """\
id: f
name: f
nodes:
  n:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      qty: 1
    output: int
output: ${n.output}
"""


# --- typed slots ------------------------------------------------------------- #


_COMPACT = """\
id: f
name: f
kind: code
code: tests.engine._compose_codefns:echo
input:
  qty: !int
output: int
"""

_SINGLE_NODE_DEF = """\
id: f
name: f
defs:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      qty: !int
    output: int
nodes:
  n:
    kind: call
    call: echo
    input:
      qty: 1
output: ${n.output}
"""

_AGENT_VARIABLE = """\
id: f
name: f
nodes:
  writer:
    kind: agent
    variables:
      tone: !str concise
      extra: !object {depth: !int 2}
    prompt: "${tone}"
    output: str
output: ${writer.output}
"""


@pytest.mark.parametrize(
    "text",
    [
        _CODE.replace("name: f\n", "name: f\ninput:\n  n: !int 3\n  m: !str\n"),
        _CODE.replace("      qty: 1", "      qty: !int 1"),
        _CODE.replace("      qty: 1", "      qty: [1, !int 2]"),
        _CODE.replace("output: ${n.output}", "output: !int ${n.output}"),
        _CODE.replace("output: ${n.output}", "output:\n  a: !int ${n.output}"),
        _COMPACT,
        _SINGLE_NODE_DEF,
        _AGENT_VARIABLE,
    ],
    ids=[
        "flow-input",
        "binding",
        "nested-binding",
        "flow-output",
        "flow-output-field",
        "compact-input",
        "single-node-def-input",
        "agent-variable",
    ],
)
def test_tags_in_typed_slots_load(text):
    load_flow(text)


# --- positions that are not typed slots ------------------------------------ #


def test_tag_on_kind_is_rejected_at_its_line():
    text = _CODE.replace("kind: code", "kind: !str code")
    _assert_error(text, _generic("kind"), "    kind: !str code")


def test_tag_on_prompt_is_rejected_at_its_line():
    text = """\
id: f
name: f
nodes:
  writer:
    kind: agent
    prompt: !str write
    output: str
output: ${writer.output}
"""
    _assert_error(text, _generic("prompt"), "    prompt: !str write")


def test_tag_on_nodes_is_rejected_at_its_line():
    text = _CODE.replace("nodes:", "nodes: !Nodes")
    _assert_error(text, _generic("nodes"), "nodes: !Nodes")


def test_tag_on_a_case_branch_target_is_rejected_at_its_line():
    text = """\
id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: ${input.score}
    cases:
      - when: high
        then: !str hot
    else: cold
  hot:
    kind: code
    code: 'return "hot"'
    output: str
  cold:
    kind: code
    code: 'return "cold"'
    output: str
output: ${hot.output | cold.output}
"""
    _assert_error(text, _generic("then"), "        then: !str hot")


def test_tag_on_map_over_is_rejected_at_its_line():
    text = """\
id: f
name: f
nodes:
  each:
    kind: map
    over: !list(int) [1, 2]
    input:
      x: ${item}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          value: ${input.x}
        output: int
    output: ${echo.output}
output: ${each.output}
"""
    _assert_error(text, _generic("over"), "    over: !list(int) [1, 2]")


def test_tag_on_wait_until_is_rejected_at_its_line():
    text = """\
id: f
name: f
input:
  deadline: !datetime
nodes:
  settle:
    kind: wait
    until: !datetime ${input.deadline}
output: ${settle.output}
"""
    _assert_error(text, _generic("until"), "    until: !datetime ${input.deadline}")


def test_tag_inside_a_non_slot_value_names_its_key():
    line = "llm_config: {provider: openai, openai: {temperature: !float 0.5}}"
    text = _CODE.replace("name: f\n", f"name: f\n{line}\n")
    _assert_error(text, _generic("temperature"), line)


def test_tagged_mapping_key_gets_its_own_message():
    text = _CODE.replace("      qty: 1", "      !int qty: 1")
    _assert_error(
        text,
        "`!int qty:` tags a mapping key; a tag types a value, so put it after the "
        "colon: `qty: !int <value>`",
        "      !int qty: 1",
    )


_CASE_ON = """\
id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: ${input.score}
    cases:
      - when: high
        then: one
    else: two
  one:
    kind: code
    code: |
      return 1
    output: int
  two:
    kind: code
    code: |
      return 2
    output: int
output: ${route.output}
"""


def test_tagged_key_message_spells_the_key_as_written():
    text = _CASE_ON.replace("    on: ${input.score}", "    !str on: ${input.score}")
    _assert_error(
        text,
        "`!str on:` tags a mapping key; a tag types a value, so put it after the "
        "colon: `on: !str <value>`",
        "    !str on: ${input.score}",
    )


def test_generic_message_spells_the_key_as_written():
    text = _CASE_ON.replace("    on: ${input.score}", "    on: !str ${input.score}")
    _assert_error(text, _generic("on"), "    on: !str ${input.score}")


def test_a_null_key_is_reported_before_a_tag_under_it():
    text = """\
id: f
name: f
nodes:
  settle:
    kind: wait
    until:
      null: !int 3
output: ${settle.output}
"""
    _assert_error(
        text,
        "the key `null` was read by YAML as null, but names in a flow must be text. "
        'Quote it ("null") or choose another name.',
        "      null: !int 3",
    )


# --- type-definition positions ---------------------------------------------- #


def test_tagged_typedef_is_rejected_at_its_line():
    text = _CODE.replace(
        "name: f\n", "name: f\ntypedefs:\n  Buy: !Rec\n    qty: int\n"
    )
    _assert_error(text, _TYPEDEF, "  Buy: !Rec")


def test_tagged_typedef_field_is_rejected_at_its_line():
    text = _CODE.replace(
        "name: f\n", "name: f\ntypedefs:\n  Buy:\n    qty: !int int\n"
    )
    _assert_error(text, _TYPEDEF, "    qty: !int int")


def test_tagged_typedef_field_without_a_value_is_rejected_at_its_line():
    text = _CODE.replace("name: f\n", "name: f\ntypedefs:\n  Buy:\n    qty: !int\n")
    _assert_error(text, _TYPEDEF, "    qty: !int")


def test_tagged_node_output_declaration_is_rejected_at_its_line():
    text = _CODE.replace("    output: int", "    output: !int")
    _assert_error(
        text,
        "`output:` declares a type; write `output: int`, without a tag",
        "    output: !int",
    )


def test_tagged_node_output_with_a_type_after_the_tag_spells_the_type():
    text = _CODE.replace("    output: int", "    output: !Optional(int) Optional(int)")
    _assert_error(
        text,
        "`output:` declares a type; write `output: Optional(int)`, without a tag",
        "    output: !Optional(int) Optional(int)",
    )


def test_tagged_node_output_field_is_rejected_at_its_line():
    text = _CODE.replace("    output: int", "    output:\n      qty: !int")
    _assert_error(
        text,
        "`output:` declares a type; write `qty: int`, without a tag",
        "      qty: !int",
    )


def test_tagged_list_item_in_a_node_output_says_the_item_is_tagged():
    text = _CODE.replace("    output: int", "    output:\n      qty:\n        - !int")
    _assert_error(
        text,
        "`output:` declares a type; an item of `qty:` is tagged `!int`; write the "
        "type without a tag",
        "        - !int",
    )


def test_tagged_compact_output_is_rejected_at_its_line():
    text = """\
id: f
name: f
kind: code
code: tests.engine._compose_codefns:echo
input:
  qty: !int
output: !int
"""
    _assert_error(
        text,
        "`output:` declares a type; write `output: int`, without a tag",
        "output: !int",
    )


def test_tagged_case_output_is_rejected_at_its_line():
    text = """\
id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: ${input.score}
    cases:
      - when: high
        then: one
    else: one
    output: !int
  one:
    kind: code
    code: |
      return 1
    output: int
output: ${route.output}
"""
    _assert_error(
        text,
        "case node 'route': a `case` routes and has no `output:`; tag its result where "
        "you read it, e.g. `x: !int ${route.output}`",
        "    output: !int",
    )


@pytest.mark.parametrize(
    "header",
    [
        "    kind: agent\n    prompt: write\n",
        "    kind: tool\n    tool_id: place_order\n",
    ],
    ids=["agent", "tool"],
)
def test_tagged_agent_and_tool_output_is_rejected_at_its_line(header):
    text = (
        "id: f\nname: f\nnodes:\n  n:\n"
        + header
        + "    output: !str\noutput: ${n.output}\n"
    )
    _assert_error(
        text,
        "`output:` declares a type; write `output: str`, without a tag",
        "    output: !str",
    )


def test_tagged_single_node_def_output_is_rejected_at_its_line():
    text = """\
id: f
name: f
defs:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      qty: !int
    output: !int
nodes:
  n:
    kind: call
    call: echo
    input:
      qty: 1
output: ${n.output}
"""
    _assert_error(
        text,
        "`output:` declares a type; write `output: int`, without a tag",
        "    output: !int",
    )


# --- defs, inline bodies, and uses: children --------------------------------- #


_DEF_FLOW = """\
id: f
name: f
defs:
  sub:
    input:
      x: !int
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          value: !int ${input.x}
        output: int
    output: ${echo.output}
nodes:
  n:
    kind: call
    call: sub
    input:
      x: 1
output: ${n.output}
"""


def test_tags_in_a_def_slot_load():
    load_flow(_DEF_FLOW)


def test_tag_outside_a_slot_in_a_def_is_rejected_at_its_line():
    text = _DEF_FLOW.replace("        kind: code", "        kind: !str code")
    _assert_error(text, _generic("kind"), "        kind: !str code")


_INLINE_FLOW = """\
id: f
name: f
nodes:
  each:
    kind: map
    over: [1, 2]
    input:
      x: ${item}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          value: !int ${input.x}
        output: int
    output: !int ${echo.output}
output: ${each.output}
"""


def test_tags_in_an_inline_body_slot_load():
    load_flow(_INLINE_FLOW)


def test_tag_outside_a_slot_in_an_inline_body_is_rejected_at_its_line():
    text = _INLINE_FLOW.replace("        kind: code", "        kind: !str code")
    _assert_error(text, _generic("kind"), "        kind: !str code")


def _write_child(tmp_path: Path, child: str) -> str:
    """Write a `uses:` child and return a parent flow that calls it."""
    (tmp_path / "child.yaml").write_text(child)
    return """\
id: parent
name: parent
uses:
  child: child
nodes:
  n:
    kind: call
    call: child
    input:
      x: 1
output: ${n.output}
"""


_CHILD = """\
id: child
name: child
input:
  x: !int
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      value: !int ${input.x}
    output: int
output: ${echo.output}
"""


def test_tags_in_a_uses_child_slot_load(tmp_path):
    load_flow(_write_child(tmp_path, _CHILD), search_paths=[tmp_path])


def test_tag_outside_a_slot_in_a_uses_child_is_rejected(tmp_path):
    parent = _write_child(tmp_path, _CHILD.replace("kind: code", "kind: !str code"))
    error = _load_error(parent, search_paths=[tmp_path])
    assert _generic("kind") in str(error)


# --- x-* keys and anchors ---------------------------------------------------- #


def test_tag_under_an_extension_key_used_through_an_alias_in_a_slot_loads():
    text = """\
id: f
name: f
x-defaults: &d !int 1
nodes:
  n:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      qty: *d
    output: int
output: ${n.output}
"""
    loaded = load_flow(text)
    assert loaded.compiled.wiring["n"]["qty"].contract is not None


def test_nested_tag_under_an_extension_key_used_through_a_merge_loads():
    text = """\
id: f
name: f
x-defaults: &d
  qty: !int 1
nodes:
  n:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      <<: *d
    output: int
output: ${n.output}
"""
    load_flow(text)


def test_unused_tag_under_an_extension_key_loads():
    load_flow(_CODE.replace("name: f\n", "name: f\nx-notes:\n  a: !Anything 1\n"))


def test_tag_under_an_extension_key_aliased_outside_a_slot_is_rejected():
    text = _CODE.replace("name: f\n", "name: f\nx-kind: &k !str code\n").replace(
        "    kind: code", "    kind: *k"
    )
    error = _load_error(text)
    assert str(error.diagnostic.message).endswith(_generic("kind"))


# --- per-load state ----------------------------------------------------------- #


def test_loading_the_same_document_twice_behaves_the_same():
    good = SourceDocument.parse(_DEF_FLOW)
    first = _load_flow(good, None, None, _LoadCtx())
    second = _load_flow(good, None, None, _LoadCtx())
    assert first.compiled.wiring == second.compiled.wiring

    bad = SourceDocument.parse(_DEF_FLOW.replace("    nodes:", "    nodes: !N"))
    messages = []
    for _ in range(2):
        with pytest.raises(LoadError) as info:
            _load_flow(bad, None, None, _LoadCtx())
        messages.append(str(info.value))
    assert messages[0] == messages[1]
    assert _generic("nodes") in messages[0]


# --- bindings ------------------------------------------------------------ #


def test_tag_on_a_binding_value_loads_and_is_consumed():
    loaded = load_flow(_CODE.replace("      qty: 1", "      qty: !int 1"))
    assert loaded.compiled.wiring["n"]["qty"].source == 1


def test_message_names_the_typed_slots():
    with pytest.raises(LoadError, match=re.escape(_generic("kind"))):
        load_flow(_CODE.replace("kind: code", "kind: !str code"))
