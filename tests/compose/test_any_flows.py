"""Flows that use `Any`: outputs, flow inputs, field reads, wiring, narrowing, and literals.

`Any` is any JSON value except null; `Optional(Any)` adds null. A node output typed `Any`
keeps the value it returns, field reads on it are not checked at load, and wiring it into
a narrower slot is a load error unless a tag or an `isinstance` branch narrows it.
"""

import datetime

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow
from yian.typesys.values import TypeCheckError


def _flow(
    output: str | None,
    rest: str = "",
    *,
    code: str = "None",
    out: str = "${mk.output}",
    head: str = "",
) -> str:
    """Return a flow whose CODE node `mk` returns `code`, typed `output`, then `rest`.

    Args:
        output (`str | None`): The `output:` type of `mk`, or `None` for an untyped node.
        rest (`str`, *optional*, defaults to `""`): More `nodes:` entries.
        code (`str`, *optional*, defaults to `"None"`): The Python expression `mk` returns.
        out (`str`, *optional*, defaults to `"${mk.output}"`): The flow's `output:`.
        head (`str`, *optional*, defaults to `""`): Top-level blocks before `input:`.
    """
    typed = "" if output is None else f"    output: {output}\n"
    return (
        "id: p\nname: p\n"
        + head
        + "input:\n  s: !str\n"
        "nodes:\n"
        "  mk:\n"
        "    kind: code\n"
        "    input:\n"
        "      s: ${input.s}\n"
        + typed
        + f"    code: |\n      return {code}\n"
        + rest
        + f"output: {out}\n"
    )


def _child(name: str, input_type: str) -> str:
    """Return a `defs:` entry `name` taking `x: !<input_type>` and returning its repr."""
    return (
        f"  {name}:\n"
        f"    input:\n      x: !{input_type}\n"
        "    nodes:\n"
        "      n:\n"
        "        kind: code\n"
        "        input:\n          x: ${input.x}\n"
        "        output: str\n"
        "        code: return repr(inputs['x'])\n"
        "    output: ${n.output}\n"
    )


def _defs(*children: str) -> str:
    """Return a `defs:` block holding `children`."""
    return "defs:\n" + "".join(children)


def _call(binding: str, *, node_id: str = "c", call: str = "child") -> str:
    """Return a CALL node `node_id` calling `call` with `x: <binding>`."""
    return (
        f"  {node_id}:\n    kind: call\n    call: {call}\n"
        f"    input:\n      x: {binding}\n"
    )


def _run(text: str, inputs: dict | None = None):
    """Load and run `text` with `inputs` (defaults to `{"s": "x"}`)."""
    return run_flow(load_flow(text), {"s": "x"} if inputs is None else inputs)


def _ok(text: str, inputs: dict | None = None):
    """Run `text` and return its output, asserting it succeeded."""
    result = _run(text, inputs)
    assert result.status == "succeeded", result.failure
    return result.output


def _load_error(text: str) -> LoadError:
    """Return the load error `text` raises."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


# --------------------------------------------------------------------------- #
# Node outputs
# --------------------------------------------------------------------------- #


def test_a_list_of_any_output_returns_mixed_items():
    assert _ok(_flow("list(Any)", code='[1, "two"]')) == [1, "two"]


@pytest.mark.parametrize(
    "code, value", [('"hello"', "hello"), ("3", 3), ('{"a": 1}', {"a": 1}), ("[1]", [1])]
)
def test_an_any_output_returns_any_json_value(code, value):
    output = _ok(_flow("Any", code=code))
    assert output == value
    assert type(output) is type(value)


def test_an_any_output_rejects_null():
    result = _run(_flow("Any", code="None"))
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"


def test_an_optional_any_output_returns_null():
    assert _ok(_flow("Optional(Any)", code="None")) is None


def test_an_object_output_rejects_a_date_member():
    result = _run(_flow("object", code='{"d": __import__("datetime").date(2026, 1, 1)}'))
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"
    assert "is not a JSON value" in result.failure.message


def test_an_object_output_rejects_a_non_text_key_at_the_output():
    result = _run(_flow("object", code='{1: "a"}'))
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "mk"
    assert result.failure.message == "object keys must be text, got 1"


# --------------------------------------------------------------------------- #
# Flow inputs
# --------------------------------------------------------------------------- #


def _input_flow(declared: str) -> str:
    """Return a flow taking `v: !<declared>` and returning it."""
    return (
        "id: p\nname: p\n"
        f"input:\n  v: !{declared}\n"
        "nodes:\n"
        "  n:\n    kind: code\n    input:\n      v: ${input.v}\n    code: return inputs['v']\n"
        "output: ${n.output}\n"
    )


def test_an_any_flow_input_rejects_null_naming_the_input():
    result = run_flow(load_flow(_input_flow("Any")), {"v": None})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.message == "input 'v': None does not match declared type Any"


def test_an_object_flow_input_rejects_a_non_text_key_naming_the_input():
    with pytest.raises(TypeCheckError) as caught:
        run_flow(load_flow(_input_flow("object")), {"v": {1: "a"}})
    assert str(caught.value) == "input 'v': object keys must be text, got 1"


def test_a_non_text_key_nested_in_a_flow_input_names_its_path():
    with pytest.raises(TypeCheckError) as caught:
        run_flow(load_flow(_input_flow("Any")), {"v": [{"a": {2: "b"}}]})
    assert str(caught.value) == "input 'v' at [0].a: object keys must be text, got 2"
    assert caught.value.path == (0, "a")


def test_a_date_inside_an_object_flow_input_arrives_as_text():
    output = _ok(_input_flow("object"), {"v": {"d": datetime.date(2026, 1, 1)}})
    assert output == {"d": "2026-01-01"}


# --------------------------------------------------------------------------- #
# Field reads on an `Any` output
# --------------------------------------------------------------------------- #


def test_a_field_read_on_an_any_output_loads_in_a_binding():
    rest = (
        "  use:\n    kind: code\n    input:\n      v: ${mk.output.a.b}\n"
        "    code: return inputs['v']\n"
    )
    assert _ok(_flow("Any", rest, code='{"a": {"b": 2}}', out="${use.output}")) == 2


def test_a_field_read_on_an_any_output_loads_in_a_flow_output():
    assert _ok(_flow("Any", code='{"a": 1}', out="\n  a: ${mk.output.a}")) == {"a": 1}


def test_a_field_read_on_an_any_output_loads_in_a_case_condition():
    rest = (
        "  route:\n    kind: case\n    cases:\n"
        "      - when: ${mk.output.a == 1}\n        then: hit\n"
        "    else: miss\n"
        "  hit:\n    kind: code\n    output: str\n    code: return 'hit'\n"
        "  miss:\n    kind: code\n    output: str\n    code: return 'miss'\n"
    )
    text = _flow("Any", rest, code='{"a": 1}', out="${hit.output | miss.output}")
    assert _ok(text) == "hit"


def test_a_field_read_on_an_any_output_loads_in_an_assert():
    text = _flow("Any", code='{"a": 1}').replace(
        "    code: |", "    asserts:\n      - output.a == 1\n    code: |"
    )
    assert _ok(text) == {"a": 1}


# --------------------------------------------------------------------------- #
# CALL wiring
# --------------------------------------------------------------------------- #


def test_a_tagged_any_binding_into_a_child_str_input_fails_at_runtime_for_an_int():
    text = _flow("Any", _call("!str ${mk.output}"), code="3", out="${c.output}",
                 head=_defs(_child("child", "str")))
    result = _run(text)
    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "c"


def test_a_tagged_any_binding_into_a_child_str_input_runs_for_text():
    text = _flow("Any", _call("!str ${mk.output}"), code='"hi"', out="${c.output}",
                 head=_defs(_child("child", "str")))
    assert _ok(text) == "'hi'"


def test_a_str_output_into_a_child_any_input_loads_and_runs():
    text = _flow("str", _call("${mk.output}"), code='"hi"', out="${c.output}",
                 head=_defs(_child("child", "Any")))
    assert _ok(text) == "'hi'"


def test_an_optional_str_output_into_a_child_any_input_is_a_load_error():
    text = _flow("Optional(str)", _call("${mk.output}"), out="${c.output}",
                 head=_defs(_child("child", "Any")))
    message = str(_load_error(text))
    assert "child expects 'Any', source is 'Optional(str)'" in message


# --------------------------------------------------------------------------- #
# Narrowing with isinstance
# --------------------------------------------------------------------------- #


def _narrowed(binding_node: str) -> str:
    """Return a flow routing on `isinstance(mk.output, str)` with `binding_node` as `go`."""
    rest = (
        "  route:\n    kind: case\n    cases:\n"
        "      - when: isinstance(mk.output, str)\n        then: go\n"
        "    else: other\n"
        + binding_node
        + "  other:\n    kind: code\n    output: str\n    code: return 'other'\n"
    )
    return _flow("Any", rest, code='"hi"', out="${go.output | other.output}",
                 head=_defs(_child("child", "str")))


def test_an_untagged_any_into_a_child_str_input_loads_inside_an_isinstance_branch():
    assert _ok(_narrowed(_call("${mk.output}", node_id="go"))) == "'hi'"


def test_the_same_untagged_wiring_outside_the_branch_is_a_load_error():
    text = _narrowed(_call("${mk.output}", node_id="go") + _call("${mk.output}", node_id="c2"))
    message = str(_load_error(text))
    assert "child expects 'str', source is 'Any'" in message
    assert "`x: !str ${...}`" in message
    assert "isinstance" in message


# --------------------------------------------------------------------------- #
# CASE `on:` an `Any` value
# --------------------------------------------------------------------------- #


def _on_any(when: str, code: str = "1") -> str:
    """Return a flow whose `route` matches `on: ${mk.output}` (`Any`) against `when`."""
    rest = (
        "  route:\n    kind: case\n    on: ${mk.output}\n    cases:\n"
        f"      - when: {when}\n        then: hit\n"
        "    else: miss\n"
        "  hit:\n    kind: code\n    output: str\n    code: return 'hit'\n"
        "  miss:\n    kind: code\n    output: str\n    code: return 'miss'\n"
    )
    return _flow("Any", rest, code=code, out="${hit.output | miss.output}")


def test_an_int_when_on_an_any_value_loads_and_matches():
    assert _ok(_on_any("1", code="1")) == "hit"
    assert _ok(_on_any("1", code='"1"')) == "miss"


def test_a_text_when_on_an_any_value_loads_and_matches():
    assert _ok(_on_any("a", code='"a"')) == "hit"


def test_an_unquoted_date_when_on_an_any_value_matches_its_text():
    assert _ok(_on_any("2026-01-02", code='"2026-01-02"')) == "hit"


def test_an_unquoted_datetime_when_on_an_any_value_keeps_its_quote_error():
    text = _on_any("2026-01-02T10:00:00Z")
    error = _load_error(text)
    assert "quote a datetime when:" in str(error)
    assert _error_line(error) == _line_of(text, "when: 2026-01-02T10")


# --------------------------------------------------------------------------- #
# MAP over lists of `Any`
# --------------------------------------------------------------------------- #


def _map(over_output: str | None, child_type: str, *, code: str, over: str = "${mk.output}"):
    """Return a flow mapping `child` (taking `x: !<child_type>`) over `over`."""
    rest = (
        "  m:\n    kind: map\n"
        f"    over: {over}\n"
        "    call: child\n    input:\n      x: ${item}\n"
    )
    return _flow(over_output, rest, code=code, out="${m.output}",
                 head=_defs(_child("child", child_type)))


def test_a_map_over_a_list_of_any_runs_items_into_a_child_any_input():
    assert _ok(_map("list(Any)", "Any", code='[1, "two"]')) == ["1", "'two'"]


def test_a_map_over_a_list_of_optional_any_into_a_child_optional_any_input_runs():
    assert _ok(_map("list(Optional(Any))", "Optional(Any)", code="[1, None]")) == ["1", "None"]


def test_a_map_over_a_list_of_optional_any_into_a_child_any_input_is_a_load_error():
    message = str(_load_error(_map("list(Optional(Any))", "Any", code="[1]")))
    assert "child expects 'Any', source is 'Optional(Any)'" in message


def test_a_map_over_a_mixed_literal_list_into_a_child_any_input_runs():
    assert _ok(_map(None, "Any", code="None", over="['a', 1]")) == ["'a'", "1"]


def test_a_map_over_a_mixed_literal_list_still_checks_each_element_at_load():
    message = str(_load_error(_map(None, "str", code="None", over="['a', 1]")))
    assert "child expects 'str', but the element 1 does not fit" in message


# --------------------------------------------------------------------------- #
# LOOP carried `Any`
# --------------------------------------------------------------------------- #


def _inline_loop(seed: str, body_output: str, code: str) -> str:
    """Return a flow with an inline loop carrying `acc`, seeded `seed`, from a body node."""
    return (
        "id: p\nname: p\n"
        "nodes:\n"
        "  loop:\n    kind: loop\n    times: 2\n"
        f"    input:\n      acc: {seed}\n"
        "    nodes:\n"
        "      step:\n        kind: code\n        input:\n          acc: ${input.acc}\n"
        f"        output: {body_output}\n"
        f"        code: |\n          return {code}\n"
        "    output:\n      acc: ${step.output}\n"
        "output: ${loop.output}\n"
    )


def test_a_loop_carrying_an_any_seed_with_an_any_body_field_runs():
    assert _ok(_inline_loop("!Any 0", "Any", "[inputs['acc']]"), {}) == {"acc": [[0]]}


def test_a_typed_loop_seed_accepts_an_optional_any_body_field():
    text = _inline_loop("!Any 0", "Optional(Any)", "inputs['acc'] + 1")
    assert _ok(text, {}) == {"acc": 2}


def test_a_null_from_an_optional_any_body_field_fails_the_next_turn():
    result = run_flow(load_flow(_inline_loop("!Any 0", "Optional(Any)", "None")), {})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.message == "input 'acc': None does not match declared type Any"


def test_an_untyped_seed_into_an_any_body_input_rejects_an_optional_any_body_field():
    text = _inline_loop("${mk.output}", "Optional(Any)", "inputs['acc']").replace(
        "nodes:\n",
        "nodes:\n  mk:\n    kind: code\n    output: Any\n    code: return 0\n",
        1,
    )
    message = str(_load_error(text))
    assert "the body returns 'acc' as Optional(Any)" in message
    assert "typed Any from its seed" in message


# --------------------------------------------------------------------------- #
# CASE output `Any` over branches of different types
# --------------------------------------------------------------------------- #


def _routed(case_output: str = "", reader: str = "!Any ${route.output}") -> str:
    """Return a flow whose `route` case picks the value entry `one` or `a`, read by `use`."""
    return (
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        f"  route:\n    kind: case\n{case_output}    cases:\n"
        "      - when: ${input.s == 'i'}\n        then: one\n"
        "    else: a\n"
        "  one: !int 1\n"
        "  a: !str \"a\"\n"
        f"  use:\n    kind: code\n    input:\n      v: {reader}\n"
        "    code: return inputs['v']\n"
        "output: ${use.output}\n"
    )


def test_a_reader_tagging_a_case_result_any_gets_the_taken_branch_value():
    text = _routed()
    assert _ok(text, {"s": "i"}) == 1
    assert _ok(text, {"s": "x"}) == "a"


def test_a_case_output_any_is_a_load_error_suggesting_the_reader_tag():
    message = str(_load_error(_routed("    output: Any\n", "${route.output}")))
    assert "a `case` routes and has no `output:`" in message
    assert "`x: !Any ${route.output}`" in message


# --------------------------------------------------------------------------- #
# Child flow output `!Any`
# --------------------------------------------------------------------------- #


_ANY_CHILD = (
    "  mk_any:\n"
    "    input:\n      x: !str\n"
    "    nodes:\n"
    "      n:\n        kind: code\n        input:\n          x: ${input.x}\n"
    "        code: return 3\n"
    "    output: !Any ${n.output}\n"
)


def _from_any_child(consumer_type: str) -> str:
    """Return a flow calling `mk_any`, then `child` (taking `consumer_type`) on its output."""
    rest = _call("${input.s}", node_id="g", call="mk_any") + _call("${g.output}")
    return _flow(None, rest, out="${c.output}",
                 head=_defs(_ANY_CHILD, _child("child", consumer_type)))


def test_a_child_any_output_into_a_parent_any_consumer_runs():
    assert _ok(_from_any_child("Any")) == "3"


def test_a_child_any_output_into_a_parent_int_input_is_a_load_error_with_the_hint():
    message = str(_load_error(_from_any_child("int")))
    assert "child expects 'int', source is 'Any'" in message
    assert "`x: !int ${...}`" in message


# --------------------------------------------------------------------------- #
# Typed slot literals
# --------------------------------------------------------------------------- #


def _literal(binding: str) -> str:
    """Return a flow whose CODE node `n` takes `x: <binding>` and returns it."""
    return (
        "id: p\nname: p\n"
        "nodes:\n"
        f"  n:\n    kind: code\n    input:\n      x: {binding}\n"
        "    code: return inputs['x']\n"
        "output: ${n.output}\n"
    )


@pytest.mark.parametrize(
    "binding, value",
    [("!Any 3", 3), ('!Any [1, "two"]', [1, "two"]), ("!Any {a: 1}", {"a": 1}),
     ("!Optional(Any) null", None)],
)
def test_a_typed_any_literal_loads_and_runs(binding, value):
    assert _ok(_literal(binding), {}) == value


def test_a_null_any_literal_is_a_load_error():
    text = _literal("!Any null")
    error = _load_error(text)
    assert str(error).endswith(
        "`null` is not a valid Any; to allow null, write `x: !Optional(Any) null`"
    )
    assert "quote" not in str(error)
    assert _error_line(error) == _line_of(text, "x: !Any null")


def test_an_unquoted_date_any_literal_gets_the_quote_hint():
    text = _literal("!Any 2024-01-01")
    error = _load_error(text)
    assert "quote" in str(error)
    assert _error_line(error) == _line_of(text, "x: !Any 2024-01-01")


# --------------------------------------------------------------------------- #
# Value entries
# --------------------------------------------------------------------------- #


def _entry_flow(entry: str, read: str) -> str:
    """Return a flow with value entry `entry` and a CODE node reading `read`."""
    return (
        "id: p\nname: p\n"
        "nodes:\n"
        f"  {entry}\n"
        f"  use:\n    kind: code\n    input:\n      v: {read}\n"
        "    code: return inputs['v']\n"
        "output: ${use.output}\n"
    )


def test_a_field_read_on_an_any_value_entry_is_not_checked():
    load_flow(_entry_flow("v: !Any {a: 1}", "${v.output.zz}"))


def test_a_field_read_on_an_object_value_entry_is_still_checked():
    message = str(_load_error(_entry_flow("w: !object {a: 1}", "${w.output.zz}")))
    assert "reads unknown field 'zz'" in message


# --------------------------------------------------------------------------- #
# Inline bodies
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "output, code, value",
    [("list(Any)", '[1, "two"]', [1, "two"]), ("list(Optional(Any))", "[1, None]", [1, None])],
)
def test_a_list_of_any_into_an_untagged_inline_body_parameter_runs(output, code, value):
    rest = (
        "  c:\n    kind: call\n    input:\n      xs: ${mk.output}\n"
        "    nodes:\n"
        "      inner:\n        kind: code\n        input:\n          xs: ${input.xs}\n"
        "        code: return inputs['xs']\n"
        "    output: ${inner.output}\n"
    )
    assert _ok(_flow(output, rest, code=code, out="${c.output}")) == value
