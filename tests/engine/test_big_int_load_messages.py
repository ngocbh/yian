"""Load errors that name an int of any size show it as a placeholder, never crash."""

from __future__ import annotations

import re
import sys

import pytest

import yian.json_codec as json_codec
from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.llm_clients import LLMConfig
from yian.nodes.agent import AgentNode
from yian.run import run_flow


BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
DIGITS = json_codec.int_text(BIG)
D = "1" * 5000  # an int literal past the default limit
ONES = (10**5000 - 1) // 9  # the int spelled by D
PLACEHOLDER = "<int too large to display>"


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


def assert_load_message(text: str, *, placeholder: bool = True) -> str:
    """Load a flow that must fail and check its message shows no long int.

    Args:
        text (`str`):
            The flow YAML.
        placeholder (`bool`, *optional*, defaults to `True`):
            Whether the message must name the value with `PLACEHOLDER`.

    Returns:
        `str`: The message, for further checks.
    """
    with pytest.raises(LoadError) as info:
        load_flow(text)
    msg = str(info.value)
    assert "set_int_max_str_digits" not in msg
    assert re.search(r"[0-9]{600}", msg) is None
    if placeholder:
        assert PLACEHOLDER in msg
    return msg


def _flow(nodes: str, *, head: str = "", output: str = "${c.output}") -> str:
    return f"id: p\nname: p\n{head}nodes:\n{nodes}output: {output}\n"


def _code(binding: str) -> str:
    return (
        f"  c:\n    kind: code\n    input:\n      n: {binding}\n"
        "    code: |\n      return inputs['n']\n"
    )


def _agent(extra: str) -> str:
    return _flow(f"  c:\n    kind: agent\n    prompt: hi\n{extra}")


def _loop(head: str) -> str:
    return _flow(
        f"  c:\n    kind: loop\n{head}    input:\n      n: 1\n    nodes:\n"
        "      b:\n        kind: code\n        input:\n          n: ${input.n}\n"
        "        code: |\n          return {'n': 1}\n    output: ${b.output}\n"
    )


def _case(cases: str) -> str:
    return _flow(
        f"  g:\n    kind: case\n    on: ${{input.x}}\n    cases: {cases}\n",
        head="input:\n  x: !int\n",
        output="${input.x}",
    )


# --- node fields -------------------------------------------------------------

_NODE_FIELD_ROWS = {
    "retries": _agent(f"    retries: -{D}\n"),
    "agent_env_max_tool_iterations": _agent(f"    env: {{max_tool_iterations: -{D}}}\n"),
    "flow_env_max_tool_iterations": _flow(
        _code("1"), head=f"env: {{max_tool_iterations: -{D}}}\n"
    ),
    "llm_config_inherit": _agent(f"    llm_config: {{inherit: {D}}}\n"),
    "depends_on": _agent(f"    depends_on: {D}\n"),
    "runs_after": _agent(f"    runs_after: {D}\n"),
    "loop_while": _loop(f"    while: {D}\n"),
    "loop_until": _loop(f"    until: {D}\n"),
    "loop_times": _loop(f"    times: -{D}\n"),
    "loop_max": _loop(f"    max: -{D}\n    while: ${{true}}\n"),
    "case_cases": _case(D),
    "case_cases_item": _case(f"[{D}]"),
    "human_input_questions": _flow(
        f"  c:\n    kind: human_input\n    prompt: hi\n    questions: {D}\n"
    ),
    "call_target": _flow(f"  c:\n    kind: call\n    call: {D}\n"),
    "kind": _flow(f"  c:\n    kind: {D}\n"),
    "node_output": _flow(f"  c:\n    kind: code\n    output: {D}\n    code: |\n      return 1\n"),
    "node_output_item": _flow(
        f"  c:\n    kind: code\n    output: [{D}]\n    code: |\n      return 1\n"
    ),
    "flow_input_untagged": _flow(_code("${input.n}"), head=f"input:\n  n: {D}\n"),
}


@pytest.mark.parametrize("text", _NODE_FIELD_ROWS.values(), ids=_NODE_FIELD_ROWS.keys())
def test_a_node_field_message_shows_a_long_int_as_a_placeholder(text):
    assert_load_message(text)


# --- validate ----------------------------------------------------------------


def _child(slot: str) -> str:
    return (
        f"defs:\n  child:\n    input:\n      n: {slot}\n    nodes:\n      c:\n"
        "        kind: code\n        input:\n          n: ${input.n}\n"
        "        code: |\n          return inputs['n']\n    output: ${c.output}\n"
    )


def _call_node(child_slot: str, value: str) -> str:
    return _child(child_slot) + _flow(
        f"  c:\n    kind: call\n    call: child\n    input:\n      n: {value}\n"
    )


def _case_on(on_type: str, whens: list[str]) -> str:
    body = "".join(f"      - when: {w}\n        then: big\n" for w in whens)
    return _flow(
        f"  g:\n    kind: case\n    on: ${{input.x}}\n    cases:\n{body}    else: other\n"
        "  big: !str big\n  other: !str other\n",
        head=f"input:\n  x: {on_type}\n",
        output="${g.output}",
    )


def _value_branch(then: str) -> str:
    return _flow(
        f"  g:\n    kind: case\n    on: ${{input.x}}\n    cases:\n      - when: 1\n"
        f"        then: {then}\n    else: other\n  other: !str other\n",
        head="input:\n  x: !int\n",
        output="${g.output}",
    )


_VALIDATE_ROWS = {
    "typed_binding_str": _flow(_code(f"!str {D}")),
    "value_entry_str": _flow(f"  v: !str {D}\n" + _code("${v.output}")),
    "inline_call_str": _child("!str") + _flow(_code(f"call(child, n={D})")),
    "inline_call_bool": _child("!bool") + _flow(_code(f"call(child, n={D})")),
    "inline_call_range": _child("!int(0..5)") + _flow(_code(f"call(child, n={D})")),
    "call_node_str": _call_node("!str", D),
    "call_node_bool": _call_node("!bool", D),
    "call_node_range": _call_node("!int(0..5)", D),
    "case_value_branch_list": _value_branch(f"[{D}]"),
    "case_value_branch_map": _value_branch(f"{{k: {D}}}"),
    "loop_seed_into_str": _flow(
        f"  c:\n    kind: loop\n    times: 1\n    input:\n      n: {D}\n    nodes:\n"
        "      bump:\n        kind: code\n        input:\n          n: ${input.n}\n"
        "        output:\n          n: str\n        code: |\n          return {'n': 'x'}\n"
        "    output: ${bump.output}\n"
    ),
    "int_range_reversed": _flow(_code("${input.x}"), head=f"input:\n  x: !int({D}..0)\n"),
    "case_duplicate_when": _case_on("!int", [D, D]),
    "case_quoted_when_vs_int": _case_on("!int", [f'"{D}"']),
    "typed_binding_quoted_int": _flow(_code(f'!int "{D}"')),
}


@pytest.mark.parametrize("text", _VALIDATE_ROWS.values(), ids=_VALIDATE_ROWS.keys())
def test_a_validate_message_shows_a_long_int_as_a_placeholder(text):
    assert_load_message(text)


def test_a_quoted_long_int_passed_to_an_int_child_fails_with_a_message_not_a_crash():
    assert_load_message(_child("!int") + _flow(_code(f'call(child, n="{D}")')))


def test_load_diagnostics_elide_a_digit_run_past_the_limit():
    error = LoadDiagnostics().error("x " + D + " y", notes=["n " + D])
    assert str(error) == "x " + PLACEHOLDER + " y"
    assert error.diagnostic.notes == ("n " + PLACEHOLDER,)


def test_load_diagnostics_keep_a_digit_run_within_the_limit():
    kept = "1" * 4300
    assert str(LoadDiagnostics().error("x " + kept + " y")) == "x " + kept + " y"


# --- run and memory ----------------------------------------------------------


def _episodic(max_episodes: str) -> str:
    return _agent(
        "    long_term_memory:\n"
        f"      e: {{type: episodic, config: {{max_episodes: {max_episodes}}}}}\n"
    )


def test_an_episodic_memory_limit_below_its_minimum_shows_a_placeholder():
    assert_load_message(_episodic(f"-{D}"))


def test_an_episodic_memory_limit_of_the_wrong_type_shows_a_placeholder():
    msg = assert_load_message(_episodic(f"[{D}]"), placeholder=False)
    assert "got <list too large to display>" in msg


def test_the_run_start_tool_iteration_check_shows_a_placeholder():
    node = AgentNode("n", prompt="hi", llm_config=LLMConfig())
    node.env = {"max_tool_iterations": -ONES}
    with pytest.raises(ValueError) as info:
        node._max_tool_iterations()
    msg = str(info.value)
    assert "set_int_max_str_digits" not in msg
    assert msg.endswith("got " + PLACEHOLDER)


def test_a_tool_binding_label_shows_a_long_int_as_a_placeholder():
    def place_order(qty: str) -> str:
        return qty

    flow = load_flow(
        "id: p\nname: p\nnodes:\n  place:\n    kind: tool\n    tool_id: place_order\n"
        f"    input:\n      qty: {D}\noutput: ${{place.output}}\n"
    )
    with pytest.raises(ValueError) as info:
        run_flow(flow, {}, tools={"place_order": place_order})
    msg = str(info.value)
    assert "set_int_max_str_digits" not in msg
    assert re.search(r"[0-9]{600}", msg) is None
    assert msg.endswith("but the binding is " + PLACEHOLDER + " (int)")



def test_an_unknown_long_int_tool_id_shows_a_placeholder():
    flow = load_flow(
        f"id: p\nname: p\nnodes:\n  t:\n    kind: tool\n    tool_id: {D}\noutput: ${{t.output}}\n"
    )
    result = run_flow(flow, {}, tools={"place_order": lambda qty: qty})
    msg = result.failure.message if result.failure is not None else ""
    assert result.status == "failed"
    assert f"unknown tool id {PLACEHOLDER}" in msg
    assert "set_int_max_str_digits" not in msg


def test_a_wait_until_long_int_fails_with_a_placeholder():
    result = run_flow(load_flow(_flow(f"  c:\n    kind: wait\n    until: {D}\n")), {})
    msg = result.failure.message
    assert "set_int_max_str_digits" not in msg
    assert re.search(r"[0-9]{600}", msg) is None
    assert f"wait `until` ({PLACEHOLDER}) did not resolve to a date/datetime: {PLACEHOLDER}" in msg


_MEMORY_KEYS = """id: root
name: root
defs:
  child:
    long_term_memory:
      inner: {type: inner, config: {limit: CHILD}}
    nodes:
      answer:
        kind: agent
        long_term_memory: {inner: {type: inner, config: {limit: AGENT}}}
        prompt: answer
        output: str
    output: ${answer.output}
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""


def _memory_keys(child: str, agent: str) -> str:
    return _MEMORY_KEYS.replace("CHILD", child).replace("AGENT", agent)


def test_one_memory_key_declared_twice_with_one_long_int_config_loads():
    load_flow(_memory_keys(D, D))


@pytest.mark.parametrize(
    "child, agent", [(D, "1"), ("1", D), (D, D + "1")], ids=["long_short", "short_long", "two_long"]
)
def test_one_memory_key_with_long_int_configs_that_differ_fails(child, agent):
    msg = assert_load_message(_memory_keys(child, agent), placeholder=False)
    assert "memory 'inner' is declared with different type or config" in msg
