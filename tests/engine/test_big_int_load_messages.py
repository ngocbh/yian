"""Load errors that name an int of any size show it as a placeholder, never crash."""

from __future__ import annotations

import re
import sys

import pytest

import yian.json_codec as json_codec
from yian.compose import LoadError, load_flow


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
