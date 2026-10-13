"""AGENT settings that a run can't use fail at load, at the authored line."""

from __future__ import annotations

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError


def _flow(
    node_lines: str = "", *, output: str = "str", flow_lines: str = "", typedefs: str = ""
) -> str:
    return f"""\
id: f
name: f
{flow_lines}nodes:
  a:
    kind: agent
    prompt: Go.
    output: {output}
{node_lines}output: ${{a.output}}
{typedefs}"""


# Typedefs go after the nodes so the `output:` line number stays the same in every case.
_UNION_WITH_FILE = """\
typedefs:
  Report:
    kind: Literal[report]
    doc: file
  Note:
    kind: Literal[note]
    text: str
  Reply: Union[Report, Note]
"""

_UNION = """\
typedefs:
  Buy:
    action: Literal[buy]
    qty: int
  Hold:
    action: Literal[hold]
    note: str
  Decision: Union[Buy, Hold]
"""


@pytest.mark.parametrize("value", ["0", "-2", "ten", "true", "1.5"])
def test_node_env_max_tool_iterations_must_be_positive_or_minus_one(value: str) -> None:
    text = _flow(f"    env: {{max_tool_iterations: {value}}}\n")
    with pytest.raises(LoadError) as info:
        load_flow(text)
    message = str(info.value)
    assert "node 'a'" in message
    assert "`env.max_tool_iterations` must be a positive int or -1 (no cap)" in message
    assert info.value.diagnostic.frames[-1].range.start_line == 8


@pytest.mark.parametrize("value", ["0", "ten", "false"])
def test_flow_env_max_tool_iterations_must_be_positive_or_minus_one(value: str) -> None:
    text = _flow(flow_lines=f"env:\n  max_tool_iterations: {value}\n")
    with pytest.raises(LoadError) as info:
        load_flow(text)
    message = str(info.value)
    assert "flow `env.max_tool_iterations` must be a positive int or -1 (no cap)" in message
    assert info.value.diagnostic.frames[-1].range.start_line == 4


@pytest.mark.parametrize("value", ["1", "300", "-1"])
def test_valid_max_tool_iterations_loads(value: str) -> None:
    flow = load_flow(
        _flow(
            f"    env: {{max_tool_iterations: {value}}}\n",
            flow_lines=f"env: {{max_tool_iterations: {value}}}\n",
        )
    )
    assert flow.compiled.nodes["a"].env == {"max_tool_iterations": int(value)}


def test_max_tool_iterations_is_ignored_on_other_node_kinds() -> None:
    text = """\
id: f
name: f
nodes:
  c:
    kind: code
    code: return 1
    output: int
    env: {max_tool_iterations: 0}
output: ${c.output}
"""
    assert load_flow(text).compiled.nodes["c"].env == {"max_tool_iterations": 0}


@pytest.mark.parametrize(
    ("output", "typedefs"),
    [
        ("file", ""),
        ("list[file]", ""),
        ("None", ""),
        ("Optional[file]", ""),
        ("{doc: file}", ""),
        pytest.param("Reply", _UNION_WITH_FILE, id="union-with-file"),
    ],
)
def test_agent_output_without_a_structured_form_fails_at_load(output: str, typedefs: str) -> None:
    with pytest.raises(LoadError) as info:
        load_flow(_flow(output=output, typedefs=typedefs))
    message = str(info.value)
    assert "node 'a'" in message
    assert "an AGENT can't produce this `output:`" in message
    assert "Literal, record, union, and list values" in message
    assert info.value.diagnostic.frames[-1].range.start_line == 7


@pytest.mark.parametrize(
    ("output", "typedefs"),
    [
        ("str", ""),
        ("int", ""),
        ("Literal['a', 'b']", ""),
        ("list[int]", ""),
        ("{x: int, y: str}", ""),
        ("Optional[int]", ""),
        pytest.param("Decision", _UNION, id="union"),
    ],
)
def test_agent_output_with_a_structured_form_loads(output: str, typedefs: str) -> None:
    load_flow(_flow(output=output, typedefs=typedefs))
