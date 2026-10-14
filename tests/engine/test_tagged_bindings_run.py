"""Tagged bindings checked when a run binds a node's inputs or the flow's outputs.

A tag (`qty: !int ${size.output.qty}`) declares the value's type. The binder checks
the resolved value against it without converting or filling anything in, and a
mismatch fails with code `binding.type`, located on the tagged binding.
"""

from __future__ import annotations

import pytest

import yian.nodes.agent.node as agent_module
from yian.compose.loader import load_flow
from yian.nodes.base import Output
from yian.run import run_flow


def _consumer_flow(value: object, binding: str) -> str:
    """Return a flow whose producer `p` emits `value` into `c`'s `x: <binding>`."""
    return f"""
id: tagged_consumer
name: tagged_consumer
nodes:
  p:
    kind: code
    code: |
      return {value!r}
  c:
    kind: code
    input:
      x: {binding}
    code: |
      return inputs["x"]
output: ${{c.output}}
"""


def test_tagged_code_input_passes_a_matching_value():
    result = run_flow(load_flow(_consumer_flow(5, "!int ${p.output}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 5


def test_narrower_source_fails_on_its_tagged_line():
    text = _consumer_flow("5", "!int ${p.output}")
    loaded = load_flow(text)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "c"
    frame = loaded.diagnose(result.failure).frames[-1]
    line = text.splitlines().index("      x: !int ${p.output}") + 1
    assert frame.range.covered_lines(text) == [line]


def test_tagged_input_does_not_convert_an_int_declared_float():
    result = run_flow(load_flow(_consumer_flow(30, "!float ${p.output}")), {})

    assert result.status == "succeeded", result.failure
    assert type(result.output) is int


_BUY_FLOW = """
id: tagged_buy
name: tagged_buy
typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
nodes:
  d:
    kind: code
    output: object
    code: |
      return {"conviction": 0.5}
  use:
    kind: code
    input:
      decision: !Buy ${d.output}
    code: |
      return inputs["decision"]
output: ${use.output}
"""


def test_tag_does_not_fill_in_a_missing_field():
    result = run_flow(load_flow(_BUY_FLOW), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "use"
    assert "action" in result.failure.message


def test_tag_passes_a_record_with_every_field_unchanged():
    text = _BUY_FLOW.replace(
        'return {"conviction": 0.5}', 'return {"action": "buy", "conviction": 0.5}'
    )

    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"action": "buy", "conviction": 0.5}


def test_isinstance_still_reads_a_record_without_the_field_as_false():
    text = """
id: tagged_isinstance
name: tagged_isinstance
typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
nodes:
  d:
    kind: code
    output: object
    code: |
      return {"conviction": 0.5}
  check:
    kind: code
    input:
      is_buy: ${isinstance(d.output, Buy)}
    code: |
      return inputs["is_buy"]
output: ${check.output}
"""
    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure
    assert result.output is False


def _agent_flow(value: object) -> str:
    """Return a flow whose producer emits `value` into an AGENT's tagged input."""
    return f"""
id: tagged_agent
name: tagged_agent
nodes:
  src:
    kind: code
    code: |
      return {value!r}
  ask:
    kind: agent
    input:
      count: !int ${{src.output}}
    prompt: "Count ${{count}}"
    output: str
output: ${{ask.output}}
"""


@pytest.fixture
def _echo_agent(monkeypatch):
    """Answer every AGENT call with its rendered prompt instead of a model."""
    prompts: list[str] = []

    def run_agent(ctx):
        prompts.append(ctx.prompt)
        return Output(ctx.prompt)

    monkeypatch.setattr(agent_module, "run_agent", run_agent)
    return prompts


def test_tagged_agent_input_renders_a_matching_value(_echo_agent):
    result = run_flow(load_flow(_agent_flow(3)), {})

    assert result.status == "succeeded", result.failure
    assert result.output == "Count 3"


def test_tagged_agent_input_fails_before_the_model_call(_echo_agent):
    result = run_flow(load_flow(_agent_flow("three")), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "ask"
    assert _echo_agent == []


def _tool_flow(binding: str) -> str:
    """Return a flow whose TOOL node binds `qty` as `binding`."""
    return f"""
id: tagged_tool
name: tagged_tool
nodes:
  src:
    kind: code
    code: |
      return "5"
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: {binding}
output: ${{place.output}}
"""


def _place_order(qty):
    return f"placed {qty}"


def test_tagged_tool_input_passes_a_literal():
    result = run_flow(load_flow(_tool_flow("!int 5")), {}, tools={"place_order": _place_order})

    assert result.status == "succeeded", result.failure
    assert result.output == "placed 5"


def test_tagged_tool_input_fails_before_the_tool_runs():
    calls = []
    result = run_flow(
        load_flow(_tool_flow("!int ${src.output}")),
        {},
        tools={"place_order": lambda qty: calls.append(qty) or "placed"},
    )

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "place"
    assert calls == []


_CHILD = """
defs:
  child:
    input:
      x: !int
    nodes:
      twice:
        kind: code
        input:
          x: ${input.x}
        output: int
        code: |
          return inputs["x"] * 2
    output: ${twice.output}
"""


def _call_flow(returned: str) -> str:
    """Return a flow calling the strict `child` with a tagged arg over an Optional source."""
    return f"""
id: tagged_call
name: tagged_call
{_CHILD}nodes:
  maybe:
    kind: code
    output: Optional(int)
    code: |
      return {returned}
  run:
    kind: call
    call: child
    input:
      x: !int ${{maybe.output}}
output: ${{run.output}}
"""


def test_tagged_call_arg_narrows_a_nullable_source():
    result = run_flow(load_flow(_call_flow("3")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 6


def test_tagged_call_arg_rejects_a_null_at_runtime():
    result = run_flow(load_flow(_call_flow("None")), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "run"


def _map_flow(items: str) -> str:
    """Return a flow mapping an inline body over `items`, with `t: !int ${item}`."""
    return f"""
id: tagged_map
name: tagged_map
nodes:
  src:
    kind: code
    code: |
      return {items}
  m:
    kind: map
    over: ${{src.output}}
    input:
      t: !int ${{item}}
    nodes:
      twice:
        kind: code
        input:
          t: ${{input.t}}
        code: |
          return inputs["t"] * 2
    output: ${{twice.output}}
output: ${{m.output}}
"""


def test_tagged_map_item_arg_runs():
    result = run_flow(load_flow(_map_flow("[1, 2]")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == [2, 4]


def test_tagged_map_item_arg_rejects_a_mistyped_item():
    result = run_flow(load_flow(_map_flow('[1, "2"]')), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"


def _loop_flow(seed: str) -> str:
    """Return a flow whose inline LOOP seeds `n` as `seed`."""
    return f"""
id: tagged_loop
name: tagged_loop
nodes:
  src:
    kind: code
    code: |
      return "0"
  loop:
    kind: loop
    times: 2
    input:
      n: {seed}
    nodes:
      bump:
        kind: code
        input:
          n: ${{input.n}}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: ${{bump.output}}
output: ${{loop.output}}
"""


def test_tagged_literal_loop_seed_runs():
    result = run_flow(load_flow(_loop_flow("!int 0")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2}


def test_tagged_loop_seed_rejects_a_mistyped_value():
    result = run_flow(load_flow(_loop_flow("!int ${src.output}")), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "loop"


def _human_flow() -> str:
    """Return a flow whose HUMAN_INPUT gate binds a tagged context value."""
    return """
id: tagged_gate
name: tagged_gate
input:
  n: !str
nodes:
  src:
    kind: code
    input:
      n: ${input.n}
    code: |
      return int(inputs["n"]) if inputs["n"].isdigit() else inputs["n"]
  ask:
    kind: human_input
    input:
      count: !int ${src.output}
    prompt: "Proceed with ${count} items?"
output: ${ask.output}
"""


def test_tagged_human_input_context_passes_a_matching_value():
    asked = []
    result = run_flow(
        load_flow(_human_flow()),
        {"n": "3"},
        tools={"ask_user": lambda **request: asked.append(request) or "yes"},
    )

    assert result.status == "succeeded", result.failure
    assert asked[0]["question"] == "Proceed with 3 items?"


def test_tagged_human_input_context_fails_before_asking():
    asked = []
    result = run_flow(
        load_flow(_human_flow()),
        {"n": "three"},
        tools={"ask_user": lambda **request: asked.append(request) or "yes"},
    )

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "ask"
    assert asked == []


def _inline_flow(returned: str) -> str:
    """Return a flow whose inline CALL body takes `n: !int` from an untyped producer."""
    return f"""
id: tagged_inline
name: tagged_inline
nodes:
  a:
    kind: code
    code: |
      return {returned}
  wrap:
    kind: call
    input:
      n: !int ${{a.output}}
    nodes:
      twice:
        kind: code
        input:
          n: ${{input.n}}
        output: int
        code: |
          return inputs["n"] * 2
    output: ${{twice.output}}
output: ${{wrap.output}}
"""


def test_tagged_inline_body_arg_runs():
    result = run_flow(load_flow(_inline_flow("4")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 8


def test_tagged_inline_body_arg_rejects_a_mistyped_value():
    result = run_flow(load_flow(_inline_flow('"4"')), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "wrap"


def _output_flow(output: str, returned: str = '{"qty": 1}') -> str:
    """Return a flow whose untyped `size` node feeds `output:` written as `output`."""
    return f"""
id: tagged_output
name: tagged_output
typedefs:
  Size:
    qty: int
  T:
    a: int
nodes:
  size:
    kind: code
    code: |
      return {returned}
output: {output}
"""


def test_tagged_output_field_passes_a_matching_value():
    result = run_flow(load_flow(_output_flow("\n  qty: !int ${size.output.qty}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"qty": 1}


def test_tagged_output_field_rejects_a_mistyped_value():
    text = _output_flow("\n  qty: !int ${size.output.qty}", returned='{"qty": "1"}')

    result = run_flow(load_flow(text), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"


def test_tagged_single_output_returns_the_bare_value():
    result = run_flow(load_flow(_output_flow("!Size ${size.output}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"qty": 1}


def test_tagged_single_output_rejects_a_mistyped_value():
    text = _output_flow("!Size ${size.output}", returned='{"qty": "x"}')
    loaded = load_flow(text)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    frame = loaded.diagnose(result.failure).frames[-1]
    line = text.splitlines().index("output: !Size ${size.output}") + 1
    assert frame.range.covered_lines(text) == [line]


def test_tagged_mapping_output_returns_the_literal_as_one_value():
    result = run_flow(load_flow(_output_flow("!T {a: 1}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"a": 1}
