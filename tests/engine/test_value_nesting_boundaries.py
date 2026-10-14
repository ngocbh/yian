"""A value nesting deeper than 100 levels fails where it enters a run, naming the path."""

import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import yian.llm_clients as llm_clients
import yian.nodes.agent.node as agent_module
from yian import load_flow, run_flow
from yian.memory_host import build_memories
from yian.nodes.base import Output
from yian.nodes.binding import BindingError, apply_defaults, bind_params
from yian.nodes.contracts import Parameter
from yian.run import resume_flow
from yian.typesys import TypeCheckError
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, ValueKind
from tests.engine.test_agent_nonfinite_replies import (
    _NATIVE,
    _Carrier,
    _correction,
    _flow as _tool_flow,
    _generate,
    _setup as _setup_tool,
    _structured_flow,
    _texts,
)
from tests.engine.test_agent_session import _view


def _code_flow(body: str, *, output_line: str = "", flow_output: str = "${deep.output}") -> str:
    """Return a flow whose `deep` CODE node runs the inline `body`."""
    indented = "\n".join(f"      {line}" for line in body.strip().splitlines())
    return f"""
id: nesting
name: nesting
nodes:
  deep:
    kind: code
{output_line}
    code: |
{indented}
output: {flow_output}
"""


def _tree_body(levels: int) -> str:
    """Return a CODE body returning `{"tree": v}` with `v` nested `levels` levels."""
    return f"""
v = "leaf"
for _ in range({levels}):
    v = {{"a": v}}
return {{"tree": v}}
"""


def _list_body(levels: int) -> str:
    """Return a CODE body returning a list chain `levels` levels deep."""
    return f"""
v = "leaf"
for _ in range({levels}):
    v = [v]
return v
"""


def _nest_list(levels: int, leaf: object = "leaf") -> object:
    value: object = leaf
    for _ in range(levels):
        value = [value]
    return value


def _nest_dict(levels: int) -> object:
    value: object = "leaf"
    for _ in range(levels):
        value = {"a": value}
    return value


def _assert_too_deep(result, node_id: str) -> None:
    """Assert the run failed `node_id`'s output contract on value depth."""
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == node_id
    assert "deeper than 100 levels" in result.failure.message
    assert "Circular" not in result.failure.message


_TREE_OUTPUT = "    output: {tree: object}"


def test_a_record_output_at_the_limit_succeeds():
    flow = load_flow(_code_flow(_tree_body(99), output_line=_TREE_OUTPUT))
    result = run_flow(flow, {})

    assert result.status == "succeeded"
    assert result.output == {"tree": _nest_dict(99)}


def test_a_record_output_over_the_limit_fails_at_the_field():
    flow = load_flow(_code_flow(_tree_body(100), output_line=_TREE_OUTPUT))
    result = run_flow(flow, {})

    _assert_too_deep(result, "deep")
    assert result.failure.message.startswith("output at tree.a")
    assert result.failure.trace[0].key == (
        "/s:root/s:node/s:deep/s:output/s:contract/s:path/s:tree"
    )


# Before the limit, 200 levels raised pydantic's "Invalid JSON: recursion limit exceeded" out
# of run_flow, and 1500 failed the inline CODE serializer with "Circular reference detected
# (depth exceeded)".
@pytest.mark.parametrize("levels", [200, 1500])
def test_a_far_too_deep_inline_output_fails_without_raising(levels):
    flow = load_flow(_code_flow(_tree_body(levels), output_line=_TREE_OUTPUT))
    result = run_flow(flow, {})

    _assert_too_deep(result, "deep")


def test_a_far_too_deep_reference_output_fails_on_depth():
    flow = load_flow("""
id: nesting
name: nesting
input:
  n: int
nodes:
  deep:
    kind: code
    input:
      n: ${input.n}
    code: tests.engine._compose_codefns:deep_tree
output: ${deep.output}
""")
    result = run_flow(flow, {"n": 1500})

    _assert_too_deep(result, "deep")
    assert "canonical data nesting" not in result.failure.message


def test_an_untyped_output_over_the_limit_fails():
    result = run_flow(load_flow(_code_flow(_list_body(101))), {})

    _assert_too_deep(result, "deep")


def test_an_untyped_output_at_the_limit_flows_out_whole():
    result = run_flow(load_flow(_code_flow(_list_body(100))), {})

    assert result.status == "succeeded"
    assert result.output == _nest_list(100)


def _map_flow(child_levels: int) -> str:
    """Return a flow whose `m` MAP collects children nested `child_levels` levels."""
    return f"""
id: nesting
name: nesting
nodes:
  m:
    kind: map
    over: "${{[1, 2]}}"
    input:
      y: int = ${{item}}
    nodes:
      work:
        kind: code
        input:
          y: ${{input.y}}
        code: |
          v = inputs["y"]
          for _ in range({child_levels}):
              v = [v]
          return v
    output: ${{work.output}}
output: ${{m.output}}
"""


def test_a_map_result_list_over_the_limit_fails_at_the_map_end():
    result = run_flow(load_flow(_map_flow(100)), {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "m/__end__"
    assert "output at [0]" in result.failure.message
    assert "100 levels" in result.failure.message


def test_a_map_result_list_at_the_limit_succeeds():
    result = run_flow(load_flow(_map_flow(99)), {})

    assert result.status == "succeeded"
    assert result.output == [_nest_list(99, 1), _nest_list(99, 2)]


def test_a_record_flow_output_around_a_value_at_the_limit_fails_at_end():
    flow = load_flow(_code_flow(_list_body(100), flow_output="\n  t: ${deep.output}"))
    result = run_flow(flow, {})

    _assert_too_deep(result, "__end__")


def test_a_run_input_record_may_carry_a_value_at_the_limit():
    flow = load_flow("""
id: nesting
name: nesting
input:
  x: object
nodes:
  echo:
    kind: code
    input:
      x: ${input.x}
    code: |
      return inputs["x"]
output: ${echo.output}
""")
    value = _nest_dict(100)
    result = run_flow(flow, {"x": value})

    assert result.status == "succeeded"
    assert result.output == value


# START's committed record may sit one level over the limit, so run_flow's input check is
# what keeps a run input record from going further.
@pytest.mark.parametrize("levels", [101, 1500])
def test_a_run_input_over_the_limit_is_rejected_before_the_run_starts(tmp_path, levels):
    flow = load_flow("""
id: nesting
name: nesting
input:
  x: object
nodes: {}
output: ${input.x}
""")
    log = tmp_path / "r.jsonl"

    with pytest.raises(ValueError) as caught:
        run_flow(flow, {"x": _nest_dict(levels)}, log=log)

    assert isinstance(caught.value, TypeCheckError)
    assert str(caught.value).startswith("input 'x' at a.a")
    assert "deeper than 100 levels" in str(caught.value)
    assert len(caught.value.path) == 100
    assert not log.exists()


def _assert_binding_too_deep(result, node_id: str, prefix: str) -> None:
    """Assert `node_id` failed `binding.evaluate` on value depth with a `prefix` message."""
    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == node_id
    assert result.failure.message.startswith(prefix)
    assert "deeper than 100 levels" in result.failure.message


_WRAPPED_BINDING_FLOW = """
id: nesting
name: nesting
input:
  x: object
nodes:
  c:
    kind: code
    input:
      y: ${[input.x]}
    code: |
      return 1
output: ${c.output}
"""


def test_a_binding_around_a_value_at_the_limit_fails_its_node():
    result = run_flow(load_flow(_WRAPPED_BINDING_FLOW), {"x": _nest_dict(100)})

    _assert_binding_too_deep(result, "c", "input 'y' at [0].a")
    assert result.failure.trace[0].key == "/s:root/s:node/s:c/s:input/s:y/s:binding"


def _call_flow(binding: str, declared: str) -> str:
    """Return a flow whose `k` CALL node binds the def input `y: declared` to `binding`."""
    return f"""
id: nesting
name: nesting
input:
  x: object
defs:
  body:
    input:
      y: {declared}
    nodes:
      inner:
        kind: code
        input:
          y: ${{input.y}}
        code: |
          return 1
    output: ${{inner.output}}
nodes:
  k:
    kind: call
    call: body
    input:
      y: {binding}
output: ${{k.output}}
"""


def test_a_call_bound_to_a_value_at_the_limit_succeeds():
    result = run_flow(load_flow(_call_flow("${input.x}", "object")), {"x": _nest_dict(100)})

    assert result.status == "succeeded"
    assert result.output == 1


def test_a_call_bound_over_the_limit_fails_the_call_not_its_start():
    result = run_flow(load_flow(_call_flow("${[input.x]}", "list[object]")), {"x": _nest_dict(100)})

    _assert_binding_too_deep(result, "k", "input 'y' at [0].a")


def test_a_map_over_list_around_a_value_at_the_limit_fails_the_map():
    flow = load_flow("""
id: nesting
name: nesting
input:
  x: object
nodes:
  m:
    kind: map
    over: ${[input.x]}
    input:
      y: object = ${item}
    nodes:
      work:
        kind: code
        code: |
          return 1
    output: ${work.output}
output: ${m.output}
""")
    result = run_flow(flow, {"x": _nest_dict(100)})

    _assert_binding_too_deep(result, "m", "MAP node 'm': `over`")
    assert result.failure.trace[0].key == "/s:root/s:node/s:m/s:reserved/s:over/s:binding"


def test_apply_defaults_rejects_a_default_over_the_limit():
    with pytest.raises(BindingError) as caught:
        apply_defaults([Parameter("p", has_default=True, default=_nest_dict(101))], {})

    assert caught.value.failure_code == "binding.evaluate"
    assert str(caught.value).startswith("default for 'p' at a.a")
    assert "deeper than 100 levels" in str(caught.value)


def test_bind_params_rejects_a_filled_default_over_the_limit():
    parameter = Parameter("p", has_default=True, default=_nest_dict(101))

    with pytest.raises(BindingError) as caught:
        bind_params([parameter], {}, VariablePool())

    assert caught.value.failure_code == "binding.evaluate"
    assert str(caught.value).startswith("default for 'p' at a.a")
    assert "deeper than 100 levels" in str(caught.value)


def test_backward_rejects_feedback_over_the_limit_and_stays_usable():
    flow = load_flow(_code_flow(_list_body(1)))
    result = run_flow(flow, {}, learnable=True)
    assert result.status == "succeeded"

    with pytest.raises(ValueError) as caught:
        result.backward(_nest_list(1500, "bad"))

    assert str(caught.value).startswith("feedback at [0][0]")
    assert "deeper than 100 levels" in str(caught.value)
    assert result.backward_outcome is None
    outcome = result.backward("good")
    assert outcome.status == "succeeded"


def test_replay_accepts_a_value_recorded_before_the_limit(monkeypatch, tmp_path):
    flow = load_flow(_code_flow(_list_body(120)))
    path = tmp_path / "r.jsonl"
    monkeypatch.setattr("yian.typesys.values.MAX_VALUE_NESTING", 500)
    recorded = run_flow(flow, {}, log=path)
    assert recorded.status == "succeeded"
    monkeypatch.undo()

    replayed = resume_flow(flow, log=path)

    assert replayed.status == "succeeded"
    assert replayed.failure is None
    assert replayed.output == recorded.output == _nest_list(120)


def _memory_flow(text: str):
    """Return `text` loaded with episodic long-term memory declared on the flow."""
    return load_flow(text.replace("name: nesting\n", "name: nesting\nlong_term_memory: episodic\n", 1))


def _backward_with_less_stack(result, feedback):
    """Run `result.backward(feedback)` with 300 fewer frames of recursion headroom.

    A caller may already sit deep in its own stack, so learning from a value at the
    limit must not need the remaining headroom of a shallow test frame.
    """
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(limit - 300)
    try:
        return result.backward(feedback)
    finally:
        sys.setrecursionlimit(limit)


def _assert_learns(flow, inputs, feedback) -> None:
    """Assert a memory run on `inputs` succeeds and learns from `feedback`."""
    result = run_flow(flow, inputs, memory=build_memories(flow), learnable=True)
    assert result.status == "succeeded"

    outcome = _backward_with_less_stack(result, feedback)

    assert outcome.status == "succeeded"
    assert outcome.learning.status == "succeeded"


def test_value_at_the_limit_learns_from_an_output():
    _assert_learns(_memory_flow(_code_flow(_list_body(100))), {}, "good")


def test_value_at_the_limit_learns_from_an_input():
    flow = _memory_flow("""
id: nesting
name: nesting
input:
  x: object
nodes:
  echo:
    kind: code
    input:
      x: ${input.x}
    code: |
      return inputs["x"]
output: ${echo.output}
""")
    _assert_learns(flow, {"x": _nest_dict(100)}, "good")


def test_value_at_the_limit_learns_from_a_record_output():
    flow = _memory_flow(_code_flow(
        _tree_body(99), output_line=_TREE_OUTPUT, flow_output="${deep.output.tree}",
    ))
    _assert_learns(flow, {}, {"a": "bad"})


# A model reply past the limit is checked alongside NaN and infinity: a structured value
# self-corrects under `retries:`, and tool-call arguments fail the node before any call.

_FREEFORM = Type(kind=ValueKind.OBJECT)


def _deep_text(levels: int) -> str:
    """Return a structured reply whose `value` nests `levels` levels, built without recursion."""
    return '{"value": ' + '{"a": ' * levels + '"leaf"' + "}" * levels + "}"


def _assert_reply_too_deep(message: str, where: str) -> None:
    """Assert `message` names `where`, a path into the value, and the limit."""
    assert f"{where} at " in message
    assert "deeper than 100 levels" in message
    assert "recursion" not in message


@pytest.mark.parametrize("levels", [101, 1500])
def test_a_fallback_structured_value_over_the_limit_self_corrects(levels):
    model = _texts(_deep_text(levels), '{"value": {"k": 1}}')

    assert _generate(model, _FREEFORM) == {"k": 1}
    assert len(model.calls) == 2
    _assert_reply_too_deep(_correction(model), "structured output")


def test_an_exhausted_fallback_structured_value_over_the_limit_is_located():
    with pytest.raises(TypeCheckError) as caught:
        _generate(_texts(_deep_text(101)), _FREEFORM, max_retries=0)
    assert caught.value.path == ("a",) * 100


@pytest.mark.parametrize("levels", [101, 1500])
def test_a_native_structured_value_over_the_limit_self_corrects(levels):
    model = _Carrier([_nest_dict(levels), {"k": 1}])

    assert _generate(model, _FREEFORM, config=_NATIVE) == {"k": 1}
    assert len(model.calls) == 2
    _assert_reply_too_deep(_correction(model), "structured output")


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("session", [True, False])
def test_a_structured_agent_over_the_limit_self_corrects_under_retries(monkeypatch, native, session):
    model = (_Carrier([_nest_dict(101), {"k": 1}]) if native
             else _texts(_deep_text(101), '{"value": {"k": 1}}'))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("object", native=native, session=session, retries=1)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "succeeded" and result.output == {"k": 1}
    _assert_reply_too_deep(_correction(model), "structured output")
    assert "provider error (" not in _correction(model)
    if session and not native:
        # A JSON-text reply is text, so memory keeps it.
        assert [effect.status for effect in _view(memory["session"]).effects] == ["completed", "completed"]


def test_a_native_session_value_at_the_limit_is_stored(monkeypatch):
    model = _Carrier([_nest_dict(100)])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("object", native=True, session=True, retries=0)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "succeeded" and result.output == _nest_dict(100)
    assert [effect.status for effect in _view(memory["session"]).effects] == ["completed"]


@pytest.mark.parametrize("levels", [101, 120, 1500])
def test_a_native_session_value_over_the_limit_is_not_stored_and_self_corrects(monkeypatch, levels):
    # The raw reply carries the value, so it is rejected before memory stores it.
    model = _Carrier([_nest_dict(levels), {"k": 1}])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("object", native=True, session=True, retries=1)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "succeeded" and result.output == {"k": 1}
    assert [effect.status for effect in _view(memory["session"]).effects] == ["failed", "completed"]
    _assert_reply_too_deep(_correction(model), "structured output")
    assert "provider error (" not in _correction(model)


class _RecordCarrier:
    """A native model that honours `include_raw`, sending each of `replies` as the raw
    tool call's arguments for a record output."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def with_structured_output(self, schema, *, include_raw=False):
        outer = self

        class Bound:
            def invoke(self, messages):
                outer.calls.append(list(messages))
                args = next(outer.replies)
                if not include_raw:
                    return schema.model_validate(args)
                raw = AIMessage(content="", tool_calls=[
                    {"name": schema.__name__, "args": args, "id": "format-id"}])
                try:
                    parsed = schema.model_validate(args)
                except Exception as error:
                    return {"raw": raw, "parsed": None, "parsing_error": error}
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return Bound()


def _schema_miss(levels: int) -> dict:
    """Return a `{t: object, n: int}` reply nesting `levels` levels under `t`, with a bad `n`."""
    return {"t": _nest_dict(levels), "n": "notint"}


@pytest.mark.parametrize("levels", [125, 1500])
@pytest.mark.parametrize("session", [True, False])
def test_a_deep_native_reply_that_misses_the_schema_retries_on_the_schema_error(
    monkeypatch, session, levels
):
    model = _RecordCarrier([_schema_miss(levels), {"t": {}, "n": 1}])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("{t: object, n: int}", native=True, session=session, retries=1)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "succeeded" and result.output == {"t": {}, "n": 1}
    correction = _correction(model)
    assert "n\n  Input should be a valid integer" in correction
    assert "recursion" not in correction and "provider error (" not in correction
    if session:
        assert [effect.status for effect in _view(memory["session"]).effects] == ["failed", "completed"]


@pytest.mark.parametrize("levels", [125, 1500])
@pytest.mark.parametrize("session", [True, False])
def test_an_exhausted_deep_native_schema_miss_is_located_at_the_field(monkeypatch, session, levels):
    model = _RecordCarrier([_schema_miss(levels)])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("{t: object, n: int}", native=True, session=session, retries=0)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "failed" and result.failure.code == "node.execute"
    assert result.failure.trace[-1].key.endswith("/s:path/s:n")


@pytest.mark.parametrize("levels", [101, 1500])
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("session", [True, False])
def test_a_structured_agent_over_the_limit_without_retries_fails(monkeypatch, native, session, levels):
    model = _Carrier([_nest_dict(levels)]) if native else _texts(_deep_text(levels))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("object", native=native, session=session, retries=0)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "agent"
    _assert_reply_too_deep(result.failure.message, "structured output")


def _deep_call(levels: int, *, first_args: object = None) -> AIMessage:
    """Return a reply calling `lookup` with arguments `{"x": ...}` nested `levels` levels.

    With `first_args`, a valid call with those arguments comes first.
    """
    calls = [{"name": "lookup", "args": {"x": _nest_dict(levels - 1)}, "id": "c2"}]
    if first_args is not None:
        calls.insert(0, {"name": "lookup", "args": first_args, "id": "c1"})
    return AIMessage(content="", tool_calls=calls)


@pytest.mark.parametrize("levels", [101, 1500])
@pytest.mark.parametrize("session", [True, False])
def test_tool_arguments_over_the_limit_fail_the_agent_before_any_call(monkeypatch, session, levels):
    tool, chat = _setup_tool(monkeypatch, [_deep_call(levels)])
    flow, memory = _tool_flow(session)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "agent"
    _assert_reply_too_deep(result.failure.message, "tool call 'lookup' arguments")
    assert tool.calls == [] and len(chat.calls) == 1
    if session:
        effects = _view(memory["session"]).effects
        assert [(effect.kind, effect.status) for effect in effects] == [("model", "failed")]


@pytest.mark.parametrize("session", [True, False])
def test_tool_arguments_over_the_limit_in_a_later_call_stop_every_call(monkeypatch, session):
    tool, _ = _setup_tool(monkeypatch, [_deep_call(101, first_args={"x": 1.0})])
    flow, memory = _tool_flow(session)

    result = run_flow(flow, {"question": "q"}, memory=memory)

    assert result.status == "failed" and result.failure.code == "node.execute"
    assert "deeper than 100 levels" in result.failure.message
    assert tool.calls == []


_REFLECTION = json.dumps({"input_feedback": {}, "self_feedback": [], "variable_feedback": {}})


class _StructuredChat:
    """Chat model replying `values` in turn (the last repeats), then an empty reflection."""

    def __init__(self, *values):
        self.values = list(values)

    def invoke(self, _messages):
        return AIMessage(content=_REFLECTION)

    def with_structured_output(self, schema, **_kwargs):
        values = self.values

        class Bound:
            def invoke(self, _messages):
                return schema.model_validate(values.pop(0) if len(values) > 1 else values[0])

        return Bound()

    def bind_tools(self, *_args, **_kwargs):
        return self


_STRUCTURED_FLOW = """
id: nesting
name: nesting
llm_config: {provider: ollama, model: test}
nodes:
  w:
    kind: agent
    prompt: go
    output: {t: object}
output: ${w.output}
"""


def test_a_model_value_at_the_limit_is_learned_from(monkeypatch):
    chat = _StructuredChat({"t": _nest_dict(99)})
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)

    _assert_learns(_memory_flow(_STRUCTURED_FLOW), {}, "good")


def test_a_model_value_over_the_limit_self_corrects_and_backward_learns(monkeypatch):
    chat = _StructuredChat({"t": _nest_dict(126)}, {"t": {"k": 1}})
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)

    _assert_learns(_memory_flow(_STRUCTURED_FLOW), {}, "good")


@pytest.mark.parametrize("levels", [126, 1500])
def test_a_model_value_over_the_limit_fails_the_agent_and_backward_still_learns(monkeypatch, levels):
    chat = _StructuredChat({"t": _nest_dict(levels)})
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    flow = _memory_flow(_STRUCTURED_FLOW)

    result = run_flow(flow, {}, memory=build_memories(flow), learnable=True)

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "w"
    assert "structured output at t.a" in result.failure.message
    assert "deeper than 100 levels" in result.failure.message

    outcome = result.backward("good")

    assert outcome.status == "succeeded"


class _ToolCallChat:
    """Chat model whose tool-bound replies call `lookup` with `{"q": arguments}`, then
    finish; unbound, it replies with an empty reflection."""

    def __init__(self, arguments):
        self.replies = [
            AIMessage(content="", tool_calls=[
                {"name": "lookup", "args": {"q": arguments}, "id": "1", "type": "tool_call"},
            ]),
            AIMessage(content="finished"),
        ]

    def bind_tools(self, *_args, **_kwargs):
        return SimpleNamespace(invoke=lambda _messages: self.replies.pop(0))

    def invoke(self, _messages):
        return AIMessage(content=_REFLECTION)


_TOOL_CALL_FLOW = """
id: nesting
name: nesting
llm_config: {provider: ollama, model: test}
nodes:
  w:
    kind: agent
    prompt: go
    tools: [lookup]
    output: str
output: ${w.output}
"""


def _run_lookup(monkeypatch, arguments):
    """Run `_TOOL_CALL_FLOW` learnably, calling `lookup` on `arguments`.

    Returns the run result and the arguments each `lookup` call received.
    """
    calls = []

    def lookup(q: object) -> str:
        """Look up `q`."""
        calls.append(q)
        return "ok"

    chat = _ToolCallChat(arguments)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    flow = _memory_flow(_TOOL_CALL_FLOW)
    result = run_flow(
        flow, {}, memory=build_memories(flow), learnable=True, tools={"lookup": lookup}
    )
    return result, calls


def test_tool_arguments_at_the_limit_run_and_are_learned_from(monkeypatch):
    result, calls = _run_lookup(monkeypatch, _nest_dict(99))
    assert result.status == "succeeded" and result.output == "finished"
    assert calls == [_nest_dict(99)]

    outcome = _backward_with_less_stack(result, "good")

    assert outcome.status == "succeeded"
    assert outcome.learning.status == "succeeded"


@pytest.mark.parametrize("levels", [124, 1500])
def test_tool_arguments_over_the_limit_fail_the_agent_and_backward_still_learns(monkeypatch, levels):
    result, calls = _run_lookup(monkeypatch, _nest_dict(levels))
    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "w"
    assert "tool call 'lookup' arguments at q.a" in result.failure.message
    assert "deeper than 100 levels" in result.failure.message
    assert calls == []

    outcome = result.backward("good")

    assert outcome.status == "succeeded"


def _flow_literal(levels: int) -> str:
    """Return a flow-style YAML literal nesting `{a: ...}` `levels` levels deep."""
    return "{a: " * levels + "1" + "}" * levels


@pytest.mark.parametrize("learnable", [False, True])
def test_an_agent_variable_default_over_the_limit_fails_the_agent(monkeypatch, learnable):
    calls = []
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: calls.append(ctx) or Output("x"))
    flow = load_flow(f"""
id: nesting
name: nesting
llm_config: {{provider: ollama, model: test}}
nodes:
  writer:
    kind: agent
    variables: {{t: 'object = {_flow_literal(101)}'}}
    prompt: go
    output: str
output: ${{writer.output}}
""")
    result = run_flow(flow, {}, learnable=learnable)

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "writer"
    assert result.failure.message.startswith("default for 't' at a.a")
    assert "deeper than 100 levels" in result.failure.message
    assert result.failure.trace[0].key == "/s:root/s:node/s:writer/s:variables/s:t/s:default"
    assert calls == []


def test_a_start_default_over_the_limit_fails_start():
    flow = load_flow(f"""
id: nesting
name: nesting
input:
  x: 'object = {_flow_literal(101)}'
nodes:
  c:
    kind: code
    input:
      y: ${{input.x}}
    code: |
      return 1
output: ${{c.output}}
""")
    result = run_flow(flow, {})

    _assert_binding_too_deep(result, "__start__", "default for 'x' at a.a")
    assert result.failure.trace[0].key == "/s:root/s:input/s:x/s:default"
