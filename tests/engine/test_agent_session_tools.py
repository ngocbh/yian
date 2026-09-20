"""API tool effects persist as observations, never as instructions to replay."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict
from pydantic import BaseModel
from types import SimpleNamespace
import pytest

import agent_composer.llm_clients as llm_clients
import agent_composer.tools as tools
from agent_composer.compose import load_flow
from agent_composer.memory.session import SessionMemory
from agent_composer.run import run_flow
from tests.engine.test_agent_session import _FLOW, _Model, _view


_TOOL_FLOW = _FLOW.replace("mode: plain", "mode: tool_calling\n    tools: [lookup]")


class _Arguments(BaseModel):
    value: int


class _Tool:
    args_schema = tool_call_schema = _Arguments

    def __init__(self, action=None):
        self.calls = []
        self.action = action or (lambda args: f"result:{args['value']}")

    def invoke(self, arguments):
        self.calls.append(arguments)
        return self.action(arguments)


class _Chat(_Model):
    def __init__(self, replies):
        super().__init__(replies)
        self.bindings = []

    def bind_tools(self, granted):
        self.bindings.append(granted)
        return self


def _call(value=1, name="lookup", call_id="reused-provider-id"):
    return AIMessage(content="", additional_kwargs={"reasoning_content": "keep reasoning"},
                     tool_calls=[{"name": name, "args": {"value": value}, "id": call_id}])


def test_two_runs_restore_tool_exchanges_without_executing_old_calls(monkeypatch):
    memory = SessionMemory()
    def action(arguments):
        view = _view(memory)
        assert view.effects[-1].kind == "tool" and view.effects[-1].status == "pending"
        assert view.items[-1].payload["message"]["data"]["tool_calls"]
        return f"result:{arguments['value']}"
    tool = _Tool(action)
    chat = _Chat([_call(1), AIMessage(content="one"), _call(2), AIMessage(content="two")])
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    flow = load_flow(_TOOL_FLOW)
    assert run_flow(flow, {"question": "first"}, memory={"session": memory}).output == "one"
    assert tool.calls == [{"value": 1}]
    assert run_flow(flow, {"question": "second"}, memory={"session": memory}).output == "two"
    assert tool.calls == [{"value": 1}, {"value": 2}]
    history = chat.calls[2]
    assert message_to_dict(history[2]) == message_to_dict(_call(1))
    assert isinstance(history[3], ToolMessage) and history[3].content == "result:1"
    assert history[3].tool_call_id == "reused-provider-id"
    view = _view(memory)
    assert [effect.kind for effect in view.effects] == ["model", "tool", "model"] * 2
    assert len(view.items) == 10  # No repeated provider-input history or duplicated tool results.
    requests = [effect.request for effect in view.effects if effect.kind == "tool"]
    assert requests[0]["model_exchange_id"] != requests[1]["model_exchange_id"]
    assert all(effect.status == "completed" for effect in view.effects)
    chat.replies = iter([AIMessage(content="third")])
    run_flow(flow, {"question": "third"}, memory={"session": memory})
    assert [message.content for message in chat.calls[-1] if isinstance(message, ToolMessage)] == ["result:1", "result:2"]
    assert tool.calls == [{"value": 1}, {"value": 2}]


def test_changed_grants_keep_history_but_reject_new_ungranted_call(monkeypatch):
    old, current = _Tool(), _Tool()
    chat = _Chat([_call(), AIMessage(content="one"), _call()])
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", old)
    monkeypatch.setitem(tools.TOOL_REGISTRY, "current", current)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    memory = SessionMemory()
    run_flow(load_flow(_TOOL_FLOW), {"question": "first"}, memory={"session": memory})
    changed = load_flow(_TOOL_FLOW.replace("tools: [lookup]", "tools: [current]"))
    failed = run_flow(changed, {"question": "second"}, memory={"session": memory})
    assert failed.failure.code == "node.execute" and "not granted" in failed.failure.message
    assert old.calls == [{"value": 1}] and current.calls == []
    assert chat.bindings == [[old], [current]]
    assert any(isinstance(message, ToolMessage) for message in chat.calls[-1])


@pytest.mark.parametrize("representation_failure", [False, True])
def test_missing_tool_result_is_repaired_only_in_context_with_truthful_status(monkeypatch, representation_failure):
    class Unprintable:
        def __str__(self):
            raise ValueError("cannot represent result")
    def action(arguments):
        if representation_failure:
            return Unprintable()
        raise ValueError("tool failed")
    tool = _Tool(action)
    chat = _Chat([_call(), AIMessage(content="later")])
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    memory = SessionMemory()
    flow = load_flow(_TOOL_FLOW)
    assert run_flow(flow, {"question": "first"}, memory={"session": memory}).failure is not None
    prior = _view(memory)
    assert prior.effects[-1].status == ("completed" if representation_failure else "failed")
    assert len(prior.items) == 3
    assert run_flow(flow, {"question": "second"}, memory={"session": memory}).output == "later"
    repairs = [message for message in chat.calls[-1] if isinstance(message, ToolMessage)]
    assert len(repairs) == 1 and repairs[0].status == "error"
    assert ("completed" if representation_failure else "failed") in repairs[0].content
    assert "unavailable" in repairs[0].content
    assert any(isinstance(message, HumanMessage) and "invocation" in message.content and "failed" in message.content
               for message in chat.calls[-1])
    assert len(_view(memory).items) == 6 and len(tool.calls) == 1
    assert all(item.payload["message"]["type"] != "tool" for item in _view(memory).items)


@pytest.mark.parametrize("reply", [
    AIMessage(content="raw function", additional_kwargs={"function_call": {"name": "lookup", "arguments": "{}"}}),
    AIMessage(content="raw tool", additional_kwargs={"tool_calls": [{"id": "broken", "type": "function", "function": {"name": "lookup", "arguments": "{"}}]}),
    AIMessage(content="bad tool", invalid_tool_calls=[{"name": "lookup", "args": "{", "id": "broken", "error": "invalid"}]),
    AIMessage(content=[{"type": "tool_use", "id": "anthropic-call", "name": "lookup", "input": {"value": 1}}]),
    AIMessage(content="bad name", tool_calls=[{"name": "", "args": {}, "id": "call"}]),
])
def test_unpairable_provider_call_shapes_become_inert_context_only(monkeypatch, reply):
    tool, chat, memory = _Tool(), _Chat([reply, AIMessage(content="later")]), SessionMemory()
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    flow = load_flow(_TOOL_FLOW)
    run_flow(flow, {"question": "first"}, memory={"session": memory})
    original = _view(memory).items[-1].payload
    run_flow(flow, {"question": "second"}, memory={"session": memory})
    assert tool.calls == []
    assert any(isinstance(message, HumanMessage) and "context only" in message.content for message in chat.calls[-1])
    assert all(not getattr(message, "tool_calls", None) and not getattr(message, "invalid_tool_calls", None)
               and "function_call" not in message.additional_kwargs and "tool_calls" not in message.additional_kwargs
               and not isinstance(message.content, list) for message in chat.calls[-1])
    assert _view(memory).items[2].payload == original


@pytest.mark.parametrize("stage", ["before_model", "after_model", "after_tool"])
def test_stop_boundaries_retain_surfaced_tool_truth_before_unwind(monkeypatch, stage):
    from agent_composer.nodes.agent import AgentNode
    from agent_composer.nodes.base import _HostStop
    stops = []
    original_ctx = AgentNode._ctx
    def capture_ctx(node, *args, **kwargs):
        ctx = original_ctx(node, *args, **kwargs)
        stops.append(ctx.stop_event)
        return ctx
    monkeypatch.setattr(AgentNode, "_ctx", capture_ctx)
    def reply(messages):
        if stage == "after_model":
            stops[-1].set()
        return _call()
    def action(arguments):
        if stage == "after_tool":
            stops[-1].set()
        return "actual result"
    chat, tool, memory = _Chat([reply, AIMessage(content="later")]), _Tool(action), SessionMemory()
    original_bind = chat.bind_tools
    def bind(grants):
        if stage == "before_model":
            stops[-1].set()
        return original_bind(grants)
    monkeypatch.setattr(chat, "bind_tools", bind)
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    flow = load_flow(_TOOL_FLOW)
    with pytest.raises(_HostStop):
        run_flow(flow, {"question": "first"}, memory={"session": memory})
    view = _view(memory)
    assert view.closed_invocations[-1].status == "interrupted"
    assert len(chat.calls) == (0 if stage == "before_model" else 1)
    assert len(tool.calls) == (1 if stage == "after_tool" else 0)
    assert all(effect.status == "completed" for effect in view.effects)
    if stage == "before_model":
        assert view.items == ()
        return
    run_flow(flow, {"question": "second"}, memory={"session": memory})
    restored = [message for message in chat.calls[-1] if isinstance(message, ToolMessage)]
    assert len(restored) == 1
    if stage == "after_tool":
        assert restored[0].content == "actual result" and restored[0].status == "success"
        assert _view(memory).items[3].payload["message"]["type"] == "tool"
    else:
        assert restored[0].status == "error" and "outcome unknown" in restored[0].content
        assert all(item.payload["message"]["type"] != "tool" for item in _view(memory).items)


@pytest.mark.parametrize("stage", ["request", "observe", "encode"])
def test_tool_memory_fault_cannot_repeat_external_action(monkeypatch, stage):
    from agent_composer.nodes.agent import conversation
    from agent_composer.nodes.base import _CapabilityFault
    error, calls, observations = ValueError("memory refused"), [], []
    def request(*args, **kwargs):
        if stage == "request":
            raise error
        return 1
    def observe(ordinal, status, **kwargs):
        observations.append(status)
        if stage == "observe":
            raise error
    def encode(*args):
        raise error
    if stage == "encode":
        monkeypatch.setattr(conversation, "encode_message", encode)
    with pytest.raises(_CapabilityFault) as caught:
        conversation.invoke_tool_effect(
            SimpleNamespace(request=request, observe=observe), lambda: calls.append("tool") or "actual",
            name="lookup", arguments={"value": 1}, model_exchange_id="model", call_index=0,
            tool_call_id="provider", check_stopped=lambda: None,
        )
    assert caught.value.original is error
    assert calls == ([] if stage == "request" else ["tool"])
    assert observations == ([] if stage == "request" else ["completed"])


@pytest.mark.parametrize("host_stop", [False, True])
def test_tool_error_identity_survives_failed_observation(host_stop):
    from agent_composer.nodes.agent.conversation import invoke_tool_effect
    from agent_composer.nodes.base import _CapabilityFault, _HostStop
    primary = _HostStop("stopped") if host_stop else ValueError("tool failed")
    seen = []
    def invoke():
        raise primary
    def observe(ordinal, status, **kwargs):
        seen.append(status)
        raise RuntimeError("cannot save failure")
    with pytest.raises(type(primary) if host_stop else _CapabilityFault) as caught:
        invoke_tool_effect(SimpleNamespace(request=lambda *args, **kwargs: 1, observe=observe), invoke,
                           name="lookup", arguments={}, model_exchange_id="model", call_index=0,
                           tool_call_id="provider", check_stopped=lambda: None)
    assert (caught.value if host_stop else caught.value.original) is primary
    assert seen == ["interrupted" if host_stop else "failed"]
    if hasattr(primary, "add_note"):
        assert "cannot save failure" in primary.__notes__[0]


def test_session_tool_calls_keep_public_argument_validation(monkeypatch):
    tool, memory = _Tool(), SessionMemory()
    bad = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"value": 1, "private": "no"}, "id": "call"}])
    chat = _Chat([bad])
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    result = run_flow(load_flow(_TOOL_FLOW), {"question": "first"}, memory={"session": memory})
    assert "public invocation" in result.failure.message
    assert tool.calls == [] and _view(memory).effects[-1].status == "failed"


def test_restored_unclosed_invocation_reports_unknown_completion_without_changing_history():
    from agent_composer.memory import MemoryFrameSpec
    from agent_composer.nodes.agent.conversation import history_messages, invoke_model, invoke_tool_effect
    from tests.engine._session_fakes import REF, harness, request, snapshot
    memory, actions = SessionMemory(), []
    with harness(memory) as opened:
        with opened.frame.open_turn(request(run_id="older"), check_stopped=lambda: None) as turn:
            invoke_model(turn, lambda: _call(), new_messages=[HumanMessage(content="first")],
                         exchange_id="older-model", check_stopped=lambda: None)
            invoke_tool_effect(turn, lambda: actions.append("executed") or "actual result",
                               name="lookup", arguments={"value": 1}, model_exchange_id="older-model",
                               call_index=0, tool_call_id="reused-provider-id", check_stopped=lambda: None)
            saved = snapshot(memory)
    assert saved["conversations"][0]["closed_invocations"] == []
    restored = SessionMemory()
    with restored.open_run(snapshot=saved) as session:
        frame = session.open_frame(MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request(run_id="current"), check_stopped=lambda: None) as turn:
            before = turn.view
            context = history_messages(turn)
            assert turn.view == before
            assert any(isinstance(message, ToolMessage) and message.content == "actual result" for message in context)
            statuses = [message.content for message in context if isinstance(message, HumanMessage)
                        and message.content.startswith("Session history status")]
            assert len(statuses) == 1 and "open" in statuses[0] and "completion unknown" in statuses[0]
            assert "failed" not in statuses[0] and "interrupted" not in statuses[0]
    assert actions == ["executed"]
    assert len(_view(restored).items) == 3
