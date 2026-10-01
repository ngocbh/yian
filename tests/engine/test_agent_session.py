"""API messages persist through the memory-owned optional conversation turn."""

from copy import deepcopy
from contextlib import ExitStack
from types import SimpleNamespace
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, message_to_dict
import pytest

import agent_composer.llm_clients as llm_clients
from agent_composer.compose import load_flow
from agent_composer.memory.episodic import EpisodicMemory
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.memory.session import SessionMemory
from agent_composer.run import resume_flow, run_flow


_FLOW = """id: plain-session
name: plain_session
llm_config: {provider: openai, model: test-model}
input:
  question: str
nodes:
  agent:
    kind: agent
    mode: plain
    long_term_memory: session
    prompt: ${question}
    input:
      question: ${input.question}
    output: str
output: ${agent.output}
"""


class _Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(deepcopy(messages))
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply(messages)
        return reply


def _view(memory):
    from agent_composer.memory.conversation import conversation_view_from_json
    with memory.open_run() as opened:
        return conversation_view_from_json(opened.snapshot["conversations"][0])


def test_two_fresh_plain_runs_retain_actual_history_without_learning(monkeypatch):
    model = _Model([AIMessage(content="first reply"), AIMessage(content="second reply")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    flow = load_flow(_FLOW)
    first = run_flow(flow, {"question": "first question"}, memory={"session": memory})
    second = run_flow(flow, {"question": "second question"}, memory={"session": memory})
    assert first.output == "first reply" and second.output == "second reply"
    assert first.learning is None and second.learning is None
    assert [message.content for message in model.calls[1][1:]] == [
        "first question", "first reply", "second question",
    ]
    assert [type(message) for message in model.calls[1]] == [
        SystemMessage, HumanMessage, AIMessage, HumanMessage,
    ]
    view = _view(memory)
    assert len(view.items) == 6  # Two new inputs and one reply per call, never repeated history.
    assert len({item.item_id for item in view.items}) == 6
    exchange_ids = [effect.request["exchange_id"] for effect in view.effects]
    assert len(set(exchange_ids)) == 2
    assert [item.payload["exchange_id"] for item in view.items] == [exchange_ids[0]] * 3 + [exchange_ids[1]] * 3
    assert all(effect.status == "completed" for effect in view.effects)


@pytest.mark.parametrize("backend,policy", [(None, "session"), (EpisodicMemory, "episodic"),
                                           (ReflectiveMemory, "reflective")])
def test_no_session_backend_keeps_plain_calls_fresh(monkeypatch, backend, policy):
    model = _Model([AIMessage(content="one"), AIMessage(content="two")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = None if backend is None else {policy: backend()}
    flow = load_flow(_FLOW.replace("long_term_memory: session", f"long_term_memory: {policy}"))
    run_flow(flow, {"question": "first"}, memory=memory)
    run_flow(flow, {"question": "second"}, memory=memory)
    assert [message.content for message in model.calls[1][1:]] == ["second"]


@pytest.mark.parametrize("message", [
    SystemMessage(content="current rules"),
    HumanMessage(content=[{"type": "text", "text": "look"},
                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]),
    AIMessage(content="thinking", additional_kwargs={"reasoning_content": "needed by provider"},
              response_metadata={"model_name": "test", "usage": {"tokens": 12}},
              tool_calls=[{"name": "lookup", "args": {"value": 1}, "id": "provider-call"}]),
    ToolMessage(content="result", tool_call_id="provider-call", artifact={"value": [1, 2]}, status="success"),
])
def test_message_codec_round_trip_preserves_exact_provider_fields(message):
    from agent_composer.nodes.agent.memory import decode_message, encode_message
    item = encode_message(message, "exchange")
    assert item.payload["exchange_id"] == "exchange"
    assert message_to_dict(decode_message(item)) == message_to_dict(message)
    assert encode_message(message, "exchange").item_id != item.item_id
    detached = item.payload
    detached["message"]["data"]["content"] = "changed"
    assert message_to_dict(decode_message(item)) == message_to_dict(message)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object(), {1: "not a JSON key"}])
def test_message_codec_rejects_non_json_provider_payloads(bad):
    from agent_composer.nodes.agent.memory import encode_message
    message = AIMessage(content="reply", response_metadata={"bad": bad})
    with pytest.raises((TypeError, ValueError)):
        encode_message(message, "exchange")


@pytest.mark.parametrize("change", ["format", "exchange", "missing", "role", "coercion"])
def test_message_codec_rejects_unknown_or_corrupt_stored_format(change):
    from agent_composer.memory import ConversationItem
    from agent_composer.nodes.agent.memory import decode_message, encode_message
    original = encode_message(HumanMessage(content="input"), "exchange")
    payload = original.payload
    format = original.format
    if change == "format":
        format = "unknown-v9"
    elif change == "exchange":
        payload["exchange_id"] = False
    elif change == "missing":
        del payload["message"]["data"]["content"]
    elif change == "role":
        payload["message"]["type"] = "not-a-role"
    else:
        payload["message"]["data"]["id"] = 7
    with pytest.raises((TypeError, ValueError)):
        decode_message(ConversationItem(original.item_id, format, payload))


@pytest.mark.parametrize("wire", [
    {"type": "human", "data": {"content": "old question"}},
    {"type": "ai", "data": {"content": "old reply", "additional_kwargs": {"reasoning_content": "kept"}}},
    {"type": "tool", "data": {"content": "old result", "tool_call_id": "provider-call"}},
])
def test_message_codec_accepts_older_payloads_without_optional_defaults(wire):
    from agent_composer.memory import ConversationItem
    from agent_composer.nodes.agent.memory import MESSAGE_FORMAT, decode_message
    item = ConversationItem("older-item", MESSAGE_FORMAT, {"exchange_id": "older-exchange", "message": wire})
    decoded = message_to_dict(decode_message(item))
    assert decoded["type"] == wire["type"]
    assert {key: decoded["data"][key] for key in wire["data"]} == wire["data"]


@pytest.mark.parametrize("role,field,call,discriminator", [
    ("ai", "tool_calls", {"name": "lookup", "args": {"id": 1}, "id": "call-1"}, "tool_call"),
    ("ai", "invalid_tool_calls", {"name": "lookup", "args": "{", "id": "call-1", "error": "invalid JSON"}, "invalid_tool_call"),
    ("AIMessageChunk", "tool_call_chunks", {"name": "lookup", "args": '{"id":1}', "id": "call-1", "index": 0}, "tool_call_chunk"),
])
def test_message_codec_accepts_nested_tool_call_discriminator_defaults(role, field, call, discriminator):
    from agent_composer.memory import ConversationItem
    from agent_composer.nodes.agent.memory import MESSAGE_FORMAT, decode_message
    wire = {"type": role, "data": {"content": "", field: [call]}}
    item = ConversationItem("older-item", MESSAGE_FORMAT, {"exchange_id": "exchange", "message": wire})
    restored = message_to_dict(decode_message(item))["data"][field][0]
    assert restored == {**call, "type": discriminator}
    assert item.payload["message"] == wire


@pytest.mark.parametrize("change", [
    "changed", "dropped", "coerced", "added_call_key", "added_args_key",
    "added_content_key", "added_kwargs_key", "decoder_input",
])
def test_message_codec_rejects_nested_data_loss_and_opaque_payload_changes(monkeypatch, change):
    from agent_composer.memory import ConversationItem
    from agent_composer.nodes.agent import memory as conversation
    wire = {"type": "ai", "data": {
        "content": [{"type": "text", "text": "reply"}],
        "additional_kwargs": {"reasoning_content": "reasoning"},
        "tool_calls": [{"name": "lookup", "args": {"id": 1}, "id": "call-1", "type": "tool_call"}],
    }}
    item = ConversationItem("older-item", conversation.MESSAGE_FORMAT, {"exchange_id": "exchange", "message": wire})
    real_encode, real_decode = conversation.message_to_dict, conversation.messages_from_dict
    def changed_encode(message):
        encoded = real_encode(message)
        data = encoded["data"]
        call = data["tool_calls"][0]
        if change == "changed":
            call["name"] = "different"
        elif change == "dropped":
            del call["id"]
        elif change == "coerced":
            call["args"]["id"] = True  # Python equality must not hide integer coercion.
        elif change == "added_call_key":
            call["unrecognized"] = "not a schema default"
        elif change == "added_args_key":
            call["args"]["extra"] = "not a schema default"
        elif change == "added_content_key":
            data["content"][0]["extra"] = "not a schema default"
        elif change == "added_kwargs_key":
            data["additional_kwargs"]["extra"] = "not a schema default"
        return encoded
    def changed_decode(messages):
        messages[0]["data"]["tool_calls"][0]["args"]["id"] = True
        return real_decode(messages)
    if change == "decoder_input":
        monkeypatch.setattr(conversation, "messages_from_dict", changed_decode)
    else:
        monkeypatch.setattr(conversation, "message_to_dict", changed_encode)
    with pytest.raises(ValueError, match="invalid stored"):
        conversation.decode_message(item)
    assert item.payload["message"] == wire


def test_message_codec_rejects_replaced_supplied_tool_call_discriminator():
    from agent_composer.memory import ConversationItem
    from agent_composer.nodes.agent.memory import MESSAGE_FORMAT, decode_message
    wire = {"type": "ai", "data": {"content": "", "tool_calls": [
        {"name": "lookup", "args": {"id": 1}, "id": "call-1", "type": "different"},
    ]}}
    item = ConversationItem("older-item", MESSAGE_FORMAT, {"exchange_id": "exchange", "message": wire})
    with pytest.raises(ValueError, match="invalid stored"):
        decode_message(item)


@pytest.mark.parametrize("stage", ["request", "observe", "encode_input", "encode_reply", "history", "view"])
def test_adapter_memory_faults_are_distinct_from_provider_failures(stage):
    from agent_composer.memory import ConversationItem
    from agent_composer.nodes.agent.memory import history_messages, invoke_model
    from agent_composer.nodes.base import _CapabilityFault
    calls = []
    expected = ValueError("memory refused write")
    def request(*args, **kwargs):
        if stage == "request":
            raise expected
        return 1
    def observe(*args, **kwargs):
        if stage == "observe":
            raise expected
    def invoke():
        calls.append("provider")
        return AIMessage(content="reply", response_metadata={"bad": object()} if stage == "encode_reply" else {})
    message = HumanMessage(content="new", response_metadata={"bad": object()} if stage == "encode_input" else {})
    class BrokenView:
        @property
        def view(self):
            raise expected
    with pytest.raises(_CapabilityFault) as caught:
        if stage == "history":
            history_messages(SimpleNamespace(view=SimpleNamespace(items=(ConversationItem("old", "unknown", {}),))))
        elif stage == "view":
            history_messages(BrokenView())
        else:
            invoke_model(SimpleNamespace(request=request, observe=observe), invoke,
                         new_messages=[message], check_stopped=lambda: None)
    if stage in {"request", "observe", "view"}:
        assert caught.value.original is expected
    assert len(calls) == (1 if stage in {"observe", "encode_reply"} else 0)


def test_compatible_model_and_prompt_changes_retain_history_with_current_system(monkeypatch):
    from agent_composer.nodes.agent.modes import plain as plain_module
    model = _Model([AIMessage(content="one"), AIMessage(content="two")])
    configs = []
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: configs.append(config) or model)
    memory = SessionMemory()
    monkeypatch.setattr(plain_module, "DEFAULT_SYSTEM", "old system instructions")
    run_flow(load_flow(_FLOW), {"question": "first"}, memory={"session": memory})
    monkeypatch.setattr(plain_module, "DEFAULT_SYSTEM", "current system instructions")
    changed = _FLOW.replace("test-model", "different-model").replace("prompt: ${question}", "prompt: changed ${question}")
    run_flow(load_flow(changed), {"question": "second"}, memory={"session": memory})
    assert configs[-1]["model"] == "different-model"
    assert [message.content for message in model.calls[-1]] == [
        "current system instructions", "first", "one", "changed second",
    ]
    assert set(_view(memory).compatibility) == {"provider", "message_format"}


def test_admission_follows_render_and_recall_and_only_current_recall_is_reused(monkeypatch):
    from agent_composer.memory import MemoryContext, MemorySection
    from agent_composer.nodes.agent import AgentNode
    from tests.engine._session_fakes import REF, harness, request
    memory = SessionMemory()
    model = _Model([AIMessage(content="one"), AIMessage(content="two")])
    events = []
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: events.append("model") or model)
    node = AgentNode("agent", prompt="question ${value}")
    for index, note in enumerate(("old recall", "current recall")):
        with harness(memory) as opened, ExitStack() as stack:
            def open_context(system, prompt, compatibility):
                assert prompt == f"question {index}"
                events.append("recall")
                sections = (MemorySection(REF, note),)
                events.append("admit")
                turn = stack.enter_context(opened.frame.open_turn(
                    request(run_id=str(index), compatibility=compatibility), check_stopped=lambda: None,
                ))
                return MemoryContext(sections=sections, transcript=turn)
            node.forward({"value": index}, open_memory_context=open_context,
                         capture_render=lambda _: events.append("render"))
    assert events == ["render", "recall", "admit", "model"] * 2
    assert "current recall" in model.calls[-1][0].content
    assert all("old recall" not in message.content for message in model.calls[-1])
    assert [message.content for message in model.calls[-1][1:]] == ["question 0", "one", "question 1"]


def test_provider_change_requires_reset_then_starts_fresh(monkeypatch):
    from agent_composer.memory import MemoryOperationError
    model = _Model([AIMessage(content="one"), AIMessage(content="two")])
    factories = []
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: factories.append(config) or model)
    memory = SessionMemory()
    run_flow(load_flow(_FLOW), {"question": "first"}, memory={"session": memory})
    changed = load_flow(_FLOW.replace("provider: openai", "provider: anthropic"))
    with pytest.raises(MemoryOperationError, match="reset"):
        run_flow(changed, {"question": "second"}, memory={"session": memory})
    assert len(factories) == 1
    memory.reset()
    assert run_flow(changed, {"question": "second"}, memory={"session": memory}).output == "two"
    assert [message.content for message in model.calls[-1][1:]] == ["second"]


def test_compatibility_matches_factory_defaults_and_config_carriers(monkeypatch):
    from agent_composer.nodes.agent import memory as conversation
    from agent_composer.llm_clients import LLMConfig
    monkeypatch.setattr(conversation, "default_llm_provider", lambda: "OpenAI")
    expected = {"provider": "openai", "message_format": conversation.MESSAGE_FORMAT}
    assert conversation.compatibility(None) == expected
    assert conversation.compatibility(LLMConfig()) == expected
    assert conversation.compatibility({"provider": "OpenAI", "model": "other", "api_key": "not-retained"}) == expected


def test_request_is_stored_before_provider_and_failed_effect_remains_truthful(monkeypatch):
    error = RuntimeError("provider refused; secret-marker")
    memory = SessionMemory()
    def fail(messages):
        view = _view(memory)
        assert len(view.effects) == 1 and view.effects[0].status == "pending"
        assert len(view.items) == 2
        raise error
    model = _Model([fail, AIMessage(content="later")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow = load_flow(_FLOW)
    failed = run_flow(flow, {"question": "first"}, memory={"session": memory})
    assert failed.failure.code == "node.execute"
    view = _view(memory)
    assert view.effects[0].status == "failed"
    assert view.closed_invocations[-1].status == "failed"
    assert view.effects[0].failure.message == "Model invocation failed (RuntimeError)."
    with memory.open_run() as opened:
        assert "secret-marker" not in json.dumps(opened.snapshot)
    assert run_flow(flow, {"question": "second"}, memory={"session": memory}).output == "later"
    assert [message.content for message in model.calls[-1][1:]] == [
        "first", 'Session history status (context only): [{"invocation":"failed"},{"effect":"model","status":"failed"}]', "second",
    ]


@pytest.mark.parametrize("host_stop", [False, True])
def test_failed_observation_does_not_replace_provider_or_host_error(host_stop):
    from agent_composer.nodes.agent.memory import invoke_model
    from agent_composer.nodes.base import _CapabilityFault, _HostStop
    primary = _HostStop("provider cancelled") if host_stop else ValueError("provider failed")
    seen = []
    def invoke():
        raise primary
    def observe(ordinal, status, **kwargs):
        seen.append(status)
        raise RuntimeError("cannot save failure")
    turn = SimpleNamespace(request=lambda *args, **kwargs: 1, observe=observe)
    with pytest.raises(type(primary) if host_stop else _CapabilityFault) as caught:
        invoke_model(turn, invoke, new_messages=[HumanMessage(content="input")], check_stopped=lambda: None)
    assert (caught.value if host_stop else caught.value.original) is primary
    assert seen == ["interrupted" if host_stop else "failed"]
    if hasattr(primary, "add_note"):
        assert any("cannot save failure" in note for note in primary.__notes__)


def test_reply_after_stop_is_saved_and_resume_retries_using_latest_history(monkeypatch, tmp_path):
    from agent_composer.nodes.agent import AgentNode
    from agent_composer.nodes.base import _HostStop
    memory = SessionMemory()
    stops = []
    original_ctx = AgentNode._ctx
    def capture_ctx(node, *args, **kwargs):
        ctx = original_ctx(node, *args, **kwargs)
        stops.append(ctx.stop_event)
        return ctx
    monkeypatch.setattr(AgentNode, "_ctx", capture_ctx)
    def finish_after_stop(messages):
        stops[-1].set()
        return AIMessage(content="surfaced before cancellation", additional_kwargs={"reasoning_content": "retained"})
    model = _Model([finish_after_stop, AIMessage(content="retry done")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow = load_flow(_FLOW)
    path = tmp_path / "retry.jsonl"
    with pytest.raises(_HostStop):
        run_flow(flow, {"question": "question"}, memory={"session": memory}, log=path)
    prefix = path.read_bytes()
    view = _view(memory)
    assert view.effects[0].status == "completed"
    assert view.closed_invocations[-1].status == "interrupted"
    assert view.items[-1].payload["message"]["data"]["content"] == "surfaced before cancellation"
    resumed = resume_flow(flow, log=path, memory={"session": memory})
    assert resumed.output == "retry done" and path.read_bytes().startswith(prefix)
    assert [message.content for message in model.calls[-1][1:]] == [
        "question", "surfaced before cancellation",
        'Session history status (context only): [{"invocation":"interrupted"}]', "question",
    ]
    view = _view(memory)
    assert len(view.effects) == 2 and len(view.items) == 6
    assert view.effects[0].invocation != view.effects[1].invocation


def test_postassertion_failure_retains_actual_reply_and_failed_turn(monkeypatch):
    from agent_composer.nodes.contracts import Assertion
    model = _Model([AIMessage(content="actual reply")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    flow = load_flow(_FLOW)
    flow.compiled.nodes["agent"].post_asserts = [Assertion("false", "must fail")]
    result = run_flow(flow, {"question": "input"}, memory={"session": memory})
    assert result.failure.code == "assertion.failed"
    view = _view(memory)
    assert view.effects[-1].status == "completed"
    assert view.items[-1].payload["message"]["data"]["content"] == "actual reply"
    assert view.closed_invocations[-1].status == "failed"
