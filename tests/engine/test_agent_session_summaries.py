"""API history summaries use the current unbound model and safe context only."""

from agent_composer.memory import MemoryContext

from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError
import json
from threading import Event
from types import SimpleNamespace

import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

import agent_composer.llm_clients as llm_clients
from agent_composer.memory.session import SessionMemory
from agent_composer.memory import MemoryFrameSpec, MemorySection
from agent_composer.nodes.agent import memory as conversation
from agent_composer.nodes.agent.modes.common import AgentRunContext
from agent_composer.nodes.agent.modes.plain import plain
from agent_composer.nodes.agent.modes.tool_calling import tool_calling
from agent_composer.nodes.base import _CapabilityFault, _HostStop
from agent_composer.typesys.values import Type, ValueKind
from tests.engine._session_fakes import REF, harness, request, snapshot
from tests.engine.test_agent_session import _Model
from tests.engine.test_agent_session_structured import _Native, _config
from tests.engine.test_agent_session_tools import _Chat, _Tool, _call


def _seed(memory, *, count=1):
    with harness(memory) as opened:
        for index in range(count):
            with opened.frame.open_turn(
                request(str(index)), check_stopped=lambda: None
            ) as turn:
                conversation.invoke_model(
                    turn,
                    lambda: AIMessage(content=f"reply-{index}"),
                    new_messages=[HumanMessage(content=f"question-{index}")],
                    check_stopped=lambda: None,
                )


def test_adapter_summary_is_context_only_and_uses_same_model():
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    model = _Model([AIMessage(content="summary text")])
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            before = turn.view
            context = conversation.prepare_history(
                turn, model, check_stopped=lambda: None
            )
            assert len(model.calls) == 1
            assert "question-0" in str(model.calls[0])
            assert "summary text" in str(context)
            assert turn.view == before


@pytest.mark.parametrize(
    "reply",
    [
        AIMessage(content=""),
        AIMessage(content="   "),
        AIMessage(content=[{"type": "reasoning", "encrypted_content": "opaque"}]),
        AIMessage(content=[{"type": "reasoning", "opaque": object()},
                           {"type": "text", "text": "summary"}]),
        "not a message",
        AIMessage(content="", tool_calls=[{"name": "tool", "args": {}, "id": "call"}]),
        AIMessage(
            content="text", additional_kwargs={"function_call": {"name": "tool"}}
        ),
        AIMessage(
            content="text",
            invalid_tool_calls=[{"name": "bad", "args": "{", "id": "bad"}],
        ),
        AIMessage(
            content=[{"type": "tool_use", "id": "call", "name": "tool", "input": {}}]
        ),
        AIMessage(
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
            ]
        ),
        AIMessage(content="text", response_metadata={"not_json": object()}),
    ],
)
def test_invalid_summary_reply_is_an_ordinary_invocation_failure(reply):
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            with pytest.raises((TypeError, ValueError)) as caught:
                conversation.prepare_history(
                    turn, _Model([reply]), check_stopped=lambda: None
                )
            assert not isinstance(caught.value, _CapabilityFault)


def test_summary_provider_exception_preserves_identity_outside_capability_boundary():
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    error = RuntimeError("provider failed")
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            with pytest.raises(RuntimeError) as caught:
                conversation.prepare_history(
                    turn, _Model([error]), check_stopped=lambda: None
                )
            assert caught.value is error


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, FutureCancelledError, AsyncCancelledError, KeyboardInterrupt],
)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", [plain, tool_calling])
def test_summary_failure_never_enters_answer_structured_retry(
    monkeypatch, error_type, native, mode
):
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    error = error_type("summary error")
    model = _Chat([error])
    model.with_structured_output = lambda *args, **kwargs: pytest.fail(
        "structured binding after failed summary"
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    before = snapshot(memory)["conversation"]
    with pytest.raises(error_type) as caught:
        with harness(memory) as opened:
            with opened.frame.open_turn(
                request("current"), check_stopped=lambda: None
            ) as turn:
                mode(
                    AgentRunContext(
                        node_id="agent",
                        prompt="current question",
                        memory_context=MemoryContext(transcript=turn),
                        output_type=Type.scalar(ValueKind.INTEGER),
                        retries=7,
                        llm_config=_config(native),
                    )
                )
    assert caught.value is error and len(model.calls) == 1 and model.bindings == []
    after = snapshot(memory)["conversation"]
    assert after["items"] == before["items"] and after["effects"] == before["effects"]
    assert after["closed_invocations"][-1]["status"] != "completed"


@pytest.mark.parametrize("native", [False, True])
def test_tool_then_structured_uses_summary_once_and_safe_current_projection(
    monkeypatch, native
):
    import agent_composer.tools as tools

    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    tool = _Tool()
    # The final chat reply has hidden malformed tool metadata, requiring an
    # inert projection when it becomes input to the structured pass.
    malformed = AIMessage(
        content="ready", additional_kwargs={"function_call": {"name": "stale"}}
    )
    replies = [AIMessage(content="old summary"), _call(), malformed]
    if not native:
        replies.extend(
            [AIMessage(content='{"value":"bad"}'), AIMessage(content='{"value":9}')]
        )
    chat = _Chat(replies)
    structured = _Native(["bad", 9])
    if native:
        chat.with_structured_output = structured.with_structured_output
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            result = tool_calling(
                AgentRunContext(
                    node_id="agent",
                    prompt="current question",
                    tools=["lookup"],
                    memory_context=MemoryContext(transcript=turn),
                    output_type=Type.scalar(ValueKind.INTEGER),
                    llm_config=_config(native),
                )
            )
    assert result.value == 9 and tool.calls == [{"value": 1}]
    structured_calls = structured.calls if native else chat.calls[-2:]
    for messages in structured_calls:
        assert (
            sum(
                "Historical conversation summary" in str(message.content)
                for message in messages
            )
            == 1
        )
        assert "question-0" not in [message.content for message in messages]
        assert "reply-0" not in [message.content for message in messages]
        assert "current question" in [message.content for message in messages]
        assert any(
            isinstance(message, ToolMessage) and message.content == "result:1"
            for message in messages
        )
        assert any(
            isinstance(message, HumanMessage) and "unpairable" in message.content
            for message in messages
        )
        assert not any(
            "function_call" in message.additional_kwargs for message in messages
        )
        assert not any(
            "completion unknown" in str(message.content) for message in messages
        )
    raw = snapshot(memory)["conversation"]
    assert "old summary" not in json.dumps(raw)
    assert sum(effect["kind"] == "model" for effect in raw["effects"]) == 5
    assert (
        sum(
            item["payload"]["message"]["data"]["content"] == "current question"
            for item in raw["items"]
        )
        == 1
    )


def test_summary_uses_unbound_model_and_current_instructions_and_grants(monkeypatch):
    import agent_composer.tools as tools

    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    with harness(memory) as opened:
        with opened.frame.open_turn(request("old"), check_stopped=lambda: None) as turn:
            conversation.invoke_model(
                turn,
                lambda: AIMessage(content="old reply"),
                new_messages=[
                    SystemMessage(content="OLD SYSTEM"),
                    ChatMessage(role="developer", content="OLD DEVELOPER"),
                    HumanMessage(content="old question"),
                ],
                check_stopped=lambda: None,
            )
    bound = _Chat([_call(name="current"), AIMessage(content="answer")])
    events = []

    class Unbound(_Model):
        def invoke(self, messages):
            events.append("summarize")
            return super().invoke(messages)

        def bind_tools(self, granted):
            events.append("bind")
            assert granted == [tool]
            return bound

    model, tool = Unbound([AIMessage(content="facts")]), _Tool()
    monkeypatch.setitem(tools.TOOL_REGISTRY, "current", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            result = tool_calling(
                AgentRunContext(
                    node_id="agent",
                    prompt="new question",
                    tools=["current"],
                    memory_context=MemoryContext(sections=(MemorySection(REF, "CURRENT NOTE"),), transcript=turn),
                )
            )
    assert result.value == "answer" and events == ["summarize", "bind"]
    assert "OLD SYSTEM" not in str(model.calls) and "OLD DEVELOPER" not in str(
        model.calls
    )
    assert "CURRENT NOTE" in bound.calls[0][0].content
    assert tool.calls == [{"value": 1}]


def test_projection_uses_whole_model_exchanges_and_inert_repairs():
    memory = SessionMemory(history_budget_chars=1, recent_turns=1)
    with harness(memory) as opened:
        with pytest.raises(RuntimeError):
            with opened.frame.open_turn(
                request("older"), check_stopped=lambda: None
            ) as turn:
                conversation.invoke_model(
                    turn,
                    lambda: _call(),
                    new_messages=[HumanMessage(content="older question")],
                    exchange_id="older",
                    check_stopped=lambda: None,
                )
                raise RuntimeError("before tool result")
        with opened.frame.open_turn(
            request("recent"), check_stopped=lambda: None
        ) as turn:
            conversation.invoke_model(
                turn,
                lambda: _call(),
                new_messages=[HumanMessage(content="recent question")],
                exchange_id="recent",
                check_stopped=lambda: None,
            )
            conversation.invoke_tool_effect(
                turn,
                lambda: "recent actual tool result",
                name="lookup",
                arguments={"value": 1},
                model_exchange_id="recent",
                call_index=0,
                tool_call_id="reused-provider-id",
                check_stopped=lambda: None,
            )
            conversation.invoke_model(
                turn,
                lambda: _call(name="ScalarWrapper"),
                new_messages=[],
                purpose="structured_native",
                check_stopped=lambda: None,
            )
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            before = turn.view
            model = _Model([AIMessage(content="older facts")])
            messages = conversation.prepare_history(
                turn, model, check_stopped=lambda: None
            )
            assert "older question" in str(
                model.calls[0]
            ) and "recent question" not in str(model.calls[0])
            assert "outcome unknown" in str(model.calls[0]) and "failed" in str(
                model.calls[0]
            )
            assert [
                message.content
                for message in messages
                if isinstance(message, ToolMessage)
            ] == ["recent actual tool result"]
            assert any(
                isinstance(message, HumanMessage) and "structured" in message.content
                for message in messages
            )
            assert turn.view == before


def test_empty_crash_status_selected_and_active_current_not_unknown():
    memory = SessionMemory(history_budget_chars=1, recent_turns=1)
    _seed(memory)
    with harness(memory) as opened:
        with opened.frame.open_turn(request("crashed"), check_stopped=lambda: None):
            saved = snapshot(memory)
    restored = SessionMemory(history_budget_chars=1, recent_turns=1)
    with restored.open_run(snapshot=saved) as session:
        frame = session.open_frame(MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request("current"), check_stopped=lambda: None) as turn:
            messages = conversation.prepare_history(
                turn, _Model([AIMessage(content="facts")]), check_stopped=lambda: None
            )
            statuses = [
                message.content
                for message in messages
                if "Session history status" in str(message.content)
            ]
            assert len(statuses) == 1 and statuses[0].count("completion unknown") == 1
            assert "failed" not in statuses[0] and "interrupted" not in statuses[0]
            assert conversation.current_history_messages(turn) == []


@pytest.mark.parametrize("corruption", ["missing", "ambiguous"])
def test_exchange_attribution_faults_do_not_call_summary_model(corruption):
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    with harness(memory) as opened:
        with opened.frame.open_turn(request("old"), check_stopped=lambda: None) as turn:
            if corruption == "missing":
                ordinal = turn.request(
                    "tool",
                    {},
                    items=(
                        conversation.encode_message(
                            HumanMessage(content="orphan"), "missing"
                        ),
                    ),
                )
                turn.observe(ordinal, "completed")
            else:
                for _ in range(2):
                    conversation.invoke_model(
                        turn,
                        lambda: AIMessage(content="reply"),
                        new_messages=[],
                        exchange_id="reused-exchange",
                        check_stopped=lambda: None,
                    )
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            model = _Model([])
            with pytest.raises(_CapabilityFault) as caught:
                conversation.prepare_history(turn, model, check_stopped=lambda: None)
            assert isinstance(caught.value.original, ValueError) and model.calls == []


def test_old_turn_without_summary_capability_preserves_full_context():
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            legacy = SimpleNamespace(view=turn.view)
            model = _Model([])
            assert conversation.prepare_history(
                legacy, model, check_stopped=lambda: None
            ) == conversation.history_messages(legacy)
            assert not any(
                "completion unknown" in str(message.content)
                for message in conversation.history_messages(legacy)
            )
            assert model.calls == []


@pytest.mark.parametrize(
    "bad", [("summary", "not JSON"), ("", "[]"), (None, "{}"), (None, "[ ]")]
)
def test_backend_projection_faults_remain_capability_faults(bad):
    turn = SimpleNamespace(prepare_history=lambda **kwargs: bad)
    with pytest.raises(_CapabilityFault):
        conversation.prepare_history(turn, _Model([]), check_stopped=lambda: None)


def test_stop_after_summary_preserves_actual_cache_without_answer(monkeypatch):
    memory = SessionMemory(history_budget_chars=1, recent_turns=1)
    _seed(memory, count=2)
    stop = Event()

    def reply(messages):
        stop.set()
        return AIMessage(content="actual summary before stop")

    model = _Model([reply])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    primary = _HostStop(primary=False)

    def check():
        if stop.is_set():
            raise primary

    before = snapshot(memory)["conversation"]
    with pytest.raises(_HostStop) as caught:
        with harness(memory) as opened:
            with opened.frame.open_turn(
                request("current"), check_stopped=check
            ) as turn:
                plain(
                    AgentRunContext(
                        node_id="agent",
                        prompt="current",
                        memory_context=MemoryContext(transcript=turn),
                        stop_event=stop,
                    )
                )
    assert caught.value is primary and len(model.calls) == 1
    after = snapshot(memory)["conversation"]
    assert after["items"] == before["items"] and after["effects"] == before["effects"]
    assert after["closed_invocations"][-1]["status"] == "interrupted"
    assert next(iter(memory._summaries.values())).text == "actual summary before stop"


def test_valid_text_blocks_are_kept_intact_without_hard_truncation():
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    model = _Model(
        [
            AIMessage(
                content=[
                    {"type": "text", "text": "A long "},
                    {"type": "text", "text": "valid summary"},
                ]
            )
        ]
    )
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            messages = conversation.prepare_history(
                turn, model, check_stopped=lambda: None
            )
            assert "A long valid summary" in messages[0].content
            assert (
                conversation.prepare_history(
                    turn, _Model([]), check_stopped=lambda: None
                )
                == messages
            )


@pytest.mark.parametrize("native", [False, True])
def test_structured_correction_attempts_are_one_whole_recent_turn(monkeypatch, native):
    memory = SessionMemory(history_budget_chars=1, recent_turns=1)
    _seed(memory)
    model = (
        _Native(["bad", 7])
        if native
        else _Model(
            [AIMessage(content='{"value":"bad"}'), AIMessage(content='{"value":7}')]
        )
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("structured"), check_stopped=lambda: None
        ) as turn:
            assert (
                plain(
                    AgentRunContext(
                        node_id="agent",
                        prompt="structured question",
                        memory_context=MemoryContext(transcript=turn),
                        output_type=Type.scalar(ValueKind.INTEGER),
                        llm_config=_config(native),
                    )
                ).value
                == 7
            )
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            summary_model = _Model([AIMessage(content="old facts")])
            messages = conversation.prepare_history(
                turn, summary_model, check_stopped=lambda: None
            )
            assert "question-0" in str(summary_model.calls[0])
            assert "structured question" not in str(summary_model.calls[0])
            assert "structured question" in [message.content for message in messages]
            assert "previous output was invalid" in str(messages)
            assert "bad" in str(messages) and "7" in str(messages)
            assert not any(getattr(message, "tool_calls", None) for message in messages)
