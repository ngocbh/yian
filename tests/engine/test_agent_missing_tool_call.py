"""An AGENT retries a structured reply that skips the schema's tool, then fails naming it."""

import warnings

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.memory.session import SessionMemory
from yian.nodes.agent.structured import type_to_schema
from yian.run import run_flow
from yian.typesys.values import Type, ValueKind
from tests.engine.test_agent_session import _FLOW
from tests.test_llm_client_blocks_anthropic import _model, _tool_reply, server  # noqa: F401


_NAME = type_to_schema(Type.scalar(ValueKind.INTEGER)).__name__
_MESSAGE = f"the reply did not call the {_NAME} tool; call it with the answer"
_SESSION_LINE = "    long_term_memory: session\n"
_TEXT = [{"type": "text", "text": "7"}]


def _flow(retries, session, llm_config="{provider: anthropic, model: claude-opus-5-5}"):
    text = (_FLOW.replace("{provider: openai, model: test-model}", llm_config)
            .replace("    output: str", f"    retries: {retries}\n    output: int"))
    assert llm_config in text and "output: int" in text
    return load_flow(text if session else text.replace(_SESSION_LINE, ""))


def _run(flow, session):
    return run_flow(flow, {"question": "q"}, memory={"session": SessionMemory()} if session else None)


def _anthropic(monkeypatch):
    # A distinctive key: the run scrubs the key from failure messages.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    model = _model(model="claude-opus-5-5")
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)


def _texts(message):
    content = message["content"]
    return [content] if isinstance(content, str) else [b["text"] for b in content if b["type"] == "text"]


@pytest.mark.parametrize("session", [True, False])
def test_an_agent_retries_a_reply_without_the_tool_call(server, monkeypatch, session):
    _anthropic(monkeypatch)
    server.replies = [_TEXT, _tool_reply(_NAME, {"value": 7})]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(_flow(2, session), session)
    assert result.status == "succeeded", result.failure
    assert result.output == 7 and len(server.bodies) == 2
    messages = server.bodies[1]["messages"]
    assert [m["role"] for m in messages] == ["user"]
    correction = _texts(messages[0])[-1]
    assert _MESSAGE in correction
    assert "provider error (" not in correction and "For troubleshooting" not in correction
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


@pytest.mark.parametrize("session", [True, False])
@pytest.mark.parametrize("retries", [0, 2])
def test_an_agent_fails_naming_the_tool_once_retries_are_spent(server, monkeypatch, session, retries):
    _anthropic(monkeypatch)
    server.content = _TEXT
    result = _run(_flow(retries, session), session)
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "node.execute" and failure.node_id == "agent"
    assert f"the reply did not call the {_NAME} tool" in failure.message
    assert failure.trace
    assert failure.cause_type == "MissingToolCallError"
    assert len(server.bodies) == retries + 1


class _NoResult:
    """A native structured model whose reply never calls the schema's tool."""

    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, *, include_raw=False):
        outer = self

        class Bound:
            def invoke(self, messages):
                outer.calls.append(list(messages))
                if not include_raw:
                    return None
                return {"raw": AIMessage(content="7"), "parsed": None, "parsing_error": None}

        return Bound()


@pytest.mark.parametrize("session", [True, False])
def test_a_generic_agent_fails_naming_the_tool_when_the_reply_has_no_result(monkeypatch, session):
    model = _NoResult()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    result = _run(_flow(0, session, "{provider: openai, model: gpt-4o}"), session)
    assert result.status == "failed"
    assert result.failure.code == "node.execute" and result.failure.node_id == "agent"
    assert _MESSAGE in result.failure.message
    assert result.failure.cause_type == "MissingToolCallError"
    assert len(model.calls) == 1
