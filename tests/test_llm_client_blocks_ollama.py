"""The ``ollama`` block reaches the native /api/chat request as written, with no default ``think``."""

import json

import httpx
import ollama
import pytest
from langchain_core.messages import HumanMessage

from yian.llm_clients.factory import model_from_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OLLAMA_BASE_URL", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _call(config, message=None, **invoke_kwargs):
    """Invoke a model built from ``config`` against a fake Ollama; return (request JSON, reply)."""
    message = message or {"role": "assistant", "content": "ok"}
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        line = {"model": "qwen3", "message": message, "done": True, "done_reason": "stop"}
        return httpx.Response(200, content=json.dumps(line).encode() + b"\n")

    model = model_from_config({"provider": "ollama", "model": "qwen3", **config})
    model._client = ollama.Client(
        host="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    reply = model.invoke([HumanMessage("hi")], **invoke_kwargs)
    assert len(sent) == 1
    return sent[0], reply


def test_no_block_sends_no_think():
    body, _ = _call({})
    assert "think" not in body


def test_unset_block_builds_no_reasoning():
    model = model_from_config({"provider": "ollama", "model": "qwen3"})
    assert model.reasoning is None
    # `None` here is dropped by `ollama.Client.chat`; the wire test above checks that.
    assert model._chat_params([HumanMessage("hi")])["think"] is None


@pytest.mark.parametrize("think", [True, False, "high"])
def test_think_is_sent_as_written(think):
    body, _ = _call({"ollama": {"think": think}})
    assert body["think"] == think


def test_options_and_keep_alive_are_sent_as_written():
    body, _ = _call(
        {"ollama": {"options": {"num_ctx": 8192, "temperature": 0.2}, "keep_alive": "5m"}}
    )
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.2}
    assert body["keep_alive"] == "5m"


def test_block_options_merge_into_langchain_options():
    # langchain puts the call's stop words into `options`; the block's options join them.
    body, _ = _call({"ollama": {"options": {"num_ctx": 8192}}}, stop=["END"])
    assert body["options"] == {"num_ctx": 8192, "stop": ["END"]}


def test_logprobs_are_sent_as_written():
    body, _ = _call({"ollama": {"logprobs": True, "top_logprobs": 3}})
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 3


def test_thinking_reply_keeps_only_the_answer_in_content():
    body, reply = _call(
        {"ollama": {"think": True}},
        message={"role": "assistant", "thinking": "let me think", "content": "42"},
    )
    assert body["think"] is True
    assert reply.content == "42"


def test_block_is_not_mutated_or_shared():
    block = {"options": {"num_ctx": 8192}}
    model = model_from_config({"provider": "ollama", "model": "qwen3", "ollama": block})
    params = model._chat_params([HumanMessage("hi")], stop=["END"])
    assert params["options"] == {"num_ctx": 8192, "stop": ["END"]}
    assert block == {"options": {"num_ctx": 8192}}
    assert model.block == {"options": {"num_ctx": 8192}}


def test_block_stays_out_of_repr_and_dump():
    model = model_from_config({"provider": "ollama", "model": "qwen3", "ollama": {"think": True}})
    assert "block" not in repr(model)
    assert "block" not in model.model_dump()
