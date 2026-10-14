"""The ``google`` block reaches the Gemini generateContent request body as written."""

import json

import httpx
import pytest
from google.genai._api_client import SyncHttpxClient
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from yian.llm_clients.factory import model_from_config

_REPLY = {
    "candidates": [{
        "content": {"role": "model", "parts": [{"text": "{\"answer\": \"ok\"}"}]},
        "finishReason": "STOP",
        "index": 0,
    }],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


@pytest.fixture
def bodies(tmp_path, monkeypatch):
    """Answer every Gemini request with one reply and record each request body."""
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    recorded = []

    def send(self, request, **kwargs):
        recorded.append(json.loads(request.content))
        if ":streamGenerateContent" in request.url.path:
            return httpx.Response(200, text=f"data: {json.dumps(_REPLY)}\n\n", request=request)
        return httpx.Response(200, json=_REPLY, request=request)

    monkeypatch.setattr(SyncHttpxClient, "send", send)
    return recorded


def _model(model, block=None):
    config = {"provider": "google", "model": model, "api_key": "k"}
    if block is not None:
        config["google"] = block
    return model_from_config(config)


@pytest.mark.parametrize(
    "model, thinking",
    [
        ("gemini-3-pro-preview", {"thinkingLevel": "HIGH"}),
        ("gemini-2.5-flash", {"thinkingBudget": 1024}),
    ],
)
def test_thinking_config_is_sent_as_written(bodies, model, thinking):
    _model(model, {"generationConfig": {"thinkingConfig": thinking}}).invoke([HumanMessage("hi")])

    assert bodies[-1]["generationConfig"]["thinkingConfig"] == thinking


def test_no_temperature_is_sent_without_one(bodies):
    _model("gemini-2.5-flash").invoke([HumanMessage("hi")])

    assert "temperature" not in bodies[-1].get("generationConfig", {})


def test_safety_settings_pass_through(bodies):
    safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}]

    _model("gemini-2.5-flash", {"safetySettings": safety}).invoke([HumanMessage("hi")])

    assert bodies[-1]["safetySettings"] == safety


def test_streaming_carries_the_block(bodies):
    block = {"generationConfig": {"thinkingConfig": {"thinkingLevel": "LOW"}, "maxOutputTokens": 64}}

    chunks = list(_model("gemini-3-flash-preview", block).stream([HumanMessage("hi")]))

    assert chunks
    assert bodies[-1]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert bodies[-1]["generationConfig"]["maxOutputTokens"] == 64


class _Answer(BaseModel):
    answer: str


def test_structured_output_keeps_the_schema_beside_block_fields(bodies):
    model = _model("gemini-2.5-flash", {"generationConfig": {"temperature": 0.2}})

    result = model.with_structured_output(_Answer).invoke([HumanMessage("hi")])

    assert result == _Answer(answer="ok")
    config = bodies[-1]["generationConfig"]
    assert config["temperature"] == 0.2
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"] == {"answer": {"title": "Answer", "type": "string"}}


def test_block_merges_into_a_callers_extra_body(bodies):
    model = _model("gemini-2.5-flash", {"generationConfig": {"topK": 5}})
    caller = {"extra_body": {"generationConfig": {"seed": 7}}}

    model.invoke([HumanMessage("hi")], http_options=caller)

    assert bodies[-1]["generationConfig"]["topK"] == 5
    assert bodies[-1]["generationConfig"]["seed"] == 7
    assert caller == {"extra_body": {"generationConfig": {"seed": 7}}}
