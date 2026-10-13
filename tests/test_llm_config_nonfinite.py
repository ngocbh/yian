"""A NaN or infinite `llm_config` value is rejected wherever a config is built."""

import math

import pytest
from pydantic import ValidationError

from yian import load_flow, run_flow
from yian.compose import LoadError
from yian.llm_clients.config import LLMConfig


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "YIAN_DEFAULT_PROVIDER",
        "YIAN_DEFAULT_MODEL",
        "OPENAI_BASE_URL",
        "VLLM_API_KEY",
        "VLLM_BASE_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


@pytest.mark.parametrize(
    "temperature", [math.nan, math.inf, "inf"], ids=["nan", "inf", "inf-string"]
)
def test_llm_config_rejects_a_nonfinite_temperature(temperature):
    with pytest.raises(ValidationError):
        LLMConfig(temperature=temperature)


_AGENT_LLM_CONFIG = """
id: nonfinite
name: nonfinite
nodes:
  writer:
    kind: agent
    llm_config: {provider: ollama, model: test, temperature: .nan}
    prompt: go
output: ${writer.output}
"""

_FLOW_LLM_CONFIG = """
id: nonfinite
name: nonfinite
llm_config: {provider: ollama, model: test, temperature: .inf}
nodes:
  writer:
    kind: agent
    prompt: go
output: ${writer.output}
"""

_DEF_LLM_CONFIG = """
id: nonfinite
name: nonfinite
defs:
  body:
    llm_config: {provider: ollama, model: test, temperature: -.inf}
    nodes:
      writer:
        kind: agent
        prompt: go
    output: ${writer.output}
nodes:
  k:
    kind: call
    call: body
output: ${k.output}
"""


@pytest.mark.parametrize(
    "source",
    [_AGENT_LLM_CONFIG, _FLOW_LLM_CONFIG, _DEF_LLM_CONFIG],
    ids=["agent", "flow", "def"],
)
def test_a_nonfinite_authored_temperature_is_a_load_error(source):
    with pytest.raises(LoadError) as caught:
        load_flow(source)

    assert "temperature" in str(caught.value)
    assert "finite" in str(caught.value)


def test_a_nonfinite_run_temperature_is_rejected_before_the_run_starts(tmp_path):
    flow = load_flow("""
id: nonfinite
name: nonfinite
nodes:
  writer:
    kind: agent
    prompt: go
output: ${writer.output}
""")
    log = tmp_path / "run.jsonl"

    with pytest.raises(ValueError, match=r"(?s)^Agent 'writer': .*temperature.*finite"):
        run_flow(
            flow,
            {},
            llm_config={"provider": "openai", "model": "gpt-test", "temperature": math.inf},
            log=log,
        )

    assert not log.exists()
