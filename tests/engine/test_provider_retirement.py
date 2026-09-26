"""Retired native settings fail before a flow can perform external work."""

import pytest

from agent_composer.compose import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.llm_clients.config import LLMConfig, SUPPORTED_LLM_PROVIDERS
from agent_composer.llm_clients.factory import create_llm_client
from agent_composer.nodes.agent import AgentNode


@pytest.mark.parametrize(
    "selection",
    [
        {"provider": "codex", "model": "old-model"},
        {"provider": "openai", "cwd": "."},
        {"sandbox": "read-only"},
    ],
)
def test_retired_native_config_points_to_unified_auth(selection):
    with pytest.raises(ValueError, match="ac auth openai"):
        LLMConfig(**selection)


def test_codex_is_not_a_separate_provider_route():
    assert "codex" not in SUPPORTED_LLM_PROVIDERS
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client("codex", "old-model")


def test_retired_tool_marker_is_not_an_implicit_tool_grant():
    with pytest.raises(ValueError, match="grant registered Composer tools explicitly"):
        AgentNode("agent", tools=["builtin_tools"])


def test_old_native_yaml_fails_during_loading():
    with pytest.raises(LoadError, match="ac auth openai"):
        load_flow("""id: old
name: old
nodes:
  agent:
    kind: agent
    prompt: hi
    llm_config: {provider: codex, model: old-model}
output: ${agent.output}
""")
