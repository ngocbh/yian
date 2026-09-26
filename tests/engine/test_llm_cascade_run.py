"""`run_flow` applies the CLI cascade layer at run start (fresh-run path)."""

from agent_composer.compose.loader import load_flow
from agent_composer.run import run_flow
from agent_composer.llm_clients.connection import resolve_model_config


def test_run_flow_applies_cli_llm_config(monkeypatch):
    captured = {}

    def fake_model_from_config(cfg):
        captured["cfg"] = cfg

        class _M:
            def invoke(self, msgs):
                class R:
                    content = "ok"

                return R()

        return _M()

    monkeypatch.setattr(
        "agent_composer.llm_clients.model_from_config", fake_model_from_config
    )
    # mode: plain so the fake model only needs .invoke (the default mode is tool_calling).
    text = (
        "id: f\nname: f\nnodes:\n  a: {kind: agent, mode: plain, prompt: hi}\n"
        "output: ${a.output}\n"
    )
    loaded = load_flow(text)
    run_flow(loaded, {}, llm_config={"provider": "openai", "model": "gpt-5.5"})
    assert captured["cfg"] == resolve_model_config({"provider": "openai", "model": "gpt-5.5"})
