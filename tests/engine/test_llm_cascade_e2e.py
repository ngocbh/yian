"""End-to-end cascade: a 2-level flow run through `run_flow` with a CLI layer. The
per-agent effective `llm_config` each agent builds its model from must reflect the full
authored ladder (node → flow → parent → CLI), and an `inherit: false` agent must skip
those authored outer layers while retaining process provider/model defaults."""

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.compose.loader import load_flow
from agent_composer.run import run_flow

# Top flow sets provider+temperature; calls `sub`. Inside `sub`, `inner` overrides only
# `model` (so it fills provider/temperature from the flow and the CLI fields), while
# `solo` skips authored outer layers with inherit:false (its own dict already supplies
# both process-defaulted selection fields in this case).
_FLOW = """
id: top
name: top
llm_config: {provider: anthropic, temperature: 0.2}
defs:
  sub:
    nodes:
      inner: {kind: agent, mode: plain, prompt: hi, llm_config: {model: claude-opus-4-8}}
      solo:
        kind: agent
        mode: plain
        depends_on: [inner]
        prompt: hi
        llm_config: {provider: openai, model: gpt-5.5, inherit: false}
    output: ${solo.output}
nodes:
  c: {kind: call, call: sub}
output: ${c.output}
"""


def test_cascade_end_to_end_with_cli_layer(monkeypatch):
    # Capture the effective config every agent builds its model from, keyed by the model
    # field (unique per agent here) so the assertions don't depend on call order.
    seen: list = []

    def fake_model_from_config(cfg):
        seen.append(dict(cfg))

        class _M:
            def invoke(self, msgs):
                class R:
                    content = "ok"

                return R()

        return _M()

    monkeypatch.setattr(llm_clients_mod, "model_from_config", fake_model_from_config)

    loaded = load_flow(_FLOW)
    res = run_flow(loaded, {}, llm_config={"max_tokens": 256})
    assert res.status == "succeeded"

    by_model = {c["model"]: c for c in seen}
    # inner: own model + flow provider/temperature + CLI max_tokens
    assert by_model["claude-opus-4-8"] == {
        "model": "claude-opus-4-8",
        "provider": "anthropic",
        "temperature": 0.2,
        "max_tokens": 256,
    }
    # solo: inherit:false — own complete selection, no authored flow/CLI layers
    assert by_model["gpt-5.5"] == {"provider": "openai", "model": "gpt-5.5"}
