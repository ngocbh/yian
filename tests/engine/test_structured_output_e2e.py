"""Structured AGENT output end-to-end through `run_flow`.

A `plain` agent declaring a record `output:` emits a structured value via the engine's
structured-generation path; the flow output is the plain dict and it passes the write
boundary (`pool.set(..., declared=output_type)`). A sibling bare-`str` agent in the same
flow confirms the text passthrough is untouched.
"""

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.compose.loader import load_flow
from agent_composer.run import run_flow

# `rec` declares a record output -> structured generation; `txt` stays a text producer.
_FLOW = """
id: top
name: top
nodes:
  rec:
    kind: agent
    mode: plain
    prompt: extract a person
    output:
      name: str
      score: int
  txt:
    kind: agent
    mode: plain
    depends_on: [rec]
    prompt: say hi
    output: str
output:
  person: ${rec.output}
  greeting: ${txt.output}
"""


def test_structured_output_end_to_end(monkeypatch):
    def fake_model_from_config(cfg):
        class _M:
            def invoke(self, msgs):  # the bare-str (txt) path
                class R:
                    content = "hello there"

                return R()

            def with_structured_output(self, schema):  # the record (rec) path
                class _Bound:
                    def invoke(self, msgs):
                        return schema.model_validate({"name": "Ada", "score": 9})

                return _Bound()

        return _M()

    monkeypatch.setattr(llm_clients_mod, "model_from_config", fake_model_from_config)

    loaded = load_flow(_FLOW)
    res = run_flow(loaded, {})
    assert res.status == "succeeded"
    # the record is a plain dict (write boundary accepted the structured value), and the
    # bare-str agent's text passthrough is unchanged.
    assert res.output == {"person": {"name": "Ada", "score": 9}, "greeting": "hello there"}
