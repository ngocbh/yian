"""Structured AGENT output end-to-end through `run_flow`.

Tool-free agents declaring record and `Literal[...]` outputs emit structured values via the
engine's structured-generation path; both pass the write boundary
(`pool.set(..., declared=output_type)`). A sibling bare-`str` agent in the same flow
confirms the text passthrough is untouched.
"""

import yian.llm_clients as llm_clients_mod
from yian.compose.loader import load_flow
from yian.run import run_flow

# `rec` declares a record output -> structured generation; `txt` stays a text producer.
_FLOW = """
id: top
name: top
nodes:
  rec:
    kind: agent
    prompt: extract a person
    output:
      name: str
      score: int
  txt:
    kind: agent
    depends_on: [rec]
    prompt: say hi
    output: str
  sentiment:
    kind: agent
    depends_on: [txt]
    prompt: classify the greeting
    output: Literal[positive, neutral, negative]
output:
  person: ${rec.output}
  greeting: ${txt.output}
  sentiment: ${sentiment.output}
"""


def test_structured_output_end_to_end(monkeypatch):
    def fake_model_from_config(cfg):
        class _M:
            def invoke(self, msgs):  # the bare-str (txt) path
                class R:
                    content = "hello there"

                return R()

            def with_structured_output(self, schema):  # record and Literal paths
                class _Bound:
                    def invoke(self, msgs):
                        payload = (
                            {"value": "neutral"}
                            if "value" in schema.model_fields
                            else {"name": "Ada", "score": 9}
                        )
                        return schema.model_validate(payload)

                return _Bound()

        return _M()

    monkeypatch.setattr(llm_clients_mod, "model_from_config", fake_model_from_config)

    loaded = load_flow(_FLOW)
    res = run_flow(loaded, {})
    assert res.status == "succeeded"
    # Structured values crossed the write boundary, while bare-str text stayed unchanged.
    assert res.output == {
        "person": {"name": "Ada", "score": 9},
        "greeting": "hello there",
        "sentiment": "neutral",
    }
