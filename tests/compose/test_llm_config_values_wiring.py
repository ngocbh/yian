"""An AGENT whose `llm_config` holds values gets them as the reserved `__llm_config` input."""

from __future__ import annotations

import json

import pytest

from yian.compile.expand import clone_child
from yian.compile.manifest import flow_fingerprint, flow_manifest
from yian.compose.loader import load_flow
from yian.nodes.agent.llm_values import LLM_CONFIG_KEY
from yian.nodes.contracts import Binding, ValueTemplate, iter_source_strings
from yian.run import run_flow


def _compiled(text: str):
    return load_flow(text, source_label="f.yaml").compiled


def _llm_edges(flow, consumer: str) -> set[str]:
    """The producers of `consumer`'s edges that feed its reserved `llm_config` input."""
    return {
        edge.from_
        for edge in flow.edges
        if edge.to == consumer and (edge.input_group or "").startswith(LLM_CONFIG_KEY)
    }


_NODE_LEVEL = """\
id: f
name: f
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: '${pick.output}', provider: openai}
output: ${chat.output}
"""


def test_a_node_level_value_gives_an_edge_from_its_producer():
    flow = _compiled(_NODE_LEVEL)
    assert _llm_edges(flow, "chat") == {"pick"}
    assert flow.nodes["chat"].reserved_wiring_keys() == {LLM_CONFIG_KEY}
    binding = flow.wiring["chat"][LLM_CONFIG_KEY]
    assert isinstance(binding, Binding) and isinstance(binding.source, ValueTemplate)
    assert binding.source.value == {"node": {"model": "${pick.output}", "provider": "openai"}}


_FLOW_LEVEL = """\
id: f
name: f
llm_config: {model: '${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    llm_config: {model: m0}
  open:
    kind: agent
    input: {x: '${pick.output}'}
    prompt: Go ${x}.
  fixed:
    kind: agent
    input: {x: '${open.output}'}
    prompt: Go ${x}.
    llm_config: {model: m1}
output: ${fixed.output}
"""


def test_a_flow_level_value_reaches_every_agent_it_can_contribute_to():
    flow = _compiled(_FLOW_LEVEL)
    assert _llm_edges(flow, "open") == {"pick"}
    assert flow.wiring["open"][LLM_CONFIG_KEY].source.value == {
        "node": {}, "flow": {"model": "${pick.output}"}
    }
    # Their own literal `model` always overrides the flow's value.
    for agent in ("pick", "fixed"):
        assert _llm_edges(flow, agent) == set()
        assert LLM_CONFIG_KEY not in flow.wiring[agent]
        assert flow.nodes[agent].reserved_wiring_keys() == set()


def test_an_agent_with_only_literal_layers_has_no_reserved_key():
    flow = _compiled(
        _NODE_LEVEL.replace("'${pick.output}'", "m2")
    )
    assert LLM_CONFIG_KEY not in flow.wiring["chat"]
    assert flow.nodes["chat"].reserved_wiring_keys() == set()


def test_an_inherit_false_agent_does_not_carry_the_flow_layer():
    flow = _compiled(
        _FLOW_LEVEL.replace("    prompt: Go ${x}.\n  fixed:", "    prompt: Go ${x}.\n"
                            "    llm_config: {inherit: false}\n  fixed:")
    )
    assert LLM_CONFIG_KEY not in flow.wiring["open"]
    assert _llm_edges(flow, "open") == set()


def test_wiring_keys_equal_params_plus_the_reserved_key():
    flow = _compiled(_FLOW_LEVEL)
    node = flow.nodes["open"]
    expected = {parameter.name for parameter in node.params} | node.reserved_wiring_keys()
    assert set(flow.wiring["open"]) == expected == {"x", LLM_CONFIG_KEY}


_DEF_CALL = """\
id: f
name: f
defs:
  child:
    input: {m: !str }
    nodes:
      a:
        kind: agent
        prompt: Go.
        llm_config: {model: '${input.m}'}
    output: ${a.output}
nodes:
  c:
    kind: call
    call: child
    input: {m: picked}
output: ${c.output}
"""


def test_clone_child_renamespaces_the_reserved_key():
    flow = _compiled(_DEF_CALL)
    child = flow.nodes["c"].child
    assert child.wiring["a"][LLM_CONFIG_KEY].source.value == {"node": {"model": "${input.m}"}}
    cloned = clone_child(child, callsite="c", record={"m": "picked"})
    texts = [text for _, text in iter_source_strings(cloned.wiring["c/a"][LLM_CONFIG_KEY])]
    assert texts == ["${c/__start__.output.m}"]


def test_two_loads_of_the_same_text_have_the_same_fingerprint():
    assert flow_fingerprint(_compiled(_FLOW_LEVEL)) == flow_fingerprint(_compiled(_FLOW_LEVEL))
    assert flow_fingerprint(_compiled(_NODE_LEVEL)) == flow_fingerprint(_compiled(_NODE_LEVEL))


_DROP = """\
id: f
name: f
input: {m: !str , n: !Optional(str) }
llm_config: {model: '${input.m}'}
nodes:
  a:
    kind: agent
    prompt: Go.
    llm_config: {model: NODE_MODEL}
output: ${a.output}
"""


def test_changing_which_lower_value_is_dropped_changes_the_wiring_identity():
    # A higher value that can't be null drops the flow's `model`; one that can, keeps it.
    shielded = _compiled(_DROP.replace("NODE_MODEL", "!str '${input.n}'"))
    open_ = _compiled(_DROP.replace("NODE_MODEL", "!Optional(str) '${input.n}'"))
    assert shielded.wiring["a"][LLM_CONFIG_KEY].source.value.get("flow", {}) == {}
    assert open_.wiring["a"][LLM_CONFIG_KEY].source.value["flow"] == {"model": "${input.m}"}
    assert flow_manifest(shielded)["wiring"]["a"] != flow_manifest(open_)["wiring"]["a"]
    assert flow_fingerprint(shielded) != flow_fingerprint(open_)


_RUN = """\
id: f
name: f
input: {m: !str }
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: openai, model: '${input.m}'}
output: ${chat.output}
"""


@pytest.mark.xfail(strict=True, reason="input-only values are not resolved at run start yet")
def test_runs_with_different_inputs_record_the_same_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class _Model:
        def invoke(self, messages):
            class _Reply:
                content = "ok"

            return _Reply()

    monkeypatch.setattr("yian.llm_clients.model_from_config", lambda cfg: _Model())
    fingerprints = []
    for index, model in enumerate(("gpt-a", "gpt-b")):
        log = tmp_path / f"run{index}.jsonl"
        result = run_flow(load_flow(_RUN), {"m": model}, log=log)
        assert result.status == "completed"
        started = json.loads(log.read_text().splitlines()[0])
        fingerprints.append(started["payload"]["flow_fingerprint"])
    assert fingerprints[0] == fingerprints[1]
