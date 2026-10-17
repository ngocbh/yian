"""An AGENT whose `llm_config` holds values gets them as the reserved `__llm_config` input."""

from __future__ import annotations

import json

import pytest

from yian.compile.expand import clone_child
from yian.compile.manifest import flow_fingerprint, flow_manifest
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.compose.validate.skips import SkipModel
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


_SELF = """\
id: f
name: f
input: {m: !str , n: !Optional(str) }
llm_config: {model: '${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    llm_config: {model: PICK_MODEL}
  chat:
    kind: agent
    prompt: Go.
output: ${chat.output}
"""


@pytest.mark.parametrize("pick_model", ["m0", "'${input.m}'", "'x${input.n}'"])
def test_a_flow_value_read_from_an_agent_that_always_sets_the_field_is_not_a_cycle(
    pick_model,
):
    flow = _compiled(_SELF.replace("PICK_MODEL", pick_model))
    assert _llm_edges(flow, "chat") == {"pick"}
    assert "pick" not in _llm_edges(flow, "pick")


def test_a_flow_value_read_from_an_agent_whose_own_value_may_be_null_is_a_cycle():
    with pytest.raises(LoadError, match="cycle"):
        _compiled(_SELF.replace("PICK_MODEL", "'${input.n}'"))


_CASE = """\
id: f
name: f
input: {go: !bool }
LAYER
nodes:
  gate:
    kind: case
    cases:
      - when: ${input.go}
        then: pick
    else: other
  pick:
    kind: code
    output: str
    code: |
      return "gpt-x"
  other:
    kind: code
    output: str
    code: |
      return "no"
  chat:
    kind: agent
    prompt: Go.
    CHAT
output: ${chat.output}
"""


def _skip_model(flow) -> SkipModel:
    return SkipModel(set(flow.nodes), flow.edges, set())


def test_an_llm_config_value_reading_a_skippable_node_lets_the_agent_skip():
    flow = _compiled(_CASE.replace("LAYER", "").replace(
        "CHAT", "llm_config: {model: '${pick.output}'}"
    ))
    assert _skip_model(flow).may_all_skip({"chat"})


def test_a_shielded_flow_value_reading_a_skippable_node_does_not():
    flow = _compiled(_CASE.replace("LAYER", "llm_config: {model: '${pick.output}'}").replace(
        "CHAT", "llm_config: {model: m1}"
    ))
    assert _llm_edges(flow, "chat") == set()
    assert not _skip_model(flow).may_all_skip({"chat"})


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="f.yaml")
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


def _line_of(text: str, needle: str) -> int:
    return next(n for n, line in enumerate(text.splitlines(), 1) if needle in line)


def test_an_unknown_node_in_a_value_is_named_at_the_llm_config_field():
    text = _NODE_LEVEL.replace("'${pick.output}'", "'${nope.output}'")
    error = _load_error(text)
    assert "\n  node 'chat' llm_config: reference ${nope.output}" in error.diagnostic.message
    assert _compiled(_NODE_LEVEL).nodes["chat"].binding_labels[LLM_CONFIG_KEY] == (
        "node 'chat' llm_config"
    )
    assert _line(error) == _line_of(text, "${nope.output}")


def test_an_unknown_input_in_a_flow_value_is_located_at_the_flow_field():
    text = _FLOW_LEVEL.replace("'${pick.output}'}\nnodes", "'${input.nope}'}\nnodes")
    error = _load_error(text)
    assert "\n  node 'open' llm_config: reference ${input.nope}" in error.diagnostic.message
    assert _line(error) == _line_of(text, "${input.nope}")


_ASK = """\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '${input.proj}'}
    adaptive_questions:
      prompt: "Q ${proj}"
      llm_config: {model: '${nope.output}'}
output: ${ask.output}
"""


def test_a_composer_value_error_names_the_gate_field():
    error = _load_error(_ASK)
    assert (
        "\n  node 'ask' adaptive_questions.llm_config: reference ${nope.output}"
        in error.diagnostic.message
    )
    assert _line(error) == _line_of(_ASK, "${nope.output}")
    composer = _compiled(_ASK.replace("${nope.output}", "${input.proj}"))
    assert composer.nodes["ask__compose"].binding_labels[LLM_CONFIG_KEY] == (
        "node 'ask' adaptive_questions.llm_config"
    )


def test_skipped_arithmetic_in_a_value_names_the_llm_config_field():
    text = _CASE.replace("LAYER", "").replace(
        "CHAT", """llm_config: {model: '${pick.output + "x"}'}"""
    )
    error = _load_error(text)
    assert error.diagnostic.message.startswith(
        "node 'chat' llm_config: ${pick.output + \"x\"} does arithmetic on pick.output"
    ), error
    assert _line(error) == _line_of(text, "pick.output + ")


def test_an_authored_input_named_like_the_reserved_key_fails():
    text = _NODE_LEVEL.replace(
        "    prompt: Go.\n", "    input: {__llm_config: '${pick.output}'}\n    prompt: Go.\n"
    )
    error = _load_error(text)
    assert "'__llm_config' is reserved" in error.diagnostic.message, error
    assert error.diagnostic.message.startswith("node 'chat'")
