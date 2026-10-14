"""Cascade resolution and replay over stable paths in the static call tree."""

import copy

import pytest

from yian.llm_clients.config import merge_llm_config
from yian.compose.loader import load_flow
from yian.compile.llm_cascade import (
    apply_effective_llm_map,
    resolve_llm_cascade,
)


_DEFAULTS = {"provider": "anthropic", "model": "default-model"}
_PROVIDER_MODELS = {"openai": "openai-default", "google": "google-default"}


def test_merge_specific_wins_per_field():
    specific = {"model": "claude-opus-4-8"}
    base = {"provider": "anthropic", "model": "claude-sonnet-4-5", "temperature": 0.2}
    # specific wins per field; base fills the gaps (provider, temperature)
    assert merge_llm_config(specific, base) == {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "temperature": 0.2,
    }


def test_merge_empty_layers():
    assert merge_llm_config({}, {}) == {}
    assert merge_llm_config({"model": "x"}, {}) == {"model": "x"}
    assert merge_llm_config({}, {"model": "y"}) == {"model": "y"}


def test_merge_treats_none_as_an_unset_field_in_every_layer():
    specific = {
        "provider": None,
        "model": "specific-model",
        "temperature": None,
    }
    base = {
        "provider": None,
        "model": None,
        "temperature": 0.2,
        "anthropic_effort": None,
    }

    assert merge_llm_config(specific, base) == {
        "model": "specific-model",
        "temperature": 0.2,
    }


def test_merge_with_none_is_associative():
    node = {"provider": None, "model": "node-model", "temperature": None}
    flow = {"provider": "anthropic", "model": None, "temperature": 0.4}
    caller = {"provider": "openai", "model": "caller-model", "temperature": None}

    assert merge_llm_config(node, merge_llm_config(flow, caller)) == merge_llm_config(
        merge_llm_config(node, flow), caller
    )


def test_merge_does_not_mutate_inputs():
    a, b = {"model": "x"}, {"provider": "p"}
    merge_llm_config(a, b)
    assert a == {"model": "x"} and b == {"provider": "p"}


def test_node_fills_gap_from_flow_then_cli():
    text = """
id: f
name: f
llm_config: {provider: anthropic}
nodes:
  a: {kind: agent, prompt: hi, llm_config: {model: claude-opus-4-8}}
output: ${a.output}
"""
    loaded = load_flow(text)
    resolve_llm_cascade(loaded.compiled, {"temperature": 0.3}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS)
    a = loaded.compiled.nodes["a"]
    # node sets model; flow fills provider; CLI fills temperature
    assert a.llm_config == {
        "model": "claude-opus-4-8", "provider": "anthropic", "temperature": 0.3
    }


def test_authored_null_fields_fall_through_the_cascade():
    text = """
id: f
name: f
llm_config: {provider: null, model: null, temperature: 0.7}
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: {provider: null, model: null, temperature: null}
output: ${a.output}
"""
    loaded = load_flow(text)
    a = loaded.compiled.nodes["a"]
    assert loaded.compiled.flow_llm_config == {
        "provider": None,
        "model": None,
        "temperature": 0.7,
    }
    assert a.own_llm_config == {
        "provider": None,
        "model": None,
        "temperature": None,
    }

    effective = resolve_llm_cascade(loaded.compiled, {}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS)

    assert effective["a"] == {
        "provider": "anthropic",
        "model": "default-model",
        "temperature": 0.7,
    }
    assert a.llm_config == effective["a"]


def test_inherit_false_node_ignores_authored_inheritance_layers_but_gets_defaults():
    text = """
id: f
name: f
llm_config: {provider: anthropic, temperature: 0.9}
nodes:
  a: {kind: agent, prompt: hi, llm_config: {model: m, inherit: false}}
output: ${a.output}
"""
    loaded = load_flow(text)
    resolve_llm_cascade(loaded.compiled, {"temperature": 0.1}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS)
    assert loaded.compiled.nodes["a"].llm_config == {
        "model": "m",
        "provider": "anthropic",
    }


def test_resolution_is_idempotent_from_own():
    # re-resolving with a different CLI layer recomputes from own_llm_config, not the
    # previously-baked effective dict
    text = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi, llm_config: {model: m}}\noutput: ${a.output}\n"
    loaded = load_flow(text)
    resolve_llm_cascade(
        loaded.compiled, {"provider": "openai"}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS
    )
    resolve_llm_cascade(
        loaded.compiled, {"provider": "google"}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS
    )
    assert loaded.compiled.nodes["a"].llm_config == {
        "model": "m",
        "provider": "google",
    }


def test_cascade_reaches_nested_def_agents():
    text = """
id: f
name: f
llm_config: {provider: anthropic}
defs:
  sub:
    nodes:
      inner: {kind: agent, prompt: hi}
    output: ${inner.output}
nodes:
  c: {kind: call, call: sub}
output: ${c.output}
"""
    loaded = load_flow(text)
    resolve_llm_cascade(
        loaded.compiled, {"model": "claude-opus-4-8"}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS
    )
    call = loaded.compiled.nodes["c"]
    inner = call.child.nodes["inner"]
    # inner sets nothing -> fills from top flow (provider) + CLI (model)
    assert inner.llm_config == {"provider": "anthropic", "model": "claude-opus-4-8"}


def test_named_and_inline_children_receive_the_same_llm_cascade():
    named = load_flow(
        """id: named-cascade
name: named_cascade
llm_config: {provider: openai}
defs:
  child:
    nodes:
      inner:
        kind: agent
        prompt: hi
        llm_config: {temperature: 0.7}
    output: ${inner.output}
nodes:
  invoke: {kind: call, call: child}
output: ${invoke.output}
"""
    )
    inline = load_flow(
        """id: inline-cascade
name: inline_cascade
llm_config: {provider: openai}
nodes:
  invoke:
    kind: call
    nodes:
      inner:
        kind: agent
        prompt: hi
        llm_config: {temperature: 0.7}
    output: ${inner.output}
output: ${invoke.output}
"""
    )

    caller = {"model": "caller-model"}
    named_effective = resolve_llm_cascade(
        named.compiled,
        caller,
        defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS,
    )
    inline_effective = resolve_llm_cascade(
        inline.compiled,
        caller,
        defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS,
    )

    assert inline_effective == named_effective == {
        "invoke/inner": {
            "provider": "openai",
            "model": "caller-model",
            "temperature": 0.7,
        }
    }
    assert (
        inline.compiled.nodes["invoke"].child.nodes["inner"].llm_config
        == named.compiled.nodes["invoke"].child.nodes["inner"].llm_config
    )


def test_nested_re_resolution_recomputes_from_own():
    # the riskier path — resolve_llm_cascade replaces node.child with a fresh deepcopy each
    # call, so re-running over a CALL must still recompute the inner agent from its OWN config
    # (never compound the previously-baked effective dict).
    text = """
id: f
name: f
defs:
  sub:
    nodes:
      inner: {kind: agent, prompt: hi}
    output: ${inner.output}
nodes:
  c: {kind: call, call: sub}
output: ${c.output}
"""
    loaded = load_flow(text)
    resolve_llm_cascade(
        loaded.compiled,
        {"provider": "openai", "model": "x"},
        defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS,
    )
    resolve_llm_cascade(
        loaded.compiled, {"provider": "google"}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS
    )
    inner = loaded.compiled.nodes["c"].child.nodes["inner"]
    # second run dropped `model` (not in the new CLI layer); did not retain it from run 1
    assert inner.llm_config == {"provider": "google", "model": "google-default"}


def test_cascade_reaches_loop_body_agents():
    # A LOOP carries a static child body like CALL/MAP; its body agents must inherit the
    # parent flow/CLI layers too (cascade recurses into any node with a static child flow).
    text = """
id: f
name: f
llm_config: {provider: anthropic}
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes:
      think: {kind: agent, prompt: hi}
      step:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output: ${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 5
output: ${chat_loop.output}
"""
    loaded = load_flow(text)
    resolve_llm_cascade(
        loaded.compiled, {"model": "claude-opus-4-8"}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS
    )
    think = loaded.compiled.nodes["chat_loop"].child.nodes["think"]
    # the body agent sets nothing -> fills from top flow (provider) + CLI (model)
    assert think.llm_config == {"provider": "anthropic", "model": "claude-opus-4-8"}


_PATH_FLOW = """
id: path-flow
name: path_flow
defs:
  child:
    nodes:
      inner: {kind: agent, prompt: inner}
    output: ${inner.output}
nodes:
  outer: {kind: agent, prompt: outer, llm_config: {model: outer-model}}
  invoke: {kind: call, call: child}
output: ${invoke.output}
"""


def test_effective_map_round_trips_static_paths():
    resolved = load_flow(_PATH_FLOW)
    effective = resolve_llm_cascade(
        resolved.compiled,
        {"provider": "openai"},
        defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS,
    )

    assert effective == {
        "outer": {"provider": "openai", "model": "outer-model"},
        "invoke/inner": {"provider": "openai", "model": "openai-default"},
    }
    assert effective["outer"] is not resolved.compiled.nodes["outer"].llm_config
    assert effective["outer"] is not effective["invoke/inner"]
    assert (
        resolved.compiled.nodes["outer"].llm_config
        is not resolved.compiled.nodes["invoke"].child.nodes["inner"].llm_config
    )

    replayed = load_flow(_PATH_FLOW)
    apply_effective_llm_map(replayed.compiled, effective)
    assert replayed.compiled.nodes["outer"].llm_config == effective["outer"]
    assert (
        replayed.compiled.nodes["invoke"].child.nodes["inner"].llm_config
        == effective["invoke/inner"]
    )
    assert replayed.compiled.nodes["outer"].llm_config is not effective["outer"]


def test_effective_map_rejects_missing_or_extra_paths():
    loaded = load_flow(_PATH_FLOW)
    effective = resolve_llm_cascade(
        loaded.compiled, {}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS
    )

    missing = copy.deepcopy(effective)
    missing.pop("invoke/inner")
    with pytest.raises(ValueError, match=r"missing.*invoke/inner"):
        apply_effective_llm_map(load_flow(_PATH_FLOW).compiled, missing)

    extra = copy.deepcopy(effective)
    extra["ghost"] = dict(_DEFAULTS)
    with pytest.raises(ValueError, match=r"extra.*ghost"):
        apply_effective_llm_map(load_flow(_PATH_FLOW).compiled, extra)


_SHARED_LEAF_FLOW = """
id: shared-leaf
name: shared_leaf
defs:
  leaf:
    nodes:
      agent: {kind: agent, prompt: hi}
    output: ${agent.output}
  openai_parent:
    llm_config: {provider: openai}
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: ${leaf_call.output}
  anthropic_parent:
    llm_config: {provider: anthropic}
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: ${leaf_call.output}
nodes:
  left: {kind: call, call: openai_parent}
  right: {kind: call, call: anthropic_parent}
output: ${right.output}
"""


def test_shared_child_is_dealiased_per_parent_provider():
    loaded = load_flow(_SHARED_LEAF_FLOW)
    left_call = loaded.compiled.nodes["left"].child.nodes["leaf_call"]
    right_call = loaded.compiled.nodes["right"].child.nodes["leaf_call"]
    assert left_call.child is right_call.child

    effective = resolve_llm_cascade(loaded.compiled, {}, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS)

    left_call = loaded.compiled.nodes["left"].child.nodes["leaf_call"]
    right_call = loaded.compiled.nodes["right"].child.nodes["leaf_call"]
    assert left_call.child is not right_call.child
    assert effective["left/leaf_call/agent"]["provider"] == "openai"
    assert effective["right/leaf_call/agent"]["provider"] == "anthropic"
    assert left_call.child.nodes["agent"].llm_config["provider"] == "openai"
    assert right_call.child.nodes["agent"].llm_config["provider"] == "anthropic"


def test_effective_map_resolves_environment_defaults_once(monkeypatch):
    import yian.run as run_mod
    import yian.llm_clients.factory as factory_mod

    from yian.llm_clients.connection import connection_snapshot
    calls = []
    monkeypatch.setenv("YIAN_DEFAULT_PROVIDER", "ANTHROPIC")
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "environment-model")

    def snapshot():
        calls.append("snapshot")
        result = connection_snapshot()
        monkeypatch.setenv("YIAN_DEFAULT_MODEL", "later-model")
        return result

    seen = []

    class _Client:
        def get_llm(self):
            return _Model()

    class _Model:
        def invoke(self, messages):
            class _Response:
                content = "ok"

            return _Response()

    def fake_create_llm_client(*, provider, model, **kwargs):
        seen.append({"provider": provider, "model": model})
        return _Client()

    monkeypatch.setattr(run_mod, "connection_snapshot", snapshot)
    monkeypatch.setattr(factory_mod, "create_llm_client", fake_create_llm_client)

    loaded = load_flow(
        "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi, "
        "llm_config: {inherit: false}}\noutput: ${a.output}\n"
    )

    result = run_mod.run_flow(loaded, {})

    assert result.status == "succeeded"
    assert calls == ["snapshot"]
    assert seen == [{"provider": "anthropic", "model": "environment-model"}]


def test_replay_uses_logged_map_after_environment_changes(monkeypatch):
    import yian._settings as settings
    import yian.compile.llm_cascade as cascade_mod

    original = load_flow(_SHARED_LEAF_FLOW)
    recorded = resolve_llm_cascade(
        original.compiled,
        {},
        defaults={"provider": "openai", "model": "recorded-model"},
        provider_models={"anthropic": "recorded-anthropic"},
    )
    monkeypatch.setenv("YIAN_DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "changed-model")

    def unexpected_settings_read():
        raise AssertionError("replay must not read process LLM defaults")

    monkeypatch.setattr(settings, "default_llm_provider", unexpected_settings_read)
    monkeypatch.setattr(settings, "default_llm_model", unexpected_settings_read)
    monkeypatch.setattr(
        cascade_mod,
        "default_llm_provider",
        unexpected_settings_read,
        raising=False,
    )
    monkeypatch.setattr(
        cascade_mod,
        "default_llm_model",
        unexpected_settings_read,
        raising=False,
    )

    replayed = load_flow(_SHARED_LEAF_FLOW)
    left_before = replayed.compiled.nodes["left"].child.nodes["leaf_call"].child
    right_before = replayed.compiled.nodes["right"].child.nodes["leaf_call"].child
    assert left_before is right_before
    apply_effective_llm_map(replayed.compiled, recorded)

    left = replayed.compiled.nodes["left"].child.nodes["leaf_call"].child
    right = replayed.compiled.nodes["right"].child.nodes["leaf_call"].child
    assert left is not right
    assert left.nodes["agent"].llm_config == recorded["left/leaf_call/agent"]
    assert right.nodes["agent"].llm_config == recorded["right/leaf_call/agent"]
    assert left.nodes["agent"].llm_config["provider"] == "openai"
    assert right.nodes["agent"].llm_config["provider"] == "anthropic"


@pytest.mark.parametrize(
    "layer",
    [{"modle": "x"}, {"provider": "openai", "inherit": False}],
)
def test_host_layer_rejects_unknown_fields(layer):
    loaded = load_flow("""
id: f
name: f
nodes:
  a: {kind: agent, prompt: hi}
output: ${a.output}
""")
    with pytest.raises(ValueError, match=r"llm_config has unknown field\(s\)"):
        resolve_llm_cascade(loaded.compiled, layer, defaults=_DEFAULTS, provider_models=_PROVIDER_MODELS)


def test_host_layer_python_block_spelling_names_the_dashed_key():
    loaded = load_flow("""
id: f
name: f
nodes:
  a: {kind: agent, prompt: hi}
output: ${a.output}
""")
    with pytest.raises(ValueError, match="`qwen_cn` is not an llm_config key; write `qwen-cn`"):
        resolve_llm_cascade(
            loaded.compiled, {"qwen_cn": {"top_k": 4}}, defaults=_DEFAULTS,
            provider_models=_PROVIDER_MODELS,
        )


def test_host_layer_block_merges_per_field_with_authored_blocks():
    loaded = load_flow("""
id: f
name: f
llm_config:
  qwen-cn: {enable_thinking: true}
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: {qwen-cn: {top_k: 4}}
output: ${a.output}
""")
    effective = resolve_llm_cascade(
        loaded.compiled,
        {"provider": "qwen-cn", "model": "qwen3-max", "qwen-cn": {"top_k": 1, "seed": 7}},
        defaults=_DEFAULTS,
        provider_models=_PROVIDER_MODELS,
    )
    assert effective["a"]["qwen-cn"] == {"enable_thinking": True, "top_k": 4, "seed": 7}
