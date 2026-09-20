"""Native provider selection stays separate from API clients and ambient replay."""

from __future__ import annotations

import builtins
import copy
from io import StringIO
import json
import os
import subprocess
import sys

import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from agent_composer.compile.llm_cascade import (
    apply_effective_llm_map,
    resolve_llm_cascade,
)
from agent_composer.compose import load_flow
from agent_composer.llm_clients.config import LLMConfig
from agent_composer.llm_clients.factory import create_llm_client, model_from_config
from agent_composer.memory.contracts import MemoryOperationError
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Output
from agent_composer.run import resume_flow, run_flow

_DEFAULTS = {"provider": "openai", "model": "default-model"}
_OLD_LLM_CONFIG = {
    "provider": "openai",
    "model": "old-model",
    "anthropic_effort": None,
    "openai_reasoning_effort": None,
    "google_thinking_level": None,
    "temperature": None,
}


def _flow(config=None):
    config = {"provider": "codex", "model": "native-model", **(config or {})}
    return load_flow(
        "id: native-config\nname: native_config\nnodes:\n"
        f"  a: {{kind: agent, prompt: hi, llm_config: {json.dumps(config)}}}\n"
        "output: ${a.output}\n"
    )


@pytest.mark.parametrize(
    "sandbox", ["read-only", "workspace-write", "danger-full-access"]
)
def test_native_config_accepts_sdk_sandbox_values(sandbox):
    config = LLMConfig(provider="codex", sandbox=sandbox, cwd="workspace")
    assert config.sandbox == sandbox
    assert config.cwd == "workspace"


def test_native_fields_are_optional_until_cascade_resolution():
    config = LLMConfig(provider="codex")
    assert config.sandbox is None
    assert config.cwd is None
    assert config.model_dump(exclude_unset=True) == {"provider": "codex"}


def test_native_provider_and_sandbox_are_in_config_schema():
    schema = LLMConfig.model_json_schema()["properties"]
    assert "codex" in schema["provider"]["anyOf"][0]["enum"]
    assert schema["sandbox"]["anyOf"][0]["enum"] == [
        "read-only",
        "workspace-write",
        "danger-full-access",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        {"sandbox": "writable"},
        {"sandbox": True},
        {"cwd": 12},
        {"cwd": ""},
        {"cwd": "\x00bad"},
        {"sandbox_mode": "read-only"},
        {"temperature": 0.0},
    ],
)
def test_native_config_rejects_invalid_or_unsupported_options(extra):
    with pytest.raises(ValueError):
        LLMConfig(provider="codex", **extra)


@pytest.mark.parametrize("factory", ["client", "model", "model-instance"])
def test_native_provider_cannot_enter_api_factory_or_import_sdk(monkeypatch, factory):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("openai_codex") or name.endswith("openai_client"):
            pytest.fail(f"native API factory attempted import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="native.*AGENT"):
        if factory == "client":
            create_llm_client("CoDeX", "native-model")
        elif factory == "model":
            model_from_config({"provider": "codex", "model": "native-model"})
        else:
            model_from_config(LLMConfig(provider="codex", model="native-model"))


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [(None, "/launch"), ("work/../project", "/launch/project"), ("/other", "/other")],
)
def test_cascade_pins_native_defaults_without_reading_ambient_state(
    monkeypatch, cwd, expected
):
    loaded = _flow({"cwd": cwd})

    def forbidden(*args, **kwargs):
        pytest.fail("compile cascade read ambient state")

    monkeypatch.setattr(os, "getcwd", forbidden)
    monkeypatch.setattr(os, "stat", forbidden)
    effective = resolve_llm_cascade(
        loaded.compiled, {}, defaults=_DEFAULTS, launch_cwd="/launch"
    )
    assert effective == {
        "a": {
            "provider": "codex",
            "model": "native-model",
            "sandbox": "read-only",
            "cwd": expected,
        }
    }
    effective["a"]["cwd"] = "/mutated"
    assert loaded.compiled.nodes["a"].llm_config["cwd"] == expected


@pytest.mark.parametrize("launch", ["", "relative", 12, "/bad\x00path"])
def test_launch_cwd_must_be_absolute(launch):
    with pytest.raises(ValueError, match="launch_cwd"):
        resolve_llm_cascade(_flow().compiled, {}, defaults=_DEFAULTS, launch_cwd=launch)


def test_native_resolution_requires_explicit_base_for_relative_directory():
    with pytest.raises(ValueError, match="launch_cwd"):
        resolve_llm_cascade(_flow().compiled, {}, defaults=_DEFAULTS)
    assert (
        resolve_llm_cascade(
            _flow({"cwd": "/explicit"}).compiled, {}, defaults=_DEFAULTS
        )["a"]["cwd"]
        == "/explicit"
    )


def test_inherited_native_temperature_fails_before_log_or_node_work(
    tmp_path, monkeypatch
):
    log = tmp_path / "never-created.jsonl"
    monkeypatch.setattr(
        AgentNode, "forward", lambda *a, **k: pytest.fail("node called")
    )
    with pytest.raises(ValueError, match="temperature"):
        run_flow(_flow(), {}, llm_config={"temperature": 0.2}, log=log)
    assert not log.exists()


def test_api_effective_map_does_not_gain_native_defaults():
    effective = resolve_llm_cascade(
        _flow({"provider": "openai"}).compiled, {}, defaults=_DEFAULTS
    )
    assert effective == {"a": {"provider": "openai", "model": "native-model"}}


def test_native_options_follow_existing_fill_gap_and_inherit_rules():
    loaded = _flow({"cwd": "node", "sandbox": None})
    loaded.compiled.flow_llm_config = {"cwd": "flow", "sandbox": "workspace-write"}
    effective = resolve_llm_cascade(
        loaded.compiled,
        {"cwd": "caller", "sandbox": "danger-full-access"},
        defaults=_DEFAULTS,
        launch_cwd="/launch",
    )
    assert effective["a"]["cwd"] == "/launch/node"
    assert effective["a"]["sandbox"] == "workspace-write"

    node = loaded.compiled.nodes["a"]
    node.llm_inherit = False
    node.own_llm_config = {"model": "isolated"}
    effective = resolve_llm_cascade(
        loaded.compiled,
        {"provider": "openai", "sandbox": "danger-full-access"},
        defaults={"provider": "codex", "model": "default"},
        launch_cwd="/launch",
    )
    assert effective["a"] == {
        "provider": "codex",
        "model": "isolated",
        "sandbox": "read-only",
        "cwd": "/launch",
    }


def test_native_nested_calls_resolve_from_launch_not_parent_agent_cwd():
    loaded = load_flow("""id: nested-native
name: nested_native
llm_config: {provider: codex, model: native}
defs:
  leaf:
    nodes:
      agent: {kind: agent, prompt: hi}
    output: ${agent.output}
  left:
    llm_config: {cwd: left, sandbox: workspace-write}
    nodes:
      call: {kind: call, call: leaf}
    output: ${call.output}
  right:
    llm_config: {cwd: right}
    nodes:
      call: {kind: call, call: leaf}
    output: ${call.output}
nodes:
  left: {kind: call, call: left}
  right: {kind: call, call: right}
output: {left: '${left.output}', right: '${right.output}'}
""")
    authored = copy.deepcopy(loaded.compiled)
    effective = resolve_llm_cascade(
        loaded.compiled, {}, defaults=_DEFAULTS, launch_cwd="/launch"
    )
    assert set(effective) == {"left/call/agent", "right/call/agent"}
    assert effective["left/call/agent"]["cwd"] == "/launch/left"
    assert effective["left/call/agent"]["sandbox"] == "workspace-write"
    assert effective["right/call/agent"]["cwd"] == "/launch/right"
    assert effective["right/call/agent"]["sandbox"] == "read-only"
    apply_effective_llm_map(authored, effective)
    left = authored.nodes["left"].child.nodes["call"].child.nodes["agent"]
    right = authored.nodes["right"].child.nodes["call"].child.nodes["agent"]
    assert left is not right
    left.llm_config["cwd"] = "/changed"
    assert right.llm_config["cwd"] == "/launch/right"
    assert effective["left/call/agent"]["cwd"] == "/launch/left"


def test_api_factory_ignores_native_and_other_provider_options(monkeypatch):
    import agent_composer.llm_clients.factory as factory

    calls = []

    class Client:
        def get_llm(self):
            return self

    def create(**kwargs):
        calls.append(kwargs)
        return Client()

    monkeypatch.setattr(factory, "create_llm_client", create)
    factory.model_from_config(
        {
            "provider": "openai",
            "model": "api",
            "temperature": 0.2,
            "openai_reasoning_effort": "high",
            "anthropic_effort": "low",
            "cwd": "/unused",
            "sandbox": "danger-full-access",
        }
    )
    assert calls == [
        {
            "provider": "openai",
            "model": "api",
            "temperature": 0.2,
            "reasoning_effort": "high",
            "max_retries": 6,
        }
    ]


def test_replay_preserves_native_directory_and_detaches_map(monkeypatch):
    loaded = _flow()
    effective = resolve_llm_cascade(
        loaded.compiled, {}, defaults=_DEFAULTS, launch_cwd="/original"
    )
    replayed = _flow()
    monkeypatch.setattr(os, "getcwd", lambda: pytest.fail("replay read cwd"))
    apply_effective_llm_map(replayed.compiled, effective)
    assert replayed.compiled.nodes["a"].llm_config == effective["a"]
    effective["a"]["cwd"] = "/changed"
    assert replayed.compiled.nodes["a"].llm_config["cwd"] == "/original"


@pytest.mark.parametrize("field", ["cwd", "sandbox"])
def test_replay_rejects_incomplete_native_map_without_mutation(field):
    loaded = _flow()
    before = copy.deepcopy(loaded.compiled.nodes["a"].llm_config)
    config = {
        "provider": "codex",
        "model": "native",
        "cwd": "/old",
        "sandbox": "read-only",
    }
    del config[field]
    with pytest.raises(ValueError, match=field):
        apply_effective_llm_map(loaded.compiled, {"a": config})
    assert loaded.compiled.nodes["a"].llm_config == before


@pytest.mark.parametrize(
    "change",
    [{"cwd": "relative"}, {"cwd": ""}, {"sandbox": "unsafe"}, {"temperature": 0.2}],
)
def test_replay_rejects_invalid_native_config_without_mutation(change):
    loaded = _flow()
    before = copy.deepcopy(loaded.compiled.nodes["a"].llm_config)
    config = {
        "provider": "codex",
        "model": "native",
        "cwd": "/old",
        "sandbox": "read-only",
        **change,
    }
    with pytest.raises(ValueError):
        apply_effective_llm_map(loaded.compiled, {"a": config})
    assert loaded.compiled.nodes["a"].llm_config == before


def test_resumed_run_uses_recorded_native_directory_after_chdir(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    other = tmp_path / "other"
    launch.mkdir()
    other.mkdir()
    log = tmp_path / "native.jsonl"
    seen = []

    def forward(node, inputs, **caps):
        seen.append(copy.deepcopy(node.llm_config))
        if len(seen) == 1:
            raise KeyboardInterrupt("interrupt native invocation")
        return Output("ok")

    monkeypatch.setattr(AgentNode, "forward", forward)
    monkeypatch.chdir(launch)
    loaded = _flow({"cwd": "project", "sandbox": "workspace-write"})
    with pytest.raises(KeyboardInterrupt):
        run_flow(loaded, {}, log=log)
    prefix = log.read_bytes()
    first = json.loads(prefix.splitlines()[0])
    recorded = first["payload"]["effective_llm_by_static_path"]["a"]
    assert recorded["cwd"] == str(launch / "project")
    monkeypatch.chdir(other)
    result = resume_flow(loaded, log=log)
    assert result.status == "succeeded"
    assert seen == [recorded, recorded]
    assert log.read_bytes().startswith(prefix)


def test_native_config_loads_when_sdk_import_is_unavailable():
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'openai_codex' or fullname.startswith('openai_codex.'):
            raise AssertionError('SDK imported by configuration')
sys.meta_path.insert(0, Block())
from agent_composer.llm_clients.config import LLMConfig
from agent_composer.llm_clients.factory import model_from_config
from agent_composer.compose import load_flow
assert LLMConfig(provider='codex').provider == 'codex'
load_flow('id: f\\nname: f\\nnodes:\\n  a: {kind: agent, prompt: hi, llm_config: {provider: codex}}\\noutput: ${a.output}\\n')
try:
    model_from_config({'provider': 'codex', 'model': 'native'})
except ValueError as exc:
    assert 'native AGENT' in str(exc)
else:
    raise AssertionError('native provider entered API factory')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_cli_skips_native_key_preflight_but_keeps_api_checks(monkeypatch):
    import agent_composer.cli.run as cli_run
    import agent_composer.cli.utils as cli_utils

    checked = []
    monkeypatch.setattr(cli_utils, "ensure_api_key", checked.append)
    cli_run._ensure_provider_keys(_flow(), {})
    assert checked == []
    cli_run._ensure_provider_keys(_flow({"provider": "openai"}), {})
    assert checked == ["openai"]


@pytest.mark.parametrize("engine_trace", [False, True])
@pytest.mark.parametrize("failure", ["inherited-temperature", "type", "cwd", "api-key"])
def test_cli_renders_native_preflight_errors_before_effects(
    tmp_path, monkeypatch, engine_trace, failure
):
    import agent_composer.cli.run as cli_run
    import agent_composer.cli.utils as cli_utils

    source = tmp_path / "flow.yaml"
    source.write_text(
        "id: native-preflight\nname: native_preflight\n"
        "llm_config: {temperature: 0.2}\nnodes:\n"
        "  a: {kind: agent, prompt: hi, llm_config: {provider: codex, model: native}}\n"
        "output: ${a.output}\n"
    )
    log = tmp_path / "never-created.jsonl"
    output = StringIO()
    monkeypatch.setattr(
        cli_run, "err_console", Console(file=output, force_terminal=False, width=160)
    )

    def unexpected(*args, **kwargs):
        pytest.fail("configuration failure reached a provider, worker, or log effect")

    monkeypatch.setattr(cli_utils, "ensure_api_key", unexpected)
    monkeypatch.setattr(cli_run, "_run_with_terminal_controller", unexpected)
    monkeypatch.setattr(cli_run, "_report_run_log", unexpected)
    monkeypatch.setattr(cli_run, "_default_run_log_path", unexpected)
    if failure == "api-key":
        monkeypatch.setattr(cli_run, "_agent_providers", lambda *a, **k: ["openai"])

        def missing_key(*args, **kwargs):
            raise RuntimeError("missing OPENAI_API_KEY")

        monkeypatch.setattr(cli_utils, "ensure_api_key", missing_key)
    elif failure != "inherited-temperature":
        error = (
            TypeError("invalid native configuration")
            if failure == "type"
            else OSError("launch directory unavailable")
        )

        def fail_preflight(*args, **kwargs):
            raise error

        monkeypatch.setattr(cli_run, "_agent_providers", fail_preflight)
    app = typer.Typer()
    app.command()(cli_run.run)
    args = [str(source), "--log", str(log)]
    if engine_trace:
        args.append("--engine-trace")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    rendered = output.getvalue() + result.output
    assert ("run failed" in rendered) is (failure != "api-key")
    message = {
        "inherited-temperature": "temperature is unsupported for the Codex provider",
        "type": "invalid native configuration",
        "cwd": "launch directory unavailable",
        "api-key": "missing OPENAI_API_KEY",
    }[failure]
    assert message in rendered
    assert ("Traceback" in rendered) is (engine_trace and failure != "api-key")
    assert not log.exists()


def _old_snapshot():
    with ReflectiveMemory(llm_config=_OLD_LLM_CONFIG) as memory:
        with memory.open_run() as session:
            snapshot = session.snapshot
    snapshot["llm_config"] = dict(_OLD_LLM_CONFIG)
    return snapshot


@pytest.mark.parametrize("durable", [False, True])
def test_old_reflective_snapshot_keeps_exact_config_shape(tmp_path, durable):
    snapshot = _old_snapshot()
    storage = tmp_path / "reflective.sqlite" if durable else None
    with ReflectiveMemory(
        llm_config={"provider": "openai", "model": "new"}, storage=storage
    ) as memory:
        with memory.open_run(snapshot=snapshot) as session:
            assert session.snapshot == snapshot
        with memory.open_run(snapshot=snapshot) as session:
            assert session.snapshot == snapshot
        with memory.open_run() as session:
            assert session.snapshot["llm_config"]["model"] == "new"
    if durable:
        with ReflectiveMemory(storage=storage) as memory:
            with memory.open_run(snapshot=snapshot) as session:
                assert session.snapshot == snapshot


@pytest.mark.parametrize("field", list(_OLD_LLM_CONFIG))
def test_old_reflective_snapshot_still_requires_every_old_field(field):
    snapshot = _old_snapshot()
    del snapshot["llm_config"][field]
    with ReflectiveMemory() as memory:
        with pytest.raises(MemoryOperationError):
            memory.open_run(snapshot=snapshot)


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": None},
        {"provider": "invalid"},
        {"temperature": "0.2"},
        {"model": 7},
    ],
)
def test_old_reflective_snapshot_still_rejects_noncanonical_config(change):
    snapshot = _old_snapshot()
    snapshot["llm_config"].update(change)
    with ReflectiveMemory() as memory:
        with pytest.raises(MemoryOperationError):
            memory.open_run(snapshot=snapshot)


def test_recovery_accepts_old_reflective_model_shape_without_rewriting_log(
    tmp_path, monkeypatch
):
    from agent_composer.events import NodeStarted

    log = tmp_path / "old-reflective.jsonl"
    loaded = load_flow("""id: old-reflective
name: old_reflective
long_term_memory: reflective
nodes:
  a: {kind: agent, prompt: hi, llm_config: {provider: openai, model: api}}
output: ${a.output}
""")

    def interrupt(event):
        if isinstance(event, NodeStarted) and event.node_id == "a":
            raise KeyboardInterrupt("before model")

    with ReflectiveMemory(llm_config=_OLD_LLM_CONFIG) as memory:
        with pytest.raises(KeyboardInterrupt):
            run_flow(
                loaded, {}, log=log, memory={"reflective": memory}, on_event=interrupt
            )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    envelope = records[0]["payload"]["memory_snapshots"][0]
    assert envelope["policy_name"] == "reflective"
    snapshot = envelope["snapshot"]
    snapshot["llm_config"] = dict(_OLD_LLM_CONFIG)
    log.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
    )
    prefix = log.read_bytes()
    monkeypatch.setattr(AgentNode, "forward", lambda *a, **k: Output("ok"))
    with ReflectiveMemory(
        llm_config={"provider": "anthropic", "model": "new"}
    ) as memory:
        result = resume_flow(loaded, log=log, memory={"reflective": memory})
        assert result.status == "succeeded"
        with memory.open_run(snapshot=snapshot) as session:
            assert session.snapshot == snapshot
    assert log.read_bytes().startswith(prefix)
