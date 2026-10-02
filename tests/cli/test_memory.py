"""composer run memory flags, registry lifetime, and run-boundary integration."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from agent_composer.cli import app
import agent_composer.cli.run as run_cli
from agent_composer.cli.memory import MemoryOptionError, build_cli_memories
from agent_composer.compose import LoadError, load_flow
from agent_composer.runtime.log import RunStartedRecord, decode_record

_RUNNER = CliRunner()

_CODE_FLOW = """IDLINE
name: cli_memory
long_term_memory: MEMORY
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: ${work.output}
"""

_SESSION_FLOW = """id: cli-memory
name: cli_memory
nodes:
  work:
    kind: agent
    mode: plain
    long_term_memory: {chat: session}
    prompt: hello
    output: str
output: ${work.output}
"""


@pytest.fixture(autouse=True)
def _offline_model(monkeypatch):
    """Answer every AGENT call locally so session flows run without a provider."""
    from langchain_core.messages import AIMessage
    import agent_composer.llm_clients as llm_clients

    class _Model:
        def invoke(self, messages):
            return AIMessage(content="7")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: _Model())
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *args: None)


def _flow_text(memory="{notes: episodic}", flow_id="cli-memory"):
    """Return the CODE flow; ``memory=None`` drops the declaration line."""
    text = _CODE_FLOW.replace("IDLINE\n", f"id: {flow_id}\n" if flow_id else "")
    if memory is None:
        return text.replace("long_term_memory: MEMORY\n", "")
    return text.replace("MEMORY", memory)


def _flow_file(tmp_path, memory="{notes: episodic}", flow_id="cli-memory"):
    path = tmp_path / "flow.yaml"
    path.write_text(_flow_text(memory, flow_id))
    return path


def _session_flow_file(tmp_path):
    path = tmp_path / "flow.yaml"
    path.write_text(_SESSION_FLOW)
    return path


def _configured(policy, config):
    """Return a flow that declares memory ``policy`` of type ``policy`` with a YAML config."""
    if policy == "session":
        return _SESSION_FLOW.replace(
            "{chat: session}", f"{{session: {{type: session, config: {config}}}}}"
        )
    return _flow_text(f"{{{policy}: {{type: {policy}, config: {config}}}}}")


def _plain(text):
    """Strip ANSI styles and collapse whitespace, so wrapped messages match."""
    # cli.run's err_console fixes its width when the module is imported, so a
    # test cannot widen it; long messages wrap at 80 columns.
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


def _records(path):
    return [decode_record(line).record for line in path.read_bytes().splitlines()]


def _run(tmp_path, *extra, flow=None):
    flow = flow or _flow_file(tmp_path)
    log = tmp_path / "run.jsonl"
    result = _RUNNER.invoke(app, ["run", str(flow), "--log", str(log), "--quiet", *extra])
    return result, log


def _no_inputs_or_provider(monkeypatch):
    monkeypatch.setattr(run_cli, "_gather_missing_inputs", lambda *a, **k: pytest.fail("inputs"))
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *a, **k: pytest.fail("provider"))


def test_memory_root_persists_every_builtin_under_the_flow_id(tmp_path):
    root = tmp_path / "mem"
    flow = _flow_file(tmp_path, "{notes: episodic, lessons: reflective, about: profile}")
    result, log = _run(tmp_path, "--memory-root", str(root), flow=flow)
    assert result.exit_code == 0, result.output
    assert (root / "cli-memory" / "notes.sqlite").is_file()
    assert (root / "cli-memory" / "lessons.sqlite").is_file()
    assert (root / "cli-memory" / ".about.md.lock").exists()
    assert isinstance(_records(log)[0], RunStartedRecord)


def test_memory_storage_is_keyed_by_memory_key(tmp_path):
    path = tmp_path / "notes.sqlite"
    result, _ = _run(tmp_path, "--memory-storage", f"notes={path}")
    assert result.exit_code == 0, result.output
    assert path.is_file()


def test_no_flags_keep_builtins_in_ram(tmp_path):
    flow = _flow_file(tmp_path, "{notes: episodic, lessons: reflective, about: profile}")
    result, _ = _run(tmp_path, flow=flow)
    assert result.exit_code == 0, result.output
    # tests/cli/conftest.py creates tmp_path/home for every CLI test.
    stored = [p for p in tmp_path.rglob("*") if p.suffix in {".sqlite", ".md", ".lock"}]
    assert stored == []


@pytest.mark.parametrize(
    "options",
    [
        ["notes"],
        ["notes="],
        ["=x.sqlite"],
        ["missing=a.sqlite"],
        ["notes=a.sqlite", "notes=b.sqlite"],
    ],
)
def test_bad_memory_storage_is_a_bad_parameter_before_inputs_provider_and_log(
    tmp_path, options, monkeypatch
):
    _no_inputs_or_provider(monkeypatch)
    args = []
    for option in options:
        args += ["--memory-storage", option]
    result, log = _run(tmp_path, *args)
    assert result.exit_code == 2, result.output
    assert "--memory-storage" in _plain(result.output)
    assert not log.exists()


def test_bad_storage_path_shape_fails_before_inputs_provider_and_log(tmp_path, monkeypatch):
    _no_inputs_or_provider(monkeypatch)
    result, log = _run(tmp_path, "--memory-storage", "notes=:memory:")
    assert result.exit_code == 1, result.output
    assert "memory 'notes': storage ':memory:' is not a local file path" in _plain(result.output)
    assert "Traceback" not in result.output
    assert not log.exists()


def test_memory_storage_for_a_custom_key_is_a_bad_parameter(tmp_path):
    flow = _flow_file(tmp_path, "{vec: vector}")
    result, _ = _run(tmp_path, "--memory-storage", "vec=v.sqlite", flow=flow)
    assert result.exit_code == 2
    assert "custom" in _plain(result.output)


def test_memory_root_without_flow_id_is_a_bad_parameter(tmp_path):
    # YAML requires id:, so only a flow compiled in Python can lack one.
    flow = load_flow(_flow_text())
    flow.compiled.flow_id = None
    with pytest.raises(MemoryOptionError, match="--memory-root") as caught:
        build_cli_memories(flow, root=tmp_path / "mem", storage_options=())
    assert caught.value.param_hint == "--memory-root"


def test_custom_type_points_to_the_python_api(tmp_path, monkeypatch):
    _no_inputs_or_provider(monkeypatch)
    flow = _flow_file(tmp_path, "{vec: vector}")
    result, log = _run(tmp_path, flow=flow)
    assert result.exit_code == 1
    text = _plain(result.output)
    assert "Python API" in text and "build_memories" in text
    assert "Traceback" not in result.output
    assert not log.exists()


def test_flags_are_accepted_with_resume(tmp_path):
    root = tmp_path / "mem"
    result, log = _run(tmp_path, "--memory-root", str(root))
    assert result.exit_code == 0, result.output
    before = log.read_bytes()
    resumed = _RUNNER.invoke(
        app, ["run", str(_flow_file(tmp_path)), "--resume", str(log), "--quiet",
              "--memory-root", str(root)]
    )
    assert resumed.exit_code == 0, resumed.output
    assert log.read_bytes() == before


@pytest.mark.parametrize("policy, config, detail", [
    ("episodic", "{bogus: 1}", r"episodic memory config has unknown key\(s\) \['bogus'\]"),
    ("reflective", "{llm_config: {model: x}}", r"unknown key\(s\) \['llm_config'\]"),
    ("session", "{storage: x.sqlite}", "config key 'storage' is not allowed"),
])
def test_yaml_config_outside_the_allowlist_fails_at_load(policy, config, detail):
    with pytest.raises(LoadError, match=detail):
        load_flow(_configured(policy, config))
