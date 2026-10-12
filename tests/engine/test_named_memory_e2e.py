"""Named memories end to end: YAML -> load -> build_memories -> run -> backward -> rerun, and the CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import yian.llm_clients as llm_clients
from yian import build_memories, load_flow, run_flow

_ROOT_FLOW = """id: e2e-memory
name: e2e_memory
long_term_memory:
  notes:
    type: episodic
    config: {recall_limit: 3}
uses:
  lib: notes-lib
input: {question: str}
nodes:
  ask:
    kind: agent
    llm_config: {provider: openai, model: fake-model}
    long_term_memory: {chat: session}
    input: {question: "${input.question}"}
    prompt: ${question}
    output: str
  lib_step:
    kind: call
    call: lib
    input: {topic: "${input.question}"}
output:
  answer: ${ask.output}
  summary: ${lib_step.output}
"""

_LIB_FLOW = """id: notes-lib
name: notes_lib
long_term_memory: {notes: episodic}
input: {topic: str}
nodes:
  summarize:
    kind: agent
    llm_config: {provider: openai, model: fake-model}
    input: {topic: "${input.topic}"}
    prompt: Summarize ${topic}.
    output: str
output: ${summarize.output}
"""

_STATUS_PREFIX = "Session history status (context only): "


def _reply(messages) -> AIMessage:
    """Describe what the model saw: real user turns and recalled memory keys."""
    users = [m for m in messages if isinstance(m, HumanMessage)]
    count = sum(not m.content.startswith(_STATUS_PREFIX) for m in users)
    keys: list[str] = []
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    if "Memory context (canonical JSON" in system:
        payload = json.loads(system.rsplit("\n", 1)[1])
        for section in payload["sections"]:
            if section["memory"] not in keys:
                keys.append(section["memory"])
    return AIMessage(content=f"users={count} memories={','.join(keys)}")


class _Model:
    def invoke(self, messages):
        return _reply(messages)


def _write_flows(directory: Path) -> Path:
    (directory / "notes-lib.yaml").write_text(_LIB_FLOW)
    flow = directory / "flow.yaml"
    flow.write_text(_ROOT_FLOW)
    return flow


def _load(path: Path):
    return load_flow(path.read_text(), search_paths=[path.parent], source_label=path.name)


def test_python_api_named_memories_persist_and_stack(tmp_path, monkeypatch):
    # Keep any user state the library writes out of the real home.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model())
    flow = _load(_write_flows(tmp_path))
    root = tmp_path / "mem"

    with build_memories(flow, root=root) as memory:
        assert dict(memory.types) == {
            "chat": "session",
            "lib.notes": "episodic",
            "notes": "episodic",
        }
        assert not root.exists()  # construction is lazy
        first = run_flow(flow, {"question": "q1"}, memory=memory, learnable=True)
        assert first.status == "succeeded", first
        assert first.output == {"answer": "users=1 memories=", "summary": "users=1 memories="}
        outcome = first.backward()
        assert outcome.status == "succeeded"
        assert outcome.learning is not None and outcome.learning.status == "succeeded"

    base = root / "e2e-memory"
    assert sorted(p.relative_to(base).as_posix() for p in base.rglob("*.sqlite")) == [
        "chat.sqlite",
        "lib/notes.sqlite",
        "notes.sqlite",
    ]

    # A fresh registry over the same root sees the learned episodes and the session.
    with build_memories(flow, root=root) as memory:
        second = run_flow(flow, {"question": "q2"}, memory=memory)
    assert second.status == "succeeded", second
    # ask: one root-flow memory, plus two session user turns.
    assert second.output["answer"] == "users=2 memories=notes"
    # summarize: outer root memory first, then its own file's scoped memory.
    assert second.output["summary"] == "users=1 memories=notes,lib.notes"


_CHILD = """
import sys
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import yian.llm_clients as clients
import yian.cli.run as run_cli
from yian.cli import main

sys.path.insert(0, sys.argv.pop(1))
from test_named_memory_e2e import _Model

clients.model_from_config = lambda _: _Model()
run_cli._ensure_provider_keys = lambda *args: None
main()
"""


def _cli(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    # Keep CLI user state (default logs, config) out of the real home.
    env["HOME"] = str(cwd)
    # The CLI's rich consoles read COLUMNS when cli.run is imported; a wide
    # child keeps each asserted line on one line.
    env["COLUMNS"] = "200"
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(Path(__file__).resolve().parent), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    result.stdout = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    return result


def _result_line(answer_users: int) -> str:
    """Return the exact line `yian run` prints for this flow's output."""
    # The CLI never runs backward, so no episodic memory has anything to recall.
    return (
        f"{{'answer': 'users={answer_users} memories=', "
        "'summary': 'users=1 memories='}"
    )


def test_cli_memory_root_persists_session_across_processes(tmp_path):
    flow = _write_flows(tmp_path)
    root = tmp_path / "mem"
    outputs = []
    for index, question in enumerate(("first", "second")):
        result = _cli(
            tmp_path, "run", str(flow), "--quiet",
            "--input", f"question={question}", "--memory-root", str(root),
            "--log", str(tmp_path / f"run-{index}.jsonl"),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(result.stdout.splitlines())
    assert _result_line(1) in outputs[0]
    assert _result_line(2) in outputs[1]  # session history came from chat.sqlite
    base = root / "e2e-memory"
    assert (base / "chat.sqlite").exists()
    assert (base / "notes.sqlite").exists()
    assert (base / "lib" / "notes.sqlite").exists()


def test_cli_memory_storage_override_places_one_key(tmp_path):
    flow = _write_flows(tmp_path)
    root, chat = tmp_path / "mem", tmp_path / "chat.sqlite"
    for index in range(2):
        result = _cli(
            tmp_path, "run", str(flow), "--quiet", "--input", "question=hi",
            "--memory-root", str(root), "--memory-storage", f"chat={chat}",
            "--log", str(tmp_path / f"run-{index}.jsonl"),
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert _result_line(2) in result.stdout.splitlines()
    assert chat.exists()
    assert not (root / "e2e-memory" / "chat.sqlite").exists()
