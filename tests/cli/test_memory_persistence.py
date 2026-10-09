"""CLI-owned session storage survives process exits without using the run log as memory."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from yian.memory.session import SessionMemory
from yian.runtime.log import decode_record

_FLOW = """id: cli-session-restart
name: cli_session_restart
llm_config: {provider: openai, model: offline-model}
input: {question: str}
nodes:
  answer:
    kind: agent
    mode: plain
    long_term_memory: session
    input: {question: "${input.question}"}
    prompt: ${question}
    output: str
output: ${answer.output}
"""

_CHILD = """
import os
import sys
from langchain_core.messages import AIMessage, HumanMessage
import yian.llm_clients as clients
import yian.cli.run as run_cli
from yian.cli import main

crash = sys.argv.pop(1) == 'crash'
class Model:
    def invoke(self, messages):
        if crash:
            os._exit(42)
        status_prefix = 'Session history status (context only): '
        users = [message for message in messages if isinstance(message, HumanMessage)]
        count = sum(not message.content.startswith(status_prefix) for message in users)
        unknown = any('completion unknown' in message.content for message in users)
        return AIMessage(content=f'Seen user messages: {count}; Unknown history: {unknown}')

clients.model_from_config = lambda _: Model()
run_cli._ensure_provider_keys = lambda *args: None
main()
"""


def _invoke(tmp_path, *args, crash=False):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, "crash" if crash else "normal", *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    result.stdout = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    return result


def _records(path):
    return [decode_record(line).record for line in path.read_bytes().splitlines()]


@pytest.mark.parametrize("durable", [False, True])
def test_session_history_across_cli_processes_is_explicitly_durable(tmp_path, durable):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_FLOW)
    storage = tmp_path / "session.sqlite"
    options = ["--memory-storage", f"session={storage}"] if durable else []
    for index in range(2):
        log = tmp_path / f"run-{index}.jsonl"
        result = _invoke(
            tmp_path,
            "run",
            str(flow),
            "--quiet",
            "--input",
            f"question=question-{index}",
            "--log",
            str(log),
            *options,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        expected = index + 1 if durable else 1
        assert f"Seen user messages: {expected}" in result.stdout
        records = _records(log)
        assert records[0].memories is not None
        assert records[0].learnable is False
        assert not any(
            type(record).__name__.startswith("Backward") for record in records
        )
        assert str(storage).encode() not in log.read_bytes()
    assert storage.exists() is durable


def test_terminal_cli_resume_does_not_acquire_the_memory_store(tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_FLOW)
    storage, log = tmp_path / "session.sqlite", tmp_path / "run.jsonl"
    options = ["--memory-storage", f"session={storage}"]
    first = _invoke(
        tmp_path,
        "run",
        str(flow),
        "--quiet",
        "--input",
        "question=first",
        "--log",
        str(log),
        *options,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    prefix = log.read_bytes()
    with SessionMemory(storage=storage):
        inspected = _invoke(
            tmp_path,
            "run",
            str(flow),
            "--quiet",
            "--resume",
            str(log),
            *options,
        )
    assert inspected.returncode == 0, inspected.stdout + inspected.stderr
    assert "Seen user messages: 1" in inspected.stdout
    assert log.read_bytes() == prefix


def test_cli_resume_after_process_death_reuses_saved_history_and_repeats_request(
    tmp_path,
):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_FLOW)
    storage, log = tmp_path / "session.sqlite", tmp_path / "run.jsonl"
    options = ["--memory-storage", f"session={storage}"]
    crashed = _invoke(
        tmp_path,
        "run",
        str(flow),
        "--quiet",
        "--input",
        "question=retry me",
        "--log",
        str(log),
        *options,
        crash=True,
    )
    assert crashed.returncode == 42, crashed.stdout + crashed.stderr
    prefix = log.read_bytes()
    resumed = _invoke(
        tmp_path,
        "run",
        str(flow),
        "--quiet",
        "--resume",
        str(log),
        *options,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Seen user messages: 2" in resumed.stdout
    assert "Unknown history: True" in resumed.stdout
    assert log.read_bytes().startswith(prefix)
    assert str(storage).encode() not in log.read_bytes()
    with SessionMemory(storage=storage) as memory:
        with memory.open_run() as opened:
            effects = opened.snapshot["conversation"]["effects"]
            assert [effect["status"] for effect in effects] == ["pending", "completed"]
            assert effects[0]["invocation"] != effects[1]["invocation"]


def test_memory_root_keeps_session_history_under_the_flow_id(tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_FLOW)
    root = tmp_path / "mem"
    for index in range(2):
        result = _invoke(
            tmp_path,
            "run",
            str(flow),
            "--quiet",
            "--input",
            f"question=question-{index}",
            "--log",
            str(tmp_path / f"run-{index}.jsonl"),
            "--memory-root",
            str(root),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"Seen user messages: {index + 1}" in result.stdout
    assert (root / "cli-session-restart" / "session.sqlite").is_file()
