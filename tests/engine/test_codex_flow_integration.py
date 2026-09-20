"""Native flow execution keeps thread ownership in the selected memory backend."""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

from langchain_core.tools import StructuredTool
import pytest

from agent_composer import load_flow, run_flow
import agent_composer.llm_clients as clients
from agent_composer.memory.episodic import EpisodicMemory
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.memory.session import SessionMemory
from agent_composer.tools import TOOL_REGISTRY

_FLOW = """id: native-continuity
name: native_continuity
llm_config: {provider: codex, model: offline-codex}
input: {question: str}
nodes:
  answer:
    kind: agent
    mode: tool_calling
    tools: [builtin_tools]
    input: {question: '${input.question}'}
    prompt: ${question}
    output: str
output: ${answer.output}
"""


@pytest.fixture
def native(monkeypatch):
    threads, calls, instances = {}, [], []

    class Worker:
        def __init__(self, **options):
            self.options = options
            self.closed = False
            self.overlapped = any(not worker.closed for worker in instances)
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            self.closed = True

        def initialize(self):
            return {}

        def _result(self, operation, result):
            observer = self.options.get("on_result")
            if observer is not None:
                observer(operation, deepcopy(result))
            return result

        def start_thread(self, params=None):
            identity = f"thread-{len(threads) + 1}"
            threads[identity] = 0
            calls.append(("start", identity, deepcopy(params)))
            return self._result(
                "start_thread",
                {"thread": {"id": identity, "sessionId": "not-the-thread-id"}},
            )

        def resume_thread(self, thread_id, params=None):
            assert thread_id in threads
            calls.append(("resume", thread_id, deepcopy(params)))
            return self._result(
                "resume_thread",
                {"thread": {"id": thread_id, "sessionId": "not-the-thread-id"}},
            )

        def run_turn(self, thread_id, input_items, params=None):
            threads[thread_id] += 1
            turn_id = f"turn-{threads[thread_id]}"
            calls.append(("turn", thread_id, deepcopy(params)))
            event = {
                "threadId": thread_id,
                "turnId": turn_id,
                "item": {
                    "id": f"message-{turn_id}",
                    "type": "agentMessage",
                    "text": f"reply-{threads[thread_id]}",
                    "phase": "final_answer",
                },
            }
            self.options["on_event"]("item/completed", event)
            result = {
                "threadId": thread_id,
                "turn": {"id": turn_id, "status": "completed", "items": []},
            }
            self.options["on_event"]("turn/completed", deepcopy(result))
            return self._result("run_turn", result)

    monkeypatch.setattr("agent_composer.nodes.agent.codex.CodexWorkerClient", Worker)

    def forbidden(*_args, **_kwargs):
        pytest.fail("native execution reached the API model factory")

    monkeypatch.setattr(clients, "model_from_config", forbidden)
    return threads, calls, instances


def _flow(policy=None):
    source = _FLOW
    if policy is not None:
        source = source.replace(
            "name: native_continuity",
            f"name: native_continuity\nlong_term_memory: {policy}",
        )
    return load_flow(source)


@pytest.mark.parametrize("policy", [None, "episodic", "reflective"])
def test_non_session_memory_starts_a_fresh_native_thread(native, policy):
    backends = {"episodic": EpisodicMemory, "reflective": ReflectiveMemory}
    memory = None if policy is None else backends[policy]()
    try:
        flow = _flow(policy)
        for question in ("first", "second"):
            result = run_flow(
                flow,
                {"question": question},
                memory=None if memory is None else {policy: memory},
            )
            assert result.status == "succeeded", result.failure
            assert result.output == "reply-1"
        threads, calls, instances = native
        assert len(threads) == 2
        assert [call[0] for call in calls] == ["start", "turn", "start", "turn"]
        assert all(worker.closed for worker in instances)
    finally:
        if memory is not None:
            memory.close()


def test_session_memory_resumes_native_thread_and_reset_starts_fresh(native):
    flow = _flow("session")
    with SessionMemory() as memory:
        for index in range(2):
            result = run_flow(
                flow, {"question": str(index)}, memory={"session": memory}
            )
            assert result.status == "succeeded", result.failure
            assert result.output == f"reply-{index + 1}"
        memory.reset()
        result = run_flow(flow, {"question": "new"}, memory={"session": memory})
        assert result.status == "succeeded", result.failure
        assert result.output == "reply-1"
    threads, calls, instances = native
    assert len(threads) == 2
    assert [call[:2] for call in calls if call[0] != "turn"] == [
        ("start", "thread-1"),
        ("resume", "thread-1"),
        ("start", "thread-2"),
    ]
    assert all(worker.closed for worker in instances)


def test_durable_native_reference_survives_reopen_and_model_change(native, tmp_path):
    path = tmp_path / "session.sqlite"
    for index in range(2):
        flow = load_flow(
            _FLOW.replace("offline-codex", f"offline-codex-{index}").replace(
                "name: native_continuity",
                "name: native_continuity\nlong_term_memory: session",
            )
        )
        with SessionMemory(storage=path) as memory:
            result = run_flow(
                flow, {"question": str(index)}, memory={"session": memory}
            )
            assert result.status == "succeeded", result.failure
            assert result.output == f"reply-{index + 1}"
            with memory.open_run() as opened:
                reference = opened.snapshot["conversations"][0]["reference"]
                assert reference == {"provider": "codex", "session_id": "thread-1"}
    threads, calls, instances = native
    assert len(threads) == 1
    assert [call[0] for call in calls] == ["start", "turn", "resume", "turn"]
    assert all(worker.closed for worker in instances)


@pytest.mark.parametrize("kind", ["loop", "map"])
def test_graph_repetition_keeps_one_queued_native_conversation(native, kind):
    body = """id: native-repeat
name: native_repeat
long_term_memory: session
llm_config: {provider: codex, model: offline-codex}
defs:
  body:
    input: {question: str}
    nodes:
      answer:
        kind: agent
        mode: tool_calling
        tools: [builtin_tools]
        input: {question: '${input.question}'}
        prompt: ${question}
        output: str
      carry:
        kind: code
        input: {answer: '${answer.output}'}
        code: "return {'question': inputs['answer']}"
        output: {question: str}
    output: ${carry.output}
nodes:
  repeat:
"""
    if kind == "loop":
        source = body + """    kind: loop
    call: body
    input: {question: first}
    times: 3
output: ${repeat.output.question}
"""
    else:
        source = body + """    kind: map
    call: body
    over: [first, second, third]
    parallel: true
    input: {question: '${item}'}
output: ${repeat.output}
"""
    with SessionMemory() as memory:
        result = run_flow(
            load_flow(source), {}, memory={"session": memory}, num_workers=3
        )
        assert result.status == "succeeded", result.failure
        if kind == "loop":
            assert result.output == "reply-3"
        else:
            assert sorted(item["question"] for item in result.output) == [
                "reply-1",
                "reply-2",
                "reply-3",
            ]
        with memory.open_run() as opened:
            assert len(opened.snapshot["conversations"]) == 1
    threads, calls, instances = native
    assert len(threads) == 1
    assert [call[0] for call in calls] == [
        "start",
        "turn",
        "resume",
        "turn",
        "resume",
        "turn",
    ]
    assert all(worker.closed and not worker.overlapped for worker in instances)


@pytest.mark.parametrize("feedback", [None, "Improve this answer."])
def test_real_native_forward_still_rejects_backward_before_any_effect(native, feedback):
    result = run_flow(_flow(), {"question": "first"}, learnable=True)
    assert result.status == "succeeded", result.failure
    _, calls, instances = native
    before = deepcopy(calls)
    with pytest.raises(ValueError, match="Codex.*forward-only"):
        result.backward(feedback)
    assert calls == before
    assert all(worker.closed for worker in instances)


def _pinned_worker():
    try:
        installed = version("openai-codex")
    except PackageNotFoundError:
        pytest.skip("offline SDK integration requires the codex extra")
    if installed != "0.155.1":
        pytest.skip("offline SDK integration requires openai-codex==0.155.1")
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    return CodexWorkerClient


@pytest.mark.parametrize("scenario", ["normal", "callback"])
def test_native_flow_uses_pinned_sdk_and_resumes_durable_thread_offline(
    monkeypatch, tmp_path, scenario
):
    client_type = _pinned_worker()
    from agent_composer.nodes.agent import codex

    audit = tmp_path / "native.jsonl"
    server = Path(__file__).resolve().parents[1] / "fixtures" / "codex_app_server.py"

    def worker(**options):
        return client_type(
            **options,
            _launch_args_override=(
                sys.executable,
                "-u",
                str(server),
                scenario,
                str(audit),
            ),
        )

    def forbidden(*_args, **_kwargs):
        pytest.fail("native SDK flow reached the API model factory")

    monkeypatch.setattr(codex, "CodexWorkerClient", worker)
    monkeypatch.setattr(clients, "model_from_config", forbidden)
    tool_calls = []

    def lookup(value: int) -> dict:
        tool_calls.append(value)
        return {"found": value + 1}

    tool = StructuredTool.from_function(
        lookup, name="lookup", description="Find the next value."
    )
    monkeypatch.setitem(TOOL_REGISTRY, "lookup", tool)
    source = _FLOW.replace(
        "name: native_continuity",
        "name: native_continuity\nlong_term_memory: session",
    )
    if scenario == "callback":
        source = source.replace("[builtin_tools]", "[lookup, builtin_tools]")
    flow = load_flow(source)
    for question in ("first", "second"):
        with SessionMemory(storage=tmp_path / "session.sqlite") as memory:
            result = run_flow(flow, {"question": question}, memory={"session": memory})
            assert result.status == "succeeded", result.failure
            assert result.output == "offline answer"
            with memory.open_run() as opened:
                assert opened.snapshot["conversations"][0]["reference"] == {
                    "provider": "codex",
                    "session_id": "thread-1",
                }
    messages = [json.loads(line) for line in audit.read_text().splitlines()]
    session_calls = [
        message
        for message in messages
        if message.get("method") in {"thread/start", "thread/resume"}
    ]
    assert [message["method"] for message in session_calls] == [
        "thread/start",
        "thread/resume",
    ]
    assert session_calls[1]["params"]["threadId"] == "thread-1"
    assert all(message["params"]["sandbox"] == "read-only" for message in session_calls)
    assert "dynamicTools" not in session_calls[1]["params"]
    descriptors = session_calls[0]["params"].get("dynamicTools", [])
    if scenario == "callback":
        assert descriptors == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Find the next value.",
                "inputSchema": tool.tool_call_schema.model_json_schema(),
            }
        ]
        replies = [
            message["result"]
            for message in messages
            if message.get("id") == "callback-1" and "result" in message
        ]
        assert len(replies) == 2
        for reply in replies:
            assert reply["success"] is True
            assert len(reply["contentItems"]) == 1
            content = reply["contentItems"][0]
            assert content["type"] == "inputText"
            assert content["text"] == str({"found": 2})
        assert tool_calls == [1, 1]
    else:
        assert descriptors == []
        assert tool_calls == []


def test_offline_sdk_failure_has_sanitized_guidance_in_normal_diagnostic(monkeypatch):
    client_type = _pinned_worker()
    from agent_composer.nodes.agent import codex

    server = Path(__file__).resolve().parents[1] / "fixtures" / "codex_app_server.py"

    def worker(**options):
        return client_type(
            **options,
            _launch_args_override=(sys.executable, "-u", str(server), "init_failure"),
        )

    def forbidden(*_args, **_kwargs):
        pytest.fail("failed native SDK initialization reached the API model factory")

    monkeypatch.setattr(codex, "CodexWorkerClient", worker)
    monkeypatch.setattr(clients, "model_from_config", forbidden)
    flow = _flow()
    result = run_flow(flow, {"question": "offline request"})
    assert result.status == "failed"
    diagnostic = flow.diagnose(result.failure)
    for word in ("configuration", "runtime", "login"):
        assert word in diagnostic.message
    assert "secret-marker" not in diagnostic.message
    assert "authentication was rejected" not in diagnostic.message
