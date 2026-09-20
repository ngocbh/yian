"""Real AGENT execution preserves memory ownership across runs and graph growth."""

from copy import deepcopy
from functools import wraps
import multiprocessing
from threading import Barrier, Event, Lock, Thread
import traceback

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import agent_composer.llm_clients as llm_clients
from agent_composer import load_flow, run_flow
from agent_composer.memory import MemoryOperationError
from agent_composer.memory.conversation import conversation_view_from_json
from agent_composer.memory.session import SessionMemory
from agent_composer.nodes.agent.conversation import decode_message
from agent_composer.nodes.base import _HostStop
from agent_composer.run import _run_flow_controlled
from tests.engine._memory_fakes import _Memory


_FLOW = """id: execution-session
name: execution_session
long_term_memory: session
llm_config: {provider: openai, model: fake-model}
input: {question: str}
nodes:
  answer:
    kind: agent
    mode: plain
    input: {question: "${input.question}"}
    prompt: ${question}
    output: str
output: ${answer.output}
"""


class _Model:
    def __init__(self, invoke):
        self.invoke = invoke


class _Worker:
    def __init__(self, target):
        self.results = []
        self.errors = []

        def run():
            try:
                self.results.append(target())
            except BaseException as error:
                self.errors.append(error)

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()

    def join(self):
        self.thread.join(5)
        assert not self.thread.is_alive(), "run worker did not finish"


def _scenario(name, connection):
    try:
        globals()[name].__wrapped__()
    except BaseException:
        connection.send(traceback.format_exc()[-6000:])
    else:
        connection.send(None)
    finally:
        connection.close()


def _bounded(function):
    """Keep a queue or scheduler deadlock from hanging the test process."""
    @wraps(function)
    def run():
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_scenario, args=(function.__name__, writer))
        process.start()
        writer.close()
        try:
            process.join(20)
            assert not process.is_alive(), "execution scenario exceeded its deadline"
            assert reader.poll(1), f"scenario exited without a result: {process.exitcode}"
            error = reader.recv()
            assert error is None, error
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            reader.close()
            if not process.is_alive():
                process.close()
    return run


def _wait_queue(memory, count):
    # Observe actual reserved tickets, not thread-start timing or sleeps.
    with memory._condition:
        assert memory._condition.wait_for(
            lambda: sum(len(queue) for queue in memory._queues.values()) == count,
            timeout=5,
        )


def _snapshot(memory):
    with memory.open_run() as opened:
        return opened.snapshot


def _views(memory):
    return [conversation_view_from_json(raw) for raw in _snapshot(memory)["conversations"]]


def _humans(messages):
    return [message.content for message in messages if isinstance(message, HumanMessage)]


@_bounded
def test_same_authored_agent_queues_real_runs_in_admission_order():
    memory, entered, release = SessionMemory(), Event(), Event()
    flow = load_flow(_FLOW)
    calls, workers = [], []
    active, lock = 0, Lock()

    def invoke(messages):
        nonlocal active
        with lock:
            active += 1
            assert active == 1, "same conversation overlapped model calls"
            calls.append(deepcopy(messages))
        try:
            question = messages[-1].content
            if question == "first":
                entered.set()
                assert release.wait(5)
            return AIMessage(content=f"reply:{question}")
        finally:
            with lock:
                active -= 1

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(llm_clients, "model_from_config", lambda _: _Model(invoke))
        try:
            workers.append(_Worker(lambda: run_flow(flow, {"question": "first"}, memory={"session": memory})))
            assert entered.wait(5)
            for count, question in enumerate(("second", "third"), 1):
                workers.append(_Worker(lambda question=question: run_flow(
                    flow, {"question": question}, memory={"session": memory}, num_workers=2)))
                _wait_queue(memory, count)
            before = _snapshot(memory)
            with pytest.raises(MemoryOperationError) as caught:
                memory.reset()
            assert caught.value.code == "session.busy"
            assert _snapshot(memory) == before
            assert len(calls) == 1
        finally:
            release.set()
            for worker in workers:
                worker.join()
        assert all(not worker.errors for worker in workers)
        assert [worker.results[0].output for worker in workers] == [
            "reply:first", "reply:second", "reply:third",
        ]
    assert [_humans(messages) for messages in calls] == [
        ["first"], ["first", "second"], ["first", "second", "third"],
    ]
    view, = _views(memory)
    assert len(view.closed_invocations) == 3
    assert all(closed.status == "completed" for closed in view.closed_invocations)
    memory.reset()
    assert _views(memory) == []


@_bounded
def test_distinct_authored_leaves_overlap_under_one_flow_session_frame():
    memory, barrier = SessionMemory(), Barrier(2)
    flow = load_flow("""id: two-leaves
name: two_leaves
long_term_memory: session
llm_config: {provider: openai, model: fake-model}
nodes:
  left: {kind: agent, mode: plain, prompt: left, output: str}
  right: {kind: agent, mode: plain, prompt: right, output: str}
output: ${[left.output, right.output]}
""")

    def invoke(messages):
        barrier.wait(5)
        return AIMessage(content=messages[-1].content)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(llm_clients, "model_from_config", lambda _: _Model(invoke))
        result = run_flow(flow, {}, memory={"session": memory}, num_workers=2)
    assert result.status == "succeeded", result.failure
    assert result.output == ["left", "right"]
    views = _views(memory)
    assert {view.key.consumer_static_path for view in views} == {
        ("two-leaves", "left"), ("two-leaves", "right"),
    }
    assert len({view.key.frame_ref for view in views}) == 1


@_bounded
def test_cancelled_waiter_exits_but_active_run_keeps_lease_until_reply_is_saved():
    memory, flow = SessionMemory(), load_flow(_FLOW)
    entered, release, owner_stop, waiter_stop = (Event() for _ in range(4))
    calls, workers = [], []

    def invoke(messages):
        question = messages[-1].content
        calls.append(question)
        if question == "owner":
            entered.set()
            assert release.wait(5)
        return AIMessage(content=f"reply:{question}")

    def controlled(question, stop):
        return _run_flow_controlled(flow, {"question": question},
                                    memory={"session": memory}, stop_event=stop, num_workers=2)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(llm_clients, "model_from_config", lambda _: _Model(invoke))
        try:
            workers.append(_Worker(lambda: controlled("owner", owner_stop)))
            assert entered.wait(5)
            workers.append(_Worker(lambda: controlled("cancelled", waiter_stop)))
            _wait_queue(memory, 1)
            waiter_stop.set()
            workers[1].join()
            assert len(workers[1].errors) == 1 and isinstance(workers[1].errors[0], _HostStop)
            _wait_queue(memory, 0)
            owner_stop.set()
            workers.append(_Worker(lambda: run_flow(flow, {"question": "next"}, memory={"session": memory})))
            _wait_queue(memory, 1)
            assert calls == ["owner"]
            with pytest.raises(MemoryOperationError, match="active or queued"):
                memory.reset()
        finally:
            release.set()
            for worker in workers:
                worker.join()
    assert len(workers[0].errors) == 1 and isinstance(workers[0].errors[0], _HostStop)
    assert not workers[2].errors and workers[2].results[0].output == "reply:next"
    assert calls == ["owner", "next"]
    view, = _views(memory)
    assert _humans([decode_message(item) for item in view.items]) == ["owner", "next"]
    assert [effect.status for effect in view.effects] == ["completed", "completed"]
    assert [closed.status for closed in view.closed_invocations] == ["interrupted", "completed"]
    assert any(decode_message(item).content == "reply:owner" for item in view.items)


@pytest.mark.parametrize("outer_session,local_session", [(True, False), (False, True), (True, True)])
def test_innermost_session_owns_history_without_hiding_other_recall(monkeypatch, outer_session, local_session):
    outer = SessionMemory() if outer_session else _Memory(recall_text="outer recall")
    local = SessionMemory() if local_session else _Memory(recall_text="local recall")
    flow = load_flow(_FLOW.replace("long_term_memory: session", "long_term_memory: outer")
                     .replace("    mode: plain", "    mode: plain\n    long_term_memory: local"))
    calls = []

    def invoke(messages):
        calls.append(deepcopy(messages))
        return AIMessage(content="answer")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model(invoke))
    for question in ("first", "second"):
        assert run_flow(flow, {"question": question}, memory={"outer": outer, "local": local}).status == "succeeded"
    assert _humans(calls[1]) == ["first", "second"]
    if not outer_session:
        assert "outer recall" in calls[1][0].content and len(outer.recall_contexts) == 2
    if not local_session:
        assert "local recall" in calls[1][0].content and len(local.recall_contexts) == 2
    owner = local if local_session else outer
    view, = _views(owner)
    assert view.key.frame_ref.static_path == (("execution-session", "answer") if local_session else ("execution-session",))
    if outer_session and local_session:
        assert _views(outer) == []


def test_loop_iterations_share_one_authored_conversation(monkeypatch):
    flow = load_flow("""id: session-loop
name: session_loop
long_term_memory: session
llm_config: {provider: openai, model: fake-model}
defs:
  body:
    input: {question: str}
    nodes:
      answer:
        kind: agent
        mode: plain
        input: {question: "${input.question}"}
        prompt: ${question}
        output: str
      carry:
        kind: code
        input: {answer: "${answer.output}"}
        code: "return {'question': inputs['answer']}"
        output: {question: str}
    output: ${carry.output}
nodes:
  repeat:
    kind: loop
    call: body
    input: {question: first}
    times: 3
output: ${repeat.output.question}
""")
    memory, calls = SessionMemory(), []

    def invoke(messages):
        calls.append(deepcopy(messages))
        return AIMessage(content=f"reply-{len(calls)}")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model(invoke))
    result = run_flow(flow, {}, memory={"session": memory})
    assert result.status == "succeeded", result.failure
    assert result.output == "reply-3"
    assert [_humans(messages) for messages in calls] == [
        ["first"], ["first", "reply-1"], ["first", "reply-1", "reply-2"],
    ]
    view, = _views(memory)
    assert len(view.closed_invocations) == 3
    assert view.key.consumer_static_path == ("session-loop", "repeat", "answer")


@_bounded
def test_parallel_map_shares_authored_history_without_changing_result_order():
    flow = load_flow("""id: session-map
name: session_map
long_term_memory: session
llm_config: {provider: openai, model: fake-model}
input: {questions: "list[str]"}
defs:
  body:
    input: {question: str}
    nodes:
      answer:
        kind: agent
        mode: plain
        input: {question: "${input.question}"}
        prompt: ${question}
        output: str
    output: ${answer.output}
nodes:
  each:
    kind: map
    call: body
    parallel: true
    over: ${input.questions}
    input: {question: "${item}"}
output: ${each.output}
""")
    memory, calls = SessionMemory(), []

    def invoke(messages):
        calls.append(deepcopy(messages))
        return AIMessage(content=f"reply:{messages[-1].content}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(llm_clients, "model_from_config", lambda _: _Model(invoke))
        result = run_flow(flow, {"questions": ["one", "two", "three"]},
                          memory={"session": memory}, num_workers=3)
    assert result.status == "succeeded", result.failure
    assert result.output == ["reply:one", "reply:two", "reply:three"]
    # Admission order may differ from MAP input order; each admitted call sees
    # all preceding actual calls in their observed execution order.
    admitted = [messages[-1].content for messages in calls]
    assert sorted(admitted) == ["one", "three", "two"]
    assert [_humans(messages) for messages in calls] == [admitted[:index] for index in (1, 2, 3)]
    view, = _views(memory)
    assert len(view.closed_invocations) == 3
    assert view.key.consumer_static_path == ("session-map", "each", "answer")


def test_two_callsites_of_one_child_keep_separate_conversations(monkeypatch):
    flow = load_flow("""id: session-calls
name: session_calls
long_term_memory: session
llm_config: {provider: openai, model: fake-model}
defs:
  body:
    input: {question: str}
    nodes:
      answer:
        kind: agent
        mode: plain
        input: {question: "${input.question}"}
        prompt: ${question}
        output: str
    output: ${answer.output}
nodes:
  left: {kind: call, call: body, input: {question: left}}
  right: {kind: call, call: body, input: {question: right}}
output: ${[left.output, right.output]}
""")
    memory, calls = SessionMemory(), []

    def invoke(messages):
        calls.append(deepcopy(messages))
        return AIMessage(content=messages[-1].content)

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model(invoke))
    for _ in range(2):
        result = run_flow(flow, {}, memory={"session": memory})
        assert result.status == "succeeded", result.failure
        assert result.output == ["left", "right"]
    assert sorted(_humans(messages) for messages in calls) == [
        ["left"], ["left", "left"], ["right"], ["right", "right"],
    ]
    assert {view.key.consumer_static_path for view in _views(memory)} == {
        ("session-calls", "left", "answer"), ("session-calls", "right", "answer"),
    }
