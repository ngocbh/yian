"""Opaque context summaries never replace backend-owned raw conversation truth."""

import json
import multiprocessing
from threading import Event, Thread
import traceback

import pytest

from agent_composer.memory.session import SessionMemory
from agent_composer.memory import MemoryFrameSpec, MemoryOperationError
from agent_composer.nodes.base import _HostStop
from tests.engine._session_fakes import REF, append_reply, harness, request, snapshot


def _seed(memory, count, *, consumer=("root", "agent")):
    with harness(memory) as opened:
        for index in range(count):
            with opened.frame.open_turn(
                request(str(index), consumer=consumer), check_stopped=lambda: None
            ) as turn:
                append_reply(turn, f"item-{consumer[-1]}-{index}", "x" * 20)


def _project(view, invocations):
    return json.dumps([invocation.run_id for invocation in invocations])


def _prepare(turn, calls, *, version="test-v1", result="summary", project=_project):
    def summarize(text, target):
        calls.append((text, target))
        return result

    return turn.prepare_history(
        projection_version=version, project=project, summarize=summarize
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("history_budget_chars", 0),
        ("history_budget_chars", -1),
        ("history_budget_chars", True),
        ("history_budget_chars", 1.5),
        ("recent_turns", -1),
        ("recent_turns", False),
        ("recent_turns", 1.0),
    ],
)
def test_settings_require_exact_integer_bounds(field, value):
    with pytest.raises((TypeError, ValueError)):
        SessionMemory(**{field: value})


def test_defaults_and_below_budget_do_not_summarize():
    memory, calls = SessionMemory(), []
    _seed(memory, 8)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            assert _prepare(turn, calls) == (None, json.dumps(list(map(str, range(8)))))
    assert calls == []


@pytest.mark.parametrize("recent", [0, 4])
def test_whole_prior_turns_and_current_are_separate(recent):
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=recent), []
    _seed(memory, 6)
    before = snapshot(memory)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            summary, remaining = _prepare(turn, calls)
            assert summary == "summary"
            assert json.loads(remaining) == list(map(str, range(6 - recent, 6)))
            assert len(calls) == 1 and calls[0][1] == 1
            assert "current" not in calls[0][0]
            for index in range(6 - recent):
                assert f'"{index}"' in calls[0][0]
    after = snapshot(memory)
    assert after["conversations"][0]["items"] == before["conversations"][0]["items"]
    assert after["notes"] == before["notes"]


def test_oversized_summary_reuses_same_prefix_and_extends_only_new_turns():
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=1), []
    _seed(memory, 3)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            expected = _prepare(turn, calls, result="oversized summary")
            assert _prepare(turn, calls) == expected
            assert len(calls) == 1
        with opened.frame.open_turn(
            request("next"), check_stopped=lambda: None
        ) as turn:
            assert _prepare(turn, calls, result="extended")[0] == "extended"
    assert len(calls) == 2
    assert "oversized summary" in calls[1][0]
    assert '"2"' in calls[1][0]
    assert '"0"' not in calls[1][0] and '"1"' not in calls[1][0]


def test_no_eligible_older_turn_does_not_call_summary():
    memory, calls = SessionMemory(history_budget_chars=1), []
    _seed(memory, 4)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            assert _prepare(turn, calls)[0] is None
    assert calls == []


@pytest.mark.parametrize("bad", [None, "", "   ", 42, ["text"]])
def test_invalid_summary_retains_raw_history_and_previous_cache(bad):
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=0), []
    _seed(memory, 1)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            before = turn.view
            with pytest.raises((TypeError, ValueError)):
                _prepare(turn, calls, result=bad)
            assert turn.view == before
            assert _prepare(turn, calls, result="valid")[0] == "valid"
            assert len(calls) == 2


def test_summary_exception_identity_and_no_candidate_publication():
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory, 1)
    error = RuntimeError("summary callback failed")
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:

            def fail(text, target):
                raise error

            with pytest.raises(RuntimeError) as caught:
                turn.prepare_history(
                    projection_version="test-v1", project=_project, summarize=fail
                )
            assert caught.value is error
            calls = []
            assert _prepare(turn, calls)[0] == "summary" and len(calls) == 1


def test_defaults_keep_four_prior_turns_even_over_budget():
    memory, calls = SessionMemory(), []
    _seed(memory, 5)

    def large_project(view, selected):
        return "x" * (10_000 * len(selected))

    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            summary, recent = _prepare(turn, calls, project=large_project)
            assert summary == "summary" and len(recent) == 40_000
    assert len(calls) == 1 and calls[0][1] == 32_000
    assert calls[0][0].split("\n", 1)[1] == "x" * 10_000


def test_cache_is_private_versioned_and_does_not_rewind_to_old_pin():
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=0), []
    _seed(memory, 1)
    pinned = snapshot(memory)
    with memory.open_run(snapshot=pinned) as session:
        frame = session.open_frame(MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request("current"), check_stopped=lambda: None) as turn:
            assert _prepare(turn, calls, result="first")[0] == "first"
            assert (
                _prepare(turn, calls, version="test-v2", result="second")[0] == "second"
            )
            assert _prepare(turn, calls)[0] == "first"
            assert len(calls) == 2
        assert session.snapshot == pinned
    with memory.open_run(snapshot=pinned) as session:
        assert session.snapshot == pinned
        frame = session.open_frame(MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request("later"), check_stopped=lambda: None) as turn:
            assert _prepare(turn, calls, result="extended")[0] == "extended"
    assert "first" in calls[-1][0] and '"current"' in calls[-1][0]
    assert '"0"' not in calls[-1][0]
    assert set(snapshot(memory)) == {
        "version",
        "namespace",
        "generation",
        "conversations",
        "notes",
    }
    memory.reset()
    _seed(memory, 1)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("fresh"), check_stopped=lambda: None
        ) as turn:
            assert _prepare(turn, calls)[0] == "summary"
    assert "Previous historical summary" not in calls[-1][0]


def test_cache_isolated_by_leaf_and_namespace():
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=0), []
    _seed(memory, 1)
    _seed(memory, 1, consumer=("root", "other"))
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            _prepare(turn, calls, result="agent summary")
        with opened.frame.open_turn(
            request("current", sequence=2, consumer=("root", "other")),
            check_stopped=lambda: None,
        ) as turn:
            assert _prepare(turn, calls, result="other summary")[0] == "other summary"
    assert len(calls) == 2 and "agent summary" not in calls[-1][0]
    restored = SessionMemory(history_budget_chars=1, recent_turns=0)
    with restored.open_run(snapshot=snapshot(memory)) as session:
        frame = session.open_frame(MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request("restored"), check_stopped=lambda: None) as turn:
            _prepare(turn, calls)
    assert "Previous historical summary" not in calls[-1][0]


def test_cached_summary_under_budget_does_not_extend_eligible_prefix():
    memory, calls = SessionMemory(history_budget_chars=20, recent_turns=0), []
    _seed(memory, 5)
    with harness(memory) as opened:
        with opened.frame.open_turn(request("a"), check_stopped=lambda: None) as turn:
            assert _prepare(turn, calls, result="s") == ("s", "[]")
        with opened.frame.open_turn(request("b"), check_stopped=lambda: None) as turn:
            assert _prepare(turn, calls) == ("s", '["a"]')
    assert len(calls) == 1


def test_failed_extension_keeps_previous_summary():
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=0), []
    _seed(memory, 1)
    with harness(memory) as opened:
        with opened.frame.open_turn(request("a"), check_stopped=lambda: None) as turn:
            _prepare(turn, calls, result="original summary")
        with opened.frame.open_turn(request("b"), check_stopped=lambda: None) as turn:
            with pytest.raises(ValueError):
                _prepare(turn, calls, result="")
            _prepare(turn, calls, result="valid extension")
    assert "original summary" in calls[-1][0]
    assert calls[-1][0] == calls[-2][0]


def test_empty_crashed_and_failed_turns_count_in_recent_selection():
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=2), []
    _seed(memory, 1)
    with harness(memory) as opened:
        with pytest.raises(RuntimeError):
            with opened.frame.open_turn(request("failed"), check_stopped=lambda: None):
                raise RuntimeError("empty failed turn")
        turn = opened.frame.open_turn(request("crashed"), check_stopped=lambda: None)
        turn.__enter__()
        crashed_pin = snapshot(memory)
        turn.__exit__(None, None, None)
    restored = SessionMemory(history_budget_chars=1, recent_turns=2)
    with restored.open_run(snapshot=crashed_pin) as session:
        frame = session.open_frame(MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request("current"), check_stopped=lambda: None) as turn:
            assert _prepare(turn, calls)[1] == '["failed", "crashed"]'
    assert '"0"' in calls[0][0]


def test_callback_history_change_rejects_candidate_without_rewinding():
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=0), []
    _seed(memory, 1)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:

            def changed(text, target):
                append_reply(turn, "new", "real evidence")
                return "not publishable"

            with pytest.raises(MemoryOperationError, match="history changed"):
                turn.prepare_history(
                    projection_version="test-v1", project=_project, summarize=changed
                )
            assert turn.view.items[-1].payload["content"] == "real evidence"
            assert _prepare(turn, calls)[0] == "summary" and len(calls) == 1


def test_close_during_projection_prevents_starting_summary_callback():
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=0), []
    _seed(memory, 1)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:

            def close(view, selected):
                opened.session.close()
                return _project(view, selected)

            with pytest.raises(MemoryOperationError):
                _prepare(turn, calls, project=close)
    assert calls == []


@pytest.mark.parametrize("version", [1, 2])
def test_snapshot_pins_and_learned_notes_survive_summary_and_note_commit(version):
    from tests.engine.test_session_notes import _mutation, _note

    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory, 1)
    value = snapshot(memory)
    if version == 1:
        value = {key: item for key, item in value.items() if key != "notes"}
        value["version"] = 1
    else:
        value["notes"] = [_note(run_id="0", text="Already learned")]
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    with memory.open_run(snapshot=value) as session:
        frame = session.open_frame(MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request("current"), check_stopped=lambda: None) as turn:
            original = turn.view
            note = _note(run_id="0", index=1, text="Learned during callback")

            def summarize(text, target):
                session.commit("notes", (_mutation(note),))
                return "summary"

            assert (
                turn.prepare_history(
                    projection_version="test-v1", project=_project, summarize=summarize
                )[0]
                == "summary"
            )
            assert turn.view == original
        assert session.snapshot == value
    final = snapshot(memory)
    assert final["notes"] == value.get("notes", []) + [note]
    assert final["conversations"][0]["items"] == value["conversations"][0]["items"]


@pytest.mark.parametrize("bad", [None, 1, [], {}])
def test_invalid_projection_does_not_call_summary_or_poison_cache(bad):
    memory, calls = SessionMemory(history_budget_chars=1, recent_turns=0), []
    _seed(memory, 1)
    with harness(memory) as opened:
        with opened.frame.open_turn(
            request("current"), check_stopped=lambda: None
        ) as turn:
            before = turn.view
            with pytest.raises(TypeError):
                _prepare(turn, calls, project=lambda view, selected: bad)
            assert calls == [] and turn.view == before
            assert _prepare(turn, calls)[0] == "summary"


def _concurrency_scenario(action, connection):
    try:
        memory = SessionMemory(history_budget_chars=1, recent_turns=1)
        _seed(memory, 2)
        entered, release, stop = Event(), Event(), Event()
        queued_stop, queued_finished = Event(), Event()
        events, errors, sessions, workers = [], [], [], []
        primary = _HostStop(primary=False)
        queued_error = RuntimeError("queued cancelled")

        def owner():
            try:
                with harness(memory) as opened:
                    sessions.append(opened.session)

                    def check():
                        if stop.is_set():
                            raise primary

                    with opened.frame.open_turn(
                        request("owner"), check_stopped=check
                    ) as turn:

                        def summarize(text, target):
                            entered.set()
                            assert release.wait(5)
                            events.append("summary returned")
                            return "retained actual summary"

                        turn.prepare_history(
                            projection_version="test-v1",
                            project=_project,
                            summarize=summarize,
                        )
                        events.append("owner complete")
            except BaseException as error:
                errors.append(error)

        def waiter(name, cancelled=False):
            try:
                with harness(memory) as opened:

                    def check():
                        if cancelled and queued_stop.is_set():
                            raise queued_error

                    with opened.frame.open_turn(
                        request(name), check_stopped=check
                    ) as turn:
                        events.append(name)
                        assert "summary returned" in events
                        assert any(
                            entry.text == "retained actual summary"
                            for entry in memory._summaries.values()
                        )
            except BaseException as error:
                errors.append(error)
            finally:
                if cancelled:
                    queued_finished.set()

        def launch(target):
            worker = Thread(target=target, daemon=True)
            workers.append(worker)
            worker.start()

        launch(owner)
        assert entered.wait(5)
        try:
            # A distinct key can complete while callback ownership stays held;
            # this also proves the callback is outside the backend condition.
            with harness(memory) as opened:
                with opened.frame.open_turn(
                    request("other", consumer=("root", "other")),
                    check_stopped=lambda: None,
                ):
                    events.append("other")
            launch(lambda: waiter("cancelled", True))
            with memory._condition:
                assert memory._condition.wait_for(
                    lambda: sum(map(len, memory._queues.values())) == 1, 5
                )
            launch(lambda: waiter("next"))
            with memory._condition:
                assert memory._condition.wait_for(
                    lambda: sum(map(len, memory._queues.values())) == 2, 5
                )
            before = snapshot(memory)
            with pytest.raises(MemoryOperationError) as busy:
                memory.reset()
            assert busy.value.code == "session.busy" and snapshot(memory) == before
            queued_stop.set()
            with memory._condition:
                memory._condition.notify_all()
            assert queued_finished.wait(5)
            assert queued_error in errors and "cancelled" not in events
            if action == "stop":
                stop.set()
            elif action == "close":
                sessions[0].close()
            assert "next" not in events and "summary returned" not in events
            with pytest.raises(MemoryOperationError):
                memory.reset()
        finally:
            release.set()
            for worker in workers:
                worker.join(5)
                assert not worker.is_alive()
        assert events.index("summary returned") < events.index("next")
        if action == "stop":
            assert primary in errors and "owner complete" not in events
        elif action == "close":
            assert any(isinstance(error, MemoryOperationError) for error in errors)
            assert "owner complete" not in events
        else:
            assert errors == [queued_error]
        memory.reset()
        connection.send(None)
    except BaseException:
        connection.send(traceback.format_exc())
    finally:
        connection.close()


@pytest.mark.parametrize("action", ["normal", "stop", "close"])
def test_callback_drains_before_fifo_lease_release(action):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_concurrency_scenario, args=(action, writer))
    process.start()
    writer.close()
    try:
        process.join(25)
        assert not process.is_alive(), "summary scenario exceeded deadline"
        assert reader.poll(1), f"scenario exited without result: {process.exitcode}"
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
