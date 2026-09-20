"""Process-local session memory with backend-owned atomic turns."""

import pytest
from dataclasses import replace
from threading import Event, Thread
from functools import wraps
import multiprocessing
import traceback

import agent_composer.memory.contracts as c
from agent_composer.memory.session import SessionMemory
from agent_composer.memory.conversation import (
    advance_conversation,
    conversation_view_to_json,
)
from tests.engine._session_fakes import append_reply, harness, item, request, snapshot

REF = c.MemoryFrameRef("root", ("root",), c.MemoryOwner.FLOW)


def test_turn_publishes_history_without_a_checkpoint_sink():
    memory = SessionMemory()
    with memory.open_run() as session:
        frame = session.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(request(), check_stopped=lambda: None) as turn:
            assert turn.view.revision == 1
            ordinal = turn.request("model", {"prompt": "hi"}, items=(item("u", "hi"),))
            assert snapshot(memory)["conversations"][0]["revision"] == 2
            turn.observe(ordinal, "completed", items=(item("a", "hello"),))
            assert turn.view.effects[0].status == "completed"
        saved = snapshot(memory)
        assert saved["conversations"][0]["revision"] == 4
        assert [entry["item_id"] for entry in saved["conversations"][0]["items"]] == [
            "u",
            "a",
        ]


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_new_turn_preserves_incomplete_history_and_native_reference(restore, closed):
    memory = SessionMemory()
    reference = c.ConversationReference("codex", "opaque-thread")
    with memory.open_run() as session:
        frame = session.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        turn = frame.open_turn(request("a"), check_stopped=lambda: None)
        turn.__enter__()
        ordinal = turn.request("native_session", {}, items=(item("a-in", "first"),))
        turn.observe(
            ordinal, "pending", items=(item("a-out", "partial"),), reference=reference
        )
        if closed:
            primary = RuntimeError("interrupted effect")
            with pytest.raises(RuntimeError) as caught:
                turn.__exit__(RuntimeError, primary, None)
            assert caught.value is primary
        else:
            # A restored prefix can end before its owner recorded closure.
            session._release(turn._lease)
        prefix = snapshot(memory)
    if restore:
        memory = SessionMemory()
        with memory.open_run(snapshot=prefix) as restored:
            assert restored.snapshot == prefix
    with memory.open_run(snapshot=prefix) as session:
        frame = session.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        for name in ("b", "c"):
            with frame.open_turn(request(name), check_stopped=lambda: None) as later:
                assert later.view.reference == reference
                assert later.view.effects[0].status == "pending"
                assert later.view.items[:2] == (
                    item("a-in", "first"),
                    item("a-out", "partial"),
                )
                append_reply(later, name, name)
        assert session.snapshot == prefix
    final = snapshot(memory)
    original = prefix["conversations"][0]
    assert (
        final["conversations"][0]["applied"][: original["revision"]]
        == original["applied"]
    )
    assert [effect["status"] for effect in final["conversations"][0]["effects"]] == [
        "pending",
        "completed",
        "completed",
    ]
    with SessionMemory().open_run(snapshot=final) as reopened:
        assert reopened.snapshot == final
    with pytest.raises(c.MemoryOperationError, match="lease|closed"):
        turn.observe(ordinal, "completed")


def test_session_backend_implements_original_protocol():
    memory = SessionMemory()
    assert isinstance(memory, c.Memory)
    with memory.open_run() as session:
        assert isinstance(session, c.MemorySession)
        frame = session.open_frame(c.MemoryFrameSpec(REF, "custom", {}))
        assert isinstance(frame, c.MemoryFrame)
        assert isinstance(frame, c.ConversationFrame)
        assert session.open_frame(c.MemoryFrameSpec(REF, "custom", {})) is frame


def test_session_frame_rejects_unknown_config():
    with SessionMemory().open_run() as session:
        with pytest.raises(c.MemoryOperationError) as caught:
            session.open_frame(c.MemoryFrameSpec(REF, "session", {"fork": True}))
        assert (caught.value.stage, caught.value.code, caught.value.retryable) == (
            "frame",
            "session.invalid_config",
            False,
        )


def test_snapshot_getter_is_detached():
    with SessionMemory().open_run() as session:
        first = session.snapshot
        first["conversations"].append({"bad": True})
        assert session.snapshot["conversations"] == []


def _live(h):
    return next(iter(h.memory._views.values()))


def _request(run_id="a", sequence=1, consumer=("root", "agent"), compatibility=None):
    return c.ConversationRequest(
        consumer,
        c.ConversationInvocation(run_id, sequence),
        {"format": "fake-v1"} if compatibility is None else compatibility,
    )


def _reserve(session, request=None):
    frame = session.open_frame(c.MemoryFrameSpec(REF, "custom", {}))
    return session._reserve(
        frame, _request() if request is None else request, lambda: None
    )


def test_uncontended_lease_reservation_does_not_change_history():
    memory = SessionMemory()
    with memory.open_run() as session:
        pinned = session.snapshot
        lease = _reserve(session)
        assert lease.active
        assert session.snapshot == pinned
        assert memory._views == {}
        session._release(lease)


def test_first_turn_admission_uses_root_and_consumer():
    with harness(policy="custom") as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            assert turn.view.key == c.ConversationKey(REF, ("root", "agent"))
            assert turn.view.revision == 1
            assert turn._stamp().policy_name == "custom"
            append_reply(turn, "one", "hello")
        assert _live(h).revision == 4


def test_same_agent_reuses_conversation_across_runs():
    memory = SessionMemory()
    with harness(memory) as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            append_reply(turn, "one", "hello")
    with harness(memory) as h:
        with h.frame.open_turn(request("b"), check_stopped=lambda: None) as turn:
            assert turn.view.items == (item("one", "hello"),)


def test_distinct_consumer_has_no_reference_or_history():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            ordinal = turn.request("native_session", {})
            turn.observe(
                ordinal,
                "completed",
                items=(item("one", "hello"),),
                reference=c.ConversationReference("codex", "opaque"),
            )
        with h.frame.open_turn(
            request("b", consumer=("root", "other")),
            check_stopped=lambda: None,
        ) as turn:
            assert turn.view.items == ()
            assert turn.view.reference is None


def test_open_session_snapshot_stays_pinned():
    with harness() as h:
        pinned = h.session.snapshot
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            append_reply(turn, "one", "hello")
        assert h.session.snapshot == pinned
        assert snapshot(h.memory)["conversations"][0]["revision"] == 4


def test_admission_reads_live_view_not_open_snapshot():
    with harness() as first, harness(first.memory) as second:
        assert second.session.snapshot["conversations"] == []
        with first.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            append_reply(turn, "one", "hello")
        with second.frame.open_turn(request("b"), check_stopped=lambda: None) as turn:
            assert turn.view.items == (item("one", "hello"),)


def test_pristine_backend_adopts_snapshot_namespace():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            append_reply(turn, "one", "hello")
        exported = snapshot(h.memory)
    restored = SessionMemory()
    with restored.open_run(snapshot=exported) as session:
        assert session.snapshot == exported
        exported["conversations"].clear()
        assert len(session.snapshot["conversations"]) == 1


def test_used_backend_rejects_foreign_snapshot():
    memory = SessionMemory()
    before = snapshot(memory)
    with pytest.raises(c.MemoryOperationError) as caught:
        memory.open_run(snapshot=snapshot(SessionMemory()))
    assert caught.value.code == "session.invalid_snapshot"
    assert snapshot(memory) == before


def test_old_same_lineage_snapshot_stays_pinned_without_rewind():
    with harness() as h:
        old = h.session.snapshot
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            append_reply(turn, "one", "hello")
        latest = snapshot(h.memory)
        with h.memory.open_run(snapshot=old) as replay:
            assert replay.snapshot == old
            assert snapshot(h.memory) == latest


def test_session_recall_is_empty():
    with harness() as h:
        assert h.frame.recall() == ""
        assert (
            h.frame.recall(c.RecallContext("system", "prompt", {}, ("root", "agent")))
            == ""
        )
        with pytest.raises(TypeError):
            h.frame.recall({})


def test_session_learn_and_commit_accept_empty_only():
    with harness() as h:
        before = snapshot(h.memory)
        assert h.frame.learn(c.LearnContext("a", REF, ())) == []
        h.session.commit("same", ())
        h.session.commit("same", ())
        with pytest.raises(ValueError):
            h.session.commit("same", (c.MemoryMutation("m", REF, {}),))
        with pytest.raises(TypeError):
            h.session.commit("same", [])
        assert snapshot(h.memory) == before


@pytest.mark.parametrize("old,new", [(True, 1), (1, 1.0), (0.0, -0.0)])
def test_incompatible_descriptor_requires_reset(old, new):
    with harness() as h:
        with h.frame.open_turn(
            request(compatibility={"format": old}),
            check_stopped=lambda: None,
        ):
            pass
        before = snapshot(h.memory)
        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(
                request("b", compatibility={"format": new}),
                check_stopped=lambda: None,
            ):
                pytest.fail("incompatible conversation was admitted")
        assert caught.value.code == "session.incompatible"
        assert snapshot(h.memory) == before
        h.memory.reset()
    with harness(h.memory) as fresh:
        with fresh.frame.open_turn(
            request("c", compatibility={"format": new}),
            check_stopped=lambda: None,
        ):
            pass


class _Worker:
    def __init__(self, target):
        self.errors = []
        self.done = Event()

        def run():
            try:
                target()
            except BaseException as exc:
                self.errors.append(exc)
            finally:
                self.done.set()

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()

    def join(self):
        self.thread.join(3)
        assert not self.thread.is_alive(), "session worker did not terminate"


def _run_scenario(name, connection):
    try:
        globals()[name].__wrapped__()
    except BaseException:
        connection.send(traceback.format_exc()[-3000:])
    else:
        connection.send(None)
    finally:
        connection.close()


def _bounded_scenario(function):
    """Contain even regressions that deadlock before a thread can be joined."""

    @wraps(function)
    def bounded():
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(
            target=_run_scenario, args=(function.__name__, writer)
        )
        process.start()
        writer.close()
        try:
            process.join(10)
            assert (
                not process.is_alive()
            ), "session scenario exceeded its process deadline"
            assert reader.poll(
                1
            ), f"scenario exited without a result: {process.exitcode}"
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

    return bounded


def _wait_queue(memory, count):
    with memory._condition:
        assert memory._condition.wait_for(
            lambda: sum(len(queue) for queue in memory._queues.values()) == count,
            timeout=2,
        )


@_bounded_scenario
def test_second_same_key_ticket_waits():
    with harness() as h:
        entered = Event()
        with h.frame.open_turn(request(), check_stopped=lambda: None):

            def second():
                with h.frame.open_turn(request("b"), check_stopped=lambda: None):
                    entered.set()

            worker = _Worker(second)
            _wait_queue(h.memory, 1)
            assert not entered.is_set()
        worker.join()
        assert worker.errors == []
        assert entered.is_set()


@_bounded_scenario
def test_third_ticket_cannot_overtake_second():
    with harness() as h:
        order = []
        workers = []
        with h.frame.open_turn(request(), check_stopped=lambda: None):

            def waiting(name):
                with h.frame.open_turn(request(name), check_stopped=lambda: None):
                    order.append(name)

            for index, name in enumerate(("b", "c"), 1):
                workers.append(_Worker(lambda name=name: waiting(name)))
                _wait_queue(h.memory, index)
        for worker in workers:
            worker.join()
            assert worker.errors == []
        assert order == ["b", "c"]


@_bounded_scenario
def test_distinct_key_can_enter_while_first_waits():
    with harness() as h:
        entered = Event()
        with h.frame.open_turn(request(), check_stopped=lambda: None):

            def other():
                with h.frame.open_turn(
                    request("b", consumer=("root", "other")),
                    check_stopped=lambda: None,
                ):
                    entered.set()

            worker = _Worker(other)
            assert entered.wait(2)
            worker.join()
            assert worker.errors == []


@_bounded_scenario
def test_cancelled_ticket_is_removed():
    with harness() as h:
        stop = Event()
        failure = RuntimeError("stop waiting")

        def check():
            if stop.is_set():
                raise failure

        with h.frame.open_turn(request(), check_stopped=lambda: None):

            def second():
                with h.frame.open_turn(request("b"), check_stopped=check):
                    pytest.fail("cancelled waiter entered")

            worker = _Worker(second)
            _wait_queue(h.memory, 1)
            stop.set()
            worker.join()
            assert worker.errors == [failure]
            _wait_queue(h.memory, 0)


def test_reset_rejects_active_backend_without_mutation():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            before = snapshot(h.memory)
            with pytest.raises(c.MemoryOperationError) as caught:
                h.memory.reset()
            assert (caught.value.stage, caught.value.code, caught.value.retryable) == (
                "reset",
                "session.busy",
                True,
            )
            assert snapshot(h.memory) == before


@_bounded_scenario
def test_reset_rejects_queued_backend_without_mutation():
    with harness() as h, harness(h.memory) as waiting_session:
        with h.frame.open_turn(request(), check_stopped=lambda: None):

            def waiting():
                with waiting_session.frame.open_turn(
                    request("b"), check_stopped=lambda: None
                ):
                    pytest.fail("closed waiter entered")

            worker = _Worker(waiting)
            _wait_queue(h.memory, 1)
            before = snapshot(h.memory)
            try:
                with pytest.raises(c.MemoryOperationError) as caught:
                    h.memory.reset()
                assert caught.value.code == "session.busy"
                assert snapshot(h.memory) == before
            finally:
                waiting_session.session.close()
                worker.join()
            assert len(worker.errors) == 1
            assert worker.errors[0].code == "session.stale"


def test_idle_reset_clears_all_keys():
    with harness() as h:
        for name in ("agent", "other"):
            with h.frame.open_turn(
                request(name, consumer=("root", name)), check_stopped=lambda: None
            ):
                pass
        before = snapshot(h.memory)
        h.memory.reset()
        assert snapshot(h.memory) == {
            **before,
            "generation": before["generation"] + 1,
            "conversations": [],
        }


def test_reset_invalidates_old_session_frame_and_turn():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            pass
        h.memory.reset()
        for call in (
            lambda: h.session.snapshot,
            lambda: h.session.open_frame(c.MemoryFrameSpec(REF, "session", {})),
            lambda: h.frame.recall(),
            lambda: h.frame.learn(c.LearnContext("a", REF, ())),
            lambda: h.session.commit("id", ()),
            lambda: turn.request("model", {}),
            lambda: turn.observe(1, "completed"),
            lambda: h.frame.open_turn(
                request("b"), check_stopped=lambda: None
            ).__enter__(),
        ):
            with pytest.raises(c.MemoryOperationError) as caught:
                call()
            assert caught.value.code == "session.stale"
        assert snapshot(h.memory)["conversations"] == []


@pytest.mark.parametrize("hostile", [None, "str", "add_note"])
def test_cleanup_annotation_cannot_replace_primary_error(hostile, monkeypatch):
    class Primary(RuntimeError):
        def add_note(self, text):
            if hostile == "add_note":
                raise RuntimeError("note formatting failed")
            super().add_note(text)

    class Cleanup(RuntimeError):
        def __str__(self):
            if hostile == "str":
                raise RuntimeError("string formatting failed")
            return "cleanup"

    primary, cleanup = Primary("primary"), Cleanup()
    with harness() as h:
        with pytest.raises(Primary) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None):

                def reject(_view):
                    raise cleanup

                monkeypatch.setattr(h.session, "_candidate_snapshot", reject)
                raise primary
        assert caught.value is primary
        if hostile is None and hasattr(BaseException, "add_note"):
            assert primary.__notes__ == [
                "Conversation cleanup also failed: Cleanup: cleanup"
            ]
        assert not h.memory._active


def test_completed_observation_is_retained_before_checker_stop():
    failure = RuntimeError("ordinary Exception used as host stop")
    stop = Event()

    def check():
        if stop.is_set():
            raise failure

    with harness() as h:
        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(request(), check_stopped=check) as turn:
                ordinal = turn.request("tool", {})
                stop.set()
                turn.observe(
                    ordinal, "completed", items=(item("result", "already happened"),)
                )
        assert caught.value is failure
        last = _live(h)
        assert last.effects[0].status == "completed"
        assert last.items == (item("result", "already happened"),)
        assert last.closed_invocations[-1].status == "interrupted"


def test_stop_after_publication_preserves_requested_effect():
    failure = RuntimeError("stop before external call")
    with harness() as h:

        def check():
            if h.memory._views and _live(h).effects:
                raise failure

        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(request(), check_stopped=check) as turn:
                turn.request("tool", {})
                pytest.fail("stopped request granted effect permission")
        assert caught.value is failure
        assert _live(h).effects[0].status == "pending"
        assert _live(h).closed_invocations[-1].status == "interrupted"


@pytest.mark.parametrize(
    "primary", [RuntimeError("failed"), KeyboardInterrupt("interrupted")]
)
def test_body_failure_closes_truthfully_without_inventing_observations(primary):
    with harness() as h:
        with pytest.raises(type(primary)) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                turn.request("tool", {})
                raise primary
        assert caught.value is primary
        view = _live(h)
        assert view.effects[0].status == "pending"
        assert view.effects[0].observation_ordinals == ()
        expected = "failed" if isinstance(primary, Exception) else "interrupted"
        assert view.closed_invocations[-1].status == expected
        with h.frame.open_turn(request("b"), check_stopped=lambda: None) as turn:
            assert turn.view.effects == view.effects


def test_completed_close_with_pending_effect_is_rejected_without_fabrication():
    with harness() as h:
        with pytest.raises(c.MemoryOperationError):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                turn.request("model", {})
        assert _live(h).revision == 2
        assert _live(h).effects[0].status == "pending"
        assert _live(h).closed_invocations == ()
        assert not h.memory._active
        with h.frame.open_turn(request("b"), check_stopped=lambda: None):
            pass


def test_active_invocation_cannot_be_shadowed_by_another_key():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            with pytest.raises(c.MemoryOperationError) as caught:
                with h.frame.open_turn(
                    request(consumer=("root", "other")),
                    check_stopped=lambda: None,
                ):
                    pass
            assert caught.value.code == "session.conflict"
            assert len(h.memory._active) == 1


@pytest.mark.parametrize(
    "field", ["policy_name", "key", "invocation", "namespace", "generation"]
)
def test_publication_requires_owned_stamp(field):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            value = {
                "policy_name": "other",
                "key": c.ConversationKey(REF, ("root", "other")),
                "invocation": request("other").invocation,
                "namespace": "foreign",
                "generation": 1,
            }[field]
            checkpoint = c.ConversationEffectRequested(
                replace(turn._stamp(), **{field: value}), "model", {}, ()
            )
            before = snapshot(h.memory)
            with pytest.raises(c.MemoryOperationError) as caught:
                turn._publish(checkpoint)
            assert caught.value.code == "session.conflict"
            assert snapshot(h.memory) == before


@_bounded_scenario
def test_session_close_retains_active_owner_until_finalization():
    with harness() as one, harness(one.memory) as two:
        with one.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            ordinal = turn.request("model", {})
            one.session.close()
            one.session.close()
            with pytest.raises(c.MemoryOperationError) as caught:
                turn.request("model", {})
            assert caught.value.code == "session.stale"
            with pytest.raises(c.MemoryOperationError) as busy:
                one.memory.reset()
            assert busy.value.code == "session.busy"
            turn.observe(ordinal, "completed", items=(item("actual", "retained"),))
            with two.frame.open_turn(
                request("b", consumer=("root", "other")), check_stopped=lambda: None
            ):
                pass
        assert _live(one).effects[0].status == "completed"
        assert _live(one).items == (item("actual", "retained"),)
        assert _live(one).closed_invocations[-1].status == "interrupted"
        assert not one.memory._active


@_bounded_scenario
def test_session_close_removes_own_waiters_without_poisoning_other_owner():
    with harness() as one, harness(one.memory) as two:
        with one.frame.open_turn(request(), check_stopped=lambda: None):

            def waiting():
                with two.frame.open_turn(request("b"), check_stopped=lambda: None):
                    pytest.fail("closed session was admitted")

            worker = _Worker(waiting)
            _wait_queue(one.memory, 1)
            two.session.close()
            worker.join()
            assert len(worker.errors) == 1
            assert worker.errors[0].code == "session.stale"
            assert not one.memory._queues
            assert one.memory._active


@_bounded_scenario
def test_stop_and_external_call_do_not_hold_backend_lock():
    with harness() as h:

        def check():
            worker = _Worker(lambda: snapshot(h.memory))
            worker.join()
            assert not worker.errors

        with h.frame.open_turn(request(), check_stopped=check) as turn:
            ordinal = turn.request("model", {})
            worker = _Worker(lambda: snapshot(h.memory))
            worker.join()
            assert not worker.errors
            turn.observe(ordinal, "completed")


@_bounded_scenario
def test_reset_rejects_queued_ticket_even_without_an_active_lease():
    with harness() as h:
        checking = Event()
        resume = Event()

        def check():
            checking.set()
            assert resume.wait(2)

        def waiting():
            with h.frame.open_turn(request(), check_stopped=check):
                pass

        worker = _Worker(waiting)
        assert checking.wait(2)
        try:
            assert not h.memory._active
            _wait_queue(h.memory, 1)
            before = snapshot(h.memory)
            with pytest.raises(c.MemoryOperationError) as caught:
                h.memory.reset()
            assert caught.value.code == "session.busy"
            assert snapshot(h.memory) == before
        finally:
            resume.set()
            worker.join()
        assert not worker.errors


def _nested(depth):
    value = {}
    for _ in range(depth):
        value = {"nested": value}
    return value


def test_snapshot_envelope_depth_is_rejected_before_publication():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            before = turn.view
            candidate = None
            for depth in range(115, 128):
                checkpoint = c.ConversationEffectRequested(
                    turn._stamp(), "model", _nested(depth), ()
                )
                try:
                    after = advance_conversation(before, checkpoint)
                except ValueError:
                    continue
                exported = {
                    **h.session.snapshot,
                    "conversations": [conversation_view_to_json(after)],
                }
                try:
                    c._freeze_json_object(exported, "snapshot")
                except ValueError:
                    candidate = checkpoint
                    break
            assert (
                candidate is not None
            ), "expected a view-valid but snapshot-too-deep boundary"
            exported_before = snapshot(h.memory)
            with pytest.raises(ValueError, match="nesting"):
                turn.request(candidate.kind, candidate.payload)
            assert snapshot(h.memory) == exported_before
            assert turn.view == before
            assert turn._lease.active


@pytest.mark.parametrize(
    "case",
    [
        "list",
        "version_bool",
        "version_unknown",
        "generation_bool",
        "generation_negative",
        "namespace_empty",
        "namespace_surrogate",
        "extra",
        "missing",
        "tuple",
        "nan",
        "infinity",
        "cycle",
        "deep",
        "duplicate",
        "foreign_namespace",
        "foreign_generation",
        "bad_digest",
        "counter_bool",
        "bad_container",
    ],
)
def test_snapshot_rejects_invalid_container_or_nonfinite_json(case):
    with harness() as source:
        with source.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        exported = snapshot(source.memory)
    if case == "list":
        exported = []
    elif case == "version_bool":
        exported["version"] = True
    elif case == "version_unknown":
        exported["version"] = 2
    elif case == "generation_bool":
        exported["generation"] = False
    elif case == "generation_negative":
        exported["generation"] = -1
    elif case == "namespace_empty":
        exported["namespace"] = ""
    elif case == "namespace_surrogate":
        exported["namespace"] = "\ud800"
    elif case == "extra":
        exported["extra"] = None
    elif case == "missing":
        del exported["version"]
    elif case == "tuple":
        exported["conversations"] = tuple(exported["conversations"])
    elif case == "nan":
        exported["version"] = float("nan")
    elif case == "infinity":
        exported["version"] = float("inf")
    elif case == "cycle":
        exported["conversations"].append(exported)
    elif case == "deep":
        exported["conversations"] = [_nested(130)]
    elif case == "duplicate":
        exported["conversations"] *= 2
    elif case == "foreign_namespace":
        exported["conversations"][0]["namespace"] = "foreign"
    elif case == "foreign_generation":
        exported["conversations"][0]["generation"] += 1
    elif case == "bad_digest":
        exported["conversations"][0]["applied"][0]["after_digest"] = "0" * 64
    elif case == "counter_bool":
        exported["conversations"][0]["revision"] = True
    elif case == "bad_container":
        exported["conversations"] = {}
    memory = SessionMemory()
    with pytest.raises(c.MemoryOperationError) as caught:
        memory.open_run(snapshot=exported)
    assert (caught.value.stage, caught.value.code, caught.value.retryable) == (
        "snapshot",
        "session.invalid_snapshot",
        False,
    )
    assert not memory._used
    assert not memory._views


def test_snapshot_order_round_trips_without_reordering():
    with harness() as h:
        for name in ("z", "a"):
            with h.frame.open_turn(
                request(name, consumer=("root", name)),
                check_stopped=lambda: None,
            ):
                pass
        exported = snapshot(h.memory)
    with SessionMemory().open_run(snapshot=exported) as restored:
        assert restored.snapshot == exported
        assert [
            value["key"]["consumer_static_path"]
            for value in restored.snapshot["conversations"]
        ] == [["root", "z"], ["root", "a"]]


def test_stop_checker_cannot_close_session_then_grant_effect_permission():
    with harness() as h:

        def check():
            if h.memory._views and _live(h).effects:
                h.session.close()

        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(request(), check_stopped=check) as turn:
                turn.request("tool", {})
                pytest.fail("closed session granted effect permission")
        assert caught.value.code == "session.stale"
        assert _live(h).effects[0].status == "pending"
        assert _live(h).closed_invocations[-1].status == "interrupted"
        assert not h.memory._active


def test_observed_stop_cannot_be_caught_to_reenable_new_effects():
    with harness() as h:
        stop = Event()
        failure = RuntimeError("stop once")

        def check():
            if stop.is_set():
                raise failure

        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(request(), check_stopped=check) as turn:
                stop.set()
                with pytest.raises(RuntimeError):
                    turn.request("tool", {})
                stop.clear()
                turn.request("tool", {})
                pytest.fail("observed stop was forgotten")
        assert caught.value is failure
        assert not _live(h).effects


@pytest.mark.parametrize("owner", [c.MemoryOwner.FLOW, c.MemoryOwner.AGENT])
def test_frame_ownership_reuses_authored_leaf_and_keeps_qualified_calls_independent(
    owner,
):
    path = ("root",) if owner is c.MemoryOwner.FLOW else ("root", "call_a", "agent")
    ref = c.MemoryFrameRef("root", path, owner)
    with harness(ref=ref) as h:
        for run in ("a", "b"):
            with h.frame.open_turn(
                request(run, consumer=("root", "call_a", "agent")),
                check_stopped=lambda: None,
            ) as turn:
                assert len(turn.view.items) == (0 if run == "a" else 1)
                append_reply(turn, run, run)
        if owner is c.MemoryOwner.FLOW:
            with h.frame.open_turn(
                request("c", consumer=("root", "call_b", "agent")),
                check_stopped=lambda: None,
            ) as turn:
                assert not turn.view.items
        else:
            with pytest.raises(ValueError):
                with h.frame.open_turn(
                    request("c", consumer=("root", "call_b", "agent")),
                    check_stopped=lambda: None,
                ):
                    pass


def test_used_backend_rejects_newer_same_namespace_snapshot():
    with harness() as source:
        initial = source.session.snapshot
        with source.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        newer = snapshot(source.memory)
    restored = SessionMemory()
    with restored.open_run(snapshot=initial):
        pass
    with pytest.raises(c.MemoryOperationError) as caught:
        restored.open_run(snapshot=newer)
    assert caught.value.code == "session.invalid_snapshot"
    assert snapshot(restored) == initial


def test_opening_frame_does_not_change_pinned_or_live_snapshot():
    memory = SessionMemory()
    with memory.open_run() as session:
        before = session.snapshot
        session.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        assert session.snapshot == before
        assert snapshot(memory) == before
        with pytest.raises(c.MemoryOperationError) as caught:
            session.open_frame(c.MemoryFrameSpec(REF, "different", {}))
        assert caught.value.code == "session.invalid_config"
        assert snapshot(memory) == before


def test_invalid_request_wire_publishes_nothing_and_keeps_lease_usable():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            before = snapshot(h.memory)
            with pytest.raises(ValueError, match="nesting") as rejected:
                turn.request("model", _nested(127))
            assert snapshot(h.memory) == before
            assert not getattr(rejected.value, "__notes__", ())
            append_reply(turn, "valid", "still usable")
        assert _live(h).effects[0].status == "completed"


@_bounded_scenario
def test_session_close_at_reserved_admission_boundary_is_typed_stale():
    with harness() as h:
        reserved = Event()
        continue_admission = Event()
        leases = []
        reserve = h.session._reserve

        def pause_after_reservation(frame, request, check_stopped):
            lease = reserve(frame, request, check_stopped)
            leases.append(lease)
            reserved.set()
            assert continue_admission.wait(2)
            return lease

        h.session._reserve = pause_after_reservation

        def entering():
            with h.frame.open_turn(request(), check_stopped=lambda: None):
                pytest.fail("a closed session admitted its reserved turn")

        worker = _Worker(entering)
        try:
            assert reserved.wait(2)
            assert leases[0].before is not None
            h.session.close()
            assert leases[0].before is not None
            assert leases[0].active
        finally:
            continue_admission.set()
            worker.join()
        assert len(worker.errors) == 1
        failure = worker.errors[0]
        assert type(failure) is c.MemoryOperationError, repr(failure)
        assert not getattr(failure, "__notes__", ())
        assert (failure.stage, failure.code, failure.retryable) == (
            "conversation",
            "session.stale",
            False,
        )
        assert not h.memory._views
        assert not h.memory._active
        assert not h.memory._queues


@_bounded_scenario
def test_session_close_never_overlaps_an_inflight_effect_with_the_next_turn():
    with harness() as owner, harness(owner.memory) as next_run:
        in_effect, finish_effect, entered_next = Event(), Event(), Event()
        order = []

        def active():
            with owner.frame.open_turn(
                request("a"), check_stopped=lambda: None
            ) as turn:
                ordinal = turn.request("model", {})
                order.append("effect-start")
                in_effect.set()
                assert finish_effect.wait(3)
                order.append("effect-end")
                turn.observe(ordinal, "completed", items=(item("result", "actual"),))

        def waiting():
            with next_run.frame.open_turn(
                request("b"), check_stopped=lambda: None
            ) as turn:
                order.append("next-enter")
                entered_next.set()
                assert turn.view.items == (item("result", "actual"),)
                assert turn.view.closed_invocations[-1].status == "interrupted"

        active_worker = _Worker(active)
        assert in_effect.wait(2)
        waiting_worker = _Worker(waiting)
        try:
            _wait_queue(owner.memory, 1)
            owner.session.close()
            assert not entered_next.is_set()
            assert len(owner.memory._active) == 1
            with pytest.raises(c.MemoryOperationError) as busy:
                owner.memory.reset()
            assert busy.value.code == "session.busy"
            with next_run.frame.open_turn(
                request("other", consumer=("root", "other")), check_stopped=lambda: None
            ):
                assert not entered_next.is_set()
        finally:
            finish_effect.set()
            active_worker.join()
            waiting_worker.join()
        assert not active_worker.errors
        assert not waiting_worker.errors
        assert order == ["effect-start", "effect-end", "next-enter"]


@pytest.mark.parametrize("stage", ["admission", "request", "observation", "close"])
def test_candidate_snapshot_failure_publishes_nothing(stage, monkeypatch):
    failure = RuntimeError("snapshot unavailable")
    with harness() as h:
        turn = h.frame.open_turn(request(), check_stopped=lambda: None)
        if stage != "admission":
            turn.__enter__()
        if stage in ("observation", "close"):
            ordinal = turn.request("model", {})
        if stage == "close":
            turn.observe(ordinal, "completed")
        before = snapshot(h.memory)
        previous = h.session._candidate_snapshot

        def reject(_view):
            raise failure

        monkeypatch.setattr(h.session, "_candidate_snapshot", reject)
        operation = {
            "admission": turn.__enter__,
            "request": lambda: turn.request("model", {}, items=(item("new", "no"),)),
            "observation": lambda: turn.observe(
                ordinal, "completed", items=(item("new", "no"),)
            ),
            "close": lambda: turn.__exit__(None, None, None),
        }[stage]
        with pytest.raises(RuntimeError) as caught:
            operation()
        assert caught.value is failure
        assert not getattr(failure, "__notes__", ())
        assert snapshot(h.memory) == before
        monkeypatch.setattr(h.session, "_candidate_snapshot", previous)
        if stage == "observation":
            turn.observe(ordinal, "completed")
        if stage in ("request", "observation"):
            turn.__exit__(None, None, None)
        with h.frame.open_turn(request("next"), check_stopped=lambda: None):
            pass


@pytest.mark.parametrize("nonempty", [False, True])
def test_old_opening_snapshot_reads_latest_history_without_rewind(nonempty):
    with harness() as source:
        if nonempty:
            with source.frame.open_turn(
                request("first"), check_stopped=lambda: None
            ) as turn:
                append_reply(turn, "first", "first")
        old = snapshot(source.memory)
        with source.frame.open_turn(
            request("second"), check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "second", "second")
        latest = _live(source)
        with source.memory.open_run(snapshot=old) as opened:
            frame = opened.open_frame(c.MemoryFrameSpec(REF, "session", {}))
            assert opened.snapshot == old
            with frame.open_turn(request("third"), check_stopped=lambda: None) as turn:
                assert turn.view.items == latest.items
                assert turn.view.revision == latest.revision + 1
                append_reply(turn, "third", "third")
            assert opened.snapshot == old


@pytest.mark.parametrize(
    "malformed", ["kind", "payload_list", "nan", "items_list", "duplicate_items"]
)
def test_malformed_request_never_publishes_partial_evidence(malformed):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            kind, payload, items = "model", {}, ()
            if malformed == "kind":
                kind = "unknown"
            elif malformed == "payload_list":
                payload = []
            elif malformed == "nan":
                payload = {"value": float("nan")}
            elif malformed == "items_list":
                items = [item("x", "x")]
            else:
                items = (item("x", "x"), item("x", "y"))
            before = snapshot(h.memory)
            with pytest.raises((TypeError, ValueError)):
                turn.request(kind, payload, items=items)
            assert snapshot(h.memory) == before
            append_reply(turn, "valid", "retained")


@pytest.mark.parametrize(
    "malformed", ["ordinal", "status", "failure", "items", "reference"]
)
def test_malformed_observation_leaves_pending_evidence_unchanged(malformed):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            ordinal = turn.request("model", {}, items=(item("input", "question"),))
            values = {"effect_ordinal": ordinal, "status": "completed"}
            if malformed == "ordinal":
                values["effect_ordinal"] = True
            elif malformed == "status":
                values["status"] = "unknown"
            elif malformed == "failure":
                values["status"] = "failed"
            elif malformed == "items":
                values["items"] = [item("partial", "no")]
            else:
                values["reference"] = {"provider": "fake", "session_id": "no"}
            before = snapshot(h.memory)
            with pytest.raises((TypeError, ValueError)):
                turn.observe(**values)
            assert snapshot(h.memory) == before
            turn.observe(ordinal, "completed", items=(item("answer", "actual"),))


def test_alternate_same_namespace_history_is_not_a_valid_opening_snapshot():
    with harness() as live:
        initial = live.session.snapshot
        with live.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            append_reply(turn, "reply", "real")
        original = snapshot(live.memory)
        fork = SessionMemory()
        with fork.open_run(snapshot=initial):
            pass
        with harness(fork) as alternate:
            with alternate.frame.open_turn(
                request(), check_stopped=lambda: None
            ) as turn:
                append_reply(turn, "reply", "different")
        with pytest.raises(c.MemoryOperationError) as caught:
            live.memory.open_run(snapshot=snapshot(fork))
        assert caught.value.code == "session.invalid_snapshot"
        assert snapshot(live.memory) == original


@pytest.mark.parametrize("new_session", [False, True])
def test_an_invocation_identity_cannot_be_reused(new_session):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        before = snapshot(h.memory)
        with harness(h.memory) as other:
            frame = other.frame if new_session else h.frame
            with pytest.raises(c.MemoryOperationError) as caught:
                with frame.open_turn(request(), check_stopped=lambda: None):
                    pytest.fail("reused invocation entered")
            assert caught.value.code == "session.conflict"
        assert snapshot(h.memory) == before
