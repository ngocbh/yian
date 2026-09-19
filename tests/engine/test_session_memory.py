"""Process-local session memory and its checkpoint acceptance boundary."""

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


def test_session_backend_implements_original_protocol():
    memory = SessionMemory()
    assert isinstance(memory, c.Memory)
    with memory.open_run() as session:
        assert isinstance(session, c.MemorySession)
        assert isinstance(session, c.ConversationSession)
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


def _admission(lease):
    before = lease.before
    return c.ConversationAdmitted(
        c.ConversationStamp(
            lease.frame._spec.policy_name,
            before.key,
            lease.request.invocation,
            before.namespace,
            before.generation,
            before.revision,
            0,
        ),
        before,
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


def test_prepare_checkpoint_does_not_mutate_backend():
    memory = SessionMemory()
    with memory.open_run() as session:
        lease = _reserve(session)
        checkpoint = _admission(lease)
        apply = session.prepare_checkpoint(checkpoint, before=lease.before)
        assert callable(apply)
        assert memory._views == {}
        session._release(lease)


def test_same_prepared_token_returns_its_exact_result():
    memory = SessionMemory()
    with memory.open_run() as session:
        lease = _reserve(session)
        apply = session.prepare_checkpoint(_admission(lease), before=lease.before)
        first = apply()
        assert first.revision == 1
        assert apply() == first
        assert memory._views[first.key] == first
        session._release(lease)


def test_first_turn_admission_uses_root_and_consumer():
    with harness(policy="custom") as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            assert turn.view.key == c.ConversationKey(REF, ("root", "agent"))
            assert turn.view.revision == 1
            append_reply(turn, "one", "hello")
        assert h.accepted[0].stamp.policy_name == "custom"
        assert [type(event) for event in h.accepted] == [
            c.ConversationAdmitted,
            c.ConversationEffectRequested,
            c.ConversationEffectObserved,
            c.ConversationClosed,
        ]


def test_same_agent_reuses_conversation_across_runs():
    memory = SessionMemory()
    with harness(memory) as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")
    with harness(memory) as h:
        with h.frame.open_turn(
            request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            assert turn.view.items == (item("one", "hello"),)


def test_distinct_consumer_has_no_reference_or_history():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            ordinal = turn.request("native_session", {})
            turn.observe(
                ordinal,
                "completed",
                items=(item("one", "hello"),),
                reference=c.ConversationReference("codex", "opaque"),
            )
        with h.frame.open_turn(
            request("b", consumer=("root", "other")),
            checkpoint=h.checkpoint,
            check_stopped=lambda: None,
        ) as turn:
            assert turn.view.items == ()
            assert turn.view.reference is None


def test_open_session_snapshot_stays_pinned():
    with harness() as h:
        pinned = h.session.snapshot
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")
        assert h.session.snapshot == pinned
        assert snapshot(h.memory)["conversations"][0]["revision"] == 4


def test_admission_reads_live_view_not_open_snapshot():
    with harness() as first, harness(first.memory) as second:
        assert second.session.snapshot["conversations"] == []
        with first.frame.open_turn(
            request(), checkpoint=first.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")
        with second.frame.open_turn(
            request("b"), checkpoint=second.checkpoint, check_stopped=lambda: None
        ) as turn:
            assert turn.view.items == (item("one", "hello"),)


def _history(h):
    before = h.accepted[0].baseline
    pairs = []
    for checkpoint in h.accepted:
        pairs.append((checkpoint, before))
        before = advance_conversation(before, checkpoint)
    return pairs


def test_duplicate_preparation_needs_original_predecessor():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            first = turn.request("native_session", {"version": 1})
            turn.observe(
                first, "pending", reference=c.ConversationReference("codex", "old")
            )
            turn.observe(
                first,
                "completed",
                items=(item("one", "reply"),),
                reference=c.ConversationReference("codex", "new"),
            )
        pairs = _history(h)
        for checkpoint, before in pairs:
            apply = h.session.prepare_checkpoint(checkpoint, before=before)
            assert apply() == advance_conversation(before, checkpoint)
        with pytest.raises(c.MemoryOperationError):
            h.session.prepare_checkpoint(pairs[1][0], before=pairs[-1][1])


def test_duplicate_rejects_forged_before_view():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "real")
        checkpoint, before = _history(h)[-1]
        forged = c.ConversationView(
            before.namespace,
            before.generation,
            before.revision,
            before.key,
            before.compatibility,
            (item("one", "forged"),),
            before.reference,
            before.effects,
            before.closed_invocations,
            before.applied,
        )
        with pytest.raises((ValueError, c.MemoryOperationError)):
            h.session.prepare_checkpoint(checkpoint, before=forged)


def test_released_token_cannot_publish_unapplied_checkpoint():
    memory = SessionMemory()
    with memory.open_run() as session:
        lease = _reserve(session)
        apply = session.prepare_checkpoint(_admission(lease), before=lease.before)
        session._release(lease)
        with pytest.raises(c.MemoryOperationError, match="lease"):
            apply()
        assert memory._views == {}


def test_duplicate_token_after_release_is_read_only():
    memory = SessionMemory()
    with memory.open_run() as session:
        lease = _reserve(session)
        apply = session.prepare_checkpoint(_admission(lease), before=lease.before)
        expected = apply()
        session._release(lease)
        before = snapshot(memory)
        assert apply() == expected
        assert snapshot(memory) == before


def test_pristine_backend_adopts_snapshot_namespace():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")
        exported = snapshot(h.memory)
    restored = SessionMemory()
    with restored.open_run(snapshot=exported) as session:
        assert session.snapshot == exported
        exported["conversations"].clear()
        assert len(session.snapshot["conversations"]) == 1


def test_duplicate_after_latest_snapshot_restore():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")
        pairs = _history(h)
        exported = snapshot(h.memory)
    with SessionMemory().open_run(snapshot=exported) as session:
        for checkpoint, before in pairs:
            assert session.prepare_checkpoint(
                checkpoint, before=before
            )() == advance_conversation(before, checkpoint)


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
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
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
            checkpoint=h.checkpoint,
            check_stopped=lambda: None,
        ):
            pass
        before = snapshot(h.memory)
        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(
                request("b", compatibility={"format": new}),
                checkpoint=h.checkpoint,
                check_stopped=lambda: None,
            ):
                pytest.fail("incompatible conversation was admitted")
        assert caught.value.code == "session.incompatible"
        assert snapshot(h.memory) == before
        h.memory.reset()
    with harness(h.memory) as fresh:
        with fresh.frame.open_turn(
            request("c", compatibility={"format": new}),
            checkpoint=fresh.checkpoint,
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
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):

            def second():
                with h.frame.open_turn(
                    request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
                ):
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
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):

            def waiting(name):
                with h.frame.open_turn(
                    request(name), checkpoint=h.checkpoint, check_stopped=lambda: None
                ):
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
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):

            def other():
                with h.frame.open_turn(
                    request("b", consumer=("root", "other")),
                    checkpoint=h.checkpoint,
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

        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):

            def second():
                with h.frame.open_turn(
                    request("b"), checkpoint=h.checkpoint, check_stopped=check
                ):
                    pytest.fail("cancelled waiter entered")

            worker = _Worker(second)
            _wait_queue(h.memory, 1)
            stop.set()
            worker.join()
            assert worker.errors == [failure]
            _wait_queue(h.memory, 0)


@_bounded_scenario
def test_fail_turn_fences_before_release():
    with harness() as h:
        entered = Event()
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):

            def waiting():
                with h.frame.open_turn(
                    request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
                ):
                    entered.set()

            worker = _Worker(waiting)
            _wait_queue(h.memory, 1)
            h.session.fail_turn(
                request().invocation, checkpoint=h.accepted[0], acceptance="accepted"
            )
            worker.join()
            assert not entered.is_set()
            assert len(worker.errors) == 1
            assert worker.errors[0].code == "session.recovery_required"
            assert h.memory._queues == {}
            assert h.memory._active == {}


def test_fenced_waiter_fails_and_removes_ticket():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            h.session.fail_turn(
                request().invocation, checkpoint=None, acceptance="uncertain"
            )
        accepted = len(h.accepted)
        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(
                request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
            ):
                pytest.fail("fenced turn entered")
        assert caught.value.code == "session.recovery_required"
        assert len(h.accepted) == accepted
        assert h.memory._queues == {}


def test_preappend_rejection_does_not_install_fence():
    with harness() as h:
        failure = RuntimeError("append rejected before attempt")

        def reject(checkpoint):
            h.session.fail_turn(
                checkpoint.stamp.invocation,
                checkpoint=checkpoint,
                acceptance="unaccepted",
            )
            raise failure

        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=reject, check_stopped=lambda: None
            ):
                pytest.fail("rejected admission entered")
        assert caught.value is failure
        assert h.memory._fences == {}
        assert h.memory._views == {}
        with h.frame.open_turn(
            request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            pass


def test_reset_rejects_active_backend_without_mutation():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
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
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):

            def waiting():
                with h.frame.open_turn(
                    request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
                ):
                    pass

            worker = _Worker(waiting)
            _wait_queue(h.memory, 1)
            before = snapshot(h.memory)
            try:
                with pytest.raises(c.MemoryOperationError) as caught:
                    h.memory.reset()
                assert caught.value.code == "session.busy"
                assert snapshot(h.memory) == before
            finally:
                h.session.fail_turn(
                    request("b").invocation, checkpoint=None, acceptance="unaccepted"
                )
                worker.join()


def test_idle_reset_clears_all_keys_and_fences():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            h.session.fail_turn(
                request().invocation, checkpoint=None, acceptance="uncertain"
            )
        with h.frame.open_turn(
            request("b", consumer=("root", "other")),
            checkpoint=h.checkpoint,
            check_stopped=lambda: None,
        ):
            pass
        before = snapshot(h.memory)
        h.memory.reset()
        after = snapshot(h.memory)
        assert after == {
            **before,
            "generation": before["generation"] + 1,
            "conversations": [],
        }
        assert h.memory._fences == {}


@pytest.mark.parametrize("applied", [False, True])
def test_reset_fences_old_handle_and_prepared_token(applied):
    memory = SessionMemory()
    with memory.open_run() as session:
        lease = _reserve(session)
        apply = session.prepare_checkpoint(_admission(lease), before=lease.before)
        if applied:
            apply()
        session._release(lease)
        memory.reset()
        for call in (
            apply,
            lambda: session.open_frame(c.MemoryFrameSpec(REF, "custom", {})),
        ):
            with pytest.raises(c.MemoryOperationError) as caught:
                call()
            assert caught.value.code == "session.stale"
        assert snapshot(memory)["conversations"] == []


@pytest.mark.parametrize("hostile", ["str", "add_note"])
def test_cleanup_annotation_cannot_replace_primary_error(hostile):
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

    primary = Primary("primary")
    cleanup = Cleanup()
    with harness() as h:

        def accept(checkpoint):
            if type(checkpoint) is c.ConversationClosed:
                raise cleanup
            return h.checkpoint(checkpoint)

        with pytest.raises(Primary) as caught:
            with h.frame.open_turn(
                request(), checkpoint=accept, check_stopped=lambda: None
            ):
                raise primary
        assert caught.value is primary


def _authority(h):
    return c.ConversationAuthority(h.accepted[0].baseline, tuple(h.accepted))


def test_reconcile_after_opening_empty_s0_installs_accepted_baseline():
    memory = SessionMemory()
    with harness(memory) as first:
        empty = first.session.snapshot
        with first.frame.open_turn(
            request(), checkpoint=first.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "first")
    with harness(memory) as second:
        with second.frame.open_turn(
            request("b"), checkpoint=second.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "two", "second")
        authority = _authority(second)
        expected = _history(second)[-1]
        expected = advance_conversation(expected[1], expected[0])
    restored = SessionMemory()
    with restored.open_run(snapshot=empty) as session:
        assert session.reconcile(authority) == expected
        assert session.snapshot == empty
        assert snapshot(restored)["conversations"] == [
            conversation_view_to_json(expected)
        ]
        assert session.reconcile(authority) == expected


def test_reconcile_after_opening_nonempty_s0_installs_accepted_baseline():
    memory = SessionMemory()
    with harness(memory) as first:
        with first.frame.open_turn(
            request(), checkpoint=first.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "first")
        with first.frame.open_turn(
            request("other", consumer=("root", "other")),
            checkpoint=first.checkpoint,
            check_stopped=lambda: None,
        ) as turn:
            append_reply(turn, "unrelated", "preserve me")
        older = snapshot(memory)
    with harness(memory) as second:
        with second.frame.open_turn(
            request("b"), checkpoint=second.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "two", "second")
    with harness(memory) as third:
        with third.frame.open_turn(
            request("c"), checkpoint=third.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "three", "third")
        authority = _authority(third)
    restored = SessionMemory()
    with restored.open_run(snapshot=older) as session:
        result = session.reconcile(authority)
        assert len(result.items) == 3
        assert session.snapshot == older
        assert snapshot(restored)["conversations"] == [
            conversation_view_to_json(result),
            older["conversations"][1],
        ]


def test_baseline_install_conflicts_publish_nothing():
    with harness() as h:
        empty = h.session.snapshot
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "accepted")
    with harness(h.memory) as next_h:
        with next_h.frame.open_turn(
            request("b"), checkpoint=next_h.checkpoint, check_stopped=lambda: None
        ):
            pass
        authority = _authority(next_h)
    conflicting = SessionMemory()
    with conflicting.open_run(snapshot=empty):
        pass
    with harness(conflicting) as other:
        with other.frame.open_turn(
            request(), checkpoint=other.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "different")
        before = snapshot(conflicting)
        with pytest.raises(c.MemoryOperationError) as caught:
            other.session.reconcile(authority)
        assert caught.value.code == "session.conflict"
        assert snapshot(conflicting) == before


def test_reconcile_requires_exact_accepted_fence_checkpoint():
    with harness() as h:
        failure = RuntimeError("lost acknowledgement")

        def lost(checkpoint):
            after = h.checkpoint(checkpoint)
            if type(checkpoint) is c.ConversationEffectObserved:
                h.session.fail_turn(
                    checkpoint.stamp.invocation,
                    checkpoint=checkpoint,
                    acceptance="accepted",
                )
                raise failure
            return after

        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=lost, check_stopped=lambda: None
            ) as turn:
                append_reply(turn, "one", "truth")
        assert caught.value is failure
        authority = _authority(h)
        shorter = c.ConversationAuthority(
            authority.baseline, authority.checkpoints[:-1]
        )
        before = snapshot(h.memory)
        with pytest.raises(c.MemoryOperationError) as rejected:
            h.session.reconcile(shorter)
        assert rejected.value.code == "session.recovery_required"
        assert snapshot(h.memory) == before
        assert h.memory._fences
        result = h.session.reconcile(authority)
        assert result.items == (item("one", "truth"),)
        assert not h.memory._fences
        # Reconciliation proves history, not completion or new effect permission.
        with pytest.raises(c.MemoryOperationError) as blocked:
            with h.frame.open_turn(
                request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
            ):
                pytest.fail("unclosed prior effect was ignored")
        assert blocked.value.code == "session.recovery_required"


@pytest.mark.parametrize("live_revision", [1, 2, 3, 4])
def test_reconcile_installs_remaining_segment_suffix(live_revision):
    with harness() as h:
        empty = h.session.snapshot
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")
        authority = _authority(h)
        pairs = _history(h)
        live = advance_conversation(
            pairs[live_revision - 1][1], pairs[live_revision - 1][0]
        )
        target = advance_conversation(pairs[-1][1], pairs[-1][0])
    restored = SessionMemory()
    exported = {**empty, "conversations": [conversation_view_to_json(live)]}
    with restored.open_run(snapshot=exported) as session:
        assert session.reconcile(authority) == target
        assert session.snapshot == exported


def test_reconcile_older_segment_does_not_rewind_newer_live_view():
    with harness() as first:
        with first.frame.open_turn(
            request(), checkpoint=first.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")
        authority = _authority(first)
        target = advance_conversation(_history(first)[-1][1], _history(first)[-1][0])
    with harness(first.memory) as second:
        with second.frame.open_turn(
            request("b"), checkpoint=second.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "two", "newer")
        before = snapshot(second.memory)
        assert second.session.reconcile(authority) == target
        assert snapshot(second.memory) == before


@pytest.mark.parametrize("mode", ["raised", "none", "old", "unapplied", "late"])
def test_unknown_or_late_checkpoint_acknowledgement_never_grants_effect_permission(
    mode,
):
    failure = RuntimeError("channel failed")
    with harness() as h:

        def channel(checkpoint):
            if type(checkpoint) is not c.ConversationEffectRequested:
                return h.checkpoint(checkpoint)
            if mode == "raised":
                raise failure
            if mode == "none":
                return None
            if mode == "old":
                return h.accepted_views[checkpoint.stamp.key]
            if mode == "unapplied":
                return advance_conversation(
                    h.accepted_views[checkpoint.stamp.key], checkpoint
                )
            after = h.checkpoint(checkpoint)
            h.session.fail_turn(
                checkpoint.stamp.invocation,
                checkpoint=checkpoint,
                acceptance="accepted",
            )
            return after

        with pytest.raises((RuntimeError, c.MemoryOperationError)) as caught:
            with h.frame.open_turn(
                request(), checkpoint=channel, check_stopped=lambda: None
            ) as turn:
                turn.request("tool", {"name": "must-not-run"})
                pytest.fail("unknown request acknowledgement granted effect permission")
        if mode == "raised":
            assert caught.value is failure
        assert h.memory._fences
        assert h.memory._active == {}
        with h.frame.open_turn(
            request("b", consumer=("root", "other")),
            checkpoint=h.checkpoint,
            check_stopped=lambda: None,
        ):
            pass


def test_completed_observation_is_retained_before_checker_stop():
    failure = RuntimeError("ordinary Exception used as host stop")
    stop = Event()

    def check():
        if stop.is_set():
            raise failure

    with harness() as h:
        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=h.checkpoint, check_stopped=check
            ) as turn:
                ordinal = turn.request("tool", {})
                stop.set()
                turn.observe(
                    ordinal, "completed", items=(item("result", "already happened"),)
                )
        assert caught.value is failure
        last = h.accepted_views[h.accepted[0].stamp.key]
        assert last.effects[0].status == "completed"
        assert last.items == (item("result", "already happened"),)
        assert last.closed_invocations[-1].status == "interrupted"
        assert h.memory._fences == {}


def test_stop_after_ack_blocks_new_effect_but_preserves_accepted_request():
    failure = RuntimeError("stop before external call")
    stop = Event()

    def check():
        if stop.is_set():
            raise failure

    with harness() as h:

        def channel(checkpoint):
            after = h.checkpoint(checkpoint)
            if type(checkpoint) is c.ConversationEffectRequested:
                stop.set()
            return after

        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=channel, check_stopped=check
            ) as turn:
                turn.request("tool", {})
                pytest.fail("stopped request granted effect permission")
        assert caught.value is failure
        last = h.accepted_views[h.accepted[0].stamp.key]
        assert last.effects[0].status == "pending"
        assert last.closed_invocations[-1].status == "interrupted"


def test_body_failure_closes_truthfully_without_inventing_observations():
    failure = RuntimeError("tool could not be observed")
    with harness() as h:
        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=h.checkpoint, check_stopped=lambda: None
            ) as turn:
                turn.request("tool", {})
                raise failure
        assert caught.value is failure
        view = h.accepted_views[h.accepted[0].stamp.key]
        assert view.effects[0].status == "pending"
        assert view.effects[0].observation_ordinals == ()
        assert view.closed_invocations[-1].status == "failed"
        assert not h.memory._fences
        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(
                request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
            ):
                pass
        assert caught.value.code == "session.recovery_required"


def test_completed_close_with_pending_effect_is_rejected_without_fabrication():
    with harness() as h:
        with pytest.raises(c.MemoryOperationError):
            with h.frame.open_turn(
                request(), checkpoint=h.checkpoint, check_stopped=lambda: None
            ) as turn:
                turn.request("model", {})
        assert type(h.accepted[-1]) is c.ConversationEffectRequested
        assert not h.memory._fences
        assert not h.memory._active


def test_failure_finalizer_does_not_upgrade_proven_unaccepted_disposition():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            h.session.fail_turn(
                request().invocation, checkpoint=None, acceptance="unaccepted"
            )
            h.session.fail_turn(
                request().invocation, checkpoint=None, acceptance="uncertain"
            )
        assert not h.memory._fences
        with h.frame.open_turn(
            request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            pass


def test_failure_report_cannot_select_foreign_lease():
    with harness() as one, harness(one.memory) as two:
        with one.frame.open_turn(
            request(), checkpoint=one.checkpoint, check_stopped=lambda: None
        ):
            with pytest.raises(c.MemoryOperationError) as caught:
                two.session.fail_turn(
                    request().invocation,
                    checkpoint=one.accepted[0],
                    acceptance="uncertain",
                )
            assert caught.value.code == "session.conflict"
            assert not one.memory._fences
            assert one.memory._active


def test_active_invocation_cannot_be_shadowed_by_another_key():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            with pytest.raises(c.MemoryOperationError) as caught:
                with h.frame.open_turn(
                    request(consumer=("root", "other")),
                    checkpoint=h.checkpoint,
                    check_stopped=lambda: None,
                ):
                    pass
            assert caught.value.code == "session.conflict"
            assert len(h.memory._active) == 1


def test_checkpoint_policy_must_match_opened_frame():
    memory = SessionMemory()
    with memory.open_run() as session:
        lease = _reserve(session)
        admission = _admission(lease)
        wrong = replace(admission, stamp=replace(admission.stamp, policy_name="wrong"))
        with pytest.raises(c.MemoryOperationError) as caught:
            session.prepare_checkpoint(wrong, before=lease.before)
        assert caught.value.code == "session.conflict"
        assert memory._views == {}
        session._release(lease)


def test_session_close_fences_active_turn_and_keeps_other_sessions_live():
    with harness() as one, harness(one.memory) as two:
        turn = one.frame.open_turn(
            request(), checkpoint=one.checkpoint, check_stopped=lambda: None
        )
        turn.__enter__()
        one.session.close()
        one.session.close()
        with pytest.raises(c.MemoryOperationError) as caught:
            turn.request("model", {})
        assert caught.value.code == "session.stale"
        assert one.memory._fences
        with two.frame.open_turn(
            request("b", consumer=("root", "other")),
            checkpoint=two.checkpoint,
            check_stopped=lambda: None,
        ):
            pass
        assert one.memory._fences


@_bounded_scenario
def test_session_close_removes_own_waiters_without_poisoning_other_owner():
    with harness() as one, harness(one.memory) as two:
        with one.frame.open_turn(
            request(), checkpoint=one.checkpoint, check_stopped=lambda: None
        ):

            def waiting():
                with two.frame.open_turn(
                    request("b"), checkpoint=two.checkpoint, check_stopped=lambda: None
                ):
                    pytest.fail("closed session was admitted")

            worker = _Worker(waiting)
            _wait_queue(one.memory, 1)
            two.session.close()
            worker.join()
            assert len(worker.errors) == 1
            assert worker.errors[0].code == "session.stale"
            assert not one.memory._queues
            assert not one.memory._fences
            assert one.memory._active


@_bounded_scenario
def test_checkpoint_callback_does_not_hold_backend_lock():
    with harness() as h:

        def channel(checkpoint):
            result = []
            worker = _Worker(lambda: result.append(h.checkpoint(checkpoint)))
            worker.join()
            assert not worker.errors
            return result[0]

        with h.frame.open_turn(
            request(), checkpoint=channel, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "hello")


def test_duplicate_rejects_well_formed_alternate_predecessor():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "real")
        pairs = _history(h)
        observation, before_observation = pairs[2]
        alternate = replace(observation, items=(item("one", "alternate"),))
        forged_before = advance_conversation(before_observation, alternate)
        # This is a fully valid, resealed alternate history, not corrupt wire.
        conversation_view_to_json(forged_before)
        with pytest.raises(c.MemoryOperationError) as caught:
            h.session.prepare_checkpoint(pairs[-1][0], before=forged_before)
        assert caught.value.code == "session.conflict"


def test_fence_digest_in_baseline_does_not_clear_acceptance_fence():
    with harness() as first:
        with first.frame.open_turn(
            request(), checkpoint=first.checkpoint, check_stopped=lambda: None
        ):
            pass
        first_authority = _authority(first)
    with harness(first.memory) as second:
        with second.frame.open_turn(
            request("b"), checkpoint=second.checkpoint, check_stopped=lambda: None
        ):
            pass
        next_authority = _authority(second)
        latest = snapshot(second.memory)
    restored = SessionMemory()
    with restored.open_run(snapshot={**latest, "conversations": []}) as session:
        frame = session.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        lease = session._reserve(frame, request(), lambda: None)
        session.fail_turn(
            request().invocation,
            checkpoint=first_authority.checkpoints[-1],
            acceptance="accepted",
        )
        before = snapshot(restored)
        with pytest.raises(c.MemoryOperationError) as caught:
            session.reconcile(next_authority)
        assert caught.value.code == "session.recovery_required"
        assert snapshot(restored) == before
        session.reconcile(first_authority)
        assert not restored._fences


def test_unknown_checkpoint_fence_requires_reset():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            h.session.fail_turn(
                request().invocation, checkpoint=None, acceptance="uncertain"
            )
        with pytest.raises(c.MemoryOperationError) as caught:
            h.session.reconcile(_authority(h))
        assert caught.value.code == "session.recovery_required"
        assert h.memory._fences


def test_prepare_admission_cannot_change_reserved_compatibility():
    with SessionMemory().open_run() as session:
        lease = _reserve(session)
        before = lease.before
        alternate = c.ConversationView(
            before.namespace,
            before.generation,
            0,
            before.key,
            {"format": "other"},
            (),
            None,
            (),
            (),
            (),
        )
        checkpoint = replace(_admission(lease), baseline=alternate)
        with pytest.raises(c.MemoryOperationError) as caught:
            session.prepare_checkpoint(checkpoint, before=alternate)
        assert caught.value.code == "session.incompatible"
        assert not session._memory._views
        session._release(lease)


def test_reconcile_rejects_competing_active_lease_without_mutation():
    with harness() as source:
        initial = source.session.snapshot
        with source.frame.open_turn(
            request(), checkpoint=source.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "source")
        authority = _authority(source)
    restored = SessionMemory()
    with restored.open_run(snapshot=initial) as session:
        lease = _reserve(session, request("competing"))
        before = snapshot(restored)
        with pytest.raises(c.MemoryOperationError) as caught:
            session.reconcile(authority)
        assert caught.value.code == "session.conflict"
        assert snapshot(restored) == before
        assert restored._active[lease.key] is lease
        assert not restored._fences
        session._release(lease)


def test_reconcile_equal_revision_different_view_publishes_nothing():
    with harness() as source:
        initial = source.session.snapshot
        with source.frame.open_turn(
            request(), checkpoint=source.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "source")
        authority = _authority(source)
    restored = SessionMemory()
    with restored.open_run(snapshot=initial):
        pass
    with harness(restored) as other:
        with other.frame.open_turn(
            request(), checkpoint=other.checkpoint, check_stopped=lambda: None
        ) as turn:
            append_reply(turn, "one", "different")
        before = snapshot(restored)
        with pytest.raises(c.MemoryOperationError) as caught:
            other.session.reconcile(authority)
        assert caught.value.code == "session.conflict"
        assert snapshot(restored) == before
        assert not restored._fences
        assert not restored._active


@_bounded_scenario
def test_reset_rejects_queued_ticket_even_without_an_active_lease():
    with harness() as h:
        checking = Event()
        resume = Event()

        def check():
            checking.set()
            assert resume.wait(2)

        def waiting():
            with h.frame.open_turn(
                request(), checkpoint=h.checkpoint, check_stopped=check
            ):
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


def test_snapshot_envelope_depth_is_rejected_before_prepare_or_append():
    memory = SessionMemory()
    with memory.open_run() as session:
        lease = _reserve(session)
        admission = _admission(lease)
        before = session.prepare_checkpoint(admission, before=lease.before)()
        stamp = replace(admission.stamp, expected_revision=1, ordinal=1)
        candidate = None
        for depth in range(115, 128):
            checkpoint = c.ConversationEffectRequested(
                stamp, "model", _nested(depth), ()
            )
            try:
                after = advance_conversation(before, checkpoint)
            except ValueError:
                continue
            exported = {
                **session.snapshot,
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
        exported_before = snapshot(memory)
        with pytest.raises(ValueError, match="nesting"):
            session.prepare_checkpoint(candidate, before=before)
        assert snapshot(memory) == exported_before
        assert memory._active[lease.key] is lease
        session._release(lease)


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
        with source.frame.open_turn(
            request(), checkpoint=source.checkpoint, check_stopped=lambda: None
        ):
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
                checkpoint=h.checkpoint,
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


def test_stop_checker_cannot_release_lease_then_grant_effect_permission():
    with harness() as h:

        def check():
            if h.accepted and type(h.accepted[-1]) is c.ConversationEffectRequested:
                h.session.fail_turn(
                    request().invocation,
                    checkpoint=h.accepted[-1],
                    acceptance="accepted",
                )

        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=h.checkpoint, check_stopped=check
            ) as turn:
                turn.request("tool", {})
                pytest.fail("released lease granted effect permission")
        assert caught.value.code == "session.stale"
        assert h.memory._fences


def test_observed_stop_cannot_be_caught_to_reenable_new_effects():
    with harness() as h:
        stop = Event()
        failure = RuntimeError("stop once")

        def check():
            if stop.is_set():
                raise failure

        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=h.checkpoint, check_stopped=check
            ) as turn:
                stop.set()
                with pytest.raises(RuntimeError):
                    turn.request("tool", {})
                stop.clear()
                turn.request("tool", {})
                pytest.fail("observed stop was forgotten")
        assert caught.value is failure
        assert all(
            type(checkpoint) is not c.ConversationEffectRequested
            for checkpoint in h.accepted
        )


def test_accepted_unapplied_checkpoint_reconciles_without_reviving_old_turn():
    with harness() as h:
        failure = RuntimeError("apply unavailable after acceptance")
        tokens = []

        def channel(checkpoint):
            if type(checkpoint) is not c.ConversationEffectRequested:
                return h.checkpoint(checkpoint)
            before = h.accepted_views[checkpoint.stamp.key]
            tokens.append(h.session.prepare_checkpoint(checkpoint, before=before))
            h.accepted.append(checkpoint)
            h.accepted_views[checkpoint.stamp.key] = advance_conversation(
                before, checkpoint
            )
            h.session.fail_turn(
                checkpoint.stamp.invocation,
                checkpoint=checkpoint,
                acceptance="accepted",
            )
            raise failure

        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request(), checkpoint=channel, check_stopped=lambda: None
            ) as turn:
                turn.request("tool", {})
        assert caught.value is failure
        assert next(iter(h.memory._views.values())).revision == 1
        with pytest.raises(c.MemoryOperationError):
            tokens[0]()
        after = h.session.reconcile(_authority(h))
        assert after.revision == 2
        assert after.effects[0].status == "pending"
        assert tokens[0]() == after
        assert not h.memory._fences
        with pytest.raises(c.MemoryOperationError) as stale:
            turn.request("tool", {})
        assert stale.value.code == "session.stale"


def test_acceptance_fence_survives_all_run_session_closures():
    memory = SessionMemory()
    failure = RuntimeError("lost admission channel")
    with harness(memory) as h:

        def fail(checkpoint):
            raise failure

        with pytest.raises(RuntimeError):
            with h.frame.open_turn(
                request(), checkpoint=fail, check_stopped=lambda: None
            ):
                pass
    assert memory._fences
    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(
                request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
            ):
                pass
        assert caught.value.code == "session.recovery_required"
        assert h.accepted == []
        assert set(snapshot(memory)) == {
            "version",
            "namespace",
            "generation",
            "conversations",
        }


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
                checkpoint=h.checkpoint,
                check_stopped=lambda: None,
            ) as turn:
                assert len(turn.view.items) == (0 if run == "a" else 1)
                append_reply(turn, run, run)
        if owner is c.MemoryOwner.FLOW:
            with h.frame.open_turn(
                request("c", consumer=("root", "call_b", "agent")),
                checkpoint=h.checkpoint,
                check_stopped=lambda: None,
            ) as turn:
                assert not turn.view.items
        else:
            with pytest.raises(ValueError):
                with h.frame.open_turn(
                    request("c", consumer=("root", "call_b", "agent")),
                    checkpoint=h.checkpoint,
                    check_stopped=lambda: None,
                ):
                    pass


def test_used_backend_rejects_newer_same_namespace_snapshot():
    with harness() as source:
        initial = source.session.snapshot
        with source.frame.open_turn(
            request(), checkpoint=source.checkpoint, check_stopped=lambda: None
        ):
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


def test_invalid_checkpoint_wire_finalizes_local_rejection_without_reencoding():
    with harness() as h:
        with h.frame.open_turn(
            request(), checkpoint=h.checkpoint, check_stopped=lambda: None
        ) as turn:
            accepted = tuple(h.accepted)
            lease = turn._lease
            with pytest.raises(ValueError, match="nesting") as rejected:
                turn.request("model", _nested(127))
            assert tuple(h.accepted) == accepted
            assert not getattr(rejected.value, "__notes__", ())
            assert lease.poisoned
            assert lease.released
            assert not lease.active
            assert lease.failure == "unaccepted"
            assert not h.memory._fences
            assert not h.memory._active
            with pytest.raises(c.MemoryOperationError) as stale:
                turn.request("model", {})
            assert stale.value.code == "session.stale"
            assert tuple(h.accepted) == accepted
        with h.frame.open_turn(
            request("b"), checkpoint=h.checkpoint, check_stopped=lambda: None
        ):
            pass


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
            with h.frame.open_turn(
                request(), checkpoint=h.checkpoint, check_stopped=lambda: None
            ):
                pytest.fail("a closed session admitted its reserved turn")

        worker = _Worker(entering)
        try:
            assert reserved.wait(2)
            assert leases[0].before is not None
            h.session.close()
            assert leases[0].before is None
        finally:
            continue_admission.set()
            worker.join()
        assert len(worker.errors) == 1
        failure = worker.errors[0]
        assert type(failure) is c.MemoryOperationError, repr(failure)
        assert (failure.stage, failure.code, failure.retryable) == (
            "conversation",
            "session.stale",
            False,
        )
        assert h.accepted == []
        assert not h.memory._active
        assert not h.memory._queues
