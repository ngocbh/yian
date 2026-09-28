"""Learned self-feedback remains separate from automatic session history."""

from copy import deepcopy
import hashlib
import json

import pytest

import agent_composer.memory.contracts as c
from agent_composer.memory.session import SessionMemory
from tests.engine._session_fakes import REF, harness, request, snapshot


def _note(
    *,
    ref=REF,
    consumer=("root", "agent"),
    run_id="a",
    sequence=1,
    index=0,
    text="Check units.",
):
    value = {
        "frame_ref": {
            "root_flow_id": ref.root_flow_id,
            "static_path": list(ref.static_path),
            "owner": ref.owner.value,
        },
        "consumer_static_path": list(consumer),
        "run_id": run_id,
        "execution_seq": sequence,
        "message_index": index,
        "text": text,
    }
    wire = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {"note_id": hashlib.sha256(wire.encode()).hexdigest(), **value}


def _with_notes(*notes):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        value = snapshot(h.memory)
    return {**value, "version": 2, "notes": list(notes)}


def test_version_one_snapshot_stays_exact_and_detached_when_pinned():
    legacy = {key: value for key, value in _with_notes().items() if key != "notes"}
    legacy["version"] = 1
    expected = deepcopy(legacy)
    memory = SessionMemory()
    with memory.open_run(snapshot=legacy) as opened:
        legacy["conversations"].clear()
        assert opened.snapshot == expected
        exported = opened.snapshot
        exported["conversations"].clear()
        assert opened.snapshot == expected
        assert snapshot(memory) == {**expected, "version": 2, "notes": []}


@pytest.mark.parametrize("text", ["Check units.", ""])
def test_version_two_snapshot_adopts_detached_notes(text):
    value = _with_notes(_note(text=text))
    expected = deepcopy(value)
    memory = SessionMemory()
    with memory.open_run(snapshot=value) as opened:
        value["notes"][0]["text"] = "changed outside"
        assert opened.snapshot == expected
        assert snapshot(memory) == expected


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "extra",
        "missing",
        "bad_id",
        "foreign_frame",
        "foreign_leaf",
        "foreign_run",
        "foreign_sequence",
        "bool_sequence",
        "negative_index",
        "bool_index",
        "nontext",
        "nan",
        "notes_object",
        "frame_extra",
    ],
)
def test_snapshot_rejects_invalid_or_unadmitted_notes_without_adoption(case):
    value = _with_notes(_note())
    note = value["notes"][0]
    if case == "duplicate":
        value["notes"] *= 2
    elif case == "extra":
        note["extra"] = None
    elif case == "missing":
        del note["text"]
    elif case == "bad_id":
        note["note_id"] = "wrong"
    elif case == "foreign_frame":
        value["notes"] = [
            _note(
                ref=c.MemoryFrameRef("elsewhere", ("elsewhere",), c.MemoryOwner.FLOW),
                consumer=("elsewhere", "agent"),
            )
        ]
    elif case == "foreign_leaf":
        value["notes"] = [_note(consumer=("root", "sibling"))]
    elif case == "foreign_run":
        value["notes"] = [_note(run_id="elsewhere")]
    elif case == "foreign_sequence":
        value["notes"] = [_note(sequence=2)]
    elif case == "bool_sequence":
        value["notes"] = [_note(sequence=True)]
    elif case == "negative_index":
        value["notes"] = [_note(index=-1)]
    elif case == "bool_index":
        value["notes"] = [_note(index=True)]
    elif case == "nontext":
        value["notes"] = [_note(text={"text": "wrong"})]
    elif case == "nan":
        note["text"] = float("nan")
    elif case == "notes_object":
        value["notes"] = {}
    elif case == "frame_extra":
        note["frame_ref"]["extra"] = None
    memory = SessionMemory()
    with pytest.raises(c.MemoryOperationError) as caught:
        memory.open_run(snapshot=value)
    assert caught.value.code == "session.invalid_snapshot"
    assert not memory._used


def test_historical_note_prefix_is_pinned_without_rewinding_live_state():
    full = _with_notes(_note(), _note(index=1, text="Use exact arithmetic."))
    memory = SessionMemory()
    with memory.open_run(snapshot=full):
        pass
    older = deepcopy(full)
    older["notes"] = older["notes"][:1]
    with memory.open_run(snapshot=older) as opened:
        assert opened.snapshot == older
        assert snapshot(memory) == full
    for notes in (
        full["notes"][1:],
        list(reversed(full["notes"])),
        [_note(text="Changed")],
    ):
        with pytest.raises(c.MemoryOperationError):
            memory.open_run(snapshot={**full, "notes": notes})


def _context(
    *,
    ref=REF,
    consumer=("root", "agent"),
    run_id="a",
    sequence=1,
    texts=("Check units.",),
    channel="self",
    feedback_sequence=1,
):
    episode = c.MemoryEpisode(
        run_id,
        c.MemoryOccurrenceRef(ref),
        consumer,
        c.MemoryEpisodeStatus.SUCCEEDED,
        {},
        execution_seq=sequence,
    )
    feedback = tuple(
        {
            "execution_seq": feedback_sequence,
            "owner_static_path": list(consumer[1:]),
            "occurrence_path": [],
            "execution_scope": {
                "static_path": list(consumer[1:]),
                "occurrence_path": [],
            },
            "channel": channel,
            "target_name": None,
            "value": text,
            "message_index": index,
            "contribution_context": {},
        }
        for index, text in enumerate(texts)
    )
    return c.LearnContext(run_id, ref, (episode,), feedback)


def test_learn_is_pure_deterministic_self_feedback_projection():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        before = snapshot(h.memory)
        context = _context(texts=("Check units.", ""))
        first = h.frame.learn(context)
        assert len(first) == 2
        assert first == h.frame.learn(context)
        assert snapshot(h.memory) == before
        for index, mutation in enumerate(first):
            expected = _note(index=index, text=("Check units.", "")[index])
            assert mutation.frame_ref == REF
            assert mutation.mutation_id == expected["note_id"]
            assert mutation.payload == {"operation": "append_note", "note": expected}
        assert (
            first[0].mutation_id
            != h.frame.learn(_context(texts=("Distinct",)))[0].mutation_id
        )


@pytest.mark.parametrize(
    "case",
    [
        "no_feedback",
        "other_channel",
        "root_only",
        "unrelated_leaf",
        "other_run",
        "unadmitted_start",
        "no_admission",
        "zero_sequence",
        "missing_sequence",
    ],
)
def test_learn_ignores_feedback_without_owning_admitted_leaf(case):
    with harness() as h:
        if case != "no_admission":
            with h.frame.open_turn(request(), check_stopped=lambda: None):
                pass
        kwargs = {}
        if case == "no_feedback":
            kwargs["texts"] = ()
        elif case == "other_channel":
            kwargs["channel"] = "input"
        elif case == "unrelated_leaf":
            kwargs["consumer"] = ("root", "sibling")
        elif case == "other_run":
            kwargs["run_id"] = "another-run"
        elif case == "unadmitted_start":
            kwargs.update(sequence=2, feedback_sequence=2)
        elif case == "zero_sequence":
            kwargs.update(sequence=0, feedback_sequence=0)
        elif case == "missing_sequence":
            kwargs.update(sequence=None, feedback_sequence=None)
        context = _context(**kwargs)
        if case == "root_only":
            context = c.LearnContext(
                "a",
                REF,
                context.episodes,
                (
                    {
                        "execution_seq": None,
                        "owner_static_path": None,
                        "root": True,
                        "channel": "self",
                        "value": "Root only",
                        "message_index": 0,
                    },
                ),
            )
        assert h.frame.learn(context) == []


def test_outer_recall_only_owner_cannot_duplicate_inner_learning():
    leaf = c.MemoryFrameRef("root", ("root", "agent"), c.MemoryOwner.AGENT)
    with harness() as h:
        inner = h.session.open_frame(c.MemoryFrameSpec(leaf, "session", {}))
        with inner.open_turn(request(), check_stopped=lambda: None):
            pass
        assert len(inner.learn(_context(ref=leaf))) == 1
        assert h.frame.learn(_context()) == []


def test_learn_rejects_foreign_frame_context():
    other = c.MemoryFrameRef("other", ("other",), c.MemoryOwner.FLOW)
    with harness() as h:
        with pytest.raises(c.MemoryOperationError) as caught:
            h.frame.learn(c.LearnContext("a", other, ()))
        assert (caught.value.stage, caught.value.code) == (
            "learn",
            "session.invalid_context",
        )


def _mutation(note=None, *, ref=REF):
    note = _note() if note is None else note
    return c.MemoryMutation(
        note["note_id"], ref, {"operation": "append_note", "note": note}
    )


def _recall(frame, consumer=("root", "agent")):
    return frame.recall(c.RecallContext("system", "prompt", {}, consumer)).text


def test_commit_publishes_notes_only_to_future_run_recall():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        history = snapshot(h.memory)["conversations"]
        mutations = tuple(h.frame.learn(_context()))
        h.session.commit("learn-a", mutations)
        assert snapshot(h.memory)["notes"] == [_note()]
        assert snapshot(h.memory)["conversations"] == history
        assert _recall(h.frame) == ""
        with harness(h.memory) as later:
            assert _recall(later.frame) == "Check units."
            assert later.frame.recall() == c.MemoryContent(transcript=later.frame)
            assert _recall(later.frame, ("root", "sibling")) == ""
            other = later.session.open_frame(
                c.MemoryFrameSpec(
                    c.MemoryFrameRef("root", ("root", "agent"), c.MemoryOwner.AGENT),
                    "session",
                    {},
                )
            )
            assert _recall(other) == ""


def test_exact_commit_replay_and_stable_note_ids_are_idempotent():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        batch = (_mutation(),)
        h.session.commit("same", batch)
        h.session.commit("same", batch)
        h.session.commit("another", batch)
        assert snapshot(h.memory)["notes"] == [_note()]
        distinct = _mutation(_note(text="A distinct observation"))
        h.session.commit("new-feedback", (distinct,))
        expected = snapshot(h.memory)
        with pytest.raises(c.MemoryOperationError) as caught:
            h.session.commit("same", (distinct,))
        assert caught.value.code == "session.commit_collision"
        assert snapshot(h.memory) == expected


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "operation",
        "wrong_id",
        "wrong_owner",
        "unopened_owner",
        "unadmitted",
        "duplicate",
        "nonmutation",
    ],
)
def test_invalid_last_mutation_publishes_no_notes_or_commit_tombstone(case):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        before = snapshot(h.memory)
        first = _mutation()
        second = _mutation(_note(index=1))
        if case == "extra":
            second = c.MemoryMutation(
                second.mutation_id, REF, {**second.payload, "extra": None}
            )
        elif case == "operation":
            second = c.MemoryMutation(
                second.mutation_id, REF, {**second.payload, "operation": "other"}
            )
        elif case == "wrong_id":
            second = c.MemoryMutation("wrong", REF, second.payload)
        elif case in {"wrong_owner", "unopened_owner"}:
            leaf = c.MemoryFrameRef("root", ("root", "agent"), c.MemoryOwner.AGENT)
            if case == "wrong_owner":
                h.session.open_frame(c.MemoryFrameSpec(leaf, "session", {}))
            second = c.MemoryMutation(second.mutation_id, leaf, second.payload)
        elif case == "unadmitted":
            second = _mutation(_note(run_id="missing", index=1))
        elif case == "duplicate":
            second = first
        elif case == "nonmutation":
            second = {}
        with pytest.raises(c.MemoryOperationError) as caught:
            h.session.commit("atomic", (first, second))
        assert caught.value.code == "session.invalid_batch"
        assert snapshot(h.memory) == before
        h.session.commit("atomic", (first,))
        assert snapshot(h.memory)["notes"] == [_note()]


def test_empty_commit_identity_cannot_later_publish_a_different_batch():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        h.session.commit("empty", ())
        with pytest.raises(c.MemoryOperationError) as caught:
            h.session.commit("empty", (_mutation(),))
        assert caught.value.code == "session.commit_collision"
        assert snapshot(h.memory)["notes"] == []


def test_reset_clears_notes_and_keeps_commit_replay_tombstones():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        batch = (_mutation(),)
        h.session.commit("before-reset", batch)
        old = snapshot(h.memory)
        h.memory.reset()
        with pytest.raises(c.MemoryOperationError):
            h.session.commit("before-reset", batch)
        with pytest.raises(c.MemoryOperationError):
            _recall(h.frame)
        with harness(h.memory) as later:
            later.session.commit("before-reset", batch)
            assert _recall(later.frame) == ""
            current = snapshot(h.memory)
            assert current["notes"] == [] and current["conversations"] == []
            assert current["generation"] == old["generation"] + 1
            with pytest.raises(c.MemoryOperationError):
                later.session.commit("new-identity", batch)


def test_restored_notes_recall_and_accept_idempotent_note_republication():
    value = _with_notes(_note(), _note(index=1, text=""))
    memory = SessionMemory()
    with memory.open_run(snapshot=value) as opened:
        frame = opened.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        assert _recall(frame) == "Check units.\n"
        opened.commit("restored", tuple(_mutation(note) for note in value["notes"]))
        assert snapshot(memory) == value


def test_version_one_pin_cannot_observe_later_notes():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        old = snapshot(h.memory)
        del old["notes"]
        old["version"] = 1
        h.session.commit("later", (_mutation(),))
        with h.memory.open_run(snapshot=old) as opened:
            frame = opened.open_frame(c.MemoryFrameSpec(REF, "session", {}))
            assert opened.snapshot == old
            assert _recall(frame) == ""
        assert snapshot(h.memory)["notes"] == [_note()]


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_seq", True),
        ("message_index", True),
        ("message_index", -1),
        ("message_index", None),
        ("value", None),
        ("value", {"text": "not text"}),
    ],
)
def test_malformed_attributed_self_feedback_fails_without_publication(field, value):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        before = snapshot(h.memory)
        context = _context()
        feedback = {**context.feedback[0], field: value}
        with pytest.raises(c.MemoryOperationError) as caught:
            h.frame.learn(c.LearnContext("a", REF, context.episodes, (feedback,)))
        assert caught.value.code == "session.invalid_context"
        assert snapshot(h.memory) == before


def test_repeat_source_feedback_in_one_context_proposes_one_note():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        context = _context()
        duplicated = c.LearnContext("a", REF, context.episodes, context.feedback * 2)
        assert h.frame.learn(duplicated) == [_mutation()]


def test_candidate_failure_keeps_notes_and_commit_identity_unpublished(monkeypatch):
    import agent_composer.memory.session as implementation

    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        before = snapshot(h.memory)
        original = implementation._snapshot

        def reject(*args):
            raise ValueError("candidate validation failed")

        monkeypatch.setattr(implementation, "_snapshot", reject)
        with pytest.raises(c.MemoryOperationError) as caught:
            h.session.commit("candidate", (_mutation(),))
        assert caught.value.code == "session.invalid_batch"
        monkeypatch.setattr(implementation, "_snapshot", original)
        assert snapshot(h.memory) == before
        h.session.commit("candidate", (_mutation(),))
        assert snapshot(h.memory)["notes"] == [_note()]


def test_commit_requires_opened_owner_even_for_exact_replay():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        h.session.commit("owned", (_mutation(),))
        with h.memory.open_run() as unopened:
            with pytest.raises(c.MemoryOperationError) as caught:
                unopened.commit("owned", (_mutation(),))
            assert caught.value.code == "session.invalid_batch"


def test_ancestor_can_recall_old_notes_without_learning_inner_invocation():
    leaf = c.MemoryFrameRef("root", ("root", "agent"), c.MemoryOwner.AGENT)
    with harness() as first:
        with first.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        first.session.commit("flow-note", tuple(first.frame.learn(_context())))
        with harness(first.memory) as later:
            assert _recall(later.frame) == "Check units."
            inner = later.session.open_frame(c.MemoryFrameSpec(leaf, "session", {}))
            with inner.open_turn(request(run_id="b"), check_stopped=lambda: None):
                pass
            assert later.frame.learn(_context(run_id="b")) == []
            later.session.commit(
                "leaf-note", tuple(inner.learn(_context(ref=leaf, run_id="b")))
            )
        assert len(snapshot(first.memory)["notes"]) == 2


def test_busy_reset_preserves_notes_and_forward_turn_updates_preserve_notes():
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None):
            pass
        h.session.commit("note", (_mutation(),))
        with h.frame.open_turn(request(run_id="b"), check_stopped=lambda: None):
            before = snapshot(h.memory)
            with pytest.raises(c.MemoryOperationError) as caught:
                h.memory.reset()
            assert caught.value.code == "session.busy"
            assert snapshot(h.memory) == before
        assert snapshot(h.memory)["notes"] == [_note()]


def test_reset_keeps_empty_commit_tombstones():
    with harness() as h:
        h.session.commit("empty", ())
        h.memory.reset()
        with harness(h.memory) as later:
            with later.frame.open_turn(request(), check_stopped=lambda: None):
                pass
            with pytest.raises(c.MemoryOperationError) as caught:
                later.session.commit("empty", (_mutation(),))
            assert caught.value.code == "session.commit_collision"
