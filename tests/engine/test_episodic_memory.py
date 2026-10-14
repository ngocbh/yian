"""Session and snapshot foundations for process-local episodic memory."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from yian.memory.contracts import (
    LearnContext,
    Memory,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOperationError,
    MemorySection,
    MemorySession,
    ModelTextInteraction,
    ModelValueInteraction,
    RecallContext,
    ToolCallInteraction,
    ToolResultInteraction,
)
from yian.memory.episodic import EpisodicMemory
import yian.memory.episodic as episodic


def _context(memory: str = "episodic") -> LearnContext:
    episode = MemoryEpisode(
        "run", (), ("agent",), MemoryEpisodeStatus.SUCCEEDED, {}, "ok"
    )
    return LearnContext("run", memory, (episode,))


def test_constructor_and_protocol_are_exact() -> None:
    for value in (True, 0, -1, 1.0):
        error = TypeError if value in (True, 1.0) else ValueError
        with pytest.raises(error):
            EpisodicMemory(max_episodes=value)  # type: ignore[arg-type]
        with pytest.raises(error):
            EpisodicMemory(recall_limit=value)  # type: ignore[arg-type]
    memory = EpisodicMemory(max_episodes=7, recall_limit=3)
    assert isinstance(memory, Memory)
    with memory.open_run() as session:
        assert isinstance(session, MemorySession)
        assert not hasattr(session, "open_frame")


def test_session_operations_validate_context_and_generation() -> None:
    memory = EpisodicMemory()
    session = memory.open_run()
    frame = session
    assert frame.recall() == ""
    assert frame.recall(RecallContext("agent", "prompt", {}, ("root",))) == ""
    assert len(frame.learn(_context())) == 1
    for operation in (
        lambda: frame.recall(object()),
        lambda: frame.learn(object()),
    ):
        with pytest.raises(MemoryOperationError) as exc:
            operation()  # type: ignore[misc]
        assert exc.value.code == "episodic.invalid_context"
    memory.reset()
    assert frame.recall() == ""
    with pytest.raises(MemoryOperationError) as exc:
        frame.learn(_context())
    assert exc.value.code == "episodic.stale_generation"
    with pytest.raises(MemoryOperationError) as commit:
        session.commit("stale", ())
    assert commit.value.code == "episodic.stale_generation"
    session.close()


def test_keys_of_one_instance_share_one_history_and_one_limit() -> None:
    memory = EpisodicMemory(max_episodes=2, recall_limit=1)
    with memory.open_run() as session:
        batch = (
            *session.learn(_context("first")),
            *session.learn(_context("second")),
        )
        session.commit("both", batch)
    with memory.open_run() as session:
        snapshot = session.snapshot
        assert snapshot["version"] == 2
        assert set(snapshot) == {
            "config",
            "episodes",
            "epoch",
            "format",
            "namespace",
            "revision",
            "version",
        }
        assert snapshot["config"] == {"max_episodes": 2, "recall_limit": 1}
        assert len(snapshot["episodes"]) == 2
        assert len(session.recall().splitlines()) == 2
        third = tuple(session.learn(_context("third")))
        session.commit("third", third)
    with memory.open_run() as session:
        retained = [item["entry_id"] for item in session.snapshot["episodes"]]
        assert retained == [batch[1].mutation_id, third[0].mutation_id]


def test_unset_limits_adopt_recorded_ones_and_explicit_ones_must_match() -> None:
    with EpisodicMemory().open_run() as session:
        assert session.snapshot["config"] == {"max_episodes": 100, "recall_limit": 10}
    with EpisodicMemory(max_episodes=4, recall_limit=2).open_run() as session:
        recorded = session.snapshot
    with EpisodicMemory().open_run(snapshot=recorded) as session:
        assert session.snapshot == recorded
    with EpisodicMemory(max_episodes=4).open_run(snapshot=recorded) as session:
        assert session.snapshot == recorded
    for kwargs in ({"max_episodes": 5}, {"recall_limit": 3}):
        with pytest.raises(MemoryOperationError) as exc:
            EpisodicMemory(**kwargs).open_run(snapshot=recorded)
        assert exc.value.code == "episodic.config_conflict"
        assert exc.value.retryable is False


def test_snapshot_is_closed_canonical_and_defensive() -> None:
    memory = EpisodicMemory(max_episodes=4, recall_limit=2)
    with memory.open_run() as session:
        snapshot = session.snapshot
        assert set(snapshot) == {
            "config",
            "episodes",
            "epoch",
            "format",
            "namespace",
            "revision",
            "version",
        }
        assert snapshot["format"] == "yian.episodic_memory"
        assert snapshot["version"] == 2
        assert snapshot["config"] == {"max_episodes": 4, "recall_limit": 2}
        assert snapshot["episodes"] == []
        snapshot["config"]["max_episodes"] = 99
        assert session.snapshot["config"]["max_episodes"] == 4
        json.dumps(session.snapshot, sort_keys=True)


def test_snapshot_restore_rules_and_recorded_defaults() -> None:
    original = EpisodicMemory(max_episodes=4, recall_limit=2)
    with original.open_run() as session:
        session.commit("declare", ())
    with original.open_run() as session:
        recorded = session.snapshot
    with original.open_run(snapshot=recorded) as current:
        assert current.snapshot == recorded
    restored = EpisodicMemory()
    with restored.open_run(snapshot=recorded) as session:
        assert session.snapshot == recorded
    nonpristine = EpisodicMemory()
    with nonpristine.open_run():
        pass
    with pytest.raises(MemoryOperationError):
        nonpristine.open_run(snapshot=recorded)
    for field, value in (("revision", 2), ("epoch", 1)):
        changed = json.loads(json.dumps(recorded))
        changed[field] = value
        with pytest.raises(MemoryOperationError):
            original.open_run(snapshot=changed)
    changed = json.loads(json.dumps(recorded))
    changed["config"]["recall_limit"] = 3
    with pytest.raises(MemoryOperationError):
        original.open_run(snapshot=changed)
    original.reset()
    with pytest.raises(MemoryOperationError):
        original.open_run(snapshot=recorded)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(extra=True),
        lambda x: x.update(version=True),
        lambda x: x.update(format="other"),
        lambda x: x.update(namespace=""),
        lambda x: x.update(epoch=True),
        lambda x: x.update(revision=-1),
        lambda x: x.update(version=1),
        lambda x: x.update(frames=[]),
        lambda x: x.update(episodes={}),
        lambda x: x["config"].update(max_episodes=0),
        lambda x: x["config"].update(extra=1),
        lambda x: x["episodes"].append({}),
    ],
)
def test_snapshot_format_is_strict(mutate: Any) -> None:
    with EpisodicMemory().open_run() as session:
        snapshot = session.snapshot
    malformed = json.loads(json.dumps(snapshot))
    mutate(malformed)
    with pytest.raises(MemoryOperationError) as exc:
        EpisodicMemory().open_run(snapshot=malformed)
    assert exc.value.code == "episodic.invalid_snapshot"
    assert exc.value.retryable is False


def test_empty_commit_idempotency_collision_and_staleness() -> None:
    memory = EpisodicMemory()
    session = memory.open_run()
    session.commit("empty", ())
    session.commit("empty", ())
    with memory.open_run() as fresh:
        assert fresh.snapshot["revision"] == 1
    with pytest.raises(MemoryOperationError) as collision:
        session.commit("empty", (MemoryMutation("m", {}),))
    assert collision.value.code == "episodic.commit_collision"
    with pytest.raises(MemoryOperationError) as nonempty:
        memory.open_run().commit("nonempty", (MemoryMutation("m", {}),))
    assert nonempty.value.code == "episodic.invalid_batch"
    stale = memory.open_run()
    with memory.open_run() as current:
        current.commit("next", ())
    with pytest.raises(MemoryOperationError) as exc:
        stale.commit("stale", ())
    assert exc.value.code == "episodic.stale_generation"
    stale.close()
    session.close()


def test_commit_inputs_are_exact() -> None:
    with EpisodicMemory().open_run() as session:
        for commit_id in ("", True, 1, None):
            with pytest.raises(MemoryOperationError) as exc:
                session.commit(commit_id, ())  # type: ignore[arg-type]
            assert exc.value.code == "episodic.invalid_commit_id"
        with pytest.raises(MemoryOperationError) as exc:
            session.commit("id", [])  # type: ignore[arg-type]
        assert exc.value.code == "episodic.invalid_batch"


def test_closed_session_rejects_every_operation() -> None:
    session = EpisodicMemory().open_run()
    frame = session
    session.close()
    session.close()
    operations = (
        lambda: session.snapshot,
        lambda: session.__enter__(),
        lambda: session.commit("x", ()),
        lambda: frame.recall(),
        lambda: frame.learn(_context()),
    )
    for operation in operations:
        with pytest.raises(MemoryOperationError) as exc:
            operation()
        assert exc.value.code == "episodic.session_closed"


def test_module_imports_only_stdlib_and_lower_memory_modules() -> None:
    tree = ast.parse(Path(episodic.__file__ or "").read_text())
    imported = {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    imported.update(
        n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    )
    assert imported <= {
        "__future__",
        "contextlib",
        "dataclasses",
        "hashlib",
        "json",
        "os",
        "threading",
        "types",
        "typing",
        "uuid",
        "yian.json_codec",
        "yian.memory.contracts",
        "yian.memory.storage",
    }


def _episode(
    label: str,
    *,
    status: MemoryEpisodeStatus = MemoryEpisodeStatus.SUCCEEDED,
) -> MemoryEpisode:
    return MemoryEpisode(
        f"run-{label}",
        (MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ("loop",), 2),),
        ("loop", "agent"),
        status,
        {"topic": label},
        None if status is MemoryEpisodeStatus.FAILED else {"answer": label},
        (
            {"code": "failed", "message": label}
            if status is MemoryEpisodeStatus.FAILED
            else None
        ),
        (
            ModelTextInteraction(f"text {label}"),
            ModelValueInteraction({"value": label}),
            ToolCallInteraction("lookup", {"query": label}),
            ToolResultInteraction("lookup", f"result {label}"),
        ),
        "system",
        f"prompt {label}",
        (MemorySection("episodic", f"memory {label}"),),
        execution_seq=7,
    )


def _learn(memory: EpisodicMemory, episode: MemoryEpisode, commit_id: str) -> None:
    with memory.open_run() as session:
        frame = session
        mutations = tuple(frame.learn(LearnContext(episode.run_id, "episodic", (episode,))))
        session.commit(commit_id, mutations)


def test_learning_is_deterministic_complete_and_round_trips_snapshot() -> None:
    memory = EpisodicMemory()
    episode = _episode("alpha")
    with memory.open_run() as session:
        frame = session
        assert frame.learn(LearnContext("empty", "episodic", ())) == []
        context = LearnContext(episode.run_id, "episodic", (episode,))
        first = frame.learn(context)
        second = frame.learn(context)
        assert [item.mutation_id for item in first] == [
            item.mutation_id for item in second
        ]
        assert [item.payload for item in first] == [item.payload for item in second]
        assert first[0].payload["operation"] == "append"
        session.commit("learn-alpha", tuple(first))
        session.commit("learn-alpha", tuple(first))
    with memory.open_run() as session:
        snapshot = session.snapshot
        rendered = session.recall()
    entry = json.loads(rendered)
    assert entry == snapshot["episodes"][0]
    assert entry["episode"]["interactions"] == [
        {"kind": "model_text", "text": "text alpha"},
        {"kind": "model_value", "value": {"value": "alpha"}},
        {"arguments": {"query": "alpha"}, "kind": "tool_call", "name": "lookup"},
        {"kind": "tool_result", "name": "lookup", "text": "result alpha"},
    ]
    restored = EpisodicMemory()
    with restored.open_run(snapshot=snapshot) as session:
        assert session.recall() == rendered

    failed_memory = EpisodicMemory()
    _learn(
        failed_memory,
        _episode("failure", status=MemoryEpisodeStatus.FAILED),
        "learn-failure",
    )
    with failed_memory.open_run() as session:
        failed = json.loads(session.recall())["episode"]
    assert failed["status"] == "failed"
    assert failed["failure"] == {"code": "failed", "message": "failure"}
    assert failed["output"] is None


def test_commit_is_atomic_rejects_malformed_and_duplicate_entries() -> None:
    memory = EpisodicMemory()
    episode = _episode("alpha")
    with memory.open_run() as session:
        frame = session
        valid = frame.learn(LearnContext(episode.run_id, "episodic", (episode,)))[0]
        malformed = MemoryMutation("wrong", valid.payload)
        with pytest.raises(MemoryOperationError) as exc:
            session.commit("bad", (valid, malformed))
        assert exc.value.code == "episodic.invalid_batch"
        assert session.snapshot["revision"] == 0
        session.commit("good", (valid,))
    with memory.open_run() as session:
        assert len(session.snapshot["episodes"]) == 1
        frame = session
        with pytest.raises(MemoryOperationError) as duplicate:
            session.commit(
                "duplicate",
                tuple(frame.learn(LearnContext(episode.run_id, "episodic", (episode,)))),
            )
        assert duplicate.value.code == "episodic.invalid_batch"


def test_handcrafted_out_of_scope_commit_is_rejected_atomically() -> None:
    memory = EpisodicMemory()
    episode = _episode("alpha")
    with memory.open_run() as session:
        frame = session
        valid = frame.learn(LearnContext(episode.run_id, "episodic", (episode,)))[0]
        payload = json.loads(json.dumps(valid.payload))
        # The episode's occurrence path no longer contains its scoped node.
        payload["entry"]["episode"]["owner_static_path"] = ["root", "outside"]
        forged = MemoryMutation(valid.mutation_id, payload)
        with pytest.raises(MemoryOperationError) as exc:
            session.commit("forged-scope", (forged,))
        assert exc.value.code == "episodic.invalid_batch"
        assert session.snapshot["revision"] == 0
        assert session.snapshot["episodes"] == []
        session.commit("valid-after-forgery", (valid,))
    with memory.open_run() as session:
        assert len(session.snapshot["episodes"]) == 1


def test_retention_recall_order_relevance_and_pinned_history() -> None:
    memory = EpisodicMemory(max_episodes=2, recall_limit=1)
    pinned = memory.open_run()
    pinned_frame = pinned
    assert pinned_frame.recall() == ""
    _learn(memory, _episode("alpha"), "one")
    _learn(memory, _episode("beta alpha"), "two")
    _learn(memory, _episode("gamma"), "three")
    assert pinned_frame.recall() == ""
    memory.reset()
    assert pinned_frame.recall() == ""
    pinned.close()

    restored = EpisodicMemory(max_episodes=2, recall_limit=1)
    _learn(restored, _episode("alpha"), "one")
    _learn(restored, _episode("beta alpha"), "two")
    _learn(restored, _episode("gamma"), "three")
    with restored.open_run() as session:
        frame = session
        history = [json.loads(line) for line in frame.recall().splitlines()]
        assert [item["episode"]["run_id"] for item in history] == [
            "run-beta alpha",
            "run-gamma",
        ]
        relevant = json.loads(
            frame.recall(RecallContext("", "alpha", {}, ("root", "agent")))
        )
        assert relevant["episode"]["run_id"] == "run-beta alpha"
        recent = json.loads(
            frame.recall(RecallContext("", "absent", {}, ("root", "agent")))
        )
        assert recent["episode"]["run_id"] == "run-gamma"


def test_rewrite_query_is_overridable_and_must_return_text() -> None:
    class FixedQueryMemory(EpisodicMemory):
        def rewrite_query(self, context: RecallContext) -> str:
            return "alpha"

    memory = FixedQueryMemory(recall_limit=1)
    _learn(memory, _episode("alpha"), "one")
    _learn(memory, _episode("newer"), "two")
    with memory.open_run() as session:
        recalled = session.recall(
            RecallContext("", "ignored", {}, ("root", "agent"))
        )
        assert json.loads(recalled)["episode"]["run_id"] == "run-alpha"

    class InvalidQueryMemory(EpisodicMemory):
        def rewrite_query(self, context: RecallContext) -> str:
            return 1  # type: ignore[return-value]

    with InvalidQueryMemory().open_run() as session:
        with pytest.raises(MemoryOperationError) as exc:
            session.recall(
                RecallContext("", "", {}, ("root", "agent"))
            )
        assert exc.value.code == "episodic.invalid_query"


def test_nonempty_snapshot_rejects_malformed_episode_and_duplicate_identity() -> None:
    memory = EpisodicMemory()
    _learn(memory, _episode("alpha"), "one")
    with memory.open_run() as session:
        snapshot = session.snapshot
    malformed = json.loads(json.dumps(snapshot))
    malformed["episodes"][0]["episode"]["status"] = "unknown"
    with pytest.raises(MemoryOperationError):
        EpisodicMemory().open_run(snapshot=malformed)
    duplicate = json.loads(json.dumps(snapshot))
    duplicate["episodes"].append(duplicate["episodes"][0])
    with pytest.raises(MemoryOperationError):
        EpisodicMemory().open_run(snapshot=duplicate)


def test_session_recalls_and_learns_under_the_memory_key() -> None:
    memory = EpisodicMemory(max_episodes=3, recall_limit=1)
    segment = MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("fan",), 1)
    episode = MemoryEpisode(
        "run",
        (segment,),
        ("fan", "agent"),
        MemoryEpisodeStatus.SUCCEEDED,
        {},
        "ok",
        memory_sections=(MemorySection("notes", "prior"),),
    )
    with memory.open_run() as session:
        assert not hasattr(session, "open_frame")
        assert session.recall() == ""
        mutations = tuple(session.learn(LearnContext("run", "notes", (episode,))))
        again = tuple(session.learn(LearnContext("run", "notes", (episode,))))
        other = tuple(session.learn(LearnContext("run", "other", (episode,))))
        assert [m.mutation_id for m in mutations] == [m.mutation_id for m in again]
        assert mutations[0].mutation_id != other[0].mutation_id
        stored = mutations[0].payload["entry"]["episode"]
        assert stored["occurrence_path"] == [
            {"index": 1, "kind": "map_item", "static_path": ["fan"]}
        ]
        assert stored["memory_sections"] == [{"memory": "notes", "text": "prior"}]
        assert "occurrence" not in stored
        session.commit("one", mutations)
    with memory.open_run() as session:
        assert len(session.recall().splitlines()) == 1
        query = RecallContext("base", "prompt", {}, ("fan", "agent"))
        assert session.recall(query) == session.recall()
