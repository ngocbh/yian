"""Session and snapshot foundations for process-local episodic memory."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from agent_composer.memory.contracts import (
    LearnContext,
    Memory,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrame,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceRef,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOperationError,
    MemoryOwner,
    MemorySection,
    MemorySession,
    ModelTextInteraction,
    ModelValueInteraction,
    RecallContext,
    ToolCallInteraction,
    ToolResultInteraction,
)
from agent_composer.memory.episodic import EpisodicMemory
import agent_composer.memory.episodic as episodic


def _ref(name: str = "agent") -> MemoryFrameRef:
    return MemoryFrameRef("root", ("root", name), MemoryOwner.AGENT)


def _spec(
    name: str = "agent", config: dict[str, object] | None = None
) -> MemoryFrameSpec:
    return MemoryFrameSpec(_ref(name), "episodic", config or {})


def _context(ref: MemoryFrameRef | None = None) -> LearnContext:
    ref = ref or _ref()
    episode = MemoryEpisode(
        "run",
        MemoryOccurrenceRef(ref),
        ref.static_path,
        MemoryEpisodeStatus.SUCCEEDED,
        {},
        "ok",
    )
    return LearnContext("run", ref, (episode,))


def test_constructor_protocol_and_frame_configuration_are_exact() -> None:
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
        frame = session.open_frame(_spec())
        assert isinstance(frame, MemoryFrame)
        assert session.open_frame(_spec()) is frame
        cases = [
            (_spec(config={"max_episodes": 7}), "episodic.frame_conflict"),
            (MemoryFrameSpec(_ref(), "other", {}), "episodic.invalid_policy"),
            (_spec(config={"unknown": 1}), "episodic.invalid_frame_config"),
            (_spec(config={"recall_limit": True}), "episodic.invalid_frame_config"),
        ]
        for spec, code in cases:
            with pytest.raises(MemoryOperationError) as exc:
                session.open_frame(spec)
            assert exc.value.code == code
        with pytest.raises(MemoryOperationError) as exc:
            session.open_frame(object())  # type: ignore[arg-type]
        assert exc.value.code == "episodic.invalid_frame"


def test_frame_operations_validate_context_ownership_and_generation() -> None:
    memory = EpisodicMemory()
    session = memory.open_run()
    frame = session.open_frame(_spec())
    assert frame.recall() == ""
    assert frame.recall(RecallContext("agent", "prompt", {}, ("root",))) == ""
    assert len(frame.learn(_context())) == 1
    for operation in (
        lambda: frame.recall(object()),
        lambda: frame.learn(object()),
        lambda: frame.learn(_context(_ref("other"))),
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


def test_snapshot_is_closed_canonical_and_defensive() -> None:
    memory = EpisodicMemory(max_episodes=4, recall_limit=2)
    with memory.open_run() as session:
        session.open_frame(_spec())
        snapshot = session.snapshot
        assert set(snapshot) == {
            "epoch",
            "format",
            "frames",
            "namespace",
            "revision",
            "version",
        }
        assert snapshot["format"] == "agent_composer.episodic_memory"
        assert snapshot["version"] == 1
        assert snapshot["frames"] == [
            {
                "config": {"max_episodes": 4, "recall_limit": 2},
                "episodes": [],
                "policy_name": "episodic",
                "ref": {
                    "owner": "agent",
                    "root_flow_id": "root",
                    "static_path": ["root", "agent"],
                },
            }
        ]
        snapshot["frames"][0]["config"]["max_episodes"] = 99
        assert session.snapshot["frames"][0]["config"]["max_episodes"] == 4
        json.dumps(session.snapshot, sort_keys=True)


def test_snapshot_restore_rules_and_recorded_defaults() -> None:
    original = EpisodicMemory(max_episodes=4, recall_limit=2)
    with original.open_run() as session:
        session.open_frame(_spec())
        session.commit("declare", ())
    with original.open_run() as session:
        recorded = session.snapshot
    with original.open_run(snapshot=recorded) as current:
        assert current.snapshot == recorded
    restored = EpisodicMemory(max_episodes=99, recall_limit=88)
    with restored.open_run(snapshot=recorded) as session:
        session.open_frame(_spec())
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
    changed["frames"][0]["config"]["recall_limit"] = 3
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
        lambda x: x.update(frames={}),
        lambda x: x["frames"][0].update(policy_name="other"),
        lambda x: x["frames"][0]["config"].update(max_episodes=0),
        lambda x: x["frames"][0].update(episodes={}),
        lambda x: x["frames"][0]["episodes"].append({}),
    ],
)
def test_snapshot_format_is_strict(mutate: Any) -> None:
    with EpisodicMemory().open_run() as session:
        session.open_frame(_spec())
        snapshot = session.snapshot
    malformed = json.loads(json.dumps(snapshot))
    mutate(malformed)
    with pytest.raises(MemoryOperationError) as exc:
        EpisodicMemory().open_run(snapshot=malformed)
    assert exc.value.code == "episodic.invalid_snapshot"
    assert exc.value.retryable is False


def test_snapshot_frames_are_uniquely_sorted() -> None:
    with EpisodicMemory().open_run() as session:
        session.open_frame(_spec("z"))
        session.open_frame(_spec("a"))
        snapshot = session.snapshot
    snapshot["frames"].reverse()
    with pytest.raises(MemoryOperationError):
        EpisodicMemory().open_run(snapshot=snapshot)


def test_empty_commit_idempotency_collision_and_staleness() -> None:
    memory = EpisodicMemory()
    session = memory.open_run()
    session.open_frame(_spec())
    session.commit("empty", ())
    session.commit("empty", ())
    with memory.open_run() as fresh:
        assert fresh.snapshot["revision"] == 1
    with pytest.raises(MemoryOperationError) as collision:
        session.commit("empty", (MemoryMutation("m", _ref(), {}),))
    assert collision.value.code == "episodic.commit_collision"
    with pytest.raises(MemoryOperationError) as nonempty:
        memory.open_run().commit("nonempty", (MemoryMutation("m", _ref(), {}),))
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
    frame = session.open_frame(_spec())
    session.close()
    session.close()
    operations = (
        lambda: session.snapshot,
        lambda: session.__enter__(),
        lambda: session.open_frame(_spec()),
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
        "agent_composer.memory.contracts",
        "agent_composer.memory.storage",
    }


def _episode(
    label: str,
    *,
    ref: MemoryFrameRef | None = None,
    status: MemoryEpisodeStatus = MemoryEpisodeStatus.SUCCEEDED,
) -> MemoryEpisode:
    ref = ref or _ref()
    occurrence = MemoryOccurrenceRef(
        ref,
        (MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ("root",), 2),),
    )
    return MemoryEpisode(
        f"run-{label}",
        occurrence,
        ref.static_path,
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
        (MemorySection(ref, f"memory {label}"),),
        execution_seq=7,
    )


def _learn(memory: EpisodicMemory, episode: MemoryEpisode, commit_id: str) -> None:
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        mutations = tuple(frame.learn(LearnContext(episode.run_id, _ref(), (episode,))))
        session.commit(commit_id, mutations)


def test_learning_is_deterministic_complete_and_round_trips_snapshot() -> None:
    memory = EpisodicMemory()
    episode = _episode("alpha")
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        assert frame.learn(LearnContext("empty", _ref(), ())) == []
        context = LearnContext(episode.run_id, _ref(), (episode,))
        first = frame.learn(context)
        second = frame.learn(context)
        assert [item.mutation_id for item in first] == [
            item.mutation_id for item in second
        ]
        assert [item.payload for item in first] == [item.payload for item in second]
        assert first[0].payload["operation"] == "append"
        outside = MemoryEpisode(
            "outside",
            MemoryOccurrenceRef(_ref()),
            ("root", "child"),
            MemoryEpisodeStatus.SUCCEEDED,
            {},
            "value",
        )
        with pytest.raises(MemoryOperationError) as owner_error:
            frame.learn(LearnContext("outside", _ref(), (outside,)))
        assert owner_error.value.code == "episodic.invalid_context"
        session.commit("learn-alpha", tuple(first))
        session.commit("learn-alpha", tuple(first))
    with memory.open_run() as session:
        snapshot = session.snapshot
        rendered = session.open_frame(_spec()).recall()
    entry = json.loads(rendered)
    assert entry == snapshot["frames"][0]["episodes"][0]
    assert entry["episode"]["interactions"] == [
        {"kind": "model_text", "text": "text alpha"},
        {"kind": "model_value", "value": {"value": "alpha"}},
        {"arguments": {"query": "alpha"}, "kind": "tool_call", "name": "lookup"},
        {"kind": "tool_result", "name": "lookup", "text": "result alpha"},
    ]
    restored = EpisodicMemory()
    with restored.open_run(snapshot=snapshot) as session:
        assert session.open_frame(_spec()).recall() == rendered

    failed_memory = EpisodicMemory()
    _learn(
        failed_memory,
        _episode("failure", status=MemoryEpisodeStatus.FAILED),
        "learn-failure",
    )
    with failed_memory.open_run() as session:
        failed = json.loads(session.open_frame(_spec()).recall())["episode"]
    assert failed["status"] == "failed"
    assert failed["failure"] == {"code": "failed", "message": "failure"}
    assert failed["output"] is None


def test_commit_is_atomic_validates_ownership_and_rejects_duplicate_entries() -> None:
    memory = EpisodicMemory()
    episode = _episode("alpha")
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        valid = frame.learn(LearnContext(episode.run_id, _ref(), (episode,)))[0]
        malformed = MemoryMutation("wrong", _ref(), valid.payload)
        with pytest.raises(MemoryOperationError) as exc:
            session.commit("bad", (valid, malformed))
        assert exc.value.code == "episodic.invalid_batch"
        assert session.snapshot["revision"] == 0
        session.commit("good", (valid,))
    with memory.open_run() as session:
        assert len(session.snapshot["frames"][0]["episodes"]) == 1
        frame = session.open_frame(_spec())
        with pytest.raises(MemoryOperationError) as duplicate:
            session.commit(
                "duplicate",
                tuple(frame.learn(LearnContext(episode.run_id, _ref(), (episode,)))),
            )
        assert duplicate.value.code == "episodic.invalid_batch"
    with memory.open_run() as session:
        unopened = MemoryMutation(valid.mutation_id, _ref(), valid.payload)
        with pytest.raises(MemoryOperationError):
            session.commit("unopened", (unopened,))


def test_handcrafted_out_of_owner_commit_is_rejected_atomically() -> None:
    memory = EpisodicMemory()
    episode = _episode("alpha")
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        valid = frame.learn(LearnContext(episode.run_id, _ref(), (episode,)))[0]
        payload = json.loads(json.dumps(valid.payload))
        payload["entry"]["episode"]["owner_static_path"] = ["root", "outside"]
        forged = MemoryMutation(valid.mutation_id, _ref(), payload)
        with pytest.raises(MemoryOperationError) as exc:
            session.commit("forged-owner", (forged,))
        assert exc.value.code == "episodic.invalid_batch"
        assert session.snapshot["revision"] == 0
        assert session.snapshot["frames"][0]["episodes"] == []
        session.commit("valid-after-forgery", (valid,))
    with memory.open_run() as session:
        assert len(session.snapshot["frames"][0]["episodes"]) == 1


def test_retention_recall_order_relevance_and_pinned_history() -> None:
    memory = EpisodicMemory(max_episodes=2, recall_limit=1)
    pinned = memory.open_run()
    pinned_frame = pinned.open_frame(_spec())
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
        frame = session.open_frame(_spec())
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
        recalled = session.open_frame(_spec()).recall(
            RecallContext("", "ignored", {}, ("root", "agent"))
        )
        assert json.loads(recalled)["episode"]["run_id"] == "run-alpha"

    class InvalidQueryMemory(EpisodicMemory):
        def rewrite_query(self, context: RecallContext) -> str:
            return 1  # type: ignore[return-value]

    with InvalidQueryMemory().open_run() as session:
        with pytest.raises(MemoryOperationError) as exc:
            session.open_frame(_spec()).recall(
                RecallContext("", "", {}, ("root", "agent"))
            )
        assert exc.value.code == "episodic.invalid_query"


def test_nonempty_snapshot_rejects_malformed_episode_and_duplicate_identity() -> None:
    memory = EpisodicMemory()
    _learn(memory, _episode("alpha"), "one")
    with memory.open_run() as session:
        snapshot = session.snapshot
    malformed = json.loads(json.dumps(snapshot))
    malformed["frames"][0]["episodes"][0]["episode"]["status"] = "unknown"
    with pytest.raises(MemoryOperationError):
        EpisodicMemory().open_run(snapshot=malformed)
    duplicate = json.loads(json.dumps(snapshot))
    duplicate["frames"][0]["episodes"].append(duplicate["frames"][0]["episodes"][0])
    with pytest.raises(MemoryOperationError):
        EpisodicMemory().open_run(snapshot=duplicate)


def test_nonempty_snapshot_rejects_episode_from_another_frame() -> None:
    memory = EpisodicMemory()
    _learn(memory, _episode("alpha"), "one")
    with memory.open_run() as session:
        snapshot = session.snapshot
    cross_frame = json.loads(json.dumps(snapshot))
    episode_ref = cross_frame["frames"][0]["episodes"][0]["episode"]["occurrence"][
        "frame_ref"
    ]
    episode_ref["static_path"] = ["root", "other"]
    with pytest.raises(MemoryOperationError) as exc:
        EpisodicMemory().open_run(snapshot=cross_frame)
    assert exc.value.code == "episodic.invalid_snapshot"
