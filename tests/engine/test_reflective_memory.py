"""Process-local reflective-memory behavior."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import pytest

from agent_composer.memory.contracts import (
    LearnContext,
    Memory,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceRef,
    MemoryOperationError,
    MemoryOwner,
    RecallContext,
)
from agent_composer.memory.reflective import ReflectiveMemory
import agent_composer.memory.reflective as reflective


def _ref(name: str = "agent") -> MemoryFrameRef:
    return MemoryFrameRef("root", ("root", name), MemoryOwner.AGENT)


def _spec(
    name: str = "agent",
    config: dict[str, object] | None = None,
) -> MemoryFrameSpec:
    return MemoryFrameSpec(_ref(name), "reflective", config or {})


def _episode(
    ref: MemoryFrameRef | None = None,
    *,
    run_id: str = "run-1",
    text: str = "hello",
) -> MemoryEpisode:
    frame_ref = ref or _ref()
    return MemoryEpisode(
        run_id=run_id,
        occurrence=MemoryOccurrenceRef(frame_ref),
        owner_static_path=frame_ref.static_path,
        status=MemoryEpisodeStatus.SUCCEEDED,
        inputs={"text": text},
        output={"answer": text.upper()},
    )


def _context(
    ref: MemoryFrameRef | None = None,
    *,
    run_id: str = "run-1",
    text: str = "hello",
) -> LearnContext:
    frame_ref = ref or _ref()
    return LearnContext(
        run_id,
        frame_ref,
        (_episode(frame_ref, run_id=run_id, text=text),),
    )


class _StructuredModel:
    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.messages: list[object] | None = None

    def invoke(self, messages: list[object]) -> object:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.result


class _Model:
    def __init__(self, structured: _StructuredModel) -> None:
        self.structured = structured
        self.schema: type[object] | None = None

    def with_structured_output(self, schema: type[object]) -> _StructuredModel:
        self.schema = schema
        return self.structured


def _install_model(
    monkeypatch: pytest.MonkeyPatch,
    result: object = None,
    error: Exception | None = None,
) -> tuple[_Model, list[dict[str, object]]]:
    model = _Model(_StructuredModel(result, error))
    configs: list[dict[str, object]] = []

    def factory(config: dict[str, object]) -> _Model:
        configs.append(dict(config))
        return model

    monkeypatch.setattr(reflective, "model_from_config", factory)
    return model, configs


def _learn_actions(
    monkeypatch: pytest.MonkeyPatch,
    memory: ReflectiveMemory,
    actions: list[dict[str, object]],
    *,
    spec: MemoryFrameSpec | None = None,
    context: LearnContext | None = None,
) -> list[MemoryMutation]:
    _install_model(monkeypatch, {"actions": actions})
    with memory.open_run() as session:
        frame = session.open_frame(spec or _spec())
        return frame.learn(context or _context())


def test_constructor_and_frame_configuration_are_exact() -> None:
    with pytest.raises(TypeError, match="max_items"):
        ReflectiveMemory(max_items=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="recall_limit"):
        ReflectiveMemory(recall_limit=0)
    with pytest.raises(ValueError):
        ReflectiveMemory({"provider": "not-a-provider"})

    memory = ReflectiveMemory(max_items=7, recall_limit=3)
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        assert session.open_frame(_spec()) is frame
        with pytest.raises(MemoryOperationError, match="different"):
            session.open_frame(_spec(config={"max_items": 7}))
        with pytest.raises(MemoryOperationError, match="policy") as exc:
            session.open_frame(MemoryFrameSpec(_ref(), "other", {}))
        assert exc.value.retryable is False
        with pytest.raises(MemoryOperationError, match="unknown"):
            session.open_frame(_spec(config={"unknown": 1}))
        with pytest.raises(MemoryOperationError, match="max_items"):
            session.open_frame(_spec(config={"max_items": True}))


def test_defaults_are_read_once_and_recorded_snapshot_never_rereads_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        reflective,
        "default_llm_provider",
        lambda: calls.append("provider") or "ollama",
    )
    monkeypatch.setattr(
        reflective,
        "default_llm_model",
        lambda: calls.append("model") or "qwen",
    )
    first = ReflectiveMemory({"temperature": 0.2})
    with first.open_run() as session:
        snapshot = session.snapshot
    assert calls == ["provider", "model"]
    assert snapshot["llm_config"] == {
        "anthropic_effort": None,
        "google_thinking_level": None,
        "model": "qwen",
        "openai_reasoning_effort": None,
        "provider": "ollama",
        "temperature": 0.2,
    }

    calls.clear()
    restored = ReflectiveMemory()
    with restored.open_run(snapshot=snapshot) as resumed:
        assert resumed.snapshot == snapshot
    assert calls == []


def test_snapshot_is_defensive_and_foreign_restore_is_conservative() -> None:
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    with memory.open_run() as session:
        session.open_frame(_spec(config={"max_items": 4}))
        snapshot = session.snapshot
        snapshot["namespace"] = "changed"
        recorded = session.snapshot
        assert recorded["namespace"] != "changed"

    foreign = ReflectiveMemory()
    with foreign.open_run() as opened:
        opened.snapshot
    with pytest.raises(MemoryOperationError, match="namespace"):
        foreign.open_run(snapshot=recorded)

    reset = ReflectiveMemory()
    reset.reset()
    with pytest.raises(MemoryOperationError, match="namespace"):
        reset.open_run(snapshot=recorded)


def test_pristine_restore_keeps_recorded_effective_frame_defaults() -> None:
    original = ReflectiveMemory(
        {"provider": "ollama", "model": "qwen"},
        max_items=4,
        recall_limit=2,
    )
    with original.open_run() as session:
        session.open_frame(_spec())
        snapshot = session.snapshot

    restored = ReflectiveMemory(max_items=99, recall_limit=88)
    with restored.open_run(snapshot=snapshot) as session:
        frame = session.open_frame(_spec())
        assert frame.recall() == ""
        assert session.snapshot == snapshot


def test_snapshot_rejects_noncanonical_scalars_and_circular_json() -> None:
    circular: dict[str, object] = {}
    circular["value"] = circular
    with pytest.raises(ValueError, match="circular"):
        ReflectiveMemory(circular)  # type: ignore[arg-type]

    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    with memory.open_run() as session:
        snapshot = session.snapshot
    snapshot["version"] = True
    with pytest.raises(MemoryOperationError, match="version"):
        ReflectiveMemory().open_run(snapshot=snapshot)


def test_frame_idempotency_and_closed_session_operations() -> None:
    memory = ReflectiveMemory()
    session = memory.open_run()
    frame = session.open_frame(_spec(config={"recall_limit": 2}))
    assert session.open_frame(_spec(config={"recall_limit": 2})) is frame
    with pytest.raises(MemoryOperationError, match="different"):
        session.open_frame(_spec(config={"recall_limit": 3}))
    session.close()
    session.close()
    with pytest.raises(MemoryOperationError, match="closed"):
        session.open_frame(_spec())
    with pytest.raises(MemoryOperationError, match="closed"):
        frame.recall()
    with pytest.raises(MemoryOperationError, match="closed"):
        frame.learn(_context())
    with pytest.raises(MemoryOperationError, match="closed"):
        session.commit("commit", ())


def test_recall_all_and_context_ranking_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = ReflectiveMemory(
        {"provider": "ollama", "model": "qwen"}, recall_limit=2
    )
    mutations = _learn_actions(
        monkeypatch,
        memory,
        [
            {"action": "add", "lesson_id": "b", "text": "red apple"},
            {"action": "add", "lesson_id": "a", "text": "blue berry"},
            {"action": "add", "lesson_id": "c", "text": "red berry"},
        ],
    )
    with memory.open_run() as session:
        session.open_frame(_spec())
        session.commit("seed", tuple(mutations))

    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        assert frame.recall() == "[b] red apple\n[a] blue berry\n[c] red berry"
        recalled = frame.recall(
            RecallContext("", "RED berry", {"tone": "BLUE"}, ("root", "user"))
        )
        assert recalled == "[a] blue berry\n[c] red berry"


def test_subclass_controls_query_rewriting(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Memory(ReflectiveMemory):
        def rewrite_query(self, context: RecallContext) -> str:
            return "apple"

    memory = _Memory({"provider": "ollama", "model": "qwen"}, recall_limit=1)
    mutations = _learn_actions(
        monkeypatch,
        memory,
        [
            {"action": "add", "lesson_id": "berry", "text": "fresh berry"},
            {"action": "add", "lesson_id": "apple", "text": "green apple"},
        ],
    )
    with memory.open_run() as session:
        session.open_frame(_spec())
        session.commit("seed", tuple(mutations))
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        assert frame.recall(RecallContext("", "berry", {}, ("root",))) == (
            "[apple] green apple"
        )


def test_learn_uses_pinned_config_and_returns_deterministic_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, configs = _install_model(
        monkeypatch,
        {
            "actions": [
                {"action": "add", "lesson_id": "one", "text": "Keep it short"},
                {"action": "noop", "lesson_id": None, "text": None},
            ]
        },
    )
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        pinned_config = session.snapshot["llm_config"]
        with pytest.raises(MemoryOperationError, match="no-op"):
            frame.learn(_context())
    assert configs == [pinned_config]
    assert model.schema is not None

    model.structured.result = {
        "actions": [
            {"action": "add", "lesson_id": "one", "text": "Keep it short"}
        ]
    }
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        first = frame.learn(_context())
        second = frame.learn(_context())
        assert first == second
        assert first[0].payload == {
            "action": "add",
            "lesson_id": "one",
            "source": {
                "frame_ref": {
                    "owner": "agent",
                    "root_flow_id": "root",
                    "static_path": ["root", "agent"],
                },
                "run_id": "run-1",
            },
            "text": "Keep it short",
        }


@pytest.mark.parametrize(
    "actions",
    [
        [{"action": "add", "lesson_id": "", "text": "x"}],
        [{"action": "add", "lesson_id": "one", "text": "   "}],
        [{"action": "remove", "lesson_id": "missing", "text": None}],
        [
            {"action": "add", "lesson_id": "same", "text": "x"},
            {"action": "add", "lesson_id": "same", "text": "y"},
        ],
    ],
)
def test_learn_rejects_malformed_or_conflicting_actions(
    monkeypatch: pytest.MonkeyPatch,
    actions: list[dict[str, object]],
) -> None:
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    with pytest.raises(MemoryOperationError) as exc:
        _learn_actions(monkeypatch, memory, actions)
    assert exc.value.stage == "learn"
    assert exc.value.retryable is False


def test_learn_provider_failure_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    _install_model(monkeypatch, error=RuntimeError("offline"))
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        with pytest.raises(MemoryOperationError) as exc:
            frame.learn(_context())
    assert exc.value.code == "reflective.provider"
    assert exc.value.retryable is True


def test_episode_material_is_capped_in_one_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, configs = _install_model(
        monkeypatch,
        {"actions": [{"action": "noop", "lesson_id": None, "text": None}]},
    )
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    huge = "x" * (reflective._MAX_EPISODE_MATERIAL_CHARS * 2)
    with memory.open_run() as session:
        frame = session.open_frame(_spec())
        result = frame.learn(_context(text=huge))
    assert result == []
    assert configs
    assert model.structured.messages is not None
    material = model.structured.messages[-1].content
    assert isinstance(material, str)
    assert len(material) <= reflective._MAX_REFLECTION_PROMPT_CHARS
    assert "truncated" in material


def test_learning_is_propose_only_and_rejects_a_stale_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    _install_model(
        monkeypatch,
        {"actions": [{"action": "add", "lesson_id": "one", "text": "first"}]},
    )
    session = memory.open_run()
    frame = session.open_frame(_spec())
    proposals = frame.learn(_context())
    assert frame.recall() == ""

    with memory.open_run() as winner:
        winner.open_frame(_spec())
        winner.commit("winner", tuple(proposals))
    with pytest.raises(MemoryOperationError, match="stale"):
        frame.learn(_context())
    session.close()


def test_commit_is_atomic_idempotent_and_empty_batches_advance_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    add = _learn_actions(
        monkeypatch,
        memory,
        [{"action": "add", "lesson_id": "one", "text": "first"}],
    )
    with memory.open_run() as session:
        session.open_frame(_spec())
        session.commit("add", tuple(add))
        session.commit("add", tuple(add))
        with pytest.raises(MemoryOperationError, match="collision"):
            session.commit("add", ())

    with memory.open_run() as session:
        session.open_frame(_spec())
        before = session.snapshot["revision"]
        session.commit("empty", ())
    with memory.open_run() as later:
        later.open_frame(_spec())
        assert later.snapshot["revision"] == before + 1


def test_concurrent_same_generation_commits_have_one_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    first = _learn_actions(
        monkeypatch,
        memory,
        [{"action": "add", "lesson_id": "one", "text": "first"}],
    )
    second = _learn_actions(
        monkeypatch,
        memory,
        [{"action": "add", "lesson_id": "two", "text": "second"}],
    )
    left = memory.open_run()
    right = memory.open_run()
    left.open_frame(_spec())
    right.open_frame(_spec())

    def commit(session: Any, commit_id: str, batch: list[MemoryMutation]) -> str:
        try:
            session.commit(commit_id, tuple(batch))
        except MemoryOperationError as exc:
            return exc.code
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = set(
            pool.map(
                lambda args: commit(*args),
                ((left, "left", first), (right, "right", second)),
            )
        )
    assert results == {"ok", "reflective.stale_generation"}
    left.close()
    right.close()


def test_reset_preserves_commit_ledger_ordering_and_old_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    add = _learn_actions(
        monkeypatch,
        memory,
        [{"action": "add", "lesson_id": "one", "text": "first"}],
    )
    old = memory.open_run()
    old_frame = old.open_frame(_spec())
    old.commit("applied", tuple(add))
    memory.reset()

    assert old_frame.recall() == ""
    old.commit("applied", tuple(add))
    with pytest.raises(MemoryOperationError, match="collision"):
        old.commit("applied", ())
    with pytest.raises(MemoryOperationError, match="stale"):
        old.commit("not-applied", tuple(add))
    with memory.open_run() as fresh:
        assert fresh.open_frame(_spec()).recall() == ""
    old.close()


def test_reflective_memory_implements_protocol_and_imports_only_lower_layers() -> None:
    assert isinstance(ReflectiveMemory(), Memory)
    path = Path(reflective.__file__ or "")
    tree = ast.parse(path.read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    forbidden = ("nodes", "compile", "compose", "runtime", "run", "cli")
    assert not any(
        any(part in imported.split(".") for part in forbidden)
        for imported in imports
    )
    json.dumps(ReflectiveMemory().open_run().snapshot, sort_keys=True)
