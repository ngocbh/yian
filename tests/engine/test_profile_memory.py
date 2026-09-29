"""ProfileMemory sessions, frames, learning, and compare-and-set commits."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
import pytest

import agent_composer.memory.profile as profile
from agent_composer.memory import (
    LearnContext,
    Memory,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrame,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceRef,
    MemoryOperationError,
    MemoryOwner,
    MemorySession,
    ModelTextInteraction,
    RecallContext,
    ToolCallInteraction,
    ToolResultInteraction,
)
from agent_composer.memory.profile import ProfileMemory
from agent_composer.memory.profile_store import FileProfileStore

ROOT = MemoryFrameRef("chat", ("chat",), MemoryOwner.FLOW)
CHILD = MemoryFrameRef("chat", ("chat", "reply"), MemoryOwner.AGENT)
OTHER = MemoryFrameRef("chat", ("chat", "other"), MemoryOwner.AGENT)
RECALL = RecallContext("system", "Say hi.", {}, ("chat", "reply"))
CONFIG = {"provider": "ollama", "model": "local-profiler"}


class _Learner:
    """Stub chat model that records requests and returns scripted replies."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return AIMessage(content=reply)


def _install(monkeypatch, *replies) -> tuple[_Learner, list]:
    learner = _Learner(*replies)
    configs = []
    monkeypatch.setattr(profile, "model_from_config", lambda config: configs.append(config) or learner)
    return learner, configs


def _memory(tmp_path, text: str = "# Ada\n", **kwargs) -> ProfileMemory:
    store = FileProfileStore(tmp_path / "ada.md")
    store.write(text, base_version=store.read()[1])
    return ProfileMemory(store, llm_config=CONFIG, **kwargs)


def _spec(ref: MemoryFrameRef, config=None) -> MemoryFrameSpec:
    return MemoryFrameSpec(ref, "profile", config or {})


def _feedback(*values) -> tuple:
    return tuple({"channel": "output", "target_name": None, "value": value,
                  "owner_static_path": None, "execution_seq": None} for value in values)


def _episode(prompt: str, output: str, interactions: tuple = ()) -> MemoryEpisode:
    return MemoryEpisode(
        "run-1", MemoryOccurrenceRef(ROOT), ("chat", "reply"), MemoryEpisodeStatus.SUCCEEDED,
        {"topic": prompt}, output, interactions=interactions, base_system_prompt="system", prompt=prompt,
    )


def _request_material(learner: _Learner, index: int = -1) -> dict:
    return json.loads(learner.requests[index][1].content)


def test_constructor_validates_store_config_and_budgets(tmp_path) -> None:
    store = FileProfileStore(tmp_path / "ada.md")
    with pytest.raises(TypeError):
        ProfileMemory(object())
    with pytest.raises(TypeError):
        ProfileMemory(store, llm_config="ollama")
    for budget in ({"max_chars": 0}, {"input_budget_chars": -1}, {"max_chars": 1.5}):
        with pytest.raises(ValueError):
            ProfileMemory(store, **budget)
    memory = ProfileMemory(store)
    assert (memory.max_chars, memory.input_budget_chars) == (32_000, 24_000)


def test_host_read_and_write_use_the_store_without_a_budget(tmp_path) -> None:
    memory = _memory(tmp_path, max_chars=10)
    text, version = memory.read()
    long_text = "# Ada\n" + "x" * 100 + "\n"
    new_version = memory.write(long_text, base_version=version, provenance={"by": "ada"})
    assert memory.read() == (long_text, new_version)
    assert memory.read(version) == (text, version)


def test_reset_is_refused(tmp_path) -> None:
    with pytest.raises(MemoryOperationError) as caught:
        _memory(tmp_path).reset()
    assert caught.value.code == "profile.reset_unsupported"


def test_snapshot_pins_version_and_hash_without_text(tmp_path) -> None:
    memory = _memory(tmp_path)
    text, version = memory.read()
    with memory.open_run() as session:
        assert session.snapshot == {
            "format": "agent_composer.profile_memory", "version": 1,
            "profile_version": version, "text_sha256": profile._sha256(text),
        }


def test_restore_accepts_the_recorded_snapshot_after_later_edits(tmp_path) -> None:
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        recorded = session.snapshot
    memory.write("# Ada\nLater.\n", base_version=memory.read()[1])
    with memory.open_run(snapshot=recorded) as restored:
        assert restored.snapshot == recorded


@pytest.mark.parametrize("change", [
    {"profile_version": "sha256:" + "0" * 64},
    {"text_sha256": "0" * 64},
    {"format": "other"},
    {"extra": 1},
])
def test_restore_rejects_unknown_or_altered_snapshots(tmp_path, change) -> None:
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        recorded = {**session.snapshot, **change}
    with pytest.raises(MemoryOperationError) as caught:
        memory.open_run(snapshot=recorded)
    assert caught.value.code == "profile.snapshot_unavailable"


def test_undecodable_document_is_a_permanent_open_failure(tmp_path) -> None:
    memory = _memory(tmp_path)
    (tmp_path / "ada.md").write_bytes(b"# Ada \xff\n")
    with pytest.raises(MemoryOperationError) as caught:
        memory.open_run()
    assert (caught.value.code, caught.value.retryable) == ("profile.store_invalid", False)
