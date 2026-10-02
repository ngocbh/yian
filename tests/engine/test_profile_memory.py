"""ProfileMemory sessions, learning, and compare-and-set commits."""

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
    MemoryOperationError,
    MemorySession,
    ModelTextInteraction,
    RecallContext,
    ToolCallInteraction,
    ToolResultInteraction,
)
from agent_composer.memory.profile import ProfileMemory
from agent_composer.memory.storage import FileMarkdownStore, markdown_version

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
    store = FileMarkdownStore(tmp_path / "ada.md")
    store.write(text, base_version=store.read()[1])
    return ProfileMemory(store, llm_config=CONFIG, **kwargs)


def _feedback(*values) -> tuple:
    return tuple({"channel": "output", "target_name": None, "value": value,
                  "owner_static_path": None, "execution_seq": None} for value in values)


def _episode(prompt: str, output: str, interactions: tuple = ()) -> MemoryEpisode:
    return MemoryEpisode(
        "run-1", (), ("chat", "reply"), MemoryEpisodeStatus.SUCCEEDED,
        {"topic": prompt}, output, interactions=interactions, base_system_prompt="system", prompt=prompt,
    )


def _request_material(learner: _Learner, index: int = -1) -> dict:
    return json.loads(learner.requests[index][1].content)


def test_constructor_validates_store_config_and_budgets(tmp_path) -> None:
    store = FileMarkdownStore(tmp_path / "ada.md")
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


def test_closed_backend_rejects_later_use_without_touching_the_store(tmp_path) -> None:
    memory = ProfileMemory(FileMarkdownStore(tmp_path / "ada.md"))
    memory.close()
    memory.close()
    calls = (
        memory.open_run,
        memory.reset,
        memory.read,
        lambda: memory.write("x", base_version=markdown_version("")),
    )
    for call in calls:
        with pytest.raises(MemoryOperationError) as caught:
            call()
        assert caught.value.code == "profile.closed"
        assert not caught.value.retryable
    assert list(tmp_path.iterdir()) == []


def test_closing_the_backend_invalidates_its_open_sessions(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, "<profile># Ada\nBrief.</profile>")
    memory = _memory(tmp_path)
    session = memory.open_run()
    batch = tuple(session.learn(LearnContext("run-1", "profile", (), _feedback("Be brief."))))
    memory.close()
    calls = (
        lambda: session.recall(RECALL),
        lambda: session.learn(LearnContext("run-1", "profile", (), _feedback("Be brief."))),
        lambda: session.commit("commit-1", batch),
    )
    for call in calls:
        with pytest.raises(MemoryOperationError) as caught:
            call()
        assert caught.value.code == "profile.closed"
    assert (tmp_path / "ada.md").read_text() == "# Ada\n"


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


def test_recall_stays_pinned_after_a_later_write(tmp_path) -> None:
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        memory.write("# Ada\nChanged.\n", base_version=memory.read()[1])
        assert root.recall(RECALL) == "# Ada\n"


def test_restore_recalls_the_recorded_version_after_later_edits(tmp_path) -> None:
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        recorded = session.snapshot
    memory.write("# Ada\nLater.\n", base_version=memory.read()[1])
    with memory.open_run(snapshot=recorded) as restored:
        assert restored.recall(RECALL) == "# Ada\n"


@pytest.mark.parametrize(("reply", "code"), [
    (AIMessage(content="# Ada without tags"), "profile.contract"),
    (AIMessage(content="<profile>   </profile>"), "profile.contract"),
    (AIMessage(content="<profile>" + "y" * 101 + "</profile>"), "profile.over_budget"),
    (object(), "profile.contract"),
])
def test_parse_reply_enforces_tags_content_and_budget(reply, code) -> None:
    with pytest.raises(MemoryOperationError) as caught:
        profile._parse_reply(reply, "# Ada\n", 100)
    assert (caught.value.stage, caught.value.code, caught.value.retryable) == ("learn", code, False)


def test_parse_reply_joins_text_blocks_and_ends_with_one_newline() -> None:
    reply = AIMessage(content=["Here:\n<profile>\n# Ada\n", {"type": "text", "text": "- Brief.\n</profile>"}])
    assert profile._parse_reply(reply, "# Ada\n", 100) == "# Ada\n- Brief.\n"
    assert profile._parse_reply(AIMessage(content="<profile></profile>"), "", 100) == ""


def test_learn_without_feedback_calls_no_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(profile, "model_from_config", lambda config: pytest.fail("model constructed"))
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        assert root.learn(LearnContext("run-1", "profile", (_episode("hi", "hello"),))) == []


def test_learn_rewrites_the_current_document_from_feedback_and_episodes(tmp_path, monkeypatch) -> None:
    learner, configs = _install(monkeypatch, "<profile>\n# Ada\n- Prefers brief replies.\n</profile>")
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        seen_version = memory.read()[1]
        memory.write("# Ada\nHuman note.\n", base_version=seen_version)
        current_version = memory.read()[1]
        call = ToolCallInteraction("calendar", {"day": "today"})
        episodes = (_episode("first", "long answer", (call,)), _episode("second", "longer answer"))
        context = LearnContext("run-1", "profile", episodes, _feedback("Be brief.", "Be brief."))
        mutation, = root.learn(context)
    assert configs == [CONFIG]
    material = _request_material(learner)
    assert material["current_profile"] == "# Ada\nHuman note.\n"
    assert material["seen_profile"] == "# Ada\n"
    assert [entry["value"] for entry in material["feedback"]] == ["Be brief.", "Be brief."]
    assert [episode["prompt"] for episode in material["episodes"]] == ["first", "second"]
    assert material["episodes"][0]["interactions"] == [
        {"kind": "tool_call", "name": "calendar", "arguments": {"day": "today"}}]
    assert material["omitted_episodes"] == 0
    assert mutation.payload == {
        "base_version": current_version, "text": "# Ada\n- Prefers brief replies.\n",
        "seen_version": seen_version, "run_id": "run-1", "feedback": ["Be brief."],
    }


def test_learner_sees_the_tool_calls_of_each_episode(tmp_path, monkeypatch) -> None:
    learner, _ = _install(monkeypatch, "<profile># Ada\n- Check the calendar first.</profile>")
    interactions = (
        ModelTextInteraction("Let me look."),
        ToolCallInteraction("calendar", {"day": "friday"}),
        ToolResultInteraction("calendar", "busy 9-17"),
    )
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        root.learn(LearnContext("run-1", "profile", (_episode("free friday?", "Yes.", interactions),),
                                _feedback("Check the calendar before answering.")))
    episode, = _request_material(learner)["episodes"]
    assert episode["interactions"] == [
        {"kind": "model_text", "text": "Let me look."},
        {"kind": "tool_call", "name": "calendar", "arguments": {"day": "friday"}},
        {"kind": "tool_result", "name": "calendar", "text": "busy 9-17"},
    ]


def test_learn_keeps_only_the_newest_episodes_within_the_input_budget(tmp_path, monkeypatch) -> None:
    learner, _ = _install(monkeypatch, "<profile># Ada\nNew.</profile>")
    memory = _memory(tmp_path, input_budget_chars=700)
    episodes = tuple(_episode(f"prompt {index}", "o" * 100) for index in range(5))
    with memory.open_run() as session:
        session.learn(LearnContext("run-1", "profile", episodes, _feedback("x")))
    material = _request_material(learner)
    assert [episode["prompt"] for episode in material["episodes"]] == ["prompt 3", "prompt 4"]
    assert material["omitted_episodes"] == 3


def test_an_oversized_newest_episode_is_trimmed_not_dropped(tmp_path, monkeypatch) -> None:
    learner, _ = _install(monkeypatch, "<profile># Ada\nRead less.</profile>")
    memory = _memory(tmp_path)
    big = (ToolCallInteraction("read_file", {"path": "notes.md"}), ToolResultInteraction("read_file", "x" * 30_000))
    episodes = (_episode("older", "short"), _episode("newest", "done", big))
    with memory.open_run() as session:
        session.learn(LearnContext("run-1", "profile", episodes, _feedback("x")))
    material = _request_material(learner)
    newest, = material["episodes"]
    assert material["omitted_episodes"] == 1
    assert newest["truncated"] is True and newest["original_chars"] > memory.input_budget_chars
    assert '"name":"read_file"' in newest["prefix"] and '"prompt":"newest"' in newest["suffix"]
    assert len(profile._canonical_json(newest)) <= memory.input_budget_chars


def test_provider_failure_is_retryable_and_bad_replies_are_not(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, RuntimeError("provider down"), "no tags")
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        for code, retryable in (("profile.provider", True), ("profile.contract", False)):
            with pytest.raises(MemoryOperationError) as caught:
                root.learn(LearnContext("run-1", "profile", (), _feedback("x")))
            assert (caught.value.code, caught.value.retryable) == (code, retryable)


def test_unchanged_document_proposes_nothing(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, "<profile># Ada</profile>")
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        assert session.learn(LearnContext("run-1", "profile", (), _feedback("x"))) == []


def test_backend_satisfies_the_memory_protocols(tmp_path) -> None:
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        assert isinstance(memory, Memory) and isinstance(session, MemorySession)


def test_commit_writes_with_compare_and_set_and_is_idempotent(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, "<profile># Ada\nBrief.</profile>")
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        batch = tuple(root.learn(LearnContext("run-1", "profile", (), _feedback("Be brief."))))
        monkeypatch.setattr(profile, "model_from_config", lambda config: pytest.fail("commit built a model"))
        session.commit("commit-1", batch)
        session.commit("commit-1", batch)
        session.commit("commit-empty", ())
        with pytest.raises(MemoryOperationError) as caught:
            session.commit("commit-1", ())
        assert caught.value.code == "profile.invalid_batch"
    assert memory.read()[0] == "# Ada\nBrief.\n"
    writes = (tmp_path / ".ada.md.writes.jsonl").read_text().splitlines()
    assert len(writes) == 2
    provenance = json.loads(writes[-1])["provenance"]
    assert provenance["commit_id"] == "commit-1" and provenance["run_id"] == "run-1"
    assert set(provenance) == {"commit_id", "run_id", "seen_version", "feedback"}
    assert provenance["feedback"] == ["Be brief."]


def test_concurrent_write_after_learning_is_a_retryable_mismatch_without_a_model(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, "<profile># Ada\nFrom model.</profile>")
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        batch = tuple(root.learn(LearnContext("run-1", "profile", (), _feedback("x"))))
        memory.write("# Ada\nFrom human.\n", base_version=memory.read()[1])
        monkeypatch.setattr(profile, "model_from_config", lambda config: pytest.fail("commit built a model"))
        with pytest.raises(MemoryOperationError) as caught:
            session.commit("commit-1", batch)
    assert (caught.value.stage, caught.value.code, caught.value.retryable) == (
        "commit", "profile.version_mismatch", True)
    assert memory.read()[0] == "# Ada\nFrom human.\n"


def test_store_failure_during_commit_is_not_retryable(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, "<profile># Ada\nNew.</profile>")
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        root = session
        batch = tuple(root.learn(LearnContext("run-1", "profile", (), _feedback("x"))))
        for error, code in ((OSError("disk"), "profile.store_unavailable"),
                            (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"), "profile.store_invalid")):
            def fail(text, *, base_version, provenance=None, error=error):
                raise error
            monkeypatch.setattr(memory._store, "write", fail)
            with pytest.raises(MemoryOperationError) as caught:
                session.commit(f"commit-{code}", batch)
            assert (caught.value.code, caught.value.retryable) == (code, False)


def test_learner_errors_chain_a_cause_without_the_key(tmp_path, monkeypatch) -> None:
    key = "sk-sentinel-profile-zqxw"
    echo = f"Incorrect API key provided: {key[:8]}****{key[-4:]} ({key})"
    store = FileMarkdownStore(tmp_path / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(store, llm_config={**CONFIG, "api_key": key})

    def refuse(config):
        raise ValueError(echo)

    for code, install in (
        ("profile.provider", lambda: _install(monkeypatch, RuntimeError(echo))),
        ("profile.model_construction", lambda: monkeypatch.setattr(profile, "model_from_config", refuse)),
    ):
        install()
        with memory.open_run() as session:
            with pytest.raises(MemoryOperationError) as caught:
                session.learn(LearnContext("run-1", "profile", (), _feedback("x")))
        cause = caught.value.__cause__
        assert caught.value.code == code
        assert str(cause) == "Incorrect API key provided: [REDACTED] ([REDACTED])"
        assert key[:8] not in str(cause) and key[-4:] not in str(cause)


def test_session_recalls_and_learns_the_whole_document(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, "<profile>\n# Ada\nBrief.\n</profile>")
    memory = _memory(tmp_path)
    with memory.open_run() as session:
        assert not hasattr(session, "open_frame")
        assert session.recall() == "# Ada\n"
        assert session.recall(RECALL) == "# Ada\n"
        mutation, = session.learn(LearnContext("run-1", "profile", (), _feedback("Be brief.")))
        assert not hasattr(mutation, "frame_ref")
        session.commit("commit-1", (mutation,))
        with pytest.raises(MemoryOperationError) as caught:
            session.commit("commit-2", (mutation, mutation))
        assert caught.value.code == "profile.invalid_batch"
    assert memory.read()[0] == "# Ada\nBrief.\n"
    (tmp_path / "large").mkdir()
    large = _memory(tmp_path / "large", "# Ada\n" + "x" * 40_000, max_chars=100)
    with large.open_run() as session:
        # max_chars bounds learner replies, never recall.
        assert session.recall(RECALL) == large.read()[0]
        assert session.recall() == large.read()[0]
