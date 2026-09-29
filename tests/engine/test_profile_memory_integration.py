"""Public runs, backward attempts, and recovery with ProfileMemory."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
import pytest

import agent_composer.llm_clients as llm_clients
import agent_composer.memory.profile as profile
from agent_composer.compose import load_flow
from agent_composer.memory import MemorySetupError
from agent_composer.memory.profile import ProfileMemory
from agent_composer.memory.storage import FileProfileStore
from agent_composer.memory.session import SessionMemory
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import JsonlRunLog, MemoryCommitFailedRecord
from tests.engine.test_agent_backward import _BackwardChat, _reply

_CHAT = """id: chat
name: chat
long_term_memory: profile
input:
  message: str
nodes:
  reply:
    kind: agent
    mode: plain
    input:
      message: ${input.message}
    prompt: ${message}
    output: str
output: ${reply.output}
"""

_NESTED = """id: nested
name: nested
long_term_memory: profile
input:
  message: str
defs:
  child:
    long_term_memory: profile
    input:
      message: str
    nodes:
      reply:
        kind: agent
        mode: plain
        long_term_memory: profile
        input:
          message: ${input.message}
        prompt: ${message}
        output: str
    output: ${reply.output}
nodes:
  invoke:
    kind: call
    call: child
    input:
      message: ${input.message}
output: ${invoke.output}
"""

_SIBLINGS = """id: siblings
name: siblings
input:
  message: str
nodes:
  first:
    kind: agent
    mode: plain
    long_term_memory: profile
    input:
      message: ${input.message}
    prompt: ${message}
    output: str
  second:
    kind: agent
    mode: plain
    long_term_memory: profile
    input:
      message: ${input.message}
    prompt: ${message}
    output: str
output: ${first.output}
"""

_OFFICE = """id: office
name: office
long_term_memory: profile
input:
  message: str
nodes:
  reply:
    kind: agent
    mode: plain
    long_term_memory: session
    input:
      message: ${input.message}
    prompt: ${message}
    output: str
output: ${reply.output}
"""


class _Learner:
    """Stub profile learner that can run a side effect before replying."""

    def __init__(self, text: str, before_reply=None) -> None:
        self.text = text
        self.before_reply = before_reply
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        if self.before_reply is not None:
            self.before_reply()
        return AIMessage(content=f"<profile>\n{self.text}\n</profile>")


def _profile(tmp_path) -> ProfileMemory:
    store = FileProfileStore(tmp_path / "ada.md")
    store.write("# Ada\n- Works on reefs.\n", base_version=store.read()[1])
    return ProfileMemory(store, llm_config={"provider": "ollama", "model": "local-profiler"})


def _forward(monkeypatch, *answers) -> list:
    chats = []

    def factory(config):
        chat = _BackwardChat([AIMessage(content=answer) for answer in answers])
        chats.append(chat)
        return chat

    monkeypatch.setattr(llm_clients, "model_from_config", factory)
    return chats


def _learner(monkeypatch, learner: _Learner) -> list:
    built = []
    monkeypatch.setattr(profile, "model_from_config", lambda config: built.append(config) or learner)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: _BackwardChat([_reply()]))
    return built


def _material(learner: _Learner, index: int = -1) -> dict:
    return json.loads(learner.requests[index][1].content)


def test_recall_learn_and_commit_through_a_public_run(tmp_path, monkeypatch) -> None:
    memory = _profile(tmp_path)
    flow = load_flow(_CHAT)
    chats = _forward(monkeypatch, "Hello.")
    result = run_flow(flow, {"message": "hi"}, learnable=True, memory={"profile": memory})
    assert result.status == "succeeded"
    system = chats[0].messages[0][0].content
    assert json.dumps("# Ada\n- Works on reefs.\n") in system

    learner = _Learner("# Ada\n- Works on reefs.\n- Prefers brief replies.")
    built = _learner(monkeypatch, learner)
    outcome = result.backward("Keep replies brief.")
    assert outcome.status == "succeeded", outcome.failure
    assert len(built) == 1 and len(learner.requests) == 1
    assert "Keep replies brief." in [entry["value"] for entry in _material(learner)["feedback"]]
    assert memory.read()[0] == "# Ada\n- Works on reefs.\n- Prefers brief replies.\n"
    last = json.loads((tmp_path / ".ada.md.writes.jsonl").read_text().splitlines()[-1])
    assert last["provenance"]["feedback"] == ["Keep replies brief."]


def test_nested_frames_show_and_rewrite_the_document_once(tmp_path, monkeypatch) -> None:
    memory = _profile(tmp_path)
    chats = _forward(monkeypatch, "Hello.")
    result = run_flow(load_flow(_NESTED), {"message": "hi"}, learnable=True, memory={"profile": memory})
    assert result.status == "succeeded"
    assert chats[0].messages[0][0].content.count("Works on reefs.") == 1
    learner = _Learner("# Ada\n- Brief.")
    _learner(monkeypatch, learner)
    assert result.backward("Be brief.").status == "succeeded"
    assert len(learner.requests) == 1


def test_unrelated_profile_frames_fail_at_run_start(tmp_path, monkeypatch) -> None:
    _forward(monkeypatch, "a", "b")
    with pytest.raises(MemorySetupError, match="outermost"):
        run_flow(load_flow(_SIBLINGS), {"message": "hi"}, memory={"profile": _profile(tmp_path)})


def test_profile_notes_and_a_session_transcript_share_one_agent_context(tmp_path, monkeypatch) -> None:
    profile_memory = _profile(tmp_path)
    memory = {"profile": profile_memory, "session": SessionMemory()}
    flow = load_flow(_OFFICE)
    _forward(monkeypatch, "First reply.")
    assert run_flow(flow, {"message": "hello"}, memory=memory).status == "succeeded"
    chats = _forward(monkeypatch, "Second reply.")
    result = run_flow(flow, {"message": "again"}, learnable=True, memory=memory)
    system, *history = chats[0].messages[0]
    assert system.content.count("Works on reefs.") == 1
    assert [message.content for message in history] == ["hello", "First reply.", "again"]
    _learner(monkeypatch, _Learner("# Ada\n- Works on reefs.\n- Brief."))
    outcome = result.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    assert profile_memory.read()[0] == "# Ada\n- Works on reefs.\n- Brief.\n"


def test_human_edit_during_learning_fails_the_commit_without_a_model(tmp_path, monkeypatch) -> None:
    memory = _profile(tmp_path)
    path = tmp_path / "run.jsonl"
    _forward(monkeypatch, "Hello.")
    result = run_flow(load_flow(_CHAT), {"message": "hi"}, log=path, learnable=True,
                      memory={"profile": memory})

    def human_edit():
        memory.write("# Ada\n- Edited by Ada.\n", base_version=memory.read()[1])

    learner = _Learner("# Ada\n- From the model.", before_reply=human_edit)
    built = _learner(monkeypatch, learner)
    outcome = result.backward("Keep replies brief.")
    assert outcome.status == "failed"
    assert (outcome.failure.stage, outcome.failure.code) == ("commit", "profile.version_mismatch")
    assert outcome.failure.cause.retryable is True
    assert len(built) == len(learner.requests) == 1
    assert memory.read()[0] == "# Ada\n- Edited by Ada.\n"
    with JsonlRunLog.resume(path) as log:
        failed = [item.record for item in log if type(item.record) is MemoryCommitFailedRecord]
    assert [record.failure.code for record in failed] == ["profile.version_mismatch"]


def test_resume_reopens_the_recorded_version_for_backward(tmp_path, monkeypatch) -> None:
    memory = _profile(tmp_path)
    flow = load_flow(_CHAT)
    path = tmp_path / "run.jsonl"
    recorded_text, recorded_version = memory.read()
    _forward(monkeypatch, "Hello.")
    run_flow(flow, {"message": "hi"}, log=path, learnable=True, memory={"profile": memory})
    with JsonlRunLog.resume(path) as log:
        started = next(iter(log)).record
    snapshot, = started.memory_snapshots
    assert snapshot.snapshot["profile_version"] == recorded_version
    memory.write("# Ada\n- Edited later.\n", base_version=recorded_version)

    resumed = resume_flow(flow, log=path, memory={"profile": ProfileMemory(
        FileProfileStore(tmp_path / "ada.md"), llm_config={"provider": "ollama", "model": "local-profiler"})})
    learner = _Learner("# Ada\n- Edited later.\n- Brief.")
    _learner(monkeypatch, learner)
    outcome = resumed.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    material = _material(learner)
    assert material["seen_profile"] == recorded_text
    assert material["current_profile"] == "# Ada\n- Edited later.\n"
    assert memory.read()[0] == "# Ada\n- Edited later.\n- Brief.\n"


def test_a_new_attempt_after_a_mismatch_learns_from_the_edited_document(tmp_path, monkeypatch) -> None:
    memory = _profile(tmp_path)
    _forward(monkeypatch, "Hello.")
    result = run_flow(load_flow(_CHAT), {"message": "hi"}, learnable=True, memory={"profile": memory})
    learner = _Learner("# Ada\n- From the model.", before_reply=lambda: memory.write(
        "# Ada\n- Edited by Ada.\n", base_version=memory.read()[1]))
    _learner(monkeypatch, learner)
    assert result.backward("Keep replies brief.").status == "failed"

    retry = _Learner("# Ada\n- Edited by Ada.\n- Prefers brief replies.")
    _learner(monkeypatch, retry)
    outcome = result.backward("Keep replies brief.")
    assert outcome.status == "succeeded", outcome.failure
    assert _material(retry)["current_profile"] == "# Ada\n- Edited by Ada.\n"
    assert memory.read()[0] == "# Ada\n- Edited by Ada.\n- Prefers brief replies.\n"


class _KilledAfterWrite(FileProfileStore):
    """Store that writes, then dies like a process killed before the runtime records success."""

    kill = True

    def write(self, text, *, base_version, provenance=None):
        version = super().write(text, base_version=base_version, provenance=provenance)
        if self.kill and provenance is not None:
            raise SystemExit("killed after the store write")
        return version


def test_an_attempt_killed_after_its_write_is_unknown_and_findable_in_provenance(tmp_path, monkeypatch) -> None:
    store = _KilledAfterWrite(tmp_path / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(store, llm_config={"provider": "ollama", "model": "local-profiler"})
    flow, path = load_flow(_CHAT), tmp_path / "run.jsonl"
    _forward(monkeypatch, "Hello.")
    run_flow(flow, {"message": "hi"}, log=path, learnable=True, memory={"profile": memory})
    _learner(monkeypatch, _Learner("# Ada\n- Brief."))
    with pytest.raises(SystemExit):
        resume_flow(flow, log=path, memory={"profile": memory}).backward("Be brief.")
    assert store.read()[0] == "# Ada\n- Brief.\n"

    store.kill = False
    resumed = resume_flow(flow, log=path, memory={"profile": memory})
    unknown, = resumed.backward_outcome.learning.policies
    assert unknown.status == "unknown" and unknown.commit_id.startswith("ac-memory-v1:")
    log = tmp_path / ".ada.md.writes.jsonl"
    written = [json.loads(line)["provenance"] for line in log.read_text().splitlines()][1:]
    assert [entry["commit_id"] for entry in written] == [unknown.commit_id]

    learner = _Learner("# Ada\n- Brief.")
    _learner(monkeypatch, learner)
    outcome = resumed.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    assert "may already reflect this feedback" in learner.requests[-1][0].content
    material = _material(learner)
    assert (material["seen_profile"], material["current_profile"]) == ("# Ada\n", "# Ada\n- Brief.\n")
    assert store.read()[0] == "# Ada\n- Brief.\n"
    assert len(log.read_text().splitlines()) == 2
