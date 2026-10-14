"""Memory, AGENT backward, structured replies and fingerprints hold ints of any size."""

from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.messages import AIMessage
import pytest

import yian.json_codec as json_codec
import yian.llm_clients as llm_clients
import yian.memory.profile as profile
import yian.memory.reflective as reflective
import yian.nodes.agent.backward as backward
from yian import load_flow, run_flow
from yian.compile.manifest import _canonical_json
from yian.memory import LearnContext, MemoryEpisode, MemoryEpisodeStatus, MemoryMutation
from yian.memory.episodic import EpisodicMemory
from yian.memory.profile import ProfileMemory
from yian.memory.reflective import ReflectiveMemory
from yian.memory.storage import FileMarkdownStore, SQLiteStorage
from yian.nodes.agent import AgentNode
from yian.nodes.base import BackwardContext, BackwardResult
from yian.run import resume_flow
from yian.runtime.learning import memory_commit_id
from yian.runtime.log import decode_record
from tests.engine.test_agent import _FakeChat


BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
DIGITS = json_codec.int_text(BIG)


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


def _episode() -> MemoryEpisode:
    return MemoryEpisode(
        "run-1", (), ("chat", "reply"), MemoryEpisodeStatus.SUCCEEDED,
        {"n": BIG}, {"m": [-BIG]}, base_system_prompt="system", prompt="count",
    )


def test_episodic_memory_commits_and_recalls_a_big_int():
    memory = EpisodicMemory()
    with memory.open_run() as session:
        batch = tuple(session.learn(LearnContext("run-1", "episodic", (_episode(),))))
        session.commit("big", batch)

    with memory.open_run() as session:
        recalled = session.recall()
        snapshot = session.snapshot

    entry, = (json_codec.loads(line) for line in recalled.splitlines())
    assert entry["episode"]["inputs"] == {"n": BIG}
    assert entry["episode"]["output"] == {"m": [-BIG]}
    assert snapshot["episodes"][0]["episode"]["inputs"] == {"n": BIG}


def test_reflective_memory_shows_a_big_int_episode_to_the_reflector(monkeypatch):
    class _Structured:
        messages = None

        def invoke(self, messages):
            self.messages = messages
            return {"actions": [{"action": "noop", "lesson_id": None, "text": None}]}

    structured = _Structured()

    class _Reflector:
        def with_structured_output(self, schema):
            return structured

    monkeypatch.setattr(reflective, "model_from_config", lambda config: _Reflector())
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    with memory.open_run() as session:
        assert session.learn(LearnContext("run-1", "reflective", (_episode(),))) == []

    prompt = structured.messages[-1].content
    episodes = json_codec.loads(prompt.split("Episodes: ", 1)[1])["episodes"]
    assert episodes[0]["inputs"] == {"n": BIG}
    assert episodes[0]["output"] == {"m": [-BIG]}


def test_sqlite_storage_state_round_trips_a_big_int(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "episodic")
    try:
        store.save({"n": BIG, "xs": [-BIG]})
        assert store.load() == {"n": BIG, "xs": [-BIG]}
    finally:
        store.close()
    reopened = SQLiteStorage(path, "episodic")
    try:
        assert reopened.load() == {"n": BIG, "xs": [-BIG]}
    finally:
        reopened.close()


def test_profile_learner_prompt_holds_a_big_int_episode(tmp_path, monkeypatch):
    class _Learner:
        def __init__(self):
            self.requests = []

        def invoke(self, messages):
            self.requests.append(messages)
            return AIMessage(content="<profile>\n# Ada\n- Counts.\n</profile>")

    learner = _Learner()
    monkeypatch.setattr(profile, "model_from_config", lambda config: learner)
    store = FileMarkdownStore(tmp_path / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(store, llm_config={"provider": "ollama", "model": "local-profiler"})
    feedback = ({"channel": "output", "target_name": None, "value": "Count.",
                 "owner_static_path": None, "execution_seq": None},)
    with memory.open_run() as session:
        session.learn(LearnContext("run-1", "profile", (_episode(),), feedback))

    prompt = learner.requests[-1][1].content
    assert DIGITS in prompt
    assert json_codec.loads(prompt)["episodes"][0]["inputs"] == {"n": BIG}


def test_markdown_store_writes_a_big_int_provenance_to_its_log(tmp_path):
    store = FileMarkdownStore(tmp_path / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1], provenance={"n": BIG})

    line, = (tmp_path / ".ada.md.writes.jsonl").read_text().splitlines()
    assert DIGITS in line
    assert json_codec.loads(line)["provenance"] == {"n": BIG}


def test_memory_commit_id_digests_a_big_int_payload():
    mutations = (MemoryMutation("m1", {"n": BIG}),)

    first = memory_commit_id("run-1", "episodic", mutations, 1)

    assert first.startswith("ac-memory-v3:")
    assert first == memory_commit_id("run-1", "episodic", mutations, 1)
    assert first != memory_commit_id("run-1", "episodic", (MemoryMutation("m1", {"n": BIG + 1}),), 1)


_STRUCTURED_FLOW = """
id: big-structured
name: big_structured
nodes:
  count:
    kind: agent
    llm_config: {provider: vllm, model: no-structured-sentinel}
    prompt: Count.
    output:
      n: int
output: ${count.output}
"""


def test_a_structured_fallback_reply_holding_a_big_int_succeeds(tmp_path, monkeypatch):
    """`_FakeChat` has no `with_structured_output`, and the catalog marks
    `vllm`/`no-structured-sentinel` as lacking native structured output, so the
    AGENT node parses the free-text reply on the prompt-injection fallback path."""
    chat = _FakeChat([AIMessage(content='{"n": ' + DIGITS + "}")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    path = tmp_path / "structured.jsonl"

    result = run_flow(load_flow(_STRUCTURED_FLOW), {}, log=path)

    assert result.status == "succeeded", result.failure
    assert result.output == {"n": BIG}
    assert chat.calls == 1
    text = path.read_text()
    assert DIGITS in text
    for line in text.splitlines():
        decode_record(line)


def _backward_context() -> BackwardContext:
    return BackwardContext(
        inputs={"n": BIG}, input_names=("n",), variable_names=(),
        output={"m": -BIG}, run_id="recorded-run", node_id="counter", node_started_seq=3,
        node_data={
            "prompt_template": "Count.", "variables": {}, "variable_domains": {},
            "rendered_prompt": "Count.",
            "effective_llm_config": {"provider": "test", "model": "recorded"},
            "_evaluation_evidence": None,
        },
        downstream_context=(),
    )


def test_agent_backward_prompt_holds_big_ints(monkeypatch):
    class _Chat(_FakeChat):
        def __init__(self, replies):
            super().__init__(replies)
            self.messages = []

        def invoke(self, messages):
            self.messages.append(messages)
            return super().invoke(messages)

    reply = '{"input_feedback": {"n": ["Recount."]}, "self_feedback": [], "variable_feedback": {}}'
    chat = _Chat([AIMessage(content=reply)])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)

    result = AgentNode("counter").backward(_backward_context(), ("Check the count.",))

    prompt = json_codec.loads(chat.messages[0][1].content)
    assert prompt["forward"]["inputs"] == {"n": BIG}
    assert prompt["forward"]["output"] == {"m": -BIG}
    assert result == BackwardResult({"n": ("Recount.",)})


def test_agent_backward_reply_holding_a_big_int_parses_and_is_checked_as_feedback():
    reply = (
        '{"input_feedback": {"n": [' + DIGITS + ']}, "self_feedback": [], '
        '"variable_feedback": {}}'
    )

    # The reply decodes; an int is not a feedback shape.
    with pytest.raises(TypeError, match="must be a string, object, list, or None"):
        backward._parse_result(reply)


def test_manifest_canonical_json_writes_a_big_int():
    assert _canonical_json({"v": BIG}) == json_codec.dumps(
        {"v": BIG}, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


_LOOP_FLOW = """
id: big-loop
name: big_loop
input:
  x: int
defs:
  step:
    input:
      n: int
    nodes:
      s:
        kind: code
        input:
          n: ${input.n}
        code: |
          return {"n": inputs["n"] + 1}
        output:
          n: int
    output: ${s.output}
nodes:
  loop:
    kind: loop
    call: step
    input:
      n: ${input.x}
    times: 3
output: ${loop.output}
"""

_MAP_FLOW = """
id: big-map
name: big_map
input:
  items: list[int]
defs:
  double:
    input:
      item: int
    nodes:
      c:
        kind: code
        input:
          v: ${input.item}
        code: |
          return inputs["v"] * 2
    output: ${c.output}
nodes:
  m:
    kind: map
    over: ${input.items}
    call: double
    input:
      item: ${item}
output: ${m.output}
"""

_CALL_FLOW = """
id: big-call
name: big_call
input:
  x: int
defs:
  inc:
    input:
      y: int
    nodes:
      c:
        kind: code
        input:
          y: ${input.y}
        code: |
          return inputs["y"] + 1
    output: ${c.output}
nodes:
  call:
    kind: call
    call: inc
    input:
      y: ${input.x}
output: ${call.output}
"""


def _assert_grown_and_replayed(
    text: str, inputs: dict, path: Path, expected: object, *, learnable: bool = False
) -> None:
    """Run `text` with a log, then assert resuming the terminal log changes no bytes."""
    result = run_flow(load_flow(text), inputs, log=path, learnable=learnable)
    assert result.status == "succeeded", result.failure
    assert result.output == expected
    before = path.read_bytes()

    resumed = resume_flow(load_flow(text), log=path)

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == expected
    assert path.read_bytes() == before


def test_a_loop_seeded_with_a_big_int_grows_and_replays(tmp_path):
    _assert_grown_and_replayed(_LOOP_FLOW, {"x": -BIG}, tmp_path / "loop.jsonl", {"n": -BIG + 3})


def test_a_map_over_big_ints_grows_and_replays(tmp_path):
    _assert_grown_and_replayed(
        _MAP_FLOW, {"items": [BIG, -BIG]}, tmp_path / "map.jsonl", [2 * BIG, -2 * BIG]
    )


@pytest.mark.parametrize("learnable", [False, True], ids=["plain", "learnable"])
def test_a_call_bound_to_a_big_int_grows_and_replays(tmp_path, learnable: bool):
    _assert_grown_and_replayed(
        _CALL_FLOW, {"x": BIG}, tmp_path / "call.jsonl", BIG + 1, learnable=learnable
    )
