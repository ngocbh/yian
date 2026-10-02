"""Host-side construction of the memories a flow declares."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from agent_composer.compose import load_flow
from agent_composer.memory.contracts import MemorySetupError
from agent_composer.memory.episodic import EpisodicMemory
from agent_composer.memory.profile import ProfileMemory
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.memory.session import SessionMemory
from agent_composer.memory_host import MemoryRegistry, build_memories


class _Closable:
    def __init__(self, log: list[str], name: str, error: BaseException | None = None):
        self.log, self.name, self.error = log, name, error

    def open_run(self, *, snapshot=None):
        raise AssertionError("not opened in these tests")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.log.append(self.name)
        if self.error is not None:
            raise self.error


def test_registry_is_a_read_only_mapping_with_types() -> None:
    log: list[str] = []
    a, b = _Closable(log, "a"), _Closable(log, "b")
    registry = MemoryRegistry({"a": a, "b": b}, {"a": "episodic", "b": "custom"}, owned=(a,))
    assert isinstance(registry, Mapping)
    assert dict(registry) == {"a": a, "b": b}
    assert dict(registry.types) == {"a": "episodic", "b": "custom"}
    with pytest.raises(TypeError):
        registry.types["a"] = "x"  # type: ignore[index]


def test_close_closes_owned_objects_in_reverse_order_once() -> None:
    log: list[str] = []
    a, b, c = (_Closable(log, n) for n in "abc")
    registry = MemoryRegistry({"a": a, "b": b, "c": c}, dict.fromkeys("abc", "t"), owned=(a, b))
    with registry as entered:
        assert entered is registry
    assert log == ["b", "a"]
    registry.close()
    assert log == ["b", "a"]


def test_close_keeps_the_first_failure_and_notes_the_rest() -> None:
    log: list[str] = []
    first, second = RuntimeError("first"), RuntimeError("second")
    a, b = _Closable(log, "a", second), _Closable(log, "b", first)
    registry = MemoryRegistry({"a": a, "b": b}, {"a": "t", "b": "t"}, owned=(a, b))
    with pytest.raises(RuntimeError, match="first") as caught:
        registry.close()
    assert log == ["b", "a"]
    assert any("RuntimeError" in note for note in caught.value.__notes__)


def test_exit_keeps_the_primary_exception() -> None:
    log: list[str] = []
    a = _Closable(log, "a", RuntimeError("cleanup"))
    with pytest.raises(KeyError) as caught:
        with MemoryRegistry({"a": a}, {"a": "t"}, owned=(a,)):
            raise KeyError("primary")
    assert any("cleanup failed" in note for note in caught.value.__notes__)


def _flow(memory: str, *, flow_id: str | None = "host-flow", agent_memory: str | None = None):
    lines = [] if flow_id is None else [f"id: {flow_id}"]
    lines += ["name: host", f"long_term_memory: {memory}"]
    lines += ["llm_config: {provider: openai, model: fake-model}", "nodes:"]
    if agent_memory is None:
        lines += [
            "  echo: {kind: code, code: tests.engine._compose_codefns:echo, "
            "input: {topic: hi}, output: str}",
            "output: ${echo.output}",
        ]
    else:
        lines += [
            "  ask:",
            "    kind: agent",
            "    mode: plain",
            f"    long_term_memory: {agent_memory}",
            "    prompt: hi",
            "    output: str",
            "output: ${ask.output}",
        ]
    return load_flow("\n".join(lines) + "\n")


class _Custom:
    def __init__(self, **config):
        self.config = config

    def open_run(self, *, snapshot=None):
        raise AssertionError("not opened")

    def reset(self) -> None:
        pass


def test_builtins_are_constructed_from_declared_config() -> None:
    flow = _flow(
        "{lessons: reflective, notes: {type: episodic, config: {max_episodes: 5}}, "
        "about: profile}",
        agent_memory="{chat: {type: session, config: {recent_turns: 2}}}",
    )
    with build_memories(flow) as registry:
        assert sorted(registry) == ["about", "chat", "lessons", "notes"]
        assert type(registry["notes"]) is EpisodicMemory
        assert type(registry["lessons"]) is ReflectiveMemory
        assert type(registry["chat"]) is SessionMemory
        assert type(registry["about"]) is ProfileMemory
        assert dict(registry.types) == {
            "about": "profile",
            "chat": "session",
            "lessons": "reflective",
            "notes": "episodic",
        }


def test_supplied_wins_and_is_not_closed() -> None:
    log: list[str] = []
    mine = _Closable(log, "mine")
    with build_memories(_flow("{notes: episodic}"), supplied={"notes": mine}) as registry:
        assert registry["notes"] is mine
    assert log == []


def test_factories_build_custom_types_with_config() -> None:
    flow = _flow("{vec: {type: vector, config: {top_k: 3}}}")
    registry = build_memories(flow, factories={"vector": _Custom})
    assert registry["vec"].config == {"top_k": 3}
    assert registry.types["vec"] == "vector"


@pytest.mark.parametrize(
    ("memory", "kwargs", "message"),
    [
        ("{vec: vector}", {}, "no factory"),
        ("{notes: episodic}", {"factories": {"episodic": _Custom}}, "built-in"),
        ("{notes: {type: episodic, config: {max_episodes: 5}}}",
         {"supplied": {"notes": _Custom()}}, "config"),
        ("{notes: episodic}", {"supplied": {"other": _Custom()}}, "not declared"),
        ("{notes: episodic}", {"storage": {"other": "x.sqlite"}}, "not declared"),
        ("{notes: episodic}", {"supplied": {"notes": _Custom()}, "storage": {"notes": "x"}},
         "storage"),
        ("{vec: vector}", {"factories": {"vector": _Custom}, "storage": {"vec": "x"}},
         "storage"),
        ("{notes: episodic}", {"storage": {"notes": ":memory:"}}, "not a local file path"),
        ("{notes: episodic}", {"supplied": {"notes": object()}}, "not a Memory"),
        ("{vec: vector}", {"factories": {"vector": lambda **_: object()}}, "not a Memory"),
        ("{notes: {type: episodic, config: {max_episodes: 0}}}", {}, "rejected its config"),
        ("{vec: {type: vector, config: {bad: 1}}}",
         {"factories": {"vector": lambda: _Custom()}}, "rejected its config"),
    ],
)
def test_setup_errors(memory: str, kwargs: dict, message: str) -> None:
    with pytest.raises(MemorySetupError, match=message):
        build_memories(_flow(memory), **kwargs)


def test_setup_errors_carry_the_declaration_origin() -> None:
    flow = _flow("{vec: vector}")
    with pytest.raises(MemorySetupError) as caught:
        build_memories(flow)
    frame = flow.origin_resolver.resolve(caught.value.origin)
    assert frame.range.start_line == 3


def test_flow_without_memory_builds_an_empty_registry() -> None:
    flow = load_flow(
        "id: none\nname: none\nnodes:\n"
        "  echo: {kind: code, code: tests.engine._compose_codefns:echo, "
        "input: {topic: hi}, output: str}\n"
        "output: ${echo.output}\n"
    )
    with build_memories(flow) as registry:
        assert len(registry) == 0
