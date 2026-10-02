"""Host-side construction of the memories a flow declares."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from agent_composer.compose import load_flow
from agent_composer.events import RunStarted
from agent_composer.memory.contracts import MemoryOperationError, MemorySetupError
from agent_composer.memory.episodic import EpisodicMemory
from agent_composer.memory.profile import ProfileMemory
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.memory.session import SessionMemory
from agent_composer.memory_host import MemoryRegistry, build_memories
from agent_composer.run import resume_flow, run_flow
from tests.engine._memory_fakes import _Memory


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
    assert caught.value.__notes__ == ["Additional memory cleanup failed: RuntimeError: second"]


def _failure_with_note(message: str, note: str) -> RuntimeError:
    error = RuntimeError(message)
    error.add_note(note)
    return error


def test_close_copies_the_notes_of_later_failures() -> None:
    log: list[str] = []
    second = _failure_with_note("second", "second detail")
    a, b = _Closable(log, "a", second), _Closable(log, "b", RuntimeError("first"))
    registry = MemoryRegistry({"a": a, "b": b}, {"a": "t", "b": "t"}, owned=(a, b))
    with pytest.raises(RuntimeError, match="first") as caught:
        registry.close()
    assert caught.value.__notes__ == [
        "Additional memory cleanup failed: RuntimeError: second",
        "second detail",
    ]


def test_exit_keeps_the_primary_exception() -> None:
    log: list[str] = []
    a = _Closable(log, "a", _failure_with_note("cleanup", "cleanup detail"))
    with pytest.raises(KeyError) as caught:
        with MemoryRegistry({"a": a}, {"a": "t"}, owned=(a,)):
            raise KeyError("primary")
    assert caught.value.__notes__ == [
        "Memory cleanup failed: RuntimeError: cleanup",
        "cleanup detail",
    ]


def _flow(memory: str, *, flow_id: str | None = "host-flow", agent_memory: str | None = None):
    lines = [f"id: {'host-flow' if flow_id is None else flow_id}"]
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
    flow = load_flow("\n".join(lines) + "\n")
    if flow_id is None:
        # YAML requires id:, but a flow compiled in Python may have none.
        flow.compiled.flow_id = None
    return flow


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
        ("{about: profile}", {"storage": {"about": "file:about.md"}}, "not a local file path"),
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


class _BytesPath:
    def __fspath__(self) -> bytes:
        return b"notes.sqlite"


@pytest.mark.parametrize(
    ("memory", "value"),
    [
        ("{notes: episodic}", 123),
        ("{notes: profile}", b"notes.md"),
        ("{notes: episodic}", _BytesPath()),
        ("{notes: reflective}", None),
    ],
)
def test_storage_values_must_be_text_paths(memory: str, value: object) -> None:
    with pytest.raises(MemorySetupError, match=r"storage\['notes'\] must be a str or os.PathLike"):
        build_memories(_flow(memory), storage={"notes": value})


def test_storage_path_errors_show_the_plain_path() -> None:
    with pytest.raises(MemorySetupError) as caught:
        build_memories(_flow("{notes: episodic}"), storage={"notes": Path("file:x")})
    assert "storage 'file:x' is not a local file path" in str(caught.value)


def test_flow_must_be_a_composed_flow() -> None:
    with pytest.raises(TypeError, match="ComposedFlow"):
        build_memories(object())  # type: ignore[arg-type]


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


def _import_flow(tmp_path: Path):
    (tmp_path / "lib.yaml").write_text(
        "id: lib\nname: lib\nlong_term_memory: {notes: episodic}\n"
        "input: {x: str}\nnodes:\n"
        '  a: {kind: code, code: tests.engine._compose_codefns:echo_x, '
        'input: {x: "${input.x}"}, output: str}\n'
        'output: {y: "${a.output}"}\n'
    )
    return load_flow(
        "id: app\nname: app\n"
        "long_term_memory: {notes: episodic, about: profile}\n"
        "input: {x: str}\nnodes:\n"
        '  a: {kind: call, call: lib, input: {x: "${input.x}"}}\n'
        'output: {y: "${input.x}"}\n'
        "uses:\n  lib: lib\n",
        search_paths=[tmp_path],
    )


def test_root_layout_follows_flow_id_scope_and_name(tmp_path: Path) -> None:
    flow = _import_flow(tmp_path)
    root = tmp_path / "mem"
    with build_memories(flow, root=root) as registry:
        assert sorted(registry) == ["about", "lib.notes", "notes"]
        assert not root.exists()  # construction is effect-free
        for key in ("notes", "lib.notes", "about"):
            registry[key].open_run()
        assert (root / "app" / "notes.sqlite").is_file()
        assert (root / "app" / "lib" / "notes.sqlite").is_file()
    assert (root / "app" / ".about.md.lock").exists()


def test_explicit_storage_overrides_root_and_needs_an_existing_parent(tmp_path: Path) -> None:
    flow = _import_flow(tmp_path)
    explicit = tmp_path / "missing" / "notes.sqlite"
    with build_memories(flow, root=tmp_path / "mem", storage={"notes": explicit}) as registry:
        with pytest.raises(MemoryOperationError) as caught:
            registry["notes"].open_run()
        assert caught.value.code == "storage.io"
        assert not explicit.parent.exists()
        registry["lib.notes"].open_run()
    assert (tmp_path / "mem" / "app" / "lib" / "notes.sqlite").is_file()


def test_no_root_means_ram_and_no_flow_id_is_needed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    flow = _flow("{notes: episodic, about: profile}", flow_id=None)
    with build_memories(flow) as registry:
        registry["notes"].open_run()
        registry["about"].open_run()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flow_id", [None, ".", ".."])
def test_root_requires_a_safe_flow_id(tmp_path: Path, flow_id) -> None:
    flow = _flow("{notes: episodic}", flow_id=flow_id)
    with pytest.raises(MemorySetupError, match="flow id"):
        build_memories(flow, root=tmp_path)


@pytest.mark.parametrize("flow_id", ["a/b", "a\\b", "a\x00b", ""])
def test_unsafe_flow_ids_are_rejected_before_touching_disk(tmp_path: Path, flow_id) -> None:
    flow = _flow("{notes: episodic}")
    flow.compiled.flow_id = flow_id  # bypass the loader's id grammar
    with pytest.raises(MemorySetupError, match="flow id"):
        build_memories(flow, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_partial_failure_closes_what_was_built_in_reverse(tmp_path: Path) -> None:
    log: list[str] = []
    flow = _flow("{a: alpha, b: beta, c: gamma}")

    def failing(**_):
        raise ValueError("gamma refused")

    factories = {
        "alpha": lambda: _Closable(log, "a"),
        "beta": lambda: _Closable(log, "b", _failure_with_note("b close", "b detail")),
        "gamma": failing,
    }
    with pytest.raises(MemorySetupError, match="gamma refused") as caught:
        build_memories(flow, factories=factories)
    assert log == ["b", "a"]
    assert caught.value.__notes__ == ["Memory cleanup failed: RuntimeError: b close", "b detail"]


def test_second_registry_on_one_root_is_busy_until_the_first_closes(tmp_path: Path) -> None:
    flow = _flow("{notes: episodic}")
    first = build_memories(flow, root=tmp_path)
    first["notes"].open_run()
    second = build_memories(flow, root=tmp_path)
    with pytest.raises(MemoryOperationError) as caught:
        second["notes"].open_run()
    assert caught.value.code == "storage.busy"
    first.close()
    second["notes"].open_run()
    second.close()


@pytest.mark.parametrize("memory", ["episodic", "reflective", "profile"])
def test_open_after_close_is_closed_and_opens_no_file(tmp_path: Path, memory: str) -> None:
    flow = _flow(f"{{notes: {memory}}}")
    registry = build_memories(flow, root=tmp_path)
    registry.close()
    for call in (registry["notes"].open_run, registry["notes"].reset):
        with pytest.raises(MemoryOperationError) as caught:
            call()
        assert caught.value.code == f"{memory}.closed"
    assert list(tmp_path.iterdir()) == []


def test_package_exports_the_host_builder() -> None:
    import agent_composer

    assert agent_composer.build_memories is build_memories
    assert agent_composer.MemoryRegistry is MemoryRegistry
    assert {"build_memories", "MemoryRegistry"} <= set(agent_composer.__all__)


def test_run_flow_accepts_any_mapping() -> None:
    flow = _flow("{notes: episodic}")
    result = run_flow(flow, {}, memory=MappingProxyType({"notes": _Memory()}))
    assert result.status == "succeeded"


def test_registry_built_for_another_flow_is_rejected() -> None:
    with build_memories(_flow("{notes: reflective}")) as other:
        with pytest.raises(MemorySetupError, match=r"'notes'.*'episodic'.*'reflective'"):
            run_flow(_flow("{notes: episodic}"), {}, memory=other)


def test_registry_for_this_flow_runs() -> None:
    flow = _flow("{notes: episodic}")
    with build_memories(flow) as registry:
        assert run_flow(flow, {}, memory=registry).status == "succeeded"


def test_resume_checks_registry_types_too(tmp_path: Path) -> None:
    flow = _flow("{notes: episodic}")
    log, root = tmp_path / "run.jsonl", tmp_path / "mem"

    def stop_after_start(event: object) -> None:
        if isinstance(event, RunStarted):
            raise RuntimeError("host stop")

    with build_memories(flow, root=root) as registry:
        with pytest.raises(RuntimeError, match="host stop"):
            run_flow(flow, {}, log=log, memory=registry, on_event=stop_after_start)
    with build_memories(_flow("{notes: reflective}")) as other:
        with pytest.raises(MemorySetupError, match=r"'notes'.*'episodic'.*'reflective'"):
            resume_flow(flow, log=log, memory=other)
    with build_memories(flow, root=root) as registry:
        assert resume_flow(flow, log=log, memory=registry).status == "succeeded"
