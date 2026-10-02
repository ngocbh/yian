"""Host-side construction of the memories a flow declares."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from agent_composer.memory_host import MemoryRegistry


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
