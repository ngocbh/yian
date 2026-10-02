"""Built-in SQLite backends open storage on first use, never in __init__."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import pytest

from agent_composer.memory import episodic as episodic_module
from agent_composer.memory.contracts import MemoryOperationError
from agent_composer.memory.episodic import EpisodicMemory

# (module, factory) pairs; C14 and C15 append reflective and session.
BACKENDS: list[tuple[object, Callable[..., object]]] = [
    (episodic_module, lambda path, **kw: EpisodicMemory(storage=path, **kw)),
]

from agent_composer.memory import reflective as reflective_module
from agent_composer.memory.reflective import ReflectiveMemory

_REFLECTIVE_LLM = {"provider": "ollama", "model": "current"}

BACKENDS.append(
    (
        reflective_module,
        lambda path, **kw: ReflectiveMemory(_REFLECTIVE_LLM, storage=path, **kw),
    )
)

from agent_composer.memory import session as session_module
from agent_composer.memory.session import SessionMemory

BACKENDS.append(
    (session_module, lambda path, **kw: SessionMemory(storage=path, **kw))
)


def _ids(case) -> str:
    return case[0].__name__.rsplit(".", 1)[-1]


@pytest.fixture(params=BACKENDS, ids=_ids)
def backend(request):
    return request.param


def _closed_code(module) -> str:
    return f"{module.__name__.rsplit('.', 1)[-1]}.closed"


def test_construction_touches_no_file(tmp_path: Path, backend) -> None:
    _module, make = backend
    path = tmp_path / "m.sqlite"
    memory = make(path)
    assert list(tmp_path.iterdir()) == []
    memory.open_run().close()
    assert path.exists()
    memory.close()


@pytest.mark.parametrize("bad", ["", ":memory:", "file:m.sqlite", "a\x00b"])
def test_bad_path_shape_fails_at_construction_without_io(
    tmp_path: Path, backend, bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _module, make = backend
    monkeypatch.chdir(tmp_path)
    with pytest.raises(MemoryOperationError) as caught:
        make(bad)
    assert caught.value.code == "storage.path"
    assert list(tmp_path.iterdir()) == []


def test_reset_also_opens_storage(tmp_path: Path, backend) -> None:
    _module, make = backend
    path = tmp_path / "m.sqlite"
    memory = make(path)
    memory.reset()
    assert path.exists()
    memory.close()


def test_missing_parent_fails_at_first_open_without_create_parent(
    tmp_path: Path, backend
) -> None:
    _module, make = backend
    memory = make(tmp_path / "missing" / "m.sqlite")
    with pytest.raises(MemoryOperationError) as caught:
        memory.open_run()
    assert caught.value.code == "storage.io"
    assert not (tmp_path / "missing").exists()
    memory.close()


def test_create_parent_makes_the_directory_at_first_open(tmp_path: Path, backend) -> None:
    _module, make = backend
    path = tmp_path / "a" / "b" / "m.sqlite"
    memory = make(path, create_parent=True)
    assert not (tmp_path / "a").exists()
    memory.open_run().close()
    assert path.exists()
    memory.close()


def test_close_before_open_does_no_io_and_blocks_later_use(tmp_path: Path, backend) -> None:
    module, make = backend
    memory = make(tmp_path / "m.sqlite")
    memory.close()
    for operation in (memory.open_run, memory.reset):
        with pytest.raises(MemoryOperationError) as caught:
            operation()
        assert caught.value.code == _closed_code(module)
    assert list(tmp_path.iterdir()) == []


def test_pid_ownership_starts_at_construction(
    tmp_path: Path, backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _module, make = backend
    memory = make(tmp_path / "m.sqlite")
    parent = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent + 1)
    with pytest.raises(MemoryOperationError) as caught:
        memory.open_run()
    assert caught.value.code == "storage.forked"
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []
    memory.close()


def test_ram_backend_ignores_pid(backend, monkeypatch: pytest.MonkeyPatch) -> None:
    _module, make = backend
    memory = make(None)
    parent = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent + 1)
    memory.open_run().close()
    monkeypatch.undo()
    memory.close()


def test_concurrent_first_open_constructs_storage_once(
    tmp_path: Path, backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, make = backend
    real = module.SQLiteStorage
    calls: list[object] = []
    calling = [threading.Event(), threading.Event()]

    def counting(*args, **kwargs):
        calls.append(args)
        # Hold the first construction until both threads have called open_run.
        for event in calling:
            assert event.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "SQLiteStorage", counting)
    memory = make(tmp_path / "m.sqlite")
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            calling[index].set()
            memory.open_run().close()
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(calls) == 1
    memory.close()


def test_failed_open_can_be_retried(tmp_path: Path, backend) -> None:
    _module, make = backend
    holder = make(tmp_path / "m.sqlite")
    holder.open_run().close()  # takes the exclusive file lock
    waiting = make(tmp_path / "m.sqlite")
    with pytest.raises(MemoryOperationError) as caught:
        waiting.open_run()
    assert caught.value.code == "storage.busy"
    holder.close()
    waiting.open_run().close()
    waiting.close()


def test_entering_the_backend_opens_storage(tmp_path: Path, backend) -> None:
    _module, make = backend
    path = tmp_path / "m.sqlite"
    with make(path):
        assert path.exists()
        with pytest.raises(MemoryOperationError) as caught:
            make(path).open_run()
        assert caught.value.code == "storage.busy"


def test_failed_load_can_be_retried(
    tmp_path: Path, backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, make = backend
    real_load = module.SQLiteStorage.load
    calls: list[int] = []

    def flaky_load(self):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryOperationError(
                "storage", "storage.io", "transient load failure", True
            )
        return real_load(self)

    monkeypatch.setattr(module.SQLiteStorage, "load", flaky_load)
    memory = make(tmp_path / "m.sqlite")
    with pytest.raises(MemoryOperationError) as caught:
        memory.open_run()
    assert caught.value.code == "storage.io"
    memory.open_run().close()
    assert len(calls) == 2
    memory.close()
