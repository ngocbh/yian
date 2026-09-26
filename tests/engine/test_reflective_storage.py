"""Optional reflective persistence preserves lessons and callback ownership."""

from contextlib import ExitStack
from copy import deepcopy
import json
import multiprocessing
import os
import signal
import sqlite3
from threading import Event, Thread
import time

import pytest

import agent_composer.memory.contracts as c
import agent_composer.memory.reflective as reflective
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.memory.storage import SQLiteStorage
from tests.engine.test_reflective_memory import (
    _context,
    _install_model,
    _mutation,
    _ref,
    _spec,
)

_CONFIG = {"provider": "ollama", "model": "current"}


def _memory(path=None, **kwargs):
    return ReflectiveMemory(_CONFIG, storage=path, **kwargs)


def _snapshot(memory):
    with memory.open_run() as opened:
        return opened.snapshot


def _learn(monkeypatch, memory, commit_id="learned", action="add", text="Use evidence"):
    actions = [{"action": action, "lesson_id": "lesson"}]
    if action != "remove":
        actions[0]["text"] = text
    _install_model(monkeypatch, {"actions": actions})
    with memory.open_run() as opened:
        frame = opened.open_frame(_spec())
        batch = tuple(frame.learn(_context(run_id=commit_id)))
        opened.commit(commit_id, batch)
    return batch


def test_learned_lessons_and_limits_survive_reopen(monkeypatch, tmp_path):
    path = tmp_path / "memory.sqlite"
    with _memory(path, max_items=2, recall_limit=1) as memory:
        old = _snapshot(memory)
        _learn(monkeypatch, memory)
        expected = _snapshot(memory)
    with _memory(path, max_items=9, recall_limit=7) as memory:
        assert _snapshot(memory) == expected
        with memory.open_run() as opened:
            assert opened.open_frame(_spec()).recall() == "[lesson] Use evidence"
            opened.open_frame(_spec("new"))
            configs = {
                item["ref"]["static_path"][-1]: item["config"]
                for item in opened.snapshot["frames"]
            }
            assert configs == {
                "agent": {"max_items": 2, "recall_limit": 1},
                "new": {"max_items": 9, "recall_limit": 7},
            }
        with memory.open_run(snapshot=old) as stale:
            assert stale.snapshot == old
            assert stale.open_frame(_spec()).recall() == ""
            with pytest.raises(c.MemoryOperationError) as caught:
                stale.open_frame(_spec()).learn(_context())
            assert caught.value.code == "reflective.stale_generation"


def test_saved_snapshots_pin_model_but_new_sessions_use_host_config(
    monkeypatch, tmp_path
):
    path = tmp_path / "memory.sqlite"
    with ReflectiveMemory(
        {"provider": "ollama", "model": "old"}, storage=path
    ) as memory:
        saved = _snapshot(memory)
    with _memory(path) as memory:
        _, configs = _install_model(monkeypatch, {"actions": [{"action": "noop"}]})
        with memory.open_run(snapshot=saved) as resumed:
            assert resumed.snapshot == saved
            resumed.open_frame(_spec()).learn(_context())
        with memory.open_run() as fresh:
            fresh.open_frame(_spec()).learn(_context())
        assert [config["model"] for config in configs] == ["old", "current"]


def test_pristine_snapshot_adoption_keeps_constructor_model(monkeypatch, tmp_path):
    with ReflectiveMemory({"provider": "ollama", "model": "old"}) as source:
        _learn(monkeypatch, source)
        saved = _snapshot(source)
    path = tmp_path / "memory.sqlite"
    with _memory(path) as memory:
        with memory.open_run(snapshot=saved) as resumed:
            assert resumed.snapshot == saved
        assert _snapshot(memory)["llm_config"]["model"] == "current"
    with _memory(path) as memory:
        current = _snapshot(memory)
        assert current["frames"] == saved["frames"]
        assert current["llm_config"]["model"] == "current"


def test_constructor_persists_identity_without_resolving_model(monkeypatch, tmp_path):
    def forbidden():
        raise AssertionError("ambient model must not be resolved at construction")

    monkeypatch.setattr(reflective, "resolve_model_config", forbidden)
    path = tmp_path / "memory.sqlite"
    memory = ReflectiveMemory(storage=path)
    namespace = memory._namespace
    with pytest.raises(c.MemoryOperationError) as caught:
        ReflectiveMemory(storage=path)
    assert caught.value.code == "storage.busy"
    memory.close()
    memory.close()
    with ReflectiveMemory(storage=path) as reopened:
        assert reopened._namespace == namespace
        assert not reopened._ever_opened
    storage = SQLiteStorage(path, "reflective")
    try:
        encoded = json.dumps(storage.load())
        for forbidden_key in (
            "llm_config",
            "provider",
            "model",
            "api_key",
            "temperature",
        ):
            assert forbidden_key not in encoded
    finally:
        storage.close()


def test_ambient_config_is_current_after_reopen(monkeypatch, tmp_path):
    path = tmp_path / "memory.sqlite"
    monkeypatch.setenv("AGENT_COMPOSER_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("AGENT_COMPOSER_DEFAULT_MODEL", "old")
    with ReflectiveMemory(storage=path) as memory:
        saved = _snapshot(memory)
    monkeypatch.setenv("AGENT_COMPOSER_DEFAULT_MODEL", "new")
    with ReflectiveMemory(storage=path) as memory:
        with memory.open_run(snapshot=saved) as resumed:
            assert resumed.snapshot == saved
        assert _snapshot(memory)["llm_config"]["model"] == "new"


def test_none_storage_never_opens_sqlite(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SQLite must remain dormant")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with _memory() as memory:
        assert _snapshot(memory)["frames"] == []


def test_invalid_defaults_do_not_create_storage(tmp_path):
    path = tmp_path / "memory.sqlite"
    with pytest.raises(ValueError):
        _memory(path, max_items=0)
    assert not path.exists()


def test_whole_commit_is_validated_before_saving(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite"
    with _memory(path) as memory:
        _learn(monkeypatch, memory)
        before = _snapshot(memory)
        with memory.open_run() as opened:
            opened.open_frame(_spec())
            with pytest.raises(c.MemoryOperationError) as caught:
                opened.commit(
                    "invalid",
                    (
                        _mutation(_ref(), "good", "revise", "lesson", "Changed"),
                        _mutation(_ref(), "bad", "remove", "missing"),
                    ),
                )
            assert caught.value.code == "reflective.invalid_mutation"
        assert _snapshot(memory) == before
        assert "invalid" not in memory._applied_commits
    with _memory(path) as memory:
        assert _snapshot(memory) == before


@pytest.mark.parametrize("durable", [False, True])
def test_idle_handles_do_not_block_close_and_are_invalidated(tmp_path, durable):
    memory = _memory(tmp_path / "memory.sqlite" if durable else None)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())
    memory.close()
    for operation in (
        memory.open_run,
        memory.reset,
        memory.__enter__,
        lambda: opened.snapshot,
        lambda: opened.open_frame(_spec()),
        lambda: opened.commit("c", ()),
        frame.recall,
        lambda: frame.learn(_context()),
        lambda: memory.rewrite_query(
            c.RecallContext("", "query", {}, _ref().static_path)
        ),
    ):
        with pytest.raises(c.MemoryOperationError) as caught:
            operation()
        assert caught.value.code == "reflective.closed"
    opened.close()
    opened.close()


@pytest.mark.parametrize("prior", ["opened", "reset", "committed"])
def test_nonpristine_flags_survive_reopen(tmp_path, prior):
    path = tmp_path / "memory.sqlite"
    with _memory() as source:
        foreign = _snapshot(source)
    with _memory(path) as memory:
        if prior == "opened":
            _snapshot(memory)
        elif prior == "reset":
            memory.reset()
        else:
            with memory.open_run() as opened:
                opened.commit("empty", ())
    with _memory(path) as memory:
        with pytest.raises(c.MemoryOperationError) as caught:
            memory.open_run(snapshot=foreign)
        assert caught.value.code == "reflective.invalid_snapshot"
        assert getattr(memory, "_ever_" + prior)


def test_commit_tombstones_survive_removal_reset_and_stale_replay(
    monkeypatch, tmp_path
):
    path = tmp_path / "memory.sqlite"
    with _memory(path) as memory:
        old = _snapshot(memory)
        batch = _learn(monkeypatch, memory)
        _learn(monkeypatch, memory, "removed", "remove")
        expected = _snapshot(memory)
        assert expected["frames"][0]["lessons"] == []
    with _memory(path) as memory:
        with memory.open_run(snapshot=old) as stale:
            stale.commit("learned", batch)
            with pytest.raises(c.MemoryOperationError) as caught:
                stale.commit("learned", ())
            assert caught.value.code == "reflective.commit_collision"
        assert _snapshot(memory) == expected
        memory.reset()
    with _memory(path) as memory:
        with memory.open_run() as opened:
            opened.commit("learned", batch)
            opened.commit("learned", batch)
            with pytest.raises(c.MemoryOperationError) as caught:
                opened.commit("learned", ())
            assert caught.value.code == "reflective.commit_collision"
        assert _snapshot(memory)["frames"] == []
        assert memory._epoch == 1 and memory._revision == 0


def test_recorded_empty_frame_limits_survive_reopen(tmp_path):
    path = tmp_path / "memory.sqlite"
    with _memory(path, max_items=4, recall_limit=2) as memory:
        with memory.open_run() as opened:
            opened.open_frame(_spec())
            saved = opened.snapshot
    with _memory(path, max_items=99, recall_limit=88) as memory:
        with memory.open_run(snapshot=saved) as resumed:
            resumed.open_frame(_spec())
            assert resumed.snapshot == saved
            resumed.commit("declare", ())
    with _memory(path) as memory:
        assert _snapshot(memory)["frames"][0]["config"] == {
            "max_items": 4,
            "recall_limit": 2,
        }


def _ram_state(memory):
    return deepcopy(
        (
            memory._namespace,
            memory._epoch,
            memory._revision,
            memory._frames,
            memory._applied_commits,
            memory._ever_opened,
            memory._ever_reset,
            memory._ever_committed,
            memory._llm_config,
        )
    )


@pytest.mark.parametrize("operation", ["open", "adopt", "reset", "commit"])
def test_failed_save_never_publishes_candidate_ram(tmp_path, monkeypatch, operation):
    memory = _memory(tmp_path / "memory.sqlite")
    session = frame = None
    if operation == "commit":
        session = memory.open_run()
        frame = session.open_frame(_spec())
    before = _ram_state(memory)
    primary = RuntimeError("save failed")

    def reject(candidate):
        raise primary

    monkeypatch.setattr(memory._storage, "save", reject)
    try:
        with pytest.raises(RuntimeError) as caught:
            if operation == "open":
                memory.open_run()
            elif operation == "adopt":
                with _memory() as source:
                    memory.open_run(snapshot=_snapshot(source))
            elif operation == "reset":
                memory.reset()
            else:
                session.commit("commit", (_mutation(_ref(), "m", "add", "l", "text"),))
        assert caught.value is primary
        assert _ram_state(memory) == before
    finally:
        memory.close()


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_uncertain_save_poisoning_blocks_ram_reads_and_keeps_primary(
    tmp_path, monkeypatch, after_commit, cancel
):
    connect = sqlite3.connect
    primary = (
        KeyboardInterrupt("cancel")
        if cancel
        else sqlite3.OperationalError("private path")
    )

    class Connection(sqlite3.Connection):
        armed = False

        def commit(self):
            if self.armed:
                if after_commit:
                    super().commit()
                raise primary
            return super().commit()

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=Connection)
    )
    path = tmp_path / "memory.sqlite"
    memory = _memory(path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())
    before = _ram_state(memory)
    memory._storage._connection.armed = True
    try:
        with pytest.raises(BaseException) as caught:
            opened.commit("uncertain", (_mutation(_ref(), "m", "add", "l", "text"),))
        if cancel:
            assert caught.value is primary
        else:
            assert caught.value.code == "storage.io"
            assert "private path" not in str(caught.value)
        assert _ram_state(memory) == before
        for operation in (
            memory.open_run,
            memory.reset,
            lambda: opened.snapshot,
            frame.recall,
            lambda: frame.learn(_context()),
            lambda: opened.open_frame(_spec()),
            lambda: opened.commit("c", ()),
        ):
            with pytest.raises(c.MemoryOperationError) as failed:
                operation()
            assert failed.value.code == "storage.poisoned"
        with pytest.raises(c.MemoryOperationError) as busy:
            _memory(path)
        assert busy.value.code == "storage.busy"
    finally:
        opened.close()
        memory.close()
    with _memory(path) as reopened:
        assert _snapshot(reopened)["revision"] == int(after_commit)


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "version",
        "counter",
        "flag",
        "digest",
        "unopened",
        "uncommitted",
        "reset",
        "config",
        "frame_order",
        "source",
        "duplicate_lesson",
        "overflow",
        "host_config",
    ],
)
def test_invalid_inventory_releases_ownership_without_rewriting(
    tmp_path, monkeypatch, case
):
    path = tmp_path / "memory.sqlite"
    with _memory(path, max_items=1) as memory:
        _learn(monkeypatch, memory)
    storage = SQLiteStorage(path, "reflective")
    inventory = storage.load()
    state = inventory["state"]
    frame = state["frames"][0]
    if case == "extra":
        inventory["extra"] = None
    elif case == "version":
        inventory["version"] = True
    elif case == "counter":
        state["revision"] = -1
    elif case == "flag":
        inventory["ever_opened"] = 1
    elif case == "digest":
        inventory["applied_commits"] = {"c": "not-a-digest"}
    elif case == "unopened":
        inventory["ever_opened"] = False
    elif case == "uncommitted":
        inventory["ever_committed"] = False
    elif case == "reset":
        inventory["ever_reset"] = True
    elif case == "config":
        frame["config"]["max_items"] = True
    elif case == "frame_order":
        state["frames"].append(deepcopy(frame))
    elif case == "source":
        frame["lessons"][0]["source"]["frame_ref"]["static_path"] = ["other"]
    elif case == "duplicate_lesson":
        frame["config"]["max_items"] = 2
        frame["lessons"].append(deepcopy(frame["lessons"][0]))
    elif case == "overflow":
        extra = deepcopy(frame["lessons"][0])
        extra["id"] = "extra"
        frame["lessons"].append(extra)
    else:
        state["llm_config"] = _CONFIG
    storage.save(inventory)
    storage.close()
    with pytest.raises(c.MemoryOperationError) as caught:
        _memory(path)
    assert caught.value.code == "reflective.invalid_storage"
    storage = SQLiteStorage(path, "reflective")
    try:
        assert storage.load() == inventory
    finally:
        storage.close()


def _close_during_callback(path, phase, reentrant):
    entered, release = Event(), Event()
    errors = []
    memory = _memory(path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())

    def callback():
        if reentrant:
            opened.close()
            with pytest.raises(c.MemoryOperationError) as caught:
                memory.close()
            assert caught.value.code == "reflective.busy"
        else:
            entered.set()
            assert release.wait(5)

    class Model:
        def with_structured_output(self, schema):
            if phase == "structured":
                callback()
            return self

        def invoke(self, messages):
            if phase == "invoke":
                callback()
            return {"actions": [{"action": "noop"}]}

    def factory(config):
        if phase == "construction":
            callback()
        return Model()

    def rewrite(context):
        callback()
        return "query"

    def work():
        try:
            if phase == "query":
                frame.recall(c.RecallContext("", "query", {}, _ref().static_path))
            else:
                frame.learn(_context())
        except BaseException as error:
            errors.append(error)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reflective, "model_from_config", factory)
        patch.setattr(memory, "rewrite_query", rewrite)
        if reentrant:
            work()
        else:
            worker = Thread(target=work, daemon=True)
            worker.start()
            assert entered.wait(5)
            try:
                # This must finish while the model callback is still blocked.
                opened.close()
                with pytest.raises(c.MemoryOperationError) as caught:
                    memory.close()
                assert caught.value.code == "reflective.busy"
                with pytest.raises(c.MemoryOperationError) as busy:
                    _memory(path)
                assert busy.value.code == "storage.busy"
            finally:
                release.set()
                worker.join(5)
            assert not worker.is_alive()
    if phase == "query":
        assert not errors
    else:
        assert len(errors) == 1
        assert isinstance(errors[0], c.MemoryOperationError)
        assert errors[0].code == "reflective.session_closed"
    opened.close()
    memory.close()
    with _memory(path):
        pass


def _fork_while_locked(path):
    memory = _memory(path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())
    locked, release = Event(), Event()

    def hold():
        with memory._lock:
            locked.set()
            release.wait(10)

    thread = Thread(target=hold, daemon=True)
    thread.start()
    assert locked.wait(5)
    pid = os.fork()
    if pid == 0:
        try:
            for operation in (
                memory.open_run,
                memory.reset,
                memory.close,
                lambda: opened.snapshot,
                opened.close,
                frame.recall,
                lambda: frame.learn(_context()),
                lambda: opened.open_frame(_spec()),
                lambda: opened.commit("c", ()),
                lambda: memory.rewrite_query(
                    c.RecallContext("", "query", {}, _ref().static_path)
                ),
            ):
                try:
                    operation()
                except c.MemoryOperationError as error:
                    assert error.code == "storage.forked"
                else:
                    raise AssertionError("inherited handle accepted")
        except BaseException:
            os._exit(2)
        os._exit(0)
    try:
        deadline = time.monotonic() + 4
        while True:
            finished, status = os.waitpid(pid, os.WNOHANG)
            if finished:
                break
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise AssertionError("inherited handle waited on a parent lock")
            time.sleep(0.01)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        thread.join(5)
        opened.close()
        memory.close()


def _write_in_another_process(path):
    with pytest.MonkeyPatch.context() as patch:
        with _memory(path) as memory:
            _learn(patch, memory)


def _bounded(target, args):
    child = multiprocessing.get_context("spawn").Process(target=target, args=args)
    child.start()
    child.join(12)
    if child.is_alive():
        child.terminate()
        child.join(3)
        pytest.fail("owned lifecycle probe timed out")
    assert child.exitcode == 0


def test_actual_process_restart_restores_learned_lessons(tmp_path):
    path = tmp_path / "memory.sqlite"
    _bounded(_write_in_another_process, (str(path),))
    with _memory(path) as memory:
        with memory.open_run() as opened:
            assert opened.open_frame(_spec()).recall() == "[lesson] Use evidence"
            assert opened.snapshot["revision"] == 1


@pytest.mark.parametrize("phase", ["construction", "structured", "invoke", "query"])
@pytest.mark.parametrize("reentrant", [False, True])
def test_active_callback_blocks_close_without_holding_backend_lock(
    tmp_path, phase, reentrant
):
    _bounded(
        _close_during_callback, (str(tmp_path / "memory.sqlite"), phase, reentrant)
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_guard_precedes_inherited_backend_lock(tmp_path):
    _bounded(_fork_while_locked, (str(tmp_path / "memory.sqlite"),))


def _fork_inside_callback(path, phase, fail=False, managed=False):
    memory = _memory(path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())
    child_pid = None
    primary = KeyboardInterrupt("callback interrupted after fork")

    def callback():
        nonlocal child_pid
        locked, release = Event(), Event()

        def hold():
            with memory._lock:
                locked.set()
                release.wait(5)

        thread = Thread(target=hold, daemon=True)
        thread.start()
        assert locked.wait(5)
        child_pid = os.fork()
        if child_pid == 0:
            signal.signal(signal.SIGALRM, lambda *_: os._exit(91))
            signal.alarm(2)
            if fail:
                raise primary
            return
        release.set()
        thread.join(5)
        assert not thread.is_alive()

    class Model:
        inspected = False

        def with_structured_output(self, schema):
            if child_pid == 0:
                os._exit(95)
            if phase == "structured":
                callback()
            return self

        @property
        def invoke(self):
            if child_pid == 0:
                os._exit(97)
            if phase == "inspection" and not self.inspected:
                self.inspected = True
                callback()
            return self._invoke

        def _invoke(self, messages):
            if child_pid == 0:
                os._exit(96)
            if phase == "invoke":
                callback()
            return {"actions": [{"action": "noop"}]}

    def construct(config):
        if phase == "construction":
            callback()
        return Model()

    def rewrite(context):
        callback()
        return "query"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reflective, "model_from_config", construct)
        patch.setattr(memory, "rewrite_query", rewrite)
        try:
            with ExitStack() as contexts:
                if managed:
                    contexts.enter_context(memory)
                    contexts.enter_context(opened)
                if phase == "query":
                    frame.recall(c.RecallContext("", "query", {}, _ref().static_path))
                else:
                    frame.learn(_context())
        except c.MemoryOperationError as error:
            if child_pid == 0:
                os._exit(0 if not fail and error.code == "storage.forked" else 92)
            raise
        except BaseException as error:
            if child_pid == 0:
                os._exit(0 if fail and error is primary else 93)
            raise
        if child_pid == 0:
            os._exit(94)
        try:
            _, status = os.waitpid(child_pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)
            assert exit_code == 0, f"forked callback exit code {exit_code}"
        finally:
            opened.close()
            memory.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.parametrize(
    "phase", ["construction", "structured", "inspection", "query", "invoke"]
)
def test_callback_fork_rejects_before_cleanup_acquires_inherited_lock(tmp_path, phase):
    _bounded(_fork_inside_callback, (str(tmp_path / "memory.sqlite"), phase))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.parametrize(
    "phase", ["construction", "structured", "inspection", "query", "invoke"]
)
def test_callback_fork_cleanup_preserves_primary_exception(tmp_path, phase):
    _bounded(_fork_inside_callback, (str(tmp_path / "memory.sqlite"), phase, True))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.parametrize(
    "phase", ["construction", "structured", "inspection", "query", "invoke"]
)
@pytest.mark.parametrize("fail", [False, True])
def test_callback_fork_contexts_preserve_primary_exception(tmp_path, phase, fail):
    _bounded(
        _fork_inside_callback,
        (str(tmp_path / "memory.sqlite"), phase, fail, True),
    )


@pytest.mark.parametrize("raises", [False, True])
def test_structured_result_inspection_preserves_error_translation(
    tmp_path, monkeypatch, raises
):
    primary = RuntimeError("private inspection failure")

    class Model:
        def with_structured_output(self, schema):
            return self

        @property
        def invoke(self):
            if raises:
                raise primary
            return None

    monkeypatch.setattr(reflective, "model_from_config", lambda _: Model())
    with _memory(tmp_path / "memory.sqlite") as memory:
        with memory.open_run() as opened:
            with pytest.raises(c.MemoryOperationError) as caught:
                opened.open_frame(_spec()).learn(_context())
    assert caught.value.code == "reflective.structured_output"
    assert caught.value.retryable is False
    if raises:
        assert caught.value.__cause__ is primary
        assert "private inspection failure" not in str(caught.value)
    else:
        assert type(caught.value.__cause__) is TypeError


@pytest.mark.parametrize("run_context", [False, True])
def test_context_cleanup_preserves_primary_exception(
    tmp_path, monkeypatch, run_context
):
    memory = _memory(tmp_path / "memory.sqlite")
    managed = memory.open_run() if run_context else memory
    close = managed.close
    primary = KeyboardInterrupt("primary")

    def reject():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(managed, "close", reject)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            with managed:
                raise primary
        assert caught.value is primary
        with pytest.raises(RuntimeError, match="cleanup failed"):
            with managed:
                pass
    finally:
        close()
        if run_context:
            memory.close()


def test_constructor_cleanup_preserves_cancellation_and_releases_owner(
    tmp_path, monkeypatch
):
    path = tmp_path / "memory.sqlite"
    _memory(path).close()
    primary = KeyboardInterrupt("initialization stopped")
    parse, close = reflective._parse_inventory, SQLiteStorage.close

    def reject(value):
        raise primary

    def cleanup(self):
        close(self)
        raise RuntimeError("cleanup problem")

    monkeypatch.setattr(reflective, "_parse_inventory", reject)
    monkeypatch.setattr(SQLiteStorage, "close", cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        _memory(path)
    assert caught.value is primary
    monkeypatch.setattr(reflective, "_parse_inventory", parse)
    monkeypatch.setattr(SQLiteStorage, "close", close)
    with _memory(path):
        pass
