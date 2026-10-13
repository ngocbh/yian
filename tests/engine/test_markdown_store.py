"""Versioned compare-and-set markdown store."""

from __future__ import annotations

import datetime
import json
import math
import threading

import pytest

from yian.memory.contracts import MemoryOperationError
from yian.memory.storage import (
    FileMarkdownStore,
    InMemoryMarkdownStore,
    MarkdownStore,
    MarkdownVersionMismatch,
    markdown_version,
)


def test_missing_file_reads_as_empty_document(tmp_path) -> None:
    store = FileMarkdownStore(tmp_path / "ada.md")
    assert store.read() == ("", markdown_version(""))
    assert not (tmp_path / "ada.md").exists()


def test_unknown_or_malformed_version_raises_lookup_error(tmp_path) -> None:
    store = FileMarkdownStore(tmp_path / "ada.md")
    for version in ("sha256:" + "0" * 64, "sha256:../../etc", "md5:abc"):
        with pytest.raises(LookupError):
            store.read(version)


def test_missing_parent_directory_fails_on_first_use(tmp_path) -> None:
    store = FileMarkdownStore(tmp_path / "missing" / "ada.md")
    assert not (tmp_path / "missing").exists()
    with pytest.raises(FileNotFoundError, match="document directory does not exist"):
        store.read()
    with pytest.raises(FileNotFoundError):
        store.write("x", base_version=markdown_version(""))
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    "path", ["", ":memory:", "file:ada.md", "FILE:ada.md", "ada\x00.md", b"ada.md", "\udcff.md"]
)
def test_non_file_paths_are_rejected_at_construction(tmp_path, monkeypatch, path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(MemoryOperationError) as caught:
        FileMarkdownStore(path)
    assert caught.value.code == "storage.path"
    assert list(tmp_path.iterdir()) == []


def test_directory_path_fails_on_first_use(tmp_path) -> None:
    (tmp_path / "ada.md").mkdir()
    store = FileMarkdownStore(tmp_path / "ada.md")
    with pytest.raises(OSError):
        store.read()


def test_create_parent_makes_the_directory_on_first_use(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "ada.md"
    store = FileMarkdownStore(path, create_parent=True)
    assert not (tmp_path / "a").exists()
    assert store.read() == ("", markdown_version(""))
    assert path.parent.is_dir()


def test_relative_path_is_bound_to_the_construction_directory(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    store = FileMarkdownStore("ada.md")
    assert list((tmp_path / "a").iterdir()) == []
    monkeypatch.chdir(tmp_path / "b")
    store.write("hello", base_version=markdown_version(""))
    assert (tmp_path / "a" / "ada.md").read_text(encoding="utf-8") == "hello"
    assert not (tmp_path / "b" / "ada.md").exists()


def test_symlinked_path_writes_through_to_its_target(tmp_path) -> None:
    (tmp_path / "repo").mkdir()
    (tmp_path / "work").mkdir()
    target = tmp_path / "repo" / "ada.md"
    target.write_text("# Ada\n", encoding="utf-8")
    link = tmp_path / "work" / "ada.md"
    link.symlink_to(target)
    store = FileMarkdownStore(link)
    first = store.write("# Ada\nLikes tea.\n", base_version=store.read()[1])
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "# Ada\nLikes tea.\n"
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["ada.md"]
    assert (tmp_path / "repo" / ".ada.md.writes.jsonl").is_file()
    target.write_text("# Ada\nEdited by hand.\n", encoding="utf-8")
    with pytest.raises(MarkdownVersionMismatch):
        FileMarkdownStore(target).write("lost update\n", base_version=first)
    assert store.read()[0] == "# Ada\nEdited by hand.\n"


def test_symlinked_parent_follows_the_link_before_parent_steps(tmp_path) -> None:
    (tmp_path / "real" / "sub").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "real" / "sub")
    store = FileMarkdownStore(tmp_path / "link" / ".." / "ada.md")
    store.write("hello", base_version=markdown_version(""))
    assert (tmp_path / "real" / "ada.md").read_text(encoding="utf-8") == "hello"
    assert (tmp_path / "real" / ".ada.md.lock").is_file()
    assert not (tmp_path / "ada.md").exists()


def test_dangling_symlink_creates_its_target(tmp_path) -> None:
    link = tmp_path / "ada.md"
    link.symlink_to(tmp_path / "notes.md")
    store = FileMarkdownStore(link)
    assert store.read() == ("", markdown_version(""))
    store.write("hello", base_version=markdown_version(""))
    assert link.is_symlink()
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "hello"


def test_write_is_compare_and_set_and_keeps_every_version(tmp_path) -> None:
    store = FileMarkdownStore(tmp_path / "ada.md")
    assert isinstance(store, MarkdownStore)
    _, empty = store.read()
    first = store.write("# Ada\n", base_version=empty)
    assert first == markdown_version("# Ada\n")
    assert (tmp_path / "ada.md").read_text() == "# Ada\n"
    second = store.write("# Ada\nPrefers email.\n", base_version=first)
    with pytest.raises(MarkdownVersionMismatch) as caught:
        store.write("lost update\n", base_version=first)
    assert (caught.value.expected, caught.value.current) == (first, second)
    assert store.read(first) == ("# Ada\n", first)
    assert store.read(empty) == ("", empty)
    assert store.read() == ("# Ada\nPrefers email.\n", second)


def test_direct_file_edit_is_a_new_version_and_breaks_stale_writes(tmp_path) -> None:
    path = tmp_path / "ada.md"
    store = FileMarkdownStore(path)
    base = store.write("# Ada\n", base_version=store.read()[1])
    path.write_text("# Ada\nEdited by hand.\n")
    with pytest.raises(MarkdownVersionMismatch):
        store.write("# Ada\nFrom a model.\n", base_version=base)
    text, version = store.read()
    assert text == "# Ada\nEdited by hand.\n" and version == markdown_version(text)
    path.write_text("# Ada\nEdited again.\n")
    assert store.read(version) == ("# Ada\nEdited by hand.\n", version)


@pytest.mark.parametrize("kind", ["file", "memory"])
def test_carriage_returns_round_trip_exactly(tmp_path, kind) -> None:
    store = FileMarkdownStore(tmp_path / "ada.md") if kind == "file" else InMemoryMarkdownStore()
    base = store.read()[1]
    for text in ("# Ada\r\nPrefers email.\r\n", "# Ada\rOld Mac line.\n"):
        version = store.write(text, base_version=base)
        assert version == markdown_version(text)
        assert store.read() == (text, version)
        base = version
    first = markdown_version("# Ada\r\nPrefers email.\r\n")
    assert store.read(first) == ("# Ada\r\nPrefers email.\r\n", first)


def test_hand_edited_crlf_file_reads_back_byte_for_byte(tmp_path) -> None:
    path = tmp_path / "ada.md"
    path.write_bytes(b"# Ada\r\nEdited on Windows.\r\n")
    store = FileMarkdownStore(path)
    text, version = store.read()
    assert text == "# Ada\r\nEdited on Windows.\r\n" and version == markdown_version(text)
    path.write_bytes(b"# Ada\n")
    assert store.read(version) == (text, version)


def test_every_write_appends_its_provenance_and_nothing_is_replayed(tmp_path) -> None:
    store = FileMarkdownStore(tmp_path / "ada.md")
    base = store.read()[1]
    provenance = {"commit_id": "c1", "run_id": "r1"}
    first = store.write("# Ada\n", base_version=base, provenance=provenance)
    with pytest.raises(MarkdownVersionMismatch):
        store.write("# Ada\n", base_version=base, provenance=provenance)
    store.write("# Ada\nMore.\n", base_version=first, provenance={"commit_id": "c2"})
    log = [json.loads(line) for line in (tmp_path / ".ada.md.writes.jsonl").read_text().splitlines()]
    assert [entry["provenance"]["commit_id"] for entry in log] == ["c1", "c2"]
    assert (log[0]["base_version"], log[0]["version"]) == (base, first)


@pytest.mark.parametrize(
    ("provenance", "error"),
    [
        ({"at": datetime.datetime(2026, 10, 13)}, TypeError),
        ({"score": math.nan}, ValueError),
        ({"tags": ("a",)}, TypeError),
        ({1: "x"}, TypeError),
        ([], TypeError),
    ],
)
def test_non_json_provenance_is_rejected_before_anything_is_written(
    tmp_path, provenance, error
) -> None:
    path = tmp_path / "ada.md"
    store = FileMarkdownStore(path)
    base = store.write("# Ada\n", base_version=store.read()[1], provenance={"commit_id": "c1"})
    log = tmp_path / ".ada.md.writes.jsonl"
    before = log.read_text()
    with pytest.raises(error, match="provenance"):
        store.write("# Ada\nEdited.\n", base_version=base, provenance=provenance)
    assert path.read_text() == "# Ada\n"
    assert log.read_text() == before
    assert not (tmp_path / ".ada.md.versions" / (markdown_version("# Ada\nEdited.\n")[7:] + ".md")).exists()


def test_concurrent_writers_from_one_base_have_exactly_one_winner(tmp_path) -> None:
    store = FileMarkdownStore(tmp_path / "ada.md")
    base = store.read()[1]
    results = []

    def attempt(index: int) -> None:
        try:
            results.append(store.write(f"writer {index}\n", base_version=base))
        except MarkdownVersionMismatch:
            results.append(None)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    winners = [version for version in results if version is not None]
    assert len(winners) == 1 and store.read()[1] == winners[0]


def test_in_memory_store_is_a_compare_and_set_markdown_store() -> None:
    store = InMemoryMarkdownStore()
    assert isinstance(store, MarkdownStore)
    text, empty = store.read()
    assert (text, empty) == ("", markdown_version(""))
    first = store.write("# Ada\n", base_version=empty, provenance={"run": "r1"})
    assert store.read() == ("# Ada\n", first)
    with pytest.raises(MarkdownVersionMismatch) as caught:
        store.write("lost\n", base_version=empty)
    assert (caught.value.expected, caught.value.current) == (empty, first)
    assert store.read(empty) == ("", empty)
    with pytest.raises(LookupError):
        store.read("sha256:" + "0" * 64)


def test_in_memory_store_accepts_initial_text_and_checks_types() -> None:
    store = InMemoryMarkdownStore("seed\n")
    current = markdown_version("seed\n")
    assert store.read() == ("seed\n", current)
    with pytest.raises(TypeError):
        store.write(b"x", base_version=current)
    with pytest.raises(TypeError):
        store.write("x", base_version=None)
    with pytest.raises(TypeError):
        InMemoryMarkdownStore(None)
    assert store.read() == ("seed\n", current)
