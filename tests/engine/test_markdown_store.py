"""Versioned compare-and-set markdown store."""

from __future__ import annotations

import json
import threading

import pytest

from agent_composer.memory.storage import (
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
