"""Versioned compare-and-set markdown profile store."""

from __future__ import annotations

import json
import threading

import pytest

from agent_composer.memory.profile_store import (
    FileProfileStore,
    ProfileStore,
    ProfileVersionMismatch,
    profile_version,
)


def test_missing_file_reads_as_empty_document(tmp_path) -> None:
    store = FileProfileStore(tmp_path / "ada.md")
    assert store.read() == ("", profile_version(""))
    assert not (tmp_path / "ada.md").exists()


def test_unknown_or_malformed_version_raises_lookup_error(tmp_path) -> None:
    store = FileProfileStore(tmp_path / "ada.md")
    for version in ("sha256:" + "0" * 64, "sha256:../../etc", "md5:abc"):
        with pytest.raises(LookupError):
            store.read(version)


def test_missing_parent_directory_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        FileProfileStore(tmp_path / "missing" / "ada.md")
