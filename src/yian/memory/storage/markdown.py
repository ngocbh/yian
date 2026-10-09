"""Versioned markdown document stores.

A markdown store holds one current markdown document and can return any version it
has handed out. Writes are compare-and-set against a base version so that a
concurrent human edit is never silently overwritten.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Iterator, Protocol, runtime_checkable

from yian.memory.contracts import JsonObject
from yian.memory.storage.sqlite import check_storage_path


class MarkdownVersionMismatch(Exception):
    """A compare-and-set write named a base version that is no longer current."""

    def __init__(self, expected: str, current: str) -> None:
        """Record the stale base version and the version that is current.

        Args:
            expected (`str`): Base version supplied by the writer.
            current (`str`): Version found in the store.
        """
        self.expected = expected
        self.current = current
        super().__init__(f"base version {expected!r} is not current ({current!r})")


@runtime_checkable
class MarkdownStore(Protocol):
    """Versioned storage for one markdown document.

    Implementations raise `LookupError` for an unknown version,
    `MarkdownVersionMismatch` for a stale base version, `ValueError` for stored
    text they cannot decode, and `OSError` when storage cannot be reached.
    Memory backends such as `ProfileMemory` turn these into typed memory
    failures. Any other exception escapes the backward attempt.
    """

    def read(self, version: str | None = None) -> tuple[str, str]:
        """Return ``(text, version)`` for the current or a named version.

        Args:
            version (`str | None`, *optional*, defaults to `None`): A version this
                store returned earlier, or `None` for the current document.

        Returns:
            `tuple[str, str]`: The document text and its version.

        Raises:
            `LookupError`: The named version is unknown.
        """
        ...

    def write(
        self,
        text: str,
        *,
        base_version: str,
        provenance: JsonObject | None = None,
    ) -> str:
        """Replace the current document if ``base_version`` is still current.

        ``provenance`` is caller-defined; from `ProfileMemory` it holds ``run_id``, ``commit_id``,
        ``seen_version``, and ``feedback``. ``commit_id`` is unique per
        backward attempt and memory, so a store that retries its own transport
        may use it to avoid writing twice.

        Args:
            text (`str`): Complete new document.
            base_version (`str`): Version the writer read before editing.
            provenance (`JsonObject | None`, *optional*, defaults to `None`):
                JSON describing who wrote and why, for example for a commit
                message.

        Returns:
            `str`: The version of the stored document.

        Raises:
            `MarkdownVersionMismatch`: ``base_version`` is not current. Nothing
                was written.
            `OSError`: Storage could not be reached. The write may or may not
                have landed.
        """
        ...


def markdown_version(text: str) -> str:
    """Return the content-addressed version of one document.

    Args:
        text (`str`): Document text.

    Returns:
        `str`: ``"sha256:<hex>"`` of the UTF-8 bytes.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class FileMarkdownStore:
    """Markdown store backed by one editable file.

    The current document is ``path``; a missing file reads as an empty document.
    Every version this store returns or writes is kept under
    ``.<name>.versions/`` next to it, so recorded runs can reopen it later.
    Each write appends its versions and provenance to ``.<name>.writes.jsonl``.
    Writers on one host serialize through ``.<name>.lock``. Nothing on disk is
    touched until the first read or write. A person may edit ``path`` directly;
    the next compare-and-set write then reports a mismatch.
    """

    def __init__(self, path: str | os.PathLike[str], *, create_parent: bool = False) -> None:
        """Bind the store to one document path without touching the filesystem.

        Args:
            path (`str | os.PathLike[str]`): Markdown file.
            create_parent (`bool`, *optional*, defaults to `False`): Create the
                parent directory on first use. Otherwise a missing parent fails
                on first use with `FileNotFoundError`.

        Raises:
            `TypeError`: ``create_parent`` is not a bool.
            `MemoryOperationError`: ``path`` is not a local file path: empty,
                ``:memory:``, a ``file:`` URI, bytes, or text with NUL or
                non-UTF-8 characters (`storage.path`). A directory is detected
                on first use.
        """
        if type(create_parent) is not bool:
            raise TypeError("create_parent must be a bool")
        # Bind a relative path to the current directory now so a later chdir
        # does not redirect the store; ".." stays for the OS to resolve.
        self.path = Path(check_storage_path(path))
        self._create_parent = create_parent
        self._versions = self.path.parent / f".{self.path.name}.versions"
        self._lock_path = self.path.parent / f".{self.path.name}.lock"
        self._writes = self.path.parent / f".{self.path.name}.writes.jsonl"

    def read(self, version: str | None = None) -> tuple[str, str]:
        """Return ``(text, version)`` for the current or a named version.

        Args:
            version (`str | None`, *optional*, defaults to `None`): A version this
                store returned earlier, or `None` for the current document.

        Returns:
            `tuple[str, str]`: The document text and its version.

        Raises:
            `LookupError`: The named version is unknown.
            `ValueError`: The file is not valid UTF-8.
            `FileNotFoundError`: The parent directory does not exist and
                `create_parent` is off.
        """
        with self._locked():
            if version is None:
                text = self._current()
                current = markdown_version(text)
                self._keep(current, text)
                return text, current
            text = self._current()
            if markdown_version(text) == version:
                self._keep(version, text)
                return text, version
            kept = self._version_path(version)
            if kept is None or not kept.is_file():
                raise LookupError(f"unknown document version {version!r}")
            return kept.read_text(encoding="utf-8"), version

    def write(
        self,
        text: str,
        *,
        base_version: str,
        provenance: JsonObject | None = None,
    ) -> str:
        """Replace the current document if ``base_version`` is still current.

        Args:
            text (`str`): Complete new document.
            base_version (`str`): Version the writer read before editing.
            provenance (`JsonObject | None`, *optional*, defaults to `None`):
                JSON appended with the versions to ``.<name>.writes.jsonl``.

        Returns:
            `str`: The version of the stored document.

        Raises:
            `TypeError`: ``text`` or ``base_version`` is not a string.
            `ValueError`: The current file is not valid UTF-8.
            `MarkdownVersionMismatch`: ``base_version`` is not current.
            `FileNotFoundError`: The parent directory does not exist and
                `create_parent` is off.
        """
        if type(text) is not str or type(base_version) is not str:
            raise TypeError("text and base_version must be strings")
        new_version = markdown_version(text)
        with self._locked():
            current = markdown_version(self._current())
            if current != base_version:
                raise MarkdownVersionMismatch(base_version, current)
            self._keep(new_version, text)
            self._replace(self.path, text)
            with self._writes.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"base_version": base_version, "version": new_version,
                     "provenance": provenance},
                    sort_keys=True,
                ) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return new_version

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the host-wide exclusive lock for this document."""
        if self._create_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.parent.is_dir():
            raise FileNotFoundError(
                f"document directory does not exist: {self.path.parent}"
            )
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _current(self) -> str:
        """Return the current file text, or an empty document when missing."""
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _version_path(self, version: str) -> Path | None:
        """Return the kept-copy path for a well-formed version, else `None`."""
        prefix, _, digest = version.partition(":")
        if prefix != "sha256" or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            return None
        return self._versions / f"{digest}.md"

    def _keep(self, version: str, text: str) -> None:
        """Store one content-addressed copy unless it already exists."""
        kept = self._version_path(version)
        if kept is not None and not kept.exists():
            self._versions.mkdir(exist_ok=True)
            self._replace(kept, text)

    @staticmethod
    def _replace(target: Path, text: str) -> None:
        """Write ``text`` to ``target`` through a synced temporary file."""
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise


class InMemoryMarkdownStore:
    """Process-local markdown store with the same compare-and-set contract.

    It keeps every version it has returned or written, so recorded runs in the
    same process can reopen them. Provenance is accepted and discarded.
    """

    def __init__(self, text: str = "") -> None:
        """Start with ``text`` as the current document.

        Args:
            text (`str`, *optional*, defaults to `""`): Initial document.

        Raises:
            `TypeError`: ``text`` is not a string.
        """
        if type(text) is not str:
            raise TypeError("text must be a string")
        self._lock = Lock()
        self._current = text
        self._versions = {markdown_version(text): text}

    def read(self, version: str | None = None) -> tuple[str, str]:
        """Return ``(text, version)`` for the current or a kept version.

        Args:
            version (`str | None`, *optional*, defaults to `None`): A version this
                store returned earlier, or `None` for the current document.

        Returns:
            `tuple[str, str]`: The document text and its version.

        Raises:
            `LookupError`: The named version is unknown.
        """
        with self._lock:
            if version is None:
                return self._current, markdown_version(self._current)
            if version not in self._versions:
                raise LookupError(f"unknown document version {version!r}")
            return self._versions[version], version

    def write(
        self,
        text: str,
        *,
        base_version: str,
        provenance: JsonObject | None = None,
    ) -> str:
        """Replace the document if ``base_version`` is still current.

        Args:
            text (`str`): Complete new document.
            base_version (`str`): Version the writer read before editing.
            provenance (`JsonObject | None`, *optional*, defaults to `None`):
                Accepted for the protocol and discarded.

        Returns:
            `str`: The version of the stored document.

        Raises:
            `TypeError`: ``text`` or ``base_version`` is not a string.
            `MarkdownVersionMismatch`: The current version is not ``base_version``.
        """
        if type(text) is not str or type(base_version) is not str:
            raise TypeError("text and base_version must be strings")
        with self._lock:
            current = markdown_version(self._current)
            if base_version != current:
                raise MarkdownVersionMismatch(base_version, current)
            self._current = text
            version = markdown_version(text)
            self._versions[version] = text
            return version
