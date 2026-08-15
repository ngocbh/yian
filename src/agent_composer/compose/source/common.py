"""Shared contracts for source documents and their occurrence cursors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import yaml

from agent_composer.events import SourceRange


@dataclass(frozen=True, slots=True)
class _MapStep:
    """One mapping-pair step in an occurrence path.

    ``constructed_key`` is the SafeLoader-produced key and ``pair_ordinal`` is
    its zero-based position in the YAML mapping, including duplicate pairs.
    """

    constructed_key: Any
    pair_ordinal: int


@dataclass(frozen=True, slots=True)
class _SeqStep:
    """One zero-based sequence-item step in an occurrence path."""

    index: int


OccurrencePath: TypeAlias = tuple[_MapStep | _SeqStep, ...]


@dataclass(frozen=True, slots=True)
class _Occurrence:
    """One retained YAML value occurrence indexed by its authored path.

    ``value`` is the SafeLoader-constructed value for this occurrence,
    ``yaml_node`` supplies its value range, and ``key_node`` supplies the range
    of the mapping key that selected it (or ``None`` outside a mapping pair).
    """

    value: Any = field(repr=False, compare=False)
    yaml_node: yaml.Node = field(repr=False, compare=False)
    key_node: yaml.Node | None = field(default=None, repr=False, compare=False)


class SourceAccessError(ValueError):
    """A source cursor cannot satisfy a requested navigation or typed access.

    ``range`` is the closest concrete source location, or ``None`` for a
    virtual or otherwise unlocated value.
    """

    def __init__(self, message: str, range: SourceRange | None = None):
        super().__init__(message)
        self.range = range


class SourceSyntaxError(ValueError):
    """A YAML source document cannot be composed or constructed safely.

    ``range`` identifies PyYAML's problem mark when one is available; otherwise
    it is ``None``.
    """

    def __init__(self, message: str, range: SourceRange | None = None):
        super().__init__(message)
        self.range = range
