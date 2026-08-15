"""One-pass parsing and private retention of a Compose YAML document."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from agent_composer.compose.source.common import (
    OccurrencePath,
    SourceSyntaxError,
    _Occurrence,
)
from agent_composer.events import SourceRange


class _DocumentLoader(yaml.SafeLoader):
    """The private SafeLoader instance that composes and constructs one document."""

    def __init__(self, stream: str):
        super().__init__(stream)


def _offset_to_line_column(text: str, offset: int) -> tuple[int, int]:
    """Convert a string offset to 1-based coordinates using YAML line breaks."""

    line = 1
    column = 1
    index = 0
    while index < offset:
        character = text[index]
        if character == "\r":
            if index + 1 < offset and text[index + 1] == "\n":
                index += 2
            else:
                index += 1
            line += 1
            column = 1
        elif character in ("\n", "\x85", "\u2028", "\u2029"):
            index += 1
            line += 1
            column = 1
        else:
            index += 1
            column += 1
    return line, column


def _problem_range(
    error: yaml.YAMLError | yaml.reader.ReaderError,
    text: str,
) -> SourceRange | None:
    """Return a one-character range at PyYAML's best available problem mark."""

    if isinstance(error, yaml.reader.ReaderError):
        line, column = _offset_to_line_column(text, error.position)
        return SourceRange(line, column, line, column + 1)

    mark = getattr(error, "problem_mark", None)
    if mark is None:
        mark = getattr(error, "context_mark", None)
    if mark is None:
        return None
    line = mark.line + 1
    column = mark.column + 1
    return SourceRange(line, column, line, column + 1)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One safely constructed YAML document with its source graph kept private.

    ``text`` is the original YAML and ``label`` identifies it to later
    diagnostics.  The constructed value, composed node tree, and occurrence
    index remain private; navigation is introduced through source cursors.
    """

    text: str
    label: str
    _value: Any = field(repr=False, compare=False)
    _root_node: yaml.Node | None = field(repr=False, compare=False)
    _index: Mapping[OccurrencePath, _Occurrence] = field(
        repr=False,
        compare=False,
    )

    @classmethod
    def parse(cls, text: str, label: str = "<flow>") -> "SourceDocument":
        """Compose and construct exactly one YAML document with one loader."""

        loader = None
        try:
            loader = _DocumentLoader(text)
            root_node = loader.get_single_node()
            value = (
                loader.construct_document(root_node)
                if root_node is not None
                else None
            )
        except (yaml.YAMLError, yaml.reader.ReaderError) as exc:
            raise SourceSyntaxError(str(exc), _problem_range(exc, text)) from exc
        finally:
            if loader is not None:
                loader.dispose()

        return cls(
            text=text,
            label=label,
            _value=value,
            _root_node=root_node,
            _index=MappingProxyType({}),
        )
