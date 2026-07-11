"""A span-recording YAML loader: construct values AND record source ranges in ONE parse.

The compose front end needs both the constructed Python value of a flow document and
the source range of every yaml node (for diagnostics). Rather than double-parse (once
with `yaml.safe_load` for values, again with `yaml.compose` for line marks), this
module subclasses `yaml.SafeLoader` so a single parse yields both: SafeConstructor
builds the value (anchors, `<<:` merge, scalar coercion), and the overrides here record
each yaml *node*'s SourceRange keyed by `id(node)`.
"""

from __future__ import annotations

import yaml

from agent_composer.events import SourceRange


def _mark_to_range(start_mark, end_mark) -> SourceRange:
    """Convert a PyYAML (start_mark, end_mark) pair to a 1-based half-open SourceRange.

    PyYAML marks are 0-based (`mark.line`, `mark.column`); SourceRange is 1-based, so
    each coordinate is offset by +1. PyYAML's `end_mark` already points one-past the
    last char, matching SourceRange's exclusive (half-open) end, so end maps directly
    after the same +1 shift.
    """
    return SourceRange(
        start_line=start_mark.line + 1,
        start_col=start_mark.column + 1,
        end_line=end_mark.line + 1,
        end_col=end_mark.column + 1,
    )
