"""Unit tests for the `BuildContext`/`BuildResult` build-side dataclasses.

These are the immutable carriers a node `.build()` will consume/return once wired
in a later commit; here we only pin their defaults and frozen-ness."""

import dataclasses

import pytest

from agent_composer.compose.nodes.base import BuildContext, BuildResult
from agent_composer.diagnostics import SourceRange


def test_build_context_defaults():
    """BuildContext requires only `registry`; every other field defaults."""
    ctx = BuildContext(registry={})
    assert ctx.resolver is None
    assert ctx.source_file == "<inline-code>"
    assert ctx.resolve_origin is None
    assert ctx.output_contract_reader is None


def test_build_result_defaults():
    """BuildResult defaults optional graph and Compose metadata to empty values."""
    result = BuildResult(node=object(), wiring={"x": "y"})
    assert result.edges == ()
    assert result.node_range is None
    assert result.field_ranges == {}


def test_build_result_range_metadata_does_not_affect_equality():
    """Source coordinates are not part of a build product's semantic identity."""
    node = object()
    left = BuildResult(
        node=node,
        wiring={},
        node_range=SourceRange(1, 1, 1, 2),
        field_ranges={("field", "code"): SourceRange(2, 1, 2, 5)},
    )
    right = BuildResult(node=node, wiring={})

    assert left == right


def test_build_context_frozen():
    """BuildContext is immutable — attribute assignment raises FrozenInstanceError."""
    ctx = BuildContext(registry={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.source_file = "other"


def test_build_result_frozen():
    """BuildResult is immutable — attribute assignment raises FrozenInstanceError."""
    result = BuildResult(node=object(), wiring={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.node = object()
