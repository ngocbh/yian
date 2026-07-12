"""Unit tests for the `BuildContext`/`BuildResult` build-side dataclasses.

These are the immutable carriers a node `.build()` will consume/return once wired
in a later commit; here we only pin their defaults and frozen-ness."""

import dataclasses

import pytest

from agent_composer.compose.nodes.base import BuildContext, BuildResult


def test_build_context_defaults():
    """BuildContext requires only `registry`; every other field defaults."""
    ctx = BuildContext(registry={})
    assert ctx.resolver is None
    assert ctx.source_file == "<inline-code>"
    assert ctx.code_line is None
    assert ctx.code_field_line is None


def test_build_result_defaults():
    """BuildResult requires `node` + `wiring`; `edges` defaults to the empty tuple."""
    result = BuildResult(node=object(), wiring={"x": "y"})
    assert result.edges == ()


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
