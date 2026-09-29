"""In-loop AGENT observer payloads are frozen, positional, and default-free of detail."""

import dataclasses

import pytest

from agent_composer.events import (
    IN_LOOP_EVENTS,
    ModelCallFinished,
    ModelCallStarted,
    TokenUsage,
    ToolCallFinished,
    ToolCallStarted,
)


def test_in_loop_observer_shapes_and_defaults():
    usage = TokenUsage(3, 2, 5)
    assert (usage.cache_read_tokens, usage.cache_creation_tokens, usage.reasoning_tokens) == (
        None, None, None,
    )
    finished = ModelCallFinished("n", "x", "chat", "anthropic", "m", usage=usage)
    assert finished.error is None
    assert ModelCallStarted("n", "x", "chat").provider is None
    assert ToolCallStarted("n", "x", 0, "call-1", "lookup").arguments is None
    tool = ToolCallFinished("n", "x", 0, "call-1", "lookup", result="ok")
    assert tool.error is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        finished.usage = None


def test_in_loop_events_names_exactly_the_four_observers():
    assert IN_LOOP_EVENTS == (ModelCallStarted, ModelCallFinished, ToolCallStarted, ToolCallFinished)
