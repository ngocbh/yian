"""Pure helper functions for agent modes (no shared state — that's in `common`)."""

from __future__ import annotations

from typing import Any

from agent_composer.memory.contracts import (
    EvaluationTrace,
    ModelTextInteraction,
    ToolResultInteraction,
)


def text_of(message: Any) -> str:
    """Return only model content explicitly represented as visible text."""
    content = message if isinstance(message, str) else getattr(message, "content", None)
    if isinstance(content, str):
        return str(content)
    if isinstance(content, list):
        visible: list[str] = []
        for part in content:
            if type(part) is not dict or part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                visible.append(str(text))
        return "\n".join(visible)
    return ""


def record_model_text(
    trace: EvaluationTrace | None,
    text: str,
) -> None:
    """Record encodable visible model text without affecting node behavior."""
    if trace is None:
        return
    try:
        interaction = ModelTextInteraction(text)
    except (TypeError, ValueError):
        return
    trace.record_interaction(interaction)


def record_tool_result(
    trace: EvaluationTrace | None,
    name: str,
    text: str,
) -> None:
    """Record an encodable visible tool result without affecting its invocation."""
    if trace is None:
        return
    try:
        interaction = ToolResultInteraction(name, text)
    except (TypeError, ValueError):
        return
    trace.record_interaction(interaction)
