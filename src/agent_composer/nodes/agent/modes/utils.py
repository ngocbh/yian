"""Text extraction and observational trace helpers shared by agent modes."""

from __future__ import annotations

import json
from typing import Any

from agent_composer.memory.contracts import (
    EvaluationTrace,
    MemorySection,
    ModelTextInteraction,
    ToolResultInteraction,
)


def system_prompt_with_memory(
    base_system_prompt: str,
    memory_sections: tuple[MemorySection, ...],
) -> str:
    """Frame opaque policy text as labelled outer-to-local system sections."""
    if not memory_sections:
        return base_system_prompt
    payload = {
        "order": "outer-to-local",
        "sections": [
            {
                "index": index,
                "owner": section.frame_ref.owner.value.upper(),
                "static_path": section.frame_ref.static_path,
                "text": section.text,
            }
            for index, section in enumerate(memory_sections, start=1)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        base_system_prompt
        + "\n\nMemory context (canonical JSON; sections are ordered from outer "
        + "workflow to local agent):\n"
        + encoded
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
