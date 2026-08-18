"""Pure helper functions for agent modes (no shared state — that's in `common`)."""

from __future__ import annotations

from typing import Any


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
