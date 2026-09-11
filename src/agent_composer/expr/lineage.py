"""Ordinary feedback values and source-relative structural validation.

Feedback addresses recorded values without depending on nodes, memory, or the
runtime. A list is one positional selection; callers use tuples to group
independent messages.
"""

from __future__ import annotations

from typing import Any, TypeAlias


FeedbackValue: TypeAlias = (
    str | dict[str, "FeedbackValue"] | list["FeedbackValue"] | None
)


def validate_feedback(
    feedback: FeedbackValue,
    value: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> None:
    """Require feedback to select the exact shape of a recorded value.

    Args:
        feedback (`FeedbackValue`): A string, absent feedback, a partial field
            selection, or a positional list with the recorded list's length.
        value (`Any`): The recorded JSON-like value addressed by the feedback.
        path (`tuple[str | int, ...]`, *optional*, defaults to `()`):
            Address of this value, used to locate validation failures.

    Raises:
        `TypeError`: Raised for an unsupported feedback value or non-string key.
        `ValueError`: Raised for an unknown field or a mismatched container shape.
    """
    if feedback is None or type(feedback) is str:
        return

    location = "feedback" + "".join(f"[{part!r}]" for part in path)
    if type(feedback) is dict:
        if type(value) is not dict:
            raise ValueError(f"{location} requires a recorded object")
        for name, message in feedback.items():
            if type(name) is not str:
                raise TypeError(f"{location} object keys must be strings")
            if name not in value:
                raise ValueError(f"{location} selects unknown field {name!r}")
            validate_feedback(message, value[name], path=(*path, name))
        return

    if type(feedback) is list:
        if type(value) is not list:
            raise ValueError(f"{location} requires a recorded list")
        if len(feedback) != len(value):
            raise ValueError(
                f"{location} must have exactly {len(value)} items; "
                f"got {len(feedback)}"
            )
        for index, (message, item) in enumerate(zip(feedback, value)):
            validate_feedback(message, item, path=(*path, index))
        return

    raise TypeError(f"{location} must be a string, object, list, or None")


def has_feedback(feedback: FeedbackValue) -> bool:
    """Return whether validated feedback contains any supplied string.

    Args:
        feedback (`FeedbackValue`): One ordinary feedback value. Its container
            shape must already have been validated against the recorded value.

    Returns:
        `bool`: Whether a string occurs, including the empty string.

    Raises:
        `TypeError`: Raised when a visited value has an unsupported type.
    """
    if feedback is None:
        return False
    if type(feedback) is str:
        return True
    if type(feedback) is dict:
        return any(has_feedback(message) for message in feedback.values())
    if type(feedback) is list:
        return any(has_feedback(message) for message in feedback)
    raise TypeError("feedback must be a string, object, list, or None")


__all__ = ["FeedbackValue", "has_feedback", "validate_feedback"]
