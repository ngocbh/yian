"""Helpers for tests that load and walk flows nested far past the call nesting limit.

`lifted_limit` lifts the call nesting limit, `deep_load` loads a flow under it, and
`tight_stack` lowers the recursion limit so a walker that recurses once per nested
flow overflows.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import dataclasses
import inspect
import sys
from typing import Any

from yian import defaults
from yian.compose import ComposedFlow, load_flow

# Loading a deep chain still recurses once per nested flow.
_LOAD_RECURSION_LIMIT = 20_000


@contextmanager
def lifted_limit() -> Iterator[None]:
    """Lift the call nesting limit to a million levels, restoring the previous values after.

    The values are installed with no range check, as `set_defaults.unchecked` does.
    """
    previous = defaults.current()
    defaults._install(dataclasses.replace(previous, max_call_nesting=10**6))
    try:
        yield
    finally:
        defaults._install(previous)


@contextmanager
def _recursion_limit(limit: int) -> Iterator[None]:
    """Set the recursion limit to `limit`, restoring the old limit after."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def deep_load(text: str, **kwargs: Any) -> ComposedFlow:
    """Load `text` with `load_flow(text, **kwargs)` under `lifted_limit()`.

    The recursion limit is raised to at least 20 000 around the load only, since
    loading still recurses once per nested flow; a higher limit is kept.

    Args:
        text (`str`): The flow's YAML text.
        **kwargs (`Any`): Passed to `load_flow` unchanged.

    Returns:
        `ComposedFlow`: The loaded flow.
    """
    limit = max(sys.getrecursionlimit(), _LOAD_RECURSION_LIMIT)
    with lifted_limit(), _recursion_limit(limit):
        return load_flow(text, **kwargs)


@contextmanager
def tight_stack(frames: int) -> Iterator[None]:
    """Allow only `frames` frames past tight_stack's frame, restoring the limit after.

    A walker that recursed once per nested flow would overflow in this headroom.
    The headroom is counted from tight_stack's own frame, so the `with` block
    starts with a few frames of it already used, and a very small `frames` can
    overflow while setting up. Size `frames` with this rule: with `baseline` the
    frames the same call needs at depth 3, `tight_stack(k)` catches a walker that
    takes `f` frames per level at depth `N` only when `N * f > k - baseline`.

    Args:
        frames (`int`): The frames allowed past tight_stack's own frame.
    """
    with _recursion_limit(len(inspect.stack(0)) + frames):
        yield
