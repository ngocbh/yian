"""Run nested generator walkers at constant stack depth.

A walker over nested flows would naturally recurse once per nested CALL, MAP, or
LOOP body, so its depth would be bounded by Python's recursion limit. Written as a
generator instead, the walker hands its nested call to a trampoline rather than
making it:

- `drive` runs a walker that returns a value. Where the recursive walker called
  `child(...)`, the generator walker writes `(yield child(...))`.
- `flatten` runs a walker that streams items. Where the recursive walker wrote
  `yield from child(...)`, the generator walker writes `yield Descend(child(...))`.

Both keep the suspended walkers on a list, so the Python stack stays a few frames
deep whatever the nesting. Side-effect order, `try`/`finally` cleanup, the error
that escapes, and its `__context__` are the same as in the recursive walker.

Two rules apply to walkers written this way:

- A walker must not descend from inside an `except` handler. The trampoline
  resumes the parent outside any handler, so an error raised in the nested walker
  would not get the handled error as its `__context__`, as it would recursively.
- A `StopIteration` that escapes a hook a walker calls becomes `RuntimeError`
  (PEP 479), because the walker is a generator.

This module imports only the standard library, so any layer may import it.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

__all__ = ["Descend", "drive", "flatten"]


@dataclass(frozen=True, slots=True)
class Descend:
    """A nested stream a `flatten` walker yields where it would `yield from` it.

    Attributes:
        steps (`Generator`): The nested walker, not yet started.
    """

    steps: Generator[Any, None, None]


def _not_a_walker(value: object) -> TypeError:
    """Return the error for a nested walker that is not a generator."""
    return TypeError(f"a nested walker must be a generator, not {type(value).__name__}")


def drive(steps: Generator[Any, Any, Any]) -> Any:
    """Run a generator walker and its nested walkers, returning the outer result.

    The walker yields each nested walker (a generator) where it would call it, and
    gets the nested walker's return value back as the value of that `yield`. An
    error a nested walker raises is thrown in at the `yield`, so the parent's
    `try`/`except` and `finally` blocks see it as they would see it raised by a
    call. Yielding anything but a generator throws `TypeError` in at the `yield`.

    Nested walkers wait on a list rather than on the Python stack, so a walker over
    nested flows does not recurse once per level.

    Args:
        steps (`Generator`): The outer walker, not yet started.

    Returns:
        `Any`: The outer walker's return value.

    Raises:
        Exception: Any error that escapes the outer walker, after every open
            walker's `finally` blocks have run, inner to outer.

    Example:
        >>> def depth(n):
        ...     return 0 if n == 0 else (yield depth(n - 1)) + 1
        >>> drive(depth(100_000))
        100000
    """
    stack = [steps]
    sent: Any = None
    error: BaseException | None = None
    while True:
        top = stack[-1]
        try:
            if error is None:
                nested = top.send(sent)
            else:
                # Thrown here, outside the handler below, so an error the parent
                # raises after handling this one gets no `__context__` from drive.
                thrown, error = error, None
                nested = top.throw(thrown)
        except StopIteration as stop:
            stack.pop()
            if not stack:
                return stop.value
            sent = stop.value
            continue
        except BaseException as raised:
            stack.pop()
            if not stack:
                raise
            error = raised
            continue
        if isinstance(nested, Generator):
            stack.append(nested)
            sent = None
        else:
            error = _not_a_walker(nested)


def flatten(steps: Generator[Any, None, None]) -> Generator[Any, None, None]:
    """Yield the items of a generator walker and its nested streams, in order.

    The walker yields `Descend(child(...))` where it would `yield from child(...)`;
    every other value it yields is an item. Items come out in the order the
    `yield from` walker would produce them. An error a nested stream raises is
    thrown into its parent at the `Descend` point, and a `Descend` of anything but a
    generator throws `TypeError` there. Closing this generator early closes every
    open walker, inner to outer, as closing a `yield from` chain would.

    Nested streams wait on a list rather than on the Python stack, so a walker over
    nested flows does not recurse once per level.

    Args:
        steps (`Generator`): The outer walker, not yet started.

    Yields:
        `Any`: Each item the walkers yield, `Descend` markers excluded.

    Raises:
        Exception: Any error that escapes the outer walker.

    Example:
        >>> def leaves(tree):
        ...     for child in tree:
        ...         yield Descend(leaves(child)) if isinstance(child, list) else child
        >>> list(flatten(leaves([1, [2, [3]], 4])))
        [1, 2, 3, 4]
    """
    stack = [steps]
    error: BaseException | None = None
    try:
        while stack:
            top = stack[-1]
            try:
                if error is None:
                    item = next(top)
                else:
                    # Thrown outside the handler below, as in `drive`.
                    thrown, error = error, None
                    item = top.throw(thrown)
            except StopIteration:
                stack.pop()
                continue
            except BaseException as raised:
                stack.pop()
                if not stack:
                    raise
                error = raised
                continue
            if not isinstance(item, Descend):
                yield item
            elif isinstance(item.steps, Generator):
                stack.append(item.steps)
            else:
                error = _not_a_walker(item.steps)
    finally:
        while stack:
            stack.pop().close()
