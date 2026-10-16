"""Tool functions whose annotations are strings, for the ToolContext detection tests."""

from __future__ import annotations

from yian.tools.context import ToolContext


class Db:
    pass


def place(ctx: ToolContext[Db], symbol: str, qty: int = 1) -> str:
    """Place an order.

    Args:
        symbol: The ticker.
        qty: How many shares.
    """
    return f"{symbol}:{qty}"


def ask_with_unresolved_request(ctx: ToolContext, **request: Missing) -> str:  # noqa: F821
    """An ask_user handler whose later annotation can't be resolved."""
    return ctx.call_id


class AskWithUnresolvedRequest:
    """A callable-object ask_user handler whose later annotation can't be resolved."""

    def __call__(self, ctx: ToolContext, **request: Missing) -> str:  # noqa: F821
        return ctx.call_id


def ask_with_unresolved_context(ctx: ToolContext[Missing], **request: object) -> str:  # noqa: F821
    """An ask_user handler whose ToolContext annotation can't be resolved."""
    return ""
