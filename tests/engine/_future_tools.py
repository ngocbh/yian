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
