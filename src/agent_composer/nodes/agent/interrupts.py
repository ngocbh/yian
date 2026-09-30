"""The signal a host interrupt raises out of a streamed AGENT model reply.

A leaf module, so the stream loop in `live.py` and the effect recorder in
`memory.py` share it without importing each other.

Layer: nodes — imports only LangChain messages.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage


class ReplyInterrupted(Exception):
    """A streamed model reply cut off at a chunk boundary by a host interrupt.

    `reply` is an `AIMessage` holding only the visible text streamed so far; a
    partial tool call is never kept because it cannot run.
    """

    def __init__(self, reply: BaseMessage) -> None:
        super().__init__("model reply interrupted by the host")
        self.reply = reply
