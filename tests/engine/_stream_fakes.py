"""Provider-free streaming chat model for AGENT live-event tests.

`ScriptedStreamChat` is a real LangChain `BaseChatModel`: `invoke` returns each
scripted `AIMessage` whole, while `stream` yields its text word by word, each
tool call as two chunks sharing its index, and then one text-free chunk carrying
the reply's usage. No network, no API keys.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def usage(input_tokens, output_tokens, **details):
    """Build LangChain usage metadata with optional token details."""
    data = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if "cache_read" in details:
        data["input_token_details"] = {"cache_read": details["cache_read"]}
    if "reasoning" in details:
        data["output_token_details"] = {"reasoning": details["reasoning"]}
    return data


def reply(text="", *, calls=(), tokens=None):
    """Build one scripted reply; `calls` holds `(name, args, id)` triples."""
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, args, call_id in calls
        ],
        usage_metadata=tokens,
    )


class ScriptedStreamChat(BaseChatModel):
    """Serve scripted replies in order through `invoke` or `stream`."""

    script: list
    mode: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted-stream"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[getattr(tool, "name", "tool") for tool in tools])

    def _next(self, how):
        self.mode.append(how)
        message = self.script.pop(0)
        if isinstance(message, Exception):
            raise message
        return message

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next("invoke"))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._next("stream")
        words = message.content.split(" ") if message.content else []
        for index, word in enumerate(words):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=word if index == 0 else " " + word)
            )
        # Each tool call arrives in two chunks sharing its index, as providers send them.
        for position, call in enumerate(message.tool_calls):
            args = json.dumps(call["args"])
            half = len(args) // 2
            for part in ({"name": call["name"], "id": call["id"], "args": args[:half]},
                         {"name": None, "id": None, "args": args[half:]}):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="", tool_call_chunks=[{**part, "index": position}]
                    )
                )
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=message.usage_metadata)
        )
