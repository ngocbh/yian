"""Declarative per-model capability table for OpenAI-compatible providers.

This is the single place that knows which model IDs reject which API
parameters or require which structured-output method. The LLM client
subclasses consult ``get_capabilities(model_name)`` instead of hardcoding
model-name ``if`` ladders, so adding a new model (or a new provider quirk)
means editing this table — not the client code.

Pattern adapted from the per-model ``compat:`` flags DeepSeek themselves
publish in their integration guides (e.g. the Oh My Pi config schema
documents ``supportsToolChoice``, ``requiresReasoningContentForToolCalls``
as declarative per-model fields).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


StructuredMethod = Literal[
    "function_calling",  # uses tools; respects supports_tool_choice
    "json_mode",         # uses response_format={"type":"json_object"}
    "json_schema",       # uses response_format={"type":"json_schema",...}
    "none",              # no structured output available; caller falls back to free-text
]


@dataclass(frozen=True)
class ModelCapabilities:
    """What an OpenAI-compatible model accepts at the API level."""

    supports_tool_choice: bool
    preferred_structured_method: StructuredMethod
    # MiniMax M2.x reasoning models need ``reasoning_split=True`` so the
    # <think> block lands in ``reasoning_details`` instead of polluting
    # ``content``. The flag is rejected by non-reasoning MiniMax models
    # (Coding Plan, MiniMax-Text-01, etc.), so we only set it where the
    # model actually consumes it. (#826)
    requires_reasoning_split: bool = False


# DeepSeek's thinking models accept the ``tools`` array but reject the
# ``tool_choice`` parameter (official Oh My Pi integration guide and the
# 400 response in issue #678). Their official tool-calling examples
# (api-docs.deepseek.com/guides/tool_calls) pass ``tools=[...]`` without
# ``tool_choice`` — we mirror that pattern by setting supports_tool_choice
# to False and letting the client suppress the kwarg. Their echo of
# ``reasoning_content`` on later turns lives in ``DeepSeekChatOpenAI``.
_DEEPSEEK_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    preferred_structured_method="function_calling",
)

_DEEPSEEK_CHAT = ModelCapabilities(
    supports_tool_choice=True,
    preferred_structured_method="function_calling",
)

# MiniMax M2.x reasoning models accept the tools array, but their
# tool_choice parameter is restricted to the enum {"none", "auto"}
# (platform.minimax.io/docs/api-reference/text-post). Langchain's
# function_calling path sends tool_choice as a function-spec dict, which
# MiniMax 400s — same shape as the DeepSeek bug. supports_tool_choice=False
# makes the dispatch in NormalizedChatOpenAI suppress the kwarg; the schema
# still ships as a tool. json_mode response_format is only for
# MiniMax-Text-01, not M2.x.
_MINIMAX_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    preferred_structured_method="function_calling",
    requires_reasoning_split=True,
)

_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    preferred_structured_method="function_calling",
)

# Catalog sentinel: a single entry whose structured method is `none`. It exists so the
# prompt-injection fallback (`structured.py`) is reachable through the *real* capability
# source on a concrete (provider, model) pair — no real provider is marked no-native today.
_NO_NATIVE_STRUCTURED = ModelCapabilities(
    supports_tool_choice=False,
    preferred_structured_method="none",
)


# Exact-ID matches take precedence over pattern matches.
_BY_ID: dict[str, ModelCapabilities] = {
    "deepseek-chat": _DEEPSEEK_CHAT,
    # Test/escape-hatch sentinel — the only entry with no native structured output.
    "no-structured-sentinel": _NO_NATIVE_STRUCTURED,
}

# Thinking-mode families: ``deepseek-reasoner``, ``deepseek-v4-*`` and MiniMax's
# M2.x lineup (platform.minimax.io/docs/api-reference/text-openai-api), and future
# ``deepseek-v5-*`` or ``MiniMax-M3*`` variants without a table edit.
_BY_PATTERN: list[tuple[re.Pattern[str], ModelCapabilities]] = [
    (re.compile(r"^deepseek-v\d"), _DEEPSEEK_THINKING),
    (re.compile(r"^deepseek-reasoner"), _DEEPSEEK_THINKING),
    (re.compile(r"^MiniMax-M\d"), _MINIMAX_THINKING),
]


def get_capabilities(model_name: str) -> ModelCapabilities:
    """Resolve capabilities by exact ID, then pattern, then default."""
    if model_name in _BY_ID:
        return _BY_ID[model_name]
    for pattern, caps in _BY_PATTERN:
        if pattern.match(model_name):
            return caps
    return _DEFAULT


# Providers whose langchain client supports `with_structured_output` natively and which are
# NOT covered by the OpenAI-compat capability table above. For these, native structured
# output is always available regardless of model id.
_NATIVE_STRUCTURED_PROVIDERS = frozenset({"anthropic", "google", "azure", "ollama"})


def supports_native_structured(provider: str, model: str) -> bool:
    """Whether `(provider, model)` can use langchain's native `with_structured_output`.

    Native-by-default providers (`anthropic`/`google`/`azure`/`ollama`) always can — they
    route to their own clients, not the OpenAI-compat table. Every other provider
    (OpenAI-compatible names, or an unknown future provider) consults the per-model table:
    a model whose `preferred_structured_method` is `none` (today only the catalog sentinel)
    has no native path, so the caller falls back to prompt-injection. An unknown model
    resolves to `_DEFAULT` (`function_calling`) and thus stays native.
    """
    if provider.lower() in _NATIVE_STRUCTURED_PROVIDERS:
        return True
    return get_capabilities(model).preferred_structured_method != "none"


# JSON-schema keys whose value is one subschema, a name -> subschema map, or a list of them.
_SUBSCHEMA_KEYS = ("items", "additionalProperties")
_SUBSCHEMA_MAP_KEYS = ("properties", "$defs")
_SUBSCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf")


def has_open_schema(schema: Any) -> bool:
    """Whether the JSON schema sent for `schema` has an open part.

    It recognizes the open shapes `type_to_schema` emits for an ``object``, ``Any`` or
    elementless-list AGENT output: ``additionalProperties: true`` or an empty ``{}``
    subschema (any value), anywhere, ``$defs`` included. Other JSON-schema constructs
    are not inspected. Strict structured-output modes (OpenAI and Azure strict,
    Responses ``strict: true``) reject these shapes. It lives beside the
    capability table because, like the table, it decides which structured-output method
    and options a client sends.

    Args:
        schema (`Any`):
            The schema given to ``with_structured_output``: a pydantic class (read through
            ``model_json_schema()``) or a JSON-schema dict (read as given).

    Returns:
        `bool`:
            `True` when the schema has an open part; `False` when it has none or is any
            other kind of input (a TypedDict, a pydantic v1 class, a callable).
    """
    if isinstance(schema, dict):
        root = schema
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        root = schema.model_json_schema()
    else:
        return False
    stack = [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if not node or node.get("additionalProperties") is True:
            return True
        for key in _SUBSCHEMA_KEYS:
            if isinstance(node.get(key), dict):
                stack.append(node[key])
        # Walk the maps by value, so an empty `properties: {}` is not itself open.
        for key in _SUBSCHEMA_MAP_KEYS:
            if isinstance(node.get(key), dict):
                stack.extend(node[key].values())
        for key in _SUBSCHEMA_LIST_KEYS:
            if isinstance(node.get(key), list):
                stack.extend(node[key])
    return False
