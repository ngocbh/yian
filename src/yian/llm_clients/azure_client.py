import copy
import json
from typing import Any, Optional

from langchain_openai import AzureChatOpenAI
from openai.lib._parsing._completions import type_to_response_format_param

from .base_client import BaseLLMClient, normalize_content
from .capabilities import has_open_schema
from .config import AZURE_NEEDS_BASE_URL, azure_resource_endpoint
from .openai_client import RequestBlockMixin, _strict_false_for_open_schema
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "callbacks", "http_client",
    "http_async_client",
)

# Keywords Azure OpenAI's strict structured outputs reject, from "JSON Schema support and
# limitations" in learn.microsoft.com/azure/foundry/openai/how-to/structured-outputs; the
# list covers both Chat Completions and the Responses API. It is Azure's own: OpenAI's
# strict mode accepts some of these. Adding a keyword is always safe, since the reply's
# parse still enforces it (e.g. ``exclusiveMinimum``).
_AZURE_STRICT_UNSUPPORTED = (
    "minLength", "maxLength", "pattern", "format",
    "minimum", "maximum", "multipleOf",
    "patternProperties", "unevaluatedProperties", "propertyNames", "minProperties",
    "maxProperties",
    "unevaluatedItems", "contains", "minContains", "maxContains", "minItems", "maxItems",
    "uniqueItems",
)


def _strict_schema_without(schema_cls: type) -> Optional[dict[str, Any]]:
    """Return the strict ``response_format`` for `schema_cls` without Azure's rejected keywords.

    Each subschema loses the keywords in `_AZURE_STRICT_UNSUPPORTED` and has them restated
    at the end of its ``description`` as ``{key: value, ...}`` with JSON spellings of the
    values, as Anthropic's SDK ``transform_schema`` does. Only schema positions are
    visited (``properties`` and ``$defs`` values, ``items``, ``prefixItems`` members, a
    dict ``additionalProperties``, ``anyOf``/``oneOf``/``allOf`` members), so a property
    named like a keyword is kept.

    Args:
        schema_cls (`type`):
            A pydantic v2 model class.

    Returns:
        `dict | None`:
            The ``{"type": "json_schema", "json_schema": {...}}`` format with the keywords
            moved, or `None` when no subschema holds one.
    """
    # The private openai SDK helper `chat.completions.parse` uses today, so the schema
    # matches what a strict Azure request sends now.
    fmt = copy.deepcopy(type_to_response_format_param(schema_cls))
    moved = False
    stack = [fmt["json_schema"]["schema"]]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        notes = [
            f"{key}: {json.dumps(node.pop(key))}"
            for key in _AZURE_STRICT_UNSUPPORTED if key in node
        ]
        if notes:
            moved = True
            note = "{" + ", ".join(notes) + "}"
            description = node.get("description")
            node["description"] = f"{description}\n\n{note}" if description else note
        for key in ("properties", "$defs"):
            stack.extend((node.get(key) or {}).values())
        for key in ("prefixItems", "anyOf", "oneOf", "allOf"):
            stack.extend(node.get(key) or ())
        stack.append(node.get("items"))
        stack.append(node.get("additionalProperties"))
    return fmt if moved else None


class NormalizedAzureChatOpenAI(RequestBlockMixin, AzureChatOpenAI):
    """AzureChatOpenAI with normalized content output and the `azure` block in each request."""

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it."""
        return normalize_content(response)

    def with_structured_output(self, schema=None, *, method=None, **kwargs):
        """Bind `schema` for structured output, as a non-strict function tool when it is open.

        Azure's default (``json_schema``, strict) rejects a schema with an open part
        (`has_open_schema`), so without an explicit ``method`` such a schema goes as a
        forced function tool without ``strict`` instead. When an open schema is bound as
        a function tool on a deployment langchain routes to the Responses API, where an
        omitted ``strict`` lets the API try strict mode, ``strict=False`` is sent unless
        the caller gave ``strict``. Any other schema or method reaches langchain unchanged.

        Args:
            schema (`dict | type | None`, *optional*, defaults to `None`):
                The output schema: a pydantic class, a TypedDict or a JSON-schema dict.
            method (`str`, *optional*, defaults to `None`):
                The structured-output method; `None` picks as described above.
            **kwargs:
                Passed to `AzureChatOpenAI.with_structured_output`.

        Returns:
            `Runnable`:
                The structured-output runnable langchain builds.
        """
        if method is None and has_open_schema(schema):
            method = "function_calling"
        _strict_false_for_open_schema(self, schema, method, kwargs)
        if method is not None:
            kwargs["method"] = method
        return super().with_structured_output(schema, **kwargs)


class AzureOpenAIClient(BaseLLMClient):
    """Client for Azure OpenAI deployments.

    ``model`` is the Azure deployment name and ``base_url`` the resource endpoint
    (e.g. ``https://<resource>.openai.azure.com/``; an ``/openai`` or
    ``/openai/v1`` suffix is stripped). Both come from ``llm_config``, so a run
    records them; ``AZURE_OPENAI_DEPLOYMENT_NAME`` and ``AZURE_OPENAI_ENDPOINT``
    are not read.

    Reads from the environment:
        AZURE_OPENAI_API_KEY: API key, unless the config passes ``api_key``
        OPENAI_API_VERSION: API version (e.g. 2025-03-01-preview)
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured AzureChatOpenAI instance.

        Raises:
            `ValueError`: No ``base_url`` was given.
        """
        if not self.base_url:
            raise ValueError(AZURE_NEEDS_BASE_URL)
        self.warn_if_unknown_model()

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "azure_deployment": self.model,
            "azure_endpoint": azure_resource_endpoint(self.base_url),
        }

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedAzureChatOpenAI(
            **llm_kwargs, block=self.kwargs.get("block") or {}, block_provider="azure",
        )

    def validate_model(self) -> bool:
        """Azure accepts any deployed model name."""
        return True
