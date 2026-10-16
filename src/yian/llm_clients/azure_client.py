import copy
import json
from functools import partial
from operator import itemgetter
from types import NoneType, UnionType
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

import annotated_types
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

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
# strict mode accepts some of these. Keywords are moved only from classes built of types
# and pydantic constraints the reply's parse enforces (`_reply_checks_every_keyword`), so
# adding a keyword such a class can emit is safe (e.g. ``exclusiveMinimum``).
_AZURE_STRICT_UNSUPPORTED = (
    "minLength", "maxLength", "pattern", "format",
    "minimum", "maximum", "multipleOf",
    "patternProperties", "unevaluatedProperties", "propertyNames", "minProperties",
    "maxProperties",
    "unevaluatedItems", "contains", "minContains", "maxContains", "minItems", "maxItems",
    "uniqueItems",
)

_SCALAR_TYPES = frozenset({str, int, float, bool, NoneType, None})

# The metadata pydantic makes for ``Field(gt=..., ge=..., lt=..., le=..., multiple_of=...,
# min_length=..., max_length=..., strict=..., allow_inf_nan=..., pattern=...)``. The class
# holding ``allow_inf_nan`` and ``pattern`` is private, so it is read from a ``Field``.
_CONSTRAINT_TYPES = frozenset({
    annotated_types.Gt, annotated_types.Ge, annotated_types.Lt, annotated_types.Le,
    annotated_types.MultipleOf, annotated_types.MinLen, annotated_types.MaxLen,
    *(type(meta) for meta in Field(strict=True, allow_inf_nan=False).metadata),
})


def _strict_schema_without(schema_cls: type) -> Optional[dict[str, Any]]:
    """Return the strict ``response_format`` for `schema_cls` without Azure's rejected keywords.

    Each subschema loses the keywords in `_AZURE_STRICT_UNSUPPORTED` and has them restated
    at the end of its ``description``, in the ``{key: value, ...}`` shape Anthropic's SDK
    ``transform_schema`` uses, with JSON spellings of the values. Only schema positions are
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
    # matches what a strict Azure request sends now. Imported here so that if a release
    # moves it, only this call fails, not every Azure client import.
    from openai.lib._parsing._completions import type_to_response_format_param

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


def _langchain_sends_a_tool(model_name: Optional[str]) -> bool:
    """Whether langchain sends ``json_schema`` structured output to `model_name` as a tool.

    Mirrors the model check in langchain-openai 1.6.2's
    ``BaseChatOpenAI.with_structured_output``, which falls back to ``function_calling``
    for models without structured outputs. On Azure the model name is the deployment name.

    Args:
        model_name (`str`, *optional*):
            The model's ``model_name``.

    Returns:
        `bool`:
            `True` for a name starting with ``gpt-3`` or ``gpt-4-``, or equal to ``gpt-4``.
    """
    return bool(model_name) and (
        model_name.startswith(("gpt-3", "gpt-4-")) or model_name == "gpt-4"
    )


def _parse_strict_reply(message: AIMessage, schema: type[BaseModel]) -> BaseModel:
    """Validate the text of `message` as JSON for `schema`.

    The JSON is parsed strictly, so a reply cut off mid-value is an error rather than
    repaired into a shorter value, as langchain's partial-JSON parser would.

    Args:
        message (`AIMessage`):
            The model's reply.
        schema (`type[BaseModel]`):
            The pydantic v2 output class.

    Returns:
        `BaseModel`:
            The validated output.

    Raises:
        `OutputParserException`:
            The text is not valid JSON for `schema`, naming any refusal the reply holds;
            the pydantic `ValidationError` is its ``__cause__``.
    """
    text = message.text
    try:
        return schema.model_validate_json(text)
    except ValueError as err:
        refusal = message.additional_kwargs.get("refusal")
        reason = f"the model refused: {refusal}" if refusal else str(err)
        raise OutputParserException(
            f"Failed to parse {schema.__name__} from completion {text!r}. Got: {reason}",
            llm_output=text,
        ) from err


def _reply_checks_every_keyword(schema_cls: type[BaseModel]) -> bool:
    """Whether parsing a reply as `schema_cls` checks every keyword in its JSON schema.

    Only types whose schema keywords all come from pydantic validation qualify: plain
    pydantic v2 models (`_is_plain_model`), ``str``, ``int``, ``float``, ``bool``,
    ``None``, ``list``, ``dict``, unions, ``Literal``, and ``Annotated`` with pydantic
    ``Field`` constraints (`_CONSTRAINT_TYPES`). Anything else (a TypedDict, a dataclass,
    ``WithJsonSchema``, a custom schema hook) can add a keyword no validator checks, so
    moving it out of the sent schema would drop it.

    Args:
        schema_cls (`type[BaseModel]`):
            A pydantic v2 model class.

    Returns:
        `bool`:
            `True` when every type reachable from `schema_cls` qualifies.
    """
    seen: set[type] = set()
    stack: list[Any] = [schema_cls]
    while stack:
        item = stack.pop()
        if isinstance(item, FieldInfo):
            if item.json_schema_extra is not None:
                return False
            if any(type(meta) not in _CONSTRAINT_TYPES for meta in item.metadata):
                return False
            stack.append(item.annotation)
            continue
        origin = get_origin(item)
        # Only plain classes are looked up: an alias's metadata may be unhashable.
        if origin is None and (item in _SCALAR_TYPES or item in seen):
            continue
        if origin is Literal:
            continue
        if origin is Annotated:
            annotated, *metadata = get_args(item)
            stack.append(annotated)
            for meta in metadata:
                if isinstance(meta, FieldInfo):
                    stack.append(meta)
                elif type(meta) not in _CONSTRAINT_TYPES:
                    return False
        elif origin in (list, dict, Union, UnionType) or item in (list, dict):
            stack.extend(get_args(item))
        elif isinstance(item, type) and issubclass(item, BaseModel):
            if not _is_plain_model(item):
                return False
            seen.add(item)
            stack.extend(item.model_fields.values())
        else:
            return False
    return True


def _is_plain_model(model: type[BaseModel]) -> bool:
    """Whether `model` keeps pydantic's own schema hooks and builds its schema for validation.

    Args:
        model (`type[BaseModel]`):
            A pydantic v2 model class.

    Returns:
        `bool`:
            `True` when no schema hook (``model_json_schema``, which the openai helper
            calls, included) is overridden, the config has no ``json_schema_extra``, and
            the schema isn't built in serialization mode, whose serializer return types
            and computed fields validation never checks.
    """
    config = model.model_config
    return (
        config.get("json_schema_extra") is None
        and config.get("json_schema_mode_override") in (None, "validation")
        and all(
            getattr(getattr(model, hook), "__func__", None)
            is getattr(BaseModel, hook).__func__
            for hook in (
                "model_json_schema",
                "__get_pydantic_json_schema__",
                "__get_pydantic_core_schema__",
            )
        )
    )


class NormalizedAzureChatOpenAI(RequestBlockMixin, AzureChatOpenAI):
    """AzureChatOpenAI with normalized content output and the `azure` block in each request."""

    # langchain's Azure client sets its own User-Agent as an SDK custom header.
    _sdk_header_names = ("User-Agent",)

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it.

        A Responses refusal block is kept as ``additional_kwargs["refusal"]``, where Chat
        Completions puts a refusal, since joining the content drops non-text blocks.
        """
        if isinstance(response.content, list):
            refusals = [
                block["refusal"] for block in response.content
                if isinstance(block, dict) and block.get("type") == "refusal"
            ]
            if refusals and not response.additional_kwargs.get("refusal"):
                response.additional_kwargs["refusal"] = "\n".join(refusals)
        return normalize_content(response)

    def with_structured_output(self, schema=None, *, method=None, include_raw=False, **kwargs):
        """Bind `schema` for structured output in a form Azure's strict mode accepts.

        Azure's default (``json_schema``, strict) rejects a schema with an open part
        (`has_open_schema`), so without an explicit ``method`` such a schema goes as a
        forced function tool without ``strict`` instead. When an open schema is bound as
        a function tool on a deployment langchain routes to the Responses API, where an
        omitted ``strict`` lets the API try strict mode, ``strict=False`` is sent unless
        the caller gave ``strict``.

        Azure's strict mode also rejects keywords such as ``minimum``, ``maximum`` and
        ``minItems`` (`_AZURE_STRICT_UNSUPPORTED`). A closed pydantic v2 schema holding one,
        given without ``method`` or ``strict``, is sent as a strict ``json_schema`` with
        those keywords moved into the subschema's description (`_strict_schema_without`).
        The reply's JSON is then parsed strictly against `schema` itself, which enforces
        the moved keywords that come from pydantic constraints. A refusal, an empty reply
        or one cut off mid-value raises an `OutputParserException`, naming any refusal,
        that yian's structured-output retry corrects. A class holding a type whose schema
        keywords the parse might not check (`_reply_checks_every_keyword`), such as
        ``json_schema_extra``, ``WithJsonSchema``, a schema hook, a TypedDict or a
        dataclass, keeps langchain's request, where Azure rejects the keyword. So does a
        deployment langchain sends as a function tool (`_langchain_sends_a_tool`). Any other
        schema or method reaches langchain unchanged.

        Args:
            schema (`dict | type | None`, *optional*, defaults to `None`):
                The output schema: a pydantic class, a TypedDict or a JSON-schema dict.
            method (`str`, *optional*, defaults to `None`):
                The structured-output method; `None` picks as described above.
            include_raw (`bool`, *optional*, defaults to `False`):
                Whether to return ``{"raw", "parsed", "parsing_error"}`` instead of the
                parsed output.
            **kwargs:
                Passed to `AzureChatOpenAI.with_structured_output`, or bound into the model
                call when the keywords are moved, as langchain's ``json_schema`` path does.

        Returns:
            `Runnable`:
                A runnable that returns outputs matching `schema`.
        """
        if (
            method is None
            and "strict" not in kwargs
            and not _langchain_sends_a_tool(self.model_name)
            and isinstance(schema, type)
            and issubclass(schema, BaseModel)
            and not has_open_schema(schema)
            and _reply_checks_every_keyword(schema)
            and (response_format := _strict_schema_without(schema)) is not None
        ):
            return self._stripped_strict_output(
                schema, response_format, include_raw=include_raw, **kwargs
            )
        if method is None and has_open_schema(schema):
            method = "function_calling"
        _strict_false_for_open_schema(self, schema, method, kwargs)
        if method is not None:
            kwargs["method"] = method
        return super().with_structured_output(schema, include_raw=include_raw, **kwargs)

    def _stripped_strict_output(self, schema, response_format, *, include_raw, **kwargs):
        """Bind `response_format` and parse the reply's content as `schema`.

        Args:
            schema (`type`):
                The pydantic v2 output class.
            response_format (`dict`):
                The strict format from `_strict_schema_without`.
            include_raw (`bool`):
                Whether to return ``{"raw", "parsed", "parsing_error"}``, assembled as
                `AzureChatOpenAI.with_structured_output` does.
            **kwargs:
                Bound into the model call as langchain's ``json_schema`` path binds them;
                ``tools`` are converted to OpenAI tools as it converts them.

        Returns:
            `Runnable`:
                A runnable that returns outputs matching `schema`.
        """
        if tools := kwargs.pop("tools", None):
            kwargs["tools"] = [convert_to_openai_tool(tool) for tool in tools]
        llm = self.bind(
            response_format=response_format,
            ls_structured_output_format={
                "kwargs": {"method": "json_schema", "strict": True},
                "schema": convert_to_openai_tool(schema),
            },
            **kwargs,
        )
        parser = RunnableLambda(partial(_parse_strict_reply, schema=schema))
        if not include_raw:
            return llm | parser
        # Kept in step with `AzureChatOpenAI.with_structured_output`'s include_raw assembly.
        parser_assign = RunnablePassthrough.assign(
            parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
        )
        parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
        return RunnableMap(raw=llm) | parser_assign.with_fallbacks(
            [parser_none], exception_key="parsing_error"
        )


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
