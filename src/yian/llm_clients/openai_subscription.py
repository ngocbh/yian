"""Direct OpenAI subscription requests behind the ordinary chat-model interface.

The caller supplies current access credentials and owns any injected HTTP clients.
This module owns request formatting and completed-response validation; it never
logs in, persists credentials, executes tools, or retains conversation history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import aclosing, closing
from dataclasses import dataclass, field

import httpx
from langchain_core.messages import ChatMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

from .capabilities import has_open_schema
from .openai_client import RequestBlockMixin

_BASE_URL = "https://chatgpt.com/backend-api/codex"
_ACCESS_ARGUMENT = "_composer_subscription_access"


@dataclass(frozen=True)
class SubscriptionAccess:
    """Current request credentials, supplied by a refresh-aware access source.

    Attributes:
        access_token (`str`): Nonempty bearer credential, omitted from repr.
        account_id (`str`): OpenAI account identifier for the selected connection.
        residency (`str | None`): Optional service routing hint; defaults to None.
    """

    access_token: str = field(repr=False)
    account_id: str
    residency: str | None = None

    def __post_init__(self) -> None:
        for name in ("access_token", "account_id", "residency"):
            value = getattr(self, name)
            if name == "residency" and value is None:
                continue
            if (
                not isinstance(value, str)
                or not value
                or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
            ):
                raise ValueError(
                    f"Subscription {name} must be nonempty HTTP header text"
                )


def _instruction_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            parts.append(block["text"])
        else:
            raise ValueError("OpenAI subscription instructions must contain only text")
    return "".join(parts)


class _SubscriptionChatOpenAI(RequestBlockMixin, ChatOpenAI):
    """Adapt Responses requests without owning an agent or tool loop.

    The ``openai`` block is merged into langchain's payload before the subscription
    guards run, so a block can never switch off the protocol settings they enforce.
    """

    access_source: Callable[[], SubscriptionAccess] = Field(exclude=True, repr=False)
    stop_check: Callable[[], None] | None = Field(
        default=None, exclude=True, repr=False
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def _check_stopped(self) -> None:
        if self.stop_check is not None:
            self.stop_check()

    def with_structured_output(self, schema, *, method=None, **kwargs):
        """Use the subscription transport's ordinary function-tool protocol.

        Every request is Responses-shaped, where function tools default to strict, so a
        schema with an open part (`has_open_schema`) is sent with ``strict=False``;
        strict mode rejects it. An explicit ``method`` or ``strict`` is respected.

        Args:
            schema (`dict | type`):
                The output schema: a pydantic class, a TypedDict or a JSON-schema dict.
            method (`str`, *optional*, defaults to `None`):
                The structured-output method; `None` means ``function_calling``.
            **kwargs:
                Passed to `ChatOpenAI.with_structured_output`.

        Returns:
            `Runnable`:
                The structured-output runnable langchain builds.
        """
        method = method or "function_calling"
        if method == "function_calling" and "strict" not in kwargs and has_open_schema(schema):
            kwargs["strict"] = False
        return super().with_structured_output(schema, method=method, **kwargs)

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        access = kwargs.pop(_ACCESS_ARGUMENT, None)
        messages = self._convert_input(input_).to_messages()
        instructions, ordinary = [], []
        for message in messages:
            if isinstance(message, SystemMessage) or (
                isinstance(message, ChatMessage)
                and message.role in ("system", "developer")
            ):
                instructions.append(_instruction_text(message))
            else:
                ordinary.append(message)
        payload = super()._get_request_payload(ordinary, stop=stop, **kwargs)
        payload.setdefault(
            "instructions", "\n\n".join(instructions) or "You are a helpful assistant."
        )
        # The SDK merges extra_body after this hook. It must not overwrite the
        # validated subscription protocol settings on the actual wire request.
        if payload.get("extra_body"):
            raise ValueError("OpenAI subscription requests do not support extra_body")
        if payload.get("store") is not False or payload.get("stream") is not True:
            raise ValueError(
                "OpenAI subscription requests require store=False and stream=True"
            )
        if payload.get("previous_response_id") is not None:
            raise ValueError(
                "OpenAI subscription requests require the full conversation"
            )
        payload["include"] = list(
            dict.fromkeys(
                [
                    *(payload.get("include") or []),
                    "reasoning.encrypted_content",
                ]
            )
        )
        self._check_stopped()
        if access is None:
            access = self.access_source()
        if type(access) is not SubscriptionAccess:
            raise TypeError("Subscription access source must return SubscriptionAccess")
        self._check_stopped()
        headers = httpx.Headers(payload.get("extra_headers") or {})
        headers["authorization"] = f"Bearer {access.access_token}"
        headers["ChatGPT-Account-Id"] = access.account_id
        headers["originator"] = "yian"
        if access.residency is not None:
            headers["x-openai-internal-codex-residency"] = access.residency
        extra_headers = dict(headers)
        # The SDK merges plain dictionaries before making case-insensitive HTTP
        # headers. Match its spelling so the placeholder bearer is replaced.
        extra_headers["Authorization"] = extra_headers.pop("authorization")
        payload["extra_headers"] = extra_headers
        return payload

    def _stream(self, *args, **kwargs):
        self._check_stopped()
        completed = False
        with closing(super()._stream(*args, **kwargs)) as stream:
            for chunk in stream:
                status = chunk.message.response_metadata.get("status")
                if status is not None:
                    if status != "completed":
                        raise RuntimeError(
                            "OpenAI subscription response did not complete"
                        )
                    completed = True
                else:
                    self._check_stopped()
                yield chunk
        if not completed:
            raise RuntimeError(
                "OpenAI subscription stream ended before a complete response"
            )

    async def _astream(self, *args, **kwargs):
        self._check_stopped()
        # Refresh may wait for a credential-file lock or network request.
        kwargs[_ACCESS_ARGUMENT] = await asyncio.to_thread(self.access_source)
        self._check_stopped()
        completed = False
        async with aclosing(super()._astream(*args, **kwargs)) as stream:
            async for chunk in stream:
                status = chunk.message.response_metadata.get("status")
                if status is not None:
                    if status != "completed":
                        raise RuntimeError(
                            "OpenAI subscription response did not complete"
                        )
                    completed = True
                else:
                    self._check_stopped()
                yield chunk
        if not completed:
            raise RuntimeError(
                "OpenAI subscription stream ended before a complete response"
            )


def create_subscription_model(
    model: str,
    *,
    access_source: Callable[[], SubscriptionAccess],
    check_stopped: Callable[[], None] | None = None,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
    block: Mapping | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> ChatOpenAI:
    """Build a direct subscription model with externally managed credentials.

    Args:
        model (`str`): Explicit provider model identifier.
        access_source (`Callable[[], SubscriptionAccess]`): Resolve current access
            for the selected connection; owns refresh and identity validation.
        check_stopped (`Callable[[], None] | None`): Optional cooperative check.
        http_client (`httpx.Client | None`): Optional caller-owned sync transport.
        http_async_client (`httpx.AsyncClient | None`): Optional caller-owned async
            transport. Omitted clients use the SDK's standard client lifecycle.
        block (`Mapping | None`): Optional ``openai`` block, shaped like a Responses
            request body and already checked against `OpenAISubscriptionBlock`;
            merged into every request body as written. Defaults to None (nothing extra).
        timeout (`float`): Network-operation timeout in seconds; defaults to 60.
        max_retries (`int`): SDK transport retries; defaults to two. Does not retry
            partial response streams or execute tools.

    Returns:
        `ChatOpenAI`: A model supporting ordinary invoke, stream, and tool binding.

    Raises:
        ValueError: Model construction rejects an invalid SDK setting. Request-time
            credential, transport, cancellation, and incomplete-response failures
            propagate from invocation without a successful model reply.
    """
    return _SubscriptionChatOpenAI(
        model=model,
        access_source=access_source,
        stop_check=check_stopped,
        base_url=_BASE_URL,
        api_key="managed-by-yian",
        use_responses_api=True,
        use_previous_response_id=False,
        streaming=True,
        stream_usage=False,
        store=False,
        output_version="responses/v1",
        include=["reasoning.encrypted_content"],
        http_client=http_client,
        http_async_client=http_async_client,
        block=dict(block or {}),
        block_provider="openai",
        block_api="responses",
        timeout=timeout,
        max_retries=max_retries,
    )
