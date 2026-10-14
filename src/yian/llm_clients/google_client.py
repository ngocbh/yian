import copy
from typing import Any, Optional

from google.genai.types import HttpOptions
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field

from .base_client import BaseLLMClient, normalize_content
from .provider_blocks import merge_request_body
from .validators import validate_model


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.

    Attributes:
        block (`dict[str, Any]`, *optional*, defaults to `{}`):
            The ``google`` block, shaped like the REST ``generateContent`` request body
            (camelCase names, REST enum spellings). It is deep-merged into every request
            body as written. Empty sends nothing extra.
    """

    block: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    def _prepare_request(self, messages, **kwargs):
        """Return langchain's request with `block` added to the request's ``extra_body``.

        google-genai deep-merges ``http_options.extra_body`` into the REST body after
        building it, aligning key case, so the block's camelCase keys land beside
        langchain's own fields (structured output, tools) and win where both set one.

        Args:
            messages (`list[BaseMessage]`):
                The messages being sent.
            **kwargs:
                Call options langchain folds into the request, including a caller's
                ``http_options`` (`dict` or `HttpOptions`), whose ``extra_body`` the block
                is merged into.

        Returns:
            `dict`:
                Keyword arguments for google-genai's ``generate_content`` call.
        """
        if self.block:
            options = kwargs.get("http_options")
            if options is None:
                options = HttpOptions()
            elif isinstance(options, dict):
                options = HttpOptions(**options)
            else:
                options = options.model_copy()
            extra_body = copy.deepcopy(options.extra_body or {})
            options.extra_body = merge_request_body(extra_body, self.block)
            kwargs["http_options"] = options
        return super()._prepare_request(messages, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it."""
        return normalize_content(response)


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models.

    With a ``base_url``, the model's httpx client does not follow redirects, so the
    key goes only to that URL.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
            # The SDK sends the key in x-goog-api-key, which httpx re-sends on a
            # cross-origin redirect.
            llm_kwargs["client_args"] = {"follow_redirects": False}

        for key in ("timeout", "max_retries", "temperature", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # An explicit None keeps langchain-google-genai from sending its 0.7 default.
        llm_kwargs.setdefault("temperature", None)

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Map thinking_level to appropriate API param based on model
        # Gemini 3 Pro: low, high
        # Gemini 3 Flash: minimal, low, medium, high
        # Gemini 2.5: thinking_budget (0=disable, -1=dynamic)
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = self.model.lower()
            if "gemini-3" in model_lower:
                # Gemini 3 Pro doesn't support "minimal", use "low" instead
                if "pro" in model_lower and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                # Gemini 2.5: map to thinking_budget
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs, block=self.kwargs.get("block") or {})

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
