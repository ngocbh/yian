import re
from contextvars import ContextVar
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+
# and Sonnet 4.5+ only. Haiku (any version shipped to date) 400s with
# ``"This model does not support the effort parameter"`` (#831). Future
# ``claude-{opus,sonnet}-X-Y`` releases inherit effort support via the
# forward-compat pattern below; future Haiku stays excluded by default.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
}
_EFFORT_PATTERN = re.compile(r"^claude-(opus|sonnet)-\d+-\d+$")


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    return model_lc in _EFFORT_EXACT or bool(_EFFORT_PATTERN.match(model_lc))


# The raw responses opened by the `_stream` call running in this context.
_OPENED_STREAMS: ContextVar[Optional[list]] = ContextVar("anthropic_opened_streams", default=None)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output and streams that close early.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def _create(self, payload: dict) -> Any:
        response = super()._create(payload)
        opened = _OPENED_STREAMS.get()
        if opened is not None:
            opened.append(response)
        return response

    def _stream(self, *args: Any, **kwargs: Any):
        """Stream like `ChatAnthropic`, closing the HTTP response when closed early.

        langchain_anthropic iterates the SDK stream without closing it, so a
        caller that stops early would hold the connection until garbage
        collection. `_create` hands the raw response to this call through a
        context variable, set only while the inner generator runs.
        """
        opened: list = []
        chunks = super()._stream(*args, **kwargs)
        try:
            while True:
                token = _OPENED_STREAMS.set(opened)
                try:
                    chunk = next(chunks)
                except StopIteration:
                    return
                finally:
                    _OPENED_STREAMS.reset(token)
                yield chunk
        finally:
            chunks.close()
            for response in opened:
                response.http_response.close()

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it."""
        return normalize_content(response)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
