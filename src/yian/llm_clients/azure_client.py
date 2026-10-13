from typing import Any, Optional

from langchain_openai import AzureChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .config import AZURE_NEEDS_BASE_URL
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "reasoning_effort", "temperature",
    "callbacks", "http_client", "http_async_client",
)


def _base_endpoint(endpoint: str) -> str:
    """Return the base Azure resource URL, stripping an OpenAI-v1 suffix.

    Some Azure portals hand out the `…/openai/v1` (OpenAI-compatible) surface.
    LangChain's AzureChatOpenAI appends its own `/openai/deployments/…` path,
    so that suffix yields a 404 ("Resource not found"). Strip it back to the
    base resource URL.
    """
    e = endpoint.rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if e.endswith(suffix):
            e = e[: -len(suffix)]
            break
    return e.rstrip("/")


class NormalizedAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI with normalized content output."""

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it."""
        return normalize_content(response)


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
            "azure_endpoint": _base_endpoint(self.base_url),
        }

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedAzureChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Azure accepts any deployed model name."""
        return True
