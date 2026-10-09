import os
from typing import Any, Optional

from langchain_openai import AzureChatOpenAI

from .base_client import BaseLLMClient, normalize_content
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

    Requires environment variables:
        AZURE_OPENAI_API_KEY: API key
        AZURE_OPENAI_ENDPOINT: Endpoint URL (e.g. https://<resource>.openai.azure.com/)
        AZURE_OPENAI_DEPLOYMENT_NAME: Deployment name
        OPENAI_API_VERSION: API version (e.g. 2025-03-01-preview)

    It reads its endpoint only from the environment, so ``LLMConfig`` rejects a
    ``base_url`` for ``provider: azure``. To reach a resource's v1 surface with a
    host key, use ``provider: openai`` with
    ``base_url: https://<resource>.openai.azure.com/openai/v1/`` instead.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured AzureChatOpenAI instance."""
        self.warn_if_unknown_model()

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", self.model),
        }
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            llm_kwargs["azure_endpoint"] = _base_endpoint(endpoint)

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedAzureChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Azure accepts any deployed model name."""
        return True
