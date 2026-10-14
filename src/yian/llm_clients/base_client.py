from abc import ABC, abstractmethod
from typing import Any, Optional
import warnings


def normalize_content(response):
    """Coerce an LLM response's `.content` to a plain string in place.

    Provider adapters that require plain-text replies use this helper to join text
    blocks and discard reasoning/metadata blocks. OpenAI Responses clients retain
    their original content blocks and do not use this normalization.

    Args:
        response (`Any`):
            A langchain message-like object with a mutable `.content` attribute, either
            a string (left untouched) or a list of block dicts / strings.

    Returns:
        `Any`:
            The same `response` object, with `.content` replaced by the joined text.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


def check_no_redirects(name: str, http_client: Any) -> None:
    """Reject a caller's HTTP client that follows redirects.

    Args:
        name (`str`):
            The keyword the client came in, for the message.
        http_client (`Any`):
            The caller's httpx client.

    Raises:
        `ValueError`:
            Raised when the client follows redirects.
    """
    # The SDKs that use this send the key in a custom header, which httpx re-sends on a
    # cross-origin redirect.
    if getattr(http_client, "follow_redirects", False):
        raise ValueError(
            f"{name} follows redirects, which would send the API key past base_url; "
            "build it with follow_redirects=False"
        )


class BaseLLMClient(ABC):
    """
    Abstract base for provider-specific LLM clients.

    A client wraps one provider's langchain chat model behind a uniform surface: build
    the model with [`get_llm`][yian.llm_clients.base_client.BaseLLMClient.get_llm]
    and check the model id against the provider's known list with
    [`validate_model`][yian.llm_clients.base_client.BaseLLMClient.validate_model].
    Subclasses implement both; the factory picks the subclass.

    Args:
        model (`str`):
            The provider-specific model id (e.g. `"claude-sonnet-4-5"`).
        base_url (`str`, *optional*, defaults to `None`):
            Override the provider's API endpoint; `None` uses the provider default.
        **kwargs:
            Extra provider-specific arguments, stashed on `self.kwargs` for `get_llm`.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs: Any):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_provider_name(self) -> str:
        """Return the provider name used in warning messages."""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """Warn when the model is outside the known list for the provider."""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass
