from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional
import warnings

from langchain_core.exceptions import OutputParserException

# The warnings `warn_once` already showed in this process. Python's own once-per-location
# registry is cleared whenever `warnings.catch_warnings()` is entered, which langchain-core
# does when binding tools, so without this a warning repeats on every agent invocation.
_warned: set[str] = set()


def warn_once(message: str, *, stacklevel: int = 2) -> None:
    """Warn `message` as a `UserWarning` the first time this process sees it.

    Args:
        message (`str`):
            The full warning text, never holding secret values.
        stacklevel (`int`, *optional*, defaults to `2`):
            As for `warnings.warn`, counted from the caller.
    """
    if message in _warned:
        return
    _warned.add(message)
    warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)


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


def drop_environment_headers(
    client: Any, explicit: Iterable[str], variables: tuple[str, ...]
) -> None:
    """Remove the headers an SDK client took from the environment when it was built.

    Keeps only the custom headers whose name is exactly (case-sensitive) one of
    `explicit`, and sets `organization`/`project` to `None` and
    `_ambient_authorizations` to `frozenset()` where the client has them. Warns with a
    `UserWarning` naming the dropped headers (not their values) when it drops any, once
    per process for each distinct message.

    Args:
        client (`Any`):
            An OpenAI or Anthropic SDK client, changed in place. `None` is skipped.
        explicit (`Iterable[str]`):
            Names of the custom headers the caller passed itself.
        variables (`tuple[str, ...]`):
            The environment variables this SDK reads headers from, for the warning.
    """
    if client is None:
        return
    keep = set(explicit)
    dropped = {name for name in client._custom_headers if name not in keep}
    # The SDK merges header variables into the private `_custom_headers`, and
    # `copy`/`with_options` re-read the environment, so this filters the built
    # client's dict directly.
    client._custom_headers = {
        name: value for name, value in client._custom_headers.items() if name in keep
    }
    for attribute, header in (
        ("organization", "OpenAI-Organization"), ("project", "OpenAI-Project")
    ):
        if hasattr(client, attribute):
            if getattr(client, attribute) is not None:
                dropped.add(header)
            setattr(client, attribute, None)
    if hasattr(client, "_ambient_authorizations"):
        client._ambient_authorizations = frozenset()
    if not dropped:
        return
    *rest, last = variables
    sources = f"{', '.join(rest)} or {last}" if rest else last
    message = (
        f"yian sends only the headers it sets, so it dropped {', '.join(sorted(dropped))}, "
        f"read from {sources}; flow runs cannot send them. Unset the variable to silence "
        "this; code that builds models with create_llm_client can set headers on the "
        "http_client it passes."
    )
    warn_once(message, stacklevel=2)


class MissingToolCallError(OutputParserException):
    """A structured reply that did not call the schema's tool.

    Its `str` is the message alone: `OutputParserException.__init__` appends
    langchain's troubleshooting link, so this skips that initializer and sets the
    parent's attributes itself.

    Args:
        message (`str`):
            The full error message.
    """

    def __init__(self, message: str):
        Exception.__init__(self, message)
        self.observation = None
        self.llm_output = None
        self.send_to_llm = False


def missing_tool_call(name: str) -> MissingToolCallError:
    """Return the error for a structured reply that did not call the schema's tool.

    The AGENT sends this message as the correction before a retry, and raises the error
    once retries are spent.

    Args:
        name (`str`):
            The name of the schema's tool.

    Returns:
        `MissingToolCallError`:
            The error, an `OutputParserException` whose message names the tool.

    Example:
        >>> str(missing_tool_call("Answer"))
        'the reply did not call the Answer tool; call it with the answer'
    """
    return MissingToolCallError(f"the reply did not call the {name} tool; call it with the answer")


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
