from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Optional
import warnings

from langchain_core.exceptions import OutputParserException
from pydantic import SecretStr

from .connection_settings import validate_headers, validate_organization

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


# What a redirect would leak, for `check_no_redirects`: from a model with a base_url, and
# from a model that sends connection headers.
BASE_URL_LEAK = "the API key past base_url"
HEADERS_LEAK = "the API key or connection headers to another host"
# The headers that carry `provider: openai`'s organization and project ids, lowercase ->
# the `llm_config` field they come from.
_OPENAI_ID_HEADERS = {"openai-organization": "organization", "openai-project": "project"}
# The environment variables the OpenAI SDK reads the organization and project ids from.
_OPENAI_ID_VARIABLES = frozenset({"OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID"})


def check_connection_headers(headers: Any, provider: str) -> dict[str, SecretStr]:
    """Check the connection headers a caller gave a client and return them as secrets.

    `create_llm_client` callers skip `LLMConfig`, so each client checks them itself. For
    ``openai``, ``OpenAI-Organization`` and ``OpenAI-Project`` carry the organization and
    project ids (as `model_from_config` passes them) and are checked as ids; every other
    header is checked by `validate_headers`.

    Args:
        headers (`Any`): The candidate mapping of header name to `str` or `SecretStr`.
        provider (`str`): The provider the headers go to.

    Returns:
        `dict[str, SecretStr]`: The headers, each value a `SecretStr`.

    Raises:
        `ValueError`: ``headers`` is not a mapping, or a name, value or id is refused;
            the message never holds a value.
    """
    if not isinstance(headers, Mapping):
        raise ValueError("connection_headers must be a mapping of header name to value")
    ids: dict[str, Any] = {}
    others: dict[Any, Any] = {}
    for name, value in headers.items():
        is_id = provider == "openai" and isinstance(name, str)
        (ids if is_id and name.lower() in _OPENAI_ID_HEADERS else others)[name] = value
    checked = validate_headers(others, provider)
    seen = {name.lower() for name in checked}
    for name, value in ids.items():
        if name.lower() in seen:
            raise ValueError(f"header {name} is given more than once (names ignore case)")
        seen.add(name.lower())
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        checked[name] = validate_organization(value, _OPENAI_ID_HEADERS[name.lower()])
    return {name: SecretStr(value) for name, value in checked.items()}


def check_no_redirects(name: str, http_client: Any, reason: str = BASE_URL_LEAK) -> None:
    """Reject a caller's HTTP client that follows redirects.

    Args:
        name (`str`):
            The keyword the client came in, for the message.
        http_client (`Any`):
            The caller's httpx client.
        reason (`str`, *optional*, defaults to `"the API key past base_url"`):
            What a redirect would send where, for the message.

    Raises:
        `ValueError`:
            Raised when the client follows redirects.
    """
    # The SDKs that use this send the key (and any connection headers) in custom
    # headers, which httpx re-sends on a cross-origin redirect.
    if getattr(http_client, "follow_redirects", False):
        raise ValueError(
            f"{name} follows redirects, which would send {reason}; "
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
    per process for each distinct message. A header or organization or project the
    client read counts as dropped only when `explicit` has no header of that name (in
    any case), since that header replaces it. The warning names
    ``llm_config.organization`` / ``project`` only when `variables` holds the OpenAI
    SDK's id variables.

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
    replaced = {name.lower() for name in keep}
    # A header named like a kept one in another case is replaced, not dropped, though the
    # filter below is exact-case.
    dropped = {name for name in client._custom_headers if name.lower() not in replaced}
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
            if getattr(client, attribute) is not None and header.lower() not in replaced:
                dropped.add(header)
            setattr(client, attribute, None)
    if hasattr(client, "_ambient_authorizations"):
        client._ambient_authorizations = frozenset()
    if not dropped:
        return
    *rest, last = variables
    sources = f"{', '.join(rest)} or {last}" if rest else last
    ids = (
        " (llm_config.organization and llm_config.project for OpenAI's ids)"
        if _OPENAI_ID_VARIABLES.intersection(variables) else ""
    )
    message = (
        f"yian sends only the headers it sets, so it dropped {', '.join(sorted(dropped))}, "
        f"read from {sources}. To send them, set the host's llm_config.headers{ids}, save "
        "them with yian auth, or pass connection_headers to create_llm_client; unset the "
        "variable to silence this."
    )
    warn_once(message)


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
