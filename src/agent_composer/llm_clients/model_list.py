"""List the model ids a provider's host offers, for model pickers and key checks.

A listing uses only the key and base URL passed in, never a key from the
environment, so a host can try a key before it saves it. Providers echo rejected
keys in error bodies, so every failure raises a `ModelListError` with a short
fixed reason and nothing chained to it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
import math
import time
from typing import Any, Optional
from urllib.parse import urlencode, urlsplit

from .auth.common import HttpResponse, _text
from .auth.http import _request
from .config import validate_base_url
from .connection import _API_ENDPOINT as _OPENAI_BASE_URL
from .factory import _OPENAI_COMPATIBLE, _PROVIDER_BASE_URL

_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com"
# Mirror of `ollama_client._DEFAULT_BASE_URL`, which cannot be imported without the
# ollama extra.
_OLLAMA_BASE_URL = "http://localhost:11434"
# The largest page Anthropic and Google serve, so most listings take one request.
_PAGE_SIZE = "1000"
# OpenRouter's listing carries a description and pricing for every model.
_BODY_LIMIT = 16 * 1024 * 1024

_Fetch = Callable[[str, dict], dict]


class ModelListError(Exception):
    """A model listing failed.

    The message is one of `HTTP <status>`, `can't reach host`, `timed out`,
    `invalid request`, `invalid response`, or `unsupported provider`; it never holds
    the key, the URL, or the response body.
    """


def list_models(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    timeout: float = 8,
    request: Optional[Callable[..., HttpResponse]] = None,
) -> Optional[tuple[str, ...]]:
    """Return the model ids the provider's host offers, sorted and without duplicates.

    Anthropic lists `{base}/v1/models` and Google `{base}/v1beta/models`, both
    following every page; Google keeps only models that support `generateContent`,
    without the `models/` prefix. OpenAI-compatible providers list `{base}/models`.
    Ollama lists its local tags at `{base}/api/tags`. Redirects are refused, so the
    key goes only to the host asked.

    Args:
        provider (`str`):
            Provider name, case-insensitive: `anthropic`, `google`, `ollama`, `azure`,
            or any provider routed through the OpenAI client (`openai`, `openrouter`,
            `xai`, `vllm`, and so on).
        api_key (`str`, *optional*, defaults to `None`):
            Key sent as the provider's auth header. `None` sends no auth header; no
            key is read from the environment. Ollama never sends one.
        base_url (`str`, *optional*, defaults to `None`):
            Host to list instead of the provider default. For Anthropic and Google it
            is the API root without the version path; for OpenAI-compatible providers
            it ends where `/models` is appended (often `/v1`). A trailing `/v1` on an
            Ollama URL is dropped. When a key is sent, the URL must pass
            `validate_base_url`. Environment base-URL overrides are not read.
        timeout (`float`, *optional*, defaults to `8`):
            Seconds for the whole listing, shared by every page. Must be positive.
        request (`Callable[..., HttpResponse]`, *optional*, defaults to `None`):
            Sends one GET, given the method, URL, and `headers` and `timeout`
            keywords; tests inject a fake. `None` uses the stdlib transport in
            `auth.http`, which refuses redirects.

    Returns:
        `tuple[str, ...] | None`:
            The model ids, or `None` for `azure`, which has no model listing.

    Raises:
        `ModelListError`:
            Raised for an unsupported provider, an invalid key, URL, or timeout, a
            non-2xx status, an unreachable host, an exhausted timeout, or a response
            that is not a model list.

    Example:
        >>> list_models("openrouter", api_key)  # doctest: +SKIP
        ('anthropic/claude-sonnet-4.5', 'openai/gpt-5', ...)
    """
    name = provider.lower() if isinstance(provider, str) else ""
    if name == "azure":
        return None
    headers = {"User-Agent": "agent-composer"}
    if name == "anthropic":
        base, pages = base_url or _ANTHROPIC_BASE_URL, _anthropic_ids
        headers["anthropic-version"] = _ANTHROPIC_VERSION
        auth = {"x-api-key": api_key}
    elif name == "google":
        base, pages = base_url or _GOOGLE_BASE_URL, _google_ids
        auth = {"x-goog-api-key": api_key}
    elif name == "ollama":
        base, pages, auth = base_url or _OLLAMA_BASE_URL, _ollama_ids, {}
    elif name in _OPENAI_COMPATIBLE:
        # Only openai itself defaults to OpenAI's host; a routed provider with no
        # endpoint of its own must not send its key there.
        default = _OPENAI_BASE_URL if name == "openai" else _PROVIDER_BASE_URL.get(name)
        if not (base_url or default):
            raise ModelListError("unsupported provider")
        base, pages = base_url or default, _openai_ids
        auth = {"Authorization": f"Bearer {api_key}"}
    else:
        raise ModelListError("unsupported provider")

    sends_key = bool(api_key) and bool(auth)
    if not _valid_request(base, api_key, timeout, check_url=sends_key and base_url is not None):
        raise ModelListError("invalid request")
    if sends_key:
        headers.update(auth)
    base = base.rstrip("/")
    if name == "ollama" and base.endswith("/v1"):
        base = base[: -len("/v1")]

    fetch = partial(
        _fetch, request or partial(_request, body_limit=_BODY_LIMIT),
        base, headers, time.monotonic() + timeout,
    )
    invalid = False
    try:
        ids = set(pages(fetch))
    except (KeyError, TypeError, AttributeError):
        invalid = True
    if invalid or not all(isinstance(model, str) for model in ids):
        # Raised outside the handler so the parse error is not chained as context.
        raise ModelListError("invalid response")
    return tuple(sorted(ids))


def _valid_request(base: Any, api_key: Any, timeout: Any, *, check_url: bool) -> bool:
    """Say whether the URL, key, and timeout can make a request."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        return False
    if api_key is not None and not isinstance(api_key, str) or not isinstance(base, str):
        return False
    if api_key and not _text(api_key):
        # A header value http.client would refuse with the key in its message.
        return False
    try:
        if check_url:
            # The key goes to this host, so hold it to the rules for a config base_url.
            validate_base_url(base)
        parts = urlsplit(base)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _fetch(
    request: Callable[..., HttpResponse],
    base: str,
    headers: dict,
    deadline: float,
    path: str,
    params: dict,
) -> dict:
    """GET `base + path` within what is left before `deadline` and return its JSON object."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModelListError("timed out")
    url = base + path + (f"?{urlencode(params)}" if params else "")
    failure = response = None
    try:
        response = request("GET", url, headers=headers, timeout=remaining)
    except TimeoutError:
        failure = "timed out"
    except ValueError:
        failure = "invalid response"
    except Exception:
        failure = "can't reach host"
    # Raised outside every handler so no transport exception is chained as context.
    if failure is not None:
        raise ModelListError(failure)
    if type(response) is not HttpResponse or type(response.status_code) is not int:
        raise ModelListError("invalid response")
    if not 200 <= response.status_code < 300:
        raise ModelListError(f"HTTP {response.status_code}")
    if type(response.data) is not dict:
        raise ModelListError("invalid response")
    return response.data


def _anthropic_ids(fetch: _Fetch) -> Iterator[Any]:
    """Yield model ids from every page of Anthropic's model listing."""
    params = {"limit": _PAGE_SIZE}
    while True:
        page = fetch("/v1/models", params)
        yield from (model["id"] for model in page["data"])
        if not page.get("has_more"):
            return
        params = {"limit": _PAGE_SIZE, "after_id": page["last_id"]}


def _google_ids(fetch: _Fetch) -> Iterator[Any]:
    """Yield the `generateContent` model ids from every page of Google's listing."""
    params = {"pageSize": _PAGE_SIZE}
    while True:
        page = fetch("/v1beta/models", params)
        for model in page.get("models", ()):
            if "generateContent" in model.get("supportedGenerationMethods", ()):
                yield model["name"].removeprefix("models/")
        token = page.get("nextPageToken")
        if not token:
            return
        params = {"pageSize": _PAGE_SIZE, "pageToken": token}


def _openai_ids(fetch: _Fetch) -> Iterator[Any]:
    """Yield model ids from an OpenAI-compatible `/models` listing."""
    yield from (model["id"] for model in fetch("/models", {})["data"])


def _ollama_ids(fetch: _Fetch) -> Iterator[Any]:
    """Yield the local model tags from Ollama's native API."""
    yield from (model["name"] for model in fetch("/api/tags", {})["models"])
