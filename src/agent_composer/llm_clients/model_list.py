"""List the model ids a provider's host offers, for model pickers and key checks.

A listing uses only the key and base URL passed in, never a key from the
environment, so a host can try a key before it saves it. Providers echo rejected
keys in error bodies, and httpx exceptions can carry the request and its headers,
so a failure raises a `ModelListError` with a short fixed reason and nothing
chained to it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import time
from typing import Any, Optional

import httpx

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

_Fetch = Callable[[str, dict[str, str]], Any]


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
    transport: Optional[httpx.BaseTransport] = None,
) -> Optional[tuple[str, ...]]:
    """Return the model ids the provider's host offers, sorted and without duplicates.

    Anthropic lists `{base}/v1/models` and Google `{base}/v1beta/models`, both
    following every page; Google keeps only models that support `generateContent`,
    without the `models/` prefix. OpenAI-compatible providers list `{base}/models`.
    Ollama lists its local tags at `{base}/api/tags`. Redirects are not followed, so
    the key goes only to the host asked.

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
            Ollama URL is dropped. Environment base-URL overrides are not read.
        timeout (`float`, *optional*, defaults to `8`):
            Seconds for the whole listing, shared by every page.
        transport (`httpx.BaseTransport`, *optional*, defaults to `None`):
            Transport for the requests, such as an `httpx.MockTransport` in tests.
            `None` uses the network.

    Returns:
        `tuple[str, ...] | None`:
            The model ids, or `None` for `azure`, which has no model listing.

    Raises:
        `ModelListError`:
            Raised for an unsupported provider, a non-2xx status, an unreachable host,
            an exhausted timeout, or a response that is not a model list.

    Example:
        >>> list_models("openrouter", api_key)  # doctest: +SKIP
        ('anthropic/claude-sonnet-4.5', 'openai/gpt-5', ...)
    """
    name = provider.lower()
    headers: dict[str, str] = {}
    pages: Callable[[_Fetch, str], Iterator[Any]]
    if name == "azure":
        return None
    if name == "anthropic":
        base, pages = base_url or _ANTHROPIC_BASE_URL, _anthropic_ids
        headers["anthropic-version"] = _ANTHROPIC_VERSION
        if api_key:
            headers["x-api-key"] = api_key
    elif name == "google":
        base, pages = base_url or _GOOGLE_BASE_URL, _google_ids
        if api_key:
            headers["x-goog-api-key"] = api_key
    elif name == "ollama":
        base, pages = base_url or _OLLAMA_BASE_URL, _ollama_ids
        base = base.rstrip("/")
        base = base[: -len("/v1")] if base.endswith("/v1") else base
    elif name in _OPENAI_COMPATIBLE:
        base = base_url or _PROVIDER_BASE_URL.get(name, _OPENAI_BASE_URL)
        pages = _openai_ids
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise ModelListError("unsupported provider")

    deadline = time.monotonic() + timeout
    invalid = False
    with httpx.Client(transport=transport, follow_redirects=False) as client:
        try:
            ids = set(pages(lambda url, params: _get_json(client, url, params, headers, deadline),
                            base.rstrip("/")))
        except (KeyError, TypeError, AttributeError):
            invalid = True
    if invalid or not all(isinstance(model, str) for model in ids):
        # Raised outside the handler so the parse error is not chained as context.
        raise ModelListError("invalid response")
    return tuple(sorted(ids))


def _get_json(
    client: httpx.Client,
    url: str,
    params: dict[str, str],
    headers: dict[str, str],
    deadline: float,
) -> Any:
    """GET `url` and decode its JSON body within what is left before `deadline`."""
    failure = None
    body = bytearray()
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure = "timed out"
        else:
            with client.stream("GET", url, params=params, headers=headers, timeout=remaining) as response:
                if not response.is_success:
                    failure = f"HTTP {response.status_code}"
                else:
                    for chunk in response.iter_bytes():
                        if time.monotonic() > deadline:
                            failure = "timed out"
                            break
                        body += chunk
    except httpx.TimeoutException:
        failure = "timed out"
    except httpx.HTTPError:
        failure = "can't reach host"
    except (httpx.InvalidURL, ValueError):
        # A malformed base URL or a header value httpx cannot encode.
        failure = "invalid request"
    if failure is None:
        try:
            return json.loads(body)
        except ValueError:
            failure = "invalid response"
    # Raised outside every handler so no httpx exception is chained as context.
    raise ModelListError(failure)


def _anthropic_ids(fetch: _Fetch, base: str) -> Iterator[Any]:
    """Yield model ids from every page of Anthropic's model listing."""
    params = {"limit": _PAGE_SIZE}
    while True:
        page = fetch(base + "/v1/models", params)
        yield from (model["id"] for model in page["data"])
        if not page.get("has_more"):
            return
        params = {"limit": _PAGE_SIZE, "after_id": page["last_id"]}


def _google_ids(fetch: _Fetch, base: str) -> Iterator[Any]:
    """Yield the `generateContent` model ids from every page of Google's listing."""
    params = {"pageSize": _PAGE_SIZE}
    while True:
        page = fetch(base + "/v1beta/models", params)
        for model in page.get("models", ()):
            if "generateContent" in model.get("supportedGenerationMethods", ()):
                name = model["name"]
                yield name.removeprefix("models/") if isinstance(name, str) else name
        token = page.get("nextPageToken")
        if not token:
            return
        params = {"pageSize": _PAGE_SIZE, "pageToken": token}


def _openai_ids(fetch: _Fetch, base: str) -> Iterator[Any]:
    """Yield model ids from an OpenAI-compatible `/models` listing."""
    yield from (model["id"] for model in fetch(base + "/models", {})["data"])


def _ollama_ids(fetch: _Fetch, base: str) -> Iterator[Any]:
    """Yield the local model tags from Ollama's native API."""
    yield from (model["name"] for model in fetch(base + "/api/tags", {})["models"])
