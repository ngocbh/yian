"""The httpx2 client model calls share: no redirects, yian's User-Agent, SDK timeouts.

The client sends only the headers its caller passes plus httpx2's own defaults; unlike
the provider SDKs it reads no ``*_CUSTOM_HEADERS`` or organization variables. httpx2
still honors proxy and CA environment settings (``trust_env``), as the SDKs do, and
reads no ``.netrc`` since no ``NetRCAuth`` is passed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import httpx2


def _yian_version() -> str:
    try:
        return version("yian")
    except PackageNotFoundError:
        return "0+unknown"


USER_AGENT = f"yian/{_yian_version()}"
# The openai and anthropic SDKs' default: ten minutes overall, five seconds to connect.
DEFAULT_TIMEOUT = httpx2.Timeout(600.0, connect=5.0)


def build_client(
    *,
    timeout: float | httpx2.Timeout = DEFAULT_TIMEOUT,
    transport: httpx2.BaseTransport | None = None,
) -> httpx2.Client:
    """Build an httpx2 client for model calls that never follows redirects.

    A redirect could carry the credential headers to another host, so the client
    returns 3xx responses to the caller instead.

    Args:
        timeout (`float | httpx2.Timeout`): The default request timeout; defaults to
            `DEFAULT_TIMEOUT`.
        transport (`httpx2.BaseTransport | None`): A transport to use instead of the
            network, for tests; defaults to None (httpx2's own).

    Returns:
        `httpx2.Client`: A client with ``follow_redirects=False`` and yian's User-Agent.
    """
    return httpx2.Client(
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        transport=transport,
    )


def check_client(name: str, client: httpx2.Client) -> None:
    """Refuse a client that follows redirects.

    Args:
        name (`str`): What the client is called by the caller, for the error message.
        client (`httpx2.Client`): The client to check.

    Raises:
        ValueError: If ``client.follow_redirects`` is true.
    """
    if client.follow_redirects:
        raise ValueError(f"{name} must not follow redirects; build it with follow_redirects=False")
