"""Routes: where a model call goes, how it authenticates, and how often it retries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import httpx2

from yian.llm.auth import Auth


@dataclass(frozen=True)
class Route:
    """One provider endpoint and the policy for calling it.

    Attributes:
        provider (`str`): The provider name, used in error messages.
        base_url (`str`): The API base, for example ``"https://api.openai.com/v1"``; a
            trailing slash is allowed.
        path (`str`): The endpoint relative to ``base_url``, for example ``"responses"``
            or ``"chat/completions"``; a leading slash is allowed.
        auth (`Auth`): The credential headers, resolved once per attempt.
        query (`Mapping[str, str]`): Query parameters, for example ``{"api-version":
            "..."}``; stored as a read-only copy.
        headers (`Mapping[str, str]`): Non-secret extra headers; stored as a read-only
            copy. They replace the framing defaults (``Content-Type`` and ``Accept``)
            of the same name, case-insensitively; auth headers of the same name
            replace them.
        max_retries (`int`): Retries after the first attempt; defaults to 2.
        max_retry_after (`float | None`): The longest server-requested delay that still
            allows a retry; `None` allows any. Defaults to 120 seconds.
        timeout (`float | httpx2.Timeout | None`): The request timeout; `None` uses the
            client's default.
    """

    provider: str
    base_url: str
    path: str
    auth: Auth
    query: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    max_retries: int = 2
    max_retry_after: float | None = 120.0
    timeout: float | httpx2.Timeout | None = None

    def __post_init__(self) -> None:
        # Copied so a caller mutating its own dict cannot change a route in use.
        object.__setattr__(self, "query", MappingProxyType(dict(self.query)))
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))

    def url(self) -> str:
        """Join ``base_url`` and ``path`` with exactly one slash.

        The query is not included; httpx2 encodes ``query`` when the request is built.

        Returns:
            `str`: The endpoint URL.
        """
        return f"{self.base_url.rstrip('/')}/{self.path.lstrip('/')}"
