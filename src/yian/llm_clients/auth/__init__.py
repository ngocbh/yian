"""Provider credentials, private selection storage, and explicit login effects.

This model-access leaf stages credentials before selection is committed. The store
owns atomic credential/default updates and refresh serialization; OpenAI services
own authentication HTTP and browser callback lifetimes. Neither owns a model,
agent loop, conversation, CLI prompt, or native Codex process.
The stdlib HTTP transport guards connected response parsing with the request
deadline and cancellation, owns socket cleanup, and never follows redirects.

Knows about: the standard library, the stdlib-only yian.defaults, the connection
settings checks in yian.llm_clients.connection_settings, and injected HTTP, clock,
and host callbacks.
Never imports: provider SDKs, settings, nodes, memory, compile, runtime, or CLI.
"""

from .common import (
    ApiKeyCredentials,
    AuthError,
    AuthState,
    Credentials,
    HttpResponse,
    OAuthCredentials,
    ProviderSelection,
)
from .openai import OpenAIAuth
from .store import AuthStore

__all__ = [
    "ApiKeyCredentials",
    "AuthError",
    "AuthState",
    "AuthStore",
    "Credentials",
    "HttpResponse",
    "OAuthCredentials",
    "OpenAIAuth",
    "ProviderSelection",
]
