"""Snapshot model defaults and nonsecret connection identities at host boundaries.

Auth storage owns credentials and refresh. This module resolves selection and
provides injected credential callables to model transports. Recorded metadata is
internal, never a YAML option, and never contains a token or API key. Compilation
may use the detached snapshot without authentication effects.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import os
from typing import Any
from urllib.parse import urlsplit

from agent_composer._settings import DEFAULT_PROVIDER, PROVIDER_DEFAULT_MODELS
from .auth import ApiKeyCredentials, AuthError, AuthStore, OAuthCredentials, OpenAIAuth
from .config import LLMConfig

_API_ENDPOINT = "https://api.openai.com/v1"
_SUBSCRIPTION_ENDPOINT = "https://chatgpt.com/backend-api/codex"
_FIELDS = {"method", "source", "credential_id", "endpoint", "store_path"}


@dataclass(frozen=True)
class ConnectionSnapshot:
    """One detached host selection shared across every static agent in a run."""

    defaults: dict[str, Any]
    provider_models: dict[str, str]
    connections: dict[str, dict[str, Any]]
    api_endpoint: str = _API_ENDPOINT

    def bind(
        self,
        effective: dict[str, dict[str, Any]],
        *,
        host_key_provider: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Attach selected connection identity without changing any input maps.

        Agents of ``host_key_provider`` authenticate with a key the host supplies
        outside the map, so an OpenAI agent records a ``host`` connection instead
        of the saved or environment one.
        """
        result = copy.deepcopy(effective)
        for config in result.values():
            if config["provider"] == "openai" and host_key_provider == "openai":
                config["_connection"] = self.host_connection()
                continue
            connection = self.connections.get(config["provider"])
            if connection is not None:
                config["_connection"] = copy.deepcopy(connection)
        return result

    def host_connection(self) -> dict[str, Any]:
        """Return the nonsecret identity of an OpenAI key supplied by the host."""
        return validate_connection({
            "method": "api_key",
            "source": "host",
            "credential_id": None,
            "endpoint": self.api_endpoint,
            "store_path": None,
        })


def _key_identity(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def connection_snapshot() -> ConnectionSnapshot:
    """Read saved choices and environment once, without resolving credentials."""
    store = AuthStore()
    state = store.read()
    environment = dict(os.environ)
    provider = (
        environment.get("AGENT_COMPOSER_DEFAULT_PROVIDER")
        or state.default_provider
        or DEFAULT_PROVIDER
    ).lower()
    models = dict(PROVIDER_DEFAULT_MODELS)
    connections = {}
    for selection in state.selections:
        models[selection.provider] = selection.model
        if selection.provider == "openai":
            connections[selection.provider] = {
                "method": selection.method,
                "source": "store",
                "credential_id": selection.credential_id,
                "endpoint": (
                    _SUBSCRIPTION_ENDPOINT
                    if selection.method == "subscription"
                    else environment.get("OPENAI_BASE_URL") or _API_ENDPOINT
                ),
                "store_path": str(store.path),
            }
    if environment.get("AGENT_COMPOSER_DEFAULT_MODEL"):
        models[provider] = environment["AGENT_COMPOSER_DEFAULT_MODEL"]
    if "openai" not in connections:
        key = environment.get("OPENAI_API_KEY")
        connections["openai"] = {
            "method": "api_key",
            "source": "environment" if key else "missing",
            "credential_id": _key_identity(key) if key else None,
            "endpoint": environment.get("OPENAI_BASE_URL") or _API_ENDPOINT,
            "store_path": None,
        }
    defaults = {"provider": provider}
    if provider in models:
        defaults["model"] = models[provider]
    connections = {
        provider: validate_connection(value) for provider, value in connections.items()
    }
    return ConnectionSnapshot(
        defaults,
        models,
        connections,
        environment.get("OPENAI_BASE_URL") or _API_ENDPOINT,
    )


def validate_connection(value: Any) -> dict[str, Any]:
    """Validate the exact nonsecret metadata schema before use or replay."""
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("Invalid internal connection metadata fields")
    method, source = value["method"], value["source"]
    if method not in ("api_key", "subscription") or source not in (
        "store",
        "environment",
        "missing",
        "legacy_environment",
        "host",
    ):
        raise ValueError("Invalid internal connection method or source")
    if type(value["endpoint"]) is not str:
        raise ValueError("Invalid internal connection endpoint")
    endpoint = urlsplit(value["endpoint"])
    if (
        endpoint.scheme not in ("https", "http")
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError(
            "Connection endpoint must be an HTTP URL without credentials, query, or fragment"
        )
    identity, path = value["credential_id"], value["store_path"]
    if source == "store":
        if (
            type(identity) is not str
            or not identity
            or type(path) is not str
            or not os.path.isabs(path)
        ):
            raise ValueError("Invalid stored connection identity")
    elif (
        path is not None
        or (
            source == "environment"
            and (type(identity) is not str or len(identity) != 64)
        )
        or (source != "environment" and identity is not None)
    ):
        raise ValueError("Invalid environment connection identity")
    if method == "subscription" and (
        source != "store" or value["endpoint"] != _SUBSCRIPTION_ENDPOINT
    ):
        raise ValueError("Invalid subscription connection routing")
    return copy.deepcopy(value)


def split_model_config(
    config: dict[str, Any] | LLMConfig,
) -> tuple[LLMConfig, dict[str, Any] | None]:
    """Validate public fields separately from host-owned connection metadata."""
    if isinstance(config, LLMConfig):
        raw = config.model_dump(exclude_none=True)
        # Dumps leave the key out; carry it across explicitly.
        if config.api_key is not None:
            raw["api_key"] = config.api_key
    else:
        raw = dict(config)
    metadata = raw.pop("_connection", None)
    validated = LLMConfig(**raw)
    if metadata is not None:
        if validated.provider != "openai":
            raise ValueError("Internal connection metadata requires provider openai")
        metadata = validate_connection(metadata)
    return validated, metadata


def resolve_model_config(
    config: dict[str, Any] | LLMConfig, *, snapshot: ConnectionSnapshot | None = None
) -> dict[str, Any]:
    """Resolve a fresh config, preserving any existing pinned connection."""
    validated, metadata = split_model_config(config)
    raw = validated.model_dump(exclude_none=True)
    if validated.api_key is not None:
        raw["api_key"] = validated.api_key
    if validated.provider and validated.model and validated.provider != "openai":
        return raw
    if metadata is not None and validated.model:
        raw["_connection"] = metadata
        return raw
    snapshot = snapshot or connection_snapshot()
    provider = validated.provider or snapshot.defaults["provider"]
    model = validated.model or snapshot.provider_models.get(provider)
    if not model:
        raise ValueError(
            f"Provider {provider!r} needs an explicit model; set llm_config.model or --model"
        )
    raw.update(provider=provider, model=model)
    if metadata is not None:
        raw["_connection"] = metadata
    elif provider == "openai" and validated.api_key is not None:
        raw["_connection"] = snapshot.host_connection()
    elif provider in snapshot.connections:
        raw["_connection"] = copy.deepcopy(snapshot.connections[provider])
    return raw


def legacy_connection_config(config: dict[str, Any]) -> dict[str, Any]:
    """Pin old OpenAI records to their original API-key route without I/O."""
    result = copy.deepcopy(config)
    # Old reflective snapshots serialized every optional public field, including
    # these empty retired defaults. They carried no native execution behavior.
    if result.get("provider") != "codex":
        for field in ("sandbox", "cwd"):
            if field in result and result[field] is None:
                del result[field]
    if result.get("provider") == "openai" and "_connection" not in result:
        result["_connection"] = {
            "method": "api_key",
            "source": "legacy_environment",
            "credential_id": None,
            "endpoint": _API_ENDPOINT,
            "store_path": None,
        }
    return result


def legacy_connection_map(
    effective: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Adapt a recorded effective map at the host replay boundary."""
    return {
        path: legacy_connection_config(config) for path, config in effective.items()
    }


def connection_credentials(
    connection: dict[str, Any],
) -> ApiKeyCredentials | OAuthCredentials:
    """Resolve fresh credentials while enforcing the recorded connection identity."""
    connection = validate_connection(connection)
    source = connection["source"]
    if source == "store":
        try:
            credentials = AuthStore(connection["store_path"]).credentials(
                "openai",
                connection["credential_id"],
                refresh=OpenAIAuth().refresh,
            )
        except AuthError as exc:
            raise AuthError(f"{exc}. Run 'composer auth openai' and start a new run") from exc
        if credentials.method != connection["method"]:
            raise AuthError(
                "The saved connection method changed; run 'composer auth openai' and start a new run"
            )
        return credentials
    if source == "host":
        raise AuthError(
            "This run's OpenAI API key came from the host's llm_config.api_key and is not "
            "recorded; start a new run with the key"
        )
    if source == "missing":
        raise AuthError(
            "No OpenAI connection was selected; run 'composer auth openai' or set OPENAI_API_KEY before starting a new run"
        )
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise AuthError(
            "This run needs its OpenAI API key in OPENAI_API_KEY; use 'composer auth openai' for a new run"
        )
    if source == "environment" and _key_identity(key) != connection["credential_id"]:
        raise AuthError(
            "The OpenAI environment API key changed; restore it or start a new run"
        )
    return ApiKeyCredentials(key)
