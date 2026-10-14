"""Snapshot model defaults and nonsecret connection identities at host boundaries.

There is no built-in provider or model: defaults come only from the environment
(``YIAN_DEFAULT_PROVIDER``/``YIAN_DEFAULT_MODEL``) and the saved logins. Auth
storage owns credentials and refresh. This module resolves selection, records
on every keyed agent which connection it uses (the saved login or the
provider's key environment variable; OpenAI also a host key, subscription, or
missing key), and provides injected credential callables to model transports.
Recorded metadata is internal, never a YAML option, and never contains a token
or API key. Compilation may use the detached snapshot without authentication
effects.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import os
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from .auth import (
    ApiKeyCredentials,
    AuthError,
    AuthStore,
    OAuthCredentials,
    OpenAIAuth,
    ProviderSelection,
)
from .config import (
    AZURE_NEEDS_BASE_URL,
    RETIRED_LLM_CONFIG_KEYS,
    LLMConfig,
    SUPPORTED_LLM_PROVIDERS,
    azure_resource_endpoint,
    validate_base_url,
)
from .provider_blocks import block_schema

_API_ENDPOINT = "https://api.openai.com/v1"
_SUBSCRIPTION_ENDPOINT = "https://chatgpt.com/backend-api/codex"
_FIELDS = {"method", "source", "credential_id", "endpoint", "store_path"}
# Providers whose agents authenticate with a key and so carry a connection.
_KEYED = frozenset(SUPPORTED_LLM_PROVIDERS) - {"ollama"}
# Providers whose saved login may hold no credential: Ollama sends none, and vLLM
# runs keyless unless its server was started with a key.
_KEYLESS_LOGINS = frozenset({"ollama", "vllm"})
_ENVIRONMENT_CONNECTION = {
    "method": "api_key",
    "source": "environment",
    "credential_id": None,
    "endpoint": None,
    "store_path": None,
}
# Recorded on each agent that authenticates with a key the host supplies outside
# the record, so a later model call without that key fails instead of falling
# back to the environment.
HOST_KEY_MARKER = "_host_key"
_HOST_KEY_MISSING = (
    "This run's {provider} API key came from the host's llm_config.api_key and is not "
    "recorded; pass api_key to resume_flow or RunResult.backward"
)


def missing_model_message(
    path: str | None,
    provider: str | None,
    *,
    stray_model: bool = False,
    label: str | None = None,
) -> str:
    """Say which agent has no model and the three ways to set one.

    Args:
        path (`str | None`): The agent's static path, or `None` outside a flow.
        provider (`str | None`): The selected provider, if one is set.
        stray_model (`bool`, *optional*, defaults to `False`): Whether
            ``YIAN_DEFAULT_MODEL`` is set without a default provider.
        label (`str | None`, *optional*, defaults to `None`): The authored site of
            a synthesized agent's ``llm_config``, named instead of its path.

    Returns:
        `str`: The error message.
    """
    if label is not None:
        agent = label[:1].upper() + label[1:]
    elif path:
        agent = f"Agent {path!r}"
    else:
        agent = None
    if provider is None:
        subject = f"{agent} has no provider or model" if agent else "No model is set"
        login = "<provider>"
    else:
        subject = (
            f"{agent} has no model for provider {provider!r}"
            if agent
            else f"No model is set for provider {provider!r}"
        )
        login = provider
    message = (
        f"{subject}; set llm_config.provider/model, pass --provider/--model, "
        f"or run 'yian auth {login}'"
    )
    if stray_model:
        message += (
            "; YIAN_DEFAULT_MODEL is set but names no provider; "
            "set YIAN_DEFAULT_PROVIDER or run 'yian auth <provider>'"
        )
    return message


@dataclass(frozen=True)
class ConnectionSnapshot:
    """One detached host selection shared across every static agent in a run.

    ``defaults`` holds the selected default provider and its model, either of which
    may be absent. ``connections`` holds one connection per keyed provider: the
    saved login, else the provider's key environment variable (for OpenAI, the
    environment key or a missing one). A saved Azure login's connection carries
    its resource endpoint. ``stray_model`` records a ``YIAN_DEFAULT_MODEL`` that
    names no provider, so an agent that needs a default can say why it was ignored.
    """

    defaults: dict[str, Any]
    provider_models: dict[str, str]
    connections: dict[str, dict[str, Any]]
    api_endpoint: str = _API_ENDPOINT
    stray_model: bool = False

    def bind(
        self,
        effective: dict[str, dict[str, Any]],
        *,
        host_key_provider: str | None = None,
        host_base_url: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Attach selected connection identity without changing any input maps.

        Every keyed agent without a host key records its provider's connection
        from the snapshot; an ``ollama`` agent records none. An ``azure`` agent with a
        saved login and no ``base_url`` gets the login's endpoint as its
        ``base_url``; one whose ``base_url`` names another resource records the
        environment connection instead, since the saved key belongs to the saved
        resource.

        Agents of ``host_key_provider`` authenticate with a key the host supplies
        outside the map. Each records the nonsecret ``HOST_KEY_MARKER``, and an
        OpenAI agent records a ``host`` connection instead of the saved or
        environment one; other providers record no connection. ``host_base_url``,
        the endpoint the host gave with that key, is recorded on those agents
        only, and is the OpenAI host connection's endpoint.
        """
        result = copy.deepcopy(effective)
        for config in result.values():
            if config["provider"] == host_key_provider:
                config[HOST_KEY_MARKER] = True
                if host_base_url is not None:
                    config["base_url"] = host_base_url
                if config["provider"] == "openai":
                    config["_connection"] = self.host_connection(host_base_url)
                continue
            self._attach(config)
        return result

    def _attach(self, config: dict[str, Any]) -> None:
        """Record the provider's snapshot connection on one config without a host key.

        Args:
            config (`dict[str, Any]`): A mutable config with its ``provider`` set.
                An ``azure`` config may get its ``base_url`` from the saved login.
        """
        provider = config["provider"]
        connection = self.connections.get(provider)
        if connection is None:
            return
        connection = validate_connection(connection, provider)
        if provider == "azure" and connection["source"] == "store":
            base_url = config.get("base_url")
            if base_url is None:
                config["base_url"] = connection["endpoint"]
            elif azure_resource_endpoint(base_url) != azure_resource_endpoint(
                connection["endpoint"]
            ):
                # The saved key belongs to its own resource; another one reads
                # AZURE_OPENAI_API_KEY as before.
                connection = copy.deepcopy(_ENVIRONMENT_CONNECTION)
        config["_connection"] = connection

    def host_connection(self, endpoint: str | None = None) -> dict[str, Any]:
        """Return the nonsecret identity of an OpenAI key supplied by the host.

        Args:
            endpoint (`str | None`, *optional*, defaults to `None`):
                The host's ``base_url`` for that key. `None` uses the snapshot's
                API endpoint.

        Returns:
            `dict[str, Any]`: Validated connection metadata with source ``host``,
                no credential identity, and that endpoint.
        """
        return validate_connection({
            "method": "api_key",
            "source": "host",
            "credential_id": None,
            "endpoint": self.api_endpoint if endpoint is None else endpoint,
            "store_path": None,
        })


def key_identity(key: str) -> str:
    """Return the nonsecret SHA-256 identity recorded for an environment API key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def connection_snapshot() -> ConnectionSnapshot:
    """Read saved choices and environment once, without resolving credentials.

    The default provider is ``YIAN_DEFAULT_PROVIDER``, else the saved default, else
    none. Each saved selection supplies its provider's model; ``YIAN_DEFAULT_MODEL``
    overrides the default provider's. Each keyed provider gets a connection: its
    saved API-key login (or, for OpenAI, subscription), else its key environment
    variable. A saved login for a provider this version does not support is
    skipped unless it is the default.

    Raises:
        `ValueError`: ``YIAN_DEFAULT_PROVIDER`` or the saved default names an
            unsupported provider, or a supported provider's saved login is of a
            kind it cannot use (such as a subscription outside OpenAI, or an
            Azure login without a valid endpoint).
    """
    store = AuthStore()
    state = store.read()
    environment = dict(os.environ)
    env_provider = environment.get("YIAN_DEFAULT_PROVIDER")
    provider = (env_provider or state.default_provider or "").lower() or None
    if env_provider and provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"YIAN_DEFAULT_PROVIDER={env_provider!r} is not a "
            f"supported provider; set it to one of: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
        )
    models = {}
    connections = {}
    for selection in state.selections:
        if selection.provider not in SUPPORTED_LLM_PROVIDERS:
            if not env_provider and selection.provider == state.default_provider:
                raise ValueError(
                    f"The saved default provider {selection.provider!r} is not supported; "
                    f"run 'yian auth {selection.provider} --remove'"
                )
            # A login for a provider this version no longer runs selects nothing.
            continue
        _check_selection(selection)
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
        elif selection.method == "api_key":
            connections[selection.provider] = {
                "method": "api_key",
                "source": "store",
                "credential_id": selection.credential_id,
                "endpoint": selection.endpoint,
                "store_path": str(store.path),
            }
    for keyed in SUPPORTED_LLM_PROVIDERS:
        if keyed in _KEYED and keyed != "openai" and keyed not in connections:
            connections[keyed] = copy.deepcopy(_ENVIRONMENT_CONNECTION)
    stray_model = False
    if environment.get("YIAN_DEFAULT_MODEL"):
        if provider is None:
            stray_model = True
        else:
            models[provider] = environment["YIAN_DEFAULT_MODEL"]
    if "openai" not in connections:
        key = environment.get("OPENAI_API_KEY")
        connections["openai"] = {
            "method": "api_key",
            "source": "environment" if key else "missing",
            "credential_id": key_identity(key) if key else None,
            "endpoint": environment.get("OPENAI_BASE_URL") or _API_ENDPOINT,
            "store_path": None,
        }
    defaults = {}
    if provider is not None:
        defaults["provider"] = provider
        if provider in models:
            defaults["model"] = models[provider]
    connections = {
        name: validate_connection(value, name) for name, value in connections.items()
    }
    return ConnectionSnapshot(
        defaults,
        models,
        connections,
        environment.get("OPENAI_BASE_URL") or _API_ENDPOINT,
        stray_model,
    )


def _check_selection(selection: ProviderSelection) -> None:
    """Check that a supported provider's saved login is a kind it can use.

    Args:
        selection (`ProviderSelection`): A saved login of a supported provider.

    Raises:
        `ValueError`: A provider other than Ollama or vLLM has no credential;
            a provider other than OpenAI has a subscription; a provider other
            than Azure has an endpoint; Azure has no valid endpoint; or Ollama
            has a key.
    """
    provider, method, endpoint = selection.provider, selection.method, selection.endpoint
    invalid = (
        (method == "none" and provider not in _KEYLESS_LOGINS)
        or (provider != "openai" and method == "subscription")
        or (provider != "azure" and endpoint is not None)
        or (provider == "azure" and endpoint is None)
        or (provider == "ollama" and method != "none")
    )
    if not invalid and provider == "azure":
        try:
            validate_base_url(endpoint)
        except ValueError:
            invalid = True
    if invalid:
        raise ValueError(f"The saved {provider} login is invalid; run 'yian auth {provider}' again")


def validate_connection(value: Any, provider: str | None = "openai") -> dict[str, Any]:
    """Validate the exact nonsecret metadata schema before use or replay.

    An OpenAI connection records its endpoint and may be a subscription, a host
    key, or a missing or legacy environment key. Another keyed provider's
    connection is an API key, either a saved login (``store``, pinned by
    credential id and store path, with the resource endpoint for ``azure``) or
    the provider's key environment variable (``environment``, unpinned).

    Args:
        value (`Any`): The candidate metadata.
        provider (`str | None`, *optional*, defaults to ``"openai"``): The
            provider of the config that carries it.

    Returns:
        `dict[str, Any]`: A deep copy of ``value``.

    Raises:
        `ValueError`: The metadata does not fit the schema for ``provider``, or
            ``provider`` takes no connection. Errors never echo values.
    """
    if not isinstance(provider, str) or provider not in _KEYED:
        name = (
            f"provider {provider}"
            if isinstance(provider, str) and provider in SUPPORTED_LLM_PROVIDERS
            else "an unsupported provider"
        )
        raise ValueError(f"Internal connection metadata is not used for {name}")
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("Invalid internal connection metadata fields")
    if provider != "openai":
        return _validate_keyed_connection(value, provider)
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


def _validate_keyed_connection(value: dict[str, Any], provider: str) -> dict[str, Any]:
    """Validate a non-OpenAI keyed provider's connection fields.

    Args:
        value (`dict[str, Any]`): Metadata with exactly the connection fields.
        provider (`str`): A keyed provider other than ``openai``.

    Returns:
        `dict[str, Any]`: A deep copy of ``value``.

    Raises:
        `ValueError`: The method, source, identity, or endpoint is invalid.
    """
    if value["method"] != "api_key" or value["source"] not in ("store", "environment"):
        raise ValueError("Invalid internal connection method or source")
    identity, path, endpoint = value["credential_id"], value["store_path"], value["endpoint"]
    if value["source"] == "environment":
        if identity is not None or path is not None or endpoint is not None:
            raise ValueError("Invalid environment connection identity")
        return copy.deepcopy(value)
    if (
        type(identity) is not str
        or not identity
        or type(path) is not str
        or not os.path.isabs(path)
    ):
        raise ValueError("Invalid stored connection identity")
    if provider == "azure":
        try:
            validate_base_url(endpoint)
        except ValueError:
            raise ValueError("Invalid stored Azure connection endpoint") from None
    elif endpoint is not None:
        raise ValueError("Invalid stored connection endpoint")
    return copy.deepcopy(value)


def split_model_config(
    config: dict[str, Any] | LLMConfig,
) -> tuple[LLMConfig, dict[str, Any] | None]:
    """Validate public fields separately from host-owned connection metadata.

    Raises:
        `AuthError`: The config carries ``HOST_KEY_MARKER`` but no ``api_key``.
        `ValueError`: A public field or the metadata is invalid, an OpenAI
            ``base_url`` differs from the endpoint of its pinned host connection,
            or an Azure ``base_url`` names another resource than its saved login.
    """
    if isinstance(config, LLMConfig):
        raw = config.model_dump(exclude_none=True)
        # Dumps leave the key out; carry it across explicitly.
        if config.api_key is not None:
            raw["api_key"] = config.api_key
    else:
        raw = dict(config)
    if HOST_KEY_MARKER in raw:
        if raw.pop(HOST_KEY_MARKER) is not True:
            raise ValueError("Invalid internal host key marker")
        if raw.get("api_key") is None:
            raise AuthError(_HOST_KEY_MISSING.format(provider=raw.get("provider")))
    metadata = raw.pop("_connection", None)
    validated = LLMConfig(**raw)
    if metadata is not None:
        metadata = validate_connection(metadata, validated.provider)
        if (
            validated.provider == "azure"
            and validated.base_url is not None
            and metadata["source"] == "store"
            and azure_resource_endpoint(validated.base_url)
            != azure_resource_endpoint(metadata["endpoint"])
        ):
            # The saved key belongs to one resource; never send it to another.
            raise ValueError("llm_config.base_url names another Azure resource than its saved login")
        if validated.provider == "openai" and validated.base_url is not None and (
            metadata["source"] != "host" or metadata["endpoint"] != validated.base_url
        ):
            # A base_url serves only the host key; it must match the pinned route.
            raise ValueError("llm_config.base_url must match its host connection endpoint")
    return validated, metadata


def check_model_config(
    config: dict[str, Any] | LLMConfig,
) -> tuple[LLMConfig, dict[str, Any] | None]:
    """Validate a resolved config as `model_from_config` does, without building a model.

    Returns:
        `tuple[LLMConfig, dict[str, Any] | None]`: The validated public fields and
            the pinned connection metadata, if any.

    Raises:
        `AuthError`: The config carries ``HOST_KEY_MARKER`` but no ``api_key``.
        `ValueError`: A field is invalid, the connection rejects a set field, an
            ``azure`` config has no ``base_url``, or the provider's block does not fit
            the API the call actually reaches (e.g. an ``openai`` block with
            ``temperature`` on a ChatGPT subscription connection).
    """
    validated, connection = split_model_config(config)
    if validated.provider == "azure" and validated.base_url is None:
        raise ValueError(AZURE_NEEDS_BASE_URL)
    if validated.provider is not None:
        _check_block(validated, connection)
    return validated, connection


def _check_block(validated: LLMConfig, connection: dict[str, Any] | None) -> None:
    """Check the provider's block against the schema of the API the call reaches.

    Load time checks a block against the provider's usual API; the connection can route
    the call to a narrower one (the ChatGPT subscription endpoint), known only here.

    Args:
        validated (`LLMConfig`): The validated config; its ``provider`` names the block.
        connection (`dict[str, Any] | None`): The pinned connection, if any.

    Raises:
        `ValueError`: The block has a key or value that API refuses; the message is
            the block class's own and never echoes values.
    """
    schema = block_schema(validated.provider, connection, validated.api_key)
    try:
        schema.model_validate(validated.block_for(validated.provider))
    except ValidationError as exc:
        raise ValueError(
            "; ".join(_block_error_text(validated.provider, error) for error in exc.errors())
        ) from None


def _block_error_text(provider: str, error: dict[str, Any]) -> str:
    """Return one block validation error as text naming its key path, without the input."""
    if error["type"] == "value_error":
        return str(error["ctx"]["error"])
    path = ".".join(str(part) for part in error["loc"])
    return f"`{provider}.{path}`: {error['msg']}"


def resolve_model_config(
    config: dict[str, Any] | LLMConfig, *, snapshot: ConnectionSnapshot | None = None
) -> dict[str, Any]:
    """Resolve a fresh config, preserving any existing pinned connection.

    An unset provider or model comes from the snapshot's defaults and saved models.
    A config with a model and a pinned connection, or a non-OpenAI config with a
    provider and model that needs no snapshot connection (keyless, or with its
    own ``api_key``), returns without reading the snapshot. Otherwise an OpenAI
    config with an ``api_key`` gets a ``host`` connection, and any other keyed
    config gets its provider's connection as `ConnectionSnapshot.bind` attaches
    it, including an Azure login's endpoint as ``base_url``.

    Raises:
        `ValueError`: No layer sets the provider or its model, or a saved login
            is invalid.
        `AuthError`: The credential store cannot be read.
    """
    validated, metadata = split_model_config(config)
    raw = validated.model_dump(exclude_none=True)
    if validated.api_key is not None:
        raw["api_key"] = validated.api_key
    if metadata is not None and validated.model:
        raw["_connection"] = metadata
        return raw
    if (
        validated.provider != "openai"
        and validated.provider
        and validated.model
        and (validated.provider not in _KEYED or validated.api_key is not None)
    ):
        return raw
    snapshot = snapshot or connection_snapshot()
    provider = validated.provider or snapshot.defaults.get("provider")
    model = validated.model or (snapshot.provider_models.get(provider) if provider else None)
    if not provider or not model:
        raise ValueError(
            missing_model_message(None, provider, stray_model=snapshot.stray_model)
        )
    raw.update(provider=provider, model=model)
    if metadata is not None:
        raw["_connection"] = metadata
    elif provider == "openai" and validated.api_key is not None:
        raw["_connection"] = snapshot.host_connection(validated.base_url)
    elif validated.api_key is None:
        snapshot._attach(raw)
    return raw


def legacy_connection_config(config: dict[str, Any]) -> dict[str, Any]:
    """Pin old records to their original route without I/O.

    An old OpenAI record gets its legacy environment API-key route. Another keyed
    provider's record, unless its key came from the host, gets the environment
    connection: those records always read the provider's key variable.
    """
    result = copy.deepcopy(config)
    # Old records serialized every optional public field, including these empty
    # retired defaults (sandbox/cwd, and the keys now written in provider blocks).
    # As nulls they configured nothing; a set one still fails validation.
    if result.get("provider") != "codex":
        for field in ("sandbox", "cwd"):
            if field in result and result[field] is None:
                del result[field]
    for field in RETIRED_LLM_CONFIG_KEYS:
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
    elif (
        result.get("provider") in _KEYED
        and "_connection" not in result
        and HOST_KEY_MARKER not in result
    ):
        result["_connection"] = copy.deepcopy(_ENVIRONMENT_CONNECTION)
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
            raise AuthError(f"{exc}. Run 'yian auth openai' and start a new run") from exc
        if credentials.method != connection["method"]:
            raise AuthError(
                "The saved connection method changed; run 'yian auth openai' and start a new run"
            )
        return credentials
    if source == "host":
        raise AuthError(_HOST_KEY_MISSING.format(provider="openai"))
    if source == "missing":
        raise AuthError(
            "No OpenAI connection was selected; run 'yian auth openai' or set OPENAI_API_KEY before starting a new run"
        )
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise AuthError(
            "This run needs its OpenAI API key in OPENAI_API_KEY; use 'yian auth openai' for a new run"
        )
    if source == "environment" and key_identity(key) != connection["credential_id"]:
        raise AuthError(
            "The OpenAI environment API key changed since this run started; start a new run"
        )
    return ApiKeyCredentials(key)
