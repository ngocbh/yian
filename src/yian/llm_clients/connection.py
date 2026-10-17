"""Snapshot model defaults and nonsecret connection identities at host boundaries.

There is no built-in provider or model: defaults come only from the environment
(``YIAN_DEFAULT_PROVIDER``/``YIAN_DEFAULT_MODEL``) and the saved logins. Auth
storage owns credentials and refresh. This module resolves selection, records
on every keyed agent which connection it uses (the saved login or the
provider's key environment variable; OpenAI also a host key, subscription, or
missing key), provides injected credential callables to model transports,
lists the credentials a call sends so its errors can be scrubbed of them, and
records the selection an agent whose config resolves only when it runs binds from.
Recorded metadata is internal, never a YAML option, and never contains a token
or API key. Compilation may use the detached snapshot without authentication
effects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
from dataclasses import dataclass
import hashlib
import os
import re
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr, ValidationError

from .api_key_env import get_api_key_env
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
    HostConnection,
    LLMConfig,
    SUPPORTED_LLM_PROVIDERS,
    azure_resource_endpoint,
    validate_base_url,
)
from .connection_settings import validate_header_name, validate_organization
from .provider_blocks import block_schema
from .redact import scrub_keys

_API_ENDPOINT = "https://api.openai.com/v1"
_SUBSCRIPTION_ENDPOINT = "https://chatgpt.com/backend-api/codex"
_FIELDS = {"method", "source", "credential_id", "endpoint", "store_path"}
# Optional fields of a saved API-key login's connection: the connection settings saved
# with it, header names only. Records made before them carry none.
_SETTINGS_FIELDS = {"organization", "project", "header_names"}
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
# Recorded beside HOST_KEY_MARKER: the sorted names of the headers the host sent with
# its key. Their values, like the key, are never recorded.
HEADER_NAMES_KEY = "_header_names"
_HOST_HEADERS_MISSING = (
    "This run's {provider} connection headers {names} came from the host's "
    "llm_config.headers and are not recorded; pass headers to resume_flow or "
    "RunResult.backward"
)
# Key variables a provider SDK reads after the provider's own, and only when yian
# passes it no key.
_KEY_ENV_FALLBACKS = {"google": ("GEMINI_API_KEY",), "azure": ("OPENAI_API_KEY",)}
# Token variables listed whenever set, though only a call yian passes no key sends
# them: AZURE_OPENAI_AD_TOKEN from the Azure SDK, ANTHROPIC_AUTH_TOKEN from a 0.x
# Anthropic SDK.
_ALWAYS_SENT_ENV = {"azure": ("AZURE_OPENAI_AD_TOKEN",), "anthropic": ("ANTHROPIC_AUTH_TOKEN",)}


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
        host_settings: HostConnection | None = None,
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
        only, and is the OpenAI host connection's endpoint. ``host_settings``, the
        connection settings the host gave with that key, is recorded on the same
        agents: ``organization`` and ``project`` as given, and the header names
        (never their values) sorted under ``HEADER_NAMES_KEY``. Its key is ignored.
        """
        recorded = _host_settings_record(host_settings)
        result = copy.deepcopy(effective)
        for config in result.values():
            self._bind_one(config, host_key_provider, host_base_url, recorded)
        return result

    def _bind_one(
        self,
        config: dict[str, Any],
        host_key_provider: str | None,
        host_base_url: str | None,
        recorded: dict[str, Any],
    ) -> None:
        """Mark one config for the host key, or attach its snapshot connection, in place.

        Args:
            config (`dict[str, Any]`): A mutable config with its ``provider`` set.
            host_key_provider (`str | None`): The host key's provider, if any.
            host_base_url (`str | None`): Where the host key is sent, if given.
            recorded (`dict[str, Any]`): The host settings recorded beside the marker:
                ``organization``, ``project`` and ``HEADER_NAMES_KEY``, each when set.
        """
        if config["provider"] == host_key_provider:
            config[HOST_KEY_MARKER] = True
            if host_base_url is not None:
                config["base_url"] = host_base_url
            config.update(copy.deepcopy(recorded))
            if config["provider"] == "openai":
                config["_connection"] = self.host_connection(host_base_url)
            return
        self._attach(config)

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


def _host_settings_record(host_settings: HostConnection | None) -> dict[str, Any]:
    """Return the nonsecret host settings recorded beside ``HOST_KEY_MARKER``.

    Args:
        host_settings (`HostConnection | None`): The host's connection; its key and
            header values are ignored.

    Returns:
        `dict[str, Any]`: ``organization`` and ``project`` when set, and the sorted
            header names under ``HEADER_NAMES_KEY`` when there are headers.
    """
    recorded: dict[str, Any] = {}
    if host_settings is not None:
        for field in ("organization", "project"):
            if getattr(host_settings, field) is not None:
                recorded[field] = getattr(host_settings, field)
        if host_settings.headers:
            recorded[HEADER_NAMES_KEY] = sorted(host_settings.headers)
    return recorded


_SELECTION_FIELDS = frozenset(
    {"defaults", "provider_models", "connections", "api_endpoint", "stray_model", "host"}
)
_HOST_RECORD_FIELDS = frozenset(
    {"provider", "base_url", "organization", "project", "header_names"}
)


@dataclass(frozen=True)
class ConnectionSelection:
    """A recorded selection that binds the connection of a config resolved later.

    Attributes:
        snapshot (`ConnectionSnapshot`): The snapshot, holding only the recorded
            providers' saved models and connections.
        host_key_provider (`str | None`): The provider whose agents use the host key;
            `None` when the selection names none.
        host_base_url (`str | None`): Where the host key is sent, if the host gave it.
        host_settings (`dict[str, Any]`): The nonsecret settings sent with the host key:
            ``organization``, ``project`` and ``HEADER_NAMES_KEY``, each when set.
    """

    snapshot: ConnectionSnapshot
    host_key_provider: str | None
    host_base_url: str | None
    host_settings: dict[str, Any]

    def bind(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return ``config`` with its connection bound as `ConnectionSnapshot.bind` binds it.

        Args:
            config (`dict[str, Any]`): One agent's resolved config with its ``provider``
                set. Not mutated.

        Returns:
            `dict[str, Any]`: A new config: marked for the host key with its base_url and
                settings when its provider is the host key's, else with the recorded
                connection of its provider.
        """
        result = copy.deepcopy(config)
        self.snapshot._bind_one(
            result, self.host_key_provider, self.host_base_url, self.host_settings
        )
        return result


def record_selection(
    snapshot: ConnectionSnapshot,
    *,
    providers: Iterable[str] | None,
    host_key_provider: str | None = None,
    host_base_url: str | None = None,
    host_settings: HostConnection | None = None,
) -> dict[str, Any]:
    """Return the JSON record of a selection, limited to the given providers.

    Only the given providers' saved models and connections are recorded; the defaults
    only when their provider is one of them; the OpenAI endpoint only when OpenAI is
    one of them (else the built-in one); and the host key's provider, endpoint and
    settings only when it is one of them. The record never holds a key or header value.

    Args:
        snapshot (`ConnectionSnapshot`): The run's selection snapshot.
        providers (`Iterable[str] | None`): The providers whose connections may be
            needed; `None` records every one.
        host_key_provider (`str | None`, *optional*, defaults to `None`): The provider
            of the host key, if the host gave one.
        host_base_url (`str | None`, *optional*, defaults to `None`): Where the host key
            is sent, if the host gave it.
        host_settings (`HostConnection | None`, *optional*, defaults to `None`): The
            host's connection; only its organization, project and header names are
            recorded.

    Returns:
        `dict[str, Any]`: The record `restore_selection` reads.
    """
    allowed = None if providers is None else frozenset(providers)

    def reachable(provider: str | None) -> bool:
        return provider is not None and (allowed is None or provider in allowed)

    host = None
    if reachable(host_key_provider):
        settings = _host_settings_record(host_settings)
        host = {"provider": host_key_provider}
        if host_base_url is not None:
            host["base_url"] = host_base_url
        for name in ("organization", "project"):
            if name in settings:
                host[name] = settings[name]
        if HEADER_NAMES_KEY in settings:
            host["header_names"] = settings[HEADER_NAMES_KEY]
    return {
        "defaults": (
            copy.deepcopy(snapshot.defaults)
            if reachable(snapshot.defaults.get("provider")) else {}
        ),
        "provider_models": {
            name: model for name, model in snapshot.provider_models.items() if reachable(name)
        },
        "connections": {
            name: copy.deepcopy(connection)
            for name, connection in snapshot.connections.items()
            if reachable(name)
        },
        # The endpoint configures only OpenAI; another run's records keep the built-in one.
        "api_endpoint": snapshot.api_endpoint if reachable("openai") else _API_ENDPOINT,
        "stray_model": snapshot.stray_model,
        "host": host,
    }


def restore_selection(record: Any) -> ConnectionSelection:
    """Validate a `record_selection` record and return the selection it holds.

    Args:
        record (`Any`): The recorded selection.

    Returns:
        `ConnectionSelection`: The selection, sharing no value with ``record``.

    Raises:
        `ValueError`: The record does not have the recorded shape, names an
            unsupported provider, or holds an invalid connection, endpoint or host
            setting. Errors never echo values.
    """
    try:
        return _restore_selection(record)
    except ValueError as exc:
        raise ValueError(f"Invalid recorded connection selection: {exc}") from None


def _restore_selection(record: Any) -> ConnectionSelection:
    """Validate and restore a selection record; `restore_selection` prefixes errors."""
    if type(record) is not dict or set(record) != _SELECTION_FIELDS:
        raise ValueError("fields")
    defaults = record["defaults"]
    if (
        type(defaults) is not dict
        or not set(defaults) <= {"provider", "model"}
        or ("model" in defaults and "provider" not in defaults)
    ):
        raise ValueError("defaults")
    for value in defaults.values():
        _nonempty(value, "defaults")
    if "provider" in defaults:
        _supported(defaults["provider"], "defaults")
    models = record["provider_models"]
    if type(models) is not dict:
        raise ValueError("provider_models")
    for name, model in models.items():
        _supported(name, "provider_models")
        _nonempty(model, "provider_models")
    connections = record["connections"]
    if type(connections) is not dict:
        raise ValueError("connections")
    for name, connection in connections.items():
        _supported(name, "connections")
        validate_connection(connection, name)
    if type(record["stray_model"]) is not bool:
        raise ValueError("stray_model")
    _nonempty(record["api_endpoint"], "api_endpoint")
    snapshot = ConnectionSnapshot(
        copy.deepcopy(defaults),
        dict(models),
        copy.deepcopy(connections),
        record["api_endpoint"],
        record["stray_model"],
    )
    # The OpenAI host connection is the one use of the endpoint; check it the same way.
    snapshot.host_connection()
    host = record["host"]
    if host is None:
        return ConnectionSelection(snapshot, None, None, {})
    if type(host) is not dict or "provider" not in host or not set(host) <= _HOST_RECORD_FIELDS:
        raise ValueError("host fields")
    provider = _supported(host["provider"], "host provider")
    base_url = host.get("base_url")
    if base_url is not None:
        validate_base_url(base_url)
    elif provider == "azure":
        # An Azure key names no resource of its own; the host always gives its endpoint.
        raise ValueError("an azure host key needs its base_url")
    settings: dict[str, Any] = {}
    for name in ("organization", "project"):
        if name in host:
            if provider != "openai":
                raise ValueError(f"host {name} applies only to provider openai")
            settings[name] = validate_organization(host[name], name)
    if "header_names" in host:
        settings[HEADER_NAMES_KEY] = validate_header_names(host["header_names"], provider)
    return ConnectionSelection(snapshot, provider, base_url, settings)


def _nonempty(value: Any, where: str) -> str:
    """Return ``value`` if it is a non-empty string; otherwise name ``where``."""
    if type(value) is not str or not value:
        raise ValueError(where)
    return value


def _supported(value: Any, where: str) -> str:
    """Return ``value`` if it names a supported provider; otherwise name ``where``."""
    if value not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(f"{where} names an unsupported provider")
    return value


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
        if selection.method == "api_key":
            connections[selection.provider].update(_selection_settings(selection))
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


def _selection_settings(selection: ProviderSelection) -> dict[str, Any]:
    """Return the connection fields that record a saved login's settings.

    Args:
        selection (`ProviderSelection`): A saved API-key login.

    Returns:
        `dict[str, Any]`: ``organization`` and ``project`` when set, and the sorted
            ``header_names`` when the login has headers; never a header value.
    """
    settings: dict[str, Any] = {}
    for field in ("organization", "project"):
        if getattr(selection, field) is not None:
            settings[field] = getattr(selection, field)
    if selection.header_names:
        settings["header_names"] = sorted(selection.header_names)
    return settings


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
    the provider's key environment variable (``environment``, unpinned). A saved
    API-key login may also record the settings saved with it: ``organization``
    and ``project`` (``openai`` only) and the sorted ``header_names``.

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
    if type(value) is not dict or not _FIELDS <= set(value) <= _FIELDS | _SETTINGS_FIELDS:
        raise ValueError("Invalid internal connection metadata fields")
    _check_connection_settings(value, provider)
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


def _check_connection_settings(value: dict[str, Any], provider: str) -> None:
    """Check the optional settings fields of a connection.

    Args:
        value (`dict[str, Any]`): Metadata with the connection fields.
        provider (`str`): A keyed provider.

    Raises:
        `ValueError`: A settings field is on a connection other than a saved
            API-key login, ``organization`` or ``project`` is on a provider other
            than ``openai``, or a value is malformed. Errors name only the field.
    """
    if not set(value) & _SETTINGS_FIELDS:
        return
    if value["source"] != "store" or value["method"] != "api_key":
        raise ValueError("Internal connection settings are only for a saved API-key login")
    for field in ("organization", "project"):
        if field not in value:
            continue
        if provider != "openai":
            raise ValueError(f"Internal connection {field} is only for provider openai")
        try:
            validate_organization(value[field], field)
        except ValueError:
            raise ValueError(f"Invalid internal connection {field}") from None
    if "header_names" in value:
        try:
            validate_header_names(value["header_names"], provider)
        except ValueError:
            raise ValueError("Invalid internal connection header_names") from None


def _validate_keyed_connection(value: dict[str, Any], provider: str) -> dict[str, Any]:
    """Validate a non-OpenAI keyed provider's connection fields.

    Args:
        value (`dict[str, Any]`): Metadata with the required connection fields and any optional ones.
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


def validate_header_names(value: Any, provider: Any) -> list[str]:
    """Validate a recorded ``HEADER_NAMES_KEY`` value: the sorted names of host headers.

    Args:
        value (`Any`): The candidate value.
        provider (`Any`): The provider of the config that carries it.

    Returns:
        `list[str]`: A copy of ``value``.

    Raises:
        `ValueError`: ``value`` is not a non-empty sorted list of distinct (ignoring
            case) header names that ``provider`` may be sent; errors name only the key.
    """
    invalid = ValueError(f"Invalid internal {HEADER_NAMES_KEY}")
    if type(value) is not list or not value or value != sorted(map(str, value)):
        raise invalid
    if len({name.lower() for name in value}) != len(value):
        raise invalid
    try:
        for name in value:
            validate_header_name(name, provider if isinstance(provider, str) else None)
    except ValueError:
        raise invalid from None
    return list(value)


def _check_host_headers(raw: dict[str, Any], names: list[str]) -> None:
    """Require a marked config's headers to have exactly its recorded header names.

    Args:
        raw (`dict[str, Any]`): The config, ``HOST_KEY_MARKER`` already removed.
        names (`list[str]`): The recorded header names; empty when none were recorded.

    Raises:
        `AuthError`: The header names differ, ignoring case; the message names the
            missing and unexpected header names, never a value.
    """
    headers = raw.get("headers") or {}
    if not isinstance(headers, Mapping):
        # `LLMConfig` names the malformed field.
        return
    diff = header_name_diff(names, [name for name in headers if isinstance(name, str)])
    if diff is None:
        return
    if not headers:
        expected = {name.lower(): name for name in names}
        raise AuthError(_HOST_HEADERS_MISSING.format(
            provider=raw.get("provider"),
            names=", ".join(expected[name] for name in sorted(expected)),
        ))
    raise AuthError(
        "llm_config.headers must have the header names this run recorded for the host "
        f"key: {diff}"
    )


def header_name_diff(expected: Iterable[str], given: Iterable[str]) -> str | None:
    """Describe how the given header names differ from the expected ones, ignoring case.

    Args:
        expected (`Iterable[str]`): The header names that should be given.
        given (`Iterable[str]`): The header names given.

    Returns:
        `str | None`: ``"missing A, B; not recorded C"`` (either part only when the
            other is empty), names sorted ignoring case; `None` when the names match.
    """
    expected_names = {name.lower(): name for name in expected}
    given_names = {name.lower(): name for name in given}
    missing = [expected_names[name] for name in sorted(expected_names) if name not in given_names]
    extra = [given_names[name] for name in sorted(given_names) if name not in expected_names]
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if extra:
        details.append(f"not recorded {', '.join(extra)}")
    return "; ".join(details) or None


def split_model_config(
    config: dict[str, Any] | LLMConfig,
) -> tuple[LLMConfig, dict[str, Any] | None]:
    """Validate public fields separately from host-owned connection metadata.

    A config marked with ``HOST_KEY_MARKER`` needs its ``api_key``, and ``headers``
    with exactly the recorded ``HEADER_NAMES_KEY`` names (none when absent),
    compared without case; both are popped from the validated fields.

    Raises:
        `AuthError`: The config carries ``HOST_KEY_MARKER`` but no ``api_key``, or
            its headers do not have the recorded names.
        `ValueError`: A public field or the metadata is invalid, the header names
            are malformed or appear without the marker, an OpenAI ``base_url``
            differs from the endpoint of its pinned host connection, or an Azure
            ``base_url`` names another resource than its saved login.
    """
    if isinstance(config, LLMConfig):
        raw = config.model_dump(exclude_none=True)
        # Dumps leave the key and header values out; carry them across explicitly.
        for field in ("api_key", "headers"):
            if getattr(config, field) is not None:
                raw[field] = getattr(config, field)
    else:
        raw = dict(config)
    header_names = raw.pop(HEADER_NAMES_KEY, None)
    if header_names is not None:
        header_names = validate_header_names(header_names, raw.get("provider"))
    if HOST_KEY_MARKER in raw:
        if raw.pop(HOST_KEY_MARKER) is not True:
            raise ValueError("Invalid internal host key marker")
        if raw.get("api_key") is None:
            raise AuthError(_HOST_KEY_MISSING.format(provider=raw.get("provider")))
        _check_host_headers(raw, header_names or [])
    elif header_names is not None:
        raise ValueError(f"Internal {HEADER_NAMES_KEY} needs the host key marker")
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


def refused_block(config: Mapping[str, Any]) -> str:
    """Return ``.<provider>`` when `check_model_config` refused only the provider's block.

    Its fields and connection are checked first and an Azure config needs a
    ``base_url``; a config that passes those was refused for its block alone.

    Args:
        config (`Mapping[str, Any]`): The config `check_model_config` refused.

    Returns:
        `str`: ``.<provider>``, or ``""`` when something else was refused.
    """
    try:
        validated, _ = split_model_config(dict(config))
    except (ValueError, AuthError):
        return ""
    if validated.provider is None or (
        validated.provider == "azure" and validated.base_url is None
    ):
        return ""
    return f".{validated.provider}"


_LEADING_KEY_PATH = re.compile(r"`[^`.]+\.([^`]+)`: (.*)", re.DOTALL)


def block_problem(where: str, block: str, message: str, callsite: str = "") -> str:
    """Name a provider block's problem under ``where``, giving its key path once.

    A single problem that opens with its backticked key path and a colon, such as
    ``openai.temperature``, reads ``<where>.openai.temperature<callsite>: ...``. Any
    other reads ``<where>.<block><callsite>: <message>``, or
    ``<where><callsite>: <message>`` when ``block`` is empty.

    Args:
        where (`str`): Names the layer, such as ``node 'chat' llm_config``.
        block (`str`): The block refused, such as ``openai``; empty when unknown.
        message (`str`): The problem, as a block check words it.
        callsite (`str`, *optional*, defaults to `""`): Follows the field, such as
            `` in 'sub'``.

    Returns:
        `str`: The message.
    """
    match = _LEADING_KEY_PATH.fullmatch(message)
    # Several problems are joined with "; "; moving the first one's path would misname
    # the rest.
    if block and match is not None and "; `" not in match[2]:
        return f"{where}.{block}.{match[1]}{callsite}: {match[2]}"
    return f"{where}{'.' + block if block else ''}{callsite}: {message}"


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
    # Dumps leave the key and header values out; carry them across explicitly.
    for field in ("api_key", "headers"):
        if getattr(validated, field) is not None:
            raw[field] = getattr(validated, field)
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
    connection: dict[str, Any], provider: str = "openai"
) -> ApiKeyCredentials | OAuthCredentials:
    """Resolve fresh credentials while enforcing the recorded connection identity.

    A ``store`` connection resolves the saved login it was pinned to, so a later
    login or removal fails instead of switching accounts; an API key comes with
    the header values saved beside it. OpenAI also resolves its environment key,
    checking that it is the recorded one. Another provider's ``environment``
    connection resolves nothing here: its client reads the key variable itself.

    Args:
        connection (`dict[str, Any]`): The recorded connection metadata.
        provider (`str`, *optional*, defaults to ``"openai"``): The provider of
            the config that carries it.

    Returns:
        `ApiKeyCredentials | OAuthCredentials`: The credentials to send.

    Raises:
        `AuthError`: The pinned login changed, was removed, or cannot be read; or
            an OpenAI key is missing or changed.
        `ValueError`: The metadata is invalid for ``provider``, or it is a
            non-OpenAI ``environment`` connection.
    """
    connection = validate_connection(connection, provider)
    source = connection["source"]
    if source == "store":
        try:
            credentials = AuthStore(connection["store_path"]).credentials(
                provider,
                connection["credential_id"],
                refresh=OpenAIAuth().refresh if provider == "openai" else None,
            )
        except AuthError as exc:
            raise AuthError(f"{exc}. Run 'yian auth {provider}' and start a new run") from exc
        if credentials.method != connection["method"]:
            raise AuthError(
                f"The saved connection method changed; run 'yian auth {provider}' and start a new run"
            )
        return credentials
    if provider != "openai":
        raise ValueError(f"An environment connection for {provider} is read by its client")
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


def call_credentials(config: dict[str, Any] | LLMConfig | None) -> tuple[str, ...]:
    """Return every credential the model call built from ``config`` may send.

    This follows the precedence of `model_from_config` and the provider SDKs. A
    config ``api_key`` is sent in place of any saved login or key variable, and
    the config's ``headers`` values (`SecretStr` or plain strings) are listed
    whenever given. Otherwise a ``store`` connection sends its provider's saved
    login, read without refresh: its key and saved header values, or its
    subscription tokens, expired ones too. Otherwise (an
    ``environment`` or ``legacy_environment`` connection, or none) the client
    reads the first set variable of the provider's key variable, then the SDK
    fallbacks. ``AZURE_OPENAI_AD_TOKEN`` (``azure``) and ``ANTHROPIC_AUTH_TOKEN``
    (``anthropic``) are listed whenever set, though the Azure token is sent only by
    a call given no key and the Anthropic token only by a 0.x SDK given no key. SDK
    header variables such as ``OPENAI_CUSTOM_HEADERS`` are never sent, since the
    clients drop them, so their values are not listed. ``OLLAMA_API_KEY`` is never
    sent, since the ollama client drops it. A config without connection
    metadata is resolved first, as `model_from_config` does; `None` resolves like
    ``{}``. Each step is guarded, so a failing one still leaves the others'
    credentials. Every header value is listed however short, so a value such as
    ``prod`` is scrubbed wherever it appears.

    Args:
        config (`dict[str, Any] | LLMConfig | None`): The config the model call
            was built from.

    Returns:
        `tuple[str, ...]`: The distinct plain credentials, possibly empty. Never raises.
    """
    found: list[str] = []
    provider = host_key = connection = headers = None
    try:
        if config is None:
            config = {}
        if isinstance(config, LLMConfig):
            provider, host_key, headers = config.provider, config.api_key, config.headers
        elif isinstance(config, dict):
            provider, host_key = config.get("provider"), config.get("api_key")
            connection, headers = config.get("_connection"), config.get("headers")
        if isinstance(host_key, SecretStr):
            host_key = host_key.get_secret_value()
        if isinstance(host_key, str) and host_key:
            found.append(host_key)
        else:
            host_key = None
    except Exception:
        pass
    try:
        if isinstance(headers, Mapping):
            # A Python caller's dict may hold plain strings where LLMConfig holds SecretStr.
            for value in headers.values():
                if isinstance(value, SecretStr):
                    value = value.get_secret_value()
                if isinstance(value, str) and value:
                    found.append(value)
    except Exception:
        pass
    if connection is None and (host_key is None or provider is None):
        try:
            resolved = resolve_model_config(config)
            provider, connection = resolved.get("provider"), resolved.get("_connection")
        except Exception:
            pass
    if not isinstance(provider, str):
        return tuple(dict.fromkeys(found))
    provider = provider.lower()
    if host_key is None:
        try:
            source = None if connection is None else connection["source"]
            if source == "store":
                found.extend(AuthStore(connection["store_path"]).saved_secrets(provider))
            elif source in (None, "environment", "legacy_environment"):
                names = (get_api_key_env(provider), *_KEY_ENV_FALLBACKS.get(provider, ()))
                values = (os.environ.get(name) for name in names if name)
                key = next((value for value in values if value), None)
                if key:
                    found.append(key)
        except Exception:
            pass
    try:
        for name in _ALWAYS_SENT_ENV.get(provider, ()):
            if os.environ.get(name):
                found.append(os.environ[name])
    except Exception:
        pass
    return tuple(dict.fromkeys(found))


def scrub_call_credentials(
    exc: BaseException, config: dict[str, Any] | LLMConfig | None
) -> None:
    """Redact every credential `call_credentials` finds for ``config`` from ``exc`` in place.

    Args:
        exc (`BaseException`): The exception the model call raised.
        config (`dict[str, Any] | LLMConfig | None`): The config the model call
            was built from.
    """
    try:
        scrub_keys(exc, call_credentials(config))
    except Exception:
        # Scrubbing is best effort; the provider error must survive it.
        pass
