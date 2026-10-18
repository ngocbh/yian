"""Provider setup over staged authentication and an atomic store, `llm_config` checks, and
the provider list.

`config_check` checks a host `llm_config` layer read from stdin, and `providers` lists
the providers and the login kinds `auth` can save for each.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import sys
from typing import Optional
import warnings
import webbrowser

from pydantic import ValidationError
import typer

from yian.cli.utils import print_json
from yian.llm_clients.auth import (
    ApiKeyCredentials,
    AuthError,
    AuthStore,
    OpenAIAuth,
)
from yian.llm_clients.api_key_env import (
    OPTIONAL_API_KEY_PROVIDERS,
    PROVIDER_API_KEY_ENV,
)
from yian.llm_clients.config import (
    LLM_CONFIG_KEYS,
    SUPPORTED_LLM_PROVIDERS,
    LLMConfig,
    pop_host_credentials,
    validate_base_url,
)
from yian.llm_clients.connection import call_credentials
from yian.llm_clients.connection_settings import (
    validate_header_name,
    validate_header_value,
    validate_headers,
    validate_organization,
)
from yian.llm_clients.factory import create_llm_client
from yian.llm_clients.model_list import ModelListError, list_models
from yian.llm_clients.redact import redact_keys, scrub_keys

_KINDS = {"api_key": "api-key", "subscription": "subscription", "none": "none"}
_CANCELLED = "Authentication cancelled; the previous login is unchanged"
_NO_DEFAULT = (
    "No default provider is saved; choose one with 'yian auth <provider> --default'"
)
# Environment overrides of a local server's URL, read the way its client reads them.
_BASE_URL_ENV = {"ollama": "OLLAMA_BASE_URL", "vllm": "VLLM_BASE_URL"}
# Variables the provider's SDK reads to send runs to another host than the one listed;
# a gateway's key must not go to the public host's listing.
_ELSEWHERE_ENV = {
    "anthropic": ("ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"),
    "google": ("GOOGLE_GEMINI_BASE_URL",),
}
_SETTINGS_NEED_API_KEY = "--organization, --project and --header apply to API-key authentication"
_EXTRA_FIELD = (
    f"llm_config has an extra field; the fields are {', '.join(sorted(LLM_CONFIG_KEYS))}"
)


def _require_terminal(option: str) -> None:
    """Refuse a prompt when no interactive terminal is available.

    Raises:
        AuthError: Standard input is not a terminal; the message names `option`.
    """
    if not sys.stdin.isatty():
        raise AuthError(
            f"Authentication needs {option} when no interactive terminal is available"
        )


def _choose(message: str, choices: list[str], *, default: str, option: str) -> str:
    """Select a value on a terminal, or require an explicit command option."""
    _require_terminal(option)
    import questionary

    answer = questionary.select(message, choices=choices, default=default).ask()
    if answer is None:
        raise AuthError(_CANCELLED)
    return answer


def _ask(message: str, *, option: str) -> str:
    """Read a typed value on a terminal, or require an explicit command option."""
    _require_terminal(option)
    import questionary

    answer = questionary.text(message).ask()
    if answer is None or not answer.strip():
        raise AuthError(_CANCELLED)
    return answer.strip()


def _api_key(provider: str, variable: str, *, optional: bool = False) -> Optional[str]:
    """Read the provider's key from `variable` or a hidden prompt.

    Args:
        provider (`str`): Provider name shown in the prompt.
        variable (`str`): Environment variable checked first.
        optional (`bool`): The provider can run without a key; off a terminal or
            with an empty answer, no key is returned.

    Returns:
        `str | None`: The key, or `None` when an optional key is not given.

    Raises:
        AuthError: A required key is missing, or the prompt is cancelled.
    """
    key = os.environ.get(variable)
    if key:
        return key
    if not sys.stdin.isatty():
        if optional:
            return None
        raise AuthError(
            f"Set {variable} before API-key setup, or run in an interactive terminal"
        )
    import questionary

    hint = " (leave empty for none)" if optional else ""
    key = questionary.password(f"{provider} API key{hint}:").ask()
    if key is None:
        raise AuthError(_CANCELLED)
    if not key.strip():
        if optional:
            return None
        raise AuthError("Authentication cancelled; no API key was supplied")
    return key.strip()


def _header_sources(specs: list[str], provider: str) -> dict[str, Optional[str]]:
    """Check each ``--header`` and read the values given through environment variables.

    ``NAME=VARIABLE`` reads the value from that variable; ``NAME`` alone leaves it to a
    hidden prompt, which needs a terminal. Messages never contain a value or the
    variable's name, since a mistyped option may put a value where the variable goes.

    Args:
        specs (`list[str]`): The ``--header`` options as given.
        provider (`str`): The provider the headers go to.

    Returns:
        `dict[str, str | None]`: Each header name and its value, or `None` to prompt.

    Raises:
        AuthError: A name is not an HTTP token, is refused or repeated (names ignore
            case), a variable is empty or unset, a value is not allowed, or a value
            needs a prompt and no terminal is available.
    """
    sources: dict[str, Optional[str]] = {}
    seen: set[str] = set()
    for spec in specs:
        name, has_variable, variable = spec.partition("=")
        try:
            validate_header_name(name, provider)
        except ValueError as error:
            raise AuthError(f"Invalid --header: {error}") from None
        if name.lower() in seen:
            raise AuthError(f"--header {name} is given more than once (names ignore case)")
        seen.add(name.lower())
        if not has_variable:
            if not sys.stdin.isatty():
                raise AuthError(
                    f"--header {name} reads its value from a hidden prompt, which needs a "
                    "terminal; use --header NAME=VARIABLE to read it from a variable"
                )
            sources[name] = None
            continue
        if not variable:
            raise AuthError(f"--header {name}=VARIABLE needs an environment variable name")
        value = os.environ.get(variable)
        if not value:
            raise AuthError(f"The environment variable given for --header {name} is not set")
        try:
            sources[name] = validate_header_value(name, value)
        except ValueError as error:
            raise AuthError(f"Invalid --header: {error}") from None
    return sources


def _headers(sources: Mapping[str, Optional[str]], provider: str) -> dict[str, str]:
    """Prompt, hidden, for each header value not read yet, and check all of them.

    Args:
        sources (`Mapping[str, str | None]`): From `_header_sources`.
        provider (`str`): The provider the headers go to.

    Returns:
        `dict[str, str]`: Each header name and its value.

    Raises:
        AuthError: A prompt is cancelled or left empty, or a value or the number of
            headers is not allowed.
    """
    headers = {}
    for name, value in sources.items():
        if value is None:
            import questionary

            value = questionary.password(f"{name} header value:").ask()
            if value is None:
                raise AuthError(_CANCELLED)
            if not value:
                raise AuthError(f"Authentication cancelled; no value was given for {name}")
        headers[name] = value
    try:
        return validate_headers(headers, provider)
    except ValueError as error:
        raise AuthError(f"Invalid --header: {error}") from None


def _open_browser(url: str) -> None:
    """Present the login URL and open it with the host's browser."""
    typer.echo(f"Open this URL to sign in:\n{url}")
    try:
        webbrowser.open(url)
    except webbrowser.Error:
        pass  # The displayed URL remains usable when browser discovery fails.


def _device_code(url: str, code: str) -> None:
    """Present the verification page and one-time user code."""
    typer.echo(f"Open {url} and enter code: {code}")


def _text_models(models: tuple[str, ...]) -> list[str]:
    """Offer general text-model families from the authenticated API catalog."""
    return [
        name
        for name in models
        if name.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4"))
        and not any(
            part in name for part in ("realtime", "audio", "image", "transcribe", "tts")
        )
    ]


def _list() -> None:
    """Print each saved login and the default without reading any credential."""
    state = AuthStore().read()
    overrides = [
        name
        for name in ("YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL")
        if os.environ.get(name)
    ]
    if not state.selections:
        typer.echo("No login is saved; run 'yian auth <provider>' to add one")
    else:
        typer.echo("Saved logins:")
        for selection in state.selections:
            details = [_KINDS[selection.method], f"model {selection.model}"]
            if selection.endpoint:
                details.append(f"endpoint {selection.endpoint}")
            if selection.organization:
                details.append(f"organization {selection.organization}")
            if selection.project:
                details.append(f"project {selection.project}")
            if selection.header_names:
                # Header values are secrets; only their names are shown.
                details.append(f"headers {', '.join(selection.header_names)}")
            default = " (default)" if selection.provider == state.default_provider else ""
            typer.echo(f"  {selection.provider}: {', '.join(details)}{default}")
        if state.default_provider is None:
            typer.echo(_NO_DEFAULT)
    if overrides and state.selections:
        verbs = ("is", "overrides") if len(overrides) == 1 else ("are", "override")
        typer.echo(
            f"{' and '.join(overrides)} {verbs[0]} set in the environment and "
            f"{verbs[1]} the saved choice"
        )


def _check_endpoint(endpoint: str) -> None:
    """Refuse an Azure endpoint that a config's `base_url` could not name.

    Raises:
        AuthError: The endpoint is invalid.
    """
    try:
        validate_base_url(endpoint)
    except ValueError as error:
        raise AuthError(f"Invalid Azure endpoint: {error}") from None


def _check_options(
    provider: str,
    *,
    method: Optional[str],
    login: Optional[str],
    model: Optional[str],
    api_key_env: Optional[str],
    endpoint: Optional[str],
    default: bool,
    remove: bool,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    header: Optional[list[str]] = None,
) -> tuple[str, dict[str, Optional[str]]]:
    """Validate the option combination and return the provider name and headers to use.

    Azure setup also needs `OPENAI_API_VERSION` and a valid `endpoint`, when given.
    `organization` and `project` apply to an OpenAI API key, and `header` to every
    provider sent a key (see `_header_sources`).

    Returns:
        `tuple[str, dict[str, str | None]]`: The provider name, and each header name
        with its value from the environment or `None` to prompt.

    Raises:
        AuthError: The provider or an option does not apply, a connection setting is
            invalid, or Azure setup lacks its API version or has an invalid endpoint.
    """
    setup = {
        "--method": method,
        "--login": login,
        "--model": model,
        "--api-key-env": api_key_env,
        "--endpoint": endpoint,
        "--organization": organization,
        "--project": project,
        "--header": header or None,
    }
    given = [option for option, value in setup.items() if value is not None]
    if default and remove:
        raise AuthError("Use either --default or --remove, not both")
    if (default or remove) and given:
        action = "--default" if default else "--remove"
        raise AuthError(f"{action} does not take {', '.join(given)}")
    if remove:
        # A login saved by another yian version may name any provider.
        return provider, {}
    name = provider.lower()
    if name not in SUPPORTED_LLM_PROVIDERS:
        raise AuthError(
            f"Unsupported provider {provider!r}; choose one of: "
            f"{', '.join(SUPPORTED_LLM_PROVIDERS)}"
        )
    if name != "openai" and (method is not None or login is not None):
        raise AuthError("--method and --login apply only to openai")
    if name != "azure" and endpoint is not None:
        raise AuthError("--endpoint applies only to azure")
    if api_key_env is not None and not api_key_env.strip():
        raise AuthError("--api-key-env needs an environment variable name")
    if name == "ollama" and api_key_env is not None:
        raise AuthError("ollama takes no API key; omit --api-key-env")
    if method is not None and method not in ("subscription", "api-key"):
        raise AuthError("Choose --method subscription or --method api-key")
    if login is not None and login not in ("browser", "device"):
        raise AuthError("Choose --login browser or --login device")
    if method == "api-key" and login is not None:
        raise AuthError("--login applies to subscription authentication")
    if organization is not None or project is not None:
        if name != "openai":
            raise AuthError("--organization and --project apply only to openai")
        if method == "subscription":
            raise AuthError("--organization and --project apply to API-key authentication")
        for field, value in (("organization", organization), ("project", project)):
            if value is not None:
                try:
                    validate_organization(value, field)
                except ValueError as error:
                    raise AuthError(f"Invalid --{field}: {error}") from None
    if header:
        if name == "ollama":
            raise AuthError("ollama takes no API key, so it takes no --header")
        if method == "subscription":
            raise AuthError("--header applies to API-key authentication")
    if name == "azure" and not default:
        # The Azure client reads the API version only from the environment.
        if not os.environ.get("OPENAI_API_VERSION"):
            raise AuthError(
                "Azure needs OPENAI_API_VERSION set to its API version (runs read it too)"
            )
        if endpoint is not None:
            _check_endpoint(endpoint)
    return name, _header_sources(header or [], name)


def _setup_openai(
    store: AuthStore,
    method: Optional[str],
    login: Optional[str],
    model: Optional[str],
    api_key_env: str,
    make_default: bool,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    header_sources: Optional[Mapping[str, Optional[str]]] = None,
) -> None:
    """Sign in to OpenAI, choose an available model, and save the login.

    An API key is saved with the given organization, project and headers, which its
    model listing sends too; a subscription takes none of them.

    Raises:
        AuthError: Sign-in, a setting, the listing, or the model choice fails.
    """
    method = method or _choose(
        "OpenAI authentication method:",
        ["subscription", "api-key"],
        default="subscription",
        option="--method",
    )
    if method == "subscription" and (organization or project or header_sources):
        raise AuthError(_SETTINGS_NEED_API_KEY)
    service = OpenAIAuth()
    if method == "subscription":
        login = login or _choose(
            "Subscription sign-in:",
            ["browser", "device"],
            default="device" if os.environ.get("SSH_CONNECTION") else "browser",
            option="--login",
        )
        credentials = (
            service.device_login(_device_code)
            if login == "device"
            else service.browser_login(_open_browser)
        )
        models = service.subscription_models(credentials)
        choices = list(models)
    else:
        if login is not None:
            raise AuthError("--login applies to subscription authentication")
        key = _api_key("openai", api_key_env)
        headers = _headers(header_sources or {}, "openai")
        credentials = ApiKeyCredentials(key, headers)
        try:
            models = service.api_models(
                credentials, organization=organization, project=project
            )
        except AuthError as error:
            # The messages are fixed today; scrubbed so no sent secret can reach output.
            raise AuthError(redact_keys(str(error), (key, *headers.values()))) from None
        choices = _text_models(models)
    if model is None:
        if not choices:
            raise AuthError(
                "No general text models were returned; use --model with an available text model"
            )
        model = _choose("Model:", choices, default=choices[0], option="--model")
    if model not in models:
        raise AuthError("The selected model is not available for this connection")
    store.save(
        "openai",
        credentials,
        model,
        organization=organization,
        project=project,
        make_default=make_default,
    )
    typer.echo(f"Saved OpenAI {method} authentication with model {model}")


def _model_answers(
    provider: str,
    model: str,
    key: Optional[str],
    endpoint: Optional[str],
    headers: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Send the model one short prompt and say why it failed, if it did.

    The client is built from the given key, endpoint and headers only. Without a
    key, vLLM gets the dummy key its client sends to a keyless server, and Ollama
    sends none.

    Args:
        provider (`str`): Provider name.
        model (`str`): Model id, or the Azure deployment name.
        key (`str | None`): API key to send, or `None` for none.
        endpoint (`str | None`): Azure resource endpoint, or `None`.
        headers (`Mapping[str, str] | None`): Connection headers sent beside the key.

    Returns:
        `str | None`: `None` when the model replied; otherwise the error type and
        a short message without any credential the call sent, or the install
        hint for a missing extra.
    """
    # The Ollama client ignores timeout and max_retries.
    options: dict = {"max_retries": 0, "timeout": 30}
    if key is not None:
        options["api_key"] = key
    elif provider in OPTIONAL_API_KEY_PROVIDERS:
        # Passed explicitly so an ambient VLLM_API_KEY doesn't stand in for the
        # keyless login being saved.
        options["api_key"] = "EMPTY"
    if headers:
        options["connection_headers"] = dict(headers)
    try:
        with warnings.catch_warnings():
            # The reply decides whether the model is usable, not yian's catalog.
            warnings.filterwarnings(
                "ignore", message="Model .* is not in the known model list"
            )
            llm = create_llm_client(provider, model, base_url=endpoint, **options)
            llm.get_llm().invoke("Reply with OK.")
    except ImportError as error:
        return str(error)
    except Exception as error:
        # call_credentials also lists a set token variable (AZURE_OPENAI_AD_TOKEN,
        # ANTHROPIC_AUTH_TOKEN), though a call given a key doesn't send it. A keyless
        # check sends no saved or ambient key, so none is looked up.
        sent = {"provider": provider, "api_key": key, "headers": dict(headers or {})}
        secrets = tuple((headers or {}).values()) if key is None else call_credentials(sent)
        scrub_keys(error, secrets)
        # Some errors, such as pydantic's ValidationError, build their text from
        # fields scrub_keys does not rewrite, so the text is redacted again.
        text = " ".join(redact_keys(str(error), secrets).split())
        return f"{type(error).__name__}: {text[:200]}"
    return None


def _setup_provider(
    store: AuthStore,
    provider: str,
    model: Optional[str],
    api_key_env: Optional[str],
    endpoint: Optional[str],
    make_default: bool,
    header_sources: Optional[Mapping[str, Optional[str]]] = None,
) -> None:
    """Read a provider's key, endpoint, and model, check the model, and save the login.

    A model in the provider's listing is accepted as is. Any other model, every
    Azure deployment, and every model when a variable in `_ELSEWHERE_ENV` sends runs
    to another host, must answer one short prompt. vLLM's key is optional unless
    `api_key_env` names its variable or headers are given, since headers are sent
    only beside a key. The listing and the reply check send the headers, which are
    saved with the key.

    Raises:
        AuthError: A value is missing or invalid, or the model did not answer.
    """
    if provider == "azure" and endpoint is None:
        endpoint = _ask(
            "Azure OpenAI endpoint (https://<resource>.openai.azure.com/):",
            option="--endpoint",
        )
        _check_endpoint(endpoint)
    key = None
    if provider != "ollama":
        key = _api_key(
            provider,
            api_key_env or PROVIDER_API_KEY_ENV[provider],
            optional=provider in OPTIONAL_API_KEY_PROVIDERS
            and api_key_env is None
            and not header_sources,
        )
    headers = _headers(header_sources or {}, provider)
    models = None
    if provider == "azure":
        model = model or _ask("Deployment name:", option="--model")
    else:
        if model is None:
            _require_terminal("--model")  # Before the listing request.
        elsewhere = next(
            (name for name in _ELSEWHERE_ENV.get(provider, ()) if os.environ.get(name)),
            None,
        )
        variable = _BASE_URL_ENV.get(provider)
        base_url = (os.environ.get(variable) or None) if variable else None
        failure = None
        if elsewhere is None:
            try:
                models = list_models(provider, key, base_url, headers=headers)
            except ModelListError as error:
                # The reasons are fixed today; scrubbed so no sent secret can reach output.
                failure = redact_keys(str(error), (key, *headers.values()))
        if model is None:
            if elsewhere is not None:
                typer.echo(
                    f"{elsewhere} is set, so runs use another host than the {provider} "
                    "listing; type a model name"
                )
                model = _ask("Model:", option="--model")
            elif models:
                model = _choose(
                    "Model:", list(models), default=models[0], option="--model"
                )
            else:
                typer.echo(
                    f"Couldn't list the {provider} models "
                    f"({failure or 'none were returned'}); type a model name"
                )
                model = _ask("Model:", option="--model")
    if models is None or model not in models:
        problem = _model_answers(provider, model, key, endpoint, headers)
        if problem:
            raise AuthError(
                f"The model didn't answer: {problem}; the previous login is unchanged"
            )
    store.save(
        provider,
        ApiKeyCredentials(key, headers) if key else None,
        model,
        endpoint=endpoint,
        make_default=make_default,
    )
    kind = "api-key" if key else "none"
    typer.echo(f"Saved {provider} login ({kind}) with model {model}")


def auth(
    provider: Optional[str] = typer.Argument(
        None, help="Provider to set up, make the default, or remove; omit to list logins."
    ),
    method: Optional[str] = typer.Option(
        None, help="OpenAI only: subscription or api-key; prompts when omitted."
    ),
    login: Optional[str] = typer.Option(
        None, help="OpenAI subscription sign-in: browser or device (for SSH)."
    ),
    model: Optional[str] = typer.Option(
        None, help="Model to save (azure: the deployment name); prompts when omitted."
    ),
    api_key_env: Optional[str] = typer.Option(
        None,
        help="Environment variable holding the API key; defaults to the provider's "
        "own, such as ANTHROPIC_API_KEY.",
    ),
    endpoint: Optional[str] = typer.Option(
        None, help="Azure resource endpoint, such as https://NAME.openai.azure.com/."
    ),
    organization: Optional[str] = typer.Option(
        None, help="OpenAI API key only: organization id sent with each request."
    ),
    project: Optional[str] = typer.Option(
        None, help="OpenAI API key only: project id sent with each request."
    ),
    header: Optional[list[str]] = typer.Option(
        None,
        "--header",
        help="Extra request header, such as a gateway token, saved with the key; "
        "repeatable. NAME=VARIABLE reads the value from that environment variable, "
        "NAME alone from a hidden prompt.",
    ),
    default: bool = typer.Option(
        False, "--default", help="Make an already-saved provider the default."
    ),
    remove: bool = typer.Option(
        False, "--remove", help="Delete the provider's saved login."
    ),
) -> None:
    """List, set up, choose the default among, or remove saved provider logins.

    `yian auth` lists the saved logins and the default.

    `yian auth PROVIDER` saves the provider's credential and model; it becomes
    the default only when no default is saved. openai signs in with a
    subscription or an API key. Other providers read the key from their
    environment variable or a hidden prompt; ollama takes none, and vllm's is
    optional unless --api-key-env names its variable. A model the provider
    doesn't list, and every Azure deployment, must answer one short prompt
    before the login is saved; with a base-URL variable such as
    ANTHROPIC_BASE_URL set, the listing is skipped and the model must answer.
    azure also needs OPENAI_API_VERSION set.

    An openai API key may be saved with --organization and --project, and
    every provider sent a key with --header; the model listing and the reply
    check send them too. Values never come from the command line: --header
    NAME=VARIABLE reads one from an environment variable, and --header NAME
    from a hidden prompt. Each save states them again: a save without them
    keeps none.

    `yian auth PROVIDER --default` makes a saved login the default.
    `yian auth PROVIDER --remove` deletes one saved login.

    A failed or cancelled setup leaves the saved logins unchanged.

    \f
    Args:
        provider (`str | None`): Provider name; omitted lists the saved logins.
        method (`str | None`): `subscription` or `api-key` (openai); omitted values prompt.
        login (`str | None`): `browser` or `device` for openai subscription sign-in.
        model (`str | None`): Model id, or the Azure deployment name; omitted values
            prompt.
        api_key_env (`str | None`): API-key environment variable; the provider's
            entry in `PROVIDER_API_KEY_ENV` when omitted. Refused for ollama.
        endpoint (`str | None`): Azure resource endpoint; azure only; omitted values
            prompt.
        organization (`str | None`): OpenAI organization id; openai API key only.
        project (`str | None`): OpenAI project id; openai API key only.
        header (`list[str] | None`): Connection headers as ``NAME=VARIABLE`` or
            ``NAME``; refused for ollama and an openai subscription.
        default (`bool`): Make the saved provider the default without signing in.
        remove (`bool`): Delete the provider's saved login; accepts any saved name,
            and a lowercase saved name in any case.

    Raises:
        typer.Exit: The command fails or is cancelled; saved logins stay unchanged.
    """
    try:
        if provider is None:
            if default or remove or any(
                value is not None
                for value in (
                    method, login, model, api_key_env, endpoint, organization, project
                )
            ) or header:
                raise AuthError("Name a provider: yian auth <provider> [options]")
            _list()
            return
        provider, header_sources = _check_options(
            provider,
            method=method,
            login=login,
            model=model,
            api_key_env=api_key_env,
            endpoint=endpoint,
            default=default,
            remove=remove,
            organization=organization,
            project=project,
            header=header,
        )
        store = AuthStore()
        if remove:
            state = store.read()
            saved = {selection.provider for selection in state.selections}
            if provider not in saved and provider.lower() in saved:
                provider = provider.lower()
            was_default = state.default_provider == provider
            store.remove(provider)
            typer.echo(f"Removed the saved {provider} login")
            if was_default:
                typer.echo(_NO_DEFAULT)
            return
        if default:
            store.set_default(provider)
            typer.echo(f"Default provider: {provider}")
            return
        # Reading first also diagnoses an unusable store before account sign-in.
        current = store.read().default_provider
        if provider == "openai":
            _setup_openai(
                store,
                method,
                login,
                model,
                api_key_env or "OPENAI_API_KEY",
                current is None,
                organization,
                project,
                header_sources,
            )
        else:
            _setup_provider(
                store, provider, model, api_key_env, endpoint, current is None, header_sources
            )
        if current in (None, provider):
            typer.echo(f"Default provider: {provider}")
        else:
            typer.echo(
                f"Default provider: {current} "
                f"(switch with 'yian auth {provider} --default')"
            )
    except AuthError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    except (KeyboardInterrupt, EOFError):
        typer.echo(_CANCELLED, err=True)
        raise typer.Exit(1) from None


def _config_errors(layer: dict) -> list[dict]:
    """Check a host `llm_config` layer as `run_flow` does and return its errors.

    The credentials are popped from a copy, then the rest is validated; the first
    failing step's errors are returned. An unknown top-level key is reported without
    its name, since a pasted key could be the name; a provider block reports its own
    field problems at its block's name.

    Args:
        layer (`dict`): The layer, key included.

    Returns:
        `list[dict]`: Each distinct error as `{"field", "message"}`, where `field` is
        the error's location joined with `.`, or `None` for an unknown top-level key, a
        plain `ValueError` or a model-level error; empty when the layer is valid.
    """
    rest = dict(layer)
    try:
        pop_host_credentials(rest)
        LLMConfig.model_validate(rest)
    except ValidationError as error:
        errors = []
        for item in error.errors(include_url=False, include_input=False, include_context=False):
            loc = [str(part) for part in item["loc"]]
            if item["type"] == "extra_forbidden" and len(loc) == 1:
                entry = {"field": None, "message": _EXTRA_FIELD}
            else:
                message = item["msg"].removeprefix("Value error, ")
                entry = {"field": ".".join(loc) or None, "message": message}
            if entry not in errors:
                errors.append(entry)
        return errors
    except ValueError as error:
        return [{"field": None, "message": str(error)}]
    return []


def config_check(
    json_output: bool = typer.Option(
        False, "--json", help="Print one yian.cli.v1 JSON document on stdout."
    ),
) -> None:
    """Check the host llm_config layer given as a JSON object on stdin, key included.

    The layer is checked as a run checks the llm_config it is given; values never
    appear in the errors.

    \f
    Args:
        json_output (`bool`): Print a `yian.cli.v1` document instead of text.

    Raises:
        typer.BadParameter: stdin is not a JSON object.
        typer.Exit: With code 1 when the layer has errors.
    """

    # json.loads accepts NaN and Infinity, which aren't JSON.
    def refuse(token: str) -> None:
        raise ValueError(f"{token} is not JSON")

    try:
        layer = json.loads(sys.stdin.read(), parse_constant=refuse)
    except ValueError:  # JSONDecodeError is a ValueError
        layer = None
    if not isinstance(layer, dict):
        raise typer.BadParameter("stdin must be a JSON object (an llm_config layer)")
    errors = _config_errors(layer)
    if json_output:
        print_json({"ok": not errors, "errors": errors})
    elif not errors:
        typer.echo("ok")
    else:
        for error in errors:
            field = error["field"]
            typer.echo(f"{field}: {error['message']}" if field else error["message"], err=True)
    if errors:
        raise typer.Exit(1)


def providers(
    json_output: bool = typer.Option(
        False, "--json", help="Print one yian.cli.v1 JSON document on stdout."
    ),
) -> None:
    """List the supported providers, the login kinds each takes, and its API-key variable.

    The login kinds are those `yian auth` can save: openai also takes a
    subscription, ollama takes no key, and vllm's key is optional.

    \f
    Args:
        json_output (`bool`): Print a `yian.cli.v1` document instead of text.
    """
    listed = []
    for name in SUPPORTED_LLM_PROVIDERS:
        if name == "openai":
            methods = ["api-key", "subscription"]
        elif name == "ollama":
            methods = ["none"]
        elif name in OPTIONAL_API_KEY_PROVIDERS:
            methods = ["api-key", "none"]
        else:
            methods = ["api-key"]
        listed.append(
            {"name": name, "methods": methods, "api_key_env": PROVIDER_API_KEY_ENV[name]}
        )
    if json_output:
        print_json({"providers": listed})
        return
    for provider in listed:
        variable = provider["api_key_env"]
        suffix = f" ({variable})" if variable else ""
        typer.echo(f"{provider['name']}: {', '.join(provider['methods'])}{suffix}")
