"""Interactive provider setup over staged authentication and an atomic store."""

from __future__ import annotations

import os
import sys
from typing import Optional
import warnings
import webbrowser

import typer

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
from yian.llm_clients.config import SUPPORTED_LLM_PROVIDERS, validate_base_url
from yian.llm_clients.factory import create_llm_client
from yian.llm_clients.model_list import ModelListError, list_models
from yian.llm_clients.redact import redact_key, scrub_key

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
    "google": ("GOOGLE_GEMINI_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI"),
}


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
            endpoint = f", endpoint {selection.endpoint}" if selection.endpoint else ""
            default = " (default)" if selection.provider == state.default_provider else ""
            typer.echo(
                f"  {selection.provider}: {_KINDS[selection.method]}, "
                f"model {selection.model}{endpoint}{default}"
            )
        if state.default_provider is None:
            typer.echo(_NO_DEFAULT)
    if overrides:
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
) -> str:
    """Validate the option combination and return the provider name to use.

    Azure setup also needs `OPENAI_API_VERSION` and a valid `endpoint`, when given.

    Raises:
        AuthError: The provider or an option does not apply, or Azure setup lacks
            its API version or has an invalid endpoint.
    """
    setup = {
        "--method": method,
        "--login": login,
        "--model": model,
        "--api-key-env": api_key_env,
        "--endpoint": endpoint,
    }
    given = [option for option, value in setup.items() if value is not None]
    if default and remove:
        raise AuthError("Use either --default or --remove, not both")
    if (default or remove) and given:
        action = "--default" if default else "--remove"
        raise AuthError(f"{action} does not take {', '.join(given)}")
    if remove:
        return provider  # A login saved by another yian version may name any provider.
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
    if name == "ollama" and api_key_env is not None:
        raise AuthError("ollama takes no API key; omit --api-key-env")
    if method is not None and method not in ("subscription", "api-key"):
        raise AuthError("Choose --method subscription or --method api-key")
    if login is not None and login not in ("browser", "device"):
        raise AuthError("Choose --login browser or --login device")
    if method == "api-key" and login is not None:
        raise AuthError("--login applies to subscription authentication")
    if name == "azure" and not default:
        # The Azure client reads the API version only from the environment.
        if not os.environ.get("OPENAI_API_VERSION"):
            raise AuthError(
                "Azure needs OPENAI_API_VERSION set to its API version (runs read it too)"
            )
        if endpoint is not None:
            _check_endpoint(endpoint)
    return name


def _setup_openai(
    store: AuthStore,
    method: Optional[str],
    login: Optional[str],
    model: Optional[str],
    api_key_env: str,
    make_default: bool,
) -> None:
    """Sign in to OpenAI, choose an available model, and save the login."""
    method = method or _choose(
        "OpenAI authentication method:",
        ["subscription", "api-key"],
        default="subscription",
        option="--method",
    )
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
        credentials = ApiKeyCredentials(_api_key("openai", api_key_env))
        models = service.api_models(credentials)
        choices = _text_models(models)
    if model is None:
        if not choices:
            raise AuthError(
                "No general text models were returned; use --model with an available text model"
            )
        model = _choose("Model:", choices, default=choices[0], option="--model")
    if model not in models:
        raise AuthError("The selected model is not available for this connection")
    store.save("openai", credentials, model, make_default=make_default)
    typer.echo(f"Saved OpenAI {method} authentication with model {model}")


def _model_answers(
    provider: str, model: str, key: Optional[str], endpoint: Optional[str]
) -> Optional[str]:
    """Send the model one short prompt and say why it failed, if it did.

    The client is built from the given key and endpoint only. Without a key, vLLM
    gets the dummy key its client sends to a keyless server, and Ollama sends none.

    Args:
        provider (`str`): Provider name.
        model (`str`): Model id, or the Azure deployment name.
        key (`str | None`): API key to send, or `None` for none.
        endpoint (`str | None`): Azure resource endpoint, or `None`.

    Returns:
        `str | None`: `None` when the model replied; otherwise the error type and
        a short message without the key, or the install hint for a missing extra.
    """
    # The Ollama client ignores timeout and max_retries.
    options: dict = {"max_retries": 0, "timeout": 30}
    if key is not None:
        options["api_key"] = key
    elif provider in OPTIONAL_API_KEY_PROVIDERS:
        # Passed explicitly so an ambient VLLM_API_KEY doesn't stand in for the
        # keyless login being saved.
        options["api_key"] = "EMPTY"
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
        scrub_key(error, key)
        # Some errors, such as pydantic's ValidationError, build their text from
        # fields scrub_key does not rewrite, so the text is redacted again.
        text = " ".join(redact_key(str(error), key).split())
        return f"{type(error).__name__}: {text[:200]}"
    return None


def _setup_provider(
    store: AuthStore,
    provider: str,
    model: Optional[str],
    api_key_env: Optional[str],
    endpoint: Optional[str],
    make_default: bool,
) -> None:
    """Read a provider's key, endpoint, and model, check the model, and save the login.

    A model in the provider's listing is accepted as is. Any other model, every
    Azure deployment, and every model when a variable in `_ELSEWHERE_ENV` sends runs
    to another host, must answer one short prompt. vLLM's key is optional unless
    `api_key_env` names its variable.

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
            optional=provider in OPTIONAL_API_KEY_PROVIDERS and api_key_env is None,
        )
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
                models = list_models(provider, key, base_url)
            except ModelListError as error:
                failure = str(error)
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
        problem = _model_answers(provider, model, key, endpoint)
        if problem:
            raise AuthError(
                f"The model didn't answer: {problem}; the previous login is unchanged"
            )
    store.save(
        provider,
        ApiKeyCredentials(key) if key else None,
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
        default (`bool`): Make the saved provider the default without signing in.
        remove (`bool`): Delete the provider's saved login; accepts any saved name.

    Raises:
        typer.Exit: The command fails or is cancelled; saved logins stay unchanged.
    """
    try:
        if provider is None:
            if default or remove or any(
                value is not None
                for value in (method, login, model, api_key_env, endpoint)
            ):
                raise AuthError("Name a provider: yian auth <provider> [options]")
            _list()
            return
        provider = _check_options(
            provider,
            method=method,
            login=login,
            model=model,
            api_key_env=api_key_env,
            endpoint=endpoint,
            default=default,
            remove=remove,
        )
        store = AuthStore()
        if remove:
            was_default = store.read().default_provider == provider
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
                store, method, login, model, api_key_env or "OPENAI_API_KEY", current is None
            )
        else:
            _setup_provider(store, provider, model, api_key_env, endpoint, current is None)
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
