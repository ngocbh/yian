"""Interactive provider setup over staged authentication and an atomic store."""

from __future__ import annotations

import os
import sys
from typing import Optional
import webbrowser

import typer

from yian.llm_clients.auth import (
    ApiKeyCredentials,
    AuthError,
    AuthStore,
    OpenAIAuth,
)
from yian.llm_clients.config import SUPPORTED_LLM_PROVIDERS

_KINDS = {"api_key": "api-key", "subscription": "subscription", "none": "none"}
_NO_DEFAULT = (
    "No default provider is saved; choose one with 'yian auth <provider> --default'"
)


def _choose(message: str, choices: list[str], *, default: str, option: str) -> str:
    """Select a value on a terminal, or require an explicit command option."""
    if not sys.stdin.isatty():
        raise AuthError(
            f"Authentication needs {option} when no interactive terminal is available"
        )
    import questionary

    answer = questionary.select(message, choices=choices, default=default).ask()
    if answer is None:
        raise AuthError(
            "Authentication cancelled; the previous connection is unchanged"
        )
    return answer


def _api_key(variable: str) -> str:
    """Read a selected environment key or ask without echoing its contents."""
    key = os.environ.get(variable)
    if key:
        return key
    if not sys.stdin.isatty():
        raise AuthError(
            f"Set {variable} before API-key setup, or run in an interactive terminal"
        )
    import questionary

    key = questionary.password("OpenAI API key:").ask()
    if not key or not key.strip():
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
        verb = "is" if len(overrides) == 1 else "are"
        typer.echo(
            f"{' and '.join(overrides)} {verb} set in the environment and "
            "override the saved choice"
        )


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

    Raises:
        AuthError: The provider or an option does not apply.
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
        credentials = ApiKeyCredentials(_api_key(api_key_env))
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
        None, help="Available model to save for the provider; prompts when omitted."
    ),
    api_key_env: Optional[str] = typer.Option(
        None, help="Environment variable containing the API key (openai: OPENAI_API_KEY)."
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

    `yian auth` lists the saved logins and the default. `yian auth PROVIDER` signs in
    and saves the provider's credential and model; it becomes the default only when
    none is saved. `yian auth PROVIDER --default` makes a saved login the default
    without signing in. `yian auth PROVIDER --remove` deletes one saved login.

    Options are checked before the store is read or any request is made. Login and
    model discovery finish before the private store changes. Tokens and keys are
    supplied by sign-in, a hidden prompt, or the named environment variable.

    \f
    Args:
        provider (`str | None`): Provider name; omitted lists the saved logins.
        method (`str | None`): `subscription` or `api-key` (openai); omitted values prompt.
        login (`str | None`): `browser` or `device` for openai subscription sign-in.
        model (`str | None`): An available model identifier; omitted values prompt.
        api_key_env (`str | None`): API-key environment variable; `OPENAI_API_KEY`
            for openai when omitted. Refused for ollama.
        endpoint (`str | None`): Azure resource endpoint; azure only.
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
        if provider != "openai":
            raise AuthError(f"Setting up {provider} is not supported yet")
        _setup_openai(
            store, method, login, model, api_key_env or "OPENAI_API_KEY", current is None
        )
        if current is None:
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
        typer.echo(
            "Authentication cancelled; the previous connection is unchanged", err=True
        )
        raise typer.Exit(1) from None
