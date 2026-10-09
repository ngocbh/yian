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


def auth(
    provider: str = typer.Argument("openai", help="Provider to connect (openai)."),
    method: Optional[str] = typer.Option(
        None, help="subscription or api-key; prompts when omitted."
    ),
    login: Optional[str] = typer.Option(
        None, help="Subscription sign-in: browser or device (for SSH)."
    ),
    model: Optional[str] = typer.Option(
        None, help="Available model to save as the default; prompts when omitted."
    ),
    api_key_env: str = typer.Option(
        "OPENAI_API_KEY", help="Environment variable containing the API key."
    ),
) -> None:
    """Connect OpenAI and save its authentication method and default model together.

    Login and model discovery finish before the private store changes. Tokens and
    keys are supplied by sign-in, a hidden prompt, or the named environment variable.

    \f
    Args:
        provider (`str`): Provider identifier; defaults to `openai`.
        method (`str | None`): `subscription` or `api-key`; omitted values prompt.
        login (`str | None`): `browser` or `device` for subscription sign-in.
        model (`str | None`): An available model identifier; omitted values prompt.
        api_key_env (`str`): API-key environment variable; defaults to `OPENAI_API_KEY`.

    Raises:
        typer.Exit: Setup fails or is cancelled; the previous selection stays active.
    """
    try:
        if provider.lower() != "openai":
            raise AuthError(
                "Authentication setup currently supports openai; other providers use their API-key environment variables"
            )
        if method is not None and method not in ("subscription", "api-key"):
            raise AuthError("Choose --method subscription or --method api-key")
        if login is not None and login not in ("browser", "device"):
            raise AuthError("Choose --login browser or --login device")
        store = AuthStore()
        store.read()  # Diagnose an unusable existing store before account sign-in.
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
            model = _choose(
                "Default model:", choices, default=choices[0], option="--model"
            )
        if model not in models:
            raise AuthError("The selected model is not available for this connection")
        store.save("openai", credentials, model)
        typer.echo(f"Saved OpenAI {method} authentication. Default model: {model}")
    except AuthError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    except (KeyboardInterrupt, EOFError):
        typer.echo(
            "Authentication cancelled; the previous connection is unchanged", err=True
        )
        raise typer.Exit(1) from None
