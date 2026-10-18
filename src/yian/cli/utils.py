"""CLI helpers: provider credentials and endpoints, and the `--json` writer.

These are the integration points the `llm_clients` modules point at by name:

- `ensure_env_key` — the interactive key prompt referenced by
  `llm_clients/api_key_env.py`. `yian run` and `yian chat` call it pre-flight
  (see `cli/run.py`) for agents without a saved login, so a missing key
  surfaces as an upfront prompt instead of a cryptic failure on the first API
  call.
- `ensure_api_key` — resolves a provider's saved login first, then falls back
  to `ensure_env_key`.
- `confirm_ollama_endpoint` — referenced by `llm_clients/model_catalog.py`
  as the step that surfaces the resolved Ollama endpoint right after provider
  selection; the future provider-selection wizard is its caller.

`ensure_env_key` and `confirm_ollama_endpoint` are TTY-aware: they prompt only
when attached to an interactive terminal and otherwise fall back (return the
ambient value, or raise for a hard-missing required key) so non-interactive
runs stay scriptable.

`print_json` writes the one `yian.cli.v1` document a `--json` command prints.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional, TextIO

from yian.defaults import current
from yian.llm_clients.api_key_env import OPTIONAL_API_KEY_PROVIDERS, get_api_key_env

# Mirror of `ollama_client._DEFAULT_BASE_URL`. Duplicated (not imported) so this
# module stays import-light — importing the ollama client pulls `langchain_ollama`
# at module top, which we don't want on the CLI's hot path.
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

JSON_FORMAT = "yian.cli.v1"


def print_json(document: dict[str, Any], file: Optional[TextIO] = None) -> None:
    """Write `document` as one `yian.cli.v1` JSON object and a newline, then flush.

    Args:
        document (`dict[str, Any]`):
            The document's keys; `format` is added first.
        file (`TextIO`, *optional*, defaults to `None`):
            Where to write; `None` means `sys.stdout`.

    Raises:
        `ValueError`:
            `document` holds a NaN or infinite float, which JSON can't represent.
    """
    out = sys.stdout if file is None else file
    out.write(json.dumps({"format": JSON_FORMAT, **document}, allow_nan=False) + "\n")
    out.flush()


def ensure_api_key(provider: str, *, interactive: bool = True) -> Optional[str]:
    """Return the key `provider`'s saved login or environment supplies, prompting if neither does.

    A saved login of any provider resolves its stored credential, so it never
    prompts. Otherwise the provider's key environment variable is checked with
    [`ensure_env_key`][yian.cli.utils.ensure_env_key].

    Args:
        provider (`str`):
            The LLM provider name (e.g. `"anthropic"`, `"openai"`), matched
            case-insensitively.
        interactive (`bool`, *optional*, defaults to `True`):
            Whether an interactive prompt is permitted. Set `False` to force the
            non-interactive path (return existing or raise) regardless of TTY.

    Returns:
        `Optional[str]`:
            The resolved API key, or `None` for a saved subscription or a provider
            that needs no key (keyless, unknown, or an optional key left unset).

    Raises:
        `AuthError`:
            The saved login cannot be read.
        `RuntimeError`:
            No saved login, a required key is unset, and it cannot be prompted
            for; see [`ensure_env_key`][yian.cli.utils.ensure_env_key].
        `ValueError`:
            The saved logins or default-provider environment are invalid.
    """
    from yian.llm_clients.auth import ApiKeyCredentials
    from yian.llm_clients.connection import connection_credentials, connection_snapshot

    name = provider.lower()
    connection = connection_snapshot().connections.get(name)
    if connection is not None and connection["source"] == "store":
        credentials = connection_credentials(connection, name)
        return credentials.api_key if isinstance(credentials, ApiKeyCredentials) else None
    return ensure_env_key(provider, interactive=interactive)


def ensure_env_key(provider: str, *, interactive: bool = True) -> Optional[str]:
    """Return `provider`'s API key from its environment variable, prompting for it if unset.

    The env var for the key is resolved via
    [`get_api_key_env`][yian.llm_clients.api_key_env.get_api_key_env]; saved
    logins are not consulted. A keyless provider (`ollama`) or an unknown one maps
    to no env var, so nothing is checked and `None` is returned; so does a
    provider whose key is optional (`vllm`) when its variable is unset. When the
    env var is set its value is returned unchanged. When it is unset and a TTY is
    attached, the user is prompted (the answer is stripped and exported into the
    process env so downstream clients read it); an empty answer is treated as a
    fingerslip and re-prompts, up to `current().max_key_prompts` times. With no
    TTY (or `interactive=False`) a required-but-missing key raises rather than
    hanging on a prompt.

    Args:
        provider (`str`):
            The LLM provider name (e.g. `"anthropic"`, `"openai"`), matched
            case-insensitively against the provider → env-var table.
        interactive (`bool`, *optional*, defaults to `True`):
            Whether an interactive prompt is permitted. Set `False` to force the
            non-interactive path (return existing or raise) regardless of TTY.

    Returns:
        `Optional[str]`:
            The resolved API key, or `None` when the provider needs none (keyless,
            unknown, or an optional key left unset).

    Raises:
        `RuntimeError`:
            The provider requires a key, none is set, and one of: no interactive
            prompt is available (no TTY, or `interactive=False`); the user cancels
            the prompt (Ctrl-C / Esc); or the user submits an empty answer
            `current().max_key_prompts` times in a row. A single empty answer is
            not an error — it re-prompts.
    """
    env_var = get_api_key_env(provider)
    if env_var is None:
        return None  # keyless (e.g. ollama) or unknown provider — nothing to ensure
    existing = os.environ.get(env_var)
    if existing:
        return existing
    if provider.lower() in OPTIONAL_API_KEY_PROVIDERS:
        return None
    if not (interactive and sys.stdin.isatty()):
        raise RuntimeError(
            f"provider {provider!r} needs an API key but ${env_var} is not set "
            f"(set it in the environment, or run interactively to be prompted). "
            f"Or run 'yian auth {provider.lower()}' to save a login"
        )
    import questionary

    # Interactive: an empty/whitespace answer is a fingerslip (the key can't be checked
    # locally, so it is never a wrong guess) -> re-prompt, capped so a caller feeding
    # empty answers forever can't hang. An explicit cancel (Ctrl-C / Esc,
    # where questionary returns None) aborts at once. The no-TTY path above already raised up front.
    tries = current().max_key_prompts
    for _ in range(tries):
        key = questionary.password(f"Enter {env_var} for provider {provider!r}:").ask()
        if key is None:  # Ctrl-C / Esc — the user chose to cancel; does not spend an extra try
            raise RuntimeError(f"no API key provided for {provider!r} (${env_var})")
        key = key.strip()  # strip once; the guard and the export share the stripped value
        if key:
            os.environ[env_var] = key
            return key
        # empty / whitespace-only — hit Enter with no real input; this attempt is spent, ask again
    raise RuntimeError(
        f"no API key provided for {provider!r} (${env_var}) after {tries} attempts"
    )


def confirm_ollama_endpoint(*, interactive: bool = True) -> str:
    """Confirm (or override) the Ollama base URL, prompting when interactive.

    The current endpoint is `$OLLAMA_BASE_URL` if set, else the localhost default
    that mirrors `ollama_client`. When a TTY is attached the value is offered as an
    editable default and the confirmed answer is exported back into
    `$OLLAMA_BASE_URL` so the client resolves the same endpoint. Non-interactively
    (or `interactive=False`) the current value is returned unchanged.

    Args:
        interactive (`bool`, *optional*, defaults to `True`):
            Whether an interactive prompt is permitted. Set `False` to force the
            non-interactive path (return the current value) regardless of TTY.

    Returns:
        `str`:
            The resolved Ollama base URL.
    """
    current = os.environ.get("OLLAMA_BASE_URL") or _OLLAMA_DEFAULT_BASE_URL
    if not (interactive and sys.stdin.isatty()):
        return current
    import questionary

    answer = questionary.text("Ollama endpoint:", default=current).ask()
    endpoint = (answer or current).strip()
    os.environ["OLLAMA_BASE_URL"] = endpoint
    return endpoint
