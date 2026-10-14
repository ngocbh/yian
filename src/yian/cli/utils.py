"""Interactive CLI helpers for provider credentials and endpoints.

These are the integration points the `llm_clients` modules point at by name:

- `ensure_api_key` — the interactive key prompt referenced by
  `llm_clients/api_key_env.py`. Called pre-flight by `yian run` (see
  `cli/run.py`) so a missing key surfaces as an upfront prompt instead of a
  cryptic failure on the first API call.
- `confirm_ollama_endpoint` — referenced by `llm_clients/model_catalog.py`
  as the step that surfaces the resolved Ollama endpoint right after provider
  selection; the future provider-selection wizard is its caller.
- `displayable` — lets `yian run` and `yian chat` print values that hold ints
  too long for `repr`.

Both are TTY-aware: they prompt only when attached to an interactive terminal
and otherwise fall back (return the ambient value, or raise for a hard-missing
required key) so non-interactive runs stay scriptable.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import yian.json_codec as json_codec
from yian.llm_clients.api_key_env import get_api_key_env

# Mirror of `ollama_client._DEFAULT_BASE_URL`. Duplicated (not imported) so this
# module stays import-light — importing the ollama client pulls `langchain_ollama`
# at module top, which we don't want on the CLI's hot path.
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

# How many times the interactive key prompt re-asks on an empty answer before giving
# up. An empty answer is a fingerslip (the key can't be validated locally, so it is
# never a "wrong guess"); we re-prompt to be friendly, but bound the loop so a
# non-interactive-but-TTY caller feeding empty forever can't hang. 3 mirrors sudo's
# `passwd_tries` default — the most familiar credential prompt on Linux.
MAX_KEY_PROMPTS = 3

# Ints up to 2000 bits are always below Python's int/str digit limit, so `repr`
# prints them; longer ones may exceed it and are spelled by `json_codec`.
_SHORT_INT_BITS = 2000


class _IntDigits:
    """Stands in for a long int when printing; its `repr` is the int's digits."""

    def __init__(self, value: int) -> None:
        self._text = json_codec.int_text(value)

    def __repr__(self) -> str:
        return self._text


def displayable(value: Any) -> Any:
    """Return `value` with every long int swapped for an object whose `repr` is its digits.

    `repr` and `str` of an int past Python's process-wide digit limit raise
    `ValueError`; the returned value prints instead. Dicts, lists and tuples are
    walked (and rebuilt only when they hold a long int); any other value is
    returned unchanged.

    Args:
        value (`Any`): The value to print.

    Returns:
        `Any`: `value` itself, or a copy whose long ints print as their digits.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return _IntDigits(value) if value.bit_length() > _SHORT_INT_BITS else value
    if isinstance(value, dict):
        items = {key: displayable(item) for key, item in value.items()}
        if all(items[key] is item for key, item in value.items()):
            return value
        return items
    if isinstance(value, (list, tuple)):
        items = [displayable(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return items if isinstance(value, list) else tuple(items)
    return value


def ensure_api_key(provider: str, *, interactive: bool = True) -> Optional[str]:
    """Return `provider`'s API key, prompting for it interactively if it is unset.

    The env var for the key is resolved via
    [`get_api_key_env`][yian.llm_clients.api_key_env.get_api_key_env].
    A keyless provider (`ollama`) or an unknown one maps to no env var, so nothing
    is checked and `None` is returned. When the env var is set its value is
    returned unchanged. When it is unset and a TTY is attached, the user is
    prompted (the answer is stripped and exported into the process env so downstream
    clients read it); an empty answer is treated as a fingerslip and re-prompts, up to
    `MAX_KEY_PROMPTS` times. With no TTY (or `interactive=False`) a required-but-missing
    key raises rather than hanging on a prompt.

    Args:
        provider (`str`):
            The LLM provider name (e.g. `"anthropic"`, `"openai"`), matched
            case-insensitively against the provider → env-var table.
        interactive (`bool`, *optional*, defaults to `True`):
            Whether an interactive prompt is permitted. Set `False` to force the
            non-interactive path (return existing or raise) regardless of TTY.

    Returns:
        `Optional[str]`:
            The resolved API key, or `None` when the provider needs none (keyless
            or unknown).

    Raises:
        `RuntimeError`:
            The provider requires a key, none is set, and one of: no interactive
            prompt is available (no TTY, or `interactive=False`); the user cancels
            the prompt (Ctrl-C / Esc); or the user submits an empty answer
            `MAX_KEY_PROMPTS` times in a row. A single empty answer is not an
            error — it re-prompts.
    """
    if provider.lower() == "openai":
        from yian.llm_clients.connection import connection_credentials, connection_snapshot
        from yian.llm_clients.auth import ApiKeyCredentials
        connection = connection_snapshot().connections["openai"]
        if connection["source"] == "store":
            credentials = connection_credentials(connection)
            return credentials.api_key if isinstance(credentials, ApiKeyCredentials) else None
    env_var = get_api_key_env(provider)
    if env_var is None:
        return None  # keyless (e.g. ollama) or unknown provider — nothing to ensure
    existing = os.environ.get(env_var)
    if existing:
        return existing
    if not (interactive and sys.stdin.isatty()):
        raise RuntimeError(
            f"provider {provider!r} needs an API key but ${env_var} is not set "
            f"(set it in the environment, or run interactively to be prompted)"
            + (". Or run 'yian auth openai' to choose a connection" if provider.lower() == "openai" else "")
        )
    import questionary

    # Interactive: an empty/whitespace answer is a fingerslip -> re-prompt, capped at
    # MAX_KEY_PROMPTS tries so the loop can't run forever. An explicit cancel (Ctrl-C / Esc,
    # where questionary returns None) aborts at once. The no-TTY path above already raised up front.
    for _ in range(MAX_KEY_PROMPTS):
        key = questionary.password(f"Enter {env_var} for provider {provider!r}:").ask()
        if key is None:  # Ctrl-C / Esc — the user chose to cancel; does not spend an extra try
            raise RuntimeError(f"no API key provided for {provider!r} (${env_var})")
        key = key.strip()  # strip once; the guard and the export share the stripped value
        if key:
            os.environ[env_var] = key
            return key
        # empty / whitespace-only — hit Enter with no real input; this attempt is spent, ask again
    raise RuntimeError(
        f"no API key provided for {provider!r} (${env_var}) after {MAX_KEY_PROMPTS} attempts"
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
