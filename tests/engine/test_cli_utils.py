"""`cli/utils.py` credential/endpoint helpers + `yian run`'s pre-flight key check.

`ensure_api_key` / `confirm_ollama_endpoint` are exercised with a fake `questionary`
and a monkeypatched `sys.stdin.isatty` so both the interactive-prompt and the
non-interactive fallback paths are observable without a real TTY. `_agent_connections`
/ `_ensure_provider_keys` are checked against a compiled flow to confirm the
cascade-resolved providers and their connections drive which keys get ensured.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import yian.cli.run as climod
from yian.cli import utils
from yian.compose.loader import load_flow

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


class _FakeQuestionary:
    """Fake `questionary`: `.password`/`.text` return canned answers via `.ask()`.

    `answer` may be a scalar (returned on every call) or a list (consumed in order,
    one per prompt) so a single test can script a sequence like empty-then-real-key.
    """

    def __init__(self, answer):
        self._seq = list(answer) if isinstance(answer, list) else None
        self._answer = answer
        self.prompts: list[str] = []

    def _next(self):
        return self._seq.pop(0) if self._seq is not None else self._answer

    def password(self, label, **kwargs):
        self.prompts.append(label)
        return SimpleNamespace(ask=self._next)

    def text(self, label, default="", **kwargs):
        self.prompts.append(label)
        return SimpleNamespace(ask=self._next)


def _tty(monkeypatch, is_tty: bool) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: is_tty))


# --- ensure_api_key ---------------------------------------------------------- #
def test_ensure_api_key_returns_existing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-existing")
    assert utils.ensure_api_key("anthropic") == "sk-existing"


def test_ensure_api_key_keyless_provider_is_none(monkeypatch):
    # ollama is keyless — no env var, nothing to ensure.
    assert utils.ensure_api_key("ollama") is None


def test_ensure_api_key_unknown_provider_is_none():
    assert utils.ensure_api_key("totally-made-up") is None


def test_ensure_api_key_case_insensitive(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    assert utils.ensure_api_key("OpenAI") == "sk-oai"


def test_ensure_api_key_missing_non_interactive_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _tty(monkeypatch, False)  # no TTY -> can't prompt
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        utils.ensure_api_key("anthropic")


def test_ensure_api_key_interactive_prompts_and_exports(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _tty(monkeypatch, True)
    fake = _FakeQuestionary("sk-typed")
    monkeypatch.setitem(sys.modules, "questionary", fake)
    assert utils.ensure_api_key("anthropic") == "sk-typed"
    # The typed key is exported so downstream clients read it.
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-typed"
    assert any("ANTHROPIC_API_KEY" in p for p in fake.prompts)


def test_ensure_api_key_reprompts_on_empty(monkeypatch):
    """An empty interactive answer re-prompts; the next non-empty key is accepted."""
    import os

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _tty(monkeypatch, True)
    fake = _FakeQuestionary(["", "sk-typed"])  # fingerslip, then a real key
    monkeypatch.setitem(sys.modules, "questionary", fake)
    assert utils.ensure_api_key("anthropic") == "sk-typed"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-typed"
    assert len(fake.prompts) == 2  # the loop re-asked


def test_ensure_api_key_cancel_raises(monkeypatch):
    """Ctrl-C/Esc (questionary returns None) raises at once -> clean exit-1 upstream."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _tty(monkeypatch, True)
    fake = _FakeQuestionary([None])
    monkeypatch.setitem(sys.modules, "questionary", fake)
    with pytest.raises(RuntimeError, match="no API key provided"):
        utils.ensure_api_key("anthropic")
    assert len(fake.prompts) == 1  # cancel does not spend extra tries


def test_ensure_api_key_exhausts_after_max(monkeypatch):
    """Three empty answers exhaust the cap: a distinct 'after N attempts' raise, exactly 3 prompts."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _tty(monkeypatch, True)
    fake = _FakeQuestionary(["", "", ""])
    monkeypatch.setitem(sys.modules, "questionary", fake)
    with pytest.raises(RuntimeError, match="after 3 attempts"):  # not the cancel message
        utils.ensure_api_key("anthropic")
    assert len(fake.prompts) == 3  # capped at MAX_KEY_PROMPTS


def test_ensure_api_key_accepts_on_last_attempt(monkeypatch):
    """A non-empty key on the 3rd (last) try is still accepted -> range(3) gives 3 usable tries.

    The leading whitespace entry also exercises whitespace-only being treated as empty.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _tty(monkeypatch, True)
    fake = _FakeQuestionary(["  ", "", "sk-typed"])  # whitespace, empty, then a real key
    monkeypatch.setitem(sys.modules, "questionary", fake)
    assert utils.ensure_api_key("anthropic") == "sk-typed"
    assert len(fake.prompts) == 3


def test_ensure_api_key_strips_surrounding_whitespace(monkeypatch):
    """A pasted key with edge whitespace is stripped before export/return."""
    import os

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _tty(monkeypatch, True)
    monkeypatch.setitem(sys.modules, "questionary", _FakeQuestionary(["  sk-abc  "]))
    assert utils.ensure_api_key("anthropic") == "sk-abc"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-abc"


def test_ensure_api_key_interactive_false_forces_non_interactive(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    _tty(monkeypatch, True)  # a TTY is present, but interactive=False overrides
    with pytest.raises(RuntimeError, match=r"OPENAI_API_KEY.*run 'yian auth openai'"):
        utils.ensure_api_key("openai", interactive=False)


# --- confirm_ollama_endpoint ------------------------------------------------- #
def test_confirm_ollama_endpoint_non_interactive_returns_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    _tty(monkeypatch, False)
    assert utils.confirm_ollama_endpoint() == "http://localhost:11434"


def test_confirm_ollama_endpoint_non_interactive_returns_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
    _tty(monkeypatch, False)
    assert utils.confirm_ollama_endpoint() == "http://gpu-box:11434"


def test_confirm_ollama_endpoint_interactive_prompts_and_exports(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    _tty(monkeypatch, True)
    monkeypatch.setitem(sys.modules, "questionary", _FakeQuestionary("http://h200:11434"))
    assert utils.confirm_ollama_endpoint() == "http://h200:11434"
    import os

    assert os.environ["OLLAMA_BASE_URL"] == "http://h200:11434"


# --- _agent_connections / _ensure_provider_keys pre-flight ------------------- #
def _hello_agent():
    return load_flow((_SEEDS / "00-hello-agent.yaml").read_text(), search_paths=[_SEEDS])


def _connection_sources(loaded, cli_cfg):
    """Return each distinct agent connection as ``(provider, source)``."""
    return [
        (provider, connection["source"])
        for provider, connection in climod._agent_connections(loaded, cli_cfg)
    ]


def test_agent_connections_fails_when_nothing_sets_a_provider(monkeypatch):
    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("YIAN_DEFAULT_MODEL", raising=False)
    with pytest.raises(ValueError, match=r"Agent 'note' has no provider or model; .*--provider/--model"):
        climod._agent_connections(_hello_agent(), {})


def test_agent_connections_use_the_environment_then_the_saved_provider(monkeypatch):
    from yian.llm_clients.auth import ApiKeyCredentials, AuthStore

    monkeypatch.setenv("YIAN_DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "env-model")
    assert _connection_sources(_hello_agent(), {}) == [("anthropic", "environment")]
    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER")
    monkeypatch.delenv("YIAN_DEFAULT_MODEL")
    AuthStore().save("xai", ApiKeyCredentials("key"), "saved-model")
    assert _connection_sources(_hello_agent(), {}) == [("xai", "store")]


def test_agent_connections_cli_override_fills_gap(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-present")
    # The agent sets no provider, so the CLI layer fills it.
    assert _connection_sources(_hello_agent(), {"provider": "openai"}) == [
        ("openai", "environment")
    ]


def test_ensure_provider_keys_ok_when_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-present")
    climod._ensure_provider_keys(_hello_agent(), {})  # no raise


def test_ensure_provider_keys_raises_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _tty(monkeypatch, False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        climod._ensure_provider_keys(_hello_agent(), {})


def test_saved_login_needs_no_key_prompt(monkeypatch):
    from yian.llm_clients.auth import ApiKeyCredentials, AuthStore

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = _FakeQuestionary("typed")
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _tty(monkeypatch, True)
    AuthStore().save("anthropic", ApiKeyCredentials("saved-k"), "m", make_default=False)

    assert utils.ensure_api_key("anthropic") == "saved-k"
    assert fake.prompts == []


def test_ensure_env_key_suggests_a_saved_login(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"ANTHROPIC_API_KEY.*run 'yian auth anthropic'"):
        utils.ensure_env_key("anthropic", interactive=False)


def test_ensure_env_key_skips_an_unset_optional_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert utils.ensure_env_key("vllm", interactive=False) is None
    monkeypatch.setenv("VLLM_API_KEY", "sk-vllm")
    assert utils.ensure_env_key("vllm", interactive=False) == "sk-vllm"
