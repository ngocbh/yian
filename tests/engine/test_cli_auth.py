"""Provider setup activates one verified method/model pair at the final step."""

import importlib
import json

import pytest
from typer.testing import CliRunner

from yian.cli import app
from yian.llm_clients.auth import (
    ApiKeyCredentials,
    AuthError,
    AuthStore,
    OAuthCredentials,
)

auth_cli = importlib.import_module("yian.cli.auth")
runner = CliRunner()


class Service:
    def __init__(self):
        self.calls = []

    def device_login(self, display):
        self.calls.append("device")
        display("https://example.invalid/device", "TEST-CODE")
        return OAuthCredentials(
            "private-access", "private-refresh", 9999999999, "test-account"
        )

    def browser_login(self, present):
        self.calls.append("browser")
        return OAuthCredentials(
            "private-access", "private-refresh", 9999999999, "test-account"
        )

    def subscription_models(self, credentials):
        self.calls.append("subscription_models")
        return ("gpt-test", "gpt-other")

    def api_models(self, credentials):
        self.calls.append("api_models")
        assert credentials.api_key == "private-key"
        return ("embedding-test", "gpt-test", "gpt-other")


@pytest.fixture
def setup_auth(monkeypatch, tmp_path):
    path = tmp_path / "auth" / "auth.json"
    monkeypatch.setenv("YIAN_AUTH_FILE", str(path))
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    service = Service()
    monkeypatch.setattr(auth_cli, "OpenAIAuth", lambda: service)
    return AuthStore(path), service


@pytest.mark.parametrize(
    "method,login",
    [("subscription", "device"), ("subscription", "browser"), ("api-key", None)],
)
def test_explicit_setup_saves_method_and_model_without_echoing_secrets(
    setup_auth, method, login
):
    store, service = setup_auth
    args = ["auth", "openai", "--method", method, "--model", "gpt-test"]
    if login:
        args += ["--login", login]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    selected = store.selection()
    assert selected.provider == "openai" and selected.model == "gpt-test"
    assert selected.method == ("api_key" if method == "api-key" else "subscription")
    assert "private-" not in result.output
    assert "Saved OpenAI" in result.output
    assert service.calls == (
        [login, "subscription_models"] if login else ["api_models"]
    )


def test_interactive_setup_defaults_to_subscription_and_catalog_first_model(
    monkeypatch, setup_auth
):
    store, service = setup_auth
    choices = []

    def choose(message, options, *, default, option):
        choices.append((option, default, options))
        return "device" if option == "--login" else default

    monkeypatch.setattr(auth_cli, "_choose", choose)
    result = runner.invoke(app, ["auth", "openai"])
    assert result.exit_code == 0, result.output
    assert choices[0][1] == "subscription"
    assert choices[-1][1] == "gpt-test"
    assert store.selection().model == "gpt-test"


@pytest.mark.parametrize("failure", ["login", "model", "cancel", "missing_model"])
def test_failed_or_cancelled_setup_preserves_previous_connection(
    monkeypatch, setup_auth, failure
):
    store, service = setup_auth
    previous = store.save("openai", ApiKeyCredentials("old-key"), "old-model")
    before = store.path.read_bytes()

    def fail(*args, **kwargs):
        raise AuthError("Simulated setup failure")

    args = ["auth", "openai", "--method", "subscription", "--login", "device"]
    if failure == "login":
        monkeypatch.setattr(service, "device_login", fail)
    elif failure == "model":
        monkeypatch.setattr(service, "subscription_models", fail)
    elif failure == "cancel":
        monkeypatch.setattr(auth_cli, "_choose", fail)
    if failure != "cancel":
        args += ["--model", "unavailable" if failure == "missing_model" else "gpt-test"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert store.path.read_bytes() == before
    assert store.selection() == previous
    assert "old-key" not in result.output and "private-" not in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["auth", "unknown"],
        ["auth", "openai", "--method", "unknown"],
        ["auth", "openai", "--method", "subscription", "--login", "unknown"],
        ["auth", "openai", "--method", "api-key", "--login", "device"],
    ],
)
def test_invalid_setup_options_fail_before_external_requests(setup_auth, args):
    store, service = setup_auth
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert service.calls == [] and not store.path.exists()


def test_noninteractive_setup_requires_explicit_choices(setup_auth):
    store, service = setup_auth
    result = runner.invoke(app, ["auth", "openai"])
    assert result.exit_code == 1 and "--method" in result.output
    assert service.calls == [] and not store.path.exists()


def _save_each_kind(store):
    store.save(
        "openai",
        OAuthCredentials("private-access", "private-refresh", 9999999999, "test-account"),
        "gpt-test",
    )
    store.save("anthropic", ApiKeyCredentials("private-key"), "claude-test")
    store.save("ollama", None, "llama-test")
    store.save(
        "azure",
        ApiKeyCredentials("private-azure"),
        "gpt-azure",
        endpoint="https://r.openai.azure.com/",
    )


def test_list_shows_saved_logins_without_secrets(monkeypatch, setup_auth):
    store, service = setup_auth
    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER")
    monkeypatch.delenv("YIAN_DEFAULT_MODEL")
    result = runner.invoke(app, ["auth"])
    assert result.exit_code == 0, result.output
    assert "No login is saved" in result.output and "yian auth <provider>" in result.output

    _save_each_kind(store)
    store.set_default("anthropic")
    result = runner.invoke(app, ["auth"])
    assert result.exit_code == 0, result.output
    lines = {
        line.split(":")[0].strip(): line
        for line in result.output.splitlines()
        if line.startswith("  ")
    }
    assert set(lines) == {"openai", "anthropic", "ollama", "azure"}
    assert "subscription" in lines["openai"] and "gpt-test" in lines["openai"]
    assert "api-key" in lines["anthropic"] and "claude-test" in lines["anthropic"]
    assert "none" in lines["ollama"] and "llama-test" in lines["ollama"]
    assert "api-key" in lines["azure"] and "https://r.openai.azure.com/" in lines["azure"]
    assert [name for name, line in lines.items() if "(default)" in line] == ["anthropic"]
    assert "private-" not in result.output and "YIAN_DEFAULT" not in result.output
    assert service.calls == []

    monkeypatch.setenv("YIAN_DEFAULT_PROVIDER", "openai")
    result = runner.invoke(app, ["auth"])
    assert result.exit_code == 0, result.output
    assert "YIAN_DEFAULT_PROVIDER" in result.output
    assert "YIAN_DEFAULT_MODEL" not in result.output


def test_default_switches_without_sign_in(setup_auth):
    store, service = setup_auth
    _save_each_kind(store)
    for provider in ("openai", "ollama", "azure", "anthropic"):
        result = runner.invoke(app, ["auth", provider, "--default"])
        assert result.exit_code == 0, result.output
        assert store.read().default_provider == provider
        assert f"Default provider: {provider}" in result.output
    assert service.calls == []

    result = runner.invoke(app, ["auth", "google", "--default"])
    assert result.exit_code == 1 and "yian auth" in result.output
    result = runner.invoke(app, ["auth", "openai", "--default", "--model", "x"])
    assert result.exit_code == 1
    assert store.read().default_provider == "anthropic"


def test_remove_deletes_one_login(monkeypatch, setup_auth):
    store, service = setup_auth
    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER")
    monkeypatch.delenv("YIAN_DEFAULT_MODEL")
    _save_each_kind(store)
    store.set_default("openai")

    result = runner.invoke(app, ["auth", "anthropic", "--remove"])
    assert result.exit_code == 0, result.output
    assert "Removed the saved anthropic login" in result.output
    state = store.read()
    assert state.default_provider == "openai"
    assert {item.provider for item in state.selections} == {"openai", "ollama", "azure"}

    result = runner.invoke(app, ["auth", "openai", "--remove"])
    assert result.exit_code == 0, result.output
    assert "No default provider is saved" in result.output
    assert store.read().default_provider is None
    result = runner.invoke(app, ["auth"])
    assert result.exit_code == 0, result.output
    assert "No default provider is saved" in result.output
    assert "yian auth <provider> --default" in result.output

    for provider in ("ollama", "azure"):
        result = runner.invoke(app, ["auth", provider, "--remove"])
        assert result.exit_code == 0, result.output
    assert store.read().selections == ()
    result = runner.invoke(app, ["auth", "anthropic", "--remove"])
    assert result.exit_code == 1 and "yian auth" in result.output
    result = runner.invoke(app, ["auth", "anthropic", "--remove", "--default"])
    assert result.exit_code == 1
    assert service.calls == []


def test_remove_accepts_an_unsupported_saved_provider(setup_auth):
    store, service = setup_auth
    store.path.parent.mkdir(mode=0o700, parents=True)
    store.path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_provider": None,
                "providers": {
                    "legacy-x": {
                        "method": "api_key",
                        "model": "m",
                        "credential_id": "abcd",
                        "credentials": {"api_key": "k"},
                    }
                },
            }
        )
    )
    store.path.chmod(0o600)
    result = runner.invoke(app, ["auth", "legacy-x", "--remove"])
    assert result.exit_code == 0, result.output
    assert store.read().selections == ()


@pytest.mark.parametrize(
    "args",
    [
        ["auth", "codex"],
        ["auth", "codex", "--default"],
        ["auth", "anthropic", "--method", "api-key"],
        ["auth", "anthropic", "--login", "device"],
        ["auth", "openai", "--endpoint", "https://x.openai.azure.com/"],
        ["auth", "ollama", "--api-key-env", "X"],
        ["auth", "openai", "--remove", "--model", "x"],
        ["auth", "azure", "--default", "--endpoint", "https://x.openai.azure.com/"],
    ],
)
def test_option_combinations_fail_before_any_request(monkeypatch, setup_auth, args):
    store, service = setup_auth

    def unread(*args, **kwargs):
        raise AssertionError("the store was opened")

    monkeypatch.setattr(auth_cli, "AuthStore", unread)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert type(result.exception) is SystemExit, result.exception
    assert service.calls == []
    if args[1] == "codex":
        assert "anthropic" in result.output


def test_setup_keeps_existing_default(setup_auth):
    store, service = setup_auth
    args = ["auth", "openai", "--method", "api-key", "--model", "gpt-test"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert store.read().default_provider == "openai"
    assert "Default provider: openai" in result.output

    store.save("anthropic", ApiKeyCredentials("private-other"), "claude-test")
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert store.read().default_provider == "anthropic"
    assert store.selection("openai").model == "gpt-test"
    assert "Default provider: anthropic" in result.output
    assert "yian auth openai --default" in result.output
