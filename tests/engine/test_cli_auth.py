"""Provider setup activates one verified method/model pair at the final step."""

import importlib

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
