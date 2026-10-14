"""Provider setup activates one verified method/model pair at the final step."""

import importlib
import json
from types import SimpleNamespace
import warnings

from langchain_core.messages import AIMessage
import pytest
from typer.testing import CliRunner

from yian.cli import app
from yian.llm_clients.auth import (
    ApiKeyCredentials,
    AuthError,
    AuthStore,
    OAuthCredentials,
)
from yian.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from yian.llm_clients.model_list import ModelListError

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

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Default provider: openai" in result.output
    assert "switch with" not in result.output

    store.save("anthropic", ApiKeyCredentials("private-other"), "claude-test")
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert store.read().default_provider == "anthropic"
    assert store.selection("openai").model == "gpt-test"
    assert "Default provider: anthropic" in result.output
    assert "yian auth openai --default" in result.output


class Checks:
    """Fake model listing and one-reply check that record their calls."""

    def __init__(self):
        self.models = ("m1", "m2")
        self.listings = []
        self.answers = []
        self.problem = None

    def list_models(self, provider, api_key=None, base_url=None):
        self.listings.append((provider, api_key, base_url))
        if isinstance(self.models, Exception):
            raise self.models
        return self.models

    def model_answers(self, provider, model, key, endpoint):
        self.answers.append((provider, model, key, endpoint))
        return self.problem


@pytest.fixture
def checks(monkeypatch, setup_auth):
    fake = Checks()
    monkeypatch.setattr(auth_cli, "list_models", fake.list_models)
    monkeypatch.setattr(auth_cli, "_model_answers", fake.model_answers)
    variables = ("OLLAMA_BASE_URL", "VLLM_BASE_URL", "VLLM_API_KEY", "OPENAI_API_VERSION")
    for name in variables:
        monkeypatch.delenv(name, raising=False)
    return setup_auth[0], fake


@pytest.mark.parametrize(
    "provider",
    sorted(set(PROVIDER_API_KEY_ENV) - {"openai", "azure", "ollama", "vllm"}),
)
def test_api_key_provider_setup_saves_key_and_model(monkeypatch, checks, provider):
    store, fake = checks
    monkeypatch.setenv(PROVIDER_API_KEY_ENV[provider], "private-key")
    result = runner.invoke(app, ["auth", provider, "--model", "m1"])
    assert result.exit_code == 0, result.output
    selected = store.selection(provider)
    assert selected.method == "api_key" and selected.model == "m1"
    assert store.credentials(provider, selected.credential_id).api_key == "private-key"
    assert store.read().default_provider == provider
    assert f"Saved {provider} login (api-key) with model m1" in result.output
    assert f"Default provider: {provider}" in result.output
    assert "switch with" not in result.output
    assert "private-" not in result.output
    assert fake.listings == [(provider, "private-key", None)]
    assert fake.answers == []


def test_api_key_setup_keeps_another_default(monkeypatch, checks):
    store, fake = checks
    store.save("openai", ApiKeyCredentials("private-other"), "gpt-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-key")
    result = runner.invoke(app, ["auth", "anthropic", "--model", "m2"])
    assert result.exit_code == 0, result.output
    assert store.selection("anthropic").model == "m2"
    assert store.read().default_provider == "openai"
    assert (
        "Default provider: openai (switch with 'yian auth anthropic --default')"
        in result.output
    )


def test_api_key_env_option_names_the_variable(monkeypatch, checks):
    store, fake = checks
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["auth", "anthropic", "--model", "m1"])
    assert result.exit_code == 1 and "ANTHROPIC_API_KEY" in result.output
    assert fake.listings == [] and not store.path.exists()

    monkeypatch.setenv("MY_KEY", "private-key")
    args = ["auth", "anthropic", "--model", "m1", "--api-key-env", "MY_KEY"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert fake.listings == [("anthropic", "private-key", None)]


def test_unlisted_model_is_checked_with_one_reply(monkeypatch, checks):
    store, fake = checks
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-key")
    fake.models = ("claude-x-20250101",)
    result = runner.invoke(app, ["auth", "anthropic", "--model", "claude-x"])
    assert result.exit_code == 0, result.output
    assert fake.answers == [("anthropic", "claude-x", "private-key", None)]
    assert store.selection("anthropic").model == "claude-x"

    before = store.path.read_bytes()
    fake.problem = "NotFoundError: model not found"
    result = runner.invoke(app, ["auth", "anthropic", "--model", "claude-y"])
    assert result.exit_code == 1
    assert "didn't answer" in result.output and "NotFoundError" in result.output
    assert store.path.read_bytes() == before


def test_listing_failure_falls_back_to_one_reply(monkeypatch, checks):
    store, fake = checks
    previous = store.save("anthropic", ApiKeyCredentials("old-key"), "claude-old")
    before = store.path.read_bytes()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-key")
    fake.models = ModelListError("HTTP 401")
    fake.problem = "AuthenticationError: invalid key"
    result = runner.invoke(app, ["auth", "anthropic", "--model", "claude-x"])
    assert result.exit_code == 1
    assert fake.answers == [("anthropic", "claude-x", "private-key", None)]
    assert store.path.read_bytes() == before
    assert store.selection() == previous and store.read().default_provider == "anthropic"
    assert "private-" not in result.output and "old-key" not in result.output


def test_listing_failure_on_a_terminal_asks_for_a_model_name(monkeypatch, checks):
    store, fake = checks
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-key")
    fake.models = ModelListError("can't reach host")
    asked = []

    def ask(message, *, option):
        asked.append(option)
        return "claude-x"

    monkeypatch.setattr(auth_cli, "_require_terminal", lambda option: None)
    monkeypatch.setattr(auth_cli, "_ask", ask)
    result = runner.invoke(app, ["auth", "anthropic"])
    assert result.exit_code == 0, result.output
    assert "can't reach host" in result.output and asked == ["--model"]
    assert fake.answers == [("anthropic", "claude-x", "private-key", None)]
    assert store.selection("anthropic").model == "claude-x"


def test_ollama_setup_saves_model_only(monkeypatch, checks):
    store, fake = checks
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999")
    fake.models = ("llama3",)
    result = runner.invoke(app, ["auth", "ollama", "--model", "llama3"])
    assert result.exit_code == 0, result.output
    assert fake.listings == [("ollama", None, "http://localhost:9999")]
    assert fake.answers == []
    selected = store.selection("ollama")
    assert selected.method == "none" and selected.model == "llama3"
    assert "Saved ollama login (none) with model llama3" in result.output


def test_vllm_key_is_optional(monkeypatch, checks):
    store, fake = checks
    result = runner.invoke(app, ["auth", "vllm", "--model", "m1"])
    assert result.exit_code == 0, result.output
    assert fake.listings == [("vllm", None, None)]
    assert store.selection("vllm").method == "none"

    monkeypatch.setenv("VLLM_API_KEY", "private-key")
    monkeypatch.setenv("VLLM_BASE_URL", "https://vllm.example.com/v1")
    result = runner.invoke(app, ["auth", "vllm", "--model", "m1"])
    assert result.exit_code == 0, result.output
    assert fake.listings[-1] == ("vllm", "private-key", "https://vllm.example.com/v1")
    assert store.selection("vllm").method == "api_key"
    assert "private-" not in result.output


def test_vllm_named_key_variable_is_required(monkeypatch, checks):
    store, fake = checks
    monkeypatch.delenv("MY_KEY", raising=False)
    args = ["auth", "vllm", "--model", "m1", "--api-key-env", "MY_KEY"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "MY_KEY" in result.output
    assert fake.listings == [] and not store.path.exists()


class Tty:
    def isatty(self):
        return True


@pytest.fixture
def terminal(monkeypatch):
    """Run the CLI as if on a terminal and answer the hidden key prompt."""
    import questionary

    prompts = []
    answer = SimpleNamespace(value="")

    def password(message):
        prompts.append(message)
        return SimpleNamespace(ask=lambda: answer.value)

    # CliRunner swaps sys.stdin during invoke, so the module's sys is replaced.
    monkeypatch.setattr(auth_cli, "sys", SimpleNamespace(stdin=Tty()))
    monkeypatch.setattr(questionary, "password", password)
    return prompts, answer


def test_hidden_prompt_reads_the_key(monkeypatch, checks, terminal):
    store, fake = checks
    prompts, answer = terminal
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answer.value = "private-key"
    result = runner.invoke(app, ["auth", "anthropic", "--model", "m1"])
    assert result.exit_code == 0, result.output
    assert prompts == ["anthropic API key:"]
    assert fake.listings == [("anthropic", "private-key", None)]
    assert store.selection("anthropic").method == "api_key"

    answer.value = None
    result = runner.invoke(app, ["auth", "anthropic", "--model", "m2"])
    assert result.exit_code == 1
    assert "the previous login is unchanged" in result.output
    assert store.selection("anthropic").model == "m1"


def test_hidden_prompt_empty_answer_means_no_vllm_key(monkeypatch, checks, terminal):
    store, fake = checks
    prompts, answer = terminal
    result = runner.invoke(app, ["auth", "vllm", "--model", "m1"])
    assert result.exit_code == 0, result.output
    assert prompts == ["vllm API key (leave empty for none):"]
    assert store.selection("vllm").method == "none"

    monkeypatch.delenv("MY_KEY", raising=False)
    args = ["auth", "vllm", "--model", "m2", "--api-key-env", "MY_KEY"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "no API key was supplied" in result.output
    assert prompts[-1] == "vllm API key:"
    assert store.selection("vllm").model == "m1"


def test_noninteractive_setup_requires_model(monkeypatch, checks):
    store, fake = checks
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-key")
    for provider in ("anthropic", "ollama"):
        result = runner.invoke(app, ["auth", provider])
        assert result.exit_code == 1 and "--model" in result.output
    assert fake.listings == [] and not store.path.exists()


def test_azure_setup_saves_endpoint_key_and_deployment(monkeypatch, checks):
    store, fake = checks
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "private-key")
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    endpoint = "https://r.openai.azure.com/"
    result = runner.invoke(app, ["auth", "azure", "--endpoint", endpoint, "--model", "dep"])
    assert result.exit_code == 0, result.output
    assert fake.listings == []
    assert fake.answers == [("azure", "dep", "private-key", endpoint)]
    selected = store.selection("azure")
    assert selected.endpoint == endpoint and selected.model == "dep"
    assert selected.method == "api_key"
    assert "private-" not in result.output

    before = store.path.read_bytes()
    result = runner.invoke(app, ["auth", "azure", "--model", "dep"])
    assert result.exit_code == 1 and "--endpoint" in result.output
    result = runner.invoke(app, ["auth", "azure", "--endpoint", endpoint])
    assert result.exit_code == 1 and "--model" in result.output
    result = runner.invoke(
        app, ["auth", "azure", "--endpoint", "http://r.example.com", "--model", "dep"]
    )
    assert result.exit_code == 1 and "endpoint" in result.output
    assert len(fake.answers) == 1 and fake.listings == []
    assert store.path.read_bytes() == before


def test_azure_setup_requires_the_api_version(monkeypatch, checks):
    store, fake = checks
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    endpoint = "https://r.openai.azure.com/"
    result = runner.invoke(app, ["auth", "azure", "--endpoint", endpoint, "--model", "dep"])
    assert result.exit_code == 1 and "OPENAI_API_VERSION" in result.output
    assert "AZURE_OPENAI_API_KEY" not in result.output
    assert fake.answers == [] and fake.listings == [] and not store.path.exists()


class StubModel:
    def __init__(self, failure=None):
        self.prompts = []
        self.failure = failure

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.failure:
            raise self.failure
        return AIMessage("OK")


def test_model_answers_builds_and_invokes_once(monkeypatch):
    built = []
    model = StubModel()

    class Client:
        def get_llm(self):
            return model

    def create(provider, name, base_url=None, **kwargs):
        built.append((provider, name, base_url, kwargs))
        return Client()

    monkeypatch.setattr(auth_cli, "create_llm_client", create)
    assert auth_cli._model_answers("azure", "dep", "private-key", "https://r/") is None
    provider, name, base_url, kwargs = built[0]
    assert (provider, name, base_url) == ("azure", "dep", "https://r/")
    assert kwargs["api_key"] == "private-key"
    assert kwargs["timeout"] == 30 and kwargs["max_retries"] == 0
    assert len(model.prompts) == 1

    model.failure = RuntimeError("bad key private-key")
    problem = auth_cli._model_answers("anthropic", "claude-x", "private-key", None)
    assert "RuntimeError" in problem and "private-key" not in problem

    class Echo(Exception):
        def __str__(self):
            return "rejected\n  input private-key"

    model.failure = Echo()
    problem = auth_cli._model_answers("anthropic", "claude-x", "private-key", None)
    assert problem == "Echo: rejected input [REDACTED]"

    model.failure = None
    # A keyless vLLM login is checked with the client's dummy key, not VLLM_API_KEY.
    monkeypatch.setenv("VLLM_API_KEY", "private-ambient")
    assert auth_cli._model_answers("vllm", "m1", None, None) is None
    assert built[-1][3]["api_key"] == "EMPTY"
    assert auth_cli._model_answers("ollama", "m1", None, None) is None
    assert "api_key" not in built[-1][3]


def test_model_answers_mutes_the_unknown_model_warning(monkeypatch):
    class Client:
        def get_llm(self):
            warnings.warn(
                "Model 'claude-x' is not in the known model list for provider "
                "'anthropic'. Continuing anyway.",
                RuntimeWarning,
            )
            return StubModel()

    monkeypatch.setattr(auth_cli, "create_llm_client", lambda *a, **k: Client())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem = auth_cli._model_answers("anthropic", "claude-x", "private-key", None)
    assert problem is None and caught == []


def test_model_answers_reports_a_missing_extra(monkeypatch):
    def create(*args, **kwargs):
        raise ImportError("Install it with:  pip install 'yian[anthropic]'")

    monkeypatch.setattr(auth_cli, "create_llm_client", create)
    problem = auth_cli._model_answers("anthropic", "claude-x", "private-key", None)
    assert "pip install" in problem


def test_env_override_note_agrees_with_the_number_of_variables(monkeypatch, setup_auth):
    result = runner.invoke(app, ["auth"])
    assert "YIAN_DEFAULT_PROVIDER and YIAN_DEFAULT_MODEL are set" in result.output
    assert "and override the saved choice" in result.output
    monkeypatch.delenv("YIAN_DEFAULT_MODEL")
    result = runner.invoke(app, ["auth"])
    assert "YIAN_DEFAULT_PROVIDER is set" in result.output
    assert "and overrides the saved choice" in result.output
