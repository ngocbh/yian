"""`LLMConfig.organization`, `.project` and `.headers`: connection settings bound to the key."""

import pytest
from pydantic import SecretStr, ValidationError

from yian.llm_clients.config import LLMConfig
from yian.llm_clients.connection_settings import (
    OPENAI_REFUSED_HEADER_NAMES,
    REFUSED_HEADER_NAMES,
    validate_header_name,
    validate_header_value,
    validate_headers,
    validate_organization,
)

KEY = "sk-sentinel-7d1f3b20"
GATEWAY = "gw-secret-91c4e2"
CREDENTIAL_NAMES = ["Authorization", "Proxy-Authorization", "x-api-key", "api-key",
                    "x-goog-api-key"]
TRANSPORT_NAMES = ["Host", "Content-Length", "Transfer-Encoding", "Connection", "Content-Type"]


def _config(**fields):
    return LLMConfig(**{"provider": "openai", "api_key": KEY, **fields})


def _refused(message_part, **fields):
    """Assert the config is refused with `message_part` and never echoes a value."""
    with pytest.raises(ValidationError) as caught:
        LLMConfig(**fields)
    text = str(caught.value)
    assert message_part in text
    assert KEY not in text and GATEWAY not in text
    return text


def test_valid_settings_are_kept():
    config = _config(
        organization="org-abc123",
        project="proj_XYZ",
        headers={"X-Gateway": GATEWAY, "X-Trace": "a b\tc"},
    )
    assert config.organization == "org-abc123"
    assert config.project == "proj_XYZ"
    assert set(config.headers) == {"X-Gateway", "X-Trace"}
    assert all(isinstance(value, SecretStr) for value in config.headers.values())
    assert config.headers["X-Gateway"].get_secret_value() == GATEWAY


def test_header_values_stay_out_of_dumps_and_repr():
    config = _config(headers={"X-Gateway": GATEWAY})
    for text in (repr(config), str(config), config.model_dump_json()):
        assert GATEWAY not in text
    assert "headers" not in config.model_dump()
    assert "headers" not in config.model_dump(mode="json")


def test_organization_and_project_dump_when_set_and_add_nothing_when_unset():
    config = _config(model="gpt-5.5", organization="org-1", project="proj-1")
    assert config.model_dump() == {
        "provider": "openai", "model": "gpt-5.5", "organization": "org-1", "project": "proj-1",
    }
    # Recorded snapshots compare full dumps, so unset settings must add nothing.
    assert LLMConfig(provider="openai").model_dump() == {"provider": "openai", "model": None}
    assert LLMConfig(provider="openai").model_dump(mode="json") == {
        "provider": "openai", "model": None,
    }


def test_an_empty_header_mapping_is_no_headers():
    assert LLMConfig(provider="openai", headers={}).headers is None


@pytest.mark.parametrize("provider", ["openai", "deepseek", "anthropic", "google", "azure"])
@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
@pytest.mark.parametrize("spell", [str, str.lower, str.upper])
def test_credential_header_names_are_refused_for_every_provider(provider, name, spell):
    _refused(
        "yian sends the credential itself",
        provider=provider, api_key=KEY, headers={spell(name): GATEWAY},
    )


@pytest.mark.parametrize("name", TRANSPORT_NAMES)
def test_transport_header_names_are_refused(name):
    _refused(
        "yian's client sets it",
        provider="deepseek", api_key=KEY, headers={name.lower(): GATEWAY},
    )


@pytest.mark.parametrize("name", ["OpenAI-Organization", "openai-project"])
def test_openai_id_headers_are_refused_for_openai_only(name):
    _refused(
        "use llm_config.organization / project",
        provider="openai", api_key=KEY, headers={name: GATEWAY},
    )
    config = LLMConfig(provider="deepseek", api_key=KEY, headers={name: "proj-1"})
    assert list(config.headers) == [name]


def test_duplicate_names_by_case_are_refused():
    _refused(
        "more than once",
        provider="openai", api_key=KEY, headers={"X-Gateway": GATEWAY, "x-gateway": GATEWAY},
    )


@pytest.mark.parametrize("name", ["", "X Gateway", "X:Gateway", "X-Gätway", "X\nY", "(x)"])
def test_names_that_are_not_tokens_are_refused(name):
    text = _refused("HTTP token", provider="openai", api_key=KEY, headers={name: GATEWAY})
    if name:
        assert name not in text


@pytest.mark.parametrize(
    "value",
    [
        f"{GATEWAY}\r",
        f"{GATEWAY}\nX-Evil: 1",
        f"{GATEWAY}\x00",
        f"{GATEWAY}\x01",
        f"{GATEWAY}\x7f",
        f"{GATEWAY}é",
        f" {GATEWAY}",
        f"{GATEWAY} ",
        f"{GATEWAY}\t",
        "",
    ],
)
def test_bad_header_values_are_refused(value):
    _refused("visible ASCII", provider="openai", api_key=KEY, headers={"X-Gateway": value})


def test_header_value_and_count_caps(set_defaults):
    set_defaults(MAX_HEADER_VALUE_CHARS=10, MAX_CONNECTION_HEADERS=2)
    assert _config(headers={"A": "x" * 10, "B": "y"}).headers["A"].get_secret_value() == "x" * 10
    _refused("at most 10 characters", provider="openai", api_key=KEY, headers={"A": "x" * 11})
    _refused(
        "at most 2 headers",
        provider="openai", api_key=KEY, headers={"A": "1", "B": "2", "C": "3"},
    )


@pytest.mark.parametrize("field", ["organization", "project"])
def test_organization_and_project_values(field, set_defaults):
    set_defaults(MAX_ORGANIZATION_CHARS=8)
    assert getattr(_config(**{field: "org-1234"}), field) == "org-1234"
    secretish = "org-45678"
    text = _refused("at most 8 characters", provider="openai", api_key=KEY, **{field: secretish})
    assert secretish not in text
    for bad in ("", "org 1", "org\n1", "orgé", "org\x7f"):
        _refused("visible ASCII", provider="openai", api_key=KEY, **{field: bad})


@pytest.mark.parametrize("provider", ["deepseek", "azure", "anthropic", "vllm"])
@pytest.mark.parametrize("field", ["organization", "project"])
def test_organization_and_project_are_openai_only(provider, field):
    _refused("only to provider openai", provider=provider, api_key=KEY, **{field: "org-1"})


def test_headers_are_refused_for_ollama():
    _refused(
        "provider ollama does not take llm_config.headers",
        provider="ollama", api_key=KEY, headers={"X-Gateway": GATEWAY},
    )


@pytest.mark.parametrize(
    "fields",
    [{"organization": "org-1"}, {"project": "proj-1"}, {"headers": {"X-Gateway": GATEWAY}}],
    ids=["organization", "project", "headers"],
)
@pytest.mark.parametrize("missing", ["api_key", "provider"])
def test_each_setting_needs_key_and_provider_in_the_same_config(fields, missing):
    given = {"provider": "openai", "api_key": KEY, **fields}
    del given[missing]
    _refused("needs llm_config.api_key and llm_config.provider", **given)


def test_azure_settings_need_the_key_too():
    _refused(
        "needs llm_config.api_key",
        provider="azure", base_url="https://res.openai.azure.com/", headers={"X-G": GATEWAY},
    )


def test_refused_name_tables_are_lowercase():
    assert set(REFUSED_HEADER_NAMES) == {name.lower() for name in
                                        CREDENTIAL_NAMES + TRANSPORT_NAMES}
    assert set(OPENAI_REFUSED_HEADER_NAMES) == {"openai-organization", "openai-project"}


def test_module_functions_return_valid_input_and_raise_value_error():
    assert validate_header_name("X-Gateway", "openai") == "X-Gateway"
    assert validate_header_name("OpenAI-Project", "deepseek") == "OpenAI-Project"
    assert validate_header_value("X-Gateway", SecretStr(GATEWAY)) == GATEWAY
    assert validate_headers({"X-A": SecretStr("1"), "X-B": "2"}, "openai") == {
        "X-A": "1", "X-B": "2",
    }
    assert validate_organization("org-1", "project") == "org-1"
    for call in (
        lambda: validate_header_name("OpenAI-Project", "openai"),
        lambda: validate_header_name(3, None),
        lambda: validate_header_value("X-Gateway", 3),
        lambda: validate_headers(["X-A"], None),
        lambda: validate_organization(None, "organization"),
    ):
        with pytest.raises(ValueError):
            call()
