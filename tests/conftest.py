"""Suite-wide model defaults and an empty auth store."""

import pytest


@pytest.fixture(autouse=True)
def explicit_llm_defaults(monkeypatch, tmp_path_factory):
    """Point every test at an empty private store and an explicit environment default.

    yian has no built-in model, so tests that build agents without one need a layer
    that sets it. Tests about defaults clear these variables themselves.
    """
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path_factory.mktemp("auth") / "auth.json"))
    monkeypatch.setenv("YIAN_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "gpt-5.5")
