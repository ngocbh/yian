"""Suite-wide model defaults, an empty auth store, built-in yian defaults, a restored environment."""

import dataclasses
import os

import pytest

from yian import defaults


# First, so its snapshot comes before the other fixtures touch the environment.
@pytest.fixture(autouse=True)
def restored_environment():
    """Put `os.environ` back after each test.

    yian's CLI helpers export a typed key with `os.environ[...] = ...`, which no monkeypatch
    undoes when the variable was absent; a leaked variable changes later tests, and the
    conformance harvest would record it in their cases.
    """
    saved = dict(os.environ)
    yield
    for name in set(os.environ) - set(saved):
        del os.environ[name]
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def explicit_llm_defaults(monkeypatch, tmp_path_factory):
    """Point every test at an empty private store and an explicit environment default.

    yian has no built-in model, so tests that build agents without one need a layer
    that sets it. Tests about defaults clear these variables themselves.
    """
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path_factory.mktemp("auth") / "auth.json"))
    monkeypatch.setenv("YIAN_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "gpt-5.5")


@pytest.fixture(autouse=True)
def builtin_yian_defaults(monkeypatch):
    """Start every test from built-in defaults, whatever the shell or an earlier test set."""
    for entry in defaults.ENTRIES:
        monkeypatch.delenv(entry.variable, raising=False)
    defaults._reload()
    yield
    defaults._forget()


class _SetDefaults:
    """Set defaults for one test: by variable, or unchecked by attribute."""

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def __call__(self, **values):
        """Set `YIAN_<NAME>` for each entry name given and re-read the environment."""
        names = {entry.name for entry in defaults.ENTRIES}
        for name, value in values.items():
            if name not in names:
                raise KeyError(f"{name} is not a yian default")
            self._monkeypatch.setenv("YIAN_" + name, str(value))
        return defaults._reload()

    def unchecked(self, **values):
        """Replace attributes of the current values with no range check.

        A later `set_defaults(...)` call re-reads the environment and discards them.
        """
        replaced = dataclasses.replace(defaults.current(), **values)
        defaults._install(replaced)
        return replaced


@pytest.fixture
def set_defaults(monkeypatch):
    """Override yian defaults for one test.

    `set_defaults(MAX_VALUE_NESTING=3)` sets the variable and re-reads it, so the range
    check applies; `set_defaults.unchecked(max_value_nesting=200_000)` installs a value
    outside any range.
    """
    return _SetDefaults(monkeypatch)
