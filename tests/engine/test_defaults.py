"""Every predefined default and limit, read once from `YIAN_<NAME>`."""

from __future__ import annotations

import dataclasses
import re
import sys

import pytest

from yian import defaults
from yian.defaults import (
    ENTRIES,
    Defaults,
    DefaultsError,
    DefaultsMismatchError,
    check_recorded,
    current,
    read,
)


def _entry(name: str) -> defaults.Entry:
    return next(entry for entry in ENTRIES if entry.name == name)


def test_entries_match_defaults_fields() -> None:
    names = [entry.name for entry in ENTRIES]
    assert len(names) == len(set(names))
    fields = {field.name: field.default for field in dataclasses.fields(Defaults)}
    assert set(fields) == {name.lower() for name in names}
    for entry in ENTRIES:
        assert fields[entry.name.lower()] == entry.default, entry.name
        assert entry.group in {"limit", "setting", "input cap"}, entry.name
        assert entry.description and "\n" not in entry.description, entry.name
        if entry.default == entry.no_limit:
            continue
        assert entry.default >= entry.minimum, entry.name
        if entry.maximum is not None:
            assert entry.default <= entry.maximum, entry.name


def test_variable_names_are_prefixed_and_do_not_collide() -> None:
    taken = {"YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL", "YIAN_AUTH_FILE"}
    owners = {"MAX", "AGENT", "EPISODIC", "REFLECTIVE", "SESSION", "PROFILE"}
    for entry in ENTRIES:
        assert re.fullmatch(r"[A-Z]+(_[A-Z]+)+", entry.name), entry.name
        assert entry.name.split("_")[0] in owners, entry.name
        assert entry.variable == "YIAN_" + entry.name
        assert entry.variable not in taken
        assert not entry.name.startswith("DEFAULT_")


def test_read_with_no_variables_is_builtin() -> None:
    assert read({}) == Defaults()
    assert read({}).non_defaults() == {}


def test_override_takes_effect() -> None:
    assert read({"YIAN_MAX_VALUE_NESTING": "80"}).max_value_nesting == 80


def test_empty_value_counts_as_unset() -> None:
    assert read({"YIAN_MAX_VALUE_NESTING": ""}) == Defaults()


@pytest.mark.parametrize(
    "value",
    ["abc", "1.5", "1e3", " 5", "1_000", "+5", "-0", "007", "0", "101", "9" * 5000],
)
def test_bad_value_names_the_variable(value: str) -> None:
    with pytest.raises(DefaultsError) as caught:
        read({"YIAN_MAX_VALUE_NESTING": value})
    message = str(caught.value)
    assert "YIAN_MAX_VALUE_NESTING" in message
    assert value[:30] in message
    assert "=" + repr(value if len(value) <= 40 else value[:37] + "...") in message
    assert "from 1 to 100" in message
    assert len(message) < 300
    assert "Exceeds the limit" not in message
    assert isinstance(caught.value, ValueError)


def _accepts(entry: defaults.Entry, value: int) -> bool:
    try:
        read({entry.variable: str(value)})
    except DefaultsError:
        return False
    return True


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_each_minimum_and_maximum(entry: defaults.Entry) -> None:
    assert _accepts(entry, entry.minimum)
    if entry.minimum - 1 != entry.no_limit:
        assert not _accepts(entry, entry.minimum - 1)
    if entry.maximum is not None:
        assert _accepts(entry, entry.maximum)
        assert not _accepts(entry, entry.maximum + 1)
    elif not entry.maximum_from_interpreter:
        assert _accepts(entry, 10**18 - 1)
        assert not _accepts(entry, 10**18)


def test_an_unbounded_entry_stops_at_18_digits() -> None:
    entry = _entry("MAX_TOTAL_NODES")
    assert read({entry.variable: str(10**18 - 1)}).max_total_nodes == 10**18 - 1
    with pytest.raises(DefaultsError) as caught:
        read({entry.variable: str(10**18)})
    assert f"from 1 to {10**18 - 1}" in str(caught.value)


def test_reflective_budgets_have_a_floor_of_256() -> None:
    for name in (
        "REFLECTIVE_MAX_LESSON_CHARS",
        "REFLECTIVE_MAX_EPISODE_CHARS",
        "REFLECTIVE_MAX_LIMIT_CHARS",
    ):
        assert _entry(name).minimum == 256


def test_no_limit_sentinel() -> None:
    entry = _entry("AGENT_MAX_TOOL_ITERATIONS")
    assert _accepts(entry, -1)
    assert _accepts(entry, 5)
    assert not _accepts(entry, 0)
    assert not _accepts(entry, -2)
    with pytest.raises(DefaultsError, match="-1"):
        read({entry.variable: "0"})


def test_int_digits_cannot_exceed_the_interpreter_limit() -> None:
    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if not limit:
        pytest.skip("this interpreter has no int digit limit")
    entry = _entry("MAX_INT_DIGITS")
    assert entry.maximum_from_interpreter
    assert read({entry.variable: str(limit)}).max_int_digits == limit
    with pytest.raises(DefaultsError, match="YIAN_MAX_INT_DIGITS") as caught:
        read({entry.variable: str(limit + 1)})
    assert f"from 1 to {limit}" in str(caught.value)


def test_int_digits_has_no_upper_bound_without_an_interpreter_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "get_int_max_str_digits", lambda: 0, raising=False)
    assert read({"YIAN_MAX_INT_DIGITS": "100000"}).max_int_digits == 100_000


def test_int_digits_has_no_upper_bound_on_an_interpreter_without_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(sys, "get_int_max_str_digits", raising=False)
    assert read({"YIAN_MAX_INT_DIGITS": "100000"}).max_int_digits == 100_000


def test_memory_json_nesting_follows_value_nesting() -> None:
    assert Defaults().memory_json_nesting == 128
    assert read({"YIAN_MAX_VALUE_NESTING": "80"}).memory_json_nesting == 108


def test_non_defaults_lists_only_changed_values_sorted() -> None:
    values = read(
        {
            "YIAN_MAX_VALUE_NESTING": "80",
            "YIAN_AGENT_RETRIES": "5",
            "YIAN_MAX_TOTAL_NODES": "10000",
        }
    )
    changed = values.non_defaults()
    assert changed == {"AGENT_RETRIES": 5, "MAX_VALUE_NESTING": 80}
    assert list(changed) == sorted(changed)


def test_from_recorded_overlays_builtins() -> None:
    values = Defaults.from_recorded({"MAX_TOTAL_NODES": 5000})
    assert values == dataclasses.replace(Defaults(), max_total_nodes=5000)
    assert Defaults.from_recorded({}) == Defaults()
    with pytest.raises(DefaultsMismatchError) as caught:
        Defaults.from_recorded({"NOT_AN_ENTRY": 1})
    assert str(caught.value) == (
        "this run started with YIAN_NOT_AN_ENTRY, which this version of yian does not know"
    )


def test_current_reads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    assert current() == Defaults()
    monkeypatch.setenv("YIAN_MAX_VALUE_NESTING", "80")
    assert current().max_value_nesting == 100
    assert defaults._reload().max_value_nesting == 80
    assert current().max_value_nesting == 80


def test_failed_read_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YIAN_MAX_VALUE_NESTING", "abc")
    defaults._forget()
    for _ in range(2):
        with pytest.raises(DefaultsError, match="YIAN_MAX_VALUE_NESTING"):
            current()
    monkeypatch.delenv("YIAN_MAX_VALUE_NESTING")
    assert current() == Defaults()


def test_install_skips_the_range_check() -> None:
    defaults._install(dataclasses.replace(Defaults(), max_value_nesting=200_000))
    assert current().max_value_nesting == 200_000


def test_set_defaults_fixture(set_defaults) -> None:
    set_defaults(MAX_VALUE_NESTING=3)
    assert current().max_value_nesting == 3
    set_defaults.unchecked(max_value_nesting=200_000)
    assert current().max_value_nesting == 200_000
    assert current().max_total_nodes == 10_000


def test_check_recorded(set_defaults) -> None:
    check_recorded({})
    check_recorded({"MAX_TOTAL_NODES": 10_000})

    set_defaults(MAX_TOTAL_NODES=20_000)
    check_recorded({"MAX_TOTAL_NODES": 20_000})
    with pytest.raises(DefaultsMismatchError) as caught:
        check_recorded({"MAX_TOTAL_NODES": 5000})
    assert isinstance(caught.value, ValueError)
    message = str(caught.value)
    assert "YIAN_MAX_TOTAL_NODES is 20000 here" in message
    assert "started with 5000" in message
    assert "set YIAN_MAX_TOTAL_NODES=5000" in message

    with pytest.raises(DefaultsMismatchError) as caught:
        check_recorded({})
    message = str(caught.value)
    assert "started with the default 10000" in message
    assert "unset YIAN_MAX_TOTAL_NODES" in message

    set_defaults(MAX_TOTAL_NODES=20_000, AGENT_RETRIES=4)
    with pytest.raises(DefaultsMismatchError) as caught:
        check_recorded({})
    message = str(caught.value)
    assert "YIAN_MAX_TOTAL_NODES" in message
    assert "YIAN_AGENT_RETRIES" in message

    with pytest.raises(DefaultsMismatchError, match="YIAN_NOT_AN_ENTRY"):
        check_recorded({"NOT_AN_ENTRY": 1})


def test_check_recorded_reports_unknown_names_and_differences_together(set_defaults) -> None:
    set_defaults(MAX_TOTAL_NODES=20_000)

    with pytest.raises(DefaultsMismatchError) as caught:
        check_recorded({"NOT_AN_ENTRY": 1, "AGENT_RETRIES": 0})

    lines = str(caught.value).splitlines()
    assert lines[0] == (
        "this run started with YIAN_NOT_AN_ENTRY, which this version of yian does not know"
    )
    assert any("YIAN_AGENT_RETRIES is 2 here" in line for line in lines[1:])
    assert any("unset YIAN_MAX_TOTAL_NODES" in line for line in lines[1:])
    assert "resumed here" not in str(caught.value)


def test_check_recorded_reports_an_out_of_range_value_as_invalid() -> None:
    with pytest.raises(DefaultsMismatchError) as caught:
        check_recorded({"MAX_VALUE_NESTING": 150})

    message = str(caught.value)
    assert message == (
        "this run recorded an invalid value YIAN_MAX_VALUE_NESTING=150; "
        "YIAN_MAX_VALUE_NESTING accepts an integer from 1 to 100"
    )
    assert "set YIAN_MAX_VALUE_NESTING=150" not in message


def test_check_recorded_accepts_an_out_of_range_value_in_effect(set_defaults) -> None:
    set_defaults.unchecked(max_value_nesting=150)
    check_recorded({"MAX_VALUE_NESTING": 150})


def test_every_test_starts_from_builtin_values() -> None:
    assert current() == Defaults()
