"""Every predefined default and limit yian uses, each overridable by environment.

Each entry has one built-in value and one environment variable, `YIAN_` plus the
entry name: `YIAN_MAX_VALUE_NESTING` overrides `MAX_VALUE_NESTING`. Limits and input
caps start with `MAX_`; defaults for settings a flow author can set are named
`<OWNER>_<SETTING>` after the authored setting (`AGENT_RETRIES`). Units are suffixes
(`_BYTES`, `_CHARS`); counts have none. Code reads a value as the lower-case
attribute of `current()`, for example `current().max_value_nesting`.

Values are read once per process: the first `current()` call reads and checks every
variable, and later changes to `os.environ` have no effect. An empty variable counts
as unset. A run records the values that differ from the built-in ones, and resuming
it with different values is refused. `docs/defaults.md` lists every entry.

This module imports only the standard library, so any layer may import it.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
import os
import re
import sys
import threading


@dataclass(frozen=True)
class Entry:
    """One default or limit and the range an environment value must fall in.

    Attributes:
        name (`str`): Upper snake case name, for example `MAX_VALUE_NESTING`.
        default (`int`): Built-in value.
        description (`str`): One line saying what the value controls.
        group (`str`): `"limit"`, `"setting"`, or `"input cap"`.
        minimum (`int`): Smallest value accepted from the environment.
        maximum (`int | None`): Largest value accepted, or `None` for no upper bound.
        no_limit (`int | None`): A value accepted outside the range that means no cap.
        maximum_from_interpreter (`bool`): Whether a set value must also not exceed
            the interpreter's int digit limit (`sys.get_int_max_str_digits()`).
    """

    name: str
    default: int
    description: str
    group: str
    minimum: int
    maximum: int | None
    no_limit: int | None = None
    maximum_from_interpreter: bool = False

    @property
    def variable(self) -> str:
        """Return the environment variable that overrides this entry."""
        return "YIAN_" + self.name


ENTRIES: tuple[Entry, ...] = (
    # Language safety limits.
    Entry(
        "MAX_VALUE_NESTING",
        100,
        "Deepest nesting a value may have; also bounds expression nesting",
        "limit",
        1,
        # Deeper nested call expressions measured a raw RecursionError with reduced
        # stack headroom; 150 levels passed.
        150,
    ),
    Entry(
        "MAX_TYPE_NESTING",
        100,
        "Deepest nesting a type expression may have",
        "limit",
        1,
        # Deeper types are rejected before they can exhaust Python's recursion limit.
        150,
    ),
    Entry(
        "MAX_YAML_NESTING",
        # Twice a value's 100 levels: a value at that limit fits inside a flow's own
        # structure, and composing and walking a document this deep from a shallow
        # caller stays within Python's default recursion limit.
        200,
        "Deepest nesting of a YAML document yian reads",
        "limit",
        1,
        200,
    ),
    Entry(
        "MAX_TOTAL_NODES",
        10_000,
        "Most live nodes a run's graph may grow to",
        "limit",
        1,
        None,
    ),
    Entry(
        "MAX_SKIP_WORLDS",
        10_000,
        "Most branch combinations the skip check explores before giving up",
        "limit",
        1,
        None,
    ),
    Entry(
        "MAX_INT_DIGITS",
        # Python's own default int digit limit.
        4300,
        "Most decimal digits an int may have when read or written as text",
        "limit",
        1,
        None,
        # A set value above the interpreter's limit could never be honoured; the
        # built-in value is not checked, so a lower PYTHONINTMAXSTRDIGITS applies first.
        maximum_from_interpreter=True,
    ),
    # Defaults for settings a flow author can set.
    Entry(
        "AGENT_RETRIES",
        2,
        "AGENT retries after a failed model call when the node sets none",
        "setting",
        0,
        None,
    ),
    Entry(
        "AGENT_MAX_TOOL_ITERATIONS",
        -1,
        "AGENT model turns with tool calls when the node sets none; -1 is no cap",
        "setting",
        1,
        None,
        no_limit=-1,
    ),
    Entry(
        "EPISODIC_MAX_EPISODES",
        100,
        "Episodes an episodic memory keeps when it sets no limit",
        "setting",
        1,
        None,
    ),
    Entry(
        "EPISODIC_RECALL_LIMIT",
        10,
        "Episodes an episodic memory recalls when it sets no limit",
        "setting",
        1,
        None,
    ),
    Entry(
        "REFLECTIVE_MAX_ITEMS",
        100,
        "Lessons a reflective memory keeps when it sets no limit",
        "setting",
        1,
        None,
    ),
    Entry(
        "REFLECTIVE_RECALL_LIMIT",
        20,
        "Lessons a reflective memory recalls when it sets no limit",
        "setting",
        1,
        None,
    ),
    # The three reflective budgets floor at 256: material over budget is cut to a
    # summary with a 64-char digest that is about 142 chars whatever the budget.
    Entry(
        "REFLECTIVE_MAX_LESSON_CHARS",
        8_000,
        "Characters of current lessons shown to the reflection model",
        "setting",
        256,
        None,
    ),
    Entry(
        "REFLECTIVE_MAX_EPISODE_CHARS",
        20_000,
        "Characters of the episode shown to the reflection model",
        "setting",
        256,
        None,
    ),
    Entry(
        "REFLECTIVE_MAX_LIMIT_CHARS",
        256,
        "Characters of the memory limits shown to the reflection model",
        "setting",
        256,
        None,
    ),
    Entry(
        "SESSION_HISTORY_BUDGET_CHARS",
        32_000,
        "Characters of history a session memory sends to the model",
        "setting",
        1,
        None,
    ),
    Entry(
        "SESSION_RECENT_TURNS",
        4,
        "Recent turns a session memory always keeps whole",
        "setting",
        0,
        None,
    ),
    Entry(
        "PROFILE_MAX_CHARS",
        32_000,
        "Characters a profile memory may hold",
        "setting",
        1,
        None,
    ),
    Entry(
        "PROFILE_INPUT_BUDGET_CHARS",
        24_000,
        "Characters of run material a profile memory sends to its learner model",
        "setting",
        1,
        None,
    ),
    # Input-size caps.
    Entry(
        "MAX_AUTH_RESPONSE_BYTES",
        1024 * 1024,
        "Largest response body read from a sign-in server",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_AUTH_STORE_BYTES",
        1024 * 1024,
        "Largest auth store file read",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_OAUTH_CALLBACK_BYTES",
        16 * 1024,
        "Largest request header block read from the sign-in callback",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_MODEL_LIST_BYTES",
        # OpenRouter's listing carries a description and pricing for every model.
        16 * 1024 * 1024,
        "Largest model listing read from a provider",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_BASE_URL_CHARS",
        2048,
        "Longest provider base URL accepted",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_SKILL_FILE_BYTES",
        256 * 1024,
        "Largest SKILL.md or bundled skill file returned to a model",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_SKILL_NAME_CHARS",
        64,
        "Longest skill name accepted",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_SKILL_DESCRIPTION_CHARS",
        1024,
        "Longest skill description shown in the skill listing",
        "input cap",
        # A cut keeps one character less than the cap and adds an ellipsis.
        2,
        None,
    ),
    Entry(
        "MAX_QUESTION_OPTIONS",
        6,
        "Most options a question to the user may offer",
        "input cap",
        1,
        None,
    ),
    Entry(
        "MAX_QUESTION_OPTION_CHARS",
        80,
        "Longest question option shown to the user",
        "input cap",
        # A cut keeps one character less than the cap and adds an ellipsis.
        2,
        None,
    ),
    Entry(
        "MAX_KEY_PROMPTS",
        # A non-interactive caller feeding empty input forever can't hang; 3 mirrors
        # sudo's `passwd_tries` default.
        3,
        "Times the CLI asks for an API key before giving up",
        "input cap",
        1,
        None,
    ),
)

_BY_NAME: dict[str, Entry] = {entry.name: entry for entry in ENTRIES}

# Memory wraps a run value a few levels deeper; 28 keeps the 128 levels memory
# allowed before the value limit was configurable.
_MEMORY_WRAPPER_LEVELS = 28


@dataclass(frozen=True)
class Defaults:
    """The value of every entry, one attribute per entry named in lower case."""

    max_value_nesting: int = 100
    max_type_nesting: int = 100
    max_yaml_nesting: int = 200
    max_total_nodes: int = 10_000
    max_skip_worlds: int = 10_000
    max_int_digits: int = 4300
    agent_retries: int = 2
    agent_max_tool_iterations: int = -1
    episodic_max_episodes: int = 100
    episodic_recall_limit: int = 10
    reflective_max_items: int = 100
    reflective_recall_limit: int = 20
    reflective_max_lesson_chars: int = 8_000
    reflective_max_episode_chars: int = 20_000
    reflective_max_limit_chars: int = 256
    session_history_budget_chars: int = 32_000
    session_recent_turns: int = 4
    profile_max_chars: int = 32_000
    profile_input_budget_chars: int = 24_000
    max_auth_response_bytes: int = 1024 * 1024
    max_auth_store_bytes: int = 1024 * 1024
    max_oauth_callback_bytes: int = 16 * 1024
    max_model_list_bytes: int = 16 * 1024 * 1024
    max_base_url_chars: int = 2048
    max_skill_file_bytes: int = 256 * 1024
    max_skill_name_chars: int = 64
    max_skill_description_chars: int = 1024
    max_question_options: int = 6
    max_question_option_chars: int = 80
    max_key_prompts: int = 3

    @property
    def memory_json_nesting(self) -> int:
        """Return the deepest JSON nesting memory material and stores may have."""
        return self.max_value_nesting + _MEMORY_WRAPPER_LEVELS

    def non_defaults(self) -> dict[str, int]:
        """Return the values that differ from the built-in ones.

        Returns:
            `dict[str, int]`: Entry name to value, sorted by name.
        """
        changed = {
            entry.name: getattr(self, entry.name.lower())
            for entry in ENTRIES
            if getattr(self, entry.name.lower()) != entry.default
        }
        return dict(sorted(changed.items()))

    @classmethod
    def from_recorded(cls, recorded: Mapping[str, int]) -> Defaults:
        """Return the built-in values overlaid with a run's recorded values.

        Args:
            recorded (`Mapping[str, int]`): Entry name to value, as a run recorded it.

        Returns:
            `Defaults`: The values the run started with.

        Raises:
            DefaultsMismatchError: If a recorded name is not an entry.
        """
        unknown = sorted(name for name in recorded if name not in _BY_NAME)
        if unknown:
            named = ", ".join("YIAN_" + name for name in unknown)
            raise DefaultsMismatchError(
                f"this run started with {named}, which this version of yian does "
                "not know, so it can't be resumed here"
            )
        return cls(**{name.lower(): value for name, value in recorded.items()})


class DefaultsError(ValueError):
    """An environment variable holds a value its entry does not accept."""


class DefaultsMismatchError(ValueError):
    """A run started with values that differ from the ones in effect now."""


_CANONICAL_INT = re.compile(r"0|-?[1-9][0-9]*")
# Checked before int() so a huge string never reaches Python's own digit limit.
_MAX_DIGITS = 18
_SHOWN_CHARS = 40


def _maximum(entry: Entry) -> int | None:
    """Return the largest value `entry` accepts from the environment now."""
    if not entry.maximum_from_interpreter:
        return entry.maximum
    # The function is missing before Python 3.10.7; 0 means the limit is off.
    interpreter = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if not interpreter:
        return entry.maximum
    if entry.maximum is None:
        return interpreter
    return min(entry.maximum, interpreter)


def _range_text(entry: Entry, maximum: int | None) -> str:
    """Describe the values `entry` accepts."""
    if maximum is None:
        text = f"an integer of at least {entry.minimum}"
    else:
        text = f"an integer from {entry.minimum} to {maximum}"
    if entry.no_limit is not None:
        text += f", or {entry.no_limit} for no cap"
    return text


def _parse(entry: Entry, raw: str) -> int:
    """Return the value `raw` sets for `entry`.

    Args:
        entry (`Entry`): The entry the variable overrides.
        raw (`str`): The variable's non-empty value.

    Returns:
        `int`: The checked value.

    Raises:
        DefaultsError: If `raw` is not a canonical decimal integer in range.
    """
    maximum = _maximum(entry)
    shown = raw if len(raw) <= _SHOWN_CHARS else raw[:_SHOWN_CHARS] + "..."
    advice = f"use {_range_text(entry, maximum)}, or unset it"
    if not _CANONICAL_INT.fullmatch(raw):
        raise DefaultsError(f"{entry.variable}={shown!r} is not an integer; {advice}")
    digits = len(raw.lstrip("-"))
    value = int(raw) if digits <= _MAX_DIGITS else None
    in_range = value is not None and (
        value == entry.no_limit
        or (value >= entry.minimum and (maximum is None or value <= maximum))
    )
    if not in_range:
        raise DefaultsError(f"{entry.variable}={shown!r} is out of range; {advice}")
    return value


def read(environ: Mapping[str, str]) -> Defaults:
    """Return the values `environ` sets, with built-in values for the rest.

    Args:
        environ (`Mapping[str, str]`): Environment to read, such as `os.environ`.

    Returns:
        `Defaults`: The checked values.

    Raises:
        DefaultsError: If a variable holds a value its entry does not accept.
    """
    values = {}
    for entry in ENTRIES:
        raw = environ.get(entry.variable)
        if raw:
            values[entry.name.lower()] = _parse(entry, raw)
    return Defaults(**values)


_lock = threading.Lock()
_values: Defaults | None = None


def current() -> Defaults:
    """Return this process's values, reading the environment on the first call.

    A failed read is not kept, so every call raises until the environment is fixed.

    Returns:
        `Defaults`: The values in effect for this process.

    Raises:
        DefaultsError: If a variable holds a value its entry does not accept.
    """
    global _values
    values = _values
    if values is not None:
        return values
    with _lock:
        if _values is None:
            _values = read(os.environ)
        return _values


def check_recorded(recorded: Mapping[str, int]) -> None:
    """Check that a run's recorded values equal the ones in effect now.

    Args:
        recorded (`Mapping[str, int]`): Entry name to value, as the run recorded it.

    Raises:
        DefaultsMismatchError: Naming every variable whose value differs, or a
            recorded name that is not an entry.
    """
    started = Defaults.from_recorded(recorded)
    here = current()
    problems = []
    for entry in ENTRIES:
        attribute = entry.name.lower()
        now, then = getattr(here, attribute), getattr(started, attribute)
        if now == then:
            continue
        if then == entry.default:
            problems.append(
                f"{entry.variable} is {now} here, but this run started with the "
                f"default {then}; unset {entry.variable} to resume it"
            )
        else:
            problems.append(
                f"{entry.variable} is {now} here, but this run started with {then}; "
                f"set {entry.variable}={then} to resume it"
            )
    if problems:
        raise DefaultsMismatchError("\n".join(problems))


def _reload() -> Defaults:
    """Re-read `os.environ` into this process's values; for tests only."""
    global _values
    with _lock:
        _values = None
    return current()


def _install(values: Defaults) -> None:
    """Make `values` this process's values with no range check; for tests only."""
    global _values
    with _lock:
        _values = values


def _forget() -> None:
    """Drop this process's values so the next `current()` reads again; for tests only."""
    global _values
    with _lock:
        _values = None


__all__ = [
    "ENTRIES",
    "Defaults",
    "DefaultsError",
    "DefaultsMismatchError",
    "Entry",
    "check_recorded",
    "current",
    "read",
]
