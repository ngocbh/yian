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
        maximum (`int | None`): Largest value accepted, or `None` for any value of
            up to 18 digits.
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
    # A learnable run logs a CALL binding's expression as a route two JSON levels per
    # call deep, and pydantic refuses to encode past 255 levels, so a 125-level call
    # expression can't be logged; 100 levels log, resume, and run with reduced stack
    # headroom.
    Entry(
        "MAX_VALUE_NESTING",
        default=100,
        description="Deepest nesting a value may have; also bounds expression nesting",
        group="limit",
        minimum=1,
        maximum=100,
    ),
    # Deeper types are rejected before they can exhaust Python's recursion limit. Type
    # text 125 levels deep still parses; a record type written in `output:` counts its
    # record levels too. Copies of a flow share its types, so record depth adds no stack
    # at run setup. The limit also bounds each typedef's levels, a chain of typedefs
    # naming each other, and a type with its typedef names expanded.
    Entry(
        "MAX_TYPE_NESTING",
        default=100,
        description="Deepest nesting a type may have, counting record levels",
        group="limit",
        minimum=1,
        maximum=125,
    ),
    # A model's structured reply schema is built by recursion, about 28 frames per level
    # for the costliest shape, an `Optional(Union(...))` field at each level. With 300
    # frames below the default recursion limit, under 50 nested calls, that shape passes
    # 23 levels on Python 3.12 and 18 on 3.10 through every structured-output path. An
    # AGENT at 16 levels under the worst accepted load, 50 `uses:` imports ending in a
    # 50-level chain, loads in about 780 frames on 3.12 and 850 on 3.10. Deeper types are
    # not proven to fit, so the variable can only lower it.
    Entry(
        "MAX_AGENT_OUTPUT_NESTING",
        default=16,
        description=(
            "Deepest nesting an AGENT `output:` type may have (each record and list is a level)"
        ),
        group="limit",
        minimum=1,
        maximum=16,
    ),
    # Twice a value's 100 levels: a value at that limit fits inside a flow's own
    # structure, and composing and walking a document this deep from a shallow caller
    # stays within Python's default recursion limit.
    Entry(
        "MAX_YAML_NESTING",
        default=200,
        description="Deepest nesting of a YAML document yian reads",
        group="limit",
        minimum=1,
        maximum=200,
    ),
    # Aliases make a document's expanded size exponential in its text, and every later
    # step, including the run log (JSON has no aliases), costs per expanded node.
    Entry(
        "MAX_YAML_NODES",
        default=100_000,
        description=(
            "Most nodes a YAML document yian reads may expand to, counting each alias as a copy"
        ),
        group="limit",
        minimum=1,
        maximum=None,
    ),
    # Loading a body or a `uses:` file recurses through the loader, so the limit holds
    # both chains. A 50-level chain loads, runs, and resumes within Python's default
    # recursion limit; the worst accepted load, 50 imports whose last file calls a
    # 50-level chain, peaks near 735 frames on Python 3.10 and 3.12. Deeper chains are
    # not proven to fit, so the variable can only lower it.
    Entry(
        "MAX_CALL_NESTING",
        default=50,
        description=(
            "Deepest chain of call, map, and loop bodies a flow may nest, and of uses: "
            "files it may import"
        ),
        group="limit",
        minimum=1,
        maximum=50,
    ),
    Entry(
        "MAX_TOTAL_NODES",
        default=10_000,
        description="Most live nodes a run's graph may grow to",
        group="limit",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "MAX_SKIP_WORLDS",
        default=10_000,
        description="Most branch combinations the skip check explores before giving up",
        group="limit",
        minimum=1,
        maximum=None,
    ),
    # The default is Python's own int digit limit. A set value above the interpreter's
    # limit could never be honoured; the built-in value is not checked, so a lower
    # PYTHONINTMAXSTRDIGITS applies first.
    Entry(
        "MAX_INT_DIGITS",
        default=4300,
        description="Most decimal digits an int may have to be a yian value",
        group="limit",
        minimum=1,
        maximum=None,
        maximum_from_interpreter=True,
    ),
    # A cut keeps one character less than the cap and adds `…`, so the minimum is 2.
    Entry(
        "MAX_SHOWN_VALUE_CHARS",
        default=200,
        description="Longest rendering of a value or key an error message shows",
        group="limit",
        minimum=2,
        maximum=None,
    ),
    # Defaults for settings a flow author can set.
    Entry(
        "AGENT_RETRIES",
        default=2,
        description=(
            "Extra attempts an AGENT gets to correct an invalid structured reply "
            "when the node sets none"
        ),
        group="setting",
        minimum=0,
        maximum=None,
    ),
    Entry(
        "AGENT_MAX_TOOL_ITERATIONS",
        default=-1,
        description="AGENT tool-call turns when the node sets none; -1 is no cap",
        group="setting",
        minimum=1,
        maximum=None,
        no_limit=-1,
    ),
    Entry(
        "EPISODIC_MAX_EPISODES",
        default=100,
        description="Episodes an episodic memory keeps when it sets no limit",
        group="setting",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "EPISODIC_RECALL_LIMIT",
        default=10,
        description="Episodes an episodic memory recalls when it sets no limit",
        group="setting",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "REFLECTIVE_MAX_ITEMS",
        default=100,
        description="Lessons a reflective memory keeps when it sets no limit",
        group="setting",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "REFLECTIVE_RECALL_LIMIT",
        default=20,
        description="Lessons a reflective memory recalls when it sets no limit",
        group="setting",
        minimum=1,
        maximum=None,
    ),
    # The three reflective budgets floor at 256: material over budget is cut to a
    # summary with a 64-char digest that is about 142 chars whatever the budget.
    Entry(
        "REFLECTIVE_MAX_LESSON_CHARS",
        default=8_000,
        description="Characters of current lessons shown to the reflection model",
        group="setting",
        minimum=256,
        maximum=None,
    ),
    Entry(
        "REFLECTIVE_MAX_EPISODE_CHARS",
        default=20_000,
        description="Characters of the episode shown to the reflection model",
        group="setting",
        minimum=256,
        maximum=None,
    ),
    Entry(
        "REFLECTIVE_MAX_LIMIT_CHARS",
        default=256,
        description="Characters of the memory limits shown to the reflection model",
        group="setting",
        minimum=256,
        maximum=None,
    ),
    Entry(
        "SESSION_HISTORY_BUDGET_CHARS",
        default=32_000,
        description="Characters of history a session memory sends to the model",
        group="setting",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "SESSION_RECENT_TURNS",
        default=4,
        description="Recent turns a session memory always keeps whole",
        group="setting",
        minimum=0,
        maximum=None,
    ),
    Entry(
        "PROFILE_MAX_CHARS",
        default=32_000,
        description="Characters a profile memory may hold",
        group="setting",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "PROFILE_INPUT_BUDGET_CHARS",
        default=24_000,
        description="Characters of run material sent to a profile memory's learner",
        group="setting",
        minimum=1,
        maximum=None,
    ),
    # Input-size caps.
    Entry(
        "MAX_AUTH_RESPONSE_BYTES",
        default=1024 * 1024,
        description="Largest response body read from a sign-in server",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "MAX_AUTH_STORE_BYTES",
        default=1024 * 1024,
        description="Largest auth store file read or written",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "MAX_OAUTH_CALLBACK_BYTES",
        default=16 * 1024,
        description="Largest request header block read from the sign-in callback",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    # OpenRouter's listing carries a description and pricing for every model.
    Entry(
        "MAX_MODEL_LIST_BYTES",
        default=16 * 1024 * 1024,
        description="Largest model listing read from a provider",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "MAX_BASE_URL_CHARS",
        default=2048,
        description="Longest provider base URL accepted",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "MAX_SKILL_FILE_BYTES",
        default=256 * 1024,
        description="Largest SKILL.md or bundled skill file returned to a model",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "MAX_SKILL_NAME_CHARS",
        default=64,
        description="Longest skill name accepted",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    # A cut keeps one character less than the cap and adds an ellipsis, so the
    # minimum is 2.
    Entry(
        "MAX_SKILL_DESCRIPTION_CHARS",
        default=1024,
        description="Longest skill description shown in the skill listing",
        group="input cap",
        minimum=2,
        maximum=None,
    ),
    Entry(
        "MAX_QUESTIONS",
        default=4,
        description="Most questions a HUMAN_INPUT node may ask at once",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    Entry(
        "MAX_QUESTION_OPTIONS",
        default=6,
        description="Most options a question to the user may offer",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
    # A cut keeps one character less than the cap and adds an ellipsis, so the
    # minimum is 2.
    Entry(
        "MAX_QUESTION_OPTION_CHARS",
        default=80,
        description="Longest question option shown to the user",
        group="input cap",
        minimum=2,
        maximum=None,
    ),
    # A non-interactive caller feeding empty input forever can't hang; 3 mirrors
    # sudo's `passwd_tries` default.
    Entry(
        "MAX_KEY_PROMPTS",
        default=3,
        description="Times the CLI asks for an API key before giving up",
        group="input cap",
        minimum=1,
        maximum=None,
    ),
)

_BY_NAME: dict[str, Entry] = {entry.name: entry for entry in ENTRIES}

# Memory wraps a run value a few levels deeper; 28 keeps the default memory bound at
# 128, the bound existing memory stores were written under.
_MEMORY_WRAPPER_LEVELS = 28


@dataclass(frozen=True)
class Defaults:
    """The value of every entry, one attribute per entry named in lower case."""

    max_value_nesting: int = 100
    max_type_nesting: int = 100
    max_agent_output_nesting: int = 16
    max_yaml_nesting: int = 200
    max_yaml_nodes: int = 100_000
    max_call_nesting: int = 50
    max_total_nodes: int = 10_000
    max_skip_worlds: int = 10_000
    max_int_digits: int = 4300
    max_shown_value_chars: int = 200
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
    max_questions: int = 4
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
        unknown = _unknown_names(recorded)
        if unknown:
            raise DefaultsMismatchError("\n".join(unknown))
        return cls(**{name.lower(): value for name, value in recorded.items()})


def _unknown_names(recorded: Mapping[str, int]) -> list[str]:
    """Describe each recorded name that is not an entry, sorted by name."""
    return [
        f"this run started with YIAN_{name}, which this version of yian does not know"
        for name in sorted(recorded)
        if name not in _BY_NAME
    ]


class DefaultsError(ValueError):
    """An environment variable holds a value its entry does not accept."""


class DefaultsMismatchError(ValueError):
    """A run started with values that differ from the ones in effect now."""


_CANONICAL_INT = re.compile(r"0|-?[1-9][0-9]*")
# Values have at most 18 digits, checked before int() so a huge string never reaches
# Python's own digit limit; an entry with no upper bound stops at the largest of them.
_MAX_DIGITS = 18
_LARGEST = 10**_MAX_DIGITS - 1
_SHOWN_CHARS = 40


def _maximum(entry: Entry) -> int:
    """Return the largest value `entry` accepts from the environment now."""
    maximum = _LARGEST if entry.maximum is None else entry.maximum
    if entry.maximum_from_interpreter:
        # The function is missing before Python 3.10.7; 0 means the limit is off.
        interpreter = getattr(sys, "get_int_max_str_digits", lambda: 0)()
        if interpreter:
            maximum = min(maximum, interpreter)
    return maximum


def _range_text(entry: Entry, maximum: int) -> str:
    """Describe the values `entry` accepts."""
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
    shown = raw if len(raw) <= _SHOWN_CHARS else raw[: _SHOWN_CHARS - 3] + "..."
    advice = f"use {_range_text(entry, maximum)}, or unset it"
    if not _CANONICAL_INT.fullmatch(raw):
        raise DefaultsError(f"{entry.variable}={shown!r} is not an integer; {advice}")
    digits = len(raw.lstrip("-"))
    value = int(raw) if digits <= _MAX_DIGITS else None
    in_range = value is not None and (
        value == entry.no_limit
        or entry.minimum <= value <= maximum
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
        DefaultsMismatchError: Naming every recorded name that is not an entry
            and every variable whose value differs, in one message.
    """
    here = current()
    problems = _unknown_names(recorded)
    for entry in ENTRIES:
        attribute = entry.name.lower()
        now, then = getattr(here, attribute), recorded.get(entry.name, entry.default)
        if now == then:
            continue
        maximum = _maximum(entry)
        if then != entry.default and not (
            then == entry.no_limit or entry.minimum <= then <= maximum
        ):
            # Advising `set YIAN_X=<then>` would name a value the variable refuses.
            problems.append(
                f"this run recorded an invalid value {entry.variable}={then}; "
                f"{entry.variable} accepts {_range_text(entry, maximum)}"
            )
        elif then == entry.default:
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
