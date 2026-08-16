"""`ac run` interactive input prompts and their type-directed value adapters.

At run start the CLI prints a `_flow_banner` (name/description/version); each prompt's
label carries the input's declared `type`, a required (`*`) / `optional` mark, and any
default (`_input_label`). `_gather_missing_inputs` covers both non-TTY omission/error
policy and an interactive path with fake `questionary`, so prompt labels are observable
without a real TTY. Structural text prompts accept recursively validated JSON arrays and
objects, while scalar text and `--input` flag values retain their string-valued boundary.
A runtime failure (`_render_run_error`) boxes the `.yaml` at the node that raised,
mirroring the compile-error frame.
"""

import math
import sys
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
import typer

import agent_composer.cli.run as climod
from agent_composer.cli.run import (
    _flow_banner,
    _gather_missing_inputs,
    _input_label,
    _parse_kv,
    _parse_structured_input,
    _render_run_error,
)
from agent_composer.compose.loader import load_flow
from agent_composer.diagnostics import Diagnostic, Failure, OriginRef
from agent_composer.nodes.contracts import Parameter, ValueContract
from agent_composer.run import RunResult, run_flow
from agent_composer.tools import install_ask_user
from agent_composer.typesys import Type, ValueKind

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"
_ERRORS = _SEEDS / "errors"


def _parameter(
    name: str,
    typ: Type,
    *,
    required: bool = False,
    has_default: bool = False,
    default=None,
) -> Parameter:
    """Build one source-free canonical flow parameter for CLI presentation."""
    return Parameter(
        name=name,
        contract=ValueContract(typ),
        required=required,
        has_default=has_default,
        default=default,
        coerce_strings=True,
    )


def _decls() -> list[Parameter]:
    return [
        _parameter("topic", Type.scalar(ValueKind.STRING), required=True),
        _parameter("as_of", Type(kind=ValueKind.DATE, nullable=True)),
        _parameter(
            "window",
            Type.scalar(ValueKind.INTEGER),
            has_default=True,
            default=30,
        ),
        _parameter(
            "mode",
            Type(
                kind=ValueKind.STRING,
                tags=frozenset({"fast", "slow"}),
            ),
            required=True,
        ),
    ]


# --- _flow_banner: the "Running flow" identity block ------------------------- #
def _plain(text) -> str:
    """Render a Rich Text to a plain (un-styled) string for substring assertions."""
    buf = StringIO()
    Console(file=buf, force_terminal=False, width=100).print(text)
    return buf.getvalue()


def test_flow_banner_name_and_description():
    out = _plain(_flow_banner("my_flow", "Does a useful thing."))
    assert "Running flow: my_flow" in out
    assert "Description: Does a useful thing." in out


def test_flow_banner_includes_version():
    out = _plain(_flow_banner("f", "d", version="v2"))
    assert "version: v2" in out


def test_flow_banner_none_when_no_metadata():
    assert _flow_banner(None, None, None) is None


# --- _input_label: name + type + required/optional/default -------------------- #
def test_input_label_required():
    topic = next(d for d in _decls() if d.name == "topic")
    assert _input_label(topic) == "topic (str) *"


def test_input_label_optional_no_default():
    as_of = next(d for d in _decls() if d.name == "as_of")
    assert _input_label(as_of) == "as_of (Optional[date]) [optional] e.g. 2026-05-21"


def test_input_label_optional_with_default():
    window = next(d for d in _decls() if d.name == "window")
    assert _input_label(window) == "window (int) [default: 30]"


def test_input_label_datetime_example():
    at = _parameter("at", Type.scalar(ValueKind.DATETIME), required=True)
    assert _input_label(at) == "at (datetime) * e.g. 2026-05-21T14:30"


def test_input_label_default_suppresses_example():
    # A date default already shows the accepted format, so no redundant `e.g.`.
    as_of = _parameter(
        "as_of",
        Type(kind=ValueKind.DATE, nullable=True),
        has_default=True,
        default="2026-01-01",
    )
    assert _input_label(as_of) == "as_of (Optional[date]) [default: 2026-01-01]"


def test_input_label_derives_a_canonical_structural_type():
    profile = _parameter(
        "profile",
        Type(
            kind=ValueKind.OBJECT,
            fields={
                "z": Type.scalar(ValueKind.INTEGER),
                "a": Type(kind=ValueKind.STRING, nullable=True),
            },
            required=frozenset({"z"}),
        ),
        required=True,
    )

    assert _input_label(profile) == 'profile ({"a":Optional[str],"z":int}) *'


def test_input_label_displays_an_explicit_null_default():
    maybe = _parameter(
        "maybe",
        Type(kind=ValueKind.INTEGER, nullable=True),
        has_default=True,
        default=None,
    )

    assert _input_label(maybe) == "maybe (Optional[int]) [default: None]"


def test_input_label_supports_a_flat_list_contract():
    items = _parameter(
        "items",
        Type.scalar(ValueKind.LIST_STRING),
        required=True,
    )

    assert _input_label(items) == "items (list[str]) *"


# --- _gather_missing_inputs: policy + interactive labels --------------------- #
class _FakeChoice:
    """Questionary-compatible display/value pair used by the prompt test double."""

    def __init__(self, title, value=None, **_kwargs):
        self.title = title
        self.value = title if value is None else value


_UNSCRIPTED = object()


@dataclass(frozen=True)
class _Attempts:
    """Ordered raw text attempts supplied to one validator-aware fake widget."""

    values: tuple[object, ...]


class _FakeQuestionary:
    """Record prompts and model Questionary's accepted-default return values.

    ``answers`` maps a label substring to one explicit answer, including ``None``
    for cancellation. An unscripted widget accepts its default; a select without
    one accepts its first choice. Selecting a ``Choice`` returns its value.

    A text answer may instead be an `_Attempts` carrier. The fake feeds its ordered
    raw values only to the production validator, records each literal ``True`` or
    rejection message, and returns the first accepted raw value without parsing it.
    Cancellation bypasses validation, matching Questionary.
    """

    def __init__(self, answers):
        self.labels: list[str] = []
        self.text_defaults: list[str] = []
        self.calls: list[SimpleNamespace] = []
        self._answers = answers

    def Style(self, spec):  # questionary.Style(...) — styling is irrelevant to the labels
        return None

    def Choice(self, title, value=None, **kwargs):
        return _FakeChoice(title, value=value, **kwargs)

    def _scripted(self, label):
        self.labels.append(label)
        for key, val in self._answers.items():
            if key in label:
                return val
        return _UNSCRIPTED

    @staticmethod
    def _choice_value(value):
        return value.value if isinstance(value, _FakeChoice) else value

    def text(self, label, default="", **kwargs):
        self.text_defaults.append(default)
        scripted = self._scripted(label)
        val = default if scripted is _UNSCRIPTED else scripted
        call = SimpleNamespace(
            kind="text",
            label=label,
            default=default,
            choices=None,
            instruction=kwargs.get("instruction"),
            verdicts=[],
            rejections=[],
        )
        self.calls.append(call)

        def ask():
            attempts = val.values if isinstance(val, _Attempts) else (val,)
            validator = kwargs.get("validate")
            for attempt in attempts:
                if attempt is None:
                    return None
                if validator is None:
                    return attempt
                verdict = validator(attempt)
                if verdict is not True and not isinstance(verdict, str):
                    raise AssertionError(
                        "production validator must return literal True or a message"
                    )
                call.verdicts.append(verdict)
                if verdict is True:
                    return attempt
                call.rejections.append(verdict)
            raise AssertionError("all scripted text attempts were rejected")

        return SimpleNamespace(ask=ask)

    def confirm(self, label, default=False, **kwargs):
        scripted = self._scripted(label)
        val = default if scripted is _UNSCRIPTED else scripted
        self.calls.append(
            SimpleNamespace(kind="confirm", label=label, default=default, choices=None)
        )
        return SimpleNamespace(ask=lambda: val)

    def select(self, label, choices, default=None, **kwargs):
        scripted = self._scripted(label)
        selected = (
            default
            if scripted is _UNSCRIPTED and default is not None
            else choices[0]
            if scripted is _UNSCRIPTED
            else scripted
        )
        val = self._choice_value(selected)
        self.calls.append(
            SimpleNamespace(
                kind="select",
                label=label,
                default=default,
                choices=list(choices),
            )
        )
        return SimpleNamespace(ask=lambda: val)


def _sink(monkeypatch) -> StringIO:
    buf = StringIO()
    monkeypatch.setattr(climod, "err_console", Console(file=buf, force_terminal=False, width=100))
    return buf


def _record_list_type() -> Type:
    """Return a list of records whose required ``count`` field is an integer."""
    return Type(
        kind=ValueKind.LIST_OBJECT,
        element=Type(
            kind=ValueKind.OBJECT,
            fields={"count": Type.scalar(ValueKind.INTEGER)},
            required=frozenset({"count"}),
        ),
    )


@pytest.mark.parametrize(
    ("raw", "typ", "expected"),
    [
        ("[1, 2]", Type.scalar(ValueKind.LIST_INTEGER), [1, 2]),
        (
            '{"count": 2}',
            Type(
                kind=ValueKind.OBJECT,
                fields={"count": Type.scalar(ValueKind.INTEGER)},
                required=frozenset({"count"}),
            ),
            {"count": 2},
        ),
    ],
)
def test_parse_structured_input_returns_native_arrays_and_objects(raw, typ, expected):
    assert _parse_structured_input(raw, typ) == expected


@pytest.mark.parametrize(
    ("raw", "typ", "message"),
    [
        (
            "[",
            Type.scalar(ValueKind.LIST_ANY),
            "Invalid JSON at line 1, column 2: Expecting value",
        ),
        ("{}", Type.scalar(ValueKind.LIST_ANY), "Expected a JSON array"),
        ("[]", Type.scalar(ValueKind.OBJECT), "Expected a JSON object"),
        (
            '["bad"]',
            Type.scalar(ValueKind.LIST_INTEGER),
            "$[0]: list element 'bad' is not int (declared list[int])",
        ),
        (
            '[{"count": "bad"}]',
            _record_list_type(),
            "$[0][\"count\"]: 'bad' does not match declared type int",
        ),
    ],
)
def test_parse_structured_input_reports_concise_value_failures(raw, typ, message):
    with pytest.raises(ValueError) as captured:
        _parse_structured_input(raw, typ)

    assert str(captured.value) == message


def test_parse_structured_input_rejects_null_at_nullable_structural_root():
    typ = Type(kind=ValueKind.LIST_ANY, nullable=True)

    with pytest.raises(ValueError) as captured:
        _parse_structured_input("null", typ)

    assert str(captured.value) == "Expected a JSON array"


def test_parse_structured_input_accepts_null_inside_nullable_element():
    typ = Type(
        kind=ValueKind.LIST_INTEGER,
        element=Type(kind=ValueKind.INTEGER, nullable=True),
    )

    assert _parse_structured_input("[null, 3]", typ) == [None, 3]


def test_parse_structured_input_keeps_flat_list_any_behavior():
    raw = '[1, "two", null, {"ok": true}, [3]]'

    assert _parse_structured_input(raw, Type.scalar(ValueKind.LIST_ANY)) == [
        1,
        "two",
        None,
        {"ok": True},
        [3],
    ]


def test_parse_structured_input_keeps_json_loads_nonfinite_number_behavior():
    parsed = _parse_structured_input(
        "[NaN, Infinity, -Infinity]",
        Type.scalar(ValueKind.LIST_NUMBER),
    )

    assert math.isnan(parsed[0])
    assert parsed[1:] == [math.inf, -math.inf]


def test_parse_structured_input_does_not_hide_validator_faults(monkeypatch):
    fault = RuntimeError("validator implementation failed")

    def fail(_typ, _value):
        raise fault

    monkeypatch.setattr(climod, "build_value_as", fail)

    with pytest.raises(RuntimeError) as captured:
        _parse_structured_input("[]", Type.scalar(ValueKind.LIST_ANY))

    assert captured.value is fault


def test_structured_prompt_does_not_turn_plain_value_error_into_a_verdict(monkeypatch):
    fault = ValueError("validator implementation failed")

    def fail(_typ, _value):
        raise fault

    fake = _FakeQuestionary({"items": _Attempts(("[]",))})
    monkeypatch.setattr(climod, "build_value_as", fail)
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    with pytest.raises(ValueError) as captured:
        _gather_missing_inputs(
            [_parameter("items", Type.scalar(ValueKind.LIST_ANY), required=True)],
            {},
            prompt_available=True,
        )

    assert captured.value is fault
    assert fake.calls[0].verdicts == []


def test_parse_kv_keeps_structured_looking_values_as_strings():
    parsed = _parse_kv(["items=[1, 2]", 'profile={"count": 2}'])

    assert parsed == {"items": "[1, 2]", "profile": '{"count": 2}'}
    assert all(isinstance(value, str) for value in parsed.values())


def test_structured_prompt_retries_nested_failure_and_returns_native_value(monkeypatch):
    fake = _FakeQuestionary(
        {"rows": _Attempts(('[{"count": "bad"}]', '[{"count": 2}]'))}
    )
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(
        [_parameter("rows", _record_list_type(), required=True)],
        {},
        prompt_available=True,
    )

    assert gathered == {"rows": [{"count": 2}]}
    call = fake.calls[0]
    assert call.instruction == "Enter a JSON array"
    assert call.verdicts == [
        "$[0][\"count\"]: 'bad' does not match declared type int",
        True,
    ]
    assert call.rejections == [
        "$[0][\"count\"]: 'bad' does not match declared type int"
    ]


def test_object_prompt_returns_native_value_and_json_instruction(monkeypatch):
    fake = _FakeQuestionary({"profile": _Attempts(('{"count": 2}',))})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={"count": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"count"}),
    )

    gathered = _gather_missing_inputs(
        [_parameter("profile", typ, required=True)],
        {},
        prompt_available=True,
    )

    assert gathered == {"profile": {"count": 2}}
    assert fake.calls[0].instruction == "Enter a JSON object"
    assert fake.calls[0].verdicts == [True]


def test_optional_structured_whitespace_is_validated_then_omitted(monkeypatch):
    fake = _FakeQuestionary({"items": _Attempts(("   ",))})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(
        [_parameter("items", Type.scalar(ValueKind.LIST_STRING))],
        {},
        prompt_available=True,
    )

    assert gathered == {}
    assert fake.calls[0].verdicts == [True]


def test_required_structured_whitespace_is_rejected_then_retried(monkeypatch):
    fake = _FakeQuestionary({"items": _Attempts(("   ", "[]"))})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(
        [_parameter("items", Type.scalar(ValueKind.LIST_STRING), required=True)],
        {},
        prompt_available=True,
    )

    assert gathered == {"items": []}
    assert fake.calls[0].verdicts == [
        "Invalid JSON at line 1, column 4: Expecting value",
        True,
    ]


def test_structured_prompt_cancellation_bypasses_validation(monkeypatch):
    fake = _FakeQuestionary({"items": None})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(
        [_parameter("items", Type.scalar(ValueKind.LIST_STRING), required=True)],
        {},
        prompt_available=True,
    )

    assert gathered is None
    assert fake.calls[0].verdicts == []


def test_structured_prompt_rejects_textual_null_then_retries(monkeypatch):
    fake = _FakeQuestionary({"items": _Attempts(("null", "[null]"))})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(
        [
            _parameter(
                "items",
                Type(kind=ValueKind.LIST_ANY, nullable=True),
                required=True,
            )
        ],
        {},
        prompt_available=True,
    )

    assert gathered == {"items": [None]}
    assert fake.calls[0].verdicts == ["Expected a JSON array", True]


def test_prompt_labels_carry_marks(monkeypatch):
    fake = _FakeQuestionary({"topic": "ACME", "mode": "fast"})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    _gather_missing_inputs(_decls(), {}, prompt_available=True)
    joined = " || ".join(fake.labels)
    assert "topic (str) *" in joined
    assert "as_of (Optional[date]) [optional]" in joined
    assert "window (int) [default: 30]" in joined


def test_prompt_skips_already_supplied(monkeypatch):
    # Everything supplied -> nothing to prompt -> no prompt issued, empty gathered.
    fake = _FakeQuestionary({})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    have = {"topic": "x", "as_of": "2026-01-01", "window": 5, "mode": "fast"}
    gathered = _gather_missing_inputs(_decls(), have, prompt_available=True)
    assert gathered == {}
    assert fake.labels == []          # no prompt issued


def test_gather_treats_present_falsey_values_as_supplied_without_questionary(
    monkeypatch,
):
    parameters = [
        _parameter(
            "none",
            Type(kind=ValueKind.STRING, nullable=True),
            required=True,
        ),
        _parameter("false", Type.scalar(ValueKind.BOOLEAN), required=True),
        _parameter("zero", Type.scalar(ValueKind.INTEGER), required=True),
        _parameter("blank", Type.scalar(ValueKind.STRING), required=True),
    ]
    monkeypatch.setitem(sys.modules, "questionary", None)

    gathered = _gather_missing_inputs(
        parameters,
        {"none": None, "false": False, "zero": 0, "blank": ""},
        prompt_available=True,
    )

    assert gathered == {}


def test_gather_non_tty_leaves_optional_and_defaulted_inputs_absent(monkeypatch):
    parameters = [
        _parameter("optional", Type(kind=ValueKind.STRING, nullable=True)),
        _parameter(
            "defaulted",
            Type.scalar(ValueKind.INTEGER),
            has_default=True,
            default=7,
        ),
    ]
    monkeypatch.setitem(sys.modules, "questionary", None)

    gathered = _gather_missing_inputs(
        parameters,
        {},
        prompt_available=False,
    )

    assert gathered == {}


def test_gather_non_tty_reports_all_missing_required_inputs_in_order(monkeypatch):
    parameters = [
        _parameter("first", Type.scalar(ValueKind.STRING), required=True),
        _parameter("optional", Type(kind=ValueKind.STRING, nullable=True)),
        _parameter("second", Type.scalar(ValueKind.INTEGER), required=True),
    ]
    monkeypatch.setitem(sys.modules, "questionary", None)

    with pytest.raises(typer.BadParameter) as captured:
        _gather_missing_inputs(parameters, {}, prompt_available=False)

    assert captured.value.message == "missing required flow inputs: first, second"
    assert captured.value.param_hint == ("--input", "--inputs")


def test_gather_non_tty_filters_required_inputs_by_exact_key_membership(monkeypatch):
    parameters = [
        _parameter(
            "none",
            Type(kind=ValueKind.STRING, nullable=True),
            required=True,
        ),
        _parameter("false", Type.scalar(ValueKind.BOOLEAN), required=True),
        _parameter("blank", Type.scalar(ValueKind.STRING), required=True),
        _parameter("missing", Type.scalar(ValueKind.INTEGER), required=True),
    ]
    monkeypatch.setitem(sys.modules, "questionary", None)

    with pytest.raises(typer.BadParameter) as captured:
        _gather_missing_inputs(
            parameters,
            {"none": None, "false": False, "blank": ""},
            prompt_available=False,
        )

    assert captured.value.message == "missing required flow input: missing"


def test_prompt_accepting_omission_preserves_start_owned_defaults(monkeypatch):
    loaded = load_flow(
        """
id: prompt-defaults
name: prompt-defaults
input:
  items: list[str] = [a, b]
  mode: Literal["z", "a"] = z
  flag: Optional[bool]
nodes:
  show:
    kind: code
    code: |-
      return inputs
    input:
      items: ${input.items}
      mode: ${input.mode}
      flag: ${input.flag}
    output: object
output: ${show.output}
"""
    )
    fake = _FakeQuestionary({})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(loaded.input, {}, prompt_available=True)

    assert gathered == {}
    result = run_flow(loaded, gathered)
    assert result.status == "succeeded"
    assert result.input == {}
    assert result.output == {
        "flag": None,
        "items": ["a", "b"],
        "mode": "z",
    }

    by_name = {
        call.label.split(" ", 1)[0]: call
        for call in fake.calls
    }
    assert by_name["items"].kind == "text"
    assert by_name["items"].default == ""
    for name in ("mode", "flag"):
        call = by_name[name]
        assert call.kind == "select"
        assert isinstance(call.default, _FakeChoice)
        assert call.default is call.choices[0]
        assert call.default.value is not None
    assert [choice.value for choice in by_name["flag"].choices[1:]] == [True, False]
    assert [choice.value for choice in by_name["mode"].choices[1:]] == ["a", "z"]


def test_prompt_collects_false_enum_and_text_overrides(monkeypatch):
    fake = _FakeQuestionary(
        {"flag": False, "mode": "slow", "note": "replacement"}
    )
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    parameters = [
        _parameter("flag", Type(kind=ValueKind.BOOLEAN, nullable=True)),
        _parameter(
            "mode",
            Type(kind=ValueKind.STRING, tags=frozenset({"fast", "slow"})),
            has_default=True,
            default="fast",
        ),
        _parameter(
            "note",
            Type.scalar(ValueKind.STRING),
            has_default=True,
            default="original",
        ),
    ]

    gathered = _gather_missing_inputs(parameters, {}, prompt_available=True)

    assert gathered == {
        "flag": False,
        "mode": "slow",
        "note": "replacement",
    }


def test_prompt_distinguishes_literal_values_from_omission_labels(monkeypatch):
    collision = "↩ Use declared default"
    fake = _FakeQuestionary({"mode": collision})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    parameter = _parameter(
        "mode",
        Type(kind=ValueKind.STRING, tags=frozenset({collision, "ordinary"})),
        has_default=True,
        default="ordinary",
    )

    gathered = _gather_missing_inputs([parameter], {}, prompt_available=True)

    assert gathered == {"mode": collision}
    choices = fake.calls[0].choices
    assert len({choice.title for choice in choices}) == len(choices)
    assert any(
        choice.title == f"Set value: {collision}" and choice.value == collision
        for choice in choices
    )


def test_prompt_distinguishes_cancellation_from_omission(monkeypatch):
    fake = _FakeQuestionary({"flag": None})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(
        [_parameter("flag", Type(kind=ValueKind.BOOLEAN, nullable=True))],
        {},
        prompt_available=True,
    )

    assert gathered is None


def test_prompt_preserves_required_false_enum_and_blank_text(monkeypatch):
    fake = _FakeQuestionary({"flag": False, "mode": "slow", "note": ""})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    parameters = [
        _parameter("flag", Type.scalar(ValueKind.BOOLEAN), required=True),
        _parameter(
            "mode",
            Type(kind=ValueKind.STRING, tags=frozenset({"fast", "slow"})),
            required=True,
        ),
        _parameter("note", Type.scalar(ValueKind.STRING), required=True),
    ]

    gathered = _gather_missing_inputs(parameters, {}, prompt_available=True)

    assert gathered == {"flag": False, "mode": "slow", "note": ""}
    assert [call.kind for call in fake.calls] == ["confirm", "select", "text"]


def test_prompt_leaves_explicit_null_defaults_omitted_for_every_widget_kind(monkeypatch):
    fake = _FakeQuestionary({})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    parameters = [
        _parameter(
            "maybe",
            Type(kind=ValueKind.INTEGER, nullable=True),
            has_default=True,
            default=None,
        ),
        _parameter(
            "flag",
            Type(kind=ValueKind.BOOLEAN, nullable=True),
            has_default=True,
            default=None,
        ),
        _parameter(
            "mode",
            Type(
                kind=ValueKind.STRING,
                tags=frozenset({"fast", "slow"}),
                nullable=True,
            ),
            has_default=True,
            default=None,
        ),
    ]

    gathered = _gather_missing_inputs(parameters, {}, prompt_available=True)

    assert gathered == {}
    assert [call.kind for call in fake.calls] == ["text", "select", "select"]


def test_prompt_treats_optional_whitespace_as_omission(monkeypatch):
    fake = _FakeQuestionary({"note": "   "})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)

    gathered = _gather_missing_inputs(
        [_parameter("note", Type(kind=ValueKind.STRING, nullable=True))],
        {},
        prompt_available=True,
    )

    assert gathered == {}


# --- _render_run_error: a runtime failure boxes the .yaml at the PRECISE line ------ #
def test_run_error_uses_loaded_canonical_diagnosis(monkeypatch):
    """The CLI renders the front end's Diagnostic without resolving source itself."""
    failure = Failure(
        "node.execute",
        "semantic failure",
        "work",
        trace=(OriginRef("site"),),
    )
    result = RunResult(
        input={},
        status="failed",
        run_id="cli-test",
        log=None,
        failure=failure,
    )
    seen = []

    class _Loaded:
        def diagnose(self, candidate):
            seen.append(candidate)
            return Diagnostic(
                code=candidate.code,
                message=candidate.message,
                notes=("canonical presentation",),
                node_id=candidate.node_id,
            )

    buf = _sink(monkeypatch)
    _render_run_error(result, _Loaded())

    assert seen == [failure]
    assert "semantic failure" in buf.getvalue()
    assert "canonical presentation" in buf.getvalue()


def test_run_error_boxes_failing_node(monkeypatch):
    # e07 omits `as_of` -> the canonical failure trace points at the failing
    # `:?` binding.
    flow = _ERRORS / "e07-required-missing.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e07-required-missing.yaml:23" in out   # the as_of binding, not the node header
    assert "report" in out                         # still inside the source window
    assert "as_of is required for the report" in out


def test_run_error_uses_reference_code_origin(monkeypatch):
    # A referenced CODE failure carries its authored `code:` scalar in the canonical
    # origin trace; no CLI kind fallback is involved.
    flow = _ERRORS / "e20-code-raises.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e20-code-raises.yaml:12" in out
    assert "╭" in out


def test_run_error_boxes_boundary_assert_line(monkeypatch):
    # e18's boundary assert fails before any node runs; its authored origin points
    # at the exact `asserts:` expression.
    flow = _ERRORS / "e18-false-boundary-assert.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"window": -5})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e18-false-boundary-assert.yaml:15" in out   # the `${input.window} > 0` line
    assert "assert failed" in out


def test_run_error_boxes_code_wrong_type_output(monkeypatch):
    # e21's code node returns a value that fails its declared `output: int` -> the typed write
    # boundary rejects it and attributes the failure to the `output:` declaration.
    flow = _ERRORS / "e21-code-wrong-type.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e21-code-wrong-type.yaml:12" in out   # the `output: int` field line
    assert "╭" in out


def test_run_error_boxes_flow_input_line(monkeypatch):
    # e08's `window` cannot coerce to int, so its origin selects the declaration.
    flow = _ERRORS / "e08-input-type-mismatch.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X", "window": "soon"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e08-input-type-mismatch.yaml:20" in out     # the `window: int` decl line
    assert "╭" in out


def test_run_error_plain_when_unlocatable(monkeypatch):
    # A failure with no authored origin remains a semantic message with no frame.
    flow = _ERRORS / "e18-false-boundary-assert.yaml"
    text = flow.read_text()
    result = RunResult(
        input={},
        status="failed",
        run_id="cli-test",
        log=None,
        failure=Failure("graph.terminal_unreachable", "boom", None),
        events=[],
    )
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert out.strip() == "boom"
    assert "╭" not in out                          # no boxed frame


def test_run_error_renders_namespaced_failure_origin_trace(monkeypatch):
    # The failure trace carries the owning CALL site despite the runtime-namespaced
    # node id (`run/boom`).
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e24-nested-code-raise.yaml:27" in out   # the owning `run:` call node line
    assert "╭" in out                               # boxed, not a plain `run failed:` line


def test_run_error_boxes_loop_max_exceeded_at_loop_node(monkeypatch):
    # The LOOP origin remains stable even when the runtime driver id carries a
    # respawn suffix (`polish~5`).
    flow = _SEEDS / "28-refine-loop.yaml"
    text = flow.read_text()
    loaded = load_flow(text, source_label=flow.name)
    loop_origin = loaded.compiled.nodes["polish"].origin
    assert loop_origin is not None
    result = RunResult(
        input={},
        status="failed",
        run_id="cli-test",
        log=None,
        failure=Failure(
            "node.execute",
            "loop 'polish' exceeded max (5)",
            "polish~5",
            trace=(loop_origin,),
            cause_type="LoopMaxExceeded",
        ),
        events=[],
    )
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "╭" in out                                # boxed, not a plain `run failed:` line
    assert "28-refine-loop.yaml:45" in out           # resolved to the authored `polish:` loop node
    assert "exceeded max (5)" in out


def test_run_error_traceback_only_under_engine_trace(monkeypatch):
    # The captured Python traceback (a code/tool/agent raise) is surfaced ONLY when
    # `engine_trace=True`; the default terse display omits it.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.failure is not None
    assert result.failure.traceback  # captured at the node-failure boundary

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded, engine_trace=False)
    assert "python traceback" not in buf.getvalue()

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded, engine_trace=True)
    out = buf.getvalue()
    assert "python traceback" in out
    assert "intentional CODE failure" in out         # the originating Python exception


# --- _render_run_error: ordered canonical frames across child flows ----------------- #
def test_run_error_multi_frame_stack_into_def(monkeypatch):
    # e24's trace resolves the top-level CALL site followed by the failing CODE
    # origin inside `defs:inner`.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "e24-nested-code-raise.yaml:27" in out   # frame 0: the owning `run:` call node (header)
    assert "e24-nested-code-raise.yaml:20" in out  # frame 1: the authored `code:` value
    assert "intentional CODE failure" in out


def test_run_error_multi_frame_stack_into_external(monkeypatch):
    # e25's ordered origins cross the file boundary: the outer CALL site is in
    # this file and the failing CODE origin is in lib_boom.yaml.
    flow = _ERRORS / "e25-external-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "e25-external-raise.yaml:19" in out      # frame 0: the `go:` call node (header)
    assert "lib_boom.yaml:18" in out                # frame 1: the external file's `code:` line


def test_run_error_multi_frame_three_levels(monkeypatch):
    # e26's origin trace resolves all three levels outermost-to-innermost.
    flow = _ERRORS / "e26-three-level-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "e26-three-level-raise.yaml:37" in out   # frame 0: the top `outer:` call node (header)
    assert "e26-three-level-raise.yaml:29" in out  # frame 1: `via:` call in `middle`
    assert "e26-three-level-raise.yaml:20" in out  # frame 2: `boom` code in `deep`


def test_run_error_multi_frame_on_nested_human_input_failure(monkeypatch):
    # Seed 25 blocks on a HUMAN_INPUT inside a called `defs:` child (`gate/approve`). A host
    # INVALID answer fails the typed write boundary. The origin trace points first
    # at the `gate:` CALL and then the precise child `output:` declaration.
    flow = _SEEDS / "25-nested-suspension.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    with install_ask_user(lambda **request: "maybe"):
        result = run_flow(loaded, {"action": "ship it"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "25-nested-suspension.yaml:41" in out    # frame 0: the `gate:` call node (header)
    assert "25-nested-suspension.yaml:31" in out  # frame 1: precise `output:` line
    assert "is not a member of variant" in out


def test_run_error_unavailable_presentation_stays_unlocated(monkeypatch):
    """A missing resolver cannot be bypassed with CLI graph inference."""
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, replace(loaded, origin_resolver=None))
    out = buf.getvalue()
    assert "╭" not in out
    assert "Source presentation was unavailable." in out
    assert "intentional CODE failure" in out


# --- _ensure_cwd_importable: a flow's `code:` refs to repo-local packages resolve ---- #
def test_ensure_cwd_importable_prepends_cwd(monkeypatch, tmp_path):
    # Without the repo root on sys.path a flow's `code: tests.seeds.fns:fn` fails to import;
    # the CLI prepends the cwd so such refs resolve as `python script.py` would. Run from a
    # dir NOT already on the path and assert it lands at sys.path[0], idempotently.
    import agent_composer.cli.run as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])
    cli._ensure_cwd_importable()
    assert sys.path[0] == str(tmp_path)
    cli._ensure_cwd_importable()                      # idempotent: no duplicate entry
    assert sys.path.count(str(tmp_path)) == 1
