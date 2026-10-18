"""Run the language-neutral conformance cases in `conformance/cases/` against Python yian.

Each case file is one test, named by its id `<group>/<name>`. The test reads the case,
checks it against the closed format, sets up its directory and environment, calls yian
for the case's kind, and compares the results with the case's expectations.
`conformance/README.md` defines the case format and what each kind checks.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError

from yian import defaults
from yian.compile.manifest import flow_fingerprint
from yian.compose import LoadError, load_flow
from yian.compose.helpers.isinstance_types import (
    expression_trees,
    resolve_constructor_type,
    resolve_isinstance_type,
)
from yian.compose.source.scalars import _BOOL, _FLOAT, _INT, _NULL
from yian.diagnostics import Diagnostic, Failure, OriginRef
from yian.expr import (
    constructor_names,
    eval_binding,
    eval_expr,
    isinstance_type_names,
    parse_expr,
    render_template_record,
)
from yian.expr.expressions import ResolveMode, _resolve_in_record, evaluate_when_record
from yian.nodes.tool import node as tool_node
from yian.run import RunResult, resume_flow, run_flow
from yian.runtime.log import JsonlRunLog
from yian.tools.context import ToolContext
from yian.tools.errors import (
    PermanentToolError,
    ReportedToolError,
    ToolCheckError,
    TransientToolError,
)
from yian.tools.function_tools import FunctionTool
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import FileRef, TypeCheckError, build_value_as

_ROOT = Path(__file__).resolve().parent.parent / "conformance"
_CASES = _ROOT / "cases"
_FIXTURES = _ROOT / "fixtures"


class _CaseLoader(yaml.SafeLoader):
    """SafeLoader that reads untagged YAML 1.2 core only and refuses duplicate or non-text keys."""

    # An own table, so SafeLoader's YAML 1.1 resolvers (yes/no, timestamps, merge keys) never
    # apply; add_implicit_resolver below copies nothing from it.
    yaml_implicit_resolvers: dict = {}

    def compose_node(self, parent: yaml.Node | None, index: object) -> yaml.Node:
        """Compose the next node, refusing any tag (`!!str`, `!!timestamp`, `!x`, `!`)."""
        event = self.peek_event()
        # YAML readers disagree on tags (`! 12` is text to some, `!!timestamp` a date), and
        # without tags SafeLoader's constructors see only text the core resolvers matched.
        # An alias event has no tag.
        if getattr(event, "tag", None) is not None:
            raise ComposerError(
                None, None, f"a case file has no tags, found {event.tag!r}", event.start_mark
            )
        return super().compose_node(parent, index)

    def construct_int(self, node: yaml.Node) -> int:
        """Build a core int: `0o` octal, `0x` hex, else decimal (leading zeros included)."""
        text = self.construct_scalar(node)
        if text.startswith(("0o", "0x")):
            return int(text[2:], 8 if text[1] == "o" else 16)
        return int(text)

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict:
        """Build a mapping, refusing a key that isn't text or that appears twice."""
        if not isinstance(node, yaml.MappingNode):
            raise ConstructorError(None, None, "expected a mapping", node.start_mark)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, str):
                raise ConstructorError(
                    None, None, f"mapping key {key!r} is not text", key_node.start_mark
                )
            if key in mapping:
                raise ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


# Inside one first-character bucket int precedes float: the float pattern also matches ints.
_CaseLoader.add_implicit_resolver("tag:yaml.org,2002:bool", _BOOL, list("tTfF"))
_CaseLoader.add_implicit_resolver("tag:yaml.org,2002:int", _INT, list("-+0123456789"))
_CaseLoader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT, list("-+0123456789."))
_CaseLoader.add_implicit_resolver("tag:yaml.org,2002:null", _NULL, ["~", "n", "N", ""])
_CaseLoader.add_constructor("tag:yaml.org,2002:int", _CaseLoader.construct_int)


# The closed format, as conformance/README.md defines it.
_COMMON = {"kind", "rules", "pending", "python_skip", "expect", "python_expect"}
_RUN = {"flow", "python_flow", "files", "env", "inputs", "run", "host"}
_KIND_KEYS = {
    "load": {"flow", "python_flow", "files", "env"},
    "run": _RUN,
    "resume": _RUN | {"resume"},
    "backward": _RUN | {"backward"},
    "expr": {"expr", "template", "values", "item", "mode", "typedefs", "env"},
    "value": {"type", "typedefs", "value", "env"},
    "defaults": {"env"},
    "fingerprint": {"flow", "python_flow", "other", "files", "env"},
    "cli": {"flow", "python_flow", "files", "env", "args", "stdin"},
}
_RUN_EXPECT = {"raises", "output", "trace", "events", "stream", "tool_calls", "memory"}
_EXPECT_KEYS = {
    "load": {"load"},
    "run": _RUN_EXPECT,
    "resume": _RUN_EXPECT,
    "backward": _RUN_EXPECT | {"backward"},
    "expr": {"value", "raises"},
    "value": {"value", "kind", "raises"},
    "defaults": {"defaults", "raises"},
    "fingerprint": {"fingerprint"},
    "cli": {"exit", "stdout", "stderr", "json", "json_lines"},
}
# The keys a kind requires, and the result keys its `expect` requires: one of the groups, plus
# any of the kind's result keys that are in no group.
_REQUIRED = {
    "value": ("type", "value"), "fingerprint": ("flow", "other"), "load": ("flow",),
    "run": ("flow",), "resume": ("flow",), "backward": ("flow",),
}
_RESULTS = {
    "load": (("load",),),
    "defaults": (("defaults",), ("raises",)),
    "value": (("value", "kind"), ("raises",)),
    "expr": (("value",), ("raises",)),
    "fingerprint": (("fingerprint",),),
    "cli": (("exit",),),
}
_ALL_KEYS = _COMMON.union(*_KIND_KEYS.values())
_ALL_EXPECT_KEYS = set().union(*_EXPECT_KEYS.values())
_RUN_KEYS = {"run_id", "num_workers", "learnable", "llm_config", "context"}
_RESUME_KEYS = {"num_workers", "api_key", "headers", "context"}
_BACKWARD_KEYS = {"feedback", "api_key", "headers"}
_HOST_KEYS = {"tools", "human", "jitter", "models", "steering", "stop", "memory", "skills"}
_RESUME_HOST_KEYS = _HOST_KEYS | {"crash_after_record"}
_TOOL_KEYS = {"parameters", "description", "results"}
_RAISE_KEYS = {"class", "message", "retry_after"}
_RULES = {
    "numbers", "arithmetic", "compare", "text-form", "canonical", "columns", "messages",
    "text", "yaml", "dates", "infinity", "llm", "code",
}
_RESERVED = [
    ("host", "models"), ("host", "steering"), ("host", "stop"), ("host", "memory"),
    ("host", "skills"), ("expect", "memory"),
]
# Keys holding text that is not a case value: no codec form, no number checks.
_TEXT_KEYS = {"flow", "python_flow", "other", "files", "stdin"}
_SECRET_KEYS = {"run", "resume", "backward", "env", "host", "stdin"}
_SECRET = re.compile(r"<key:([0-9]+)>")
_FILLED = re.compile(r"sk-conformance-[0-9]+")
_ENV_SUFFIXES = ("_API_KEY", "_BASE_URL", "_ENDPOINT", "_API_VERSION")
_TOOL_ERRORS = {
    error.__name__: error
    for error in (PermanentToolError, TransientToolError, ReportedToolError, ToolCheckError,
                  TimeoutError, ConnectionError)
}


def _read(path: Path) -> dict:
    """Return the case file at `path` as read by `_CaseLoader`."""
    limit = sys.get_int_max_str_digits()
    # Case files have no digit limit; yian applies its own when a value reaches it.
    sys.set_int_max_str_digits(0)
    try:
        case = yaml.load(path.read_text(encoding="utf-8"), Loader=_CaseLoader)
    finally:
        sys.set_int_max_str_digits(limit)
    if not isinstance(case, dict):
        raise ValueError("a case file holds one mapping")
    return case


def _mapping(where: str, value: object) -> dict:
    """Return `value`, raising unless it is a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"`{where or 'the case'}` must be a mapping")
    return value


def _keys(where: str, mapping: object, allowed: set, known: frozenset | set = frozenset(),
          kind: str = "", required: tuple = ()) -> None:
    """Raise unless `mapping` is a mapping with only `allowed` keys and every `required` one.

    A key in `known` but not in `allowed` belongs to another kind of case than `kind`.
    """
    _mapping(where, mapping)
    for key in mapping:
        name = f"{where}.{key}" if where else key
        if key in known and key not in allowed:
            raise ValueError(f"`{name}` is not a key of a `{kind}` case")
        if key not in allowed:
            raise ValueError(f"unknown key `{name}`")
    for key in required:
        if key not in mapping:
            raise ValueError(f"`{where}.{key}` is required" if where else f"`{key}` is required")


def _check_host(host: object, kind: str) -> None:
    """Raise unless `host`, its tools and their scripts use only the format's keys."""
    allowed = _RESUME_HOST_KEYS if kind == "resume" else _HOST_KEYS
    _keys("host", host, allowed, _RESUME_HOST_KEYS, kind)
    tools = _mapping("host.tools", host.get("tools", {}))
    for name, tool in tools.items():
        where = f"host.tools.{name}"
        _keys(where, tool, _TOOL_KEYS, required=("parameters", "results"))
        results = tool["results"]
        scripts = results.items() if isinstance(results, dict) else [(None, results)]
        for node, script in scripts:
            if not isinstance(script, list):
                raise ValueError(f"`{where}.results` must be a list or a mapping of lists")
            label = f"{where}.results" if node is None else f"{where}.results.{node}"
            for index, entry in enumerate(script):
                at = f"{label}[{index}]"
                _keys(at, entry, {"return", "raise"})
                if len(entry) != 1:
                    raise ValueError(f"`{at}` takes `return` or `raise`")
                if "raise" in entry:
                    error = entry["raise"]
                    _keys(f"{at}.raise", error, _RAISE_KEYS, required=("class", "message"))
                    if "retry_after" in error and error["class"] != "TransientToolError":
                        raise ValueError(f"`{at}.raise.retry_after` is for TransientToolError")
    human = _mapping("host.human", host.get("human", {}))
    if any(not isinstance(answers, list) for answers in human.values()):
        raise ValueError("`host.human` maps node ids to lists of answers")
    if "human" in host and "ask_user" in tools:
        raise ValueError("a case with `host.human` declares no `ask_user` tool")
    jitter = host.get("jitter", [])
    if not isinstance(jitter, list) or any(
        type(draw) not in (int, float) or not 0 <= draw < 1 for draw in jitter
    ):
        raise ValueError("`host.jitter` must be a list of numbers in [0, 1)")


def _check(case: dict) -> dict:
    """Return `case` after checking it against the closed format; raise naming the problem."""
    kind = case.get("kind")
    if kind not in _KIND_KEYS:
        raise ValueError(f"`kind` must be one of {', '.join(_KIND_KEYS)}, not {kind!r}")
    _keys("", case, _COMMON | _KIND_KEYS[kind], _ALL_KEYS, kind,
          required=("kind", "expect", *_REQUIRED.get(kind, ())))
    for key in ("expect", "python_expect"):
        if key not in case:
            continue
        _keys(key, case[key], _EXPECT_KEYS[kind], _ALL_EXPECT_KEYS, kind)
        groups = _RESULTS.get(kind, ())
        optional = _EXPECT_KEYS[kind].difference(*groups)
        if groups and not any(set(group) == case[key].keys() - optional for group in groups):
            needs = ", or ".join(" and ".join(f"`{name}`" for name in group) for group in groups)
            raise ValueError(f"`{key}` needs {needs}")
        if kind in ("run", "resume", "backward") and (
            ("raises" in case[key]) == ("output" in case[key])
        ):
            raise ValueError(f"`{key}` takes one of `raises` and `output`")
    if "python_expect" in case and not case.get("rules"):
        raise ValueError("`python_expect` needs `rules`")
    for key, what in (("rules", "rule ids"), ("args", "text")):
        items = case.get(key, [])
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"`{key}` must be a list of {what}")
    unknown = [rule for rule in case.get("rules", []) if rule not in _RULES]
    if unknown:
        raise ValueError(f"unknown rule ids: {', '.join(unknown)}")
    for key in ("env", "files"):
        texts = _mapping(key, case.get(key, {}))
        if any(not isinstance(text, str) for text in texts.values()):
            raise ValueError(f"`{key}` values must be text")
    for key in _TEXT_KEYS - {"files"}:
        if key in case and not isinstance(case[key], str):
            raise ValueError(f"`{key}` must be text")
    if "run" in case:
        _keys("run", case["run"], _RUN_KEYS)
    if "resume" in case:
        _keys("resume", case["resume"], _RESUME_KEYS)
    for key in ("run", "resume"):
        if case.get(key, {}).get("context", "present") != "present":
            raise ValueError(f"`{key}.context` must be `present`")
    if "backward" in case:
        if not isinstance(case["backward"], list):
            raise ValueError("`backward` must be a list")
        for index, entry in enumerate(case["backward"]):
            _keys(f"backward[{index}]", entry, _BACKWARD_KEYS, required=("feedback",))
    if "host" in case:
        _check_host(case["host"], kind)
    run = case.get("run", {})
    if kind == "resume":
        cut = case.get("host", {}).get("crash_after_record")
        if type(cut) is not int or cut < 0:
            raise ValueError("a `resume` case needs `host.crash_after_record`, a record number")
        if run.get("num_workers") != 0:
            raise ValueError("`host.crash_after_record` needs `run.num_workers: 0`")
    if kind == "backward" and run.get("learnable") is not True:
        raise ValueError("a `backward` case needs `run.learnable: true`")
    if kind == "expr":
        if ("expr" in case) == ("template" in case):
            raise ValueError("an `expr` case takes one of `expr` and `template`")
        if not isinstance(case.get("expr", case.get("template")), str):
            raise ValueError("`expr` or `template` must be text")
        mode = case.get("mode", "binding")
        if mode not in ("binding", "condition", "strict"):
            raise ValueError(f"`mode` must be one of binding, condition, strict, not {mode!r}")
        if mode == "condition" and {"template", "item"} & case.keys():
            raise ValueError("`mode: condition` takes an `expr` and no `item`")
        if mode == "strict" and "template" in case and {"item", "typedefs"} & case.keys():
            raise ValueError("a strict `template` takes no `item` or `typedefs`")
    return case


def _decode(value: object) -> object:
    """Return a case value read from yian's JSON codec form; a bare non-finite float is an error."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"a bare {value} is not a case value; write {{$float: inf}} or {{$float: -inf}}"
        )
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value in ({"$float": "inf"}, {"$float": "-inf"}):
        return float(value["$float"])
    if list(value) == ["$record"]:
        record = value["$record"]
        if not isinstance(record, dict):
            raise ValueError("`$record` takes a mapping")
        return {key: _decode(item) for key, item in record.items()}
    return {key: _decode(item) for key, item in value.items()}


def _fill_secrets(value: object) -> object:
    """Return `value` with each `<key:N>` in its text replaced by `sk-conformance-N`."""
    if isinstance(value, str):
        return _SECRET.sub(r"sk-conformance-\1", value)
    if isinstance(value, list):
        return [_fill_secrets(item) for item in value]
    if isinstance(value, dict):
        return {key: _fill_secrets(item) for key, item in value.items()}
    return value


def _prepare(case: dict) -> dict:
    """Return the case as the Python runner runs it.

    Values are decoded, `python_flow` and `python_expect` replace `flow` and `expect`, and
    secrets are filled in.
    """
    case = {key: value if key in _TEXT_KEYS else _decode(value) for key, value in case.items()}
    for key in ("flow", "expect"):
        if f"python_{key}" in case:
            case[key] = case.pop(f"python_{key}")
    return {key: _fill_secrets(value) if key in _SECRET_KEYS else value
            for key, value in case.items()}


def _marks(path: Path) -> list:
    """Return the pytest marks for the case at `path`: a skip when Python doesn't run it."""
    try:
        case = _check(_read(path))
    except Exception:
        # The test reads the file again and fails with the error.
        return []
    if "python_skip" in case:
        return [pytest.mark.skip(reason=str(case["python_skip"]))]
    if case.get("rules") and "python_expect" not in case:
        rules = ", ".join(case["rules"])
        return [pytest.mark.skip(reason=f"depends on TS rules ({rules}) and has no python_expect")]
    return []


def _params() -> list:
    """Return one pytest param per case file, in sorted order."""
    return [
        pytest.param(path, id=f"{path.parent.name}/{path.stem}", marks=_marks(path))
        for path in sorted(_CASES.glob("*/*.yaml"))
    ]


def _set_up(case: dict, case_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write the case directory and, except for `cli`, give yian only the case's variables."""
    if "flow" in case:
        (case_dir / "flow.yaml").write_text(case["flow"], encoding="utf-8")
    for name, text in case.get("files", {}).items():
        path = case_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if _FIXTURES.is_dir():
        shutil.copytree(_FIXTURES, case_dir / "fixtures")
    if case["kind"] == "cli":
        return
    for name in list(os.environ):
        if name.startswith("YIAN_") or name.endswith(_ENV_SUFFIXES):
            monkeypatch.delenv(name)
    monkeypatch.setenv("YIAN_AUTH_FILE", str(case_dir / "auth.json"))
    for name, value in case.get("env", {}).items():
        monkeypatch.setenv(name, value)
    # Forget rather than reload: a bad variable must raise inside the kind, not here.
    defaults._forget()


def _same(actual: object, expected: object, by_value: bool = False,
          optional_message: bool = False) -> bool:
    """Return whether `actual` equals `expected`, kinds included.

    Bools, ints, floats, text and null are distinct kinds; mappings compare by keys and
    values, lists in order. With `by_value`, an int and a float compare by value. With
    `optional_message`, an expected mapping without `message` doesn't compare it.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        if optional_message and "message" not in expected:
            actual = {key: value for key, value in actual.items() if key != "message"}
        return set(actual) == set(expected) and all(
            _same(actual[key], expected[key], by_value, optional_message) for key in expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(a, e, by_value, optional_message) for a, e in zip(actual, expected))
        )
    numbers = (int, float)
    if by_value and type(actual) in numbers and type(expected) in numbers:
        return actual == expected
    return type(actual) is type(expected) and actual == expected


def _dump(value: object) -> str:
    """Return `value` as YAML text for a failure message."""
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=1000)


def _compare(what: str, actual: object, expected: object, **flags: bool) -> None:
    """Fail the case, showing both sides, unless `actual` is the same as `expected`."""
    if not _same(actual, expected, **flags):
        pytest.fail(
            f"{what} differs\n--- actual\n{_dump(actual)}--- expected\n{_dump(expected)}",
            pytrace=False,
        )


def _raised(exc: Exception) -> dict:
    """Return an error in the case's `raises` form."""
    raised = {"class": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, TypeCheckError):
        raised["path"] = list(exc.path)
    return raised


def _expect_raises(expected: dict, call) -> dict:
    """Return the error `call()` raises in `raises` form; fail unless it is the one expected."""
    try:
        call()
    except Exception as exc:
        raised = _raised(exc)
        _compare("raises", raised, expected)
        return raised
    pytest.fail(f"expected {expected.get('class')} to be raised, but nothing was", pytrace=False)


def _load(text: str, case_dir: Path):
    """Load flow text as the case's `flow.yaml`, resolving paths in the case directory."""
    return load_flow(text, search_paths=[case_dir], source_label="flow.yaml")


def _diagnostic(diag: Diagnostic) -> dict:
    """Return `diag` in the case's diagnostic form."""
    frames = []
    for frame in diag.frames:
        span = frame.range
        entry = {
            "label": frame.label,
            "range": [span.start_line, span.start_col, span.end_line, span.end_col],
        }
        if frame.note:
            entry["note"] = frame.note
        frames.append(entry)
    form = {"code": diag.code, "message": diag.message}
    if frames:
        form["frames"] = frames
    if diag.notes:
        form["notes"] = list(diag.notes)
    return form


def _run_load(case: dict, case_dir: Path) -> None:
    """Load `flow` and compare `ok` or its diagnostics."""
    try:
        _load(case["flow"], case_dir)
    except LoadError as exc:
        actual = {"diagnostics": [_diagnostic(exc.diagnostic)]}
    else:
        actual = "ok"
    _compare("load", actual, case["expect"]["load"], optional_message=True)


def _run_defaults(case: dict, case_dir: Path) -> None:
    """Read yian's defaults and compare the listed entries, or the error."""
    expect = case["expect"]
    if "raises" in expect:
        _expect_raises(expect["raises"], defaults.current)
        return
    unknown = set(expect["defaults"]) - {entry.name for entry in defaults.ENTRIES}
    if unknown:
        pytest.fail(f"`expect.defaults` names unknown entries: {', '.join(sorted(unknown))}",
                    pytrace=False)
    values = defaults.current()
    actual = {name: getattr(values, name.lower()) for name in expect["defaults"]}
    _compare("defaults", actual, expect["defaults"])


def _run_fingerprint(case: dict, case_dir: Path) -> None:
    """Check that `flow` fingerprints the same twice, then compare it with `other`'s."""
    def fingerprint(text: str) -> str:
        return flow_fingerprint(_load(text, case_dir).compiled)

    first = fingerprint(case["flow"])
    if fingerprint(case["flow"]) != first:
        pytest.fail("two loads of `flow` have different fingerprints", pytrace=False)
    same = fingerprint(case["other"]) == first
    _compare("fingerprint", "same" if same else "different", case["expect"]["fingerprint"])


_MODES = {"binding": ResolveMode.BINDING_NONE, "strict": ResolveMode.STRICT_RAISE}


def _run_expr(case: dict, case_dir: Path) -> None:
    """Evaluate `expr` or `template` in `mode` over `values` and `item`; compare the result."""
    mode = case.get("mode", "binding")
    template = case.get("template")
    values = case.get("values", {})

    def resolve(path: str) -> object:
        return _resolve_in_record(path, values)

    def evaluate() -> object:
        if template is not None and mode == "strict":
            return render_template_record(template, values)
        if template is not None:
            trees = expression_trees(template)
        elif mode == "condition":
            trees = expression_trees(case["expr"], condition=True)
        else:
            trees = [parse_expr(case["expr"])]
        # The type table the loader gives a node: its isinstance and constructor names.
        registry = read_typedefs(case.get("typedefs") or {})
        types = {}
        for tree in trees:
            for name in isinstance_type_names(tree):
                types[name] = resolve_isinstance_type(name, registry)
            for name in constructor_names(tree):
                types[name] = resolve_constructor_type(name, registry)
        if template is not None:
            return eval_binding(template, resolve, case.get("item"), types=types)
        if mode == "condition":
            # What every `when:`, `while:` and assert runs.
            return evaluate_when_record(case["expr"], values, types)
        return eval_expr(trees[0], resolve, case.get("item"), _MODES[mode], types=types)

    expect = case["expect"]
    if "raises" in expect:
        _expect_raises(expect["raises"], evaluate)
    else:
        _compare("value", evaluate(), expect["value"])


def _run_value(case: dict, case_dir: Path) -> None:
    """Build `value` as `type` and compare the stored value and its kind, or the error."""
    def build():
        typ = type_for(case["type"], read_typedefs(case.get("typedefs") or {}))
        return build_value_as(typ, case["value"])

    expect = case["expect"]
    if "raises" in expect:
        _expect_raises(expect["raises"], build)
        return
    typed = build()
    _compare("value", typed.to_object(), expect["value"])
    _compare("kind", typed.kind.value, expect["kind"])


def _host(case: dict, patch: pytest.MonkeyPatch) -> tuple:
    """Build the case's host and return `(tools, calls, check)`.

    `tools` is the run's tool set: a stub per `host.tools` entry and `ask_user` for
    `host.human`. Each call is recorded in `calls`, by tool, and takes its script's next
    entry; `host.jitter` replaces the TOOL backoff draws. `check()` fails the case when a
    call found its script used up, or when a script has entries left.
    """
    host = case.get("host", {})
    calls: dict = {}
    problems: list = []
    scripts = {f"host.human.{node}": list(answers)
               for node, answers in host.get("human", {}).items()}

    def take(label: str, what: str) -> object:
        if label not in scripts:
            tool, node = label.rsplit(".", 1)
            problems.append(f"`{tool}` has no list for node {node!r}")
        elif not scripts[label]:
            problems.append(f"{what} found `{label}` used up")
        else:
            return scripts[label].pop(0)
        # Permanent, so a TOOL node doesn't retry the overrun.
        raise PermanentToolError(f"`{label}` is used up")

    def record(name: str, args: dict, ctx: ToolContext) -> None:
        calls.setdefault(name, []).append(
            {"args": args, "call_id": ctx.call_id, "attempt": ctx.attempt, "node": ctx.node}
        )

    def stub(name: str, by_node: bool):
        def on_invoke(ctx: ToolContext, args: dict) -> object:
            record(name, args, ctx)
            label = f"host.tools.{name}.results" + (f".{ctx.node}" if by_node else "")
            entry = take(label, f"a `{name}` call from node {ctx.node!r}")
            if "return" in entry:
                return entry["return"]
            error = entry["raise"]
            cls = _TOOL_ERRORS.get(error["class"]) or type(error["class"], (Exception,), {})
            extra = {"retry_after": error["retry_after"]} if "retry_after" in error else {}
            raise cls(error["message"], **extra)
        return on_invoke

    tools = {}
    for name, tool in host.get("tools", {}).items():
        results = tool["results"]
        if isinstance(results, dict):
            scripts.update({f"host.tools.{name}.results.{node}": list(script)
                            for node, script in results.items()})
        else:
            scripts[f"host.tools.{name}.results"] = list(results)
        tools[name] = FunctionTool(name, tool.get("description", ""), tool["parameters"],
                                   stub(name, isinstance(results, dict)))

    def ask_user(ctx: ToolContext, **request: object) -> object:
        record("ask_user", {key: value for key, value in request.items() if value is not None},
               ctx)
        return take(f"host.human.{ctx.node}", f"a question from node {ctx.node!r}")

    if "human" in host:
        tools["ask_user"] = ask_user
    if "jitter" in host:
        scripts["host.jitter"] = list(host["jitter"])
        patch.setattr(tool_node, "_jitter", lambda: take("host.jitter", "a backoff draw"))

    def check() -> None:
        left = [f"`{label}` has {len(script)} {'entry' if len(script) == 1 else 'entries'} left"
                for label, script in scripts.items() if script]
        if problems or left:
            pytest.fail("\n".join(problems + left), pytrace=False)

    return tools, calls, check


def _failure(failure, flow) -> dict:
    """Return a run's `failure` in the case's failure form."""
    return {**_diagnostic(flow.diagnose(failure)), "node": failure.node_id}


def _data(value: object, flow) -> object:
    """Return `value` in data form.

    Dataclasses become mappings of their fields, failures the failure form, files
    `{uri, mime, name}`, tuples lists and sets sorted lists.
    """
    if isinstance(value, Failure):
        return _failure(value, flow)
    if is_dataclass(value):
        return {field.name: _data(getattr(value, field.name), flow) for field in fields(value)}
    if isinstance(value, FileRef):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_data(item, flow) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_data(item, flow) for item in value)
    if isinstance(value, dict):
        return {key: _data(item, flow) for key, item in value.items()}
    return value


def _fields(value: object, flow) -> dict:
    """Return the data form of an event or stream chunk, without its null fields."""
    return {key: item for key, item in _data(value, flow).items() if item is not None}


def _event(event: object, flow) -> list:
    """Return `event` as `[<EventName>]` or `[<EventName>, <data form>]`."""
    data = _fields(event, flow)
    return [type(event).__name__, data] if data else [type(event).__name__]


# The README's trace table: record kind -> (the payload field naming the node, the payload
# fields kept); `None` keeps every field.
_TRACE = {
    "run_started": (None, ("inputs", "learnable", "num_workers", "defaults",
                           "effective_llm_by_static_path", "call_time_llm")),
    "run_resumed": (None, ("retry_node_ids", "num_workers")),
    "node_started": ("node_id", ()),
    "node_step": ("node_id", ("step",)),
    "node_succeeded": ("node_id", ("value", "commit_target", "forward_capture")),
    "node_routed": ("node_id", ("handle", "terminal_failure", "forward_capture")),
    "node_expanded": ("spawner_id", ("seed", "prune_ids", "forward_capture")),
    "node_failed": ("failure", ("failure", "forward_capture")),
    **dict.fromkeys(
        ("memory_commit_requested", "memory_commit_succeeded", "memory_commit_failed"),
        (None, ("memory", "commit_id", "attempt_id", "mutations", "failure")),
    ),
    **dict.fromkeys(("backward_started", "backward_ready", "backward_finished"), (None, None)),
}
_LEFT_OUT_WHEN_EMPTY = {"defaults", "effective_llm_by_static_path", "call_time_llm"}
_FAILURE_FIELDS = {"node_failed": "failure", "node_routed": "terminal_failure"}


def _cut(connection: dict) -> dict:
    """Return a connection record with only its `method`, `source` and `endpoint`."""
    return {name: connection[name] for name in ("method", "source", "endpoint")
            if name in connection}


def _llm(config: dict | None) -> dict | None:
    """Return an LLM config with its `_connection` cut, when it has one."""
    if config is None or config.get("_connection") is None:
        return config
    return {**config, "_connection": _cut(config["_connection"])}


def _capture(capture: dict) -> dict:
    """Return a `forward_capture` with each route's `value_json` text read as JSON.

    Its `node_data.effective_llm_config` has its connection cut.
    """
    def route(node: dict | None) -> dict | None:
        if node is None:
            return None
        return {**node, "value_json": None if node["value_json"] is None
                else json.loads(node["value_json"]),
                "children": [route(child) for child in node["children"]],
                "snapshot": route(node["snapshot"])}

    data = capture["node_data"]
    if data is not None and "effective_llm_config" in data:
        data = {**data, "effective_llm_config": _llm(data["effective_llm_config"])}
    return {**capture, "route": route(capture["route"]), "node_data": data,
            "items": [{**item, "capture": _capture(item["capture"])}
                      for item in capture["items"]]}


def _trace(lines: list, flow) -> list:
    """Return the run log's `lines` projected to the README's trace."""
    def failure(payload: dict) -> dict:
        return _failure(Failure(payload["code"], payload["message"], payload["node_id"],
                                tuple(OriginRef(origin["key"]) for origin in payload["trace"])),
                        flow)

    entries = []
    for line in lines:
        record = json.loads(line)
        kind, payload = record["kind"], dict(record["payload"])
        node_field, kept = _TRACE[kind]
        node = payload[node_field] if node_field else None
        if kind == "node_failed":
            node = node["node_id"]
        if kind == "run_started":
            pool = payload["initial_pool"]
            payload["inputs"] = pool["store"][pool["start_id"]]["value"]
            payload["effective_llm_by_static_path"] = {
                path: _llm(config)
                for path, config in payload["effective_llm_by_static_path"].items()
            }
            call_time = payload.get("call_time_llm")
            if call_time is not None:
                selection = call_time["selection"]
                payload["call_time_llm"] = {**call_time, "selection": {
                    **selection, "connections": {provider: _cut(connection) for provider,
                                                 connection in selection["connections"].items()}
                }}
        if kind == "node_succeeded":
            payload["value"] = payload["value"]["value"]
            if payload["commit_target"] == node:
                del payload["commit_target"]
        key = _FAILURE_FIELDS.get(kind)
        if key is not None and payload[key] is not None:
            payload[key] = failure(payload[key])
        if payload.get("forward_capture") is not None:
            payload["forward_capture"] = _capture(payload["forward_capture"])
        if kind == "node_step":
            step = payload["step"] = dict(payload["step"])
            if "retry_in" in step:
                step["retry_in"] = float(step["retry_in"])
            if "effective_llm_config" in step:
                step["effective_llm_config"] = _llm(step["effective_llm_config"])
        backward = payload["outcome"]["failure"] if kind == "backward_finished" else None
        if backward is not None:
            backward = {key: value for key, value in backward.items()
                        if key not in ("cause_type", "traceback")}
            # A backward failure's `cause` is a node failure when it has a `trace`.
            if isinstance(backward["cause"], dict) and "trace" in backward["cause"]:
                backward["cause"] = failure(backward["cause"])
            payload["outcome"] = {**payload["outcome"], "failure": backward}
        payload = {
            key: value for key, value in payload.items()
            if (kept is None or key in kept) and value is not None
            and not (key in _LEFT_OUT_WHEN_EMPTY and not value)
        }
        entries.append([kind, node, payload] if payload else [kind, node])
    return entries


def _output(result: RunResult, flow) -> dict:
    """Return `result` in the form of `expect.output`."""
    if result.status == "succeeded":
        return {"status": "succeeded", "output": result.output}
    return {"status": result.status, "failure": _failure(result.failure, flow)}


def _check_secrets(case: dict, texts: dict) -> None:
    """Fail the case when a secret it filled in appears in one of `texts`, keyed by where."""
    secrets = set(_FILLED.findall(repr([case.get(key) for key in sorted(_SECRET_KEYS)])))
    leaks = [f"secret {secret} appears in {where}" for where, text in texts.items()
             for secret in sorted(set(_FILLED.findall(text)) & secrets)]
    if leaks:
        pytest.fail("\n".join(leaks), pytrace=False)


def _event_node(event: list) -> str | None:
    """Return an event's node: its `node_id`, a `NodeFailed`'s failure node, else None."""
    data = event[1] if len(event) > 1 else {}
    return data["failure"]["node"] if event[0] == "NodeFailed" else data.get("node_id")


def _group(entries: list, node_of) -> list:
    """Return `entries` with run-level ones (node None) first, then each node's, nodes sorted.

    Each group keeps its order; nodes sort by UTF-16 code units.
    """
    def key(entry: object) -> tuple:
        node = node_of(entry)
        return (node is not None, b"" if node is None else node.encode("utf-16-be"))

    return sorted(entries, key=key)


def _pooled(results: dict) -> dict:
    """Return `results` with the trace, events, stream and each tool's calls grouped by node."""
    return {**results,
            "trace": _group(results["trace"], lambda entry: entry[1]),
            "events": _group(results["events"], _event_node),
            "stream": _group(results["stream"], lambda chunk: chunk.get("node_id")),
            "tool_calls": {tool: _group(calls, lambda call: call["node"])
                           for tool, calls in results["tool_calls"].items()}}


_ID_FIELDS = ("exchange_id", "commit_id", "mutation_id", "note_id")


def _placeholders(results: dict, run_id: str | None) -> dict:
    """Return `results` with each made-up id in their text replaced by `<id:N>`.

    The ids are `run_id` (unless None) and every text value of an `_ID_FIELDS` key. They
    are numbered in order of first appearance, walking `results` in order, lists in order
    and mapping keys sorted by UTF-16 code units.
    """
    ids = set() if run_id is None else {run_id}

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _ID_FIELDS and isinstance(item, str) and item:
                    ids.add(item)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(results)
    if not ids:
        return results
    # Longest first, so an id that holds another is replaced whole.
    pattern = re.compile("|".join(map(re.escape, sorted(ids, key=len, reverse=True))))
    numbers: dict = {}

    def replace(value: object) -> object:
        if isinstance(value, str):
            return pattern.sub(
                lambda match: f"<id:{numbers.setdefault(match[0], len(numbers) + 1)}>", value
            )
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(value[key])
                    for key in sorted(value, key=lambda key: key.encode("utf-16-be"))}
        return value

    return {name: replace(value) for name, value in results.items()}


class _Cut(BaseException):
    """The refused append that cuts a `resume` case's first run."""


def _run_run(case: dict, case_dir: Path) -> None:
    """Run the flow on the case's host; compare `raises` or `output`, then the rest.

    A `resume` case's first run is cut after record `host.crash_after_record`, and the
    flow is resumed from its log; the output, events and stream are the resumed run's. A
    `backward` case then calls backward once per `backward` entry. The rest is the trace,
    events, stream, `tool_calls` and backward outcomes, each when the case expects it.
    Secrets are checked first; a pooled run is grouped, and placeholders put in, before
    anything is compared.
    """
    flow = _load(case["flow"], case_dir)
    expect = case["expect"]
    log = case_dir / "run.jsonl"
    events, chunks, results, outcomes = [], [], [], []
    with pytest.MonkeyPatch.context() as patch:
        tools, calls, check = _host(case, patch)
        run, resume = dict(case.get("run", {})), dict(case.get("resume", {}))
        for arguments in (run, resume):
            if "context" in arguments:
                arguments["context"] = object()
        hooks = {"log": log, "tools": tools, "on_event": events.append,
                 "on_stream": chunks.append}

        def start() -> None:
            results.append(run_flow(flow, case.get("inputs", {}), **hooks, **run))

        if case["kind"] == "resume":
            last = case["host"]["crash_after_record"]
            append = JsonlRunLog.append

            def refuse(self: JsonlRunLog, record: object) -> object:
                if self._next_sequence > last:
                    raise _Cut
                return append(self, record)

            with pytest.MonkeyPatch.context() as cut:
                cut.setattr(JsonlRunLog, "append", refuse)
                try:
                    start()
                except _Cut:
                    pass
                else:
                    pytest.fail(f"the run ended before record {last + 1}", pytrace=False)
            events.clear()
            chunks.clear()

            def start() -> None:
                results.append(resume_flow(flow, **hooks, **resume))

        raised = {}
        if "raises" in expect:
            raised = _expect_raises(expect["raises"], start)
        else:
            start()
        for entry in case.get("backward", []) if results else []:
            try:
                results[0].backward(entry["feedback"], api_key=entry.get("api_key"),
                                    headers=entry.get("headers"))
            except Exception as exc:
                outcomes.append({"raises": _raised(exc)})
                continue
            trace = _trace(log.read_text(encoding="utf-8").splitlines(), flow)
            finished = [record for record in trace if record[0] == "backward_finished"]
            outcomes.append(finished[-1][2]["outcome"])
        text = log.read_text(encoding="utf-8") if log.exists() else ""
        lines = text.splitlines()
        actual = {
            "trace": _trace(lines, flow),
            "events": [_event(event, flow) for event in events],
            "stream": [_fields(chunk, flow) for chunk in chunks],
            "tool_calls": calls,
            "output": _output(results[0], flow) if results else None,
            "backward": outcomes,
        }
        _check_secrets(case, {**{name: repr(value) for name, value in actual.items()},
                              "the run log": text, "the raised error": raised.get("message", "")})
        if {**run, **resume}.get("num_workers", 8) != 0:
            actual = _pooled(actual)
        run_id = None if "run_id" in run or not lines else json.loads(lines[0])["run_id"]
        actual = _placeholders(actual, run_id)
        if "output" in expect:
            _compare("output", actual["output"], expect["output"],
                     by_value=actual["output"]["status"] == "succeeded")
        for name in ("trace", "events", "stream", "tool_calls", "backward"):
            if name in expect:
                # Backward outcomes are projected like the trace, so they keep the kinds.
                _compare(name, actual[name], expect[name],
                         by_value=name not in ("trace", "backward"))
        # Last: a leftover entry usually follows from a divergence the comparisons show.
        check()


def _run_cli(case: dict, case_dir: Path) -> None:
    """Run `yian` with `args` in the case directory; compare its exit code and output."""
    (case_dir / "home").mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(case_dir / "home"), "COLUMNS": "1000",
           "PYTHONPATH": os.pathsep.join(sys.path), **case.get("env", {})}
    command = "import sys; sys.argv[0] = 'yian'; from yian.cli import main; main()"
    done = subprocess.run(
        [sys.executable, "-c", command, *case.get("args", [])],
        cwd=case_dir, input=case.get("stdin", ""), capture_output=True, encoding="utf-8",
        timeout=120, env=env,
    )

    def normal(text: str) -> str:
        # Longest first: the resolved path can hold the given one (`/private/var` on macOS).
        for path in sorted({str(case_dir), str(case_dir.resolve())}, key=len, reverse=True):
            text = text.replace(path, "<tmp>")
        return re.sub(r" +$", "", text, flags=re.MULTILINE)

    def refuse(token: str) -> None:
        raise ValueError(f"`{token}` in stdout is not JSON")

    def document(text: str) -> object:
        return _decode(json.loads(text, parse_constant=refuse))

    stdout, stderr = normal(done.stdout), normal(done.stderr)
    _check_secrets(case, {"stdout": stdout, "stderr": stderr, **{
        str(path.relative_to(case_dir)): path.read_text(encoding="utf-8")
        for path in case_dir.rglob("*.jsonl")
    }})
    expect = case["expect"]
    if not _same(done.returncode, expect["exit"]):
        pytest.fail(f"exit {done.returncode}, expected {expect['exit']}\n--- stderr\n{stderr}",
                    pytrace=False)
    for name, text in (("stdout", stdout), ("stderr", stderr)):
        if name in expect:
            _compare(name, text, expect[name])
    if "json" in expect:
        _compare("json", document(stdout), expect["json"], optional_message=True)
    if "json_lines" in expect:
        _compare("json_lines", [document(line) for line in stdout.splitlines() if line.strip()],
                 expect["json_lines"], optional_message=True)


_KINDS = {
    "load": _run_load,
    "run": _run_run,
    "resume": _run_run,
    "backward": _run_run,
    "defaults": _run_defaults,
    "fingerprint": _run_fingerprint,
    "expr": _run_expr,
    "value": _run_value,
    "cli": _run_cli,
}


@pytest.mark.parametrize("path", _params())
def test_case(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one conformance case and compare its results with its expectations."""
    case = _prepare(_check(_read(path)))
    for section, key in _RESERVED:
        if key in case.get(section, {}):
            pytest.fail(
                f"`{section}.{key}` is not supported by the Python runner yet", pytrace=False
            )
    run = _KINDS.get(case["kind"])
    if run is None:
        pytest.fail(f"kind `{case['kind']}` is not run by the Python runner yet", pytrace=False)
    _set_up(case, tmp_path, monkeypatch)
    run(case, tmp_path)
