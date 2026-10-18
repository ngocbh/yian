"""Run the language-neutral conformance cases in `conformance/cases/` against Python yian.

Each case file is one test, named by its id `<group>/<name>`. `conformance/README.md` defines
the case format and what each kind checks.
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
from typing import NoReturn

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

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Build a mapping, refusing a key that isn't text or that appears twice."""
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
_FLOW = {"flow", "python_flow", "files", "python_files", "env"}
_RUN = _FLOW | {"inputs", "run", "host"}
_KIND_KEYS = {
    "load": _FLOW,
    "run": _RUN,
    "resume": _RUN | {"resume"},
    "backward": _RUN | {"backward"},
    "expr": {"expr", "template", "values", "item", "mode", "typedefs", "env"},
    "value": {"type", "typedefs", "value", "env"},
    "defaults": {"env"},
    # Both runners load `flow` and `other` as written: no `python_flow` or `python_files`.
    "fingerprint": _FLOW - {"python_flow", "python_files"} | {"other"},
    "cli": _FLOW | {"args", "stdin"},
}
_REQUIRED = {
    "load": ("flow",), "run": ("flow",), "resume": ("flow",), "backward": ("flow", "backward"),
    "value": ("type", "value"), "fingerprint": ("flow", "other"),
}
# Each kind's `expect` holds exactly one of its result groups, and any of its optional keys.
_RESULTS = {
    "load": (("load",),),
    "run": (("raises",), ("output",)),
    "resume": (("raises",), ("output",)),
    "backward": (("raises",), ("output",)),
    "expr": (("value",), ("raises",)),
    "value": (("value", "kind"), ("raises",)),
    "defaults": (("defaults",), ("raises",)),
    "fingerprint": (("fingerprint",),),
    "cli": (("exit",),),
}
_RUN_OPTIONAL = {"trace", "events", "stream", "tool_calls", "memory"}
_OPTIONAL = {
    "run": _RUN_OPTIONAL, "resume": _RUN_OPTIONAL, "backward": _RUN_OPTIONAL | {"backward"},
    "cli": {"stdout", "stderr", "json", "json_lines"},
}
_EXPECT_KEYS = {kind: _OPTIONAL.get(kind, set()).union(*groups)
                for kind, groups in _RESULTS.items()}
_ALL_KEYS = _COMMON.union(*_KIND_KEYS.values())
_ALL_EXPECT_KEYS = set().union(*_EXPECT_KEYS.values())
_RUN_KEYS = {"run_id", "num_workers", "learnable", "llm_config", "context"}
_RESUME_KEYS = {"num_workers", "api_key", "headers", "context"}
# Host keys the format reserves and this runner doesn't support yet.
_RESERVED_HOST_KEYS = ("models", "steering", "stop", "memory", "skills")
_HOST_KEYS = {"tools", "human", "jitter", *_RESERVED_HOST_KEYS}
_RESUME_HOST_KEYS = _HOST_KEYS | {"crash_after_record"}
_RULES = {
    "numbers", "arithmetic", "compare", "text-form", "canonical", "columns", "messages",
    "text", "yaml", "dates", "infinity", "llm", "code",
}
_RESERVED = [*(("host", key) for key in _RESERVED_HOST_KEYS), ("expect", "memory")]
# Keys holding text that is not a case value: no codec form, no number checks.
_TEXT_KEYS = {"flow", "python_flow", "other", "files", "python_files", "stdin"}
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
    def name(key: str) -> str:
        return f"{where}.{key}" if where else key

    for key in _mapping(where, mapping):
        if key in known and key not in allowed:
            raise ValueError(f"`{name(key)}` is not a key of a `{kind}` case")
        if key not in allowed:
            raise ValueError(f"unknown key `{name(key)}`")
    for key in required:
        if key not in mapping:
            raise ValueError(f"`{name(key)}` is required")


def _check_host(host: object, kind: str) -> None:
    """Raise unless `host`, its tools and their scripts use only the format's keys."""
    allowed = _RESUME_HOST_KEYS if kind == "resume" else _HOST_KEYS
    _keys("host", host, allowed, _RESUME_HOST_KEYS, kind)
    tools = _mapping("host.tools", host.get("tools", {}))
    for name, tool in tools.items():
        where = f"host.tools.{name}"
        _keys(where, tool, {"parameters", "description", "results"},
              required=("parameters", "results"))
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
                    _keys(f"{at}.raise", error, {"class", "message", "retry_after"},
                          required=("class", "message"))
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
        groups = _RESULTS[kind]
        if not any(set(group) == case[key].keys() - _OPTIONAL.get(kind, set())
                   for group in groups):
            needs = ", or ".join(" and ".join(f"`{name}`" for name in group) for group in groups)
            raise ValueError(f"`{key}` needs {needs}")
    if "python_expect" in case and not case.get("rules"):
        raise ValueError("`python_expect` needs `rules`")
    for key, what in (("rules", "rule ids"), ("args", "text")):
        items = case.get(key, [])
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"`{key}` must be a list of {what}")
    unknown = [rule for rule in case.get("rules", []) if rule not in _RULES]
    if unknown:
        raise ValueError(f"unknown rule ids: {', '.join(unknown)}")
    for key in ("env", "files", "python_files"):
        if any(not isinstance(text, str) for text in _mapping(key, case.get(key, {})).values()):
            raise ValueError(f"`{key}` values must be text")
    for key in _TEXT_KEYS - {"files", "python_files"}:
        if not isinstance(case.get(key, ""), str):
            raise ValueError(f"`{key}` must be text")
    for key, allowed in (("run", _RUN_KEYS), ("resume", _RESUME_KEYS)):
        if key in case:
            _keys(key, case[key], allowed)
            if case[key].get("context", "present") != "present":
                raise ValueError(f"`{key}.context` must be `present`")
    if "backward" in case:
        if not isinstance(case["backward"], list) or not case["backward"]:
            raise ValueError("`backward` must be a non-empty list")
        for index, entry in enumerate(case["backward"]):
            _keys(f"backward[{index}]", entry, {"feedback", "api_key", "headers"},
                  required=("feedback",))
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
        value = value["$record"]
        if not isinstance(value, dict):
            raise ValueError("`$record` takes a mapping")
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

    Values are decoded, `python_flow`, `python_files` and `python_expect` replace `flow`,
    `files` and `expect`, and secrets are filled in.
    """
    case = {key: value if key in _TEXT_KEYS else _decode(value) for key, value in case.items()}
    for key in ("flow", "files", "expect"):
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


def _fail(message: str) -> NoReturn:
    """Fail the case with `message` and no traceback."""
    pytest.fail(message, pytrace=False)


def _same(actual: object, expected: object, by_value: bool = False,
          optional_message: bool = False) -> bool:
    """Return whether `actual` equals `expected`, kinds included.

    With `by_value`, an int and a float compare by value. With `optional_message`, an
    expected mapping without `message` doesn't compare it.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        if optional_message and "message" not in expected:
            actual = {key: value for key, value in actual.items() if key != "message"}
        return actual.keys() == expected.keys() and all(
            _same(actual[key], expected[key], by_value, optional_message) for key in expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same(a, e, by_value, optional_message) for a, e in zip(actual, expected)
        )
    if by_value and type(actual) in (int, float) and type(expected) in (int, float):
        return actual == expected
    return type(actual) is type(expected) and actual == expected


def _compare(what: str, actual: object, expected: object, **flags: bool) -> None:
    """Fail the case, showing both sides as YAML, unless `actual` is the same as `expected`."""
    if not _same(actual, expected, **flags):
        dump = [yaml.safe_dump(side, sort_keys=False, allow_unicode=True, width=1000)
                for side in (actual, expected)]
        _fail(f"{what} differs\n--- actual\n{dump[0]}--- expected\n{dump[1]}")


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
    _fail(f"expected {expected.get('class')} to be raised, but nothing was")


def _results(expect: dict, compute) -> None:
    """Compare the results `compute()` returns, by name, with `expect`.

    When `expect` has `raises`, compare the error `compute()` raises instead.
    """
    if "raises" in expect:
        _expect_raises(expect["raises"], compute)
        return
    for name, actual in compute().items():
        _compare(name, actual, expect[name])


def _load(text: str, case_dir: Path):
    """Load flow text as the case's `flow.yaml`, resolving paths in the case directory."""
    return load_flow(text, search_paths=[case_dir], source_label="flow.yaml")


def _diagnostic(diag: Diagnostic) -> dict:
    """Return `diag` in the case's diagnostic form."""
    frames = []
    for frame in diag.frames:
        span = frame.range
        frames.append({"label": frame.label,
                       "range": [span.start_line, span.start_col, span.end_line, span.end_col],
                       **({"note": frame.note} if frame.note else {})})
    form = {"code": diag.code, "message": diag.message, "frames": frames, "notes": list(diag.notes)}
    return {key: value for key, value in form.items() if value or key in ("code", "message")}


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
    names = case["expect"].get("defaults", {})

    def compute() -> dict:
        unknown = set(names) - {entry.name for entry in defaults.ENTRIES}
        if unknown:
            _fail(f"`expect.defaults` names unknown entries: {', '.join(sorted(unknown))}")
        values = defaults.current()
        return {"defaults": {name: getattr(values, name.lower()) for name in names}}

    _results(case["expect"], compute)


def _run_fingerprint(case: dict, case_dir: Path) -> None:
    """Check that `flow` fingerprints the same twice, then compare it with `other`'s."""
    def fingerprint(text: str) -> str:
        return flow_fingerprint(_load(text, case_dir).compiled)

    first = fingerprint(case["flow"])
    if fingerprint(case["flow"]) != first:
        _fail("two loads of `flow` have different fingerprints")
    same = fingerprint(case["other"]) == first
    _compare("fingerprint", "same" if same else "different", case["expect"]["fingerprint"])


def _run_expr(case: dict, case_dir: Path) -> None:
    """Evaluate `expr` or `template` in `mode` over `values` and `item`; compare the result."""
    mode, template = case.get("mode", "binding"), case.get("template")
    values, item = case.get("values", {}), case.get("item")

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
            types |= {name: resolve_isinstance_type(name, registry)
                      for name in isinstance_type_names(tree)}
            types |= {name: resolve_constructor_type(name, registry)
                      for name in constructor_names(tree)}
        if template is not None:
            return eval_binding(template, resolve, item, types=types)
        if mode == "condition":
            # What every `when:`, `while:` and assert runs.
            return evaluate_when_record(case["expr"], values, types)
        resolve_mode = ResolveMode.STRICT_RAISE if mode == "strict" else ResolveMode.BINDING_NONE
        return eval_expr(trees[0], resolve, item, resolve_mode, types=types)

    _results(case["expect"], lambda: {"value": evaluate()})


def _run_value(case: dict, case_dir: Path) -> None:
    """Build `value` as `type` and compare the stored value and its kind, or the error."""
    def compute() -> dict:
        typ = type_for(case["type"], read_typedefs(case.get("typedefs") or {}))
        typed = build_value_as(typ, case["value"])
        return {"value": typed.to_object(), "kind": typed.kind.value}

    _results(case["expect"], compute)


def _host(case: dict, patch: pytest.MonkeyPatch) -> tuple:
    """Build the case's host and return `(tools, calls, check)`.

    `tools` holds a stub per `host.tools` entry and `ask_user` for `host.human`; each call is
    recorded in `calls`, by tool, and takes its script's next entry, as each TOOL backoff takes
    the next `host.jitter` draw. `check()` fails the case when a script ran out, and
    `check(left=True)` also when one has entries left.
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

    def call(name: str, label: str, ctx: ToolContext, args: dict) -> object:
        calls.setdefault(name, []).append(
            {"args": args, "call_id": ctx.call_id, "attempt": ctx.attempt, "node": ctx.node}
        )
        return take(label, f"a call to `{name}` from node {ctx.node!r}")

    def stub(name: str, by_node: bool):
        def on_invoke(ctx: ToolContext, args: dict) -> object:
            label = f"host.tools.{name}.results" + (f".{ctx.node}" if by_node else "")
            entry = call(name, label, ctx, args)
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
        request = {key: value for key, value in request.items() if value is not None}
        return call("ask_user", f"host.human.{ctx.node}", ctx, request)

    if "human" in host:
        tools["ask_user"] = ask_user
    if "jitter" in host:
        scripts["host.jitter"] = list(host["jitter"])
        patch.setattr(tool_node, "_jitter", lambda: take("host.jitter", "a backoff draw"))

    def check(left: bool = False) -> None:
        leftovers = [
            f"`{label}` has {len(script)} {'entry' if len(script) == 1 else 'entries'} left"
            for label, script in scripts.items() if left and script
        ]
        if problems or leftovers:
            _fail("\n".join(problems + leftovers))

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
        return sorted((_data(item, flow) for item in value),
                      key=lambda item: _utf16(item) if isinstance(item, str) else item)
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
# fields kept).
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
    "backward_started": (None, ("attempt_id", "feedback")),
    "backward_ready": (None, ("attempt_id", "root_result", "attributed_results", "proposals")),
    "backward_finished": (None, ("attempt_id", "outcome", "attributed_results")),
}
_LEFT_OUT_WHEN_EMPTY = {"defaults", "effective_llm_by_static_path", "call_time_llm"}
_CAPTURE_LEFT_OUT_WHEN_EMPTY = {"evaluated_keys", "items", "node_data", "input_names",
                                "variable_names"}
_FAILURE_FIELDS = {"node_failed": "failure", "node_routed": "terminal_failure"}


def _cut(connection: dict) -> dict:
    """Return a connection record with only its `method`, `source` and `endpoint`."""
    return {name: connection[name] for name in ("method", "source", "endpoint")
            if name in connection}


def _llm(config: dict) -> dict:
    """Return an agent's LLM config projected to the trace.

    Its `llm_config` fields (keys not starting with `_`), `_header_names` and `_host_key` stay,
    and its `_connection` is cut; any other `_` key is dropped.
    """
    kept = {key: value for key, value in config.items()
            if not key.startswith("_") or key in ("_header_names", "_host_key")}
    if config.get("_connection") is not None:
        kept["_connection"] = _cut(config["_connection"])
    return kept


def _capture(capture: dict) -> dict:
    """Return a `forward_capture` projected to the trace, changing it in place."""
    def route(node: dict) -> dict:
        # `expression` and `result_type` are text each implementation spells its own way.
        lean = {key: value for key, value in node.items()
                if key not in ("expression", "result_type")
                and (key == "kind" or value not in (None, [], 0, ""))}
        if "value_json" in lean:
            lean["value_json"] = json.loads(lean["value_json"])
        if "children" in lean:
            lean["children"] = [route(child) for child in lean["children"]]
        if "snapshot" in lean:
            lean["snapshot"] = route(lean["snapshot"])
        return lean

    data = capture["node_data"]
    if data and data.get("effective_llm_config") is not None:
        data["effective_llm_config"] = _llm(data["effective_llm_config"])
    capture["route"] = route(capture["route"])
    for item in capture["items"]:
        item["capture"] = _capture(item["capture"])
    return {key: value for key, value in capture.items()
            if value or key not in _CAPTURE_LEFT_OUT_WHEN_EMPTY}


def _trace(lines: list, flow) -> list:
    """Return the run log's `lines` projected to the README's trace."""
    def failure(payload: dict) -> dict:
        return _failure(Failure(payload["code"], payload["message"], payload["node_id"],
                                tuple(OriginRef(origin["key"]) for origin in payload["trace"])),
                        flow)

    entries = []
    for line in lines:
        # Each record is parsed afresh, so its payload is projected in place.
        record = json.loads(line)
        kind, payload = record["kind"], record["payload"]
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
            if payload.get("call_time_llm") is not None:
                # `selection` is internal; its effect is each agent's `effective_llm_config` step.
                agents = payload["call_time_llm"]["agents"]
                payload["call_time_llm"] = {
                    "agents": {path: _llm(config) for path, config in agents.items()}
                }
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
            step = payload["step"]
            if "retry_in" in step:
                step["retry_in"] = float(step["retry_in"])
            if step.get("effective_llm_config") is not None:
                step["effective_llm_config"] = _llm(step["effective_llm_config"])
        backward = payload["outcome"]["failure"] if kind == "backward_finished" else None
        if backward is not None:
            backward.pop("cause_type", None)
            backward.pop("traceback", None)
            # A backward failure's `cause` is a node failure when it has a `trace`.
            if isinstance(backward["cause"], dict) and "trace" in backward["cause"]:
                backward["cause"] = failure(backward["cause"])
        payload = {
            key: value for key, value in payload.items()
            if key in kept and value is not None
            and not (key in _LEFT_OUT_WHEN_EMPTY and not value)
        }
        entries.append([kind, node, payload] if payload else [kind, node])
    return entries


def _output(result: RunResult, flow) -> dict:
    """Return `result` in the form of `expect.output`."""
    if result.status == "succeeded":
        return {"status": "succeeded", "output": _data(result.output, flow)}
    return {"status": result.status, "failure": _failure(result.failure, flow)}


def _check_secrets(case: dict, texts: dict) -> None:
    """Fail the case when a secret it filled in appears in one of `texts`, keyed by where."""
    secrets = set(_FILLED.findall(repr([case.get(key) for key in sorted(_SECRET_KEYS)])))
    leaks = [f"secret {secret} appears in {where}" for where, text in texts.items()
             for secret in sorted(set(_FILLED.findall(text)) & secrets)]
    if leaks:
        _fail("\n".join(leaks))


def _utf16(text: str) -> bytes:
    """Return the sort key that orders text by UTF-16 code units."""
    return text.encode("utf-16-be")


def _pooled(results: dict) -> dict:
    """Return `results` with the trace, events, stream and each tool's calls grouped by node.

    Run-level entries (node None) come first, then each node's, nodes sorted; each group keeps
    its order.
    """
    def group(entries: list, node_of) -> list:
        return sorted(entries, key=lambda entry: (node_of(entry) is not None,
                                                  _utf16(node_of(entry) or "")))

    def event_node(event: list) -> str | None:
        # A `NodeFailed` belongs to its failure's node; run events have no `node_id`.
        data = event[1] if len(event) > 1 else {}
        return data["failure"]["node"] if event[0] == "NodeFailed" else data.get("node_id")

    return {**results,
            "trace": group(results["trace"], lambda entry: entry[1]),
            "events": group(results["events"], event_node),
            "stream": group(results["stream"], lambda chunk: chunk.get("node_id")),
            "tool_calls": {tool: group(calls, lambda call: call["node"])
                           for tool, calls in results["tool_calls"].items()}}


_ID_FIELDS = ("exchange_id", "commit_id", "mutation_id", "note_id")


def _placeholders(results: dict, run_id: str | None) -> dict:
    """Return `results` with each made-up id in their text replaced by `<id:N>`.

    The ids are `run_id` (unless None) and every text value of an `_ID_FIELDS` key, numbered
    in order of first appearance, with mapping keys in sorted order.
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
            return {key: replace(value[key]) for key in sorted(value, key=_utf16)}
        return value

    return {name: replace(value) for name, value in results.items()}


class _Cut(BaseException):
    """The refused append that cuts a `resume` case's first run."""


def _run_run(case: dict, case_dir: Path) -> None:
    """Run the flow on the case's host and compare the results the case expects.

    A `resume` case's first run is cut after record `host.crash_after_record` and resumed; a
    `backward` case then calls backward once per `backward` entry.
    """
    flow = _load(case["flow"], case_dir)
    expect = case["expect"]
    log = case_dir / "run.jsonl"
    events, chunks = [], []
    with pytest.MonkeyPatch.context() as patch:
        tools, calls, check = _host(case, patch)
        run, resume = dict(case.get("run", {})), dict(case.get("resume", {}))
        for arguments in (run, resume):
            if "context" in arguments:
                arguments["context"] = object()
        hooks = {"log": log, "tools": tools, "on_event": events.append,
                 "on_stream": chunks.append}

        def start() -> RunResult:
            return run_flow(flow, case.get("inputs", {}), **hooks, **run)

        if case["kind"] == "resume":
            last = case["host"]["crash_after_record"]
            append = JsonlRunLog.append

            def refuse(self: JsonlRunLog, record: object) -> object:
                if self.next_sequence > last:
                    raise _Cut
                return append(self, record)

            with pytest.MonkeyPatch.context() as cut:
                cut.setattr(JsonlRunLog, "append", refuse)
                try:
                    start()
                except _Cut:
                    pass
                else:
                    _fail(f"the run ended before record {last + 1}")
            events.clear()
            chunks.clear()

            def start() -> RunResult:
                return resume_flow(flow, **hooks, **resume)

        result, raised = None, {}
        if "raises" in expect:
            raised = _expect_raises(expect["raises"], start)
        else:
            result = start()

        def finished() -> list:
            trace = _trace(log.read_text(encoding="utf-8").splitlines(), flow)
            return [entry for entry in trace if entry[0] == "backward_finished"]

        outcomes = []
        for index, entry in enumerate(case.get("backward", []) if result is not None else []):
            before = len(finished())
            try:
                result.backward(entry["feedback"], api_key=entry.get("api_key"),
                                headers=entry.get("headers"))
            except Exception as exc:
                outcomes.append({"raises": _raised(exc)})
                continue
            after = finished()
            if len(after) != before + 1:
                _fail(f"backward call {index + 1} appended {len(after) - before} "
                      "`backward_finished` records, not one")
            outcomes.append(after[-1][2]["outcome"])
        # A call past a script's end is the case's own error, whatever it made differ.
        check()
        text = log.read_text(encoding="utf-8") if log.exists() else ""
        lines = text.splitlines()
        actual = {
            "trace": _trace(lines, flow),
            "events": [_event(event, flow) for event in events],
            "stream": [_fields(chunk, flow) for chunk in chunks],
            "tool_calls": calls,
            "output": _output(result, flow) if result is not None else None,
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
        check(left=True)


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
        raise ValueError(f"{token} is not JSON")

    def document(text: str) -> object:
        try:
            return _decode(json.loads(text, parse_constant=refuse))
        except ValueError as error:  # JSONDecodeError is a ValueError
            _fail(f"stdout is not a JSON document ({error})\n--- stdout\n{stdout}")

    stdout, stderr = normal(done.stdout), normal(done.stderr)
    _check_secrets(case, {"stdout": stdout, "stderr": stderr, **{
        str(path.relative_to(case_dir)): path.read_text(encoding="utf-8")
        for path in case_dir.rglob("*.jsonl")
    }})
    expect = case["expect"]
    if not _same(done.returncode, expect["exit"]):
        _fail(f"exit {done.returncode}, expected {expect['exit']}\n--- stdout\n{stdout}"
              f"--- stderr\n{stderr}")
    for name, text in (("stdout", stdout), ("stderr", stderr)):
        if name in expect and text != expect[name]:
            _fail(f"{name} differs\n--- actual\n{text}--- expected\n{expect[name]}")
    if "json" in expect:
        _compare("json", document(stdout), expect["json"], optional_message=True)
    if "json_lines" in expect:
        _compare("json_lines", [document(line) for line in stdout.splitlines() if line.strip()],
                 expect["json_lines"], optional_message=True)


_KINDS = {
    "load": _run_load, "run": _run_run, "resume": _run_run, "backward": _run_run,
    "defaults": _run_defaults, "fingerprint": _run_fingerprint, "expr": _run_expr,
    "value": _run_value, "cli": _run_cli,
}


@pytest.mark.parametrize("path", _params())
def test_case(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one conformance case and compare its results with its expectations."""
    case = _prepare(_check(_read(path)))
    for section, key in _RESERVED:
        if key in case.get(section, {}):
            _fail(f"`{section}.{key}` is not supported by the Python runner yet")
    _set_up(case, tmp_path, monkeypatch)
    _KINDS[case["kind"]](case, tmp_path)
