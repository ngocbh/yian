"""Record conformance case candidates while the Python test suite runs.

A pytest plugin: while each test runs, it records the engine calls whose inputs can be written as
data (flow loads, runs, resumes, backward passes, expressions, templates, values and defaults).
The model calls a run makes through `llm_clients.model_from_config`, whatever a test patched it
with, become the run's `host.models`. Each passing test's calls become candidates; a candidate is
replayed through the conformance runner (`tests/test_conformance.py`), plain and with every TS-rule
shim on (three times each for a pooled run). The shimmed outcome is its `expect`; when that
differs, the plain one is its `python_expect` and the rules whose shims change it are its `rules`.
A rule no shim can follow (huge ints, an int in a float slot, infinity, dates, YAML, structured
model calls, CODE, some text rules) is added to `rules` by what the case holds, with
`python_expect` even when equal, and flagged `needs TS expect: <rule>`: its TS expectation is
written by hand. Before the replays, its CODE takes its TS form, keeping the Python form in
`python_flow` and `python_files`: references to ported modules point at `conformance/fixtures/`,
and inline bodies take the translations `--harvest-code FILE` gives. It is written as a case
file under `<out>/cases/<group>/<name>.yaml`, deduplicated by content. Run it with:

    PYTHONPATH=src python -m pytest -p tests.conformance_harvest --harvest-out DIR tests

`DIR/worklist.yaml` lists what is left for hand work: the CODE bodies to translate, the
references to port, and the cases needing a TS expectation.

Shards of the suite run one after another into one `DIR`: deduplication and reports are per
process, so neither shards running at once nor `-n` (xdist) are supported.

Design section 4.4 explains the harvest; `conformance/README.md` defines the case format.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection
import dataclasses
from decimal import Decimal
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import threading
from types import ModuleType
from typing import NamedTuple

from langchain_core.messages import AIMessage, AIMessageChunk
from lark import Tree
from pydantic import BaseModel
import pytest
import yaml

import yian
from yian import defaults, llm_clients
from yian.compose import LoadError, load_flow
from yian.compose.source.scalars import _FlowLoader
from yian.expr import builtins as expr_builtins, expressions, template
from yian.expr.expressions import ResolveMode, eval_expr, evaluate_when_record
from yian.expr.grammar import parse_expr
from yian.expr.template import eval_binding, render_template_record
from yian.nodes.agent import live, memory
from yian.nodes.code.node import _REFERENCE_RE, CodeNode
from yian.nodes.tool import node as tool_node
from yian.run import RunResult, resume_flow, run_flow
from yian.typesys import values as typesys_values
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import (
    FileRef,
    ValueKind,
    build_value_as,
    long_int_digits,
    shown_text,
    value_repr,
    write_as_text,
)

from tests import test_conformance as runner

# The order a case file's keys are written in. Other keys (the harvest writes none) follow in
# their given order.
_ORDER = ("kind", "rules", "flow", "python_flow", "files", "python_files", "env", "inputs",
          "run", "host", "backward", "expr", "template", "values", "item", "mode", "typedefs",
          "type", "value", "expect", "python_expect")
# The order of an `expect`'s (and `python_expect`'s) keys, and of a run's `output`: the result
# before what led to it.
_EXPECT_ORDER = ("raises", "output", "load", "value", "kind", "defaults", "trace", "events",
                 "stream", "tool_calls", "backward")
_OUTPUT_ORDER = ("status", "output", "failure")
# A list of one-line scalars whose texts add up to at most this many characters is written in
# flow style, as `[12, 3, 12, 8]`.
_FLOW_LIMIT = 60
_NAME_LIMIT = 150


def _encode(value: object) -> object:
    """Return a case value in yian's JSON codec form: the inverse of the runner's `_decode`."""
    if isinstance(value, float) and math.isinf(value):
        return {"$float": "inf" if value > 0 else "-inf"}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if not isinstance(value, dict):
        return value
    encoded = {key: _encode(item) for key, item in value.items()}
    # A record the reader would take for codec form goes inside `$record`.
    if value in ({"$float": "inf"}, {"$float": "-inf"}) or list(value) == ["$record"]:
        return {"$record": encoded}
    return encoded


class _CaseDumper(yaml.SafeDumper):
    """SafeDumper writing what `_CaseLoader` reads back: no aliases, multi-line text as blocks."""

    # The reader's table, so text the reader would resolve to another kind (`0o17`, `1e3`,
    # `Null`) is quoted, and text only YAML 1.1 resolves (`yes`) is not.
    yaml_implicit_resolvers = runner._CaseLoader.yaml_implicit_resolvers

    def ignore_aliases(self, data: object) -> bool:
        """Write every value in full, never as an alias."""
        return True

    def represent_list(self, data: list) -> yaml.SequenceNode:
        """Write a short list of one-line scalars in flow style, other lists as SafeDumper does."""
        if all(not isinstance(item, (list, dict)) and "\n" not in str(item) for item in data) and (
                sum(len(str(item)) for item in data) <= _FLOW_LIMIT):
            return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
        return super().represent_list(data)

    def represent_str(self, data: str) -> yaml.ScalarNode:
        """Write multi-line text in literal block style, other text as SafeDumper does."""
        # YAML 1.1 readers (PyYAML) take these for line breaks and YAML 1.2 readers for text;
        # their double-quoted escapes mean the same to both.
        if any(char in data for char in "\x85\u2028\u2029"):
            return self.represent_scalar("tag:yaml.org,2002:str", data, style='"')
        if "\n" in data:
            return self.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return super().represent_str(data)


_CaseDumper.add_representer(str, _CaseDumper.represent_str)
_CaseDumper.add_representer(list, _CaseDumper.represent_list)


def _ordered(mapping: dict, order: tuple) -> dict:
    """Return `mapping` with the keys in `order` first, in that order, and the rest as given."""
    return dict(sorted(mapping.items(),
                       key=lambda item: order.index(item[0]) if item[0] in order else len(order)))


def _dump(case: dict) -> str:
    """Return the case file text for `case`, whose values are plain Python.

    Raises ValueError when the text doesn't read back, through the runner's reader and
    `_decode`, to the same data.
    """
    ordered = _ordered(case, _ORDER)
    for key in ("expect", "python_expect"):
        if isinstance(ordered.get(key), dict):
            expect = ordered[key] = _ordered(ordered[key], _EXPECT_ORDER)
            if isinstance(expect.get("output"), dict):
                expect["output"] = _ordered(expect["output"], _OUTPUT_ORDER)
    limit = sys.get_int_max_str_digits()
    # As the runner's `_read`: case files have no digit limit.
    sys.set_int_max_str_digits(0)
    try:
        # A value holding itself recurses without end.
        written = {key: value if key in runner._TEXT_KEYS else _encode(value)
                   for key, value in ordered.items()}
        # No width: an expression folded over lines is harder to read.
        text = yaml.dump(written, Dumper=_CaseDumper, sort_keys=False, allow_unicode=True,
                         width=math.inf)
        read = yaml.load(text, Loader=runner._CaseLoader)
        back = {key: value if key in runner._TEXT_KEYS else runner._decode(value)
                for key, value in read.items()}
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        raise ValueError(f"the case can't be written as a case file: {exc}") from exc
    finally:
        sys.set_int_max_str_digits(limit)
    if not runner._same(back, case):
        raise ValueError("the case file doesn't read back to the same case")
    return text


def _digest(text: str) -> str:
    """Return the sha256 of `text`, in hex."""
    return hashlib.sha256(text.encode()).hexdigest()


def _case_name(path: str, test_id: str, index: int = 1, salted: bool = False) -> str:
    """Return the case name for the test `test_id` (its node id after the file) in `path`.

    `tests/engine/test_expr_eval.py` and `TestA::test_x[a-b]` give
    `engine_expr_eval__TestA__x_a-b_`. `index` is the record's position among the test's
    records, skipped ones included, so a name stays the same when a skipped record is written
    later; from 2 on, the name ends in `__<index>`. A `salted` name ends in 8 hex characters
    of the test id's hash, for two tests whose names are the same.
    """
    segments = path.removeprefix("tests/").removesuffix(".py").split("/")
    parts = test_id.split("::")
    parts[-1] = parts[-1].removeprefix("test_")
    name = "__".join(["_".join(s.removeprefix("test_") for s in segments), *parts])
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if index > 1:
        name += f"__{index}"
    if salted:
        name += f"_{_digest(test_id)[:8]}"
    if len(name) <= _NAME_LIMIT:
        return name
    # The hash keeps two long names that share their first characters apart.
    return f"{name[:_NAME_LIMIT - 9]}_{_digest(name)[:8]}"


class _Index:
    """The case files under a cases directory, by the hash of their text."""

    def __init__(self, cases_dir: Path) -> None:
        """Hash every case file under `cases_dir`."""
        self.cases_dir = cases_dir
        self.by_digest: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        # The cases this session wrote or rewrote.
        self.harvested: set[str] = set()
        for path in sorted(cases_dir.glob("*/*.yaml")):
            self._hold(f"{path.parent.name}/{path.stem}", _digest(path.read_text(encoding="utf-8")))

    def _hold(self, name: str, digest: str) -> None:
        """Record that the case `name` holds the text with `digest`, forgetting its old text."""
        old = self.by_name.get(name)
        if old is not None and self.by_digest.get(old) == name:
            del self.by_digest[old]
        self.by_name[name] = digest
        self.by_digest[digest] = name

    def write(self, group: str, name: str, text: str) -> str:
        """Write `text` as the case `<group>/<name>`; return how it went.

        "written" when it was written. Nothing is written for a text this case already holds
        ("rewrite"), a text another case holds ("duplicate"), or a new text for a case this
        session already wrote or rewrote ("collision": two tests gave one name).
        """
        case_id, digest = f"{group}/{name}", _digest(text)
        holder = self.by_digest.get(digest)
        if holder == case_id:
            self.harvested.add(case_id)
            return "rewrite"
        if holder is not None:
            return "duplicate"
        if case_id in self.harvested:
            return "collision"
        path = self.cases_dir / group / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Through a temporary file, so an interrupted harvest never leaves half a case.
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        self._hold(case_id, digest)
        self.harvested.add(case_id)
        return "written"


_WORKTREE = Path(__file__).parent.parent
_YIAN_DIR = os.path.dirname(yian.__file__) + os.sep
# Calls from these files are the engine's own or the harvest's, never a test's.
_NOT_TEST = {runner.__file__, __file__}
_GROUPS = {"value": "values", "defaults": "defaults", "run": "run-core", "backward": "backward"}
# A top-level `uses:` key; the flow text can't be read as plain YAML (tags), so it's matched.
# One the match misses (an indented document, flow style) leaves the replay without the
# child's files, so the replay differs and the candidate is rejected.
_USES = re.compile(r"^uses\s*:", re.MULTILINE)
_AGENT = re.compile(r"\bkind:\s*agent\b")
_FILES_LIMIT = 50
_FILES_BYTES = 256 * 1024
# A `run_flow` argument is written under `run` when it differs from its default here.
_RUN_DEFAULTS = {"run_id": None, "num_workers": 8, "learnable": False, "llm_config": None}
_RUN_HOSTS = ("tools", "memory", "skills", "steering")
# The case sections a test's inputs go into, which must not hold the test's secrets.
_INPUT_KEYS = ("flow", "files", "env", "inputs", "run", "host", "backward", "values", "item",
               "value")
# Results a run case leaves out of its `expect` when they are empty.
_LEFT_OUT_WHEN_EMPTY = ("stream", "tool_calls")
# The results a pooled run's case keeps only when its three replays agree on them.
_POOLED_RESULTS = ("trace", "events", "stream", "tool_calls", "backward")
# The harvest's state while it runs, None when the plugin is off.
_STATE: _Harvest | None = None


class _Skip(Exception):
    """A call or candidate that isn't written; its text is the report's reason."""


def _harvest_error(exc: Exception) -> str:
    """Return the skip reason for an error in the harvest's own code."""
    return f"harvest error: {type(exc).__name__}: {exc}"


def _plain(value: object, what: str) -> object:
    """Return a copy of `value` when it is data; raise `_Skip("not data: <what>")` otherwise.

    Data is None, bools, ints, floats but NaN, text, lists, and mappings with text keys. A value
    nested deeper than the recursion limit isn't data either.
    """
    kind = type(value)
    if value is None or kind in (bool, int, str) or (kind is float and not math.isnan(value)):
        return value
    try:
        if kind is list:
            return [_plain(item, what) for item in value]
        if kind is dict and all(type(key) is str for key in value):
            return {key: _plain(item, what) for key, item in value.items()}
    except RecursionError:
        raise _Skip(f"not data: {what} (nested too deep)") from None
    raise _Skip(f"not data: {what}")


def _written(case: dict) -> str:
    """Return the case file text for `case`, or raise `_Skip` when it can't be written."""
    try:
        return _dump(case)
    except ValueError:
        raise _Skip("not data: case file") from None


def _keyed(secret: str, keys: dict) -> str:
    """Return the `<key:N>` placeholder for `secret`, numbering secrets in `keys` from 1."""
    return f"<key:{keys.setdefault(secret, len(keys) + 1)}>"


def _env(models: bool = False, keys: dict | None = None) -> dict:
    """Return the environment a call sees, as `{"env": ...}`.

    That is every `YIAN_*` variable but `YIAN_AUTH_FILE` and, unless `models`, the model
    defaults (`expr`, `value` and `defaults` calls don't read them), and each connection
    variable whose value differs from the session's start, with `*_API_KEY` values as
    `<key:N>`, numbered in `keys` (default: a new numbering). Empty: `{}`.
    """
    left_out = ("YIAN_AUTH_FILE",) if models else (
        "YIAN_AUTH_FILE", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL")
    env = {}
    for name, value in sorted(os.environ.items()):
        if name.startswith("YIAN_"):
            keep = name not in left_out
        else:
            # The harvesting shell's own keys must never land in a case.
            keep = name.endswith(runner._ENV_SUFFIXES) and _STATE.start_env.get(name) != value
        if keep:
            env[name] = value
    keys = {} if keys is None else keys
    for name, value in env.items():
        if name.endswith("_API_KEY"):
            env[name] = _keyed(value, keys)
    return {"env": env} if env else {}


def _set_entries(case: dict) -> list:
    """Return the names of the defaults entries whose variables the case's `env` sets."""
    env = case.get("env", {})
    return [entry.name for entry in defaults.ENTRIES if entry.variable in env]


def _result(outcome: tuple, form) -> dict:
    """Return a call's outcome, `("value", v)` or `("raises", error)`, in the runner's form.

    A value goes through `form`; an error is `raises`.
    """
    kind, value = outcome
    return {"raises": runner._raised(value)} if kind == "raises" else form(value)


def _expect(case: dict, results: dict) -> dict:
    """Return the runner's `results` for `case` as its `expect`.

    The runner's own `error` and `left` go, so do a run's empty stream and tool calls, and
    `defaults` keeps the entries `env` sets.
    """
    expect = {key: value for key, value in results.items()
              if key not in ("error", "left") and (value or key not in _LEFT_OUT_WHEN_EMPTY)}
    if "defaults" in expect:
        expect["defaults"] = {name: expect["defaults"][name] for name in _set_entries(case)}
    return expect


def _trees(case: dict) -> list:
    """Return the expression trees of an `expr` case, parsed as the runner parses them."""
    if "template" in case:
        return runner.expression_trees(case["template"])
    if case.get("mode") == "condition":
        return runner.expression_trees(case["expr"], condition=True)
    return [runner.parse_expr(case["expr"])]


def _check_types(case: dict, types: dict | None) -> None:
    """Raise `_Skip` unless each of `types` the case's expressions use is the runner's.

    The runner builds the types a case's `isinstance` and constructor names use, with no
    typedefs; an entry the expressions don't use changes nothing.
    """
    if not types:
        return
    try:
        trees = _trees(case)
    except Exception:
        # The call raised on the text before it could use a type.
        return
    registry = runner.read_typedefs({})
    built: dict = {}
    for tree in trees:
        for names, resolve in ((runner.isinstance_type_names, runner.resolve_isinstance_type),
                               (runner.constructor_names, runner.resolve_constructor_type)):
            for name in names(tree):
                try:
                    built[name] = resolve(name, registry)
                except Exception:
                    built[name] = None
    if any(name in built and built[name] != typ for name, typ in types.items()):
        raise _Skip("types not expressible")


def _answering(arguments: dict) -> list:
    """Replace the call's resolver with one noting each answer; return the `(path, value)` list."""
    resolve, answers = arguments["resolve"], []

    def answer(path: str) -> object:
        value = resolve(path)
        answers.append((path, value))
        return value

    arguments["resolve"] = answer
    return answers


def _walk(value: object, steps: tuple) -> object:
    """Return what the dotted `steps` read in `value`, as the runner's resolver walks it."""
    for step in steps:
        value = value.get(step) if isinstance(value, dict) else None
    return value


def _values(answers: list) -> dict:
    """Return a resolver's answers nested as `{"values": ...}`; `a.b` gives `{a: {b: v}}`.

    Each answer is authoritative: one under a path already answered must equal walking that
    answer, else the call skips. A None answer is a miss and adds nothing. No answers: `{}`.
    """
    values: dict = {}
    answered: dict = {}
    # Shallowest first, so a whole answer is known before the answers inside it.
    for path, value in sorted(answers, key=lambda answer: answer[0].count(".")):
        parts = tuple(step.strip() for step in path.strip().split("."))
        try:
            value = _plain(value, "resolver")
        except _Skip:
            raise _Skip("resolver: value is not data") from None
        for size in range(len(parts), 0, -1):
            if parts[:size] in answered:
                if not runner._same(_walk(answered[parts[:size]], parts[size:]), value):
                    raise _Skip("resolver: conflicting answers")
                break
        else:
            answered[parts] = value
            if value is not None:
                node = values
                for step in parts[:-1]:
                    node = node.setdefault(step, {})
                node[parts[-1]] = value
    return {"values": values} if values else {}


def _expr_finish(case: dict, arguments: dict, answers: list | None = None):
    """Add the call's `item` and environment to an `expr` case; return its finish step.

    The finish step adds the resolver's `answers`, checks the call's `types` and returns the
    case and the call's result.
    """
    if arguments.get("item") is not None:
        case["item"] = _plain(arguments["item"], "item")
    case |= _env()

    def finish(outcome: tuple) -> tuple:
        full = case | _values(answers) if answers is not None else case
        _check_types(full, arguments.get("types"))
        return full, _result(outcome, lambda value: {"value": _plain(value, "result")})

    return finish


def _record_eval_expr(arguments: dict):
    """Record an `eval_expr` call as an `expr` case, in strict or binding mode."""
    if arguments["trace"]:
        raise _Skip("trace=True")
    held = _STATE.trees.get(id(arguments["tree"]))
    if held is None:
        raise _Skip("expression text unknown")
    case = {"kind": "expr", "expr": held[1]}
    # A condition-falsy miss is the binding mode's None.
    if arguments["mode"] is ResolveMode.STRICT_RAISE:
        case["mode"] = "strict"
    return _expr_finish(case, arguments, _answering(arguments))


def _record_eval_binding(arguments: dict):
    """Record an `eval_binding` call as an `expr` case with a binding `template`."""
    if arguments["trace"]:
        raise _Skip("trace=True")
    case = {"kind": "expr", "template": arguments["source"]}
    return _expr_finish(case, arguments, _answering(arguments))


def _with_record(case: dict, record: object) -> dict:
    """Return `case` with `record`, a call's input record, as its `values`."""
    record = _plain(record, "values")
    return case | {"values": record} if record else case


def _record_evaluate_when_record(arguments: dict):
    """Record an `evaluate_when_record` call as an `expr` case in condition mode."""
    case = {"kind": "expr", "expr": arguments["expression"], "mode": "condition"}
    return _expr_finish(_with_record(case, arguments["record"]), arguments)


def _record_render_template_record(arguments: dict):
    """Record a `render_template_record` call as an `expr` case with a strict `template`."""
    case = {"kind": "expr", "template": arguments["text"], "mode": "strict"}
    return _expr_finish(_with_record(case, arguments["record"]), arguments)


def _record_build_value_as(arguments: dict):
    """Record a `build_value_as` call as a `value` case."""
    declared = arguments["declared"]
    if isinstance(declared, ValueKind):
        # A bare kind is that type: storage spells lists `list[str]`, type text `list(str)`.
        text, raw = declared.value.replace("[", "(").replace("]", ")"), {}
    elif id(declared) in _STATE.types:
        _, text, raw = _STATE.types[id(declared)]
    else:
        raise _Skip("type text unknown")
    case = {"kind": "value", "type": text, "value": _plain(arguments["value"], "value"),
            **({"typedefs": raw} if raw else {}), **_env()}

    def finish(outcome: tuple) -> tuple:
        return case, _result(outcome, lambda typed: _plain(runner._value_form(typed), "result"))

    return finish


def _record_defaults(arguments: dict):
    """Record a `defaults.current()` call as a `defaults` case."""
    case = {"kind": "defaults", **_env()}
    if not _set_entries(case):
        raise _Skip("no defaults variable set")

    def finish(outcome: tuple) -> tuple:
        return case, _result(outcome, runner._defaults_form)

    return finish


def _files(text: str, search_paths: list | None) -> dict:
    """Return the files a load of `text` reads from `search_paths`, as `{"files": ...}`.

    A flow with `uses:` needs exactly one search path; each `.yaml` file under it, at most 50
    and 256 KiB in all, is a file by its relative path. A `flow.yaml` there, the runner's name
    for the flow, must hold the flow's text and is left out. Otherwise, or with no files: `{}`.
    """
    if not _USES.search(text):
        return {}
    if len(search_paths or []) != 1:
        raise _Skip("uses: needs one search path")
    root = Path(search_paths[0])
    paths = []
    for path in root.rglob("*.yaml"):
        if path.is_file():
            paths.append(path)
        # A search path can be a large tree; stop at the limit rather than walk all of it.
        if len(paths) > _FILES_LIMIT:
            raise _Skip(f"files: more than {_FILES_LIMIT}")
    files, size = {}, 0
    for path in sorted(paths):
        data = path.read_bytes()
        size += len(data)
        if size > _FILES_BYTES:
            raise _Skip("files: more than 256 KiB")
        try:
            files[path.relative_to(root).as_posix()] = data.decode("utf-8")
        except UnicodeDecodeError:
            raise _Skip("files: not UTF-8") from None
    # The runner writes the flow to `flow.yaml` and then the files, so one there would replace it.
    if files.pop("flow.yaml", text) != text:
        raise _Skip("files: flow.yaml isn't the flow")
    return {"files": files} if files else {}


def _record_load_flow(arguments: dict):
    """Record a `load_flow` call as a `load` case; its `source_label` isn't kept.

    The runner compares a load error's code and message, not its frames, which name the label.
    """
    if arguments["child_resolver"] is not None:
        raise _Skip("child_resolver")
    text, call = arguments["text"], _STATE.active
    # Read now: the test may change the files after the call.
    case = {"kind": "load", "flow": text, **_files(text, arguments["search_paths"]),
            **_env(models=True)}

    def finish(outcome: tuple) -> tuple:
        kind, value = outcome
        if kind == "value":
            _STATE.flows[id(value)] = (value, case, call)
            return case, {"load": "ok"}
        if not isinstance(value, LoadError):
            raise _Skip(f"load raised {type(value).__name__}")
        return case, {"load": {"diagnostics": [runner._diagnostic(value.diagnostic)]}}

    return finish


def _record_run_flow(arguments: dict):
    """Record a `run_flow` call as a `run` case, its flow's text from its `load_flow` record.

    `log` and the event and stream hooks aren't kept: the runner gives the run its own. The
    run claims the load record, which isn't a case of its own once the run's case is written.
    """
    for host in _RUN_HOSTS:
        if arguments[host]:
            raise _Skip(f"host: {host}")
    held = _STATE.flows.get(id(arguments["flow"]))
    if held is None:
        raise _Skip("flow text unknown")
    _, load, load_call = held
    call = _STATE.active
    load_call.runs.append(call)
    case = {"kind": "run", **{key: load[key] for key in ("flow", "files") if key in load},
            **_env(models=True, keys=call.keys)}
    inputs = _plain(arguments["inputs"], "inputs")
    if inputs != {}:
        case["inputs"] = inputs
    run = {key: _plain(arguments[key], key) for key in _RUN_DEFAULTS}
    run = {key: value for key, value in run.items() if not runner._same(value, _RUN_DEFAULTS[key])}
    if arguments["context"] is not None:
        run["context"] = "present"
    config = run.get("llm_config")
    if isinstance(config, dict):
        _hide_keys(config, call)
    if run:
        case["run"] = run

    def finish(outcome: tuple) -> tuple:
        kind, value = outcome
        if kind == "value":
            # So a backward call on the result, during the test, joins this record.
            _STATE.results[id(value)] = (value, call)
        return case, _result(
            outcome, lambda result: {"output": runner._output(result, arguments["flow"])})

    return finish


def _hide_keys(mapping: dict, call: _Call) -> None:
    """Replace `mapping`'s `api_key` value and `headers` values by `<key:N>` in `call`'s numbering.

    They are the secrets a run or backward call sends: the case holds placeholders, and `call`
    notes the secrets.
    """
    headers = mapping["headers"] if isinstance(mapping.get("headers"), dict) else {}
    for holder, name in [(mapping, "api_key"), *((headers, name) for name in headers)]:
        if isinstance(holder.get(name), str):
            call.secrets.add(holder[name])
            holder[name] = _keyed(holder[name], call.keys)


def _record_resume_flow(arguments: dict):
    """Skip a `resume_flow` call: resume cases are converted by hand."""
    raise _Skip("resume (hand-converted)")


def _taken(tools: list, schema: type | None, messages: list) -> tuple | None:
    """Take the model call being made with `messages`, `tools` bound and `schema` asked for.

    Its `host.models` entry is reserved now, in the order calls start, as the runner's replay
    takes entries. Returns the active record, the reserved entry, empty until `_add_entry`
    fills it, and the call's request in the case's form. None when no record is active, or
    when the call can't be recorded, which skips the record's run.
    """
    key = runner._take_model_key()
    state = _STATE
    if state is None:
        return None
    # A run nested in a recorded one (from a test's hook) records into the outer run, whose
    # replay makes no nested calls, so it has entries left over and is rejected.
    with state.lock:
        call = state.active
    if call is None:
        return None
    try:
        if key is None:
            raise _Skip("model: call outside any AGENT")
        if key[1] not in runner._CONSUMERS:
            raise _Skip(f"model: {key[1]} call")
        request = _model_data(runner._v1_request, messages, tools, schema)
    except _Skip as skip:
        _skip_host(call, str(skip))
        return None
    except Exception as exc:
        _skip_host(call, _harvest_error(exc))
        return None
    node, consumer = key
    entry: dict = {}
    with state.lock:
        # A call starting after its run ended (a thread the test left running) adds nothing.
        if state.active is not call:
            return None
        call.models.setdefault(node, {}).setdefault(consumer, []).append(entry)
    return call, entry, request


def _add_entry(taken: tuple | None, entry) -> None:
    """Fill the `host.models` entry `_taken` reserved with `entry()` and the call's request."""
    if taken is None:
        return
    call, reserved, request = taken
    try:
        recorded = {**entry(), "request": request}
    except _Skip as skip:
        _skip_host(call, str(skip))
        return
    except Exception as exc:
        _skip_host(call, _harvest_error(exc))
        return
    with _STATE.lock:
        # A call ending after its run did (a thread the test left running) leaves its entry
        # empty, which drops the run's case.
        if _STATE.active is call:
            reserved.update(recorded)


def _skip_host(call: _Call, reason: str) -> None:
    """Note why a host call of `call`'s run can't be recorded, unless one already is.

    A host call is a model call or a backoff draw. Nothing is noted once the run has ended.
    """
    with _STATE.lock:
        if _STATE.active is call:
            call.host_skip = call.host_skip or reason


def _model_data(convert, *args: object) -> object:
    """Return `convert(*args)`, one of the runner's translations, raising `_Skip` for `_NotData`."""
    try:
        return convert(*args)
    except runner._NotData as reason:
        raise _Skip(f"model: {reason}") from None


def _usage(metadata: dict) -> dict:
    """Return LangChain's `usage_metadata` as a case usage, or raise `_Skip` if it can't be one."""
    usage = {name: metadata.get(name) for name in runner._USAGE_COUNTS}
    for name, (details, key) in runner._USAGE_DETAILS.items():
        if key in (metadata.get(details) or {}):
            usage[name] = metadata[details][key]
    if runner._usage_metadata(usage) != metadata:
        raise _Skip("model: usage the case format can't hold")
    return usage


def _reply(reply: object) -> dict:
    """Return an invoked call's `AIMessage` as a `reply` entry, with its `usage` if it has one."""
    if type(reply) is not AIMessage:
        raise _Skip(f"model: reply is {type(reply).__name__}")
    entry = {"reply": _model_data(runner._v1_message, reply, {})}
    if reply.usage_metadata:
        entry["usage"] = _usage(reply.usage_metadata)
    return entry


def _structured_reply(result: object) -> dict:
    """Return a native structured call's result as a `reply` entry holding its JSON.

    No usage: it reaches only the callbacks of the model inside the binding.
    """
    if not isinstance(result, BaseModel):
        raise _Skip(f"model: structured result is {type(result).__name__}")
    text = json.dumps(result.model_dump(mode="json", by_alias=True))
    return {"reply": _model_data(runner._v1_message, AIMessage(content=text), {})}


def _chunks(chunks: list) -> list:
    """Return a stream's `AIMessageChunk`s as case chunks: each one's text, tool calls, usage."""
    recorded = []
    for chunk in chunks:
        if type(chunk) is not AIMessageChunk:
            raise _Skip(f"model: stream yielded {type(chunk).__name__}")
        if not isinstance(chunk.content, str):
            raise _Skip("model: chunk with list content")
        if chunk.additional_kwargs:
            raise _Skip(f"model: chunk with additional_kwargs "
                        f"({', '.join(sorted(chunk.additional_kwargs))})")
        if chunk.content:
            recorded.append({"text": chunk.content})
        for delta in chunk.tool_call_chunks:
            call = {"index": delta["index"], "arguments": delta["args"] or ""}
            call |= {key: delta[key] for key in ("id", "name") if delta[key] is not None}
            recorded.append({"tool_call": call})
        if chunk.usage_metadata:
            recorded.append({"usage": _usage(chunk.usage_metadata)})
    return recorded


def _finish_chunk(chunks: list) -> dict:
    """Return the `finish` chunk of a stream that ended: its last `finish_reason`, or null."""
    reasons = [chunk.response_metadata.get("finish_reason") for chunk in chunks]
    return {"finish": next((reason for reason in reversed(reasons) if reason is not None), None)}


def _error(exc: Exception) -> dict:
    """Return a model call's error as an entry's `raise`."""
    error = {"class": type(exc).__name__, "message": str(exc)}
    if type(getattr(exc, "status_code", None)) is int:
        error["status"] = exc.status_code
    return error


# The model methods a proxy wraps.
_RECORDED = ("invoke", "stream", "bind_tools", "with_structured_output")


class _Proxy:
    """A model that records each call it passes to the model it wraps.

    It has only the wrapped model's attributes, so `stream_reply` still finds no `stream` on a
    model without one. `invoke` and `stream` are recorded; `bind_tools` and
    `with_structured_output` return proxies sending the tools or schema; the rest is the
    wrapped model's own.
    """

    def __init__(self, model: object, tools: list | tuple = (), schema: type | None = None) -> None:
        """Wrap `model`, whose calls send `tools` and ask for `schema`."""
        self._model, self._tools, self._schema = model, list(tools), schema

    # `isinstance` sees the wrapped model's class: `invoke_structured` gives its usage callback
    # only to a `Runnable`, and a test's checks must see what they would see unwrapped.
    __class__ = property(lambda self: type(self._model))

    def __getattr__(self, name: str) -> object:
        """Return the wrapped model's attribute `name`, or the proxy's own for a recorded one."""
        # A proxy `copy` builds without `__init__` has no fields yet, and must not take the
        # wrapped model's copy protocol.
        if name.startswith("__") or name in ("_model", "_tools", "_schema"):
            raise AttributeError(name)
        value = getattr(self._model, name)
        return getattr(self, f"_recorded_{name}") if name in _RECORDED else value

    def _recorded_bind_tools(self, tools: list, *args: object, **kwargs: object) -> _Proxy:
        """Bind `tools` to the wrapped model; return the result wrapped to send them."""
        tools = list(tools)
        return _Proxy(self._model.bind_tools(tools, *args, **kwargs), tools, self._schema)

    def _recorded_with_structured_output(self, schema: type, *args: object,
                                         **kwargs: object) -> _Proxy:
        """Bind `schema` to the wrapped model; return the result wrapped to ask for it."""
        return _Proxy(self._model.with_structured_output(schema, *args, **kwargs), (), schema)

    def _recorded_invoke(self, messages: list, *args: object, **kwargs: object) -> object:
        """Invoke the wrapped model, recording its reply or error."""
        taken = _taken(self._tools, self._schema, messages)
        try:
            result = self._model.invoke(messages, *args, **kwargs)
        except Exception as exc:
            if self._schema is None:
                _add_entry(taken, lambda: {"chunks": [], "raise": _error(exc)})
            elif taken is not None:
                # A binding validates the reply itself, so its error may be a rejected reply the
                # recorder never sees, not a provider failure: no entry can stand for it.
                _skip_host(taken[0], f"model: structured call raised {type(exc).__name__} "
                                     "(a binding's errors can't be scripted)")
            raise
        _add_entry(taken, lambda: _reply(result) if self._schema is None
                   else _structured_reply(result))
        return result

    def _recorded_stream(self, messages: list, *args: object, **kwargs: object):
        """Stream from the wrapped model, recording its chunks and how the stream ended."""
        # Taken when the call is made, as the runner's replay takes its entry.
        taken = _taken(self._tools, self._schema, messages)
        return _streamed(taken, lambda: self._model.stream(messages, *args, **kwargs))


def _streamed(taken: tuple | None, stream):
    """Yield each chunk of `stream()`, then add the entry of the call `_taken` took."""
    seen = []
    try:
        for chunk in stream():
            seen.append(chunk)
            yield chunk
    except GeneratorExit:
        # An interrupted reply: the replay can't cut its stream at the same chunk.
        if taken is not None:
            _skip_host(taken[0], "model: stream closed early")
        raise
    except Exception as exc:
        _add_entry(taken, lambda: {"chunks": _chunks(seen), "raise": _error(exc)})
        raise
    _add_entry(taken, lambda: {"chunks": [*_chunks(seen), _finish_chunk(seen)]})


def _recording() -> bool:
    """Return whether a model built now is recorded: a run is recorded and an AGENT seated."""
    state = _STATE
    return (state is not None and state.active is not None
            and getattr(runner._SEAT, "seat", None) is not None)


def _factory(original):
    """Return the model factory `original` wrapped to give recorded models (see `_recording`)."""
    def model_from_config(*args: object, **kwargs: object) -> object:
        model = original(*args, **kwargs)
        # A test's factory that calls the wrapped one gives a proxy already.
        return model if type(model) is _Proxy or not _recording() else _Proxy(model)

    model_from_config.harvest_wraps = original
    return model_from_config


def _draws(original):
    """Return the backoff draw `original` wrapped to add each draw to the active record's jitter.

    A draw outside `[0, 1)` can't be a `host.jitter` entry: it skips the record's run.
    """
    def _jitter() -> object:
        draw = original()
        state = _STATE
        call = state.active if state is not None else None
        if call is None:
            return draw
        if type(draw) in (int, float) and 0 <= draw < 1:
            with state.lock:
                # As a model call's entry: a draw after the run ended changes nothing.
                if state.active is call:
                    call.jitter.append(draw)
        else:
            _skip_host(call, f"jitter: draw {draw!r} outside [0, 1)")
        return draw

    _jitter.harvest_wraps = original
    return _jitter


# What a value assigned to a watched module's name is wrapped with, by name.
_WATCHED = {"model_from_config": _factory, "_jitter": _draws}


class _Watched(ModuleType):
    """A yian module that wraps each value assigned to a `_WATCHED` name, so the harvest sees it.

    However a test patches the module (`monkeypatch.setattr`, `mock.patch`, plain assignment),
    its value is wrapped; a wrapped value isn't wrapped again, nor one the harvest's replay
    assigns.
    """

    def __setattr__(self, name: str, value: object) -> None:
        """Set the attribute `name`, wrapping `value` for a `_WATCHED` name."""
        state = _STATE
        if (name in _WATCHED and state is not None and not state.disabled
                and "harvest_wraps" not in getattr(value, "__dict__", {})):
            value = _WATCHED[name](value)
        super().__setattr__(name, value)


def _remember_tree(arguments: dict, tree: object) -> None:
    """Note the text a test parsed into `tree`."""
    _STATE.trees[id(tree)] = (tree, arguments["text"])


def _remember_typedefs(arguments: dict, registry: dict) -> None:
    """Note the `typedefs:` mapping a test read into `registry`, when it is data."""
    try:
        _STATE.registries[id(registry)] = (registry, _plain(arguments["raw"] or {}, "typedefs"))
    except _Skip:
        pass


def _remember_type(arguments: dict, typ: object) -> None:
    """Note the type text and typedefs a test resolved into `typ`, when its registry is known."""
    registry = arguments["registry"]
    if id(registry) in _STATE.registries:
        raw = _STATE.registries[id(registry)][1]
    elif isinstance(registry, dict) and not registry:
        raw = {}
    else:
        return
    _STATE.types[id(typ)] = (typ, arguments["type_str"], raw)


def _from_test(frame) -> bool:
    """Return whether the code running in `frame` is a test's: not yian's, nor the harvest's."""
    name = frame.f_code.co_filename
    return not name.startswith(_YIAN_DIR) and name not in _NOT_TEST


def _maps(frame) -> bool:
    """Return whether a text helper's call from `frame` is noted: a test's, while one runs."""
    state = _STATE
    return (state is not None and state.test is not None and not state.disabled
            and _from_test(frame))


def _begin_record(frame, entry: str) -> tuple[_Test, _Call] | None:
    """Start recording a call to `entry` from `frame`, when the call counts.

    It counts when a test's code made it while a test runs, the harvest isn't replaying, and
    no recorded call is active anywhere in the process. Returns the running test and the
    call's record, which is the active one until `_end_record`; otherwise None.
    """
    state = _STATE
    if not _maps(frame):
        return None
    with state.lock:
        # Read once: a thread a test left behind may record while the test ends.
        test = state.test
        if state.active is not None or test is None:
            return None
        state.active = _Call(entry)
        return test, state.active


def _end_record() -> None:
    """End the recorded call `_begin_record` started."""
    with _STATE.lock:
        _STATE.active = None


class _Call:
    """One recorded call: its entry point, candidate case and own result, or why it's skipped."""

    def __init__(self, entry: str) -> None:
        """Start the record of a call to `entry`."""
        self.entry = entry
        self.case: dict | None = None
        self.result: dict | None = None
        self.skip: str | None = "the call didn't finish"
        # The keys and header values the call could use; a case holding one isn't written.
        self.secrets: set[str] = set()
        # For a load, the records of the runs of its flow.
        self.runs: list[_Call] = []
        # For a run, its model calls as `host.models` and backoff draws as `host.jitter`, and
        # why one can't be recorded.
        self.models: dict = {}
        self.jitter: list = []
        self.host_skip: str | None = None
        # For a run, its case's secret numbering, and its result's backward calls as the
        # case's `backward` entries with their outcomes.
        self.keys: dict = {}
        self.backward: list = []
        self.outcomes: list = []
        self.written = False


def _finish_call(call: _Call, finish, outcome: tuple) -> None:
    """Take the call's case and result from `finish(outcome)`, or the reason it skips.

    Never raises: an error in the harvest's code is a skip, not the test's error.
    """
    if finish is None:
        return
    try:
        call.case, call.result = finish(outcome)
        call.skip = None
    except _Skip as skip:
        call.skip = str(skip)
    except Exception as exc:
        call.skip = _harvest_error(exc)


def _recorded(test: _Test, call: _Call, original, recorder, signature, args: tuple,
              kwargs: dict) -> object:
    """Call `original` with `args` and `kwargs`, recording it as `call` in `test` via `recorder`.

    `recorder(arguments)` reads the bound arguments (it may replace some, such as a resolver)
    and returns the step that turns the call's outcome into a case and result. The call's own
    result or error is returned or raised unchanged.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        # The call's own error, from the function itself.
        return original(*args, **kwargs)
    bound.apply_defaults()
    test.calls.append(call)
    call.secrets.update(value for name, value in os.environ.items() if name.endswith("_API_KEY"))
    finish = None
    try:
        finish = recorder(bound.arguments)
    except _Skip as skip:
        call.skip = str(skip)
    except Exception as exc:
        call.skip = _harvest_error(exc)
    try:
        value = original(*bound.args, **bound.kwargs)
    except Exception as exc:
        _finish_call(call, finish, ("raises", exc))
        raise
    _finish_call(call, finish, ("value", value))
    return value


def _entry(entry: str, original, recorder):
    """Return `original` wrapped to record the calls that count (see `_begin_record`)."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args: object, **kwargs: object) -> object:
        begun = _begin_record(sys._getframe(1), entry)
        if begun is None:
            return original(*args, **kwargs)
        try:
            return _recorded(*begun, original, recorder, signature, args, kwargs)
        finally:
            _end_record()

    return wrapper


def _backward_entry(call: _Call, feedback: object, api_key: object, headers: object) -> dict:
    """Return a backward call's `backward` entry, its secrets as `<key:N>` in `call`'s numbering."""
    entry = {"feedback": _plain(feedback, "feedback")}
    for name, value in (("api_key", api_key), ("headers", headers)):
        if value is not None:
            entry[name] = _plain(value, name)
    _hide_keys(entry, call)
    return entry


def _backward(original):
    """Return `RunResult.backward` wrapped to add a test's call to its result's run record.

    A call from a test's code, on a result whose run has a record that isn't skipped, while no
    other record is active, adds its entry and its outcome's `status` (or its error as
    `raises`) to that record, which is active while the call runs, so its model calls are the
    run's. Other calls record nothing.
    """
    @functools.wraps(original)
    def backward(self: RunResult, feedback: object = None, *, api_key: object = None,
                 headers: object = None) -> object:
        state, call = _STATE, None
        if _maps(sys._getframe(1)):
            with state.lock:
                held = state.results.get(id(self))
                if held is not None and held[1].skip is None and state.active is None:
                    call = state.active = held[1]
        if call is None:
            return original(self, feedback, api_key=api_key, headers=headers)
        try:
            try:
                call.backward.append(_backward_entry(call, feedback, api_key, headers))
            except _Skip as skip:
                call.skip = str(skip)
            except Exception as exc:
                call.skip = _harvest_error(exc)
            try:
                outcome = original(self, feedback, api_key=api_key, headers=headers)
            except Exception as exc:
                call.outcomes.append({"raises": runner._raised(exc)})
                raise
            call.outcomes.append({"status": outcome.status})
            return outcome
        finally:
            _end_record()

    return backward


def _mapper(original, remember):
    """Return `original` wrapped to `remember(arguments, result)` for each test's call."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        if _maps(sys._getframe(1)):
            try:
                remember(signature.bind(*args, **kwargs).arguments, result)
            except Exception:
                # A text not noted only makes a later record skip; it never fails the test.
                pass
        return result

    return wrapper


def _holders(original: object) -> list[tuple[ModuleType, str]]:
    """Return each `(module, name)` where a loaded yian module holds `original`."""
    return [(module, name) for module_name, module in list(sys.modules.items())
            if module is not None and (module_name == "yian" or module_name.startswith("yian."))
            for name, value in list(vars(module).items()) if value is original]


def _replace(original, wrapper, replaced: list) -> None:
    """Put `wrapper` in place of `original` in every loaded yian module that holds it.

    Each `(module, name, original)` replaced is added to `replaced`.
    """
    for module, name in _holders(original):
        setattr(module, name, wrapper)
        replaced.append((module, name, original))


class _Test:
    """The test running now: its recorded calls and the outcomes of its phases."""

    def __init__(self) -> None:
        """Start a test with no calls and no outcome."""
        self.calls: list[_Call] = []
        self.failed = self.skipped = self.passed = False

    @property
    def outcome(self) -> str:
        """Return `passed`, `failed` or `skipped`; a test whose call never passed failed."""
        if self.failed or not (self.passed or self.skipped):
            return "failed"
        return "skipped" if self.skipped else "passed"


class _Harvest:
    """A harvest session's state: what it wrote, the test running now and its noted texts."""

    def __init__(self, config: pytest.Config, out: Path,
                 code: dict[str, str] | None = None) -> None:
        """Start a harvest writing under `out`, translating CODE bodies by the table `code`."""
        self.config, self.out = config, out
        self.index = _Index(out / "cases")
        self.start_env = dict(os.environ)
        self.lock = threading.Lock()
        # The record of the recorded call running now, anywhere in the process; no other call
        # is recorded while it runs.
        self.active: _Call | None = None
        # Set while the harvest replays, so the runner's calls are never recorded.
        self.disabled = False
        self.test: _Test | None = None
        # Strong references, by id, for the test's duration: ids can be reused, and a lone
        # atom is a `Token`, which can't be weakly referenced.
        self.trees: dict = {}
        self.types: dict = {}
        self.registries: dict = {}
        # Each flow the test loaded, with its load case and record, so a run of it can claim it;
        # each result a recorded run gave, with the run's record, so a backward call joins it.
        self.flows: dict = {}
        self.results: dict = {}
        self.remaining: dict[str, int] = {}
        self.reports: dict[str, dict] = {}
        # Files pytest's own selection narrowed, whose reports aren't written.
        self.narrowed: set[str] = set()
        # Set while the plugin deselects reported files, to tell its call from pytest's.
        self.deselecting = False
        self.replaced: list = []
        self.paths: set | None = None
        # The seats of the runner's model replay, for the session; and the originals wrapped.
        self.patch = pytest.MonkeyPatch()
        self.factory = llm_clients.model_from_config
        self.jitter = tool_node._jitter
        self.backward = RunResult.backward
        # The `--harvest-code` table: TS bodies by the key of the Python body each replaces.
        self.code = code or {}


def _new_report(path: str) -> dict:
    """Return an empty report for the test file `path`.

    Its `skips` list the tests by reason, its `flags` the cases written by flag, its `code` the
    Python CODE bodies left untranslated by key, with their text and cases, and its `references`
    the cases of each CODE reference left unported.
    """
    return {"file": path, "tests": {"passed": 0, "failed": 0, "skipped": 0}, "records": {},
            "written": {}, "duplicates": 0, "skips": {}, "flags": {}, "code": {},
            "references": {}}


def _machine_paths(state: _Harvest) -> set:
    """Return the paths a case must not hold: temp dirs, the worktree and home."""
    if state.paths is None:
        given = [tempfile.gettempdir(), _WORKTREE, Path.home()]
        # Absent under `-p no:tmpdir`; the default base temp lies under gettempdir() anyway.
        factory = getattr(state.config, "_tmp_path_factory", None)
        if factory is not None:
            given.append(factory.getbasetemp())
        state.paths = {str(form) for path in given
                       for form in (Path(path), Path(path).resolve())} - {os.sep}
    return state.paths


class _Shim(NamedTuple):
    """A TS-era rule as a test-only patch of Python yian, on only during a harvest replay.

    `rule` is the rule's id in `conformance/README.md`; `install(patch, case)` makes Python
    follow the rule through `patch`, given the candidate `case`.
    """

    rule: str
    install: Callable[[pytest.MonkeyPatch, dict], None]


def _js_number(x: float) -> str:
    """Return `x` as JavaScript's `Number.prototype.toString` writes it."""
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if math.isnan(x):
        return "NaN"
    if x == 0:
        return "0"
    if x < 0:
        return "-" + _js_number(-x)
    # `repr` is the shortest round-tripping form, as in ECMAScript's Number::toString.
    _, digits, exponent = Decimal(repr(x)).normalize().as_tuple()
    s = "".join(map(str, digits))
    k, n = len(s), exponent + len(s)
    if k <= n <= 21:
        return s + "0" * (n - k)
    if 0 < n <= 21:
        return s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return "0." + "0" * -n + s
    mantissa = s[0] + ("." + s[1:] if k > 1 else "")
    return f"{mantissa}e{'+' if n > 0 else '-'}{abs(n - 1)}"


def _ts_json(value: object, indent: int | None = None) -> str:
    """Return `value` as yian's JSON writer writes it.

    Numbers are as in the text form (so `inf` is a bare `Infinity`), text keeps non-ASCII, a
    file is its `{uri, mime, name}` mapping and any other non-JSON value its text form, quoted.
    With `indent=None` the JSON is compact; with an int `indent` it has `json.dumps`'s layout.
    """
    colon = ":" if indent is None else ": "

    def write(value: object, level: int) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return _js_number(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, FileRef):
            value = value.model_dump()
        if isinstance(value, list):
            items, brackets = [write(item, level + 1) for item in value], "[]"
        elif isinstance(value, dict):
            items = [json.dumps(_ts_text(key), ensure_ascii=False) + colon + write(item, level + 1)
                     for key, item in value.items()]
            brackets = "{}"
        else:
            return json.dumps(_ts_text(value), ensure_ascii=False)
        if not items:
            return brackets
        if indent is None:
            return brackets[0] + ",".join(items) + brackets[1]
        inner = "\n" + " " * (indent * (level + 1))
        outer = "\n" + " " * (indent * level)
        return brackets[0] + inner + ("," + inner).join(items) + outer + brackets[1]

    return write(value, 0)


def _ts_text(value: object) -> str:
    """Return `value`'s text form: text as itself, other JSON values and files as JSON.

    `None` reads `null`; anything else (an exception a tool reported, a host object) its `str`.
    """
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float, list, dict, FileRef)):
        return _ts_json(value)
    return str(value)


# A JS array index: a canonical decimal below 2^32 - 1 (ten digits at most).
_ARRAY_INDEX = re.compile(r"0|[1-9][0-9]{0,9}")


def _is_array_index(key: object) -> bool:
    """Return whether the record key `key` is a JavaScript array index."""
    return (isinstance(key, str) and _ARRAY_INDEX.fullmatch(key) is not None
            and int(key) < 2**32 - 1)


def _js_ordered(value: object) -> object:
    """Return a copy of `value` whose records, at every depth, hold their keys in JS order.

    That is the array-index keys first, ascending, then the other keys in their order.
    """
    if isinstance(value, list):
        return [_js_ordered(item) for item in value]
    if isinstance(value, dict):
        keys = sorted(filter(_is_array_index, value), key=int)
        keys += [key for key in value if not _is_array_index(key)]
        return {key: _js_ordered(value[key]) for key in keys}
    return value


def _ts_join(value: object, sep: str = "\n") -> str:
    """Join a list's elements in their text form with `sep`, with `_join`'s checks."""
    expr_builtins._reject_null("join", value, "[]")
    if not isinstance(value, list):
        raise TypeError(f"join() takes a list, got {expr_builtins._kind(value)}")
    if any(x is None for x in value):
        raise TypeError("join() got a null element; drop or default it before joining")
    return write_as_text(lambda v: sep.join(_ts_text(x) for x in v), value)


def _ts_render_as_json(original):
    """Return `render_as_json` writing through `_ts_json` for an int `indent >= 0`.

    Any other indent goes to `original`: the design defines only `indent >= 0`.
    """
    @functools.wraps(original)
    def render_as_json(value: object, indent: int = 2) -> str:
        expr_builtins._reject_null("render_as_json", value)
        if type(indent) is int and indent >= 0:
            return write_as_text(lambda v: _ts_json(v, indent), value)
        return original(value, indent)

    return render_as_json


def _keep_text_builtins(patch: pytest.MonkeyPatch) -> None:
    """Make the installed text builtins `builtins._TEXT_BUILTINS`, so they still type as `str`."""
    fns = expr_builtins.TEMPLATE_FNS
    patch.setattr(expr_builtins, "_TEXT_BUILTINS",
                  frozenset(fns[name] for name in ("join", "render_as_json", "upper", "lower")))


def _ts_event_text(result: object) -> str | None:
    """Return a tool result's text for its `ToolCallFinished` event, in the text form.

    `None` when writing it raises, as `live._text` gives.
    """
    try:
        return _ts_text(result)
    except Exception:
        return None


def _install_text_form(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Write `${...}` spans, `join` elements, `render_as_json` and tool results in the text form.

    That covers a tool result's text both where the model reads it and in its event.
    """
    fns = expr_builtins.TEMPLATE_FNS
    # `_as_text` and `_tool_text` each write their value with `write_as_text(str, ...)`.
    for module in (template, memory):
        patch.setattr(module, "write_as_text",
                      lambda convert, value: write_as_text(_ts_text, value))
    # Only `observe_tool_call` calls it, through its module global.
    patch.setattr(live, "_text", _ts_event_text)
    patch.setitem(fns, "join", _ts_join)
    patch.setitem(fns, "render_as_json", _ts_render_as_json(fns["render_as_json"]))
    _keep_text_builtins(patch)


def _patch_held(patch: pytest.MonkeyPatch, original: object, wrapper: object) -> None:
    """Put `wrapper` in place of `original`, through `patch`, in every loaded yian module."""
    for module, name in _holders(original):
        patch.setattr(module, name, wrapper)


def _ts_value_repr(value: object) -> str:
    """Return `value` as `value_repr` shows it, but written by `_ts_json`, so text is JSON-quoted.

    An int longer than `YIAN_MAX_INT_DIGITS`, or a value holding one, gets `value_repr`'s
    placeholder. When writing it raises, the value is `<{type} object>`, or
    `<{type} too large to display>` for an int past Python's digit limit.
    """
    # Read before the `try`, so a bad `YIAN_*` value is raised, as `value_repr` raises it.
    defaults.current()
    try:
        if long_int_digits(value) is not None:
            # The long-int placeholders are the numbers rule's.
            return value_repr(value)
        text = _ts_json(value)
    except ValueError:
        return f"<{type(value).__name__} too large to display>"
    except Exception:
        return f"<{type(value).__name__} object>"
    return shown_text(text)


# The `value_repr` callers that pass it a name or a key, as `(path under src/yian, co_qualname)`.
# Design 3.2.8 keeps Python's repr for a name, so a message quoting one reads as in Python.
# The `_json_value_error` entry also covers its two value sites, which show only non-JSON values
# (no TS counterpart).
_NAME_CALLERS = frozenset({
    ("nodes/binding.py", "check_input_names.<locals>.<genexpr>"),
    ("nodes/contracts.py", "validate_output_contract.<locals>.<genexpr>"),
    ("typesys/values.py", "_build_for_type.<locals>.<genexpr>"),
    ("typesys/values.py", "reject_non_text_keys"),
    ("typesys/values.py", "plain_containers"),
    ("typesys/values.py", "_json_value_error"),
    ("typesys/values.py", "_dict_children"),
    ("typesys/values.py", "_nesting_segment"),
})


def _from_name_caller() -> bool:
    """Return whether the first calling frame outside this file is one of `_NAME_CALLERS`."""
    frame = sys._getframe(1)
    # Skip this file's frames, so a key-order wrapper in between doesn't hide the caller.
    while frame.f_code.co_filename == __file__:
        frame = frame.f_back
    code = frame.f_code
    path = code.co_filename.removeprefix(_YIAN_DIR).replace(os.sep, "/")
    return (path, code.co_qualname) in _NAME_CALLERS


def _install_messages(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Show values in yian's messages in the text form, and names with Python's repr."""
    original = typesys_values.value_repr

    @functools.wraps(original)
    def messages_value_repr(value: object) -> str:
        return original(value) if _from_name_caller() else _ts_value_repr(value)

    _patch_held(patch, original, messages_value_repr)


def _ordering(original):
    """Return `original` wrapped to take its first argument through `_js_ordered`."""
    @functools.wraps(original)
    def wrapper(value: object, *args: object, **kwargs: object) -> object:
        return original(_js_ordered(value), *args, **kwargs)

    return wrapper


def _ordering_unraised(original):
    """Return `original`, which must not raise, wrapped to take its argument through `_js_ordered`.

    A cyclic or too deep value, which `_js_ordered` can't copy, goes to `original` unordered.
    """
    @functools.wraps(original)
    def wrapper(value: object) -> object:
        try:
            ordered = _js_ordered(value)
        except RecursionError:
            return original(value)
        return original(ordered)

    return wrapper


def _install_key_order(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Hand the text points, tool-result events and `value_repr` each record in JS key order."""
    fns = expr_builtins.TEMPLATE_FNS
    for name in ("join", "render_as_json"):
        patch.setitem(fns, name, _ordering(fns[name]))
    patch.setattr(template, "_as_text", _ordering(template._as_text))
    tool_text = memory._tool_text

    @functools.wraps(tool_text)
    def ordered_tool_text(name: str, result: object) -> str:
        return tool_text(name, _js_ordered(result))

    patch.setattr(memory, "_tool_text", ordered_tool_text)
    patch.setattr(live, "_text", _ordering_unraised(live._text))
    _keep_text_builtins(patch)
    _patch_held(patch, typesys_values.value_repr, _ordering_unraised(typesys_values.value_repr))


def _ts_kind(value: object) -> str:
    """Name `value`'s kind as design 3.2.2's messages do: `null`, `bool`, `int`, `record`, ..."""
    if value is None:
        return "null"
    if isinstance(value, FileRef):
        return "file"
    return expr_builtins._kind(value)


def _shown(lhs: object, rhs: object) -> tuple[str, str]:
    """Return the operands as yian's operator messages show them."""
    # `_operand_repr` is what yian's own operator messages use; the `messages` shim reaches it
    # through `value_repr`.
    return expressions._operand_repr(lhs), expressions._operand_repr(rhs)


def _operand_kinds(op: str, lhs: object, rhs: object) -> expressions.ExpressionError:
    """Return the error for `lhs op rhs` on a pair of operand kinds `op` doesn't take."""
    l, r = _shown(lhs, rhs)
    return expressions.ExpressionError(
        f"type error evaluating {l} {op} {r}: unsupported operand kinds for `{op}`: "
        f"`{_ts_kind(lhs)}` and `{_ts_kind(rhs)}`")


def _division_by_zero(op: str, lhs: object, rhs: object) -> expressions.ExpressionError:
    """Return the error for `lhs op rhs` dividing by zero, worded as Python yian's."""
    l, r = _shown(lhs, rhs)
    return expressions.ExpressionError(f"division by zero: {l} {op} {r}")


def _is_number(value: object) -> bool:
    """Return whether arithmetic takes `value` as a number: an int, a float, or a bool (0 or 1)."""
    return isinstance(value, (int, float))


def _to_float(value: int | float) -> float:
    """Return the float nearest `value`, or a signed `inf` (noting `infinity`) past the range."""
    try:
        return float(value)
    except OverflowError:
        _NOTES.add("infinity")
        return math.inf if value > 0 else -math.inf


def _float_step(symbol: str, a: float, b: float) -> float:
    """Return IEEE 754's `a symbol b`, NaN where the operation has no numeric result."""
    try:
        if symbol == "%":
            return math.fmod(a, b)
        if symbol == "**":
            return math.pow(a, b)
        return expressions._ARITH_STEPS[symbol](a, b)
    except ValueError:
        return math.nan
    except OverflowError:
        # Only `math.pow` overflows; IEEE pow is negative then for a negative base and an odd
        # integral exponent.
        return -math.inf if a < 0 and b.is_integer() and b % 2 == 1 else math.inf


def _ts_apply_step(self, symbol: str, lhs: object, rhs: object) -> object:
    """Apply one operator of an arithmetic chain to two resolved values, by design 3.2.2."""
    if symbol == "%" and isinstance(lhs, str):
        _NOTES.add("printf")
        raise expressions.ExpressionError("text formatting with `%` was removed; use a template")
    if symbol == "+" and (isinstance(lhs, str) and isinstance(rhs, str)
                          or isinstance(lhs, list) and isinstance(rhs, list)):
        return lhs + rhs
    if not (_is_number(lhs) and _is_number(rhs)):
        if symbol == "*" and any(isinstance(x, (str, list)) and _is_number(y)
                                 for x, y in ((lhs, rhs), (rhs, lhs))):
            _NOTES.add("repetition")
        _NOTES.add("operand-kinds")
        raise _operand_kinds(symbol, lhs, rhs)
    if isinstance(lhs, int) and isinstance(rhs, int) and not (symbol == "**" and rhs < 0):
        if symbol in ("/", "%") and rhs == 0:
            raise _division_by_zero(symbol, lhs, rhs)
        if symbol == "/":
            try:
                return lhs / rhs
            except OverflowError:
                # The quotient is past the float range, and so is its floor, of the same sign.
                return _to_float(lhs // rhs)
        if symbol == "%":
            # Truncating, so the remainder has the sign of `lhs`.
            remainder = abs(lhs) % abs(rhs)
            return -remainder if lhs < 0 else remainder
        return expressions._ARITH_STEPS[symbol](lhs, rhs)
    a, b = _to_float(lhs), _to_float(rhs)
    if (symbol in ("/", "%") and b == 0) or (symbol == "**" and a == 0 and b < 0):
        raise _division_by_zero(symbol, lhs, rhs)
    result = _float_step(symbol, a, b)
    if math.isnan(result):
        l, r = _shown(lhs, rhs)
        raise expressions.ExpressionError(f"{symbol} on {l} and {r} has no numeric result")
    if math.isinf(result) and math.isfinite(a) and math.isfinite(b):
        _NOTES.add("infinity")
    return result


def _ts_neg(self, node: Tree) -> object:
    """Evaluate a unary minus node by design 3.2.2: a number's negation, a bool's as an int."""
    value = self._as_value(self._walk(node.children[0]))
    if _is_number(value):
        # `-True` is already the int -1.
        return -value
    _NOTES.add("operand-kinds")
    raise expressions.ExpressionError(
        f"unary minus on non-number {expressions._operand_repr(value)}: "
        f"unsupported operand kind for `-`: `{_ts_kind(value)}`")


def _install_arithmetic(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Make the arithmetic operators and unary minus follow design 3.2.2."""
    patch.setattr(expressions._EvalExpr, "_apply_step", _ts_apply_step)
    patch.setattr(expressions._EvalExpr, "_do_neg", _ts_neg)


def _ts_comparison(original):
    """Return `_eval_comparison` with `in` and the ordered comparisons of design 3.2.2."""
    @functools.wraps(original)
    def _eval_comparison(op: str, lhs: object, rhs: object) -> bool:
        if op in ("in", "not in"):
            if isinstance(lhs, str) and isinstance(rhs, str):
                # By code points, which is JS's code-unit test for well-formed text (lone
                # surrogates aren't yian values, design 3.2.1).
                contained = lhs in rhs
            elif isinstance(rhs, list):
                contained = any(lhs == x for x in rhs)
            elif isinstance(rhs, dict):
                contained = isinstance(lhs, str) and lhs in rhs
            else:
                raise _operand_kinds(op, lhs, rhs)
            return contained if op == "in" else not contained
        if op not in ("<", "<=", ">", ">=") or lhs is None or rhs is None:
            return original(op, lhs, rhs)
        if isinstance(lhs, str) and isinstance(rhs, str):
            # Big-endian UTF-16 orders as its code units.
            return original(op, lhs.encode("utf-16-be"), rhs.encode("utf-16-be"))
        if _is_number(lhs) and _is_number(rhs):
            return original(op, lhs, rhs)
        raise _operand_kinds(op, lhs, rhs)

    return _eval_comparison


def _install_compare(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Make `in` and the ordered comparisons follow design 3.2.2."""
    original = expressions._eval_comparison
    _patch_held(patch, original, _ts_comparison(original))


# Line breaks as PyYAML (YAML 1.1) reads them, and as YAML 1.2 does.
_PY_LINE_BREAK = re.compile("\r\n|[\r\n\x85\u2028\u2029]")
_TS_LINE_BREAK = re.compile("\r\n|[\r\n]")


def _ts_range(text: str, rng: list) -> list:
    """Return the Python source range `rng` in `text` as TS gives it.

    Python's `[start_line, start_col, end_line, end_col]` breaks lines at CRLF, CR, LF, NEL,
    U+2028 and U+2029 and counts columns in code points; TS breaks lines at CRLF, CR and LF
    and counts columns in UTF-16 code units. A position past its line keeps its distance from
    the line end; a line past the text is kept, and so is a range in a text starting with U+FEFF
    (such a case is the `yaml` rule's).
    """
    if text.startswith("\ufeff"):
        return list(rng)
    breaks = list(_PY_LINE_BREAK.finditer(text))
    starts = [0] + [match.end() for match in breaks]
    ends = [match.start() for match in breaks] + [len(text)]

    def position(line: int, column: int) -> list:
        if not (1 <= line <= len(starts) and column >= 1):
            return [line, column]
        width = ends[line - 1] - starts[line - 1]
        # A problem range at a line's end runs one column past it (`column, column + 1`).
        past = max(column - width - 1, 0)
        offset = starts[line - 1] + column - 1 - past
        before = [match.end() for match in _TS_LINE_BREAK.finditer(text, 0, offset)]
        start = before[-1] if before else 0
        return [len(before) + 1, len(text[start:offset].encode("utf-16-le")) // 2 + 1 + past]

    return position(*rng[:2]) + position(*rng[2:])


def _labelled(case: dict, label: str) -> str | None:
    """Return the text Python ran under a frame's `label`: the flow, or the one file so named."""
    if label == "flow.yaml":
        return case.get("python_flow", case.get("flow"))
    files = case.get("python_files", case.get("files", {}))
    # A child flow's label is its file name.
    named = [text for path, text in files.items() if Path(path).name == label]
    return named[0] if len(named) == 1 else None


def _install_columns(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Give every diagnostic frame's range in TS lines and columns, by its label's text."""
    diagnostic = runner._diagnostic

    @functools.wraps(diagnostic)
    def ts_diagnostic(diag: object) -> dict:
        form = diagnostic(diag)
        for frame in form.get("frames", []):
            text = _labelled(case, frame["label"])
            if text is not None:
                frame["range"] = _ts_range(text, frame["range"])
        return form

    # `_failure` calls `_diagnostic`, so run failures follow.
    patch.setattr(runner, "_diagnostic", ts_diagnostic)


def _install_dollar(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Make `!!timestamp` text match only to its end, not also before a trailing newline."""
    pattern = _FlowLoader.timestamp_regexp
    patch.setattr(_FlowLoader, "timestamp_regexp",
                  re.compile(pattern.pattern.removesuffix("$") + r"\Z", pattern.flags))


def _ts_intake(value: object) -> object:
    """Return `value` as a TS host reads it back: an integral float within ±(2^53-1) as an int.

    Lists and records are walked; `-0.0` reads as `0`, and `inf`, `nan` and anything else are
    kept. `value` itself comes back when nothing in it changes.
    """
    if type(value) is float:
        return int(value) if value.is_integer() and abs(value) <= 2**53 - 1 else value
    if type(value) is list:
        items = [_ts_intake(item) for item in value]
        return items if any(new is not old for new, old in zip(items, value)) else value
    if type(value) is dict:
        record = {key: _ts_intake(item) for key, item in value.items()}
        return record if any(record[key] is not value[key] for key in value) else value
    return value


def _install_code_intake(patch: pytest.MonkeyPatch, case: dict) -> None:
    """Read a CODE body's result as a TS host does."""
    forward = CodeNode.forward

    @functools.wraps(forward)
    def ts_forward(self: CodeNode, inputs: dict, **kwargs: object) -> object:
        output = forward(self, inputs, **kwargs)
        # The desugar names a value entry's one input after the node itself, so a CODE node
        # returning another input isn't skipped; an expanded id carries `ns/` and `#n`.
        own = self.id.rsplit("/", 1)[-1].split("#", 1)[0]
        if list(inputs) == [own] and self.code == f"return inputs[{own!r}]":
            # A value entry's value is authored, not a host's result.
            return output
        value = _ts_intake(output.value)
        return output if value is output.value else dataclasses.replace(output, value=value)

    patch.setattr(CodeNode, "forward", ts_forward)


# The shims by name, in install order: a later shim may wrap what an earlier one patched.
_SHIMS: dict[str, _Shim] = {
    "arithmetic": _Shim("arithmetic", _install_arithmetic),
    "compare": _Shim("compare", _install_compare),
    "text-form": _Shim("text-form", _install_text_form),
    "messages": _Shim("messages", _install_messages),
    # Record key order is in the text-form rule's section of the design.
    "key-order": _Shim("text-form", _install_key_order),
    "columns": _Shim("columns", _install_columns),
    # The timestamp pattern's `$` is the text rule's.
    "dollar": _Shim("text", _install_dollar),
    "code-intake": _Shim("numbers", _install_code_intake),
}
# What the shims noted during the latest replay.
_NOTES: set[str] = set()


def _noting_requests(original):
    """Return `runner._request_difference` wrapped to note a difference rather than report it.

    A request that differs from its entry's, or can't be compared with it, adds
    `request: <label>` to `_NOTES`, where `<label>` is the entry's,
    `host.models.<node>.<consumer>[<i>]`; the wrapper returns "".
    """
    def _request_difference(what: str, *args: object) -> str:
        if original(what, *args):
            # The runner's `served` passes `what` as `` `{label}[{index}].request` ``.
            _NOTES.add(f"request: {what.strip('`').removesuffix('.request')}")
        return ""

    return _request_difference


def _noting_int_in_float(original):
    """Return `_widen_int` wrapped to note `int-in-float` each time it widens an int to a float.

    It widens an int in a `float` slot and one in an untyped list holding floats. The value is
    still widened: TS keeps it an int, which no shim can follow, since pydantic widens it again
    inside `NumberValue` and `list[float]`.
    """
    @functools.wraps(original)
    def _widen_int(value: int, path: tuple) -> float:
        _NOTES.add("int-in-float")
        return original(value, path)

    return _widen_int


def _replay(state: _Harvest, case: dict, shims: Collection[str] = ()) -> tuple[dict, frozenset]:
    """Return the runner's results for the candidate `case`, read back from its file, and notes.

    The shims named in `shims` are on during the replay, and the notes are what they noted.
    Under any shim, a model call's request that differs from its entry's is noted instead of
    failing the replay: a shim may change a prompt or a tool result's text; and an int widened to
    a float (in a `float` slot or an untyped list holding floats) is noted `int-in-float`.
    Raises KeyError for a name with no shim, and `_Skip` when the replay raises anything.
    """
    unknown = set(shims) - _SHIMS.keys()
    if unknown:
        raise KeyError(f"no such shims: {sorted(unknown)}")
    text = _written(case)
    _NOTES.clear()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.yaml"
        path.write_text(text, encoding="utf-8")
        case_dir = Path(directory) / "case"
        case_dir.mkdir()
        state.disabled = True
        try:
            with pytest.MonkeyPatch.context() as patch:
                installed = [_SHIMS[name] for name in _SHIMS if name in shims]
                for shim in installed:
                    shim.install(patch, case)
                if installed:
                    patch.setattr(runner, "_request_difference",
                                  _noting_requests(runner._request_difference))
                    # Only `values` calls it, through its module global, and only for a
                    # non-bool int.
                    patch.setattr(typesys_values, "_widen_int",
                                  _noting_int_in_float(typesys_values._widen_int))
                results = runner.results(runner._read(path), case_dir, candidate=True)
            return results, frozenset(_NOTES)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # pytest's `Failed`, which the runner raises, is a BaseException.
            raise _Skip(f"replay failed: {type(exc).__name__}") from None
        finally:
            state.disabled = False
            defaults._forget()


def _agreed(case: dict, results: dict) -> dict:
    """Return what agreement compares of `results`.

    That is their `expect`, with a load error cut to its first diagnostic's code, message and
    frame ranges, and a run's cut to its `raises` or `output`, a failure to its code, message,
    node and frame ranges, and its backward outcomes to their `status` or `raises`.
    """
    agreed = _expect(case, results)
    load = agreed.get("load")
    if isinstance(load, dict):
        agreed["load"] = _cut_diagnostic(load["diagnostics"][0], ("code", "message"))
    if case["kind"] in ("run", "backward"):
        # The test's own calls give no trace, events or stream, and backward only a status.
        agreed = {key: agreed[key] for key in ("raises", "output", "backward") if key in agreed}
        output = agreed.get("output")
        if output is not None and output["status"] == "failed":
            agreed["output"] = {"status": "failed", "failure": _cut_diagnostic(
                output["failure"], ("code", "message", "node"))}
        if "backward" in agreed:
            agreed["backward"] = [outcome if "raises" in outcome else {"status": outcome["status"]}
                                  for outcome in agreed["backward"]]
    return agreed


def _cut_diagnostic(diagnostic: dict, keys: tuple) -> dict:
    """Return `diagnostic`'s `keys` and its frames' ranges.

    The labels are left out: the test names its flow, the runner's replay `flow.yaml`. A range
    still tells where yian placed the error, which a replayed error of a stand-in class can move.
    """
    return {**{key: diagnostic[key] for key in keys},
            "ranges": [frame["range"] for frame in diagnostic.get("frames", [])]}


def _leaves(value: object):
    """Yield every scalar in the case value `value`, mapping keys included."""
    if isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _leaves(item)
    else:
        yield value


def _texts(value: object):
    """Yield every text in the case value `value`, mapping keys included."""
    return (leaf for leaf in _leaves(value) if isinstance(leaf, str))


def _scan_secrets(case: dict, secrets: set) -> None:
    """Raise `_Skip` when one of `secrets` (those of 8 or more characters) is in a case input."""
    secrets = {secret for secret in secrets if len(secret) >= 8}
    for key in _INPUT_KEYS:
        if any(secret in text for text in _texts(case.get(key)) for secret in secrets):
            raise _Skip(f"secret in {key}")


def _group(case: dict) -> str:
    """Return the group a case is written in."""
    if case["kind"] == "load":
        return "load-agent" if _AGENT.search(case["flow"]) else "load-core"
    if case["kind"] == "run" and case.get("host", {}).get("models"):
        return "run-agent"
    return _GROUPS.get(case["kind"]) or ("templates" if "template" in case else "expr")


def _completed(call: _Call) -> tuple[dict, dict]:
    """Return a recorded call's case and result, a run's with its host and backward calls.

    They are added after the test: a backward call can come after its run returned.
    """
    case, result = call.case, call.result
    host = {key: value for key, value in (("models", call.models), ("jitter", call.jitter))
            if value}
    if host:
        case = case | {"host": host}
    if call.backward:
        case = case | {"kind": "backward", "backward": call.backward}
        result = result | {"backward": call.outcomes}
    return case, result


def _shimmed(state: _Harvest, case: dict, shims: Collection[str]) -> tuple[dict, frozenset]:
    """Return `_replay(state, case, shims)`.

    Raises `_Skip("shims: <reason>")` when the replay fails or leaves script entries.
    """
    try:
        results, notes = _replay(state, case, shims)
    except _Skip as skip:
        raise _Skip(f"shims: {skip}") from None
    if results.get("left"):
        raise _Skip("shims: replay left script entries")
    return results, notes


def _without_requests(case: dict, notes: frozenset) -> dict:
    """Return `case` without the `request` of each model entry a `request: <label>` note names."""
    host = case.get("host", {})
    if "models" not in host:
        return case
    # The labels are built from the case's own keys: a consumer such as `memory:<key>` may hold
    # a `.`, so a label can't be split.
    models = {}
    for node, consumers in host["models"].items():
        models[node] = {}
        for consumer, script in consumers.items():
            models[node][consumer] = [
                {key: value for key, value in entry.items() if key != "request"}
                if f"request: host.models.{node}.{consumer}[{index}]" in notes else entry
                for index, entry in enumerate(script)]
    return case | {"host": host | {"models": models}}


class _CodeScalar(NamedTuple):
    """A CODE node's `code:` value in a flow text.

    `node` is the node's id when its mapping sits under `nodes:`, else None; `value` the
    scalar's text; `start` and `end` its half-open span as indexes into the flow text; and
    `start_line` and `end_line` the 1-based lines, as frames count them, of its first and last
    characters.
    """

    node: str | None
    value: str
    start: int
    end: int
    start_line: int
    end_line: int


def _code_scalars(text: str) -> list[_CodeScalar]:
    """Return the `code:` scalars of the mappings with `kind: code` in a flow text, in order.

    The text is composed with yian's loader; a text that doesn't compose has none.
    """
    try:
        root = yaml.compose(text, Loader=_FlowLoader)
    except yaml.YAMLError:
        return []
    found = []
    # An alias shares its anchor's node, so a recursive one would walk forever; a node already
    # on the path is skipped, while a reused one elsewhere is walked again.
    path = set()

    def walk(node: yaml.Node | None, node_id: str | None) -> None:
        if id(node) in path:
            return
        path.add(id(node))
        try:
            visit(node, node_id)
        finally:
            path.discard(id(node))

    def visit(node: yaml.Node | None, node_id: str | None) -> None:
        if isinstance(node, yaml.SequenceNode):
            for item in node.value:
                walk(item, None)
            return
        if not isinstance(node, yaml.MappingNode):
            return
        fields = {key.value: value for key, value in node.value
                  if isinstance(key, yaml.ScalarNode)}
        kind, code = fields.get("kind"), fields.get("code")
        if (isinstance(kind, yaml.ScalarNode) and kind.value == "code"
                and isinstance(code, yaml.ScalarNode)):
            start, end = code.start_mark, code.end_mark
            # A block scalar's end mark is at the start of the line after its last one.
            last = end.line if end.column or end.line == start.line else end.line - 1
            found.append(_CodeScalar(node_id, code.value, start.index, end.index,
                                     start.line + 1, last + 1))
        for key, value in node.value:
            if key.value == "nodes" and isinstance(value, yaml.MappingNode):
                for child_key, child in value.value:
                    walk(child, child_key.value if isinstance(child_key, yaml.ScalarNode)
                         else None)
            else:
                walk(value, None)

    walk(root, None)
    return found


# The Python modules whose CODE functions have TS ports, by their file in `conformance/fixtures/`.
_FIXTURE_MODULES = {
    "tests.seeds.fns": "seeds.ts",
    "tests.engine._compose_codefns": "compose_codefns.ts",
}


@functools.cache
def _exported(file: str) -> frozenset[str]:
    """Return the names of the functions `conformance/fixtures/<file>` exports."""
    text = (runner._FIXTURES / file).read_text(encoding="utf-8")
    return frozenset(re.findall(r"\bexport (?:async )?function ([A-Za-z_]\w*)\b", text))


def _code_key(body: str) -> str:
    """Return the key of a Python CODE body in the translation table and the worklist."""
    return _digest(body)[:16]


def _fitted(span: str, body: str) -> str | None:
    """Return the TS `body` as the text replacing a Python body's scalar text `span`, or None.

    A `|` block keeps its header line and content indentation and is padded with empty lines to
    the span's line count; a one-line scalar takes a one-line body, double-quoted. None when the
    body doesn't fit: it has more lines, or the scalar is a one-line one and the body isn't, or
    the scalar is another multi-line one.
    """
    lines = body.rstrip("\n").split("\n")
    if "\n" not in span:
        return json.dumps(lines[0], ensure_ascii=False) if len(lines) == 1 else None
    header, _, content = span.partition("\n")
    indents = [line[:len(line) - len(line.lstrip(" "))]
               for line in content.split("\n") if line.strip()]
    # The span of a block that isn't at the end of the text ends with its last line's break.
    pad = span.count("\n") - len(lines)
    if not header.startswith("|") or not indents or pad < span.endswith("\n"):
        return None
    return header + "".join(f"\n{indents[0]}{line}" if line else "\n" for line in lines) + (
        "\n" * pad)


def _reads_as(text: str, start: int, end: int, value: str) -> bool:
    """Return whether the `code:` scalar at `start` in `text` reads as `value` within `end`.

    Trailing line breaks don't count, and the scalar may stop before blank text up to `end`.
    """
    found = [scalar for scalar in _code_scalars(text) if scalar.start == start]
    return (bool(found) and found[0].value.rstrip("\n") == value.rstrip("\n")
            and found[0].end <= end and not text[found[0].end:end].strip())


def _translate(text: str, depth: int, table: dict[str, str]) -> tuple[str, dict, set]:
    """Return a flow text with its CODE in TS form, and the bodies and references left.

    `depth` is how many directories below the case directory the text's file sits. A reference
    into `_FIXTURE_MODULES` to a function the fixture exports becomes `./fixtures/<file>:<name>`
    (`../` once per level of `depth`), and an inline body whose key is in `table` takes the TS
    body there, when it fits (`_fitted`); no line below either moves. Each replacement must read
    back as its TS form, or it is left. The bodies left are by key, the references as written.
    """
    untranslated, unported = {}, set()
    # From the end, so each span's indexes still hold; an alias's node is found again at its
    # anchor's span, which is replaced once.
    scalars = sorted({scalar.start: scalar for scalar in _code_scalars(text)}.values(),
                     key=lambda scalar: scalar.start, reverse=True)
    for scalar in scalars:
        span = text[scalar.start:scalar.end]
        reference = _REFERENCE_RE.fullmatch(scalar.value)
        if reference:
            module, _, name = scalar.value.partition(":")
            file = _FIXTURE_MODULES.get(module)
            value = (f"{'../' * depth or './'}fixtures/{file}:{name}"
                     if file is not None and name in _exported(file) else None)
            # A reference written as a block keeps the block's lines.
            new = None if value is None else value + "\n" * span.count("\n")
        else:
            value = table.get(_code_key(scalar.value))
            new = None if value is None else _fitted(span, value)
        candidate = None if new is None else text[:scalar.start] + new + text[scalar.end:]
        if candidate is not None and _reads_as(candidate, scalar.start, scalar.start + len(new),
                                               value):
            text = candidate
        elif reference:
            unported.add(scalar.value)
        else:
            untranslated[_code_key(scalar.value)] = scalar.value
    return text, untranslated, unported


def _ts_code(case: dict, table: dict[str, str]) -> tuple[dict, dict, set]:
    """Return `case` with the CODE of its flow and files translated (`_translate`), and what's left.

    A `flow` or `files` the translation changes keeps its Python form in `python_flow` or
    `python_files`. What's left is the bodies left untranslated, by key, and the references.
    """
    untranslated, unported = {}, set()

    def translate(text: str, depth: int) -> str:
        text, bodies, references = _translate(text, depth, table)
        untranslated.update(bodies)
        unported.update(references)
        return text

    if "flow" in case:
        flow = translate(case["flow"], 0)
        if flow != case["flow"]:
            case = case | {"flow": flow, "python_flow": case["flow"]}
    if "files" in case:
        files = {path: translate(text, len(PurePosixPath(path).parent.parts))
                 for path, text in case["files"].items()}
        if files != case["files"]:
            case = case | {"files": files, "python_files": case["files"]}
    return case, untranslated, unported


_INFINITY_TEXTS = ("non-finite number", "numbers must be finite", "int is too large for float",
                   "Result too large", "Numerical result out of range",
                   "Input should be a finite number")
_LONG_INT_TEXTS = ("IntTooLongError", "<int with", "larger than yian allows")
_INF_SPELLING = re.compile(r"\.(?:inf|Inf|INF)\b")
# Candidates only: `float()` decides which are numbers and which overflow.
_FLOAT_LITERAL = re.compile(r"[0-9.]+[eE][-+]?[0-9]+|[0-9]+\.[0-9]*")
_DATES = re.compile(r"\bsystem\.(?:now|today)\b")
_STRUCTURED_PROMPTS = ("Respond with ONLY a JSON object matching this JSON schema",
                       "Your previous output was invalid:")
# `\d` is Unicode category Nd: Python's expression lexer reads any decimal digit as a number;
# U+001C-U+001F and U+0085 are whitespace to Python, and U+FEFF is whitespace to JS.
_TEXT_CHARS = re.compile(r"(?![0-9])\d|[\x1c-\x1f\x85\ufeff]")
# Where PyYAML's error text marks its place; it can reach any message, such as a `call(...)`
# keyword value's or a skill frontmatter's error.
_YAML_MARK = 'in "<unicode string>", line'
# The arithmetic shim's notes for what TS rejects at load, where both operand types are known.
_LOAD_ARITHMETIC = frozenset({"repetition", "printf", "operand-kinds"})
# The failure codes whose text a CODE node's own behavior gives: its body raised (TS writes
# `<name>: <message>`) or its return failed the output check (TS names JS types). A CODE node's
# other failures, such as an input binding's, read the same in both.
_CODE_FAILURES = frozenset({"node.execute", "contract.output"})


def _diagnostics(expect: dict) -> list:
    """Return the diagnostics in an `expect`: a load's, a run's failure and its trace's failures."""
    load = expect.get("load")
    found = list(load["diagnostics"]) if isinstance(load, dict) else []
    failure = expect.get("output", {}).get("failure")
    if failure:
        found.append(failure)
    for entry in expect.get("trace", []):
        field = runner._FAILURE_FIELDS.get(entry[0])
        if field is not None and len(entry) > 2 and entry[2].get(field):
            found.append(entry[2][field])
    return found


def _overflows(literal: str) -> bool:
    """Return whether the number text `literal` reads as an infinite float."""
    try:
        return math.isinf(float(literal))
    except ValueError:
        return False


def _on_code_line(case: dict, frame: dict) -> bool:
    """Return whether a frame of Python's outcome is on a line holding a `code:` value."""
    text = _labelled(case, frame["label"])
    first, _, last, _ = frame["range"]
    return text is not None and any(scalar.start_line <= last and first <= scalar.end_line
                                    for scalar in _code_scalars(text))


def _unshimmable(case: dict, py: dict, notes: frozenset) -> set[str]:
    """Return the rules no shim covers that a case touches, by what it holds.

    `py` is the case's plain outcome and `notes` what the shims noted in its replays. The
    signals are loose: a rule found where it changes nothing only adds the case to the review
    list.
    """
    flows = [case.get(key) for key in ("flow", "python_flow")]
    flows += [text for key in ("files", "python_files") for text in case.get(key, {}).values()]
    flows = [text for text in flows if isinstance(text, str)]
    exprs = [case[key] for key in ("expr", "template") if isinstance(case.get(key), str)]
    texts = flows + exprs
    outcome = list(_texts(py))
    rules = set()

    if (any(type(x) is float and math.isinf(x) for part in (case, py) for x in _leaves(part))
            or any(_INF_SPELLING.search(text) for text in texts)
            or any(_overflows(literal) for text in texts
                   for literal in _FLOAT_LITERAL.findall(text))
            or any(part in text for text in outcome for part in _INFINITY_TEXTS)
            or "infinity" in notes):
        rules.add("infinity")

    if (any(part in text for text in outcome for part in _LONG_INT_TEXTS)
            or "YIAN_MAX_INT_DIGITS" in case.get("env", {})):
        rules.add("numbers")

    if "int-in-float" in notes:
        rules.add("int-in-float")

    if any(_DATES.search(text) for text in texts):
        rules.add("dates")

    diagnostics = _diagnostics(py)
    if (any(diagnostic["message"].startswith("invalid YAML:") for diagnostic in diagnostics)
            or any(_YAML_MARK in text for text in outcome)
            or any(text.startswith("\ufeff") for text in flows)):
        rules.add("yaml")

    entries = [(consumer, entry)
               for consumers in case.get("host", {}).get("models", {}).values()
               for consumer, script in consumers.items() for entry in script]
    prompts = [part.get("text", "") for _, entry in entries
               for message in entry.get("request", {}).get("messages", [])
               for part in message.get("parts", []) if part.get("type") == "text"]
    if (any(consumer == "structured_retry" or "schema" in entry.get("request", {})
            for consumer, entry in entries)
            or any(prompt.startswith(_STRUCTURED_PROMPTS) for prompt in prompts)
            or any(len(event) > 1 and str(event[1].get("purpose", "")).startswith("structured_")
                   for event in py.get("events", []))):
        rules.add("llm")

    code_ids = {scalar.node for text in flows for scalar in _code_scalars(text) if scalar.node}
    failed = [diagnostic["node"] for diagnostic in diagnostics
              if diagnostic.get("node") and diagnostic["code"] in _CODE_FAILURES]
    frames = [frame for diagnostic in diagnostics for frame in diagnostic.get("frames", [])]
    if (any(segment.split("#", 1)[0] in code_ids
            for node in failed for segment in node.split("/"))
            or any(_on_code_line(case, frame) for frame in frames)
            or any("`code:`" in diagnostic["message"] for diagnostic in diagnostics)):
        rules.add("code")

    # A flow's leading U+FEFF is a BOM, the yaml rule's; one anywhere else is the text rule's.
    data = [text for key in ("inputs", "values", "value") for text in _texts(case.get(key))]
    if any(_TEXT_CHARS.search(text) for text in [*(flow.removeprefix("\ufeff") for flow in flows),
                                                  *exprs, *data]):
        rules.add("text")
    return rules


def _expected(state: _Harvest, call: _Call) -> tuple[str, str, list, dict, set]:
    """Return a recorded call's group, case file text, flags, and CODE bodies and references left.

    The case's `expect` is from the runner. First its flow and files take their TS CODE
    (`_ts_code`), and the replays run the Python form. A body with no translation is flagged
    `code: untranslated`, one whose translation doesn't fit `code: translation doesn't fit`, and
    a reference left `code: unported reference`; the bodies left are returned by key.

    A pooled run is replayed three times, and its output or error, and its backward statuses,
    must agree in each. The call is then replayed as many times with every shim on, where they
    must agree too, and that outcome is the `expect`. When it differs from the plain one, the
    call is replayed once with each shim alone: the case's `rules` are those of the shims that
    change the outcome (none, and no rule `_unshimmable` finds, is a skip), and the plain outcome
    is its `python_expect`. A pooled run's trace, events, stream, tool calls and backward
    outcomes are kept only when the plain and shimmed runs succeeded and the replays of each are
    equal, each one dropped giving a `pooled: <result> dropped` flag. A model entry whose
    request the shims change loses its `request` (flag `shims: request dropped`). Each rule
    `_unshimmable` finds joins `rules` and is flagged `needs TS expect: <rule>`; a load, run or
    backward pass whose arithmetic TS rejects at load (repetition, `%` on text, operand kinds)
    gets the `arithmetic` rule and is flagged `needs TS expect: arithmetic`. A case whose plain
    outcome holds Lark's `Expected one of` token list is a skip.
    """
    if call.skip:
        raise _Skip(call.skip)
    if call.host_skip:
        raise _Skip(call.host_skip)
    if any(not entry for consumers in call.models.values() for script in consumers.values()
           for entry in script):
        raise _Skip("model: a call didn't end during its run")
    case, result = _completed(call)
    _scan_secrets(case, call.secrets)
    case, untranslated, unported = _ts_code(case, state.code)
    is_run = case["kind"] in ("run", "backward")
    pooled = is_run and case.get("run", {}).get("num_workers", _RUN_DEFAULTS["num_workers"]) != 0
    times = 3 if pooled else 1
    replays = [_replay(state, case)[0] for _ in range(times)]
    for actual in replays:
        if actual.get("left"):
            raise _Skip("replay left script entries")
        # A run's output is a host value, whose numbers the runner compares by value.
        if not runner._same(_agreed(case, actual), _agreed(case, result), by_value=is_run):
            raise _Skip("replay differs")
    py = _expect(case, replays[0])
    # Lark lists the expected tokens in an order PYTHONHASHSEED sets, so the text differs
    # across processes.
    if any("Expected one of" in text for text in _texts(py)):
        raise _Skip("unstable parser message (Lark token order)")
    shimmed, noted = zip(*(_shimmed(state, case, _SHIMS) for _ in range(times)))
    notes = frozenset().union(*noted)
    for actual in shimmed[1:]:
        if not runner._same(_agreed(case, actual), _agreed(case, shimmed[0]), by_value=is_run):
            raise _Skip("shims: replays differ")
    ts = _expect(case, shimmed[0])
    # Before a pooled result or a request is dropped: either may hold a rule's signal.
    unshimmable = _unshimmable(case, py, notes)
    flags = []
    if untranslated.keys() - state.code.keys():
        flags.append("code: untranslated")
    if untranslated.keys() & state.code.keys():
        flags.append("code: translation doesn't fit")
    if unported:
        flags.append("code: unported reference")
    if pooled:
        # When a pooled run fails, siblings racing the failure make its trace depend on timing
        # across processes, even when three replays in one process agree.
        succeeded = all(expect.get("output", {}).get("status") == "succeeded"
                        for expect in (py, ts))
        for key in _POOLED_RESULTS:
            stable = all(runner._same(actual.get(key), runs[0].get(key))
                         for runs in (replays, shimmed) for actual in runs[1:])
            if (key in py or key in ts) and not (succeeded and stable):
                py.pop(key, None)
                ts.pop(key, None)
                flags.append(f"pooled: {key} dropped")
    if any(note.startswith("request: ") for note in notes):
        case = _without_requests(case, notes)
        flags.append("shims: request dropped")
    rules = set()
    if not runner._same(ts, py):
        for name, shim in _SHIMS.items():
            alone = _expect(case, _shimmed(state, case, (name,))[0])
            # On the results `py` kept: a timing difference in this one replay of a pooled run
            # can at worst add a rule (the case keeps `python_expect`), not lose a case.
            if not runner._same({key: alone[key] for key in alone.keys() & py.keys()}, py):
                rules.add(shim.rule)
        # With the rules no shim covers, the case still goes to review by its flags.
        if not rules and not unshimmable:
            raise _Skip("shims: no single shim changes the outcome")
    rules |= unshimmable
    flags += [f"needs TS expect: {rule}" for rule in sorted(unshimmable)]
    if case["kind"] in ("load", "run", "backward") and notes & _LOAD_ARITHMETIC:
        rules.add("arithmetic")
        flags.append("needs TS expect: arithmetic")
    written = {**case, "expect": ts}
    if rules:
        written |= {"rules": sorted(rules), "python_expect": py}
    text = _written(written)
    if any(path in text for path in _machine_paths(state)):
        raise _Skip("machine path")
    return _group(case), text, flags, untranslated, unported


def _write(state: _Harvest, report: dict, call: _Call, path: str, test_id: str,
           number: int) -> None:
    """Write a recorded call's case as the test's `number`th record, counting it in `report`.

    A case written is listed under each of its flags, CODE bodies left untranslated and
    references left unported; a duplicate's are its holder's.
    """
    group, text, flags, untranslated, unported = _expected(state, call)
    name = _case_name(path, test_id, number)
    done = state.index.write(group, name, text)
    if done == "collision":
        name = _case_name(path, test_id, number, salted=True)
        done = state.index.write(group, name, text)
    if done == "collision":
        raise _Skip("name collision")
    if done == "duplicate":
        report["duplicates"] += 1
        return
    report["written"][group] = report["written"].get(group, 0) + 1
    for flag in flags:
        report["flags"].setdefault(flag, []).append(f"{group}/{name}")
    for key, body in untranslated.items():
        report["code"].setdefault(key, {"python": body, "cases": []})["cases"].append(
            f"{group}/{name}")
    for reference in sorted(unported):
        report["references"].setdefault(reference, []).append(f"{group}/{name}")


def _end_test(state: _Harvest) -> _Test:
    """Return the running test, leaving no test running and no texts noted."""
    test, state.test = state.test, None
    for table in (state.trees, state.types, state.registries, state.flows, state.results):
        table.clear()
    return test


def _report_path(state: _Harvest, path: str) -> Path:
    """Return where the report of the test file `path` goes."""
    return state.out / "report" / f"{path.replace('/', '__')}.json"


def _write_report(state: _Harvest, report: dict) -> None:
    """Write a test file's report, through a temporary file: a report marks the file done."""
    target = _report_path(state, report["file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _finish(state: _Harvest, item: pytest.Item) -> None:
    """End the running test: harvest its calls if it passed, and report its file if last.

    A test's path is the file's in its node id, relative to the rootdir. A file pytest's own
    selection narrowed is never reported, so a partial harvest doesn't close it.
    """
    test = _end_test(state)
    path, _, test_id = item.nodeid.partition("::")
    report = state.reports.setdefault(path, _new_report(path))
    report["tests"][test.outcome] += 1
    if test.outcome == "passed":
        # A load goes after the runs of its flow and is no case of its own once one of them is
        # written, since a run's case holds its load.
        calls = sorted(enumerate(test.calls, 1), key=lambda numbered: bool(numbered[1].runs))
        for number, call in calls:
            report["records"][call.entry] = report["records"].get(call.entry, 0) + 1
            if any(run.written for run in call.runs):
                continue
            try:
                _write(state, report, call, path, test_id, number)
                call.written = True
                continue
            except _Skip as skip:
                reason = str(skip)
            except Exception as exc:
                reason = _harvest_error(exc)
            ids = report["skips"].setdefault(reason, [])
            if test_id not in ids:
                ids.append(test_id)
    state.remaining[path] = state.remaining.get(path, 1) - 1
    if state.remaining[path] <= 0 and path not in state.narrowed:
        _write_report(state, report)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add `--harvest-out DIR`, which turns the harvest on, and `--harvest-code FILE`."""
    parser.addoption("--harvest-out", metavar="DIR",
                     help="record conformance case candidates under DIR")
    parser.addoption("--harvest-code", metavar="FILE",
                     help="translate the CODE bodies keyed in FILE (YAML: quoted key to TS body)")


def _code_table(path: str) -> dict[str, str]:
    """Return the `--harvest-code` table at `path`: a YAML mapping of body key to TS body.

    A key is quoted: unquoted hex digits can read as a number.

    Raises `pytest.UsageError` for a file that can't be read or holds anything else.
    """
    try:
        table = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=runner._CaseLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise pytest.UsageError(f"--harvest-code {path}: {exc}") from None
    if not isinstance(table, dict) or not all(
            isinstance(key, str) and isinstance(body, str) for key, body in table.items()):
        raise pytest.UsageError(f"--harvest-code {path}: not a mapping of key to TS body "
                                "(quote the keys)")
    return table


def pytest_configure(config: pytest.Config) -> None:
    """With `--harvest-out`, wrap the entry points and text helpers in every yian module.

    `RunResult.backward` is wrapped. The model factory and the TOOL backoff draw are wrapped,
    and so is each one a test assigns; the runner's seats name each model call's node and
    consumer. `--harvest-code` is read. `-n` is refused, and the temporary files a killed harvest
    left under `<out>` are removed.
    """
    global _STATE
    out = config.getoption("harvest_out")
    if not out:
        return
    if config.pluginmanager.hasplugin("xdist") and config.getoption("numprocesses", None):
        raise pytest.UsageError("--harvest-out doesn't support -n: run shards one after another")
    code = config.getoption("harvest_code")
    table = _code_table(code) if code else {}
    # A harvest killed between a write and its rename leaves the temporary file.
    for leftover in [*Path(out).glob("cases/*/*.yaml.tmp"), *Path(out).glob("report/*.json.tmp")]:
        leftover.unlink()
    _STATE = _Harvest(config, Path(out), table)
    entries = {
        "eval_expr": (eval_expr, _record_eval_expr),
        "eval_binding": (eval_binding, _record_eval_binding),
        "evaluate_when_record": (evaluate_when_record, _record_evaluate_when_record),
        "render_template_record": (render_template_record, _record_render_template_record),
        "build_value_as": (build_value_as, _record_build_value_as),
        "defaults.current": (defaults.current, _record_defaults),
        "load_flow": (load_flow, _record_load_flow),
        "run_flow": (run_flow, _record_run_flow),
        "resume_flow": (resume_flow, _record_resume_flow),
    }
    for entry, (original, recorder) in entries.items():
        _replace(original, _entry(entry, original, recorder), _STATE.replaced)
    for original, remember in ((parse_expr, _remember_tree), (type_for, _remember_type),
                               (read_typedefs, _remember_typedefs)):
        _replace(original, _mapper(original, remember), _STATE.replaced)
    runner._seated(_STATE.patch)
    RunResult.backward = _backward(_STATE.backward)
    for module in (llm_clients, tool_node):
        module.__class__ = _Watched
    llm_clients.model_from_config = _factory(_STATE.factory)
    tool_node._jitter = _draws(_STATE.jitter)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Put the originals back."""
    global _STATE
    if _STATE is None:
        return
    for module, name, original in _STATE.replaced:
        setattr(module, name, original)
    RunResult.backward = _STATE.backward
    for module in (llm_clients, tool_node):
        module.__class__ = ModuleType
    llm_clients.model_from_config = _STATE.factory
    tool_node._jitter = _STATE.jitter
    _STATE.patch.undo()
    _STATE = None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Deselect the tests of files with a report, and count each other file's selected tests.

    A file's report is written after its last selected test. A file named with `::` on the
    command line is narrowed.
    """
    if _STATE is None:
        return
    paths = {item.nodeid.partition("::")[0] for item in items}
    reported = {path for path in paths if _report_path(_STATE, path).exists()}
    deselected = [item for item in items if item.nodeid.partition("::")[0] in reported]
    if deselected:
        items[:] = [item for item in items if item.nodeid.partition("::")[0] not in reported]
        _STATE.deselecting = True
        try:
            config.hook.pytest_deselected(items=deselected)
        finally:
            _STATE.deselecting = False
    named = {(config.invocation_params.dir / arg.partition("::")[0]).resolve()
             for arg in config.args if "::" in arg}
    for item in items:
        path = item.nodeid.partition("::")[0]
        _STATE.remaining[path] = _STATE.remaining.get(path, 0) + 1
        if item.path.resolve() in named:
            _STATE.narrowed.add(path)


def pytest_deselected(items: list) -> None:
    """Note the files pytest's own selection (`-k`, `-m`, `--deselect`, `--lf`) narrowed."""
    if _STATE is None or _STATE.deselecting:
        return
    _STATE.narrowed.update(item.nodeid.partition("::")[0] for item in items)


def pytest_terminal_summary(terminalreporter) -> None:
    """Add up every report under `<out>/report`: tests, cases written, duplicates, flags, skips.

    A line counts the files this run harvested but didn't report, as its selection narrowed them.
    `<out>/worklist.yaml` is rewritten from the reports: the CODE bodies left untranslated by
    key (`code`), the references left unported (`references`), and the cases flagged
    `needs TS expect: <reason>` by reason (`expect`); a line counts each.
    """
    if _STATE is None:
        return
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((_STATE.out / "report").glob("*.json"))]
    tests, written, flags, skips, duplicates = Counter(), Counter(), Counter(), Counter(), 0
    code, references, expect = {}, {}, {}
    for report in reports:
        for key, entry in report["code"].items():
            code.setdefault(key, {"python": entry["python"], "cases": []})["cases"] += (
                entry["cases"])
        for reference, names in report["references"].items():
            references.setdefault(reference, []).extend(names)
        for flag, names in report["flags"].items():
            if flag.startswith("needs TS expect: "):
                expect.setdefault(flag.removeprefix("needs TS expect: "), []).extend(names)
        tests.update(report["tests"])
        written.update(report["written"])
        duplicates += report["duplicates"]
        flags.update({flag: len(names) for flag, names in report["flags"].items()})
        skips.update({reason: len(ids) for reason, ids in report["skips"].items()})
    write = terminalreporter.write_line
    terminalreporter.write_sep("-", "conformance harvest")
    write(f"files reported: {len(reports)}; tests: {tests['passed']} passed, "
          f"{tests['failed']} failed, {tests['skipped']} skipped")
    narrowed = _STATE.narrowed & _STATE.reports.keys()
    if narrowed:
        write(f"not reported (narrowed selection): {len(narrowed)} files")
    groups = ", ".join(f"{group} {n}" for group, n in sorted(written.items())) or "none"
    write(f"written: {groups}; {duplicates} duplicates")
    write("flags: " + (", ".join(f"{flag} {n}" for flag, n in sorted(flags.items())) or "none"))
    if skips:
        write("top skip reasons (tests):")
    for reason, n in skips.most_common(10):
        write(f"  {n:5}  {reason}")
    worklist = {"code": dict(sorted(code.items())),
                "references": dict(sorted(references.items())),
                "expect": dict(sorted(expect.items()))}
    _STATE.out.mkdir(parents=True, exist_ok=True)
    (_STATE.out / "worklist.yaml").write_text(
        yaml.dump(worklist, Dumper=_CaseDumper, sort_keys=False, allow_unicode=True,
                  width=math.inf), encoding="utf-8")
    write(f"worklist: code {len(code)}, references {len(references)}, "
          f"expect {sum(len(names) for names in expect.values())}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Record the test's calls while it runs; then harvest them if it passed."""
    if _STATE is None:
        return (yield)
    _STATE.test = _Test()
    try:
        result = yield
    except BaseException:
        # An interrupted test (Ctrl-C, `pytest.exit`) is neither harvested nor counted, so
        # its file's report isn't written as if the file were done.
        _end_test(_STATE)
        raise
    try:
        _finish(_STATE, item)
    except Exception as exc:
        # The file stays unreported, as its count of tests left never reaches zero or its
        # report wasn't written, and the session goes on.
        _end_test(_STATE)
        sys.stderr.write(f"conformance harvest: {item.nodeid}: {_harvest_error(exc)}\n")
    return result


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Note each phase's outcome for the running test."""
    test = _STATE.test if _STATE is not None else None
    if test is None:
        return
    if report.failed:
        test.failed = True
    elif report.skipped:
        test.skipped = True
    elif report.when == "call":
        test.passed = True
