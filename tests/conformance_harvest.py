"""Record conformance case candidates while the Python test suite runs.

A pytest plugin: while each test runs, it records the engine calls whose inputs can be written as
data (flow loads, runs, resumes, backward passes, expressions, templates, values and defaults).
Each passing test's calls become candidates; a candidate is replayed through the conformance
runner (`tests/test_conformance.py`) to get its `expect`, and written as a case file under
`<out>/cases/<group>/<name>.yaml`, deduplicated by content. Run it with:

    PYTHONPATH=src python -m pytest -p tests.conformance_harvest --harvest-out DIR tests

Design section 4.4 explains the harvest; `conformance/README.md` defines the case format.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading

from lark import Tree
import pytest
import yaml

import yian
from yian import defaults
from yian.expr.expressions import ResolveMode, eval_expr, evaluate_when_record
from yian.expr.grammar import parse_expr
from yian.expr.template import eval_binding, render_template_record
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import ValueKind, build_value_as

from tests import test_conformance as runner

# The order a case file's keys are written in. Other keys (the harvest writes none) follow in
# their given order.
_ORDER = ("kind", "flow", "files", "env", "inputs", "run", "host", "backward", "expr",
          "template", "values", "item", "mode", "typedefs", "type", "value", "expect")
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


def _dump(case: dict) -> str:
    """Return the case file text for `case`, whose values are plain Python.

    Raises ValueError when the text doesn't read back, through the runner's reader and
    `_decode`, to the same data.
    """
    ordered = sorted(case, key=lambda key: _ORDER.index(key) if key in _ORDER else len(_ORDER))
    limit = sys.get_int_max_str_digits()
    # As the runner's `_read`: case files have no digit limit.
    sys.set_int_max_str_digits(0)
    try:
        # A value holding itself recurses without end.
        written = {key: case[key] if key in runner._TEXT_KEYS else _encode(case[key])
                   for key in ordered}
        text = yaml.dump(written, Dumper=_CaseDumper, sort_keys=False, allow_unicode=True)
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
    `engine_expr_eval__TestA__x_a-b_`. A test's second and later cases end in `__<index>`; a
    `salted` name ends in 8 hex characters of the test id's hash, for two tests whose names
    are the same.
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


_HELD = "held: and/or result (owner deciding)"
_WORKTREE = Path(__file__).parent.parent
_YIAN_DIR = os.path.dirname(yian.__file__) + os.sep
# Calls from these files are the engine's own or the harvest's, never a test's.
_NOT_TEST = {runner.__file__, __file__}
_GROUPS = {"value": "values", "defaults": "defaults"}
# The harvest's state while it runs, None when the plugin is off.
_STATE: _Harvest | None = None


class _Skip(Exception):
    """A call or candidate that isn't written; its text is the report's reason."""


def _plain(value: object, what: str) -> object:
    """Return a copy of `value` when it is data; raise `_Skip("not data: <what>")` otherwise.

    Data is None, bools, ints, floats but NaN, text, lists, and mappings with text keys.
    """
    kind = type(value)
    if value is None or kind in (bool, int, str) or (kind is float and not math.isnan(value)):
        return value
    if kind is list:
        return [_plain(item, what) for item in value]
    if kind is dict and all(type(key) is str for key in value):
        return {key: _plain(item, what) for key, item in value.items()}
    raise _Skip(f"not data: {what}")


def _written(case: dict) -> str:
    """Return the case file text for `case`, or raise `_Skip` when it can't be written."""
    try:
        return _dump(case)
    except ValueError:
        raise _Skip("not data: case file") from None


def _env() -> dict:
    """Return the environment an `expr`, `value` or `defaults` call sees, as `{"env": ...}`.

    That is every `YIAN_*` variable but `YIAN_AUTH_FILE` and the model defaults (these kinds
    don't read them), and each connection variable whose value differs from the session's
    start, with `*_API_KEY` values as `<key:N>`. Empty: `{}`.
    """
    env = {}
    for name, value in sorted(os.environ.items()):
        if name.startswith("YIAN_"):
            keep = name not in ("YIAN_AUTH_FILE", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL")
        else:
            # The harvesting shell's own keys must never land in a case.
            keep = name.endswith(runner._ENV_SUFFIXES) and _STATE.start_env.get(name) != value
        if keep:
            env[name] = value
    keys: dict = {}
    for name, value in env.items():
        if name.endswith("_API_KEY"):
            env[name] = f"<key:{keys.setdefault(value, len(keys) + 1)}>"
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

    The runner's own `error` and `left` go, and `defaults` keeps the entries `env` sets.
    """
    expect = {key: value for key, value in results.items() if key not in ("error", "left")}
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


def _and_or(case: dict) -> bool:
    """Return whether an `expr` case's expressions use `and` or `or`."""
    try:
        trees = _trees(case)
    except Exception:
        # Text that doesn't parse holds no `and`/`or` the replay would reach.
        return False
    return any(isinstance(tree, Tree) and any(node.data in ("or_expr", "and_expr")
                                              for node in tree.iter_subtrees())
               for tree in trees)


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


def _merge(old: object, new: object) -> object:
    """Return `old` and `new` merged mapping by mapping; raise `_Skip` where they differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        return {**old, **{key: _merge(old[key], value) if key in old else value
                          for key, value in new.items()}}
    if runner._same(old, new):
        return old
    raise _Skip("resolver: conflicting answers")


def _values(answers: list) -> dict:
    """Return a resolver's answers nested as `{"values": ...}`; `a.b` gives `{a: {b: v}}`.

    A None answer is a miss and adds nothing. No answers: `{}`.
    """
    values: dict = {}
    for path, value in answers:
        if value is None:
            continue
        try:
            value = _plain(value, "resolver")
        except _Skip:
            raise _Skip("resolver: value is not data") from None
        for step in reversed(path.strip().split(".")):
            value = {step.strip(): value}
        values = _merge(values, value)
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
        return case, _result(outcome, lambda typed: {
            "value": _plain(typed.to_object(), "result"), "kind": typed.kind.value})

    return finish


def _record_defaults(arguments: dict):
    """Record a `defaults.current()` call as a `defaults` case."""
    case = {"kind": "defaults", **_env()}
    if not _set_entries(case):
        raise _Skip("no defaults variable set")

    def finish(outcome: tuple) -> tuple:
        return case, _result(outcome, lambda values: {"defaults": {
            entry.name: getattr(values, entry.name.lower()) for entry in defaults.ENTRIES}})

    return finish


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


def _begin_record(frame) -> bool:
    """Start recording a call from `frame` and return True, when it counts.

    It counts when a test's code made it while a test runs, the harvest isn't replaying, and
    no recorded call is active anywhere in the process.
    """
    state = _STATE
    if not _maps(frame):
        return False
    with state.lock:
        if state.depth:
            return False
        state.depth += 1
        return True


def _end_record() -> None:
    """End the recorded call `_begin_record` started."""
    with _STATE.lock:
        _STATE.depth -= 1


class _Call:
    """One recorded call: its entry point, candidate case and own result, or why it's skipped."""

    def __init__(self, entry: str) -> None:
        """Start the record of a call to `entry`."""
        self.entry = entry
        self.case: dict | None = None
        self.result: dict | None = None
        self.skip: str | None = "the call didn't finish"

    def finish(self, finish, outcome: tuple) -> None:
        """Take the case and result from `finish(outcome)`, or the reason it skips."""
        if finish is None:
            return
        try:
            self.case, self.result = finish(outcome)
            self.skip = None
        except _Skip as skip:
            self.skip = str(skip)


def _recorded(entry: str, original, recorder, signature, args: tuple, kwargs: dict) -> object:
    """Call `original` with `args` and `kwargs`, recording the call through `recorder`.

    `recorder(arguments)` reads the bound arguments (it may replace some, such as a resolver)
    and returns the step that turns the call's outcome into a case and result.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        # The call's own error, from the function itself.
        return original(*args, **kwargs)
    bound.apply_defaults()
    call = _Call(entry)
    _STATE.test.calls.append(call)
    finish = None
    try:
        finish = recorder(bound.arguments)
    except _Skip as skip:
        call.skip = str(skip)
    try:
        value = original(*bound.args, **bound.kwargs)
    except Exception as exc:
        call.finish(finish, ("raises", exc))
        raise
    call.finish(finish, ("value", value))
    return value


def _entry(entry: str, original, recorder):
    """Return `original` wrapped to record the calls that count (see `_begin_record`)."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args: object, **kwargs: object) -> object:
        if not _begin_record(sys._getframe(1)):
            return original(*args, **kwargs)
        try:
            return _recorded(entry, original, recorder, signature, args, kwargs)
        finally:
            _end_record()

    return wrapper


def _mapper(original, remember):
    """Return `original` wrapped to `remember(arguments, result)` for each test's call."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        if _maps(sys._getframe(1)):
            remember(signature.bind(*args, **kwargs).arguments, result)
        return result

    return wrapper


def _replace(original, wrapper, replaced: list) -> None:
    """Put `wrapper` in place of `original` in every loaded yian module that holds it.

    Each `(module, name, original)` replaced is added to `replaced`.
    """
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name == "yian" or module_name.startswith("yian.")):
            continue
        for name, value in list(vars(module).items()):
            if value is original:
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


def _new_report(path: str) -> dict:
    """Return an empty report for the test file `path`."""
    return {"file": path, "tests": {"passed": 0, "failed": 0, "skipped": 0}, "records": {},
            "written": {}, "duplicates": 0, "skips": {}, "flags": {}}


class _Harvest:
    """A harvest session: what it wrote, the test running now and its noted texts."""

    def __init__(self, config: pytest.Config, out: Path) -> None:
        """Start a harvest writing under `out`."""
        self.config, self.out = config, out
        self.index = _Index(out / "cases")
        self.start_env = dict(os.environ)
        self.lock = threading.Lock()
        self.depth = 0
        # Set while the harvest replays, so the runner's calls are never recorded.
        self.disabled = False
        self.test: _Test | None = None
        # Strong references, by id, for the test's duration: ids can be reused, and a lone
        # atom is a `Token`, which can't be weakly referenced.
        self.trees: dict = {}
        self.types: dict = {}
        self.registries: dict = {}
        self.remaining: dict[str, int] = {}
        self.reports: dict[str, dict] = {}
        self.replaced: list = []
        self._paths: set | None = None

    def machine_paths(self) -> set:
        """Return the paths a case must not hold: temp dirs, the worktree and home."""
        if self._paths is None:
            given = [tempfile.gettempdir(), self.config._tmp_path_factory.getbasetemp(),
                     _WORKTREE, Path.home()]
            self._paths = {str(form) for path in given
                           for form in (Path(path), Path(path).resolve())} - {os.sep}
        return self._paths

    def replay(self, case: dict) -> dict:
        """Return the runner's results for the candidate `case`, read back from its file.

        Raises `_Skip` when the replay raises anything.
        """
        text = _written(case)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.yaml"
            path.write_text(text, encoding="utf-8")
            case_dir = Path(directory) / "case"
            case_dir.mkdir()
            self.disabled = True
            try:
                return runner.results(runner._read(path), case_dir, candidate=True)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                # pytest's `Failed`, which the runner raises, is a BaseException.
                raise _Skip(f"replay failed: {type(exc).__name__}") from None
            finally:
                self.disabled = False
                defaults._forget()

    def expected(self, call: _Call) -> tuple[str, str]:
        """Return a recorded call's group and case file text, its `expect` from the runner."""
        if call.skip:
            raise _Skip(call.skip)
        case = call.case
        # Python's `and`/`or` give an operand, not a bool, and that decides conditions too.
        if case["kind"] == "expr" and _and_or(case):
            raise _Skip(_HELD)
        actual = self.replay(case)
        if actual.get("left"):
            raise _Skip("replay left script entries")
        expect = _expect(case, actual)
        if not runner._same(expect, _expect(case, call.result)):
            raise _Skip("replay differs")
        text = _written({**case, "expect": expect})
        if any(path in text for path in self.machine_paths()):
            raise _Skip("machine path")
        group = _GROUPS.get(case["kind"]) or ("templates" if "template" in case else "expr")
        return group, text

    def write(self, report: dict, call: _Call, path: str, test_id: str, number: int) -> None:
        """Write a recorded call's case as the test's `number`th, counting it in `report`."""
        group, text = self.expected(call)
        done = self.index.write(group, _case_name(path, test_id, number), text)
        if done == "collision":
            done = self.index.write(group, _case_name(path, test_id, number, salted=True), text)
        if done == "collision":
            raise _Skip("name collision")
        if done == "duplicate":
            report["duplicates"] += 1
        else:
            report["written"][group] = report["written"].get(group, 0) + 1

    def outcome(self, report: pytest.TestReport) -> None:
        """Note the outcome of one of the running test's phases."""
        test = self.test
        if test is None:
            return
        if report.failed:
            test.failed = True
        elif report.skipped:
            test.skipped = True
        elif report.when == "call":
            test.passed = True

    def finish(self, item: pytest.Item) -> None:
        """End the running test: harvest its calls if it passed, and report its file if last.

        A test's path is the file's in its node id, relative to the rootdir.
        """
        test, self.test = self.test, None
        for table in (self.trees, self.types, self.registries):
            table.clear()
        path, _, test_id = item.nodeid.partition("::")
        report = self.reports.setdefault(path, _new_report(path))
        report["tests"][test.outcome] += 1
        if test.outcome == "passed":
            for number, call in enumerate(test.calls, 1):
                report["records"][call.entry] = report["records"].get(call.entry, 0) + 1
                try:
                    self.write(report, call, path, test_id, number)
                except _Skip as skip:
                    ids = report["skips"].setdefault(str(skip), [])
                    if test_id not in ids:
                        ids.append(test_id)
        self.remaining[path] = self.remaining.get(path, 1) - 1
        if self.remaining[path] <= 0:
            target = self.out / "report" / f"{path.replace('/', '__')}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add `--harvest-out DIR`, which turns the harvest on."""
    parser.addoption("--harvest-out", metavar="DIR",
                     help="record conformance case candidates under DIR")


def pytest_configure(config: pytest.Config) -> None:
    """With `--harvest-out`, wrap the entry points and text helpers in every yian module."""
    global _STATE
    out = config.getoption("harvest_out")
    if not out:
        return
    _STATE = _Harvest(config, Path(out))
    entries = {
        "eval_expr": (eval_expr, _record_eval_expr),
        "eval_binding": (eval_binding, _record_eval_binding),
        "evaluate_when_record": (evaluate_when_record, _record_evaluate_when_record),
        "render_template_record": (render_template_record, _record_render_template_record),
        "build_value_as": (build_value_as, _record_build_value_as),
        "defaults.current": (defaults.current, _record_defaults),
    }
    for entry, (original, recorder) in entries.items():
        _replace(original, _entry(entry, original, recorder), _STATE.replaced)
    for original, remember in ((parse_expr, _remember_tree), (type_for, _remember_type),
                               (read_typedefs, _remember_typedefs)):
        _replace(original, _mapper(original, remember), _STATE.replaced)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Put the originals back."""
    global _STATE
    if _STATE is None:
        return
    for module, name, original in _STATE.replaced:
        setattr(module, name, original)
    _STATE = None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list) -> None:
    """Count each file's selected tests, so its report is written after its last one."""
    if _STATE is None:
        return
    for item in items:
        path = item.nodeid.partition("::")[0]
        _STATE.remaining[path] = _STATE.remaining.get(path, 0) + 1


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Record the test's calls while it runs; then harvest them if it passed."""
    if _STATE is None:
        return (yield)
    _STATE.test = _Test()
    try:
        return (yield)
    finally:
        _STATE.finish(item)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Note each phase's outcome for the running test."""
    if _STATE is not None:
        _STATE.outcome(report)
