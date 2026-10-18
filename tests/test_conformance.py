"""Run the language-neutral conformance cases in `conformance/cases/` against Python yian.

Each case file is one test, named by its id `<group>/<name>`. The test reads the case,
checks it against the closed format, sets up its directory and environment, calls yian
for the case's kind, and compares the results with the case's expectations.
`conformance/README.md` defines the case format and what each kind checks.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil

import pytest
import yaml
from yaml.constructor import ConstructorError

from yian import defaults
from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.compose.source.scalars import _BOOL, _FLOAT, _INT, _NULL, _FlowLoader
from yian.typesys.values import TypeCheckError

_ROOT = Path(__file__).resolve().parent.parent / "conformance"
_CASES = _ROOT / "cases"
_FIXTURES = _ROOT / "fixtures"


class _CaseLoader(yaml.SafeLoader):
    """SafeLoader that reads YAML 1.2 core scalars only and refuses duplicate or non-text keys."""

    # An own table, so SafeLoader's YAML 1.1 resolvers (yes/no, timestamps, merge keys)
    # never apply; add_implicit_resolver below copies nothing from it.
    yaml_implicit_resolvers: dict = {}
    # yian's flow-loader constructors check their text through this method.
    _core_text = _FlowLoader._core_text

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict:
        """Build a mapping, refusing a key that isn't text or that appears twice."""
        if not isinstance(node, yaml.MappingNode):
            raise ConstructorError(None, None, "expected a mapping", node.start_mark)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
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
_CaseLoader.add_constructor("tag:yaml.org,2002:bool", _FlowLoader.construct_yaml_bool)
_CaseLoader.add_constructor("tag:yaml.org,2002:int", _FlowLoader.construct_yaml_int)
_CaseLoader.add_constructor("tag:yaml.org,2002:float", _FlowLoader.construct_yaml_float)


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
_RUN_KEYS = {"run_id", "num_workers", "learnable", "llm_config", "context"}
_RESUME_KEYS = {"num_workers", "api_key", "headers", "context"}
_BACKWARD_KEYS = {"feedback", "api_key", "headers"}
_HOST_KEYS = {"tools", "human", "jitter", "models", "steering", "stop", "memory", "skills"}
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
_ENV_SUFFIXES = ("_API_KEY", "_BASE_URL", "_ENDPOINT", "_API_VERSION")


def _read(path: Path) -> dict:
    """Return the case file at `path` as read by `_CaseLoader`."""
    case = yaml.load(path.read_text(encoding="utf-8"), Loader=_CaseLoader)
    if not isinstance(case, dict):
        raise ValueError("a case file holds one mapping")
    return case


def _keys(where: str, mapping: object, allowed: set, known: set = frozenset(),
          required: tuple = ()) -> None:
    """Raise unless `mapping` is a mapping with only `allowed` keys and every `required` one.

    A key in `known` but not in `allowed` belongs to another kind of case.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"`{where or 'the case'}` must be a mapping")
    for key in mapping:
        name = f"{where}.{key}" if where else key
        if key in known and key not in allowed:
            raise ValueError(f"`{name}` is used with the wrong kind")
        if key not in allowed:
            raise ValueError(f"unknown key `{name}`")
    for key in required:
        if key not in mapping:
            raise ValueError(f"`{where}.{key}` is required" if where else f"`{key}` is required")


def _check_host(host: object, kind: str) -> None:
    """Raise unless `host`, its tools and their scripts use only the format's keys."""
    allowed = _HOST_KEYS | {"crash_after_record"} if kind == "resume" else _HOST_KEYS
    _keys("host", host, allowed, _HOST_KEYS | {"crash_after_record"})
    tools = host.get("tools", {})
    _keys("host.tools", tools, set(tools))
    for name, tool in tools.items():
        where = f"host.tools.{name}"
        _keys(where, tool, _TOOL_KEYS, required=("parameters", "results"))
        results = tool["results"]
        for script in results.values() if isinstance(results, dict) else [results]:
            if not isinstance(script, list):
                raise ValueError(f"`{where}.results` must be a list or a mapping of lists")
            for entry in script:
                _keys(f"{where}.results entry", entry, {"return", "raise"})
                if len(entry) != 1:
                    raise ValueError(f"a `{where}.results` entry takes `return` or `raise`")
                if "raise" in entry:
                    _keys(f"{where}.results raise", entry["raise"], _RAISE_KEYS,
                          required=("class", "message"))


def _check(case: dict) -> dict:
    """Return `case` after checking it against the closed format; raise naming the problem."""
    kind = case.get("kind")
    if kind not in _KIND_KEYS:
        raise ValueError(f"`kind` must be one of {', '.join(_KIND_KEYS)}, not {kind!r}")
    every_key = _COMMON.union(*_KIND_KEYS.values())
    _keys("", case, _COMMON | _KIND_KEYS[kind], every_key, required=("kind", "expect"))
    every_expect = set().union(*_EXPECT_KEYS.values())
    for key in ("expect", "python_expect"):
        if key in case:
            _keys(key, case[key], _EXPECT_KEYS[kind], every_expect)
    if "python_expect" in case and "rules" not in case:
        raise ValueError("`python_expect` needs `rules`")
    rules = case.get("rules", [])
    if not isinstance(rules, list) or any(not isinstance(rule, str) for rule in rules):
        raise ValueError("`rules` must be a list of rule ids")
    unknown = [rule for rule in rules if rule not in _RULES]
    if unknown:
        raise ValueError(f"unknown rule ids: {', '.join(unknown)}")
    for key in ("env", "files"):
        texts = case.get(key, {})
        _keys(key, texts, set(texts))
        if any(not isinstance(text, str) for text in texts.values()):
            raise ValueError(f"`{key}` values must be text")
    for key in ("flow", "python_flow", "other", "stdin"):
        if key in case and not isinstance(case[key], str):
            raise ValueError(f"`{key}` must be text")
    if "run" in case:
        _keys("run", case["run"], _RUN_KEYS)
    if "resume" in case:
        _keys("resume", case["resume"], _RESUME_KEYS)
    if "backward" in case:
        if not isinstance(case["backward"], list):
            raise ValueError("`backward` must be a list")
        for index, entry in enumerate(case["backward"]):
            _keys(f"backward[{index}]", entry, _BACKWARD_KEYS, required=("feedback",))
    if "host" in case:
        _check_host(case["host"], kind)
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
    if list(value) == ["$float"]:
        if value["$float"] not in ("inf", "-inf"):
            raise ValueError(f"`$float` takes inf or -inf, not {value['$float']!r}")
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


def _expect_raises(expected: dict, call) -> None:
    """Fail the case unless `call()` raises the error `expected` describes."""
    try:
        call()
    except Exception as exc:
        _compare("raises", _raised(exc), expected)
        return
    pytest.fail(f"expected {expected.get('class')} to be raised, but nothing was", pytrace=False)


def _load(text: str, case_dir: Path):
    """Load flow text as the case's `flow.yaml`, resolving paths in the case directory."""
    return load_flow(text, search_paths=[case_dir], source_label="flow.yaml")


def _run_defaults(case: dict, case_dir: Path) -> None:
    """Read yian's defaults and compare the listed entries, or the error."""
    expect = case["expect"]
    if "raises" in expect:
        _expect_raises(expect["raises"], defaults.current)
        return
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


_KINDS = {
    "defaults": _run_defaults,
    "fingerprint": _run_fingerprint,
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
