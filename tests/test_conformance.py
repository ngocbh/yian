"""Run the language-neutral conformance cases in `conformance/cases/` against Python yian.

Each case file is one test, named by its id `<group>/<name>`. `conformance/README.md` defines
the case format and what each kind checks. Set `CONFORMANCE_CASES` to run the cases under
another root instead.
"""

from __future__ import annotations

import builtins
import contextlib
from dataclasses import fields, is_dataclass
from functools import reduce
import hashlib
import json
import math
from operator import add
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import NoReturn

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
import pytest
import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError

from yian import defaults, llm_clients
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
from yian.llm.messages import (
    Message,
    MessageFormatError,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from yian.llm_clients.connection import check_model_config, resolve_model_config
from yian.nodes.agent import loop as agent_loop, node as agent_node
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
_CASES = Path(os.environ.get("CONFORMANCE_CASES") or _ROOT / "cases")
_FIXTURES = _ROOT / "fixtures"
_CLI_SCHEMA = json.loads((_ROOT / "schema" / "cli.schema.json").read_text(encoding="utf-8"))
Draft202012Validator.check_schema(_CLI_SCHEMA)
_CLI_VALIDATOR = Draft202012Validator(_CLI_SCHEMA)


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
    "cli": _FLOW | {"args", "stdin", "signal"},
    "json": {"read", "mode", "write", "form"},
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
    "json": (("value",), ("text",), ("raises",)),
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
_RESERVED_HOST_KEYS = ("steering", "stop", "memory", "skills")
_HOST_KEYS = {"tools", "human", "jitter", "models", *_RESERVED_HOST_KEYS}
# `host.models` consumers; `summary` and `memory:<key>` are reserved.
_CONSUMERS = ("agent", "structured_retry", "compose", "backward")
_USAGE_COUNTS = ("input_tokens", "output_tokens", "total_tokens")
# Each optional usage count, by where LangChain's `usage_metadata` holds it.
_USAGE_DETAILS = {
    "cache_read_tokens": ("input_token_details", "cache_read"),
    "cache_creation_tokens": ("input_token_details", "cache_creation"),
    "reasoning_tokens": ("output_token_details", "reasoning"),
}
_RESUME_HOST_KEYS = _HOST_KEYS | {"crash_after_record"}
_RULES = {
    "numbers", "arithmetic", "compare", "text-form", "canonical", "columns", "messages",
    "text", "yaml", "dates", "infinity", "llm", "code",
}
_RESERVED = [*(("host", key) for key in _RESERVED_HOST_KEYS), ("expect", "memory")]
# Keys holding text that is not a case value: no codec form, no number checks.
_TEXT_KEYS = {"flow", "python_flow", "other", "files", "python_files", "stdin", "read"}
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


def _check_usage(where: str, usage: object) -> None:
    """Raise unless `usage` is a usage: the three counts and any optional ones, as ints."""
    _keys(where, usage, {*_USAGE_COUNTS, *_USAGE_DETAILS}, required=_USAGE_COUNTS)
    if any(type(count) is not int or count < 0 for count in usage.values()):
        raise ValueError(f"`{where}` counts must be non-negative ints")


def _read_message(where: str, data: object) -> Message:
    """Return `data` read as a yian.message.v1 message, raising unless it is one."""
    try:
        return Message.from_dict(data)
    except MessageFormatError as error:
        raise ValueError(f"`{where}` is not a yian.message.v1 message: {error}") from None


def _check_reply(where: str, reply: object) -> None:
    """Raise unless `reply` is a yian.message.v1 assistant message this runner can serve.

    This runner serves only text and tool call parts.
    """
    message = _read_message(where, reply)
    if message.role != "assistant":
        raise ValueError(f"`{where}.role` must be `assistant`")
    for index, part in enumerate(message.parts):
        if not isinstance(part, (TextPart, ToolCallPart)):
            raise ValueError(f"`{where}.parts[{index}]`: this runner doesn't script "
                             f"`{reply['parts'][index]['type']}` parts yet")


def _check_chunk(where: str, chunk: object, last: bool) -> None:
    """Raise unless `chunk` is one text, tool call, usage or finish chunk; a finish comes last."""
    _keys(where, chunk, {"text", "tool_call", "usage", "finish"})
    if len(chunk) != 1:
        raise ValueError(f"`{where}` takes one of `text`, `tool_call`, `usage` and `finish`")
    (kind, value), = chunk.items()
    if kind == "text" and not isinstance(value, str):
        raise ValueError(f"`{where}.text` must be text")
    if kind == "tool_call":
        _keys(f"{where}.tool_call", value, {"index", "id", "name", "arguments"},
              required=("index", "arguments"))
        if type(value["index"]) is not int:
            raise ValueError(f"`{where}.tool_call.index` must be an int")
        if any(not isinstance(value.get(key, ""), str) for key in ("id", "name", "arguments")):
            raise ValueError(f"`{where}.tool_call` `id`, `name` and `arguments` must be text")
    if kind == "usage":
        _check_usage(f"{where}.usage", value)
    if kind == "finish" and not (value is None or isinstance(value, str)):
        raise ValueError(f"`{where}.finish` must be text or null")
    if kind == "finish" and not last:
        raise ValueError(f"`{where}`: `finish` must be the last chunk")


def _check_request(where: str, request: object) -> None:
    """Raise unless `request` is `{messages, tools, schema}` with `messages` required.

    `messages` are yian.message.v1 messages, `tools` a non-empty list of tool specs and
    `schema` a mapping.
    """
    _keys(where, request, {"messages", "tools", "schema"}, required=("messages",))
    for key in ("messages", "tools"):
        if not isinstance(request.get(key, []), list):
            raise ValueError(f"`{where}.{key}` must be a list")
    if request.get("tools") == []:
        raise ValueError(f"`{where}.tools` is empty; omit `tools` when none are bound")
    for index, message in enumerate(request["messages"]):
        _read_message(f"{where}.messages[{index}]", message)
    spec = ("name", "description", "parameters")
    for index, tool in enumerate(request.get("tools", [])):
        _keys(f"{where}.tools[{index}]", tool, set(spec), required=spec)
    _mapping(f"{where}.schema", request.get("schema", {}))


def _check_model_entry(where: str, entry: object) -> None:
    """Raise unless `entry` is a `reply` or `chunks` model entry.

    A `reply` may add a `usage`; `chunks` may add a `raise`, and without one end in a `finish`.
    Either may add a `request`.
    """
    _keys(where, entry, {"reply", "usage", "chunks", "raise", "request"})
    if ("reply" in entry) == ("chunks" in entry):
        raise ValueError(f"`{where}` takes `reply` or `chunks`")
    if "request" in entry:
        _check_request(f"{where}.request", entry["request"])
    if "reply" in entry:
        if "raise" in entry:
            raise ValueError(f"`{where}.raise` must come with `chunks`, not a `reply`")
        _check_reply(f"{where}.reply", entry["reply"])
        if "usage" in entry:
            _check_usage(f"{where}.usage", entry["usage"])
        return
    if "usage" in entry:
        raise ValueError(f"`{where}.usage` must come with a `reply`; `chunks` carry a usage chunk")
    chunks = entry["chunks"]
    if not isinstance(chunks, list):
        raise ValueError(f"`{where}.chunks` must be a list")
    for index, chunk in enumerate(chunks):
        _check_chunk(f"{where}.chunks[{index}]", chunk, index == len(chunks) - 1)
    if "raise" in entry:
        error = entry["raise"]
        _keys(f"{where}.raise", error, {"class", "message", "status"},
              required=("class", "message"))
        # The replay builds the error from these.
        if not isinstance(error["class"], str) or not isinstance(error["message"], str):
            raise ValueError(f"`{where}.raise.class` and `.message` must be text")
        if type(error.get("status", 0)) is not int:
            raise ValueError(f"`{where}.raise.status` must be an int")
    elif not chunks or "finish" not in chunks[-1]:
        raise ValueError(f"`{where}.chunks` must end in a `finish` chunk, unless a `raise` "
                         "follows them")


def _check_models(models: object) -> None:
    """Raise unless `host.models` maps node ids to consumers to lists of entries."""
    for node, consumers in _mapping("host.models", models).items():
        for consumer, script in _mapping(f"host.models.{node}", consumers).items():
            label = f"host.models.{node}.{consumer}"
            if consumer == "summary" or consumer.startswith("memory:"):
                raise ValueError(f"`{label}`: this runner doesn't script `{consumer}` calls yet")
            if consumer not in _CONSUMERS:
                raise ValueError(f"`{label}`: the consumer must be one of {', '.join(_CONSUMERS)}")
            if not isinstance(script, list):
                raise ValueError(f"`{label}` must be a list")
            for index, entry in enumerate(script):
                _check_model_entry(f"{label}[{index}]", entry)


def _check_host(host: object, kind: str) -> None:
    """Raise unless `host`, its tools, models and their scripts use only the format's keys."""
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
    _check_models(host.get("models", {}))
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


def _check(case: dict, candidate: bool = False) -> dict:
    """Return `case` after checking it against the closed format; raise naming the problem.

    A `candidate` is a case without its expectations yet: `expect` and `python_expect` aren't
    required or checked.
    """
    kind = case.get("kind")
    if kind not in _KIND_KEYS:
        raise ValueError(f"`kind` must be one of {', '.join(_KIND_KEYS)}, not {kind!r}")
    expects = () if candidate else ("expect", "python_expect")
    _keys("", case, _COMMON | _KIND_KEYS[kind], _ALL_KEYS, kind,
          required=("kind", *expects[:1], *_REQUIRED.get(kind, ())))
    for key in expects:
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
    if "signal" in case:
        _keys("signal", case["signal"], {"name", "after"}, required=("name", "after"))
        if case["signal"]["name"] != "SIGINT":
            raise ValueError("`signal.name` must be SIGINT")
        if not isinstance(case["signal"]["after"], str) or not case["signal"]["after"]:
            raise ValueError("`signal.after` must be non-empty text")
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
    if case["kind"] == "json":
        return [pytest.mark.skip(reason="json cases check TS yian's own JSON reader and encoders")]
    if "python_skip" in case:
        return [pytest.mark.skip(reason=str(case["python_skip"]))]
    if case.get("rules") and "python_expect" not in case:
        rules = ", ".join(case["rules"])
        return [pytest.mark.skip(reason=f"depends on TS rules ({rules}) and has no python_expect")]
    return []


def _params() -> list:
    """Return one pytest param per case file, in sorted order."""
    # A mistyped root would otherwise collect nothing and pass.
    if not _CASES.is_dir():
        raise ValueError(f"no case root at {_CASES}")
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


def _difference(what: str, actual: object, expected: object, **flags: bool) -> str:
    """Return both sides as YAML, or "" when `actual` is the same as `expected`."""
    if _same(actual, expected, **flags):
        return ""
    dump = [yaml.safe_dump(side, sort_keys=False, allow_unicode=True, width=1000)
            for side in (actual, expected)]
    return f"{what} differs\n--- actual\n{dump[0]}--- expected\n{dump[1]}"


def _compare(what: str, actual: object, expected: object, **flags: bool) -> None:
    """Fail the case, showing both sides as YAML, unless `actual` is the same as `expected`."""
    difference = _difference(what, actual, expected, **flags)
    if difference:
        _fail(difference)


def _raised(exc: Exception) -> dict:
    """Return an error in the case's `raises` form."""
    raised = {"class": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, TypeCheckError):
        raised["path"] = list(exc.path)
    return raised


def _catching(compute) -> dict:
    """Return the results `compute()` returns by name, or its error as `raises` and `error`.

    `raises` is the error's `raises` form and `error` the error itself, never compared.
    """
    try:
        return compute()
    except Exception as exc:
        return {"raises": _raised(exc), "error": exc}


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


def _actual_load(case: dict, case_dir: Path) -> dict:
    """Load `flow` and return `load`, `ok` or its diagnostics."""
    try:
        _load(case["flow"], case_dir)
    except LoadError as exc:
        return {"load": {"diagnostics": [_diagnostic(exc.diagnostic)]}}
    return {"load": "ok"}


def _actual_defaults(case: dict, case_dir: Path) -> dict:
    """Read yian's defaults and return every entry as `defaults`, or the error."""
    def compute() -> dict:
        values = defaults.current()
        return {"defaults": {entry.name: getattr(values, entry.name.lower())
                             for entry in defaults.ENTRIES}}

    return _catching(compute)


def _actual_fingerprint(case: dict, case_dir: Path) -> dict:
    """Check that `flow` fingerprints the same twice; return whether `other`'s is the same."""
    def fingerprint(text: str) -> str:
        return flow_fingerprint(_load(text, case_dir).compiled)

    first = fingerprint(case["flow"])
    if fingerprint(case["flow"]) != first:
        _fail("two loads of `flow` have different fingerprints")
    same = fingerprint(case["other"]) == first
    return {"fingerprint": "same" if same else "different"}


def _actual_expr(case: dict, case_dir: Path) -> dict:
    """Evaluate `expr` or `template` in `mode` over `values` and `item`; return the result."""
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

    return _catching(lambda: {"value": evaluate()})


def _actual_value(case: dict, case_dir: Path) -> dict:
    """Build `value` as `type` and return the stored value and its kind, or the error."""
    def compute() -> dict:
        typ = type_for(case["type"], read_typedefs(case.get("typedefs") or {}))
        typed = build_value_as(typ, case["value"])
        return {"value": typed.to_object(), "kind": typed.kind.value}

    return _catching(compute)


_SEAT = threading.local()


def _seated(patch: pytest.MonkeyPatch) -> None:
    """Wrap the functions that make model calls, so `_take_model_key()` names each call's seat.

    `run_agent` seats its node's `agent` calls (`compose` for a `<gate>__compose` node),
    `backward_agent` its node's `backward` calls and `prepare_history` `summary` calls;
    `generate_structured` keeps its caller's seat and starts counting the calls made under it.
    """
    def wrap(module: object, name: str, make_seat) -> None:
        inner = getattr(module, name)

        def seated(*args: object, **kwargs: object) -> object:
            outer = getattr(_SEAT, "seat", None)
            _SEAT.seat = make_seat(outer, *args)
            try:
                return inner(*args, **kwargs)
            finally:
                _SEAT.seat = outer

        patch.setattr(module, name, seated)

    wrap(agent_node, "run_agent", lambda outer, ctx: {
        "node": ctx.node_id,
        "consumer": "compose" if ctx.node_id.endswith("__compose") else "agent",
    })
    wrap(agent_node, "backward_agent",
         lambda outer, context, *rest: {"node": context.node_id, "consumer": "backward"})
    wrap(agent_loop, "generate_structured",
         lambda outer, *rest: None if outer is None else {**outer, "calls": 0})
    wrap(agent_loop, "prepare_history",
         lambda outer, *rest: None if outer is None else {**outer, "consumer": "summary"})


def _take_model_key() -> tuple[str, str] | None:
    """Take the model call being made: return its `(node, consumer)`, or None outside any seat.

    Call it once per model call: under `generate_structured` each call is counted, and the
    first keeps its caller's consumer while each later one is a `structured_retry`.
    """
    seat = getattr(_SEAT, "seat", None)
    if seat is None:
        return None
    consumer = seat["consumer"]
    if "calls" in seat:
        if seat["calls"]:
            consumer = "structured_retry"
        seat["calls"] += 1
    return seat["node"], consumer


def _usage_metadata(usage: dict) -> dict:
    """Return a case usage as LangChain's `usage_metadata`."""
    metadata = {name: usage[name] for name in _USAGE_COUNTS}
    for name, (details, key) in _USAGE_DETAILS.items():
        if name in usage:
            metadata.setdefault(details, {})[key] = usage[name]
    return metadata


def _langchain_reply(entry: dict) -> AIMessage:
    """Return a `reply` entry as the `AIMessage` a provider client returns."""
    message = Message.from_dict(entry["reply"])
    calls, invalid = [], []
    for part in message.tool_calls():
        if part.arguments is not None:
            calls.append({"name": part.name, "args": part.arguments, "id": part.id,
                          "type": "tool_call"})
        else:
            # Arguments that didn't parse, as a client reports them.
            invalid.append({"name": part.name, "args": part.raw_arguments, "id": part.id,
                            "error": None, "type": "invalid_tool_call"})
    usage = _usage_metadata(entry["usage"]) if "usage" in entry else None
    return AIMessage(content=message.text(), tool_calls=calls, invalid_tool_calls=invalid,
                     usage_metadata=usage)


def _langchain_chunk(chunk: dict) -> AIMessageChunk:
    """Return a case chunk as the `AIMessageChunk` a provider client streams."""
    if "text" in chunk:
        return AIMessageChunk(content=chunk["text"])
    if "tool_call" in chunk:
        delta = chunk["tool_call"]
        return AIMessageChunk(content="", tool_call_chunks=[{
            "index": delta["index"], "id": delta.get("id"), "name": delta.get("name"),
            "args": delta["arguments"],
        }])
    if "usage" in chunk:
        return AIMessageChunk(content="", usage_metadata=_usage_metadata(chunk["usage"]))
    reason = chunk["finish"]
    return AIMessageChunk(content="",
                          response_metadata={} if reason is None else {"finish_reason": reason})


class _NotData(Exception):
    """A LangChain message the case format can't hold; its text says which and why."""


_ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant",
          ToolMessage: "tool"}


def _v1_parts(message: BaseMessage, role: str, calls: dict) -> list:
    """Return the yian.message.v1 parts of a LangChain message with text content.

    An `AIMessage`'s tool calls are added to `calls`, which maps call ids to tool names, and a
    `ToolMessage` takes its result's name from there.
    """
    if role == "tool":
        if message.tool_call_id not in calls:
            raise _NotData(f"ToolMessage for tool call {message.tool_call_id!r}, which no "
                           "earlier AIMessage made")
        return [ToolResultPart(message.tool_call_id, calls[message.tool_call_id],
                               message.content, is_error=message.status == "error")]
    parts = [TextPart(message.content)] if message.content else []
    if role == "assistant":
        for call in message.tool_calls:
            parts.append(ToolCallPart(call["id"], call["name"], call["args"]))
            calls[call["id"]] = call["name"]
        for call in message.invalid_tool_calls:
            parts.append(ToolCallPart(call["id"], call["name"], None, call["args"]))
            calls[call["id"]] = call["name"]
    return parts


def _v1_message(message: BaseMessage, calls: dict) -> dict:
    """Return a LangChain message as a yian.message.v1 message.

    `calls` maps the tool call ids of earlier messages to their tool names; this message's
    calls are added. Raises `_NotData` for a message the format can't hold: one that isn't a
    system, human, AI or tool message, that has list content or `additional_kwargs` (where
    reasoning travels), whose parts yian.message.v1 refuses, or a tool message answering a
    call no earlier message made.
    """
    name = type(message).__name__
    role = _ROLES.get(type(message))
    if role is None:
        raise _NotData(f"{name}, which has no yian.message.v1 role")
    if not isinstance(message.content, str):
        raise _NotData(f"{name} with list content")
    if message.additional_kwargs:
        raise _NotData(f"{name} with additional_kwargs "
                       f"({', '.join(sorted(message.additional_kwargs))})")
    try:
        return Message(role, _v1_parts(message, role, calls)).to_dict()
    except MessageFormatError as error:
        raise _NotData(f"{name} that yian.message.v1 refuses ({error})") from None


def _v1_request(messages: list, tools: list, schema: type | None) -> dict:
    """Return a model call's request in the case's form.

    Its `messages` as yian.message.v1 messages; the bound `tools` as
    `convert_to_openai_tool` gives them, unless none are bound; and a structured call's
    pydantic `schema` as the JSON Schema LangChain sends for it, unless it is None. Raises
    `_NotData` as `_v1_message` does.
    """
    calls: dict = {}
    request: dict = {"messages": [_v1_message(message, calls) for message in messages]}
    if tools:
        request["tools"] = [convert_to_openai_tool(tool)["function"] for tool in tools]
    if schema is not None:
        request["schema"] = convert_to_openai_tool(schema)["function"]["parameters"]
    return request


def _request_difference(what: str, expected: dict, messages: list, tools: list,
                        schema: type | None) -> str:
    """Return how a model call's request differs from `expected`, or "" when it doesn't."""
    try:
        actual = _v1_request(messages, tools, schema)
    except _NotData as reason:
        return f"{what} can't be compared: {reason}"
    return _difference(what, actual, expected)


def _raise(error: dict) -> NoReturn:
    """Raise a model call's recorded error.

    Its class is the builtin of that name when that is an `Exception`, else a new class of that
    name; a `status` becomes the error's `status_code`.
    """
    cls = getattr(builtins, error["class"], None)
    if not (isinstance(cls, type) and issubclass(cls, Exception)):
        cls = type(error["class"], (Exception,), {})
    exc = cls(error["message"])
    if "status" in error:
        exc.status_code = error["status"]
    raise exc


def _scripted_model(take, problems: list) -> object:
    """Return a chat model that serves each call the next `host.models` entry of its seat.

    Each `invoke` or `stream` call takes the entry for `_take_model_key()` through `take`. `invoke`
    returns the reply, or the chunks summed; `stream` yields the reply as one chunk, or the
    chunks. An entry with `raise` raises after its chunks. `bind_tools` returns a model that
    sends its tools, and `with_structured_output` a `Runnable` that returns the reply's text
    read as its schema. A call's request that differs from its entry's `request` is added to
    `problems`.
    """
    taken: dict = {}

    def served(messages: list, tools: list, schema: type | None) -> dict:
        key = _take_model_key()
        if key is None:
            return take(None, "a model call was made outside any AGENT (no seat)")
        node, consumer = key
        label = f"host.models.{node}.{consumer}"
        entry = take(label, f"a model call for `{consumer}` from node {node!r}")
        index = taken[label] = taken.get(label, -1) + 1
        if "request" in entry:
            difference = _request_difference(f"`{label}[{index}].request`", entry["request"],
                                             messages, tools, schema)
            if difference:
                problems.append(difference)
        return entry

    def replay(entry: dict):
        if "reply" in entry:
            reply = _langchain_reply(entry)
            yield AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls,
                                 invalid_tool_calls=reply.invalid_tool_calls,
                                 usage_metadata=reply.usage_metadata)
            return
        yield from map(_langchain_chunk, entry["chunks"])
        if "raise" in entry:
            _raise(entry["raise"])

    def summed(entry: dict) -> AIMessage:
        return message_chunk_to_message(reduce(add, replay(entry)))

    class ScriptedModel:
        def __init__(self, tools: list) -> None:
            self.tools = tools

        # No other options: one the runner doesn't script fails as a TypeError naming it.
        def bind_tools(self, tools: list) -> ScriptedModel:
            return ScriptedModel(list(tools))

        def with_structured_output(self, schema: type) -> Runnable:
            return StructuredModel(schema)

        def invoke(self, messages: list, **kwargs: object) -> AIMessage:
            return summed(served(messages, self.tools, None))

        def stream(self, messages: list, **kwargs: object):
            # The entry is taken when the call is made, not when its first chunk is read.
            return replay(served(messages, self.tools, None))

    # A Runnable, since yian hands its usage callback only to one.
    class StructuredModel(Runnable):
        def __init__(self, schema: type) -> None:
            self.schema = schema

        def invoke(self, messages: list, config: dict | None = None, **kwargs: object) -> object:
            reply = summed(served(messages, [], self.schema))
            # What a chat model reports to the callbacks, which is where yian reads the usage.
            result = LLMResult(generations=[[ChatGeneration(message=reply)]])
            for handler in (config or {}).get("callbacks", []):
                handler.on_llm_end(result)
            return self.schema.model_validate(json.loads(reply.content))

    return ScriptedModel([])


def _host(case: dict, patch: pytest.MonkeyPatch) -> tuple:
    """Build the case's host and return `(tools, calls, check)`.

    `tools` holds a stub per `host.tools` entry and `ask_user` for `host.human`; each call is
    recorded in `calls`, by tool, and takes its script's next entry, as each TOOL backoff takes
    the next `host.jitter` draw and each model call the next `host.models` entry of its node and
    consumer. Every model yian builds is the scripted one, after yian resolves and checks its
    config. `check()` fails the case when a script ran out or was missing or a model call's
    request differed from its entry's, and returns a message per script with entries left.
    """
    host = case.get("host", {})
    calls: dict = {}
    problems: list = []
    scripts = {f"host.human.{node}": list(answers)
               for node, answers in host.get("human", {}).items()}

    def take(label: str | None, what: str) -> object:
        if label is None:
            # The call has no script to take from; `what` says why.
            problems.append(what)
        elif label not in scripts:
            problems.append(f"{what} found no `{label}` list")
        elif not scripts[label]:
            problems.append(f"{what} found `{label}` used up")
        else:
            return scripts[label].pop(0)
        # Permanent, so a TOOL node doesn't retry the overrun; a model call's overrun fails its
        # AGENT. `check()` reports it either way.
        raise PermanentToolError(what if label is None else f"`{label}` is used up")

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
    scripts.update({f"host.models.{node}.{consumer}": list(script)
                    for node, consumers in host.get("models", {}).items()
                    for consumer, script in consumers.items()})
    model = _scripted_model(take, problems)

    def model_from_config(config: object) -> object:
        # The script replaces only the provider: the config is resolved and checked as yian's
        # own factory does, so its errors stay yian's.
        check_model_config(resolve_model_config(config))
        return model

    # Installed even without `host.models`, so an unscripted model call never reaches a provider.
    patch.setattr(llm_clients, "model_from_config", model_from_config)
    _seated(patch)

    def check() -> list:
        if problems:
            _fail("\n".join(problems))
        return [
            f"`{label}` has {len(script)} {'entry' if len(script) == 1 else 'entries'} left"
            for label, script in scripts.items() if script
        ]

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


def _actual_run(case: dict, case_dir: Path) -> dict:
    """Run the flow on the case's host and return its results.

    A `resume` case's first run is cut after record `host.crash_after_record` and resumed; a
    `backward` case then calls backward once per `backward` entry. The results are `trace`,
    `events`, `stream`, `tool_calls`, `backward` for a `backward` case, `output` or the host
    error as `raises` and `error` (as `_catching` gives them), and `left`, a message per script
    with entries left.
    """
    flow = _load(case["flow"], case_dir)
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

        result, caught = None, {}
        try:
            result = start()
        except Exception as exc:
            caught = {"raises": _raised(exc), "error": exc}

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
        left = check()
        text = log.read_text(encoding="utf-8") if log.exists() else ""
        lines = text.splitlines()
        # In placeholder order: ids are numbered walking the results in this order.
        actual = {
            "trace": _trace(lines, flow),
            "events": [_event(event, flow) for event in events],
            "stream": [_fields(chunk, flow) for chunk in chunks],
            "tool_calls": calls,
            **({"output": _output(result, flow)} if result is not None else {}),
            **({"backward": outcomes} if case["kind"] == "backward" else {}),
        }
        _check_secrets(case, {**{name: repr(value) for name, value in actual.items()},
                              "the run log": text,
                              "the raised error": str(caught.get("error", ""))})
        if {**run, **resume}.get("num_workers", 8) != 0:
            actual = _pooled(actual)
        run_id = None if "run_id" in run or not lines else json.loads(lines[0])["run_id"]
        # A raised error is compared as is, without placeholders.
        return {**_placeholders(actual, run_id), **caught, "left": left}


def _interrupted(argv: list, case_dir: Path, env: dict, stdin: str, after: str) -> tuple:
    """Run `argv`, send it SIGINT once `after` is in its stdout or stderr, and return
    `(returncode, stdout, stderr)` once it exits.

    Fails the case when the command exits before `after` appears or 60 seconds pass first.
    """
    process = subprocess.Popen(
        argv, cwd=case_dir, env=env, encoding="utf-8", stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        # A shell can start background jobs with SIGINT ignored, and the child would keep that.
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    out, err = [], []

    def collect(stream, parts: list) -> None:
        # One character at a time, so text is seen as soon as it is written.
        for char in iter(lambda: stream.read(1), ""):
            parts.append(char)

    def feed() -> None:
        # A command can exit, or stop reading, before taking all of its stdin.
        with contextlib.suppress(BrokenPipeError):
            process.stdin.write(stdin)
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()

    readers = [threading.Thread(target=collect, args=pair, daemon=True)
               for pair in ((process.stdout, out), (process.stderr, err))]
    # On its own thread, so a stdin the command never reads can't block past the deadline.
    writer = threading.Thread(target=feed, daemon=True)
    for thread in (*readers, writer):
        thread.start()
    try:
        deadline = time.monotonic() + 60
        while True:
            # Taken before looking: once both readers are done, all the output is in.
            finished = not any(reader.is_alive() for reader in readers)
            if any(after in "".join(parts) for parts in (out, err)):
                break
            if finished:
                _fail(f"the command exited before {after!r} appeared\n--- stdout\n"
                      f"{''.join(out)}--- stderr\n{''.join(err)}")
            if time.monotonic() > deadline:
                _fail(f"{after!r} didn't appear within 60 seconds")
            time.sleep(0.02)
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            _fail(f"the command didn't exit within 120 seconds after SIGINT\n--- stdout\n"
                  f"{''.join(out)}--- stderr\n{''.join(err)}")
    finally:
        # Ends a command the case gave up on; a no-op once it has exited.
        process.kill()
        process.wait()
    for thread in (*readers, writer):
        # A process the command started can hold a pipe open after the command exits.
        thread.join(timeout=10)
        if thread.is_alive():
            _fail("a pipe to the command stayed open 10 seconds after it exited")
    return process.returncode, "".join(out), "".join(err)


def _actual_cli(case: dict, case_dir: Path) -> dict:
    """Run `yian` with `args` in the case directory; return `exit`, `stdout` and `stderr`.

    With `signal`, the command is sent the signal once `signal.after` appears in its output.
    """
    (case_dir / "home").mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(case_dir / "home"), "COLUMNS": "1000",
           "PYTHONPATH": os.pathsep.join(sys.path), **case.get("env", {})}
    command = "import sys; sys.argv[0] = 'yian'; from yian.cli import main; main()"
    argv = [sys.executable, "-c", command, *case.get("args", [])]
    if "signal" in case:
        returncode, out, err = _interrupted(argv, case_dir, env, case.get("stdin", ""),
                                            case["signal"]["after"])
    else:
        done = subprocess.run(
            argv, cwd=case_dir, input=case.get("stdin", ""), capture_output=True,
            encoding="utf-8", timeout=120, env=env,
        )
        returncode, out, err = done.returncode, done.stdout, done.stderr

    def normal(text: str) -> str:
        # Longest first: the resolved path can hold the given one (`/private/var` on macOS).
        for path in sorted({str(case_dir), str(case_dir.resolve())}, key=len, reverse=True):
            text = text.replace(path, "<tmp>")
        return re.sub(r" +$", "", text, flags=re.MULTILINE)

    stdout, stderr = normal(out), normal(err)
    _check_secrets(case, {"stdout": stdout, "stderr": stderr, **{
        str(path.relative_to(case_dir)): path.read_text(encoding="utf-8")
        for path in case_dir.rglob("*.jsonl")
    }})
    return {"exit": returncode, "stdout": stdout, "stderr": stderr}


_KINDS = {
    "load": _actual_load, "run": _actual_run, "resume": _actual_run, "backward": _actual_run,
    "defaults": _actual_defaults, "fingerprint": _actual_fingerprint, "expr": _actual_expr,
    "value": _actual_value, "cli": _actual_cli,
}


def _check_cli(expect: dict, actual: dict) -> None:
    """Compare a `cli` case's exit code, stdout and stderr, and stdout read as JSON."""
    stdout, stderr = actual["stdout"], actual["stderr"]

    def refuse(token: str) -> None:
        raise ValueError(f"{token} is not JSON")

    def document(text: str) -> object:
        try:
            parsed = json.loads(text, parse_constant=refuse)
        except ValueError as error:  # JSONDecodeError is a ValueError
            _fail(f"stdout is not a JSON document ({error})\n--- stdout\n{stdout}")
        # Checked as parsed: the schema describes the JSON, not the codec's values.
        invalid = best_match(_CLI_VALIDATOR.iter_errors(parsed))
        if invalid is not None:
            # When no `anyOf` or `oneOf` branch is clearly closest, the best match is the
            # combinator's own error, so each branch's errors are listed.
            reasons = "".join(f"  {error.json_path}: {error.message}\n"
                              for error in invalid.context or [invalid])
            _fail(f"stdout doesn't match conformance/schema/cli.schema.json\n{reasons}"
                  f"--- stdout\n{stdout}")
        return _decode(parsed)

    if actual["exit"] < 0:
        _fail(f"died from signal {-actual['exit']}\n--- stdout\n{stdout}--- stderr\n{stderr}")
    if not _same(actual["exit"], expect["exit"]):
        _fail(f"exit {actual['exit']}, expected {expect['exit']}\n--- stdout\n{stdout}"
              f"--- stderr\n{stderr}")
    for name, text in (("stdout", stdout), ("stderr", stderr)):
        if name in expect and text != expect[name]:
            _fail(f"{name} differs\n--- actual\n{text}--- expected\n{expect[name]}")
    if "json" in expect:
        _compare("json", document(stdout), expect["json"], optional_message=True)
    if "json_lines" in expect:
        _compare("json_lines", [document(line) for line in stdout.splitlines() if line.strip()],
                 expect["json_lines"], optional_message=True)


def _check_results(case: dict, actual: dict) -> None:
    """Fail unless the results in `actual` are the ones the case file contents `case` expect."""
    expect = _prepare(case)["expect"]
    if "raises" in expect and "raises" not in actual:
        _fail(f"expected {expect['raises'].get('class')} to be raised, but nothing was")
    if "raises" in actual and "raises" not in expect:
        _fail("unexpected error\n" + "".join(traceback.format_exception(actual["error"])))
    if "raises" in expect:
        _compare("raises", actual["raises"], expect["raises"])
    if case["kind"] == "cli":
        _check_cli(expect, actual)
        return
    if "defaults" in expect:
        unknown = set(expect["defaults"]) - set(actual["defaults"])
        if unknown:
            _fail(f"`expect.defaults` names unknown entries: {', '.join(sorted(unknown))}")
        actual = {**actual, "defaults": {name: actual["defaults"][name]
                                         for name in expect["defaults"]}}
    if "output" in expect:
        _compare("output", actual["output"], expect["output"],
                 by_value=actual["output"]["status"] == "succeeded")
    for name in ("load", "value", "kind", "defaults", "fingerprint", "trace", "events", "stream",
                 "tool_calls", "backward"):
        if name in expect:
            # Only a run's events, stream and tool calls compare numbers by value; backward
            # outcomes are projected like the trace, so they keep the kinds.
            _compare(name, actual[name], expect[name],
                     by_value=name in ("events", "stream", "tool_calls"),
                     optional_message=name == "load")
    # Last: a leftover entry usually follows from a divergence the comparisons show.
    if actual.get("left"):
        _fail("\n".join(actual["left"]))


def results(case: dict, case_dir: Path, candidate: bool = False) -> dict:
    """Return the results of running the case file contents `case` in `case_dir`, by name.

    The case is checked (as a `candidate`, without its expectations) and prepared, and fails
    when it uses a key this runner doesn't support yet; its directory and environment are set
    up, and the environment is restored afterwards.
    """
    case = _prepare(_check(case, candidate))
    for section, key in _RESERVED:
        if key in case.get(section, {}):
            _fail(f"`{section}.{key}` is not supported by the Python runner yet")
    try:
        with pytest.MonkeyPatch.context() as patch:
            _set_up(case, case_dir, patch)
            return _KINDS[case["kind"]](case, case_dir)
    finally:
        # Defaults read from the case's environment must not outlive it.
        defaults._forget()


@pytest.mark.parametrize("path", _params())
def test_case(path: Path, tmp_path: Path) -> None:
    """Run one conformance case and compare its results with its expectations."""
    case = _read(path)
    _check_results(case, results(case, tmp_path))


def test_harvest_writes_cases_that_read_back(tmp_path: Path) -> None:
    """The writer quotes text the reader would take for another kind and writes codec form."""
    from tests import conformance_harvest as harvest

    shared = ["a", "b"]
    case = {
        "inputs": {"0o17": "0o17", "yes": "yes", "exp": "1e3", "null": "Null",
                   "text": "one\ntwo\n", "inf": float("inf"), "ninf": float("-inf"),
                   "record": {"$record": 1}, "big": 10**39 + 7, "x": shared, "y": shared,
                   "breaks": ["a\u2028b", "a\u2029b\n", "a\x85b"],
                   "a\u2028b": 1, "a\x85b": 2},
        "flow": "id: f\nnodes: []\n",
        "kind": "run",
    }
    text = harvest._dump(case)
    path = tmp_path / "case.yaml"
    path.write_text(text, encoding="utf-8")
    read = _read(path)
    assert list(read) == ["kind", "flow", "inputs"]
    assert read["inputs"]["inf"] == {"$float": "inf"}
    assert read["inputs"]["record"] == {"$record": {"$record": 1}}
    decoded = {key: value if key in _TEXT_KEYS else _decode(value) for key, value in read.items()}
    assert _same(decoded, case)
    for written in ("'0o17': '0o17'", "yes: yes", "exp: '1e3'", "'null': 'Null'",
                    "flow: |\n  id: f\n", "text: |\n"):
        assert written in text
    # YAML 1.1 and 1.2 readers disagree on these unless they are escaped.
    assert '- "a\\Lb"' in text and '? "a\\Nb"\n  : 2' in text
    assert not {"\x85", "\u2028", "\u2029"} & set(text)
    assert "&" not in text and "*" not in text
    for value in ((1, 2), float("nan")):
        with pytest.raises(ValueError):
            harvest._dump({"kind": "expr", "expr": "x", "values": {"x": value}})


def test_harvest_names_and_dedupes(tmp_path: Path) -> None:
    """Case names follow the test; a held text, or a second text for one name, isn't written."""
    from tests import conformance_harvest as harvest

    name = harvest._case_name("tests/engine/test_expr_eval.py", "TestA::test_x[a-b]")
    assert name == "engine_expr_eval__TestA__x_a-b_"
    assert harvest._case_name("tests/test_a.py", "test_x[a b]", 2) == "a__x_a_b___2"
    salt = hashlib.sha256(b"test_x[a b]").hexdigest()[:8]
    assert harvest._case_name("tests/test_a.py", "test_x[a b]", salted=True) == f"a__x_a_b__{salt}"
    for index in (1, 2):
        full = "a__" + "x" * 300 + ("__2" if index > 1 else "")
        cut = full[:141] + "_" + hashlib.sha256(full.encode()).hexdigest()[:8]
        assert harvest._case_name("tests/test_a.py", "test_" + "x" * 300, index) == cut

    cases = tmp_path / "cases"
    index = harvest._Index(cases)
    assert index.write("expr", "one", "kind: expr\n") == "written"
    assert index.write("expr", "one", "kind: expr\n") == "rewrite"
    assert index.write("values", "two", "kind: expr\n") == "duplicate"
    # Two tests that give one name in one session: the second text isn't written.
    assert index.write("expr", "one", "kind: value\n") == "collision"
    assert sorted(path.relative_to(cases).as_posix() for path in cases.rglob("*")) == [
        "expr", "expr/one.yaml"]
    assert (cases / "expr/one.yaml").read_text(encoding="utf-8") == "kind: expr\n"

    again = harvest._Index(cases)
    assert again.write("expr", "three", "kind: expr\n") == "duplicate"
    # A name from disk harvested again with a new text is overwritten and gives up its old text.
    assert again.write("expr", "one", "kind: value\n") == "written"
    assert again.write("values", "two", "kind: expr\n") == "written"
    assert again.write("expr", "one", "kind: other\n") == "collision"
    assert (cases / "expr/one.yaml").read_text(encoding="utf-8") == "kind: value\n"

    last = harvest._Index(cases)
    assert last.write("expr", "one", "kind: value\n") == "rewrite"
    # A rewrite claims its name for the session too.
    assert last.write("expr", "one", "kind: other\n") == "collision"
    assert last.write("values", "two", "kind: expr\n") == "rewrite"
    assert sorted(path.name for path in cases.rglob("*.yaml*")) == ["one.yaml", "two.yaml"]


_HARVESTED_TESTS = '''
import pytest

from yian import defaults
from yian.expr import eval_binding, eval_expr, parse_expr, render_template_record
from yian.expr.expressions import evaluate_when_record
from yian.typesys.types import type_for
from yian.typesys.values import TypeCheckError, build_value_as


def _from(mapping):
    def resolve(path):
        value = mapping
        for step in path.split("."):
            value = value.get(step) if isinstance(value, dict) else None
        return value
    return resolve


def test_arithmetic():
    assert eval_expr(parse_expr("order.qty * 2 + 0.5"), _from({"order": {"qty": 3}})) == 6.5


def test_template():
    assert render_template_record("Hi ${name}!", {"name": "Ada"}) == "Hi Ada!"


def test_binding():
    assert eval_binding("${a :- 30}", _from({})) == 30


def test_fails():
    eval_expr(parse_expr("1 + 1"), _from({}))
    assert False


def test_or_binding():
    assert eval_expr(parse_expr("x or y"), _from({"x": 0, "y": 2})) is True


def test_or_condition():
    assert evaluate_when_record("(x or y) == 1", {"x": 1, "y": 0}) is True


def test_defaults(monkeypatch):
    monkeypatch.setenv("YIAN_MAX_TOTAL_NODES", "20000")
    defaults._forget()
    try:
        assert defaults.current().max_total_nodes == 20000
    finally:
        defaults._forget()


def test_value():
    with pytest.raises(TypeCheckError):
        build_value_as(type_for("list(int)", {}), [1, 2.5])
'''


def _harvest(tmp_path: Path, text: str) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the test file `text` under the harvest plugin; return the run and the output root.

    The run gets no `YIAN_*` variables but the model defaults and auth file the suite sets.
    """
    root = Path(__file__).resolve().parent.parent
    (tmp_path / "test_sample.py").write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    env = {name: value for name, value in os.environ.items() if not name.startswith("YIAN_")}
    env |= {"PYTHONPATH": f"{root / 'src'}{os.pathsep}{root}", "YIAN_DEFAULT_PROVIDER": "openai",
            "YIAN_DEFAULT_MODEL": "gpt-5.5", "YIAN_AUTH_FILE": str(tmp_path / "auth.json")}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "tests.conformance_harvest", "--harvest-out", str(out), "test_sample.py"],
        cwd=tmp_path, env=env, capture_output=True, encoding="utf-8", timeout=300,
    )
    return done, out


def test_harvest_records_expr_value_defaults(tmp_path: Path) -> None:
    """Passing tests' expr, template, value and defaults calls become cases that pass."""
    done, out = _harvest(tmp_path, _HARVESTED_TESTS)
    assert "1 failed, 7 passed" in done.stdout, done.stdout + done.stderr
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    assert sorted(cases) == [
        "defaults/sample__defaults.yaml", "expr/sample__arithmetic.yaml",
        "templates/sample__binding.yaml", "templates/sample__template.yaml",
        "values/sample__value.yaml",
    ]
    assert cases["expr/sample__arithmetic.yaml"] == {
        "kind": "expr", "expr": "order.qty * 2 + 0.5", "values": {"order": {"qty": 3}},
        "expect": {"value": 6.5}}
    assert cases["templates/sample__template.yaml"] == {
        "kind": "expr", "template": "Hi ${name}!", "values": {"name": "Ada"}, "mode": "strict",
        "expect": {"value": "Hi Ada!"}}
    assert cases["templates/sample__binding.yaml"] == {
        "kind": "expr", "template": "${a :- 30}", "expect": {"value": 30}}
    # Only the variables the test set; the entries they set.
    assert cases["defaults/sample__defaults.yaml"] == {
        "kind": "defaults", "env": {"YIAN_MAX_TOTAL_NODES": "20000"},
        "expect": {"defaults": {"MAX_TOTAL_NODES": 20000}}}
    value = cases["values/sample__value.yaml"]
    assert {key: value[key] for key in ("kind", "type", "value")} == {
        "kind": "value", "type": "list(int)", "value": [1, 2.5]}
    assert value["expect"]["raises"]["class"] == "TypeCheckError"
    assert value["expect"]["raises"]["path"] == [1]
    for name, case in cases.items():
        case_dir = tmp_path / "replay" / name
        case_dir.mkdir(parents=True)
        _check_results(case, results(case, case_dir))

    report = json.loads((out / "report" / "test_sample.py.json").read_text(encoding="utf-8"))
    assert report == {
        "file": "test_sample.py",
        "tests": {"passed": 7, "failed": 1, "skipped": 0},
        "records": {"eval_expr": 2, "render_template_record": 1, "eval_binding": 1,
                    "evaluate_when_record": 1, "defaults.current": 1, "build_value_as": 1},
        "written": {"expr": 1, "templates": 2, "defaults": 1, "values": 1},
        "duplicates": 0,
        "skips": {"held: and/or result (owner deciding)": ["test_or_binding",
                                                           "test_or_condition"]},
        "flags": {},
    }
