"""Run the language-neutral conformance cases in `conformance/cases/` against Python yian.

Each case file is one test, named by its id `<group>/<name>`. `conformance/README.md` defines
the case format and what each kind checks. Set `CONFORMANCE_CASES` to run the cases under
another root instead.
"""

from __future__ import annotations

import contextlib
from dataclasses import fields, is_dataclass
from datetime import date
from functools import reduce
import hashlib
import inspect
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
from yian.typesys.values import FileRef, TypeCheckError, TypedValue, build_value_as

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
<<<<<<< ours
    "numbers", "int-in-float", "arithmetic", "compare", "text-form", "canonical", "columns",
    "messages", "text", "yaml", "dates", "infinity", "llm", "code",
=======
    "numbers", "arithmetic", "compare", "text-form", "canonical", "columns", "messages",
    "text", "yaml", "dates", "infinity", "llm", "code", "int-in-float",
>>>>>>> theirs
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
# The builtins a scripted model error can be; any other class is a plain error of that name.
_MODEL_ERRORS = {error.__name__: error for error in (TimeoutError, ConnectionError)}


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
    # yian refuses an auth file in a directory others can read.
    case_dir.chmod(0o700)
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


def _defaults_form(values: defaults.Defaults) -> dict:
    """Return yian's defaults `values` as the results' `defaults`: every entry by name."""
    return {"defaults": {entry.name: getattr(values, entry.name.lower())
                         for entry in defaults.ENTRIES}}


def _actual_defaults(case: dict, case_dir: Path) -> dict:
    """Read yian's defaults and return every entry as `defaults`, or the error."""
    return _catching(lambda: _defaults_form(defaults.current()))


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


def _value_form(typed: TypedValue) -> dict:
    """Return a built value as the results' stored `value` and its `kind`."""
    return {"value": typed.to_object(), "kind": typed.kind.value}


def _actual_value(case: dict, case_dir: Path) -> dict:
    """Build `value` as `type` and return the stored value and its kind, or the error."""
    def compute() -> dict:
        typ = type_for(case["type"], read_typedefs(case.get("typedefs") or {}))
        return _value_form(build_value_as(typ, case["value"]))

    return _catching(compute)


_SEAT = threading.local()


def _seated(patch: pytest.MonkeyPatch) -> None:
    """Wrap the functions that make model calls, so `_take_model_key()` names each call's seat.

    `run_agent` seats its node's `agent` calls (`compose` for a `<gate>__compose` node),
    `backward_agent` its node's `backward` calls and `prepare_history` `summary` calls;
    `generate_structured` keeps its caller's seat and starts counting the calls made under it.
    A call whose context can't be read gets no seat.
    """
    def wrap(module: object, name: str, make_seat, first: str | None = None) -> None:
        inner = getattr(module, name)

        def seated(*args: object, **kwargs: object) -> object:
            outer = getattr(_SEAT, "seat", None)
            try:
                _SEAT.seat = make_seat(outer, args[0] if args else kwargs.get(first))
            except Exception:
                # A call the seat can't be read from gets none, rather than an error in yian.
                _SEAT.seat = None
            try:
                return inner(*args, **kwargs)
            finally:
                _SEAT.seat = outer

        patch.setattr(module, name, seated)

    wrap(agent_node, "run_agent", lambda outer, ctx: {
        "node": ctx.node_id,
        "consumer": "compose" if ctx.node_id.endswith("__compose") else "agent",
    }, "ctx")
    wrap(agent_node, "backward_agent",
         lambda outer, context: {"node": context.node_id, "consumer": "backward"}, "context")
    wrap(agent_loop, "generate_structured",
         lambda outer, first: None if outer is None else {**outer, "calls": 0})
    wrap(agent_loop, "prepare_history",
         lambda outer, first: None if outer is None else {**outer, "consumer": "summary"})


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
    """Raise a model call's scripted error.

    Its class is `TimeoutError` or `ConnectionError` for those names, else a new class of that
    name; a `status` becomes the error's `status_code`.
    """
    cls = _MODEL_ERRORS.get(error["class"]) or type(error["class"], (Exception,), {})
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
from yian.expr.expressions import ExpressionError, ResolveMode, evaluate_when_record
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


def test_condition():
    assert evaluate_when_record("n > 1", {"n": 2}) is True


def test_strict_miss():
    with pytest.raises(ExpressionError):
        eval_expr(parse_expr("a.b"), _from({}), mode=ResolveMode.STRICT_RAISE)


def test_item():
    assert eval_expr(parse_expr("item.t"), _from({}), item={"t": 1}) == 1


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


def _pytest(tmp_path: Path, *args: str, **variables: str) -> subprocess.CompletedProcess:
    """Run pytest with `args` in `tmp_path` and return the run.

    The run gets no `YIAN_*` variables but the model defaults and auth file the suite sets,
    and `variables` added to its environment.
    """
    root = Path(__file__).resolve().parent.parent
    env = {name: value for name, value in os.environ.items() if not name.startswith("YIAN_")}
    env |= {"PYTHONPATH": f"{root / 'src'}{os.pathsep}{root}", "YIAN_DEFAULT_PROVIDER": "openai",
            "YIAN_DEFAULT_MODEL": "gpt-5.5", "YIAN_AUTH_FILE": str(tmp_path / "auth.json"),
            **variables}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=tmp_path, env=env, capture_output=True, encoding="utf-8", timeout=300,
    )


def _harvest(tmp_path: Path, text: str, *args: str,
             **variables: str) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the test file `text` under the harvest plugin; return the run and the output root.

    `args` select the tests (default: the whole file); `variables` are as `_pytest` takes them.
    """
    (tmp_path / "test_sample.py").write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    done = _pytest(tmp_path, "-p", "tests.conformance_harvest", "--harvest-out", str(out),
                   *(args or ["test_sample.py"]), **variables)
    return done, out


def test_harvest_records_expr_value_defaults(tmp_path: Path) -> None:
    """Passing tests' expr, template, value and defaults calls become cases that pass."""
    done, out = _harvest(tmp_path, _HARVESTED_TESTS)
    assert "1 failed, 10 passed" in done.stdout, done.stdout + done.stderr
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    assert sorted(cases) == [
        "defaults/sample__defaults.yaml", "expr/sample__arithmetic.yaml",
        "expr/sample__condition.yaml", "expr/sample__item.yaml", "expr/sample__or_binding.yaml",
        "expr/sample__or_condition.yaml", "expr/sample__strict_miss.yaml",
        "templates/sample__binding.yaml", "templates/sample__template.yaml",
        "values/sample__value.yaml",
    ]
    assert cases["expr/sample__arithmetic.yaml"] == {
        "kind": "expr", "expr": "order.qty * 2 + 0.5", "values": {"order": {"qty": 3}},
        "expect": {"value": 6.5}}
    assert cases["expr/sample__condition.yaml"] == {
        "kind": "expr", "expr": "n > 1", "values": {"n": 2}, "mode": "condition",
        "expect": {"value": True}}
    assert cases["expr/sample__item.yaml"] == {
        "kind": "expr", "expr": "item.t", "item": {"t": 1}, "expect": {"value": 1}}
    # `and` and `or` give a bool, as in TS: shared cases.
    assert cases["expr/sample__or_binding.yaml"] == {
        "kind": "expr", "expr": "x or y", "values": {"x": 0, "y": 2}, "expect": {"value": True}}
    assert cases["expr/sample__or_condition.yaml"] == {
        "kind": "expr", "expr": "(x or y) == 1", "values": {"x": 1, "y": 0}, "mode": "condition",
        "expect": {"value": True}}
    strict = cases["expr/sample__strict_miss.yaml"]
    assert {key: strict[key] for key in ("kind", "expr", "mode")} == {
        "kind": "expr", "expr": "a.b", "mode": "strict"}
    assert "values" not in strict and strict["expect"]["raises"]["class"] == "ExpressionError"
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
        "tests": {"passed": 10, "failed": 1, "skipped": 0},
        "records": {"eval_expr": 4, "render_template_record": 1, "eval_binding": 1,
                    "evaluate_when_record": 2, "defaults.current": 1, "build_value_as": 1},
        "written": {"expr": 6, "templates": 2, "defaults": 1, "values": 1},
        "duplicates": 0,
        "skips": {},
        "flags": {},
    }


_LOAD_TESTS = '''
import pytest

from yian import LoadError, load_flow

OK = """\\
id: minimal
name: minimal
input:
  q: !str
nodes:
  score:
    kind: tool
    tool_id: score
    input:
      q: ${input.q}
output: ${score.output}
"""

AGENT = """\\
id: agent
name: agent
input:
  topic: !str
nodes:
  rate:
    kind: agent
    input:
      topic: ${input.topic}
    output: float
    prompt: "Rate ${topic}."
output: ${rate.output}
"""

CHILD = """\\
id: child
name: child
input:
  q: !str
nodes:
  score:
    kind: tool
    tool_id: score
    input:
      q: ${input.q}
output: ${score.output}
"""

USES = """\\
id: parent
name: parent
input:
  q: !str
uses:
  child: lib/child
nodes:
  run:
    kind: call
    call: child
    input:
      q: ${input.q}
output: ${run.output}
"""


def test_load_ok():
    load_flow(OK)


def test_load_error():
    with pytest.raises(LoadError):
        load_flow(OK.replace("${score.output}", "${scor.output}"))


def test_agent():
    load_flow(AGENT)


def test_uses(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "child.yaml").write_text(CHILD)
    # The flow's own file, under the runner's name for it, isn't one of the case's files.
    (tmp_path / "flow.yaml").write_text(USES)
    load_flow(USES, search_paths=[tmp_path], source_label="flow.yaml")


def test_child_resolver():
    load_flow(USES, child_resolver=lambda path, version=None: load_flow(CHILD))


def test_label():
    with pytest.raises(TypeError):
        load_flow(OK, source_label=1)
'''


def test_harvest_records_loads_and_resumes(tmp_path: Path) -> None:
    """Loads become `load` cases, `uses:` ones with `files`; a reported file isn't run again."""
    done, out = _harvest(tmp_path, _LOAD_TESTS)
    assert "6 passed" in done.stdout, done.stdout + done.stderr
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    assert sorted(cases) == ["load-agent/sample__agent.yaml", "load-core/sample__load_error.yaml",
                             "load-core/sample__load_ok.yaml", "load-core/sample__uses.yaml"]
    # Loads keep the model defaults the suite sets.
    env = {"YIAN_DEFAULT_MODEL": "gpt-5.5", "YIAN_DEFAULT_PROVIDER": "openai"}
    ok = cases["load-core/sample__load_ok.yaml"]
    assert ok == {"kind": "load", "flow": ok["flow"], "env": env, "expect": {"load": "ok"}}
    uses = cases["load-core/sample__uses.yaml"]
    assert uses["expect"] == {"load": "ok"}
    assert list(uses["files"]) == ["lib/child.yaml"]
    assert uses["files"]["lib/child.yaml"].startswith("id: child\n")
    error = cases["load-core/sample__load_error.yaml"]["expect"]["load"]["diagnostics"]
    assert [diag["code"] for diag in error] == ["compose.load"]
    assert "${scor.output}" in error[0]["message"]
    for name, case in cases.items():
        case_dir = tmp_path / "replay" / name
        case_dir.mkdir(parents=True)
        _check_results(case, results(case, case_dir))

    report_path = out / "report" / "test_sample.py.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {
        "file": "test_sample.py",
        "tests": {"passed": 6, "failed": 0, "skipped": 0},
        # The child_resolver's own load runs inside a recorded one.
        "records": {"load_flow": 6},
        "written": {"load-core": 3, "load-agent": 1},
        "duplicates": 0,
        "skips": {"child_resolver": ["test_child_resolver"],
                  "load raised TypeError": ["test_label"]},
        "flags": {},
    }
    assert "conformance harvest" in done.stdout
    assert "written: load-agent 1, load-core 3; 0 duplicates" in done.stdout, done.stdout
    assert "flags: none" in done.stdout, done.stdout
    assert "not reported" not in done.stdout, done.stdout

    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}
    again, _ = _harvest(tmp_path, _LOAD_TESTS)
    assert "6 deselected" in again.stdout, again.stdout + again.stderr
    assert {path: path.read_bytes() for path in out.rglob("*") if path.is_file()} == before

    # A narrowed selection harvests its tests but never closes the file.
    for name, args in (("k", ("-k", "load_ok", "test_sample.py")),
                       ("node", ("test_sample.py::test_load_ok",))):
        narrow = tmp_path / name
        narrow.mkdir()
        done, out = _harvest(narrow, _LOAD_TESTS, *args)
        assert "1 passed" in done.stdout, done.stdout + done.stderr
        assert (out / "cases" / "load-core" / "sample__load_ok.yaml").is_file()
        assert not (out / "report").exists()
        assert "not reported (narrowed selection): 1 files" in done.stdout, done.stdout


def test_harvest_files_limits(tmp_path: Path) -> None:
    """A `uses:` load's files: its `flow.yaml` must be the flow; 50 files and 256 KiB at most."""
    from tests import conformance_harvest as harvest

    text = "id: parent\nuses:\n  child: lib/child\n"
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "child.yaml").write_text("id: child\n", encoding="utf-8")
    (tmp_path / "flow.yaml").write_text(text, encoding="utf-8")
    assert harvest._files(text, [tmp_path]) == {"files": {"lib/child.yaml": "id: child\n"}}
    assert harvest._files("id: plain\n", [tmp_path]) == {}
    with pytest.raises(harvest._Skip, match="uses: needs one search path"):
        harvest._files(text, [])
    (tmp_path / "flow.yaml").write_text(text + "# changed\n", encoding="utf-8")
    with pytest.raises(harvest._Skip, match="files: flow.yaml isn't the flow"):
        harvest._files(text, [tmp_path])
    (tmp_path / "flow.yaml").unlink()
    (tmp_path / "lib" / "big.yaml").write_text("x" * (256 * 1024), encoding="utf-8")
    with pytest.raises(harvest._Skip, match="files: more than 256 KiB"):
        harvest._files(text, [tmp_path])
    (tmp_path / "lib" / "big.yaml").unlink()
    for index in range(50):
        (tmp_path / "lib" / f"extra_{index}.yaml").write_text("id: x\n", encoding="utf-8")
    with pytest.raises(harvest._Skip, match="files: more than 50"):
        harvest._files(text, [tmp_path])


_RUN_TESTS = '''
import os

import pytest

from yian import load_flow
from yian.run import resume_flow, run_flow

ECHO = """\\
id: echo
name: echo
input:
  q: !str
nodes: {}
output: ${input.q}
"""


def test_run():
    flow = load_flow(ECHO)
    assert run_flow(flow, {"q": "hi"}, run_id="r1", num_workers=0).output == "hi"


def test_pooled():
    assert run_flow(load_flow(ECHO), {"q": "pooled"}, run_id="r1").output == "pooled"


def test_failed():
    assert run_flow(load_flow(ECHO), {}, run_id="r1", num_workers=0).status == "failed"


def test_raises():
    with pytest.raises(ValueError):
        run_flow(load_flow(ECHO), {"nope": 1}, num_workers=0)


def test_keyed():
    config = {"provider": "openai", "api_key": "sk-other-key-5678",
              "headers": {"X-Team": "team-header-value"}}
    flow = load_flow(ECHO)
    assert run_flow(flow, {"q": "k"}, run_id="r1", num_workers=0, llm_config=config,
                    learnable=True, context=object()).output == "k"


def test_shared_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared-key-4321")
    config = {"provider": "openai", "api_key": "sk-shared-key-4321",
              "headers": {"X-Team": "team-header-value"}}
    flow = load_flow(ECHO)
    assert run_flow(flow, {"q": "s"}, run_id="r1", num_workers=0, llm_config=config).output == "s"


def test_shell_key():
    flow = load_flow(ECHO)
    run_flow(flow, {"q": os.environ["X_API_KEY"]}, run_id="r1", num_workers=0)


def test_secret():
    config = {"provider": "openai", "api_key": "sk-test-secret-1234"}
    flow = load_flow(ECHO)
    run_flow(flow, {"q": "sk-test-secret-1234"}, run_id="r1", num_workers=0, llm_config=config)


def test_tools():
    flow = load_flow(ECHO)
    run_flow(flow, {"q": "t"}, run_id="r1", num_workers=0, tools={"noop": lambda: None})


def test_tmp_path(tmp_path):
    flow = load_flow(ECHO)
    run_flow(flow, {"q": str(tmp_path)}, run_id="r1", num_workers=0)


def test_unknown_flow():
    flow = load_flow(ECHO, child_resolver=lambda path, version=None: None)
    run_flow(flow, {"q": "u"}, run_id="r1", num_workers=0)


def test_nested():
    inner = []

    def on_event(event):
        # A run inside a recorded run isn't recorded, nor is its load.
        if not inner:
            inner.append(run_flow(load_flow(ECHO), {"q": "inner"}, num_workers=0))

    flow = load_flow(ECHO)
    assert run_flow(flow, {"q": "outer"}, run_id="r1", num_workers=0,
                    on_event=on_event).output == "outer"
    assert inner[0].output == "inner"


def test_resume(tmp_path):
    flow = load_flow(ECHO)
    run_flow(flow, {"q": "again"}, run_id="r1", num_workers=0, log=tmp_path / "run.jsonl")
    assert resume_flow(flow, log=tmp_path / "run.jsonl").output == "again"
'''


def test_harvest_records_runs(tmp_path: Path) -> None:
    """Runs become `run-core` cases; hosts, secrets and machine paths skip."""
    # The harvesting shell's own key.
    done, out = _harvest(tmp_path, _RUN_TESTS, X_API_KEY="sk-shell-key-0123456789")
    assert "13 passed" in done.stdout, done.stdout + done.stderr
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    # A load whose run is written is no case of its own; one whose run isn't stays.
    assert sorted(cases) == [
        "load-core/sample__shell_key.yaml", "run-core/sample__failed__2.yaml",
        "run-core/sample__keyed__2.yaml", "run-core/sample__nested__2.yaml",
        "run-core/sample__pooled__2.yaml", "run-core/sample__raises__2.yaml",
        "run-core/sample__resume__2.yaml", "run-core/sample__run__2.yaml",
        "run-core/sample__shared_key__2.yaml",
    ]
    env = {"YIAN_DEFAULT_MODEL": "gpt-5.5", "YIAN_DEFAULT_PROVIDER": "openai"}
    run = cases["run-core/sample__run__2.yaml"]
    assert run == {
        "kind": "run", "flow": run["flow"], "env": env, "inputs": {"q": "hi"},
        "run": {"run_id": "r1", "num_workers": 0},
        "expect": {
            "output": {"status": "succeeded", "output": "hi"},
            "trace": [
                ["run_started", None, {"inputs": {"q": "hi"}, "learnable": False,
                                       "num_workers": 0}],
                ["node_started", "__start__"],
                ["node_succeeded", "__start__", {"value": {"q": "hi"}}],
                ["node_started", "__end__"],
                ["node_succeeded", "__end__", {"value": "hi"}],
            ],
            "events": [
                ["RunStarted"],
                ["NodeStarted", {"node_id": "__start__"}],
                ["NodeSucceeded", {"node_id": "__start__", "output": {"q": "hi"}}],
                ["NodeStarted", {"node_id": "__end__"}],
                ["NodeSucceeded", {"node_id": "__end__", "output": "hi"}],
                ["RunSucceeded", {"output": "hi"}],
            ],
        },
    }
    assert run["flow"].startswith("id: echo\n")
    # A pooled run runs with the default workers; its three replays agree, so its trace stays.
    pooled = cases["run-core/sample__pooled__2.yaml"]
    assert pooled["run"] == {"run_id": "r1"} and pooled["expect"]["trace"]
    failure = cases["run-core/sample__failed__2.yaml"]["expect"]["output"]
    assert failure["status"] == "failed" and failure["failure"]["node"] == "__start__"
    raises = cases["run-core/sample__raises__2.yaml"]
    assert "run" in raises and "run_id" not in raises["run"]
    assert raises["expect"]["raises"] == {"class": "ValueError",
                                          "message": "unknown input 'nope'; declared inputs: q"}
    # Raised before the run started: no events, which the case checks.
    assert raises["expect"]["events"] == []
    keyed = cases["run-core/sample__keyed__2.yaml"]
    assert keyed["run"] == {
        "run_id": "r1", "num_workers": 0, "learnable": True, "context": "present",
        "llm_config": {"provider": "openai", "api_key": "<key:1>",
                       "headers": {"X-Team": "<key:2>"}}}
    # One numbering: the key in `env` and in `llm_config` is one, the header another.
    shared = cases["run-core/sample__shared_key__2.yaml"]
    assert shared["env"]["OPENAI_API_KEY"] == "<key:1>"
    assert shared["run"]["llm_config"]["api_key"] == "<key:1>"
    assert shared["run"]["llm_config"]["headers"] == {"X-Team": "<key:2>"}
    assert not any("X_API_KEY" in case.get("env", {}) for case in cases.values())
    for name, case in cases.items():
        case_dir = tmp_path / "replay" / name
        case_dir.mkdir(parents=True)
        _check_results(case, results(case, case_dir))

    report = json.loads((out / "report" / "test_sample.py.json").read_text(encoding="utf-8"))
    assert report == {
        "file": "test_sample.py",
        "tests": {"passed": 13, "failed": 0, "skipped": 0},
        # The nested test's inner load and run aren't recorded.
        "records": {"load_flow": 13, "run_flow": 13, "resume_flow": 1},
        "written": {"run-core": 8, "load-core": 1},
        # The loads of the secret, tools and tmp_path tests hold the shell key test's load text.
        "duplicates": 3,
        "skips": {"secret in inputs": ["test_shell_key", "test_secret"],
                  "host: tools": ["test_tools"],
                  "machine path": ["test_tmp_path"], "child_resolver": ["test_unknown_flow"],
                  "flow text unknown": ["test_unknown_flow"],
                  "resume (hand-converted)": ["test_resume"]},
        "flags": {},
    }


_MODEL_TESTS = '''
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

import yian.llm_clients as llm_clients
from yian import load_flow
from yian.llm_clients import factory
from yian.run import run_flow
from tests.engine._stream_fakes import Cut, ScriptedStreamChat, reply, usage


def _run(output, model="{provider: openai, model: gpt-5.5}", **hooks):
    flow = f"""\\
id: write
name: write
nodes:
  write:
    kind: agent
    output: {output}
    llm_config: {model}
    prompt: Write.
output: ${{write.output}}
"""
    return run_flow(load_flow(flow), {}, run_id="r1", num_workers=0, **hooks)


class Fake:
    """A duck-typed chat model: `invoke` serves `replies`, a structured call `structured`."""

    def __init__(self, replies=(), structured=()):
        self.replies, self.structured = list(replies), list(structured)

    def invoke(self, messages):
        return self.replies.pop(0)

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class Bound:
            def invoke(self, messages):
                return schema.model_validate(outer.structured.pop(0))

        return Bound()


def test_text(monkeypatch):
    fake = Fake([reply("hello", tokens=usage(3, 2))])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: fake)
    assert _run("str").output == "hello"


def test_structured(monkeypatch):
    fake = Fake(structured=[{"name": "Ada", "score": 3}])
    monkeypatch.setattr("yian.llm_clients.model_from_config", lambda config: fake)
    assert _run("{name: str, score: int}").output == {"name": "Ada", "score": 3}


def test_structured_retry(monkeypatch):
    # A model without native structured output: each attempt is a plain call whose reply is seen.
    fake = Fake([AIMessage('{"name": "Ada", "score": "high"}'),
                 AIMessage('{"name": "Ada", "score": 3}')])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: fake)
    result = _run("{name: str, score: int}", "{provider: vllm, model: no-structured-sentinel}")
    assert result.output == {"name": "Ada", "score": 3}
    assert fake.replies == []


def test_structured_raises(monkeypatch):
    fake = Fake(structured=[{"name": "Ada", "score": "high"}, {"name": "Ada", "score": 3}])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: fake)
    assert _run("{name: str, score: int}").output == {"name": "Ada", "score": 3}
    assert fake.structured == []


def test_structured_runnable(monkeypatch):
    configs = []

    class Bound(Runnable):
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages, config=None, **kwargs):
            configs.append(config)
            return self.schema.model_validate({"name": "Bo", "score": 4})

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return Bound(schema)

    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: Model())
    result = _run("{name: str, score: int}", on_event=lambda event: None)
    assert result.output == {"name": "Bo", "score": 4}
    # yian hands its usage callback only to a Runnable binding.
    assert configs[0]["callbacks"]


def test_no_stream(monkeypatch):
    fake = Fake([reply("hello")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: fake)
    assert _run("str", on_stream=lambda chunk: None).output == "hello"


def test_list_content(monkeypatch):
    fake = Fake([AIMessage(content=[{"type": "text", "text": "hi"}])])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: fake)
    assert _run("str").output == "hi"


def test_streamed(monkeypatch):
    chat = ScriptedStreamChat(script=[reply("hello there", tokens=usage(3, 2))])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    assert _run("str", on_stream=lambda chunk: None).output == "hello there"


def test_stream_cut(monkeypatch):
    chat = ScriptedStreamChat(script=[Cut("hel lo", ConnectionError("connection reset"))])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    result = _run("str", on_stream=lambda chunk: None)
    assert result.status == "failed" and result.failure.message == "connection reset"


def test_client_type(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-client-0123")
    config = {"provider": "openai", "model": "gpt-5.5"}
    assert type(factory.model_from_config(config)).__name__ == "NormalizedChatOpenAI"
    assert type(llm_clients.model_from_config(config)).__name__ == "NormalizedChatOpenAI"
'''


def _summary(done: subprocess.CompletedProcess) -> str:
    """Return a pytest run's last line, its counts, without the time it took."""
    return done.stdout.strip().splitlines()[-1].partition(" in ")[0]


def test_harvest_records_models(tmp_path: Path) -> None:
    """Model calls through a patched factory become `host.models` entries of `run-agent` cases."""
    done, out = _harvest(tmp_path, _MODEL_TESTS)
    assert _summary(done) == "10 passed", done.stdout + done.stderr
    # The plugin changes no test's outcome.
    assert _summary(_pytest(tmp_path, "test_sample.py")) == _summary(done)
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    # The loads of the skipped runs stay.
    assert sorted(cases) == [
        "load-agent/sample__list_content.yaml", "load-agent/sample__structured_raises.yaml",
        "run-agent/sample__no_stream__2.yaml", "run-agent/sample__stream_cut__2.yaml",
        "run-agent/sample__streamed__2.yaml", "run-agent/sample__structured__2.yaml",
        "run-agent/sample__structured_retry__2.yaml",
        "run-agent/sample__structured_runnable__2.yaml", "run-agent/sample__text__2.yaml",
    ]
    models = {name.removeprefix("run-agent/sample__").removesuffix("__2.yaml"):
              case["host"]["models"]["write"] for name, case in cases.items()
              if name.startswith("run-agent/")}

    def text(value: str) -> dict:
        return {"format": "yian.message.v1", "role": "assistant",
                "parts": [{"type": "text", "text": value}]}

    def roles(entry: dict) -> list:
        return [message["role"] for message in entry["request"]["messages"]]

    (entry,), = models["text"].values()
    assert list(models["text"]) == ["agent"]
    assert entry["reply"] == text("hello")
    assert entry["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert list(entry["request"]) == ["messages"] and roles(entry) == ["system", "user"]
    assert entry["request"]["messages"][1]["parts"] == [{"type": "text", "text": "Write."}]

    (entry,), = models["structured"].values()
    assert list(models["structured"]) == ["agent"]
    # The recorder sees no usage of a native structured call.
    assert entry["reply"] == text('{"name": "Ada", "score": 3}') and "usage" not in entry
    assert entry["request"]["schema"]["required"] == ["name", "score"]

    # The first reply fails the schema; the retry's request ends in the correction.
    assert list(models["structured_retry"]) == ["agent", "structured_retry"]
    (first,), (retry,) = models["structured_retry"].values()
    assert first["reply"] == text('{"name": "Ada", "score": "high"}')
    assert retry["reply"] == text('{"name": "Ada", "score": 3}')
    assert "schema" not in first["request"] and roles(first) == ["system", "user", "user"]
    assert roles(retry) == ["system", "user", "user", "user"]
    correction = retry["request"]["messages"][3]["parts"][0]["text"]
    assert correction.startswith("Your previous output was invalid: ")

    (entry,), = models["structured_runnable"].values()
    assert entry["reply"] == text('{"name": "Bo", "score": 4}')

    # A model without `stream` is invoked even with a stream sink.
    (entry,), = models["no_stream"].values()
    assert entry["reply"] == text("hello") and "chunks" not in entry

    (entry,), = models["streamed"].values()
    assert entry["chunks"] == [
        {"text": "hello"}, {"text": " there"},
        {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}, {"finish": None}]

    (entry,), = models["stream_cut"].values()
    assert entry["chunks"] == [{"text": "hel"}, {"text": " lo"}]
    assert entry["raise"] == {"class": "ConnectionError", "message": "connection reset"}
    assert cases["run-agent/sample__stream_cut__2.yaml"]["expect"]["output"]["status"] == "failed"
    for name, case in cases.items():
        case_dir = tmp_path / "replay" / name
        case_dir.mkdir(parents=True)
        _check_results(case, results(case, case_dir))

    report = json.loads((out / "report" / "test_sample.py.json").read_text(encoding="utf-8"))
    assert report["tests"] == {"passed": 10, "failed": 0, "skipped": 0}
    assert report["written"] == {"run-agent": 7, "load-agent": 2}
    assert report["skips"] == {
        "model: structured call raised ValidationError (a binding's errors can't be scripted)": [
            "test_structured_raises"],
        "model: AIMessage with list content": ["test_list_content"],
    }


_POOLED_TESTS = '''
import json

from langchain_core.messages import AIMessage

import yian.llm_clients as llm_clients
from yian import load_flow
from yian.run import run_flow

PAIR = """\\
id: pair
name: pair
nodes:
  a:
    kind: agent
    output: str
    llm_config: {provider: openai, model: gpt-5.5}
    prompt: Write a.
  b:
    kind: agent
    output: str
    llm_config: {provider: openai, model: gpt-5.5}
    prompt: Write b.
output:
  a: ${a.output}
  b: ${b.output}
"""

ECHO = """\\
id: echo
name: echo
input:
  q: !str
nodes: {}
output: ${input.q}
"""

WRITE = """\\
id: write
name: write
nodes:
  write:
    kind: agent
    output: str
    llm_config: {provider: openai, model: gpt-5.5}
    prompt: Write.
output: ${write.output}
"""

GATE = """\\
id: gate
name: gate
input:
  n: !int
nodes:
  gate:
    kind: case
    cases:
      - when: "input.n > 5 or input.n < 0"
        then: big
    else: small
  big: {kind: code, code: "return 'big'", output: str}
  small: {kind: code, code: "return 'small'", output: str}
output: ${big.output | small.output}
"""

# A CASE's `when:` is first evaluated in the run; a CALL binding's constant fallback is
# evaluated when the flow loads, to check its type against the child's input.
EITHER = """\\
id: either
name: either
input:
  a: !bool
defs:
  child:
    input:
      flag: !bool
    nodes: {}
    output: ${input.flag}
nodes:
  pick:
    kind: call
    call: child
    input:
      flag: ${input.a :- (true or false)}
output: ${pick.output}
"""

FEEDBACK = json.dumps({"input_feedback": {}, "self_feedback": [], "variable_feedback": {}})


class Shout:
    """A chat model replying with its last message in capitals, whichever thread calls it."""

    def invoke(self, messages):
        return AIMessage(messages[-1].content.upper())


class Fake:
    """A chat model serving `replies` in order."""

    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        return AIMessage(self.replies.pop(0))


def test_pooled_agents(monkeypatch):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: Shout())
    result = run_flow(load_flow(PAIR), {}, run_id="r1")
    assert result.output == {"a": "WRITE A.", "b": "WRITE B."}


def test_pooled_failed():
    assert run_flow(load_flow(ECHO), {}, run_id="r1").status == "failed"


def test_backward(monkeypatch):
    fake = Fake(["hello", FEEDBACK])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: fake)
    result = run_flow(load_flow(WRITE), {}, run_id="r1", num_workers=0, learnable=True)
    # Called after the run returned: it still joins the run's record.
    assert result.backward("Shorter.", api_key="sk-backward-key-0001").status == "succeeded"
    assert fake.replies == []


def test_backward_unrecorded(monkeypatch):
    fake = Fake(["hello", FEEDBACK])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: fake)
    result = run_flow(load_flow(WRITE), {}, run_id="r1", num_workers=0, learnable=True,
                      tools={"noop": lambda: None})
    assert result.backward("Shorter.").status == "succeeded"


def test_or_case():
    assert run_flow(load_flow(GATE), {"n": 7}, run_id="r1", num_workers=0).output == "big"


def test_or_load():
    load_flow(EITHER)
'''


def test_harvest_records_backward_and_pooled(tmp_path: Path) -> None:
    """Pooled runs replay thrice; backward calls join their run; `and`/`or` runs are shared."""
    done, out = _harvest(tmp_path, _POOLED_TESTS)
    assert _summary(done) == "6 passed", done.stdout + done.stderr
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    assert sorted(cases) == [
        "backward/sample__backward__2.yaml", "load-agent/sample__backward_unrecorded.yaml",
        "load-core/sample__or_load.yaml", "run-agent/sample__pooled_agents__2.yaml",
        "run-core/sample__or_case__2.yaml", "run-core/sample__pooled_failed__2.yaml",
    ]
    pooled = cases["run-agent/sample__pooled_agents__2.yaml"]
    assert pooled["run"] == {"run_id": "r1"}
    assert sorted(pooled["host"]["models"]) == ["a", "b"]
    # Three equal replays of a run that succeeded keep its trace and events.
    assert pooled["expect"]["output"] == {"status": "succeeded",
                                          "output": {"a": "WRITE A.", "b": "WRITE B."}}
    assert pooled["expect"]["trace"] and pooled["expect"]["events"]
    # A pooled run that failed keeps only its output.
    failed = cases["run-core/sample__pooled_failed__2.yaml"]
    assert list(failed["expect"]) == ["output"]
    assert failed["expect"]["output"]["status"] == "failed"

    backward = cases["backward/sample__backward__2.yaml"]
    assert backward["kind"] == "backward"
    assert backward["run"] == {"run_id": "r1", "num_workers": 0, "learnable": True}
    assert backward["backward"] == [{"feedback": "Shorter.", "api_key": "<key:1>"}]
    models = backward["host"]["models"]["write"]
    assert [len(models[consumer]) for consumer in ("agent", "backward")] == [1, 1]
    assert json.loads(models["backward"][0]["reply"]["parts"][0]["text"]) == {
        "input_feedback": {}, "self_feedback": [], "variable_feedback": {}}
    assert [entry["status"] for entry in backward["expect"]["backward"]] == ["succeeded"]
    assert [entry[0] for entry in backward["expect"]["trace"]][-3:] == [
        "backward_started", "backward_ready", "backward_finished"]
    for name, case in cases.items():
        case_dir = tmp_path / "replay" / name
        case_dir.mkdir(parents=True)
        _check_results(case, results(case, case_dir))

    report = json.loads((out / "report" / "test_sample.py.json").read_text(encoding="utf-8"))
    assert report["tests"] == {"passed": 6, "failed": 0, "skipped": 0}
    assert report["written"] == {"run-agent": 1, "run-core": 2, "backward": 1, "load-agent": 1,
                                 "load-core": 1}
    assert report["skips"] == {"host: tools": ["test_backward_unrecorded"]}
    assert report["flags"] == {
        "pooled: trace dropped": ["run-core/sample__pooled_failed__2"],
        "pooled: events dropped": ["run-core/sample__pooled_failed__2"],
    }


def test_harvest_records_jitter_draws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Draws through a wrapped `_jitter` during a recorded call go to its `host.jitter`."""
    from tests import conformance_harvest as harvest

    state = harvest._Harvest(None, tmp_path)
    monkeypatch.setattr(harvest, "_STATE", state)
    module = harvest._Watched("tool_node")
    draws = iter([0.5, 0.25, 0.75, 1.0, 0.125])
    module._jitter = lambda: next(draws)
    # Outside any recorded call, a draw goes nowhere.
    assert module._jitter() == 0.5
    call = state.active = harvest._Call("run_flow")
    assert [module._jitter() for _ in range(2)] == [0.25, 0.75]
    assert call.jitter == [0.25, 0.75] and call.host_skip is None
    # A draw outside [0, 1) can't be a case's; it is still the test's.
    assert module._jitter() == 1.0
    assert call.host_skip == "jitter: draw 1.0 outside [0, 1)"
    assert module._jitter() == 0.125
    assert call.jitter == [0.25, 0.75, 0.125]
    assert call.host_skip == "jitter: draw 1.0 outside [0, 1)"


def test_harvest_checks_pooled_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pooled runs replay thrice, plain and shimmed; a result differing in one is dropped."""
    from tests import conformance_harvest as harvest

    monkeypatch.setattr(harvest, "_STATE", harvest._Harvest(None, tmp_path))
    output = {"status": "succeeded", "output": "hi"}
    raises = {"class": "ValueError", "message": "no"}

    def replay(**results: object) -> dict:
        return {"trace": [["run_started", None]], "events": [["RunStarted"]], "stream": [],
                "tool_calls": {}, "left": [], **results}

    def expected(result: dict, plain: list, shimmed: list | None = None) -> tuple:
        call = harvest._Call("run_flow")
        call.case, call.result, call.skip = {"kind": "run", "flow": "id: f\n"}, result, None
        # The shimmed replays are the plain ones unless given.
        served = {False: list(plain), True: list(plain if shimmed is None else shimmed)}
        monkeypatch.setattr(harvest, "_replay", lambda state, case, shims=(): (
            served[bool(shims)].pop(0), frozenset()))
        _, text, flags = harvest._expected(harvest._STATE, call)
        assert served == {False: [], True: []}
        return yaml.load(text, Loader=_CaseLoader)["expect"], flags

    events = [["RunStarted"], ["RunSucceeded", {"output": "hi"}]]
    succeeded = [replay(output=output), replay(output=output)]
    expect, flags = expected({"output": output}, [*succeeded, replay(output=output, events=events)])
    assert expect == {"output": output, "trace": [["run_started", None]]}
    assert flags == ["pooled: events dropped"]
    # Equal plain replays, but shimmed ones that differ.
    expect, flags = expected({"output": output}, [*succeeded, replay(output=output)],
                             [*succeeded, replay(output=output, events=events)])
    assert expect == {"output": output, "trace": [["run_started", None]]}
    assert flags == ["pooled: events dropped"]

    differing = {"status": "succeeded", "output": "ho"}
    with pytest.raises(harvest._Skip, match="replay differs"):
        expected({"output": output}, [replay(output=output), replay(output=differing),
                                      replay(output=output)])
    with pytest.raises(harvest._Skip, match="shims: replays differ"):
        expected({"output": output}, [*succeeded, replay(output=output)],
                 [*succeeded, replay(output=differing)])

    calls = {"tool": [{"args": {}, "call_id": "c1", "attempt": 1, "node": "n"}]}
    raised = replay(raises=raises, stream=[{"text": "x"}], tool_calls=calls)
    expect, flags = expected({"raises": raises}, [raised, raised, raised])
    assert expect == {"raises": raises}
    assert flags == [f"pooled: {key} dropped"
                     for key in ("trace", "events", "stream", "tool_calls")]


def test_harvest_shim_skips_and_requests(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Shim skips and dropped requests; a pooled run's kept trace is compared under each shim."""
    from tests import conformance_harvest as harvest

    state = harvest._Harvest(None, tmp_path)
    monkeypatch.setattr(harvest, "_STATE", state)
    call = harvest._Call("eval_expr")
    call.case, call.result, call.skip = {"kind": "expr", "expr": "1"}, {"value": 1}, None

    def all_shims(state: object, case: dict, shims: tuple = ()) -> tuple:
        # Only every shim together changes the outcome.
        return {"value": 2 if len(shims) == len(harvest._SHIMS) else 1}, frozenset()

    monkeypatch.setattr(harvest, "_replay", all_shims)
    with pytest.raises(harvest._Skip, match="^shims: no single shim changes the outcome$"):
        harvest._expected(state, call)
    # One a rule no shim covers touches is written for review, with that rule.
    call.case = {"kind": "expr", "expr": "\u0661"}
    _, text, flags = harvest._expected(state, call)
    assert yaml.load(text, Loader=_CaseLoader) == {
        "kind": "expr", "rules": ["text"], "expr": "\u0661", "expect": {"value": 2},
        "python_expect": {"value": 1}}
    assert flags == ["needs TS expect: text"]
    call.case = {"kind": "expr", "expr": "1"}

    def failing(state: object, case: dict, shims: tuple = ()) -> tuple:
        if shims:
            raise harvest._Skip("replay failed: Failed")
        return {"value": 1}, frozenset()

    monkeypatch.setattr(harvest, "_replay", failing)
    with pytest.raises(harvest._Skip, match="^shims: replay failed: Failed$"):
        harvest._expected(state, call)

    output = {"status": "succeeded", "output": "hi"}

    def traced(state: object, case: dict, shims: tuple = ()) -> tuple:
        # A pooled run whose stable trace only the text-form shim changes.
        trace = [["run_started", "text-form" in shims]]
        return {"output": output, "trace": trace, "events": [], "stream": [], "tool_calls": {},
                "left": []}, frozenset()

    monkeypatch.setattr(harvest, "_replay", traced)
    run = harvest._Call("run_flow")
    run.case, run.result, run.skip = {"kind": "run", "flow": "id: f\n"}, {"output": output}, None
    _, text, flags = harvest._expected(state, run)
    written = yaml.load(text, Loader=_CaseLoader)
    assert written["rules"] == ["text-form"] and flags == []
    assert written["expect"] == {"output": output, "trace": [["run_started", True]],
                                 "events": []}
    assert written["python_expect"] == {"output": output, "trace": [["run_started", False]],
                                        "events": []}

    def entry(text: str) -> dict:
        return {"reply": text, "request": {"messages": [text]}}

    case = {"kind": "run", "host": {"models": {"n": {"agent": [entry("a"), entry("b")],
                                                     "memory:a.b": [entry("c")]}},
                                    "jitter": [0.5]}}
    copy = json.loads(json.dumps(case))
    # A consumer holding a `.` is found by its own key.
    trimmed = harvest._without_requests(case, frozenset({"request: host.models.n.memory:a.b[0]"}))
    assert trimmed == {"kind": "run", "host": {
        "models": {"n": {"agent": [entry("a"), entry("b")], "memory:a.b": [{"reply": "c"}]}},
        "jitter": [0.5]}}
    assert case == copy


def test_harvest_replay_installs_shims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay turns on the shims it names; under one, a differing request is noted, not failed."""
    from tests import conformance_harvest as harvest
    from yian.expr import template

    state = harvest._Harvest(None, tmp_path)
    as_text, request_difference = template._as_text, harvest.runner._request_difference

    def text_shim(patch: pytest.MonkeyPatch, case: dict) -> None:
        harvest._NOTES.add("printf")
        patch.setattr(template, "_as_text", lambda value: "shimmed")

    monkeypatch.setattr(harvest, "_SHIMS", {
        "text": harvest._Shim("text-form", text_shim),
        "idle": harvest._Shim("messages", lambda patch, case: None),
    })
    case = {"kind": "expr", "template": "x=${v}", "values": {"v": True}}
    assert harvest._replay(state, case) == ({"value": "x=True"}, frozenset())
    assert harvest._replay(state, case, ("idle",)) == ({"value": "x=True"}, frozenset())
    assert harvest._replay(state, case, ("text",)) == ({"value": "x=shimmed"},
                                                       frozenset({"printf"}))
    # The shim is undone and its note forgotten by the next replay.
    assert template._as_text is as_text and not state.disabled
    assert harvest._replay(state, case) == ({"value": "x=True"}, frozenset())
    with pytest.raises(KeyError, match="no such shims"):
        harvest._replay(state, case, ("text", "missing"))

    # Under a shim, an int a `float` slot widens is noted; it is widened all the same.
    coerce = harvest.typesys_values._coerce_scalar
    noted = frozenset({"int-in-float"})
    for typ, value, kept, notes in (
            ("float", 2, {"value": 2.0, "kind": "float"}, noted),
            ("list(float)", [1, 2.5], {"value": [1.0, 2.5], "kind": "list[float]"}, noted),
            ("float", 2.5, {"value": 2.5, "kind": "float"}, frozenset())):
        case = {"kind": "value", "type": typ, "value": value}
        assert harvest._replay(state, case, ("idle",)) == (kept, notes), typ
    assert harvest._replay(state, {"kind": "value", "type": "float", "value": 2}) == (
        {"value": 2.0, "kind": "float"}, frozenset())
    assert harvest.typesys_values._coerce_scalar is coerce

    run = _read(_ROOT / "cases/run-agent/model_reply.yaml")
    del run["expect"]
    results, notes = harvest._replay(state, run, ("idle",))
    assert results["output"] == {"status": "succeeded", "output": "Hello Ada"}
    assert notes == frozenset()
    run["host"]["models"]["write"]["agent"][0]["request"]["messages"][1]["parts"] = [
        {"type": "text", "text": "Greet Bob."}]
    with pytest.raises(harvest._Skip, match="replay failed"):
        harvest._replay(state, run)
    results, notes = harvest._replay(state, run, ("idle",))
    assert results["output"] == {"status": "succeeded", "output": "Hello Ada"}
    assert notes == frozenset({"request: host.models.write.agent[0]"})
    assert harvest.runner._request_difference is request_difference
    # A replay failing under a shim (here, a key the runner refuses) undoes it all the same.
    with pytest.raises(harvest._Skip, match="replay failed"):
        harvest._replay(state, run | {"unknown": 1}, ("text",))
    assert template._as_text is as_text and not state.disabled
    assert harvest.runner._request_difference is request_difference


_SHIMMED_TESTS = '''
from yian.expr import eval_expr, parse_expr, render_template_record
from yian.typesys.types import type_for
from yian.typesys.values import TypeCheckError, build_value_as


def test_bool_text():
    assert render_template_record("${f}", {"f": True}) == "True"


def test_remainder():
    assert eval_expr(parse_expr("7 % -2"), lambda path: None) == -1


def test_list_order():
    assert eval_expr(parse_expr("[1] < [2]"), lambda path: None) is True


def test_message():
    try:
        build_value_as(type_for("int", {}), "x")
    except TypeCheckError as exc:
        assert str(exc) == "'x' does not match declared type int"


def test_key_order():
    assert render_template_record("${r}", {"r": {"b": 1, "1": 2}}) == "{'b': 1, '1': 2}"


def test_plain():
    assert eval_expr(parse_expr("1 + 2"), lambda path: None) == 3
'''


def test_harvest_tags_shimmed_rules(tmp_path: Path) -> None:
    """A case a shim changes holds the TS outcome, its rules and Python's outcome."""
    done, out = _harvest(tmp_path, _SHIMMED_TESTS)
    assert _summary(done) == "6 passed", done.stdout + done.stderr
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    message = cases["expr/sample__list_order.yaml"]["expect"]["raises"].pop("message")
    assert "unsupported operand kinds" in message
    assert cases == {
        "templates/sample__bool_text.yaml": {
            "kind": "expr", "rules": ["text-form"], "template": "${f}", "values": {"f": True},
            "mode": "strict", "expect": {"value": "true"}, "python_expect": {"value": "True"}},
        "expr/sample__remainder.yaml": {
            "kind": "expr", "rules": ["arithmetic"], "expr": "7 % -2",
            "expect": {"value": 1}, "python_expect": {"value": -1}},
        "expr/sample__list_order.yaml": {
            "kind": "expr", "rules": ["compare"], "expr": "[1] < [2]",
            "expect": {"raises": {"class": "ExpressionError"}}, "python_expect": {"value": True}},
        "values/sample__message.yaml": {
            "kind": "value", "rules": ["messages"], "type": "int", "value": "x",
            "expect": {"raises": {"class": "TypeCheckError",
                                  "message": '"x" does not match declared type int',
                                  "path": []}},
            "python_expect": {"raises": {"class": "TypeCheckError",
                                         "message": "'x' does not match declared type int",
                                         "path": []}}},
        # Both the text form and the key order change it; both are the text-form rule.
        "templates/sample__key_order.yaml": {
            "kind": "expr", "rules": ["text-form"], "template": "${r}",
            "values": {"r": {"b": 1, "1": 2}}, "mode": "strict",
            "expect": {"value": '{"1":2,"b":1}'},
            "python_expect": {"value": "{'b': 1, '1': 2}"}},
        "expr/sample__plain.yaml": {"kind": "expr", "expr": "1 + 2", "expect": {"value": 3}},
    }
    # The rules come first, `python_expect` after `expect`.
    text = (out / "cases/expr/sample__remainder.yaml").read_text(encoding="utf-8")
    assert text == ("kind: expr\nrules: [arithmetic]\nexpr: 7 % -2\nexpect:\n  value: 1\n"
                    "python_expect:\n  value: -1\n")
    for name in cases:
        # Read again: the message was popped above.
        case, case_dir = _read(out / "cases" / name), tmp_path / "replay" / name
        case_dir.mkdir(parents=True)
        _check_results(case, results(case, case_dir))

    report = json.loads((out / "report" / "test_sample.py.json").read_text(encoding="utf-8"))
    assert report["written"] == {"templates": 2, "expr": 3, "values": 1}
    assert report["skips"] == {} and report["flags"] == {}


_REQUEST_TESTS = '''
from langchain_core.messages import AIMessage

import yian.llm_clients as llm_clients
from yian import load_flow
from yian.run import run_flow

FLAG = """\\
id: flag
name: flag
input:
  f: !bool
nodes:
  plain:
    kind: agent
    output: str
    llm_config: {provider: openai, model: gpt-5.5}
    prompt: Write.
  flagged:
    kind: agent
    input:
      f: ${input.f}
    output: str
    llm_config: {provider: openai, model: gpt-5.5}
    prompt: Flag ${f}.
output:
  plain: ${plain.output}
  flagged: ${flagged.output}
"""


class Shout:
    """A chat model replying with its last message in capitals."""

    def invoke(self, messages):
        return AIMessage(messages[-1].content.upper())


def test_flag(monkeypatch):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: Shout())
    result = run_flow(load_flow(FLAG), {"f": True}, run_id="r1", num_workers=0)
    assert result.output == {"plain": "WRITE.", "flagged": "FLAG TRUE."}
'''


def test_harvest_drops_changed_requests(tmp_path: Path) -> None:
    """A request a shim changes leaves its entry without `request`; the others keep theirs."""
    done, out = _harvest(tmp_path, _REQUEST_TESTS)
    assert _summary(done) == "1 passed", done.stdout + done.stderr
    case = _read(out / "cases/run-agent/sample__flag__2.yaml")
    models = case["host"]["models"]
    assert models["plain"]["agent"][0]["request"]["messages"][1]["parts"] == [
        {"type": "text", "text": "Write."}]
    (flagged,) = models["flagged"]["agent"]
    assert "request" not in flagged
    assert flagged["reply"]["parts"] == [{"type": "text", "text": "FLAG TRUE."}]
    # The replies are scripted, so the outcome is the same under the shims.
    assert "rules" not in case and "python_expect" not in case
    assert case["expect"]["output"] == {
        "status": "succeeded", "output": {"plain": "WRITE.", "flagged": "FLAG TRUE."}}
    (tmp_path / "replay").mkdir()
    _check_results(case, results(case, tmp_path / "replay"))

    report = json.loads((out / "report" / "test_sample.py.json").read_text(encoding="utf-8"))
    assert report["written"] == {"run-agent": 1}
    assert report["flags"] == {"shims: request dropped": ["run-agent/sample__flag__2"]}


_UNSHIMMABLE_TESTS = '''
import pytest

from yian import load_flow
from yian.compose import LoadError
from yian.expr import eval_expr, parse_expr, render_template_record
from yian.expr.expressions import ExpressionError
from yian.run import run_flow
from yian.typesys.types import type_for
from yian.typesys.values import build_value_as

TODAY = """\\
id: today
name: today
nodes:
  c:
    kind: code
    input:
      d: ${system.today}
    output: str
    code: return inputs["d"]
output: ${c.output}
"""

BOOM = """\\
id: boom
name: boom
nodes:
  c:
    kind: code
    output: int
    code: |
      raise ValueError("boom")
      return 1
output: ${c.output}
"""


def test_infinite():
    assert build_value_as(type_for("float", {}), float("inf")).value == float("inf")


def test_digits(monkeypatch):
    monkeypatch.setenv("YIAN_MAX_INT_DIGITS", "10")
    with pytest.raises(ExpressionError, match="larger than yian allows"):
        render_template_record("${12345678901}", {})


def test_today():
    load_flow(TODAY)


def test_yaml():
    with pytest.raises(LoadError, match="invalid YAML"):
        load_flow("id: [")


def test_code():
    assert run_flow(load_flow(BOOM), {}, run_id="r1", num_workers=0).status == "failed"


def test_digit():
    assert eval_expr(parse_expr("\\u0661 + 1"), lambda path: None) == 2


def test_int_float():
    assert build_value_as(type_for("float", {}), 2).value == 2.0
'''


def test_harvest_tags_unshimmable_rules(tmp_path: Path) -> None:
    """A case touching a rule no shim covers carries it, `python_expect` and a review flag."""
    done, out = _harvest(tmp_path, _UNSHIMMABLE_TESTS)
    assert _summary(done) == "7 passed", done.stdout + done.stderr
    cases = {path.relative_to(out / "cases").as_posix(): _read(path)
             for path in sorted((out / "cases").glob("*/*.yaml"))}
    tagged = {"values/sample__infinite": "infinity", "templates/sample__digits": "numbers",
              "load-core/sample__today": "dates", "load-core/sample__yaml": "yaml",
              "run-core/sample__code__2": "code", "expr/sample__digit": "text",
              "values/sample__int_float": "int-in-float"}
    assert sorted(cases) == sorted(f"{name}.yaml" for name in tagged)
    for name, rule in tagged.items():
        case = cases[f"{name}.yaml"]
        assert case["rules"] == [rule], name
        # No shim changes these: the TS expectation is written by hand.
        assert case["python_expect"] == case["expect"], name
        case_dir = tmp_path / "replay" / name
        case_dir.mkdir(parents=True)
        _check_results(case, results(case, case_dir))
    assert cases["values/sample__infinite.yaml"]["value"] == {"$float": "inf"}
    assert cases["templates/sample__digits.yaml"]["env"] == {"YIAN_MAX_INT_DIGITS": "10"}
    assert cases["run-core/sample__code__2.yaml"]["expect"]["output"]["failure"]["node"] == "c"
    assert cases["values/sample__int_float.yaml"]["python_expect"] == {"value": 2.0,
                                                                       "kind": "float"}

    report = json.loads((out / "report" / "test_sample.py.json").read_text(encoding="utf-8"))
    assert report["skips"] == {}
    assert report["flags"] == {f"needs TS expect: {rule}": [name]
                               for name, rule in tagged.items()}


def test_harvest_flags_arithmetic_at_load(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """A load or run whose arithmetic TS rejects at load is flagged, with no tag of its own."""
    from tests import conformance_harvest as harvest

    state = harvest._Harvest(None, tmp_path)
    monkeypatch.setattr(harvest, "_STATE", state)

    def noting(state: object, case: dict, shims: tuple = ()) -> tuple:
        return {"load": "ok"}, frozenset({"printf"} if shims else ())

    monkeypatch.setattr(harvest, "_replay", noting)
    for kind, flags in (("load", ["needs TS expect: arithmetic"]), ("expr", [])):
        call = harvest._Call("load_flow")
        call.case, call.result, call.skip = {"kind": kind, "flow": "id: f\n"}, {"load": "ok"}, None
        _, text, written = harvest._expected(state, call)
        assert written == flags
        assert "rules" not in yaml.load(text, Loader=_CaseLoader)


def test_harvest_unshimmable_llm() -> None:
    """Structured model calls are the `llm` rule's, by their request, consumer or events."""
    from tests import conformance_harvest as harvest

    def run(consumer: str = "agent", text: str = "Write.", **request: object) -> dict:
        message = {"role": "user", "parts": [{"type": "text", "text": text}]}
        entry = {"reply": {"role": "assistant", "parts": []},
                 "request": {"messages": [message], **request}}
        return {"kind": "run", "flow": "id: f\n", "host": {"models": {"n": {consumer: [entry]}}}}

    plain = {"output": {"status": "succeeded", "output": "x"}}
    assert harvest._unshimmable(run(), plain, frozenset()) == set()
    for case in (run(schema={"type": "object"}), run("structured_retry"),
                 run(text="Respond with ONLY a JSON object matching this JSON schema: {}"),
                 run(text="Your previous output was invalid: no JSON.")):
        assert harvest._unshimmable(case, plain, frozenset()) == {"llm"}
    events = [["ModelCallStarted", {"node_id": "n", "purpose": "structured_native"}]]
    assert harvest._unshimmable(run(), plain | {"events": events}, frozenset()) == {"llm"}
    events = [["ModelCallStarted", {"node_id": "n", "purpose": "chat"}]]
    assert harvest._unshimmable(run(), plain | {"events": events}, frozenset()) == set()


def test_harvest_unshimmable_texts() -> None:
    """A flow's leading BOM is the `yaml` rule's, any other U+FEFF the `text` rule's."""
    from tests import conformance_harvest as harvest

    def rules(case: dict, py: dict, notes: frozenset = frozenset()) -> set:
        return harvest._unshimmable(case, py, notes)

    loaded = {"load": "ok"}
    assert rules({"kind": "load", "flow": "\ufeffid: f\n"}, loaded) == {"yaml"}
    assert rules({"kind": "load", "flow": "id: f # \ufeff\n"}, loaded) == {"text"}
    assert rules({"kind": "expr", "expr": "\ufeff1"}, {"value": 1}) == {"text"}

    value = {"kind": "value", "type": "float", "value": 2}
    finite = {"raises": {"class": "TypeCheckError", "message": "Input should be a finite number"}}
    assert rules(value, finite) == {"infinity"}
    other = {"raises": {"class": "ValueError", "message": "need a finite number of items"}}
    assert rules(value, other) == set()
    widened = {"value": 2.0, "kind": "float"}
    assert rules(value, widened) == set()
    assert rules(value, widened, frozenset({"int-in-float"})) == {"int-in-float"}


_CODE_FLOW = '''\
id: f
name: f
nodes:
  pick:
    kind: code
    code: m:pick
  body:
    kind: code
    code: |
      x = 1
      return x
  other:
    kind: tool
output: ${body.output}
'''


def test_harvest_unshimmable_code() -> None:
    """Failures of CODE nodes and frames on a `code:` line are the `code` rule's."""
    from tests import conformance_harvest as harvest

    def load_error(line: int, label: str = "flow.yaml", message: str = "bad") -> dict:
        frame = {"label": label, "range": [line, 5, line, 9]}
        return {"load": {"diagnostics": [{"code": "compose.load", "message": message,
                                          "frames": [frame]}]}}

    def failed(node: str) -> dict:
        return {"output": {"status": "failed", "failure": {
            "code": "node.execute", "message": "ValueError: no", "node": node}}}

    load = {"kind": "load", "flow": _CODE_FLOW}
    for line in (6, 9, 10, 11):
        assert harvest._unshimmable(load, load_error(line), frozenset()) == {"code"}, line
    for line in (5, 8, 12, 13):
        assert harvest._unshimmable(load, load_error(line), frozenset()) == set(), line
    assert harvest._unshimmable(load, load_error(3, message="node 'pick': bad `code:` value"),
                                frozenset()) == {"code"}
    # The Python form is what Python ran.
    both = {"kind": "load", "flow": "id: f\n", "python_flow": _CODE_FLOW}
    assert harvest._unshimmable(both, load_error(6), frozenset()) == {"code"}
    # Another label reads the file of that name.
    child = {"kind": "load", "flow": "id: f\nuses: [lib/child.yaml]\n",
             "files": {"lib/child.yaml": _CODE_FLOW}}
    assert harvest._unshimmable(child, load_error(6, "child.yaml"), frozenset()) == {"code"}
    assert harvest._unshimmable(child, load_error(6, "other.yaml"), frozenset()) == set()

    run = {"kind": "run", "flow": _CODE_FLOW}
    for node in ("pick", "body", "loop/pick#2", "pick#1/x"):
        assert harvest._unshimmable(run, failed(node), frozenset()) == {"code"}, node
    for node in ("other", "loop/other#2", "picked"):
        assert harvest._unshimmable(run, failed(node), frozenset()) == set(), node
    failure = failed("body")["output"]["failure"]
    for entry in (["node_failed", "body", {"failure": failure}],
                  ["node_routed", "body", {"terminal_failure": failure}]):
        assert harvest._unshimmable(run, {"trace": [entry]}, frozenset()) == {"code"}, entry[0]
    routed = {"trace": [["node_routed", "body", {"terminal_failure": None}]]}
    assert harvest._unshimmable(run, routed, frozenset()) == set()


def test_harvest_code_scalars() -> None:
    """`_code_scalars` finds the `code:` values of CODE mappings, with their ids and spans."""
    from tests import conformance_harvest as harvest

    def found(text: str) -> list:
        return [(scalar.node, scalar.value, text[scalar.start:scalar.end], scalar.start_line,
                 scalar.end_line) for scalar in harvest._code_scalars(text)]

    assert found(_CODE_FLOW) == [
        ("pick", "m:pick", "m:pick", 6, 6),
        ("body", "x = 1\nreturn x\n", "|\n      x = 1\n      return x\n", 9, 11)]
    flow = ("nodes:\n"
            "  a: {kind: code, code: 'return 1'}\n"
            "  outer:\n"
            "    kind: loop\n"
            "    nodes:\n"
            "      inner:\n"
            "        code: return 2\n"
            "        kind: code\n"
            "    steps: [{kind: code, code: return 3}]\n")
    assert found(flow) == [("a", "return 1", "'return 1'", 2, 2),
                           ("inner", "return 2", "return 2", 7, 7),
                           (None, "return 3", "return 3", 9, 9)]
    # A recursive alias is walked once.
    assert found("a: &x\n  b: *x\n") == []
    assert found("nodes:\n  c: &x {kind: code, code: r, again: *x}\n") == [("c", "r", "r", 2, 2)]
    # A reused alias that isn't recursive is walked each time.
    assert found("nodes:\n  a: &x {kind: code, code: r}\n  b: *x\n") == [
        ("a", "r", "r", 2, 2), ("b", "r", "r", 2, 2)]
    # Not CODE, or not text to compose.
    assert found("nodes:\n  t: {kind: tool, code: x}\n") == []
    assert found("nodes: [\n") == []
    assert found("") == []


def _install_shims(patch: pytest.MonkeyPatch, *names: str) -> None:
    """Install the harvest's shims named in `names`, given in install order, through `patch`."""
    from tests import conformance_harvest as harvest

    for name in names:
        harvest._SHIMS[name].install(patch, {})


def test_shim_js_number() -> None:
    """`_js_number` writes a float as JavaScript's `Number.prototype.toString` does."""
    from tests import conformance_harvest as harvest

    rows = [(1.0, "1"), (1.5, "1.5"), (0.1, "0.1"), (1e21, "1e+21"), (1e-7, "1e-7"),
            (123e-9, "1.23e-7"), (1e20, "100000000000000000000"), (0.000001, "0.000001"),
            (-2.5, "-2.5"), (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"), (math.inf, "Infinity"),
            (-math.inf, "-Infinity"), (-0.0, "0"), (100.0, "100")]
    assert [harvest._js_number(x) for x, _ in rows] == [text for _, text in rows]


def test_shim_ts_text() -> None:
    """`_ts_text` gives a value's text form."""
    from tests import conformance_harvest as harvest

    rows = [("a b", "a b"), (True, "true"), (False, "false"), (1, "1"),
            (12345678901234567890123, "12345678901234567890123"), (1.5, "1.5"),
            (1e21, "1e+21"), (math.inf, "Infinity"), ([1, "a"], '[1,"a"]'),
            ({"k": "v"}, '{"k":"v"}'), (FileRef(uri="s3://b/k"),
                                        '{"uri":"s3://b/k","mime":null,"name":null}'),
            (None, "null"), (ValueError("boom"), "boom"),
            ({"a": [1.0, None, {"é": []}], "b": {}}, '{"a":[1,null,{"é":[]}],"b":{}}')]
    assert [harvest._ts_text(value) for value, _ in rows] == [text for _, text in rows]


def test_shim_render_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under `text-form`, `render_as_json` lays a value out as Python does, numbers as JS."""
    from yian.expr import builtins

    original = builtins.TEMPLATE_FNS["render_as_json"]
    _install_shims(monkeypatch, "text-form")
    render = builtins.TEMPLATE_FNS["render_as_json"]
    assert inspect.signature(render) == inspect.signature(original)
    for value in ({"a": [1, {"b": None, "c": [True, "x"]}], "d": {}, "e": []}, [[], {}, [[1]]],
                  "é", {"name": "Zoë", "tags": ["ü"]}, FileRef(uri="s3://b/k")):
        for indent in (0, 2, 4):
            expected = json.dumps(value.model_dump() if isinstance(value, FileRef) else value,
                                  indent=indent, ensure_ascii=False)
            assert render(value, indent) == expected
    assert render({"x": 1.0, "y": [math.inf, -math.inf, 0.5]}) == (
        '{\n  "x": 1,\n  "y": [\n    Infinity,\n    -Infinity,\n    0.5\n  ]\n}')
    assert render([1.0], 0) == "[\n1\n]"
    # The design defines only `indent >= 0`; any other indent is Python's own layout.
    assert render({"x": 1.0}, "\t") == '{\n\t"x": 1.0\n}'
    with pytest.raises(TypeError, match=r"render_as_json\(\) got null"):
        render(None)


def test_shim_js_key_order() -> None:
    """`_js_ordered` lists a record's array-index keys first, ascending, at every depth."""
    from tests import conformance_harvest as harvest

    assert list(harvest._js_ordered({"b": 1, "10": 2, "a": 3, "2": 4})) == ["2", "10", "b", "a"]
    assert list(harvest._js_ordered({"01": 1, "4294967295": 2, "4294967294": 3, "0": 4})) == [
        "0", "4294967294", "01", "4294967295"]
    nested = harvest._js_ordered({"x": [{"1": "a", "0": {"9": 1, "3": 2}}]})
    assert list(nested["x"][0]) == ["0", "1"] and list(nested["x"][0]["0"]) == ["3", "9"]
    assert harvest._js_ordered([1, "a", None]) == [1, "a", None]
    assert harvest._js_ordered(1.5) == 1.5


def test_shim_text_form_through_yian() -> None:
    """Python yian writes values in the text form under the shims, and as `str` without them."""
    from yian.expr import builtins
    from yian.expr.expressions import ExpressionError
    from yian.nodes.agent import memory

    values = {"flag": True, "n": 1.0, "xs": [1, 1.5, True, "a", [1, "b"]],
              "r": {"a": 1.0, "b": [True, None]}, "rec": {"b": 1, "1": 2}}
    # Without the shims, Python's spellings.
    assert eval_binding("x=${flag}", values.get) == "x=True"
    assert render_template_record("${n}", values) == "1.0"
    assert memory._tool_text("t", None) == "None"
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, "key-order")
        for name in ("join", "render_as_json", "upper", "lower"):
            assert builtins.builtin_result_type(name) == builtins.builtin_result_type("upper")
        assert builtins.builtin_result_type("join") is not None
        assert render_template_record("${join(xs, \",\")}", values) == "1,1.5,True,a,[1, 'b']"
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, "text-form")
        assert eval_binding("x=${flag}", values.get) == "x=true"
        assert render_template_record("${n}", values) == "1"
        assert render_template_record("${render_as_json(r)}", values) == (
            '{\n  "a": 1,\n  "b": [\n    true,\n    null\n  ]\n}')
        assert render_template_record("${join(xs, \",\")}", values) == '1,1.5,true,a,[1,"b"]'
        assert render_template_record("r=${rec}", values) == 'r={"b":1,"1":2}'
        assert memory._tool_text("t", None) == "null"
        assert memory._tool_text("t", {"k": [1.0]}) == '{"k":[1]}'
        assert memory._reported_text("t", ReportedToolError("no such row")) == "no such row"
        assert memory._reported_text("t", ReportedToolError("")) == "ReportedToolError"
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, "text-form", "key-order")
        assert render_template_record("r=${rec}", values) == 'r={"1":2,"b":1}'
        assert render_template_record("${render_as_json(rec, 0)}", values) == (
            '{\n"1": 2,\n"b": 1\n}')
        assert render_template_record("${join([rec], \",\")}", values) == '{"1":2,"b":1}'
        assert memory._tool_text("t", {"b": 1, "0": 2}) == '{"0":2,"b":1}'
        with pytest.raises(ExpressionError, match="takes a list, got str"):
            render_template_record("${join(n2, \",\")}", {"n2": "ab"})
        with pytest.raises(ExpressionError, match=r"render_as_json"):
            render_template_record("${render_as_json(r, 1, 2)}", values)
    assert eval_binding("x=${flag}", values.get) == "x=True"
    assert memory._tool_text("t", None) == "None"


def test_shim_text_form_replay(tmp_path: Path) -> None:
    """A replay under `text-form` writes the text form; key order tags the same rule."""
    from tests import conformance_harvest as harvest

    assert harvest._SHIMS["key-order"].rule == "text-form"
    case = {"kind": "expr", "template": "x=${v}", "values": {"v": True}}
    assert harvest._replay(harvest._Harvest(None, tmp_path), case, ("text-form",)) == (
        {"value": "x=true"}, frozenset())


def test_shim_value_repr() -> None:
    """`_ts_value_repr` shows a value in the text form, text JSON-quoted, cut as `value_repr`."""
    from tests import conformance_harvest as harvest
    from yian.typesys.values import shown_text, value_repr

    rows = [(True, "true"), (None, "null"), ("x", '"x"'), (["a"], '["a"]'), ({"k": 1.0}, '{"k":1}')]
    assert [harvest._ts_value_repr(value) for value, _ in rows] == [text for _, text in rows]
    assert harvest._ts_value_repr("x" * 300) == shown_text('"' + "x" * 300 + '"')
    assert harvest._ts_value_repr("x" * 300).endswith("x…")
    # Long-int placeholders are the numbers rule's, so they stay Python's.
    assert harvest._ts_value_repr([10**5000]) == value_repr([10**5000])


def test_shim_value_repr_through_yian() -> None:
    """Under `messages`, yian's messages show values in the text form and names as Python does."""
    from tests import conformance_harvest as harvest
    from yian.expr import expressions
    from yian.nodes.binding import check_input_names
    from yian.nodes.contracts import Parameter, PartialRecordContract, validate_output_contract
    from yian.typesys import values
    from yian.typesys.values import Type, ValueKind, reject_non_text_keys

    record = Type(kind=ValueKind.OBJECT, fields={"a": Type.scalar(ValueKind.INTEGER)},
                  required=frozenset({"a"}))
    names = [
        (lambda: check_input_names([Parameter("a")], {"zz": 1}), "unknown input 'zz'"),
        (lambda: validate_output_contract(PartialRecordContract((("a", None),)),
                                          {"a": 1, "zz": 2}), "unknown fields: ['zz']"),
        (lambda: build_value_as(record, {"a": 1, "zz": 2}), "unknown fields: ['zz']"),
        (lambda: reject_non_text_keys({None: 1}, "input 'x'"), "must be text, got None"),
    ]

    def raised(call) -> str:
        with pytest.raises((TypeCheckError, ValueError)) as caught:
            call()
        return str(caught.value)

    def not_an_int() -> str:
        return raised(lambda: build_value_as(type_for("int", {}), "not-an-int"))

    original = values.value_repr
    assert not_an_int() == "'not-an-int' does not match declared type int"
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, "messages")
        assert expressions.value_repr is values.value_repr is not original
        assert inspect.signature(values.value_repr) == inspect.signature(original)
        assert not_an_int() == '"not-an-int" does not match declared type int'
        assert expressions._operand_repr(True) == "true"
        for call, message in names:
            assert message in raised(call)
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, "key-order")
        assert values.value_repr({"b": 1, "0": 2}) == "{'0': 2, 'b': 1}"
        assert expressions.value_repr is values.value_repr
        cyclic: dict = {}
        cyclic["x"] = cyclic
        assert values.value_repr(cyclic) == "{'x': {...}}"
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, *harvest._SHIMS)
        assert inspect.signature(values.value_repr) == inspect.signature(original)
        assert expressions.value_repr({"b": 1, "0": 2}) == '{"0":2,"b":1}'
        assert not_an_int() == '"not-an-int" does not match declared type int'
        for call, message in names:
            assert message in raised(call)
    assert values.value_repr is original and expressions.value_repr is original


def test_shim_value_repr_name_sites() -> None:
    """Each of the `messages` shim's name sites is one code object that calls `value_repr`."""
    import types

    from tests import conformance_harvest as harvest
    import yian

    root = Path(yian.__file__).parent
    for path, qualname in harvest._NAME_CALLERS:
        source = root / path
        stack = [compile(source.read_text(encoding="utf-8"), str(source), "exec")]
        found = 0
        while stack:
            code = stack.pop()
            stack += [const for const in code.co_consts if isinstance(const, types.CodeType)]
            found += code.co_qualname == qualname and "value_repr" in code.co_names
        assert found == 1, (path, qualname)


def _shim_outcome(expr: str, values: dict) -> tuple[str, set]:
    """Evaluate `expr` over `values`: return its repr or `raises: <message>`, and the notes."""
    from tests import conformance_harvest as harvest
    from yian.expr.expressions import ExpressionError

    harvest._NOTES.clear()
    try:
        shown = repr(eval_expr(parse_expr(expr), values.get))
    except ExpressionError as exc:
        shown = f"raises: {exc}"
    return shown, set(harvest._NOTES)


def _kinds_raise(l: object, op: str, r: object, k1: str, k2: str) -> str:
    """Return `_shim_outcome`'s text for the unsupported-operand-kinds error of `l op r`."""
    return (f"raises: type error evaluating {l} {op} {r}: unsupported operand kinds for `{op}`: "
            f"`{k1}` and `{k2}`")


def test_shim_arithmetic() -> None:
    """Under `arithmetic`, the operators follow design 3.2.2 and note where Python differs."""
    from tests import conformance_harvest as harvest
    from yian.expr import expressions

    values = {"inf": math.inf, "r": {}, "f": FileRef(uri="s3://b/k")}
    rows = [
        ("7 % -2", "1", set()),
        ("-7 % 2", "-1", set()),
        ("-7.5 % 2", "-1.5", set()),
        ("5 % inf", "5.0", set()),
        ("inf % 2", "raises: % on Infinity and 2 has no numeric result", set()),
        ("10 / 4", "2.5", set()),
        ("10 ** 400 / 10 ** 399", "10.0", set()),
        ("1 + 0.5", "1.5", set()),
        ("2 ** 100", repr(2**100), set()),
        ("2 ** -1", "0.5", set()),
        ("(10 ** 400) ** -1", "0.0", {"infinity"}),
        ("true + 1", "2", set()),
        ("-true", "-1", set()),
        ("-(0.0)", "-0.0", set()),
        ('"a" + "b"', "'ab'", set()),
        ("[1] + [2]", "[1, 2]", set()),
        ('"a" + 1', _kinds_raise('"a"', "+", 1, "str", "int"), {"operand-kinds"}),
        ("r + r", _kinds_raise("{}", "+", "{}", "record", "record"), {"operand-kinds"}),
        ("null + 1", _kinds_raise("null", "+", 1, "null", "int"), {"operand-kinds"}),
        ("f + 1", _kinds_raise('{"uri":"s3://b/k","mime":null,"name":null}', "+", 1, "file",
                               "int"), {"operand-kinds"}),
        ('-"a"', 'raises: unary minus on non-number "a": unsupported operand kind for `-`: `str`',
         {"operand-kinds"}),
        ("-null", "raises: unary minus on non-number null: unsupported operand kind for `-`: "
         "`null`", {"operand-kinds"}),
        ('"ab" * 3', _kinds_raise('"ab"', "*", 3, "str", "int"),
         {"operand-kinds", "repetition"}),
        ("[0] * 3", _kinds_raise("[0]", "*", 3, "list", "int"),
         {"operand-kinds", "repetition"}),
        ('"%s" % 1', "raises: text formatting with `%` was removed; use a template", {"printf"}),
        ("10 ** 400 / 1", "inf", {"infinity"}),
        ('"a" - 1', _kinds_raise('"a"', "-", 1, "str", "int"), {"operand-kinds"}),
        ("1 / 0", "raises: division by zero: 1 / 0", set()),
        ("1.0 % 0.0", "raises: division by zero: 1 % 0", set()),
        ("0 ** -1", "raises: division by zero: 0 ** -1", set()),
        ("0.0 ** -1", "raises: division by zero: 0 ** -1", set()),
        ("(0 - 8) ** 0.5", "raises: ** on -8 and 0.5 has no numeric result", set()),
        ("inf - inf", "raises: - on Infinity and Infinity has no numeric result", set()),
        ("0 * inf", "raises: * on 0 and Infinity has no numeric result", set()),
        ("inf / inf", "raises: / on Infinity and Infinity has no numeric result", set()),
        ("1 ** inf", "1.0", set()),
        ("(0 - 1) ** inf", "1.0", set()),
        ("2.0 ** 5000", "inf", {"infinity"}),
        ("(0 - 2.0) ** 5001", "-inf", {"infinity"}),
        ("10 ** 400 * 1.0", "inf", {"infinity"}),
    ]
    apply_step, neg = expressions._EvalExpr._apply_step, expressions._EvalExpr._do_neg
    with pytest.MonkeyPatch.context() as patch:
        # `messages` spells the operands as TS does, so the messages read as under every shim.
        _install_shims(patch, "arithmetic", "messages")
        outcomes = [_shim_outcome(expr, values) for expr, _, _ in rows]
    assert outcomes == [(shown, notes) for _, shown, notes in rows]
    assert harvest._SHIMS["arithmetic"].rule == "arithmetic"
    assert expressions._EvalExpr._apply_step is apply_step and expressions._EvalExpr._do_neg is neg
    assert _shim_outcome('"%s" % 1', values) == ("'1'", set())


def test_shim_compare() -> None:
    """Under `compare`, comparisons and `in` follow design 3.2.2, noting nothing."""
    from tests import conformance_harvest as harvest

    values = {"inf": math.inf, "r": {"k": 1}, "a": {"a": 1}, "hi": "\uffff",
              "smile": "\U0001F600"}
    rows = [
        ("[1, 2] < [1, 3]", _kinds_raise("[1,2]", "<", "[1,3]", "list", "list")),
        ('"a" < 1', _kinds_raise('"a"', "<", 1, "str", "int")),
        ("hi < smile", "False"),
        ("null < 1", "False"),
        ("true < 2", "True"),
        ("1 < inf", "True"),
        ("1 == 1.0", "True"),
        ("2 ** 60 == 2.0 ** 60", "True"),
        ("1 in [1.0]", "True"),
        ('"k" in r', "True"),
        ("[1] in a", "False"),
        ('1 in "abc"', _kinds_raise(1, "in", '"abc"', "int", "str")),
        ('"b" in "abc"', "True"),
        ("1 in null", _kinds_raise(1, "in", "null", "int", "null")),
        ("1 not in null", _kinds_raise(1, "not in", "null", "int", "null")),
    ]
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, "compare", "messages")
        outcomes = [_shim_outcome(expr, values) for expr, _ in rows]
    assert outcomes == [(shown, set()) for _, shown in rows]
    assert harvest._SHIMS["compare"].rule == "compare"
    # Python orders text by code point.
    assert _shim_outcome("hi < smile", values) == ("True", set())


def test_shim_ts_range() -> None:
    """`_ts_range` maps a Python range to TS lines (LF, CRLF, CR) and UTF-16 columns."""
    from tests import conformance_harvest as harvest

    smile = "\U0001F600"
    rows = [
        (f"a{smile}b", [1, 3, 1, 4], [1, 4, 1, 5]),
        (f"a{smile}b", [1, 1, 1, 2], [1, 1, 1, 2]),
        # NEL and U+2028 break lines for PyYAML only, so later lines move up.
        ("a\x85b\nc", [3, 1, 3, 2], [2, 1, 2, 2]),
        ("a\x85b\nc", [2, 1, 2, 2], [1, 3, 1, 4]),
        ("x\u2028y\nz", [3, 1, 3, 2], [2, 1, 2, 2]),
        (f"{smile}\x85b", [2, 1, 2, 2], [1, 4, 1, 5]),
        ("a\r\nb\rc", [3, 1, 3, 2], [3, 1, 3, 2]),
        ("a\r\nb\rc", [2, 1, 2, 2], [2, 1, 2, 2]),
        # A position past its line keeps its distance from the line end; a line past the text
        # is kept.
        (f"{smile}b", [1, 3, 1, 4], [1, 4, 1, 5]),
        ("a\r\nb", [1, 3, 1, 4], [1, 3, 1, 4]),
        ('id: f\nname: f\nk: "\U0001F600', [3, 6, 3, 7], [3, 7, 3, 8]),
        ("ab", [5, 1, 5, 2], [5, 1, 5, 2]),
        # A text starting with a BOM is the `yaml` rule's.
        (f"\ufeffa{smile}b", [1, 4, 1, 5], [1, 4, 1, 5]),
    ]
    assert [harvest._ts_range(text, rng) for text, rng, _ in rows] == [ts for _, _, ts in rows]


_ASTRAL_LOAD_FLOW = '''\
id: f
name: f
uses:
  child: child
nodes:
  go:
    kind: call
    call: child
output: ${go.output}
'''
_ASTRAL_CHILD = '''\
id: child
name: child
nodes:
  a: {code: "return '\U0001F600'", kind: code, output: str, bogus: 1}
output: ${a.output}
'''


def test_shim_columns(tmp_path: Path) -> None:
    """Under `columns`, a diagnostic's frames give TS columns in the text their label names."""
    from tests import conformance_harvest as harvest

    state = harvest._Harvest(None, tmp_path)
    flow = 'id: f\nname: f\nnodes:\n  a:\n    kind: code\n    title: "\U0001F600" ]\n'
    load = {"kind": "load", "flow": "unused", "python_flow": flow}

    def ranges(case: dict, shims: tuple = ()) -> list:
        results, _ = harvest._replay(state, case, shims)
        return [frame["range"] for frame in results["load"]["diagnostics"][0]["frames"]]

    assert ranges(load) == [[6, 16, 6, 17]]
    assert ranges(load, ("columns",)) == [[6, 17, 6, 18]]
    child = {"kind": "load", "flow": _ASTRAL_LOAD_FLOW, "files": {"child.yaml": "unused"},
             "python_files": {"child.yaml": _ASTRAL_CHILD}}
    assert ranges(child) == [[4, 10, 4, 15], [4, 59, 4, 60]]
    assert ranges(child, ("columns",)) == [[4, 10, 4, 15], [4, 60, 4, 61]]
    assert harvest._SHIMS["columns"].rule == "columns"


def test_shim_dollar() -> None:
    """Under `dollar`, `!!timestamp` text with a trailing newline is no longer a date."""
    from tests import conformance_harvest as harvest
    from yian.compose.source.scalars import _FlowLoader

    newline, plain = '!!timestamp "2026-01-01\\n"', '!!timestamp "2026-01-01"'
    # Python's `$` also matches before a trailing newline.
    assert yaml.load(newline, Loader=_FlowLoader) == date(2026, 1, 1)
    pattern = _FlowLoader.timestamp_regexp
    with pytest.MonkeyPatch.context() as patch:
        _install_shims(patch, "dollar")
        with pytest.raises(ConstructorError, match="expected a date or timestamp"):
            yaml.load(newline, Loader=_FlowLoader)
        assert yaml.load(plain, Loader=_FlowLoader) == date(2026, 1, 1)
    assert _FlowLoader.timestamp_regexp is pattern
    assert harvest._SHIMS["dollar"].rule == "text"


def test_shim_ts_intake() -> None:
    """`_ts_intake` reads an integral float within ±(2^53-1) as an int, at every depth."""
    from tests import conformance_harvest as harvest

    rows = [(2.0, 2), (-0.0, 0), (2.5, 2.5), (1e300, 1e300), (math.inf, math.inf),
            (2.0**53, 2.0**53), (-(2.0**53 - 1), -(2**53 - 1)), ([1.0, {"a": 3.0}], [1, {"a": 3}]),
            (True, True), ("2.0", "2.0")]
    assert [repr(harvest._ts_intake(value)) for value, _ in rows] == [
        repr(intake) for _, intake in rows]
    assert math.isnan(harvest._ts_intake(math.nan))
    # Unchanged values come back as themselves.
    unchanged = [1, {"a": 2.5}]
    assert harvest._ts_intake(unchanged) is unchanged


_INTAKE_FLOW = '''\
id: f
name: f
nodes:
  a:
    kind: code
    output: object
    code: |-
      return {"n": 2.0}
  v: !float 2.0
output:
  a: ${a.output}
  v: ${v.output}
'''


def test_shim_code_intake(tmp_path: Path) -> None:
    """`code-intake` stores a CODE body's integral float as an int, but not a value entry's."""
    from tests import conformance_harvest as harvest
    from yian.nodes.code.node import CodeNode

    state = harvest._Harvest(None, tmp_path)
    case = {"kind": "run", "flow": _INTAKE_FLOW, "run": {"run_id": "r1", "num_workers": 0}}
    forward = CodeNode.forward

    def stored(shims: tuple = ()) -> tuple[str, str, frozenset]:
        results, notes = harvest._replay(state, case, shims)
        [entry] = [entry for entry in results["trace"] if entry[:2] == ["node_succeeded", "a"]]
        # By repr, so 2 and 2.0 differ.
        return repr(results["output"]["output"]), repr(entry[2]["value"]), notes

    assert stored() == ("{'a': {'n': 2.0}, 'v': 2.0}", "{'n': 2.0}", frozenset())
    assert stored(("code-intake",)) == ("{'a': {'n': 2}, 'v': 2.0}", "{'n': 2}", frozenset())
    assert CodeNode.forward is forward
    assert harvest._SHIMS["code-intake"].rule == "numbers"


def test_python_counts_code_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a shim, yian's character caps count an astral character as one."""
    from yian.skills import Skill
    from yian.skills.tool import system_prompt_with_skills
    from yian.typesys.values import shown_text

    smile = "\U0001F600"
    monkeypatch.setenv("YIAN_MAX_SHOWN_VALUE_CHARS", "3")
    monkeypatch.setenv("YIAN_MAX_SKILL_DESCRIPTION_CHARS", "3")
    defaults._forget()
    try:
        assert shown_text(smile * 3) == smile * 3
        assert shown_text(smile * 4) == smile * 2 + "…"
        listed = system_prompt_with_skills("base", [Skill("a", smile * 3, ""),
                                                    Skill("b", smile * 4, "")])
        assert listed.endswith(f"- a: {smile * 3}\n- b: {smile * 2}…")
    finally:
        defaults._forget()
