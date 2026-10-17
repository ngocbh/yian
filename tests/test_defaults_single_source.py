"""Every default and limit is defined only in `yian.defaults`.

The scan parses source files and never imports `yian`, so it runs unchanged on code from
before `yian.defaults` existed.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "yian"

# Each `(path relative to src/yian, name)` the scan finds that is not a default or limit,
# with the reason. A method parameter is `Class.method.parameter`.
ALLOWED: dict[tuple[str, str], str] = {
    # Format versions and fixed codes.
    ("runtime/log.py", "SCHEMA_VERSION"): "run-record schema version",
    ("runtime/log.py", "_RETIRED_SCHEMAS"): "schema versions this version refuses by name",
    ("memory/episodic.py", "_SNAPSHOT_VERSION"): "memory snapshot format version",
    ("memory/reflective.py", "_SNAPSHOT_VERSION"): "memory snapshot format version",
    ("memory/storage/sqlite.py", "_VERSION"): "sqlite store format version",
    ("compile/validation.py", "_RELATION_RANK"): "ordering of type relations",
    ("cli/chat/tools.py", "_open_no_follow.mode"): "file permission bits, as os.open",
    ("llm/errors.py", "_STATUS_KINDS"): "HTTP status codes mapped to error kinds",
    ("tools/mcp.py", "_CONNECTION_CLOSED"): "MCP SDK error code",
    ("tools/mcp.py", "_REQUEST_TIMEOUT"): "MCP SDK error code",
    ("tools/mcp.py", "_INTERNAL_ERROR"): "JSON-RPC error code",
    ("tools/mcp.py", "_INVALID_REQUEST"): "JSON-RPC error code",
    ("tools/mcp.py", "_INVALID_REQUEST_CODES"): "JSON-RPC error codes",
    # Display and message formatting.
    ("cli/run.py", "_ERR_CONTEXT"): "lines of YAML shown around an error",
    ("compose/validate/case_branches.py", "_SHOWN_LIMIT"): "width of a value shown in a message",
    ("run.py", "_LITERAL_LABEL_WIDTH"): "width of a literal shown in a type label",
    ("expr/builtins.py", "_render_as_json.indent"): "JSON indent of the render_as_json builtin",
    ("llm_clients/redact.py", "_MIN_VISIBLE"): "key characters a masked echo must show to count",
    ("llm_clients/base_client.py", "warn_once.stacklevel"): "frames to skip, as warnings.warn",
    ("llm/errors.py", "_MAX_DETAIL_CHARS"): "length of provider error text kept in a message",
    ("llm/testing.py", "_MAX_DIFFS"): "differences listed in a cassette mismatch message",
    ("tools/mcp.py", "_STDERR_LINES"): "MCP server stderr lines shown in a start error",
    # Internal sizes that no user input reaches.
    ("expr/lineage.py", "_MAX_SNAPSHOT_DEPTH"): "diagnostic retention depth, internal",
    ("typesys/values.py", "_LOG10_2"): "log10(2), a math constant for counting digits",
    ("llm/executor.py", "_MAX_ERROR_BODY"): "bytes of an error response read, internal",
    # Host call options and network tuning, passed per call.
    ("run.py", "run_flow.num_workers"): "host run option, passed per call",
    ("run.py", "_run_flow_controlled.num_workers"): "host run option, passed per call",
    ("llm_clients/auth/openai.py", "OpenAIAuth.__init__.request_timeout"): "network timeout",
    ("llm_clients/auth/openai.py", "OpenAIAuth.browser_login.timeout"): "sign-in timeout",
    ("llm_clients/auth/openai.py", "OpenAIAuth.device_login.timeout"): "sign-in timeout",
    ("llm_clients/auth/store.py", "AuthStore.__init__.lock_timeout"): "file lock timeout",
    ("llm_clients/auth/store.py", "AuthStore.__init__.refresh_margin"): (
        "seconds before token expiry to refresh"
    ),
    ("llm_clients/model_list.py", "list_models.timeout"): "network timeout",
    ("llm/route.py", "Route.max_retries"): "transport retries, set per route",
    ("tools/mcp.py", "_MAX_PAGES"): "tool-list pages read from an MCP server, as the SDK's cap",
    ("tools/mcp.py", "_CALL_GRACE_SECONDS"): "margin past an MCP call's own timeout",
    ("tools/mcp.py", "_PING_SECONDS"): "MCP liveness probe timeout",
    ("tools/mcp.py", "_CLOSE_JOIN_SECONDS"): "wait for an MCP server's thread, covering stdio shutdown",
    ("llm_clients/openai_subscription.py", "create_subscription_model.max_retries"): (
        "SDK transport retries"
    ),
}

# Files whose int literals are all outside the scan's scope.
ALLOWED_FILES: dict[str, str] = {
    "llm_clients/provider_blocks.py": "provider API contracts",
}


def _is_flagged_int(node: ast.AST) -> bool:
    """Return whether `node` is an int constant of at least 2 or a negative int literal."""
    if isinstance(node, ast.Constant):
        value = node.value
        return isinstance(value, int) and not isinstance(value, bool) and value >= 2
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        return (
            isinstance(operand, ast.Constant)
            and isinstance(operand.value, int)
            and not isinstance(operand.value, bool)
            and operand.value > 0
        )
    return False


def _target_names(target: ast.expr) -> list[str]:
    """Return the names an assignment target binds."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_names(element)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


class _Scanner(ast.NodeVisitor):
    """Collect flagged names in one module.

    An assignment counts when it binds a name at module or class level, including inside
    `if`, `try`, `with`, and loops at that level; a parameter default counts in every
    function and lambda. Either counts when a flagged int appears anywhere in its
    expression. Names are qualified by the enclosing classes and functions, so a method
    parameter is `Class.method.parameter`.
    """

    def __init__(self) -> None:
        self.hits: set[str] = set()
        self._names: list[str] = []
        self._in_function: list[bool] = [False]

    def _record(self, name: str) -> None:
        self.hits.add(".".join([*self._names, name]))

    def _record_targets(self, targets: list[ast.expr], value: ast.expr | None) -> None:
        if self._in_function[-1] or value is None:
            return
        if any(_is_flagged_int(node) for node in ast.walk(value)):
            for target in targets:
                for name in _target_names(target):
                    self._record(name)

    # Assignments recurse so a lambda in the assigned value has its defaults checked.
    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_targets(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_targets([node.target], node.value)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter(node, in_function=False)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._record_defaults("<lambda>", node.args)
        self._names.append("<lambda>")
        self._in_function.append(True)
        self.generic_visit(node)
        self._in_function.pop()
        self._names.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._record_defaults(node.name, node.args)
        self._enter(node, in_function=True)

    def _record_defaults(self, function: str, args: ast.arguments) -> None:
        positional = [*args.posonlyargs, *args.args]
        pairs = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
        pairs += [
            (arg, default)
            for arg, default in zip(args.kwonlyargs, args.kw_defaults)
            if default is not None
        ]
        for arg, default in pairs:
            if any(_is_flagged_int(node) for node in ast.walk(default)):
                self._record(f"{function}.{arg.arg}")

    def _enter(self, node: ast.AST, *, in_function: bool) -> None:
        self._names.append(node.name)  # type: ignore[attr-defined]
        self._in_function.append(in_function)
        self.generic_visit(node)
        self._in_function.pop()
        self._names.pop()


def _flagged_names(source: str) -> set[str]:
    """Return the flagged names in one module's source."""
    scanner = _Scanner()
    scanner.visit(ast.parse(source))
    return scanner.hits


@functools.cache
def _scan() -> frozenset[tuple[str, str]]:
    """Return `(path relative to src/yian, name)` for every flagged binding."""
    hits: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative == "defaults.py" or relative in ALLOWED_FILES:
            continue
        names = _flagged_names(path.read_text(encoding="utf-8"))
        hits.update((relative, name) for name in names)
    return frozenset(hits)


def test_scan_flags_each_shape_of_default() -> None:
    source = """
MAX_BYTES = 256 * 1024
_LIMITS = _Limits(100, 10)
max_int_digits = 4300
NO_CAP = -1
ONE = 1
ENABLED = True
if True:
    GUARDED = 5

class Model:
    name: str = Field(max_length=64)
    retries = 2

    def method(self, a, b=3, *, c=-1, d=1):
        LOCAL = 99
        def inner(e=7):
            pass

def function(limit=10, flag=False):
    pass

def product(n=256 * 1024, m=1 << 20, ok=None):
    pass

def positional(p=8, /, q=1):
    pass

async def fetch(timeout=30.0, attempts=3):
    pass

handler = lambda size=64: size
call(lambda depth=-2: depth)

@dataclass
class Settings:
    width: int = field(default=2)

    class Inner:
        depth = 12

try:
    TRIED = 6
except ImportError:
    pass
with context():
    WITHIN = 9
"""
    assert _flagged_names(source) == {
        "MAX_BYTES",
        "_LIMITS",
        "max_int_digits",
        "NO_CAP",
        "GUARDED",
        "Model.name",
        "Model.retries",
        "Model.method.b",
        "Model.method.c",
        "Model.method.inner.e",
        "function.limit",
        "product.n",
        "product.m",
        "positional.p",
        "fetch.attempts",
        "handler",
        "<lambda>.size",
        "<lambda>.depth",
        "Settings.width",
        "Settings.Inner.depth",
        "TRIED",
        "WITHIN",
    }


def test_defaults_and_limits_are_defined_only_in_yian_defaults() -> None:
    unexpected = sorted(_scan() - ALLOWED.keys())
    assert not unexpected, (
        "int defaults or limits defined outside yian.defaults (move them there, or add an "
        "ALLOWED entry with the reason they are not a default or limit):\n"
        + "\n".join(f"  {path}: {name}" for path, name in unexpected)
    )


def test_allowlist_has_no_stale_entries() -> None:
    stale = sorted(ALLOWED.keys() - _scan())
    assert not stale, "ALLOWED entries the scan no longer finds:\n" + "\n".join(
        f"  {path}: {name}" for path, name in stale
    )


def test_allowed_files_exist() -> None:
    missing = sorted(name for name in ALLOWED_FILES if not (SRC / name).is_file())
    assert not missing, f"ALLOWED_FILES entries that no longer exist: {missing}"


_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Pow: lambda a, b: a**b,
    ast.LShift: lambda a, b: a << b,
}


def _constant_int(node: ast.AST) -> int | None:
    """Return the int a constant expression of int literals evaluates to, else `None`."""
    if isinstance(node, ast.Constant):
        value = node.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _constant_int(node.operand)
        return None if operand is None else -operand
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _constant_int(node.left), _constant_int(node.right)
        if left is None or right is None:
            return None
        try:
            return _BINARY_OPS[type(node.op)](left, right)
        except (ArithmeticError, ValueError):
            return None
    return None


def _parse(relative: str) -> ast.Module:
    """Return the parsed module at `relative` under src/yian."""
    return ast.parse((SRC / relative).read_text(encoding="utf-8"))


def _compares_length_to(tree: ast.AST, value: int) -> bool:
    """Return whether `tree` compares a `len(...)` call against a constant `value`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        has_length = any(
            isinstance(operand, ast.Call)
            and isinstance(operand.func, ast.Name)
            and operand.func.id == "len"
            for operand in operands
        )
        if has_length and any(_constant_int(operand) == value for operand in operands):
            return True
    return False


def test_inline_cap_guards_flag_each_shape() -> None:
    assert _constant_int(ast.parse("16 * 1024", mode="eval").body) == 16384
    assert _constant_int(ast.parse("1 << 14", mode="eval").body) == 16384
    assert _constant_int(ast.parse("2**14", mode="eval").body) == 16384
    assert _constant_int(ast.parse("n * 1024", mode="eval").body) is None
    assert _compares_length_to(ast.parse("if len(raw) > 4: pass"), 4)
    assert _compares_length_to(ast.parse("ok = 1 <= len(raw) <= 2 + 2"), 4)
    assert not _compares_length_to(ast.parse("ok = 1 <= len(raw) <= most"), 4)


def test_oauth_callback_cap_is_not_inline() -> None:
    tree = _parse("llm_clients/auth/openai.py")
    inline = [
        node.lineno  # type: ignore[attr-defined]
        for node in ast.walk(tree)
        if isinstance(node, ast.expr) and _constant_int(node) == 16384
    ]
    assert not inline, f"OAuth callback cap written inline at lines {inline}"


def test_question_count_cap_is_not_inline() -> None:
    assert not _compares_length_to(_parse("nodes/human_input/questions.py"), 4)
