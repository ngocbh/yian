"""CODE — run deterministic Python, either a `module:function` reference or inline source.

A `code` node carries one polymorphic `code:` field, classified once by shape:

- a `module:function` token → **reference mode**: import and call the function
  **in-process** (trusted, import-auditable, repo-local) — unchanged.
- real source (a `def`-less body) → **inline mode**: the author writes a **bare body**
  that reads the node's **`inputs`** dict and `return`s a value; the engine wraps it as
  `def main(inputs):`, compiles it, and calls `main(inputs)` **in-process** — the same
  one-dict calling convention as reference mode, so a body promotes to a `module:function`
  by copy-paste.
- a bare dotted token with no colon (a likely typo'd reference) → **rejected at load**
  with a "did you mean `module:function`?" hint, rather than silently run as inline.

Both modes call with the node's **bound typed input record** (a dict of the node's
declared inputs) — *not* the whole pool — and return the node's **one output value**:

    def my_step(inputs: dict) -> Any: ...        # reference target
    # inline: a bare body; `inputs` is in scope; `return` the value

The leaf code thus sees only its declared inputs (a pure function); the engine's
`eval_node` seam builds the record (the read boundary) and hands it in as `inputs`.

Inline source is **author-written and static** (it lives in the version-controlled flow
YAML), so there is no injection vector into the code itself, and running the operator's
own Python in-process is the same capability reference mode already has (trust model =
author is the operator). That safety argument holds **only** while flow YAML is
operator-authored: because inline source is `exec`'d in-process with the engine's
privileges, loading a flow from an untrusted or agent-generated source is remote code
execution. No opt-in load-time gate exists because there is no untrusted-source loader. The
current runtime executes inline code **in-process** with no isolation — a runaway body
cannot be killed, only interrupted with the run; there is no killable subprocess runtime or
security sandbox. A per-node `run()` **deep serialize-once check** makes a non-serializable
inline return fail *inside CODE*, before the dispatcher prepares its durable success record.
"""

import ast
import importlib
import re
import textwrap
from types import CodeType
from typing import Callable, Iterator, Optional, cast

from yian.diagnostics import OriginRef
from yian.nodes.base import Node, NodeKind, Output
from yian.nodes.contracts import _map_origin, _validate_optional_origin
from yian.typesys.values import (
    ANY_VALUE_ADAPTER,
    TypeCheckError,
    build_value,
    reject_nonfinite,
)

# A reference is a single dotted module path + one colon + a function identifier:
# `pkg.mod:func`, `mymod:myfunc`. Anchored (fullmatch): real source never matches.
_REFERENCE_RE = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")
# A bare (possibly dotted) identifier with NO colon — a likely typo'd reference, not
# real source. `pkg.mod.helper`, `myfunc`. The reject bucket.
_BARE_TOKEN_RE = re.compile(r"[A-Za-z_][\w.]*")


def classify_code_source(code: str) -> str:
    """
    Classify a `code:` field value by shape into an execution mode.

    Args:
        code (`str`):
            The raw `code:` field — either a `module:function` reference token or an
            inline Python body (a bare body reading `inputs`).

    Returns:
        `str`:
            `"reference"` for a `module:function` token, or `"inline"` for real source
            (anything carrying a newline, whitespace, or a non-identifier character).

    Raises:
        `ValueError`:
            When `code` is a bare (possibly dotted) identifier with no colon — neither a
            valid reference nor inline source (a likely typo'd reference). The message
            names both fixes; the loader turns it into a located `LoadError`.
    """
    # Reference shape wins: a single-line body that happens to fullmatch `token:token`
    # (e.g. `a.b:c`) is read as a reference, not inline source. Practically unreachable —
    # such a string carries no `return`, so it would fail the has-`return` gate as inline
    # anyway — but the precedence is fixed here so the classification is unambiguous.
    if _REFERENCE_RE.fullmatch(code):
        return "reference"
    if _BARE_TOKEN_RE.fullmatch(code):
        # No colon, so it can't be a reference; no source, so it can't be inline.
        suggestion = code.rsplit(".", 1)
        hint = f"{suggestion[0]}:{suggestion[1]}" if len(suggestion) == 2 else f"module:{code}"
        raise ValueError(
            f"`code:` value {code!r} is neither a `module:function` reference nor inline "
            f"source (a bare body reading `inputs`); did you mean {hint!r}?"
        )
    return "inline"


def _wrap_inline(code: str) -> str:
    """Wrap a bare inline body in the implicit `def main(inputs):` entrypoint.

    The author writes statements that read `inputs` and `return` a value; the engine
    supplies the `def` header and indents the body under it. The synthesized header is
    wrapper line 1, so authored line index zero is always wrapper line 2.
    """
    return "def main(inputs):\n" + textwrap.indent(code, "    ")


def select_inline_code_origin(
    lineno: object,
    *,
    code_origin: OriginRef | None,
    code_line_origins: tuple[OriginRef, ...],
    node_origin: OriginRef | None,
) -> OriginRef | None:
    """Select the finest source-neutral origin for one wrapped Python line.

    Args:
        lineno (`object`):
            Candidate 1-based line in the synthesized ``def main(inputs):``
            wrapper. Only an exact integer at line 2 or later can select an
            authored line.
        code_origin (`OriginRef | None`):
            Whole authored ``code:`` scalar fallback.
        code_line_origins (`tuple[OriginRef, ...]`):
            Exact authored line origins in constructed-value order.
        node_origin (`OriginRef | None`):
            Authored node fallback when no CODE-specific origin exists.

    Returns:
        `OriginRef | None`:
            Exact line origin, then whole-code origin, then node origin.

    Raises:
        TypeError:
            Raised when an origin carrier has the wrong shape.
    """
    _validate_optional_origin(code_origin, "CODE source origin")
    _validate_optional_origin(node_origin, "node origin")
    if type(code_line_origins) is not tuple:
        raise TypeError("CODE line origins must be a tuple")
    for origin in code_line_origins:
        if not isinstance(origin, OriginRef):
            raise TypeError("CODE line origins must contain only OriginRef values")

    line_index = lineno - 2 if type(lineno) is int else -1
    if 0 <= line_index < len(code_line_origins):
        return code_line_origins[line_index]
    return code_origin if code_origin is not None else node_origin


def _collect_code_objects(root: CodeType) -> tuple[CodeType, ...]:
    """Return one compiled wrapper's recursive code-object tree by identity.

    Code objects compare structurally, so equality-based containers cannot express
    ownership. The retained tuple keeps every object alive, and duplicate/cycle
    protection also uses ``is``.

    Args:
        root (`CodeType`): Compiled module wrapper.

    Returns:
        `tuple[CodeType, ...]`: Module, function, and nested code objects.
    """
    owned: list[CodeType] = []

    def visit(candidate: CodeType) -> None:
        if any(candidate is existing for existing in owned):
            return
        owned.append(candidate)
        for constant in candidate.co_consts:
            if isinstance(constant, CodeType):
                visit(constant)

    visit(root)
    return tuple(owned)


def _returns_in_scope(nodes) -> bool:
    """True if any statement in `nodes` is a `return`, NOT descending into nested scopes.

    A `return` inside a nested `def`/`lambda`/`class` belongs to that inner scope, not to
    `main`, so it must not satisfy the has-`return` gate — hence the manual recursion that
    stops at a new scope rather than `ast.walk` (which descends into everything).
    """
    for node in nodes:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue  # a new scope — its `return` is not main's
        if _returns_in_scope(ast.iter_child_nodes(node)):
            return True
    return False


class CodeNode(Node):
    """
    Run deterministic Python — a `module:function` reference or inline source, both in-process.

    The one `code:` field is classified by shape at construction with
    `classify_code_source`, and the
    mode is stored, so `run` stays pure dispatch. Reference mode imports and calls the
    function; inline mode wraps the bare body as `def main(inputs):`, compiles it once,
    and calls `main(inputs)`. Node purity is preserved — `run`
    returns `Output(value)` and never touches the pool. Inline runs are synchronous (no
    continuation state is created).

    Args:
        node_id (`str`):
            The node's unique id.
        code (`str`):
            The `code:` field — a `module:function` reference or inline source.
        source_file (`str`, *optional*, defaults to `"<inline-code>"`):
            The flow's file label, used only in inline mode to frame tracebacks at the flow
            filename.
        code_origin (`OriginRef | None`, *optional*, defaults to `None`):
            Opaque authored identity for the whole ``code:`` scalar. Diagnostic-only.
        code_line_origins (`tuple[OriginRef, ...]`, *optional*, defaults to `()`):
            Opaque identities for exact constructed literal lines in value order.
            Empty when line-level provenance is unavailable. Diagnostic-only.
        title (`str`, *optional*, defaults to `None`):
            Display title.

    Raises:
        ValueError: If `code` is neither a reference nor inline source (the reject bucket),
            or if an inline body defines no `return`.
        SyntaxError: If inline source fails to compile (the load-time syntax gate).
    """

    kind = NodeKind.CODE

    def __init__(
        self,
        node_id: str,
        *,
        code: str,
        source_file: str = "<inline-code>",
        code_origin: OriginRef | None = None,
        code_line_origins: tuple[OriginRef, ...] = (),
        title=None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.code = code
        self.code_origin = code_origin
        self.code_line_origins = code_line_origins
        self._validate_code_origins()
        self._mode = classify_code_source(code)  # "reference" | "inline"; raises on reject
        if self._mode == "reference":
            self.ref = code  # reference-mode callable target (kept for the in-process path)
            return
        # inline mode — wrapped + gated at LOAD, so a syntax slip / missing return fails
        # before any node runs. `main` sees `inputs`; the author writes a bare body.
        wrapped = _wrap_inline(code)
        # Compile the normal wrapper once for both the syntax gate and execution. Runtime
        # attribution uses retained code-object identity rather than filename or equality.
        self._code = compile(wrapped, source_file, "exec")
        self._owned_code_objects = _collect_code_objects(self._code)
        # Require an explicit `return` — a value-producing node that never returns is almost
        # certainly a mistake (it would yield None). Scoped to `main`, not nested defs.
        main_fn = ast.parse(wrapped).body[0]
        if not _returns_in_scope(main_fn.body):
            raise ValueError("inline `code:` has no `return` (the body must return a value)")

    def _validate_code_origins(self) -> None:
        """Require immutable source-neutral CODE diagnostic carriers."""
        _validate_optional_origin(self.code_origin, "CODE source origin")
        if type(self.code_line_origins) is not tuple:
            raise TypeError("CODE line origins must be a tuple")
        for origin in self.code_line_origins:
            if not isinstance(origin, OriginRef):
                raise TypeError("CODE line origins must contain only OriginRef values")

    def iter_local_origins(
        self,
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> Iterator[OriginRef]:
        """Yield common origins followed by whole-source and exact line origins."""
        self._validate_code_origins()
        yield from super().iter_local_origins(_seen_carriers=_seen_carriers)
        if self.code_origin is not None:
            yield self.code_origin
        yield from self.code_line_origins

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite common and CODE-specific origins on a copied node graph."""
        self._validate_code_origins()
        super().rewrite_local_origins(mapper, _seen_carriers=_seen_carriers)
        self.code_origin = _map_origin(self.code_origin, mapper)
        self.code_line_origins = tuple(
            cast(OriginRef, _map_origin(origin, mapper))
            for origin in self.code_line_origins
        )

    def exception_origin(self, exception: Exception) -> OriginRef | None:
        """Return the most precise authored CODE origin for this failure.

        Traceback filenames and code-object equality are deliberately ignored:
        separately compiled code can share both. Reference mode attributes the
        invocation to its whole ``code:`` scalar. Inline failures outside authored
        code, including the serialize-once check, use that same scalar fallback.

        Args:
            exception (`Exception`): Original node-body exception.

        Returns:
            `OriginRef | None`: Exact line, whole CODE scalar, node, or ``None``.
        """
        if self._mode == "reference":
            return (
                self.code_origin
                if self.code_origin is not None
                else super().exception_origin(exception)
            )

        owned_lineno: int | None = None
        traceback = exception.__traceback__
        while traceback is not None:
            frame_code = traceback.tb_frame.f_code
            if any(
                frame_code is owned_code
                for owned_code in self._owned_code_objects
            ):
                owned_lineno = traceback.tb_lineno
            traceback = traceback.tb_next

        return select_inline_code_origin(
            owned_lineno,
            code_origin=self.code_origin,
            code_line_origins=self.code_line_origins,
            node_origin=self.origin,
        )

    def forward(self, inputs: dict) -> Output:
        if self._mode == "reference":
            module_name, _, func_name = self.ref.partition(":")
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            result = func(inputs)  # strict: the user fn sees only its bound record
            return Output(value=result)  # the one value (object/list/scalar), stored whole
        # inline: run in-process. `main(inputs)` gets the bound record (one dict, like
        # reference mode). A raise propagates and eval_node funnels it into a clean NodeFailed.
        # A fresh namespace per run is deliberate, not a missed cache: re-`exec` rebuilds the
        # `main` function object (cheap — it does NOT re-run the body) into a clean module
        # scope, so no module-level state carries between invocations. That per-run isolation
        # keeps each run independent, which is what the engine's determinism wants.
        ns: dict = {}
        exec(self._code, ns)  # binds `main`
        result = ns["main"](inputs)
        self._assert_serializable(result)
        return Output(value=result)

    def _assert_serializable(self, value) -> None:
        """Fail *at the node* if an inline return can't survive the typed, serializable pool.

        The generic write-boundary validates the top-level shape only; a nested
        non-serializable value (e.g. a bare object inside a returned dict) would otherwise
        reach canonical success-record encoding before it fails. Running the typed-value
        serializer once here surfaces the error inside this CODE invocation. `None` is left
        to the declared-`output:` check.
        """
        if value is None:
            return
        try:
            reject_nonfinite(value, "output")
        except TypeCheckError:
            # The output-contract boundary rejects it with its path, the same failure NaN gets.
            return
        try:
            typed = build_value(value)  # top-level: raises if the value can't be wrapped
            ANY_VALUE_ADAPTER.dump_json(typed)  # nested: raises on a non-serializable member
        except TypeCheckError as exc:
            raise TypeError(f"inline `code:` returned a non-serializable value: {exc}") from exc
        except Exception as exc:  # a pool serialization failure (nested unknown type)
            raise TypeError(f"inline `code:` returned a non-serializable value: {exc}") from exc
