"""One unified `${...}` expression engine + the template scanner that hosts it.

The engine is three layers, all pool-agnostic (they take a `resolve` callable):
- `grammar` (`parse_expr`) — PARSE ONLY: one Lark grammar for every `${...}`
  construct (refs, arithmetic, boolean, coalesce/default/required, builtin calls).
- `expressions` (`eval_expr`, `infer_expr_type`, `expr_refs`) — evaluate a parsed
  tree against a `resolve`, conservatively infer its static type, and walk its
  reference leaves. Also the `when:`/`asserts:` condition surface
  (`evaluate_when`, `first_failing_assert`) and pool resolution
  (`resolve_reference`).
- `template` — the binding/prompt scanner: splits a value into text + `${...}`
  spans, then drives the engine per span (`eval_binding`, `eval_template`,
  `expr_refs_of`, `prompt_refs`, `render_template_record`, `scan_template`).

Flow invocation is NO LONGER a `${...}` construct — it is the compile-time
whole-value `call(...)` directive (in `compose.calls`), not part of this engine.

Knows about:   `typesys` (the pool and value types, in `expressions`).
Never imports: `nodes`, `compile`, or `runtime` (they import IT).
"""

from agent_composer.expr.expressions import (
    ExpressionError,
    RequiredError,
    condition_refs,
    eval_expr,
    evaluate_when,
    expr_refs,
    first_failing_assert,
    infer_expr_type,
    resolve_reference,
    rewrite_condition_refs,
    rewrite_expr_refs,
)
from agent_composer.expr.grammar import parse_expr
from agent_composer.expr.template import (
    binding_co_skips,
    eval_binding,
    eval_template,
    expr_refs_of,
    prompt_refs,
    render_template_record,
    rewrite_template_refs,
    scan_template,
)

__all__ = [
    "ExpressionError",
    "RequiredError",
    "binding_co_skips",
    "condition_refs",
    "eval_binding",
    "eval_expr",
    "eval_template",
    "evaluate_when",
    "expr_refs",
    "expr_refs_of",
    "first_failing_assert",
    "infer_expr_type",
    "parse_expr",
    "prompt_refs",
    "render_template_record",
    "resolve_reference",
    "rewrite_condition_refs",
    "rewrite_expr_refs",
    "rewrite_template_refs",
    "scan_template",
]
