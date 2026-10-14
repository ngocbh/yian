"""One unified `${...}` expression engine + the template scanner that hosts it.

The engine is three layers, all pool-agnostic (they take a `resolve` callable):
- `grammar` (`parse_expr`) — PARSE ONLY: one Lark grammar for every `${...}`
  construct (refs, arithmetic, boolean, coalesce/default/required, builtin calls).
- `expressions` (`eval_expr`, `infer_expr_type`, `expr_refs`) — evaluate a parsed
  tree against a `resolve`, conservatively infer its static type, and walk its
  reference leaves. Also the `when:`/`asserts:` condition surface
  (`parse_condition`, `evaluate_when`, `first_failing_assert`), the `isinstance` type
  names a tree uses (`isinstance_type_names`), its constructor names
  (`constructor_names`), and a call's tested expression
  (`isinstance_subject`), and pool resolution (`resolve_reference`).
- `template` — the binding/prompt scanner: splits a value into text + `${...}`
  spans, then drives the engine per span (`eval_binding`, `eval_template`,
  `expr_refs_of`, `prompt_refs`, `render_template_record`, `scan_template`), and
  checks every builtin call in those spans (`check_template_calls`). It also
  lists each reference a condition reads with the `:-` / `:?` / `|` escape around it
  and whether that escape co-skips as a binding (`condition_reads`, `ConditionRead`),
  and renames those references by occurrence (`rewrite_condition_reads`).

`lineage` owns ordinary feedback values, immutable expression routes, and
source-relative feedback transfer. Optional tracing in the same evaluator walk
retains selected value arms, container mappings, and opaque function boundaries.
Routes reference accepted source identities; the caller supplies recorded values
when transferring feedback. This leaf uses only the standard library and the
stdlib-only `yian.json_codec`, and stays independent of nodes, memory, and runtime.

Flow invocation is NO LONGER a `${...}` construct — it is the compile-time
whole-value `call(...)` directive (in `compose.desugar.inline_calls`), not part of this engine.

Knows about:   `typesys` (the pool and value types, in `expressions`) and the
               stdlib-only `yian.json_codec` (JSON text for ints of any size).
Never imports: `nodes`, `compile`, or `runtime` (they import IT).
"""

from yian.expr.expressions import (
    ConstructorTypeError,
    ExpressionError,
    RequiredError,
    condition_refs,
    constructor_names,
    eval_expr,
    evaluate_when,
    expr_refs,
    first_failing_assert,
    infer_expr_type,
    isinstance_subject,
    isinstance_type_names,
    parse_condition,
    resolve_reference,
    rewrite_condition_refs,
    rewrite_expr_refs,
)
from yian.expr.grammar import parse_expr
from yian.expr.lineage import FeedbackValue, has_feedback, validate_feedback
from yian.expr.template import (
    ConditionRead,
    binding_co_skips,
    check_template_calls,
    condition_reads,
    eval_binding,
    eval_template,
    expr_refs_of,
    prompt_refs,
    render_template_record,
    rewrite_condition_reads,
    rewrite_template_refs,
    scan_template,
)

__all__ = [
    "ConditionRead",
    "ConstructorTypeError",
    "ExpressionError",
    "FeedbackValue",
    "RequiredError",
    "binding_co_skips",
    "check_template_calls",
    "condition_reads",
    "condition_refs",
    "constructor_names",
    "eval_binding",
    "eval_expr",
    "eval_template",
    "evaluate_when",
    "expr_refs",
    "expr_refs_of",
    "first_failing_assert",
    "has_feedback",
    "infer_expr_type",
    "isinstance_subject",
    "isinstance_type_names",
    "parse_condition",
    "parse_expr",
    "prompt_refs",
    "render_template_record",
    "resolve_reference",
    "rewrite_condition_reads",
    "rewrite_condition_refs",
    "rewrite_expr_refs",
    "rewrite_template_refs",
    "scan_template",
    "validate_feedback",
]
