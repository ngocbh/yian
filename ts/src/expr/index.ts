/**
 * One unified `${...}` expression engine and the template scanner that hosts it, as Python's
 * `yian.expr`.
 *
 * The engine is four layers, all pool-agnostic (none reads the pool directly):
 * - `grammar` (`parseExpr`): parse only, one grammar for every `${...}` construct (references,
 *   arithmetic, boolean, coalesce/default/required, builtin calls). An expression nests at most
 *   as deep as a value, `current().maxValueNesting` levels.
 * - `expressions` (`evalExpr`, `inferExprType`, `exprRefs`): evaluate a parsed tree against a
 *   `resolve`, conservatively infer its static type, and walk its reference leaves. Also the
 *   `when:`/`asserts:` condition surface (`parseCondition`, `evaluateWhen`,
 *   `firstFailingAssert`), the `isinstance` type names a tree uses (`isinstanceTypeNames`), its
 *   constructor names (`constructorNames`), a call's tested expression (`isinstanceSubject`), and
 *   pool resolution (`resolveReference`).
 * - `template`: the binding/prompt scanner. It splits a value into text and `${...}` spans, then
 *   drives the engine per span (`evalBinding`, `evalTemplate`, `exprRefsOf`, `promptRefs`,
 *   `renderTemplateRecord`, `scanTemplate`), and checks every builtin call in those spans
 *   (`checkTemplateCalls`). It also lists each reference a condition reads with the `:-` / `:?` /
 *   `|` escape around it and whether that escape co-skips as a binding (`conditionReads`,
 *   `ConditionRead`), and renames those references by occurrence (`rewriteConditionReads`).
 * - `lineage`: the routes a traced evaluation records (`evalExpr`, `evalTemplate` and
 *   `evalBinding` with `trace: true`) and the feedback transfer over them; this module re-exports
 *   the feedback value type and its checks (`FeedbackValue`, `hasFeedback`, `validateFeedback`).
 *
 * Flow invocation is not a `${...}` construct: it is compose's compile-time whole-value
 * `call(...)` directive, not part of this engine.
 *
 * Knows about: `typesys` (the pool and value types, in `expressions`; `writeAsText`, in
 * `builtins` and `template`; `writeAsJson`, `Type` and `ValueKind`, in `builtins`; `buildValue`,
 * in `lineage`), `typesys/json` (`logJson`, in `lineage` and `template`; `parseJson`, in
 * `lineage`), `defaults` (the value nesting bound, in `grammar`) and `diagnostics`.
 * Never imports: `nodes`, `compile`, or `runtime` (they import it).
 */

export {
  ConstructorTypeError,
  conditionRefs,
  constructorNames,
  ExpressionError,
  evalExpr,
  evaluateWhen,
  exprRefs,
  firstFailingAssert,
  inferExprType,
  isinstanceSubject,
  isinstanceTypeNames,
  parseCondition,
  RequiredError,
  resolveReference,
  rewriteConditionRefs,
  rewriteExprRefs,
} from "./expressions.ts";
export { parseExpr } from "./grammar.ts";
export type { FeedbackValue } from "./lineage.ts";
export { hasFeedback, validateFeedback } from "./lineage.ts";
export type { ConditionRead } from "./template.ts";
export {
  bindingCoSkips,
  checkTemplateCalls,
  conditionReads,
  evalBinding,
  evalTemplate,
  exprRefsOf,
  promptRefs,
  renderTemplateRecord,
  rewriteConditionReads,
  rewriteTemplateRefs,
  scanTemplate,
} from "./template.ts";
