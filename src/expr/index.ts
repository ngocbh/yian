/**
 * One unified `${...}` expression engine and the template scanner that hosts it.
 *
 * The engine is four layers, all pool-agnostic (none reads the pool directly):
 * - `grammar` (`parseExpr`): parse only, one grammar for every `${...}` construct (references,
 *   arithmetic, boolean, coalesce/default/required, builtin calls). An expression nests at most
 *   as deep as a value, `current().maxValueNesting` levels.
 * - `expressions` (`evalExpr`, `inferExprType`, `exprRefs`): evaluate a parsed tree against a
 *   `resolve`, conservatively infer its static type, and walk its reference leaves. Also the
 *   `when:`/`asserts:` condition surface (`parseCondition`, `evaluateWhen`), the `isinstance`
 *   type names a tree uses (`isinstanceTypeNames`), its constructor names (`constructorNames`), a
 *   call's tested expression (`isinstanceSubject`), a reference's head and `.`/`?.` steps
 *   (`splitReference`), and pool resolution (`resolveReference`).
 * - `template`: the binding/prompt scanner. It splits a value into text and `${...}` spans, then
 *   drives the engine per span (`evalBinding`, `evalTemplate`, `exprRefsOf`, `promptRefs`,
 *   `renderTemplateRecord`, `scanTemplate`), and checks every builtin call in those spans
 *   (`checkTemplateCalls`). It also lists each reference a condition reads with the `:-` / `:?` /
 *   `|` escape around it and whether that escape co-skips as a binding (`conditionReads`,
 *   `ConditionRead`), and renames those references by occurrence (`rewriteConditionReads`).
 *
 * Flow invocation is not a `${...}` construct: it is compose's compile-time whole-value
 * `call(...)` directive, not part of this engine.
 *
 * Knows about: `typesys` (the pool and value types, in `expressions`; `writeAsText`, in
 * `builtins` and `template`; `writeAsJson`, `Type` and `ValueKind`, in `builtins`), `defaults`
 * (the value nesting bound, in `grammar`; the comprehension, int and indent caps, in `expressions`
 * and `builtins`) and `diagnostics`.
 */

export {
  ConstructorTypeError,
  conditionRefs,
  constructorNames,
  declaredAt,
  ExpressionError,
  evalExpr,
  evaluateWhen,
  exprRefs,
  inferExprType,
  isinstanceSubject,
  isinstanceTypeNames,
  localDeclared,
  MISSING_ROOT,
  namesEnd,
  optionalAfterRoots,
  optionalInsideRoot,
  parseCondition,
  RequiredError,
  referencePath,
  referenceText,
  resolveReference,
  rewriteConditionRefs,
  rewriteExprRefs,
  splitReference,
  stepsText,
} from "./expressions.ts";
export type { DeclaredTypes, RefRename, RefRenamed, RefStep, RootResolver } from "./expressions.ts";
export { parseExpr } from "./grammar.ts";
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
  Verbatim,
} from "./template.ts";
