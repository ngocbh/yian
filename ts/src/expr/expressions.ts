/**
 * `${...}` evaluation plus conservative compile-time type/reference analysis.
 *
 * For now this holds only the expression error family and the `isinstance` special form's
 * name, which the grammar shares; the evaluator, type inference and the reference walkers
 * land here later.
 */

/** A `when:` expression could not be parsed or evaluated. */
export class ExpressionError extends RangeError {
  override name = "ExpressionError";
}

/** A `${ref:?message}` whose ref was unbound (the binder maps it to BindingError). */
export class RequiredError extends ExpressionError {
  override name = "RequiredError";
}

/** A `${T(...)}` constructor's value does not fit `T` (the binder's `binding.type`). */
export class ConstructorTypeError extends ExpressionError {
  override name = "ConstructorTypeError";
}

/**
 * The type-test special form `isinstance(x, T)`: its second argument is a type name
 * looked up in the node's type table, never a reference, so the evaluator, the reference
 * walkers and rewriters, and type inference treat only its first argument as an expression.
 */
export const ISINSTANCE = "isinstance";
