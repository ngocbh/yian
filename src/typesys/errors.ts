/**
 * The bad-value errors yian throws itself, so a host or the CLI can tell a value yian refused (a
 * host's or a flow's mistake) from a built-in `TypeError` or `RangeError` a bug raised. Each is
 * named as the built-in it extends, in messages, stacks and `errorTypeName`.
 *
 * Imports nothing, so every layer may import it.
 */

/** A `TypeError` yian throws for a value of the wrong kind. */
export class YianTypeError extends TypeError {}

/** A `RangeError` yian throws for a value outside what it accepts. */
export class YianRangeError extends RangeError {}

// The constructor's name is what `errorTypeName` and run records report, so each reads as the
// built-in it extends, as the error did before yian had its own class.
Object.defineProperty(YianTypeError, "name", { value: "TypeError" });
Object.defineProperty(YianRangeError, "name", { value: "RangeError" });
