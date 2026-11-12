/** The classes yian throws for the built-in errors a test names. */

import { YianRangeError, YianTypeError } from "../../src/typesys/errors.ts";

/**
 * Return the class yian throws where a test expects the built-in `cls`: its own `YianTypeError` for
 * `TypeError` and `YianRangeError` for `RangeError`, else `cls` itself.
 */
export function thrownClass<T extends abstract new (...args: never[]) => unknown>(cls: T): T {
  const given: unknown = cls;
  if (given === TypeError) return YianTypeError as unknown as T;
  if (given === RangeError) return YianRangeError as unknown as T;
  return cls;
}
