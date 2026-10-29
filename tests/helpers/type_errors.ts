/** Catching the `TypeCheckError` a call throws, for the type-system tests. */

import { TypeCheckError } from "../../src/typesys/values.ts";

/** Return the `TypeCheckError` `run` throws; rethrow any other error, and throw if it returns. */
export function typeCheckError(run: () => unknown): TypeCheckError {
  try {
    run();
  } catch (error) {
    if (error instanceof TypeCheckError) return error;
    throw error;
  }
  throw new Error("expected a TypeCheckError");
}
