/**
 * Turn `yian run`'s memory flags into one host-built memory registry. Not ported to TypeScript
 * yet: the host memory registry it builds hasn't landed.
 */

import type { ComposedFlow } from "../compose/build/pipeline.ts";

/** A memory flag value is invalid; `paramHint` names the flag. */
export class MemoryOptionError extends RangeError {
  override name = "MemoryOptionError";
  readonly paramHint: string;

  constructor(message: string, { paramHint }: { paramHint: string }) {
    super(message);
    this.paramHint = paramHint;
  }
}

/**
 * Build every memory `flow` declares from `--memory-root` and the repeated `--memory-storage
 * KEY=PATH` values. Not ported to TypeScript yet.
 *
 * @throws {Error} Always, until the host memory registry is ported.
 */
export function buildCliMemories(
  flow: ComposedFlow,
  options: { root: string | undefined; storage: readonly string[] },
): never {
  throw new Error("buildCliMemories is not ported to TypeScript yet");
}
