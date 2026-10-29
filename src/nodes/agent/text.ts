/**
 * The interactions learning reads, for the AGENT loop and its structured output.
 *
 * Knows about:   memory contracts.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { VisibleInteraction } from "../../memory/contracts.ts";

/**
 * Append `make()`'s interaction to `interactions`, unless its carrier refuses the fields (a
 * `TypeError` or `RangeError`): what learning reads is observational and never fails the pass.
 */
export function pushVisible(interactions: VisibleInteraction[], make: () => VisibleInteraction): void {
  try {
    interactions.push(make());
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
  }
}
