/** Shared contracts for source documents and their occurrence cursors. */

import type { SourceRange } from "../../diagnostics.ts";

/** A YAML source can't be composed or constructed safely. `range` locates the problem when known, else `null`. */
export class SourceSyntaxError extends RangeError {
  override name = "SourceSyntaxError";
  readonly range: SourceRange | null;

  constructor(message: string, range: SourceRange | null = null) {
    super(message);
    this.range = range;
  }
}
