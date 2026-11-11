/**
 * Shared producer helpers: the node ids a `${...}` reference reads from.
 *
 * The two functions here map a binding value to the producer node ids it depends on: the
 * primitive the edges phase (`edges/data.ts`) and the later boundary synthesis
 * (`build/boundary.ts`, via `synthesizeBoundaryGraph`) both build on. A ref head of
 * `<id>.output[.…]` or an AGENT's `<id>.new_messages[.…]` produces from node `<id>`; an
 * `input.X` ref produces from the synthesized `START_ID`; `system.X`/`item` produce nothing.
 *
 * Knows about: the compile model (`START_ID`), `expr` reference scanning, binding sources, and
 * the pool name of an AGENT's new messages (`typesys/pool.ts`).
 */

import { START_ID } from "../../compile/model.ts";
import { ExpressionError, exprRefsOf } from "../../expr/index.ts";
import { iterSourceStrings } from "../../nodes/contracts.ts";
import { NEW_MESSAGES } from "../../typesys/pool.ts";

/**
 * Return the producer node id of a `${...}` reference head.
 *
 * - `<id>.output[.…]`, and an AGENT's `<id>.new_messages[.…]`: produce from node `<id>`.
 * - `input.<key>`: produce from the synthesized `START_ID`, so an `${input.X}` reader gets a
 *   `START_ID -> reader` data edge.
 * - `system.<key>` is run-global and produces no edge; `item` is MAP-body-local. Both are reserved
 *   node ids, so `${item.output}` reads the item's `output` field.
 *
 * Legacy plural heads (`outputs`/`inputs`) are rejected at parse time. Exported for
 * `validate/skips.ts`.
 *
 * @param path A reference path, e.g. `a.output.field`.
 * @returns The producer node id, or `null` when the reference has none.
 */
export function refProducer(path: string): string | null {
  const parts = path.split(".");
  if (parts.length >= 2 && parts[0] === "input") return START_ID;
  if (parts.length >= 2 && (parts[1] === "output" || parts[1] === NEW_MESSAGES) && parts[0] !== "system" && parts[0] !== "item") {
    return parts[0]!;
  }
  return null;
}

/**
 * Return the producer node ids a binding or expression value reads, in source order.
 *
 * `value` may be a `${...}` binding string, a `when:`/`on:` expression template, a
 * `ValueTemplate` (each nested string, in tree order), one `Binding` around either, or a
 * non-string literal (no refs). Coalesce alternatives flatten to several producers; the caller
 * assigns them all the one sink input group. An `input.X` ref produces from `START_ID` (the
 * input-reader data edge); `system.X` produces nothing. A string whose refs don't parse is
 * skipped: the ref-wiring pass reports it, located.
 */
export function bindingProducers(value: unknown): string[] {
  const producers: string[] = [];
  for (const [, text] of iterSourceStrings(value)) {
    let refs: string[];
    try {
      refs = exprRefsOf(text);
    } catch (error) {
      if (error instanceof ExpressionError) continue;
      throw error;
    }
    for (const ref of refs) {
      const producer = refProducer(ref);
      if (producer !== null) producers.push(producer);
    }
  }
  return producers;
}
