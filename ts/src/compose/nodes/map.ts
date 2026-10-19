/**
 * kind=map: mapped function application (`List.map`), a `CallDescriptor` subclass.
 *
 * A `map` is `kind: map` with `over: ${list}`: it applies the callable to each element, with
 * `${item}` in element scope. `parallel:` is kept but inert: run-level workers control
 * concurrency. `MapDescriptor` subclasses `CallDescriptor` (in `call.ts`), adding only the map
 * field surface (`over:`, `parallel:`) through `allowed` and `required`; it inherits every field
 * and the `inputs` carrier. Its `parse` wraps the template `parse` and rejects an `over:` literal
 * that is not a list, at the value: a reference or a `call(...)` directive is typed later, from
 * the graph. Registered in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, compose source and helpers, `base.ts` and `call.ts`.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { CALL_HEAD } from "../helpers/directives.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { hasSpan, QUOTED_STYLES } from "../helpers/tags.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import type { ScalarStyle } from "../source/common.ts";
import type { SourceCursor } from "../source/index.ts";
import { type NodeDescriptor, rejectContainerSpans, siteKey, valueTypeError } from "./base.ts";
import { CallDescriptor } from "./call.ts";

const OVER_FIX = "write a list, or a reference such as `over: ${input.items}`";

/**
 * Reject an `over:` literal that is not a list.
 *
 * Text holding a `${...}` span is a reference and passes, as is an unquoted plain scalar starting
 * with `call(`: a directive the loader lifts into a node after this check. A quoted or block
 * scalar is text, as in any binding.
 *
 * @param nodeId The node id, for the message.
 * @param value The authored `over:` value.
 * @param style The value's YAML scalar style; `null` reads as plain.
 * @param range Where the error points.
 * @param diagnostics Factory for located load errors.
 * @throws LoadError when `value` is neither a list nor a reference.
 */
function rejectNonListOver(
  nodeId: string,
  value: unknown,
  style: ScalarStyle | null,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
): void {
  if (Array.isArray(value)) return;
  let problem: string;
  if (typeof value === "string") {
    if (hasSpan(value)) return;
    if (!QUOTED_STYLES.has(style) && CALL_HEAD.test(value.trim())) return;
    problem = `\`over:\` must be a list, but YAML read it as text; ${OVER_FIX}`;
  } else {
    problem = valueTypeError("over", "a list", value, OVER_FIX);
  }
  throw diagnostics.error(`node ${reprText(nodeId)} (kind=map): ${problem}`, { range });
}

/**
 * kind=map: `List.map`, applying a callable to each element of an `over: ${list}` source.
 *
 * A `CallDescriptor` with more authored fields: `over` (the iteration source) is required beside
 * `call`, and `parallel` is accepted as inert metadata. It declares no new fields (`over` and
 * `parallel` already live on `CallDescriptor`); it only widens `allowed` and `required` and sets
 * `kind = "map"`. It inherits `bindingFields` (`inputs`, `over`). Its `input:` names are the
 * child's parameters, not bare locals, so an input named `inf` is allowed.
 */
export class MapDescriptor extends CallDescriptor {
  static override readonly kind = "map";
  static override readonly allowed = this.allow("over", "parallel");
  static override readonly required = this.require("over");
  static override readonly inputsAreLocals = false;

  /**
   * Reject an `over:` value nested past the limit or holding a lone surrogate, or a list or
   * mapping holding `${...}`, then run the `call:` and base checks.
   *
   * @throws LoadError at `over:` (else at the node), or as the `call:` and base checks throw.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const over = fields["over"] ?? null;
    const site = ranges.get(siteKey("field", "over")) ?? ownRange;
    // Before the span walk, which recurses through the value.
    checkWrittenValue(over, `node ${reprText(nodeId)} field 'over'`, site, diagnostics);
    rejectContainerSpans(this.kind, nodeId, "over", over, site, diagnostics);
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }

  /**
   * Run the template `parse`, then reject an `over:` literal that is not a list.
   *
   * The check runs here rather than in `_validateFields` because whether `call(...)` text is a
   * directive depends on the value's scalar style, which only the cursor knows.
   *
   * @throws LoadError for any field the template rejects, or an `over:` that is neither a list
   *   nor a reference.
   */
  static override parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const desc = super.parse(nodeId, cursor) as MapDescriptor;
    rejectNonListOver(
      nodeId,
      desc.over,
      cursor.child("over").scalarStyle,
      desc.fieldRanges.get(siteKey("field", "over")) ?? desc.range,
      LoadDiagnostics.fromCursor(cursor),
    );
    return desc as InstanceType<T>;
  }
}
