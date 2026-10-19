/**
 * kind=wait: the blocking absolute-deadline node's descriptor.
 *
 * Owns the `wait` node's `WaitDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`), rejects an `until:` literal that
 * is not a date or datetime at the value (a reference is typed later, from the graph), and builds
 * the runtime `WaitNode` with the reserved `until` wiring. Registered in `KINDS` and
 * `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, compose helpers, nodes (wait) and typesys.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { WaitNode } from "../../nodes/wait/node.ts";
import { parseDateOrDatetime } from "../../typesys/temporal.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { hasSpan } from "../helpers/tags.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  rejectContainerSpans,
  siteKey,
  valueTypeError,
} from "./base.ts";

const UNTIL_FIX = "write one such as `until: 2026-05-21`, or a reference such as `until: ${input.as_of}`";

/**
 * Reject an `until:` literal that is not a date or datetime.
 *
 * Text passes when it parses as a date or datetime, or holds a `${...}` span; TS YAML reads an
 * unquoted date or timestamp as text, so it is checked here too. A deadline whose UTC conversion
 * overflows at the calendar's ends (such as `'0001-01-01T00:00:00+05:00'`) parses here and is
 * left to fail at run time.
 *
 * @throws LoadError at `range` when `value` is neither date text nor a reference.
 */
function rejectNonDateUntil(
  nodeId: string,
  value: unknown,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
): void {
  let problem: string;
  if (typeof value === "string") {
    if (hasSpan(value)) return;
    try {
      parseDateOrDatetime(value);
      return;
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
    }
    problem =
      "`until:` must be a date or datetime, but YAML read it as text that is not a date or " +
      `datetime; ${UNTIL_FIX}`;
  } else {
    problem = valueTypeError("until", "a date or datetime", value, UNTIL_FIX);
  }
  throw diagnostics.error(`node ${reprText(nodeId)} (kind=wait): ${problem}`, { range });
}

/** The fields `WaitDescriptor`'s constructor takes: the common ones and the WAIT's own. */
export interface WaitDescriptorFields extends NodeDescriptorFields {
  until: string;
}

/**
 * kind=wait: block until an absolute date or datetime; produces `null`.
 *
 * `allowed` adds only `until` to the commons, and requires it. A WAIT's `until` source rides the
 * reserved wiring key, not the kind-blind binding walk, so it has no binding field; it never
 * populates `inputs` or `outputs`.
 */
export class WaitDescriptor extends NodeDescriptor {
  static override readonly kind = "wait";
  static override readonly allowed = this.allow("until");
  static override readonly required = this.require("until");
  static override readonly bindingFields: readonly string[] = [];

  /** The deadline: date or datetime text, or a `${...}` reference, as written. */
  readonly until: string;

  /**
   * Create a descriptor from `fields`.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: WaitDescriptorFields) {
    super(fields);
    this.until = fields.until;
  }

  /**
   * Reject an `until:` value that is not a date, a datetime, or a reference, then run the base
   * checks. A value nested past the limit or holding a lone surrogate, and a list or mapping
   * holding `${...}`, get their own messages first.
   *
   * @throws LoadError at the field for a bad value.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const until = fields["until"] ?? null;
    const site = ranges.get(siteKey("field", "until")) ?? ownRange;
    // Before the span walk, which recurses through the value.
    checkWrittenValue(until, `node ${reprText(nodeId)} field 'until'`, site, diagnostics);
    rejectContainerSpans(this.kind, nodeId, "until", until, site, diagnostics);
    rejectNonDateUntil(nodeId, until, site, diagnostics);
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }

  /**
   * Build the runtime `WaitNode` and its flow-owned wiring.
   *
   * The required `until` source rides the reserved wiring key, never a node-side param;
   * `_finish` stamps the descriptor origin.
   */
  override build(_ctx: BuildContext): BuildResult {
    const node = new WaitNode(this.id, { title: this.nodeName });
    node.params = [];
    return this._finish(node, new Map([["until", this.until]]));
  }
}
