/**
 * kind=loop: the `('a -> 'a) -> 'a -> 'a` driver's descriptor.
 *
 * Owns the `loop` node's `LoopDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) and rejects a `call:` YAML read as
 * non-text and a `while:` or `until:` that isn't an expression string. `while` and `until` are
 * read into the `while_` and `until_` fields, as Python names them. Registered in `KINDS` and
 * `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, typesys, compose helpers and `base.ts`.
 * Never imports: compose/parse, the runtime engine or the CLI.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { valueRepr } from "../../typesys/values.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { NodeDescriptor, type NodeDescriptorFields, rejectNonText, siteKey } from "./base.ts";

// Each predicate key with the example its non-string error shows.
const PREDICATE_EXAMPLES: readonly (readonly [string, string])[] = [
  ["while", "while: not done"],
  ["until", "until: done"],
];

/** The fields `LoopDescriptor`'s constructor takes: the common ones and the LOOP's own. */
export interface LoopDescriptorFields extends NodeDescriptorFields {
  call: string;
  childLabel?: string | null;
  while_?: string | null;
  until_?: string | null;
  times?: unknown;
  max?: unknown;
  asserts?: readonly string[] | null;
}

/**
 * kind=loop: a `('a -> 'a) -> 'a -> 'a` driver over a carried record.
 *
 * `call` names the body child flow, resolved as a `call`'s is. The authored `input:` mapping
 * seeds the carried record at turn 0. The body returns a subset of the carried record; each
 * omitted field keeps its previous value. Exactly one of `while_`, `until_` and `times` drives
 * the loop; `max` optionally limits a `while:` or `until:` loop's iterations. `times` and `max`
 * hold the YAML value as read (an int is a `bigint`; `null` means unset); the build checks them.
 *
 * `allowed` adds `call`, `inputs`, `while`, `until`, `times`, `max` and `asserts` to the
 * commons, and `call` is required. `bindingFields` is `inputs`: the carried-record seed bindings
 * take part in the inline-call and case-output rewrites. `keywordRemap` reads `while:` and
 * `until:` into `while_` and `until_`; their ranges stay keyed by the YAML key.
 *
 * `childLabel` is compiler-owned diagnostic text for an anonymous inline child; it is not in
 * `allowed`, so never authored.
 */
export class LoopDescriptor extends NodeDescriptor {
  static override readonly kind = "loop";
  static override readonly allowed = this.allow("call", "inputs", "while", "until", "times", "max", "asserts");
  static override readonly required = this.require("call");
  static override readonly bindingFields: readonly string[] = ["inputs"];
  static override readonly keywordRemap: ReadonlyMap<string, string> = new Map([
    ["while", "while_"],
    ["until", "until_"],
  ]);

  /** The body child flow's name: a `defs:` entry or an external flow id. */
  readonly call: string;
  /** The diagnostic label of an anonymous inline child; `null` on an authored node. */
  readonly childLabel: string | null;
  /** The `while:` predicate text; `null` when absent. */
  readonly while_: string | null;
  /** The `until:` predicate text; `null` when absent. */
  readonly until_: string | null;
  /** The authored `times:` as YAML read it; `null` when absent. */
  readonly times: unknown;
  /** The authored `max:` as YAML read it; `null` when absent. */
  readonly max: unknown;
  /** The node asserts. */
  readonly asserts: readonly string[];

  /**
   * Create a descriptor from `fields`.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: LoopDescriptorFields) {
    super(fields);
    this.call = fields.call;
    this.childLabel = fields.childLabel ?? null;
    this.while_ = fields.while_ ?? null;
    this.until_ = fields.until_ ?? null;
    this.times = fields.times ?? null;
    this.max = fields.max ?? null;
    this.asserts = fields.asserts ?? [];
  }

  /**
   * Reject a `call:` YAML read as non-text (including `null`), a `while:` or `until:` that is
   * neither `null` nor text, and predicate text holding a lone surrogate, then run the base checks.
   *
   * A YAML bool, number or list predicate would otherwise crash the predicate check, or, when
   * falsy (`while: false`), skip it and fail only at run time.
   *
   * @throws LoadError at the offending field (else at the node).
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    rejectNonText(
      this.kind,
      nodeId,
      "call",
      fields["call"] ?? null,
      ranges.get(siteKey("field", "call")) ?? ownRange,
      diagnostics,
      { required: true },
    );
    for (const [key, example] of PREDICATE_EXAMPLES) {
      const value = fields[this.keywordRemap.get(key)!] ?? null;
      if (value !== null && typeof value !== "string") {
        throw diagnostics.error(
          `loop node ${reprText(nodeId)}: \`${key}:\` must be an expression string over the ` +
            `carried record, such as \`${example}\` (got ${valueRepr(value)})`,
          { range: ranges.get(siteKey("field", key)) ?? ownRange },
        );
      }
    }
    // After both type checks, so a flow Python refuses gets Python's error.
    for (const [key] of PREDICATE_EXAMPLES) {
      const value = fields[this.keywordRemap.get(key)!];
      if (typeof value === "string") {
        const range = ranges.get(siteKey("field", key)) ?? ownRange;
        checkWrittenValue(value, `node ${reprText(nodeId)} field ${reprText(key)}`, range, diagnostics);
      }
    }
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }
}
