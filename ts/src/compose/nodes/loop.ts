/**
 * kind=loop: the `('a -> 'a) -> 'a -> 'a` driver's descriptor and its resolve-and-bake build.
 *
 * Owns the `loop` node's `LoopDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) and rejects a `call:` YAML read as
 * non-text and a `while:` or `until:` that isn't an expression string. `while` and `until` are
 * read into the `while_` and `until_` fields, as Python names them. Its `build` bakes the
 * runtime `LoopNode`: the driver and its counts, and the `'a -> 'a` field-name contract (a body
 * returns a subset of the carried record and each omitted field carries over, so the loop's
 * output contract merges the returned body fields with the seeds of the omitted ones). Registered
 * in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, the LOOP runtime node, node contracts, typesys, compose helpers,
 * `base.ts` and `resolver.ts`.
 * Never imports: compose/parse, the runtime engine or the CLI.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { isExactly, type OutputContract, PartialRecordContract } from "../../nodes/contracts.ts";
import { LoopNode } from "../../nodes/loop/node.ts";
import { valueRepr, writeAsText } from "../../typesys/values.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { combineOutputContracts, projectOutputContract } from "../helpers/output_contracts.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  rejectNonText,
  siteKey,
} from "./base.ts";
import { callableDisplay, childSignature, resolveChild } from "./resolver.ts";

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

  /**
   * Build the runtime `LoopNode` and its flow-owned wiring, resolving the body (`call:`) and
   * baking its compiled flow onto the node.
   *
   * The node carries the authored `input:` names (the carried record's seeds) as `params`; the
   * wiring holds their sources. Exactly one of `while:`, `until:` and `times:` drives the loop:
   * a predicate loop keeps its text and its optional `max:` limit; a `times:` loop has no
   * predicate, and `max:` beside it is redundant. A count must be a YAML int from 1 to
   * `Number.MAX_SAFE_INTEGER`; the node holds it as a number.
   *
   * The body's returned field names must be carried fields, at least one of them, and its input
   * names carried fields too. A body returning every carried field stamps its codomain as the
   * output contract and leaves `returnedFields` unset. A body returning a subset stamps
   * `returnedFields` (carried order) and a merged contract: each returned field projected from
   * the body's codomain, each omitted field typed from its seed (the seed's tag when it has one,
   * else through `ctx.bindingContractReader`, opaque without a reader or an unproven seed). The
   * returned fields' types are checked against the seeds' later, over the assembled flow. Every
   * problem found is reported together, at the node.
   *
   * @throws LoadError when the body can't be resolved or nests too deep, or the driver, a count
   *   or the field-name contract is illegal.
   */
  override build(ctx: BuildContext): BuildResult {
    const child = resolveChild(this.id, this.call, ctx.resolver, this.childLabel, {
      diagnostics: ctx.diagnostics,
      range: this.fieldRanges.get(siteKey("field", "call")) ?? this.range,
    });
    const sig = childSignature(child);
    const subject = `loop node ${reprText(this.id)}`;
    // The child's asserts ride its compiled boundary nodes, which every iteration's clone of the
    // body keeps.
    const node = new LoopNode(this.id, {
      flowId: this.call,
      flowVersion: null,
      child: child.compiled,
      childInputs: sig.inputs,
      title: this.nodeName,
    });
    node.params = this.sinkParams();
    const wiring = this.sinkWiring();
    node.outputContract = sig.output;

    const errors: string[] = [];
    // Only a value that isn't null names a driver, as Python's `is not None` checks.
    const present = (
      [
        ["while", this.while_],
        ["until", this.until_],
        ["times", this.times],
      ] as const
    )
      .filter(([, value]) => value !== null)
      .map(([key]) => key);
    if (present.length !== 1) {
      errors.push(`${subject}: exactly one of while:/until:/times: is required (got ${namesText(present)})`);
    } else if (present[0] === "times") {
      // A fixed count: exactly N runs, no predicate. `maxIters` doubles as N, so the runtime
      // guard needs no special case.
      const count = checkedCount(subject, "times", this.times, errors);
      node.predicateKind = "times";
      node.predicate = null;
      node.times = count;
      node.maxIters = count;
      if (this.max !== null) errors.push(`${subject}: max: is redundant with times: (times bounds the count)`);
    } else {
      const kind = present[0]!;
      node.predicateKind = kind;
      node.predicate = kind === "while" ? this.while_ : this.until_;
      node.maxIters = this.max === null ? null : checkedCount(subject, "max", this.max, errors);
    }

    // The field-name half of the `'a -> 'a` contract. The carried record's names are the seed
    // names. A record body returns its declared fields and a value body the fields of the record
    // it references; at least one of them, all carried. The body's input names are carried too.
    const carriedNames = Object.keys(this.inputs);
    const carried = new Set(carriedNames);
    const carriedText = namesText(carriedNames);
    const outNames = contractFieldNames(sig.output);
    if (outNames === null || outNames.size === 0) {
      errors.push(
        `${subject}: body ${callableDisplay(this.call, this.childLabel)} must return a record with at ` +
          `least one carried field (a subset of the carried record ${carriedText}); ` +
          "write `output: {field: ...}` or a whole reference to a node with a record output",
      );
    } else {
      const outside = [...outNames].filter((name) => !carried.has(name));
      if (outside.length > 0) {
        errors.push(
          `${subject}: body output field(s) ${namesText(outside)} are not carried record fields ` +
            `${carriedText} (a body returns a subset of the carried record)`,
        );
      } else if (outNames.size !== carried.size) {
        node.returnedFields = carriedNames.filter((name) => outNames.has(name));
        node.outputContract = combineOutputContracts(
          new Map(
            carriedNames.map((name) => [
              name,
              outNames.has(name) ? projectOutputContract(sig.output!, [name]) : this.seedContract(ctx, name),
            ]),
          ),
        );
      }
    }
    const extra = sig.inputs.map((input) => input.name).filter((name) => !carried.has(name));
    if (extra.length > 0) {
      errors.push(
        `${subject}: body input(s) ${namesText(extra)} are not carried record fields ${carriedText} ` +
          "(the body reads a subset of the carried record)",
      );
    }
    if (errors.length > 0) throw ctx.diagnostics.error(errors.join("\n  "), { range: this.range });
    return this._finish(node, wiring);
  }

  /**
   * Return the contract of omitted carried field `name`, from its seed binding: a tagged seed's
   * declared contract (the driver checks it when it binds the seed), else the one
   * `ctx.bindingContractReader` reads from the seed's value; `null` without a reader or when the
   * seed is unproven.
   */
  private seedContract(ctx: BuildContext, name: string): OutputContract | null {
    const declared = this.inputTypes.get(name);
    if (declared !== undefined) return declared;
    if (ctx.bindingContractReader === null) return null;
    return ctx.bindingContractReader(
      `${this.id}.input.${name}`,
      this.inputs[name],
      this.fieldRanges.get(siteKey("input", name)) ?? this.fieldRanges.get(siteKey("field", "inputs")) ?? null,
    );
  }
}

/**
 * Check the authored count `value` of `key` (`times` or `max`), appending one message to `errors`
 * when it isn't a YAML int (a `bigint`) from 1 to `Number.MAX_SAFE_INTEGER`.
 *
 * @param subject The message prefix naming the loop node.
 * @returns The count as a number; `null` when it was refused.
 */
function checkedCount(subject: string, key: string, value: unknown, errors: string[]): number | null {
  const field = `${subject}: \`${key}:\``;
  if (typeof value !== "bigint") {
    // A bool, a float, text or anything else: a count is a plain integer.
    errors.push(`${field} must be an integer (got ${valueRepr(value)})`);
  } else if (value < 1n) {
    // A count must permit at least one body run.
    errors.push(`${field} must be >= 1 (got ${value})`);
  } else if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    errors.push(`${field} must be at most ${Number.MAX_SAFE_INTEGER} (got ${value})`);
  } else {
    return Number(value);
  }
  return null;
}

/** Return `names` sorted by code unit, as a JSON list of text. */
function namesText(names: readonly string[]): string {
  return writeAsText(names.toSorted(compareCodeUnits));
}

/** Return the field names of a closed record contract, else `null`. */
function contractFieldNames(contract: OutputContract | null): Set<string> | null {
  if (isExactly(contract, PartialRecordContract)) return new Set(contract.names);
  if (contract === null || contract.type.fields === undefined) return null;
  return new Set(contract.type.fields.keys());
}
