/**
 * kind=loop: the `('a -> 'a) -> 'a -> 'a` driver's descriptor and its resolve-and-bake build.
 *
 * Owns the `loop` node's `LoopDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) and rejects a `call:` YAML read as
 * non-text and a `while:` or `until:` that isn't an expression string. `while` and `until` are
 * read into the `while_` and `until_` fields, as Python names them. Its `build` bakes the
 * runtime `LoopNode`: the driver and its counts, the `'a -> 'a` field-name contract (a body
 * returns a subset of the carried record and each omitted field carries over, so the loop's
 * output contract merges the returned body fields with the seeds of the omitted ones), the
 * predicate's names, checks and field reads against the carried record, and whether the loop can
 * end. Registered in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`. It also holds the
 * loop-entry helpers the load checks use: `whileNonNullPaths` finds the carried paths a `while:`
 * null check proves, `nullableAt` relaxes a type along such a path, `loopEntryInputs` gives the
 * body's inputs read that way, and `carriedFieldTypes` gives the carried record's proven field
 * types.
 *
 * Knows about: diagnostics, the LOOP runtime node, node contracts, typesys, compose helpers,
 * `base.ts` and `resolver.ts`.
 * Never imports: compose/parse, the runtime engine or the CLI.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import { ExpressionError, exprRefs, parseCondition } from "../../expr/index.ts";
import {
  comparePaths,
  isExactly,
  type OutputContract,
  outputContractFieldError,
  Parameter,
  PartialRecordContract,
  ValueContract,
} from "../../nodes/contracts.ts";
import { LoopNode } from "../../nodes/loop/node.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { Type, ValueKind, valueRepr, writeAsText } from "../../typesys/values.ts";
import { AllOf, type Guard, Holds, NotNullCheck, proves, sameGuard } from "../helpers/condition_guards.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import {
  type BaseTypeOf,
  checkedRefs,
  guardedReadTypes,
  impossibleCheckMessage,
  impossibleChecks,
  narrowedReadError,
  unresolvedType,
  widerType,
} from "../helpers/narrowing_facts.ts";
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
   * returned fields' types are checked against the seeds' later, over the assembled flow.
   *
   * A `while:` or `until:` predicate must parse, and every name it reads must be a carried field.
   * With the names valid, its checks and field reads are checked against the carried record's
   * contract: a check that can never be true is refused, as is a field read the carried record (or
   * the type the checks before the read narrow it to) can't have. A `while:` is first evaluated on
   * the seeds, so its checks take a path's type from the seed too: the wider of the seed's and the
   * carried type, else none. `ctx.registry` resolves the type names of the predicate's `isinstance`
   * checks. Without `max:`, a predicate loop whose body returns a known subset of the carried
   * record must return at least one field the predicate reads, or the loop can never end. Every
   * problem found is reported together, at the node; a predicate that doesn't parse, found alone,
   * is reported at its field.
   *
   * @throws LoadError when the body can't be resolved or nests too deep, the driver, a count, the
   *   field-name contract, the predicate or one of its checks or field reads is illegal, or the
   *   loop can never end.
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
    // The predicate is evaluated against the carried record, whose names are the only ones it can
    // resolve: an unknown name reads as null, so `while:` would run zero times and `until:`
    // without `max:` would spin forever. A `times:` loop has no predicate.
    const predicate = this.while_ ?? this.until_;
    if (predicate !== null) {
      const key = this.while_ !== null ? "while" : "until";
      // Parsing the whole value covers both spellings (bare `score >= 0.8`, whole `${score >= 0.8}`).
      let tree: ExprNode | null = null;
      try {
        tree = parseCondition(predicate);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        const message = `${subject}: \`${key}:\` is not a valid boolean expression: ${error.message}`;
        if (errors.length === 0) {
          // Alone, the error belongs to the field; beside other errors it joins them at the node.
          throw ctx.diagnostics.error(message, {
            range: this.fieldRanges.get(siteKey("field", key)) ?? this.range,
            cause: error,
          });
        }
        errors.push(message);
      }
      const heads = new Set((tree === null ? [] : exprRefs(tree)).map((ref) => ref.split(".", 1)[0]!));
      const unknown = [...heads].filter((head) => !carried.has(head));
      if (unknown.length > 0) {
        errors.push(
          `${subject}: \`${key}:\` references unknown name(s) ${namesText(unknown)} — the predicate reads ` +
            `the carried record ${carriedText}`,
        );
      }
      if (tree !== null && errors.length === 0 && node.outputContract !== null) {
        // With the field names valid, the output contract is the carried record's: the body's for
        // a total loop, merged with the omitted fields' seeds otherwise.
        const contract = node.outputContract;
        let baseTypeOf = carriedBaseType(contract);
        if (key === "while") {
          // `while:` is first evaluated on the seeds, which the type contract doesn't always force
          // to fit the body (a literal, an untyped producer), so its checks must hold on a seed's
          // type too.
          const seeds = new Map(
            [...heads].toSorted(compareCodeUnits).map((head) => [head, this.seedContract(ctx, head)] as const),
          );
          baseTypeOf = seededBaseType(baseTypeOf, seeds);
        }
        for (const message of predicateErrors(contract, tree, baseTypeOf, ctx.registry)) {
          errors.push(`${subject}: \`${key}:\` ${message}`);
        }
      }
      // A body that returns none of the fields the predicate reads can never change it, so without
      // `max:` the loop never ends. `returnedFields` is set only when the body returns a known
      // strict subset of the carried record. A driver, parse or unknown-name error already
      // explains the loop, so this one doesn't join it.
      const returned = node.returnedFields;
      if (
        present.length === 1 &&
        this.max === null &&
        heads.size > 0 &&
        unknown.length === 0 &&
        returned !== null &&
        [...heads].every((head) => !returned.includes(head))
      ) {
        const stuck = [...heads].toSorted(compareCodeUnits);
        const reads = stuck.map((name) => `\`${name}\``).join(", ");
        const fix = stuck.map((name) => `\`${name}\``).join(" or ");
        errors.push(
          `${subject}: this loop can never end: \`${key}: ${predicate}\` reads ${reads}, which the body ` +
            `never updates; return ${fix} from the body, or add \`max:\``,
        );
      }
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
function checkedCount(subject: string, key: "times" | "max", value: unknown, errors: string[]): number | null {
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

/** Return the declared type of a carried-record path, or `null` when it is unproven. */
function carriedBaseType(contract: OutputContract): BaseTypeOf {
  return (base) => {
    const projected = projectOutputContract(contract, base.split("."));
    return isExactly(projected, ValueContract) ? projected.type : null;
  };
}

/**
 * Return the type a carried-record path has on every evaluation of a `while:`, the seed's
 * included: the carried type when the seed's type fits it, the seed's when the carried type fits
 * the seed's, and otherwise, or when either is unproven, `null`.
 *
 * @param carried The carried record's declared types, from `carriedBaseType`.
 * @param seeds Each carried field the predicate reads, to its seed's contract (`null` unproven).
 */
function seededBaseType(carried: BaseTypeOf, seeds: ReadonlyMap<string, OutputContract | null>): BaseTypeOf {
  return (base) => {
    const carriedType = carried(base);
    const [head, ...rest] = base.split(".");
    const seed = seeds.get(head!) ?? null;
    if (!(carriedType instanceof Type) || seed === null) return null;
    const projected = projectOutputContract(seed, rest);
    if (!isExactly(projected, ValueContract)) return null;
    return widerType(carriedType, projected.type);
  };
}

/**
 * Return the predicate's impossible checks and field-read errors, each message once.
 *
 * Checks and narrowed reads take each path's type from `baseTypeOf`; a read nothing narrows is
 * walked on `contract`, the carried record. A read is checked under each distinct guard it is
 * read with, since the same path can be read both narrowed and not.
 */
function predicateErrors(
  contract: OutputContract,
  tree: ExprNode,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
): string[] {
  const messages = impossibleChecks(tree, baseTypeOf, registry).map(impossibleCheckMessage);
  const checked: [string, Guard][] = [];
  for (const [ref, guard] of checkedRefs(tree, baseTypeOf, registry)) {
    // A read after a reported check is reached by no value.
    if (guard === null) continue;
    if (checked.some(([seen, seenGuard]) => seen === ref && sameGuard(seenGuard, guard))) continue;
    checked.push([ref, guard]);
    const message = predicateFieldError(contract, ref, guard, baseTypeOf, registry);
    if (message !== null) messages.push(message);
  }
  return [...new Set(messages)];
}

/**
 * Return why predicate reference `ref` reads a field the carried record can't have, else `null`.
 *
 * `and`, `or` and `not` short-circuit, so the checks before a read (`isinstance` and union tag
 * comparisons) narrow the values it can see (`guardedReadTypes`): the read is fine when any
 * narrowed type allows it, and otherwise the first one's error is reported, naming the narrowed
 * type when it lacks the field. With nothing narrowed the read is walked on the carried record. A
 * read guarded by a check whose type name doesn't resolve is not checked, so the type-table check
 * reports the unknown name.
 *
 * @param guard The read's guard, from `checkedRefs`.
 */
function predicateFieldError(
  contract: OutputContract,
  ref: string,
  guard: Guard,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
): string | null {
  // The type-table pass reports the unknown name; an error here would fail the build first and
  // hide it behind a hint to add the check the author wrote.
  if (unresolvedType(guard, registry)) return null;
  const candidates = guardedReadTypes(ref, guard, baseTypeOf, registry);
  if (candidates.length === 0) {
    return outputContractFieldError(contract, ref.split("."), ref, { isinstanceHint: true });
  }
  let first: string | null = null;
  for (const candidate of candidates) {
    const [typ, fields] = candidate;
    const error = outputContractFieldError(new ValueContract(typ), fields, ref, { isinstanceHint: true });
    if (error === null) return null;
    first ??= narrowedReadError(ref, candidate, error, registry);
  }
  return first;
}

/**
 * Return the carried paths a loop's condition proves not null whenever the body starts.
 *
 * Only `while:` is checked before each iteration, the first included; `until:` is checked after
 * the body and `times:` has no condition, so they prove nothing. A path comes from a
 * `<path> != null` check the condition proves when true: each part of an `and` adds its paths,
 * and an `or` keeps only the paths every part proves.
 *
 * @param kind The loop's driver: `"while"`, `"until"` or `"times"`.
 * @param predicate The condition text, bare or whole-span; `null` for `times`.
 * @returns Each proven path once, as field names from the carried record, sorted with
 *   `comparePaths`; `[]` for a condition that doesn't parse (the build reports it).
 *
 * @example
 * `whileNonNullPaths("while", "n < 3 and page.next != null")` is `[["page", "next"]]`.
 */
export function whileNonNullPaths(kind: string | null, predicate: string | null): readonly (readonly string[])[] {
  if (kind !== "while" || predicate === null) return [];
  let tree: ExprNode;
  try {
    tree = parseCondition(predicate);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    return []; // the build already reported the condition
  }
  return [...nonNullBases(proves(tree, true))].map((base) => base.split(".")).toSorted(comparePaths);
}

/** Return the dotted paths `guard` proves not null (see `whileNonNullPaths`). */
function nonNullBases(guard: Guard): Set<string> {
  if (guard instanceof Holds) {
    return guard.check instanceof NotNullCheck && guard.value ? new Set([guard.check.base]) : new Set();
  }
  const parts = guard.parts.map(nonNullBases);
  if (guard instanceof AllOf) return new Set(parts.flatMap((part) => [...part]));
  if (parts.length === 0) return new Set();
  return new Set([...parts[0]!].filter((base) => parts.every((part) => part.has(base))));
}

/**
 * Return `typ` with the value at `path`, and each record above it, made nullable.
 *
 * The path is followed through record fields. Where it names an absent field, or stops at a union
 * or a non-record before its end, the rest of it is left as is, but the value it stopped at is
 * still made nullable. Each field on the path stops being required in its record, since a field
 * read through a null or absent record is null. The null type is returned as it is.
 *
 * @param typ The type to change; it is not changed itself.
 * @param path Field names from `typ` to the value; `[]` is `typ` itself.
 * @returns The changed copy.
 *
 * @example
 * `nullableAt({next: str}, ["next"])` with `next` required is `Optional({next: Optional(str)})`
 * with `next` not required.
 */
export function nullableAt(typ: Type, path: readonly string[]): Type {
  const [head, ...rest] = path;
  let changed = typ;
  if (head !== undefined && typ.variants === undefined && typ.fields?.has(head)) {
    const fields = new Map(typ.fields);
    fields.set(head, nullableAt(typ.fields.get(head)!, rest));
    const required = new Set(typ.required);
    required.delete(head);
    changed = typ.replace({ fields, required });
  }
  if (changed.kind === ValueKind.NONE) return changed; // the null type is already nullable
  return changed.replace({ nullable: true });
}

/**
 * Return a loop body's inputs as the values that enter the body may fill them.
 *
 * A path the loop's `while:` proves not null is never null when the body starts, so each input
 * type is read as nullable at such a path and at each record above it. The seed check and the
 * returned-field check compare against these, while the node's `childInputs` stay as declared.
 *
 * @param node The built loop node.
 * @returns The body's inputs in order: each relaxed one a copy keeping its origins, each other one
 *   the node's own parameter.
 */
export function loopEntryInputs(node: LoopNode): Parameter[] {
  const paths = whileNonNullPaths(node.predicateKind, node.predicate);
  return node.childInputs.map((parameter) => {
    const contract = parameter.contract;
    if (contract === null) return parameter;
    let typ = contract.type;
    for (const path of paths) {
      if (path[0] === parameter.name) typ = nullableAt(typ, path.slice(1));
    }
    if (typ === contract.type) return parameter;
    return new Parameter(parameter.name, {
      contract: new ValueContract(typ, contract.origins),
      required: parameter.required,
      hasDefault: parameter.hasDefault,
      default: parameter.default,
      coerceStrings: parameter.coerceStrings,
      origin: parameter.origin,
      defaultOrigin: parameter.defaultOrigin,
    });
  });
}

/**
 * Return the proven field types of a loop's carried-record contract: an exact record's fields'
 * types, a partly typed record's proven field contracts' types; anything else proves no field.
 *
 * @param contract The loop node's `outputContract`: the body's own contract for a total loop, or
 *   the merged returned-plus-seeded contract for a partial one.
 * @returns Each proven carried field's type, keyed by field name, in field order.
 */
export function carriedFieldTypes(contract: OutputContract | null): Map<string, Type> {
  if (isExactly(contract, ValueContract) && contract.type.fields !== undefined) {
    return new Map(contract.type.fields);
  }
  if (isExactly(contract, PartialRecordContract)) {
    const types = new Map<string, Type>();
    for (const [name, fieldContract] of contract.fields) {
      if (fieldContract !== null) types.set(name, fieldContract.type);
    }
    return types;
  }
  return new Map();
}
