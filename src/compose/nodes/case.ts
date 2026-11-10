/**
 * kind=case: the branch-routing node's descriptor.
 *
 * Owns the `case` node's `CaseDescriptor`. Its `parse` wraps the template `parse` from
 * `NodeDescriptor` to check the `cases:` shape and keep each `cases[N].when` / `then` value's
 * range, the `then:` and `else:` values' YAML styles and tags, and whether `else:` was written.
 * Its `withSourceOrigins` registers each indexed predicate and an optional `on:` site. Registered
 * in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Unlike the leaf kinds, a `case` lowers in its own `build` (pattern-match compilation): it
 * carries no inputs and no leaf node, and its build returns the strict `CaseNode` with the case's
 * data edges followed by its control edges. Two surface forms:
 *
 * - searched (`cases: [{when: "<bool>", then: <t>}]`, `else: <e>`): each distinct read root of the
 *   `when:`s (defined at `searchedParams`) becomes one `CaseNode` input `__rN`. Each reference's
 *   root in a `when:` is rewritten to its local `__rN`, keeping the steps after it; the `when:`
 *   itself stays lazily evaluated.
 * - `on:` (`on: ${ref}`, `cases: [{when: <value>, then: <t>}]`): one input `__on = ${ref}`; each
 *   `when: <value>` is a literal compared exactly: text becomes `__on == "<value>"`, a number
 *   `__on == <n> and not isinstance(__on, bool)`, a bool `__on == true and isinstance(__on, bool)`.
 *   The `isinstance` guard keeps a bool from equalling a number even when the `on:` type is
 *   unknown.
 *
 * Each case routes to its `then:` (the edge handle); `else:` routes to the else handle
 * (`ELSE_HANDLE`). The lowered `when:`s read only the node-local `__rN` / `__on` names, so the
 * `CaseNode` evaluates them against its bound input record. Exhaustiveness is a separate validate
 * pass: it needs the producers' types, known only once every node is built.
 *
 * Knows about: diagnostics, the compile model, expr, nodes (base, case, contracts), typesys,
 * compose source and helpers.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { Edge } from "../../compile/model.ts";
import { OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import {
  bindingCoSkips,
  type ConditionRead,
  conditionReads,
  ExpressionError,
  exprRefsOf,
  parseCondition,
  rewriteConditionReads,
} from "../../expr/index.ts";
import { ELSE_HANDLE } from "../../nodes/base.ts";
import { Case, CaseNode } from "../../nodes/case/node.ts";
import {
  Binding,
  bindingSource,
  iterSourceStrings,
  Parameter,
  validateOptionalOrigin,
} from "../../nodes/contracts.ts";
import { NEW_MESSAGES, poolRootLength } from "../../typesys/pool.ts";
import { typeName } from "../../typesys/types.ts";
import { isRecord, valueRepr } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { hasSpan } from "../helpers/tags.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { ScalarStyle } from "../source/common.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  type BuildContext,
  BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  rejectNonText,
  siteKey,
} from "./base.ts";

const ON_FIX = "write a reference such as `on: ${input.x}`";

/** One authored `cases:` entry as YAML read it: `when:` and `then:` to their values. */
export type CaseSpec = Readonly<Record<string, unknown>>;

/** The fields `CaseDescriptor`'s constructor takes: the common ones and the CASE's own. */
export interface CaseDescriptorFields extends NodeDescriptorFields {
  cases?: readonly CaseSpec[];
  else_?: unknown;
  hasElse?: boolean;
  on?: string | null;
  whenRanges?: readonly (SourceRange | null)[];
  thenRanges?: readonly (SourceRange | null)[];
  thenStyles?: readonly (ScalarStyle | null)[];
  elseStyle?: ScalarStyle | null;
  thenTags?: readonly (string | null)[];
  elseTag?: string | null;
  thenIncludes?: readonly (string | null)[];
  elseInclude?: string | null;
  whenOrigins?: readonly (OriginRef | null)[];
  onOrigin?: OriginRef | null;
}

/**
 * kind=case: branch routing, in the searched (`when:`) or the `on:`-value form.
 *
 * `allowed` adds `cases`, `else`, `on` and `outputs` to the commons, and `cases` is required. An
 * authored `output:` still parses (into `outputs`) so the loader can reject it with the rewrite.
 * `bindingFields` is empty: a `case` carries conditions (`on:` and each `when:`), not `${...}`
 * sink bindings. `else:` is read into `else_`.
 *
 * `thenStyles` (aligned with `cases`), `elseStyle` and `hasElse` let the branch check tell an
 * unquoted node name from quoted text, and `else: null` from no `else:`; `thenTags` and `elseTag`
 * keep a tagged value's tag, and `thenIncludes` and `elseInclude` a value's `!include(T)` as
 * written. A style is the YAML scalar style of the authored value (`null` for a
 * mapping, a list, a missing value or a source-free cursor); downstream reads `null` as plain.
 */
export class CaseDescriptor extends NodeDescriptor {
  static override readonly kind = "case";
  static override readonly allowed = this.allow("cases", "else", "on", "outputs");
  static override readonly required = this.require("cases");
  // A case's conditions belong to its entries, not the node, so point there instead.
  static override readonly whenHint =
    "; `when:` goes in each entry of `cases:` (`cases: [{when: ..., then: <node>}]`)";
  static override readonly bindingFields: readonly string[] = [];
  static override readonly writtenFields = [
    ["on", "the value a case matches"],
    ["cases[].when", "a condition"],
  ] as const;
  static override readonly keywordRemap: ReadonlyMap<string, string> = new Map([["else", "else_"]]);

  /**
   * The authored `cases:` entries. The template `parse` hands over whatever YAML read (its fields
   * are cast), so until `parse`'s shape check it may be any value; a direct construction gives a
   * list. The list is kept as given, not copied.
   */
  readonly cases: readonly CaseSpec[];
  /** The `else:` fallback as YAML read it; `null` when absent or written `null`. */
  readonly else_: unknown;
  /** Whether `else:` was written, even as `else: null`. */
  readonly hasElse: boolean;
  /** The `on:` reference text; `null` in the searched form. */
  readonly on: string | null;
  /**
   * The range of each `cases[N].when` value, aligned with `cases` (`null` where none exists), so a
   * generated parameter (`__rN`) locates at its own line. Position, not text, is the identity.
   */
  readonly whenRanges: readonly (SourceRange | null)[];
  /** The range of each `cases[N].then` value, aligned with `cases`, so a call target locates. */
  readonly thenRanges: readonly (SourceRange | null)[];
  /** The YAML scalar style of each `cases[N].then` value, aligned with `cases`. */
  readonly thenStyles: readonly (ScalarStyle | null)[];
  /** The YAML scalar style of the `else:` value. */
  readonly elseStyle: ScalarStyle | null;
  /** The local tag of each `cases[N].then` value (`null` when untagged), aligned with `cases`. */
  readonly thenTags: readonly (string | null)[];
  /** The local tag of the `else:` value; `null` when untagged. */
  readonly elseTag: string | null;
  /**
   * The `!include(T) PATH` each `cases[N].then` value was read with, as written (`null` for one
   * written in the flow), aligned with `cases`.
   */
  readonly thenIncludes: readonly (string | null)[];
  /** The `!include(T) PATH` the `else:` value was read with, as written; `null` when none. */
  readonly elseInclude: string | null;
  /**
   * The source-neutral site of each `cases[N].when`: empty for a direct or source-free
   * construction, otherwise aligned with `cases`.
   */
  readonly whenOrigins: readonly (OriginRef | null)[];
  /** The site of `on:`, which belongs to the generated `__on` binding, not to a case. */
  readonly onOrigin: OriginRef | null;

  /**
   * Create a descriptor from `fields`. A given non-null `else_` sets `hasElse`.
   *
   * @throws TypeError when `whenOrigins` isn't an array, holds anything but an `OriginRef` or
   *   `null`, or `onOrigin` or `origin` isn't an `OriginRef` or `null`.
   * @throws RangeError when `whenOrigins` is non-empty and `cases` isn't a list of the same length.
   */
  constructor(fields: CaseDescriptorFields) {
    super(fields);
    // Not `?? []`: a YAML `cases: null` must stay null so `_checkCasesShape` can reject it.
    this.cases = fields.cases === undefined ? [] : fields.cases;
    this.else_ = fields.else_ ?? null;
    this.hasElse = (fields.hasElse ?? false) || this.else_ !== null;
    this.on = fields.on ?? null;
    this.whenRanges = fields.whenRanges ?? [];
    this.thenRanges = fields.thenRanges ?? [];
    this.thenStyles = fields.thenStyles ?? [];
    this.elseStyle = fields.elseStyle ?? null;
    this.thenTags = fields.thenTags ?? [];
    this.elseTag = fields.elseTag ?? null;
    this.thenIncludes = fields.thenIncludes ?? [];
    this.elseInclude = fields.elseInclude ?? null;
    const whenOrigins: unknown = fields.whenOrigins ?? [];
    if (!Array.isArray(whenOrigins)) throw new TypeError("CASE predicate origins must be an array");
    // A frozen copy, so the origins can't change after the alignment check.
    this.whenOrigins = Object.freeze([...whenOrigins]);
    this._validateOriginAlignment();
    for (const origin of this.whenOrigins) {
      if (origin !== null && !(origin instanceof OriginRef)) {
        throw new TypeError("CASE predicate origin must be an OriginRef or null");
      }
    }
    this.onOrigin = fields.onOrigin ?? null;
    validateOptionalOrigin(this.onOrigin, "CASE on origin");
  }

  /**
   * Keep present predicate origins aligned with a list of cases.
   *
   * @throws RangeError when `whenOrigins` is non-empty and `cases` isn't a list of its length.
   */
  _validateOriginAlignment(): void {
    if (this.whenOrigins.length === 0) return;
    if (!Array.isArray(this.cases) || this.cases.length !== this.whenOrigins.length) {
      throw new RangeError("CASE predicate origins must align with cases");
    }
  }

  /**
   * Reject an `on:` YAML read as non-text (`null` stays absent), text holding a lone surrogate or
   * text with no `${...}` reference, then run the base checks.
   *
   * @throws LoadError at `on:` (else at the node).
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
    cursor: SourceCursor,
  ): void {
    const on = fields["on"] ?? null;
    const range = ranges.get(siteKey("field", "on")) ?? ownRange;
    rejectNonText(this.kind, nodeId, "on", on, range, diagnostics, { fix: ON_FIX });
    if (typeof on === "string") {
      checkWrittenValue(on, `node ${reprText(nodeId)} field 'on'`, range, diagnostics);
      // Text with no span is a constant, so every run would route the same way.
      if (!hasSpan(on)) {
        throw diagnostics.error(
          `node ${reprText(nodeId)} (kind=${this.kind}): \`on:\` must be a reference, ` +
            `but has no \`\${...}\`; ${ON_FIX}`,
          { range },
        );
      }
    }
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics, cursor);
  }

  /**
   * Parse as usual, check the `cases:` shape and each text `when:`, then keep each `cases[N].when`
   * and `then` value's range, each `then:` value's style and tag, the `else:` value's style and
   * tag, and whether `else:` was written.
   *
   * The `cases:` entries have no unique text key, so `fieldRanges` can't reach them; the
   * positional `sequenceFieldRanges` does.
   *
   * @throws LoadError when `cases:` isn't a list of mappings (at the field or the item), a text
   *   `when:` holds a lone surrogate (at that `when:`), or the template `parse` rejects the body.
   */
  static override parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const desc = super.parse(nodeId, cursor) as CaseDescriptor;
    const cases = CaseDescriptor._checkCasesShape(desc, cursor);
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    const whenRanges = cursor.sequenceFieldRanges("cases", "when");
    const thenRanges = cursor.sequenceFieldRanges("cases", "then");
    cases.forEach((item, index) => {
      const when = item["when"];
      if (typeof when === "string") {
        checkWrittenValue(
          when,
          `node ${reprText(nodeId)} field 'cases[${index}].when'`,
          whenRanges[index] ?? desc.range,
          diagnostics,
        );
      }
    });
    const hasElse = cursor.has("else");
    const casesCursor = cursor.child("cases");
    const thens = cases.map((item, index) =>
      Object.hasOwn(item, "then") ? casesCursor.child(index).child("then") : null,
    );
    return desc.replace({
      hasElse,
      elseStyle: hasElse ? cursor.child("else").scalarStyle : null,
      elseTag: hasElse ? cursor.child("else").tag : null,
      whenRanges,
      thenRanges,
      thenStyles: thens.map((then) => then?.scalarStyle ?? null),
      thenTags: thens.map((then) => then?.tag ?? null),
      thenIncludes: thens.map((then) => then?.typedInclude?.written ?? null),
      elseInclude: hasElse ? (cursor.child("else").typedInclude?.written ?? null) : null,
    }) as InstanceType<T>;
  }

  /**
   * Return `desc.cases` once it is a list of mappings; later passes index it so.
   *
   * @throws LoadError at `cases:` when it isn't a list (a mapping gets a hint to start each case
   *   with `- `) or is empty with no `else:`, or at the first item that isn't a mapping.
   */
  static _checkCasesShape(desc: CaseDescriptor, cursor: SourceCursor): readonly CaseSpec[] {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    const where = `case node ${reprText(desc.id)}`;
    const cases: unknown = desc.cases;
    if (!Array.isArray(cases)) {
      const hint = isRecord(cases) ? "; start each case with `- `" : "";
      throw diagnostics.error(
        `${where}: \`cases:\` must be a list of {when:, then:} mappings (got ${valueRepr(cases)})${hint}`,
        { range: desc.fieldRanges.get(siteKey("field", "cases")) ?? desc.range },
      );
    }
    if (cases.length === 0 && !desc.hasElse) {
      throw diagnostics.error(
        `${where}: \`cases:\` is empty and there is no \`else:\`, so it routes nowhere; add a case or an \`else:\``,
        { range: desc.fieldRanges.get(siteKey("field", "cases")) ?? desc.range },
      );
    }
    const casesCursor = cursor.child("cases");
    cases.forEach((item: unknown, index) => {
      if (!isRecord(item)) {
        throw diagnostics.error(
          `${where}: \`cases[${index}]\` must be a {when:, then:} mapping (got ${valueRepr(item)})`,
          { range: casesCursor.child(index).range ?? desc.range },
        );
      }
    });
    return cases as readonly CaseSpec[];
  }

  /**
   * Attach each authored predicate's origin and the optional `on:` origin.
   *
   * Predicate sites are indexed, so duplicate expression text stays distinct. A missing or
   * malformed entry keeps `null` and reaches the build's own check, so the error order is the
   * same.
   */
  override withSourceOrigins(
    cursor: SourceCursor,
    options: { sitePath: readonly (string | number)[]; catalogBuilder: SourceCatalogBuilder },
  ): this {
    const { sitePath, catalogBuilder } = options;
    const specs: readonly unknown[] = Array.isArray(this.cases) ? this.cases : [];
    const whenOrigins: (OriginRef | null)[] =
      this.whenOrigins.length > 0 ? [...this.whenOrigins] : specs.map(() => null);
    const casesCursor = cursor.child("cases");
    specs.forEach((spec, index) => {
      if (!isRecord(spec)) return;
      const itemCursor = casesCursor.child(index);
      if (!itemCursor.has("when")) return;
      const registered = catalogBuilder.register([...sitePath, "case", index, "when"], itemCursor.child("when"));
      if (registered !== null) whenOrigins[index] = registered;
    });
    let onOrigin = this.onOrigin;
    if (this.on !== null && cursor.has("on")) {
      onOrigin = catalogBuilder.register([...sitePath, "field", "on"], cursor.child("on")) ?? onOrigin;
    }
    return this.replace({ whenOrigins, onOrigin });
  }

  /**
   * Lower this `case` to a strict `CaseNode`, its flow-owned wiring (`__rN` or `__on` to its
   * source) and its edges.
   *
   * The edges are the data edges (one per `${<id>.output}` or `${<id>.new_messages}` a generated
   * input's source reads, keyed by the input's name) followed by the control edges (this node to
   * each `then:` or `else:` target, keyed by the branch handle). Exhaustiveness is not checked
   * here: that pass needs the assembled producers' types. Each generated input's range goes into
   * the build result's `fieldRanges` under `siteKey("input", name)`, and its label
   * (`cases[N].when`, `on:`) into the node's `wiringLabels`, so a reference error names the
   * authored site, not the generated name.
   *
   * @throws RangeError when `whenOrigins` no longer aligns with `cases`.
   * @throws LoadError when a case lacks its `when:` or `then:`, an `on:` case's `when:` value
   *   can't be compared, or a searched `when:` isn't text or doesn't parse (at that `when:`).
   */
  override build(ctx: BuildContext): BuildResult {
    this._validateOriginAlignment();
    const lowered =
      this.on !== null ? desugarOn(this, this.on, ctx.diagnostics) : desugarSearched(this, ctx.diagnostics);
    const result = this._finish(lowered.node, lowered.wiring, [...lowered.dataEdges, ...lowered.controlEdges]);
    result.node.wiringLabels = lowered.paramLabels;
    return new BuildResult({ ...result, fieldRanges: new Map([...result.fieldRanges, ...lowered.paramRanges]) });
  }
}

/**
 * A lowered `case`: its node, its flow-owned wiring, its data and control edges, and each
 * generated input's range (keyed by `siteKey("input", name)`) and author-facing label.
 */
interface Lowering {
  node: CaseNode;
  wiring: Map<string, unknown>;
  dataEdges: Edge[];
  controlEdges: Edge[];
  paramRanges: Map<string, SourceRange>;
  paramLabels: Map<string, string>;
}

/** The reserved heads that name no node, whatever field follows them. */
const NON_NODE_HEADS: ReadonlySet<string> = new Set(["input", "system", "item"]);

/**
 * Return the producing node id of reference path `<id>.output[...]` or an AGENT's
 * `<id>.new_messages[...]`, else `null` (also for `input.output`, a flow input so named).
 */
function nodeProducer(ref: string): string | null {
  const parts = ref.split(".");
  if (parts.length < 2 || NON_NODE_HEADS.has(parts[0]!)) return null;
  return parts[1] === "output" || parts[1] === NEW_MESSAGES ? parts[0]! : null;
}

/**
 * Return `value` as a `when:` string literal, for `__on == "<value>"`.
 *
 * The condition grammar's strings (`"[^"]*"` or `'[^']*'`) have no escapes, so the value takes
 * the quote it doesn't hold: double quotes unless it holds `"`, else single quotes. Backslashes
 * need no escaping.
 *
 * @throws LoadError at `range` when `value` holds both quote characters.
 */
function quote(value: string, diagnostics: LoadDiagnostics, range: SourceRange | null): string {
  if (!value.includes('"')) return `"${value}"`;
  if (!value.includes("'")) return `'${value}'`;
  throw diagnostics.error(
    `case \`on:\` match value ${reprText(value)} contains both ' and " — unrepresentable in a ` +
      "`when:` string literal (no escape support); rename the matched value",
    { range },
  );
}

/**
 * Return a key that is equal for two `when:` values exactly when they match the same `on:` values.
 *
 * Text matches text, a bool only the same bool, and numbers match numerically, so `1n` and `1.0`
 * share a key while `true`, `1n` and `"1"` don't. The key is a string, so a `Set` or `Map`
 * compares it by value.
 *
 * @param value An `on:` case's `when:` value: text, a bool or a number.
 * @returns `text:<s>`, `bool:true` / `bool:false`, or `number:<n>`, where `<n>` is a bigint's
 *   digits, an integral float's digits (so `-0` keys as `0`), `inf` / `-inf`, or another float's
 *   `String` form. A bigint never goes through `Number`, which would round it.
 */
export function onMatchKey(value: string | boolean | bigint | number): string {
  if (typeof value === "string") return `text:${value}`;
  if (typeof value === "boolean") return `bool:${value}`;
  if (typeof value === "bigint") return `number:${value}`;
  if (value === Infinity) return "number:inf";
  if (value === -Infinity) return "number:-inf";
  return `number:${Number.isInteger(value) ? BigInt(value) : String(value)}`;
}

/**
 * Return a number as condition-literal digits: positional (the grammar's numbers have no
 * exponent), a float keeping a `.` and the sign of `-0`, and infinity as `inf` / `-inf`. Also spells
 * the floats of a case branch's suggested value entry (`validate/case_branches.ts`).
 */
export function numberText(value: bigint | number): string {
  if (typeof value === "bigint") return String(value);
  if (value === Infinity) return "inf";
  if (value === -Infinity) return "-inf";
  // YAML reads `-0.0` as the float `0` (yian has no negative zero), so only a descriptor built
  // in code reaches this branch.
  if (Object.is(value, -0)) return "-0.0";
  // `String` is the shortest round-tripping form, but it may use an exponent: move the point.
  const [mantissa, exponent = "0"] = String(Math.abs(value)).split("e");
  const [whole, fraction = ""] = mantissa!.split(".");
  const digits = whole! + fraction;
  const point = whole!.length + Number(exponent);
  let text: string;
  if (point <= 0) text = `0.${"0".repeat(-point)}${digits}`;
  else if (point >= digits.length) text = `${digits}${"0".repeat(point - digits.length)}.0`;
  else text = `${digits.slice(0, point)}.${digits.slice(point)}`;
  return value < 0 ? `-${text}` : text;
}

/**
 * Return the `__on` condition one `on:` case's `when:` value lowers to.
 *
 * Text compares as a string literal, a number numerically and never with a bool, and a bool only
 * with a bool.
 *
 * @throws LoadError at `range` when the value is a list, a record or another non-scalar, or text
 *   holding both quote characters.
 */
function matchCondition(
  value: unknown,
  diagnostics: LoadDiagnostics,
  nodeId: string,
  range: SourceRange | null,
): string {
  if (typeof value === "string") return `__on == ${quote(value, diagnostics, range)}`;
  // `==` counts a bool as 0 or 1; the isinstance guard keeps a bool from equalling a number.
  if (typeof value === "boolean") return `__on == ${value} and isinstance(__on, bool)`;
  if (typeof value === "bigint" || typeof value === "number") {
    return `__on == ${numberText(value)} and not isinstance(__on, bool)`;
  }
  const kind = Array.isArray(value) ? "a list" : isRecord(value) ? "a record" : typeName(value);
  throw diagnostics.error(
    `case node ${reprText(nodeId)}: an on: case's when: takes text, a number, or a bool, got ${kind}`,
    { range },
  );
}

/**
 * Return the data edges of a lowered `case`: one per `${<id>.output}` or AGENT
 * `${<id>.new_messages}` reference a generated input's source reads.
 *
 * Each edge's `inputGroup` is the input's name and its `sourceHandle` is `null`, the opposite of
 * a control edge, so the flat edge list splits back by that attribute.
 */
function dataEdges(nodeId: string, wiring: ReadonlyMap<string, unknown>): Edge[] {
  const edges: Edge[] = [];
  const counts = new Map<string, number>();
  for (const [name, source] of wiring) {
    // Each source carries a binding's stance: an `on:` value may carry an escape, a searched
    // node-output read inside one is `${path :- null}` (optional), and a co-skipping escape's
    // join input is required.
    const optional = !bindingCoSkips(bindingSource(source));
    for (const [, text] of iterSourceStrings(source)) {
      let refs: string[];
      try {
        refs = exprRefsOf(text);
      } catch (error) {
        // A malformed reference is reported, located, by the reference-wiring pass.
        if (error instanceof ExpressionError) continue;
        throw error;
      }
      for (const ref of refs) {
        const producer = nodeProducer(ref);
        if (producer === null) continue;
        const index = counts.get(producer) ?? 0;
        counts.set(producer, index + 1);
        edges.push(new Edge(`${producer}->${nodeId}#${index}`, producer, nodeId, { inputGroup: name, optional }));
      }
    }
  }
  return edges;
}

/**
 * Return the control edges of a lowered `case`: this node to each target, carrying its branch
 * handle (a case's `then:` or the else handle) as `sourceHandle`, with no `inputGroup`.
 */
function controlEdges(nodeId: string, handleTargets: readonly (readonly [string, string])[]): Edge[] {
  const counts = new Map<string, number>();
  return handleTargets.map(([handle, target]) => {
    const index = counts.get(target) ?? 0;
    counts.set(target, index + 1);
    return new Edge(`${nodeId}->${target}#${index}`, nodeId, target, { sourceHandle: handle });
  });
}

/**
 * Lower the `on: ${ref}` form: one `__on` input and one `__on` match condition per case.
 *
 * The `__on` input is located at the `on:` value and labelled `on:`.
 *
 * @param on The `on:` reference text (`desc.on`, narrowed by the caller).
 * @throws LoadError when a case lacks its `when:` value or `then:` (at the missing value, else at
 *   `cases:`), or a `when:` value can't be compared (at that `when:`).
 */
function desugarOn(desc: CaseDescriptor, on: string, diagnostics: LoadDiagnostics): Lowering {
  const cases: Case[] = [];
  const handleTargets: [string, string][] = [];
  desc.cases.forEach((spec, index) => {
    const value = spec["when"] ?? null;
    const then = spec["then"] ?? null;
    if (value === null || then === null) {
      const missingRange =
        value === null && index < desc.whenRanges.length ? desc.whenRanges[index] : desc.thenRanges[index];
      throw diagnostics.error(
        `case node ${reprText(desc.id)}: each on:-form case needs a \`when:\` value and a \`then:\``,
        { range: missingRange ?? desc.fieldRanges.get(siteKey("field", "cases")) ?? desc.range },
      );
    }
    // The branch check runs before the build in the pipeline and has made each
    // `then:` and the `else:` a node name, so this cast and the `else_` one below hold.
    const handle = then as string;
    cases.push(
      new Case(handle, {
        when: matchCondition(value, diagnostics, desc.id, desc.whenRanges[index] ?? desc.range),
        origin: desc.whenOrigins[index] ?? null,
      }),
    );
    handleTargets.push([handle, handle]);
  });
  if (desc.else_ !== null) handleTargets.push([ELSE_HANDLE, desc.else_ as string]);
  const node = new CaseNode(desc.id, cases, { title: desc.nodeName });
  node.params = [new Parameter("__on")];
  const wiring = new Map<string, unknown>([
    ["__on", desc.onOrigin === null ? on : new Binding(on, { origin: desc.onOrigin })],
  ]);
  const onRange = desc.fieldRanges.get(siteKey("field", "on"));
  return {
    node,
    wiring,
    dataEdges: dataEdges(desc.id, wiring),
    controlEdges: controlEdges(desc.id, handleTargets),
    paramRanges: new Map(onRange === undefined ? [] : [[siteKey("input", "__on"), onRange]]),
    paramLabels: new Map([["__on", "on:"]]),
  };
}

/** The `__rN` inputs of a searched case, from one parse of its `when:`s. */
interface SearchedParams {
  /** Each distinct binding source and its local, in first-seen order. */
  sourceToLocal: Map<string, string>;
  /** The case index each local was first seen in (its caret and label site). */
  localCaseIndex: Map<string, number>;
  /** Per case, the local of each reference occurrence, in `conditionReads` order. */
  caseLocals: string[][];
}

/**
 * Return the binding source of one `when:` reference occurrence, also the key its local is
 * deduped by: the read's root only (`x.output`, `x.new_messages`, `input.k`), so the steps after
 * it stay in the `when:`, where `and`/`or` guards keep them lazy.
 *
 * A read of a node's output or an AGENT's new messages inside an escape is `${root :- null}`: it
 * reads the same value (a miss is already null) but takes an optional edge, so the escape decides
 * as it does in a binding. Any other read (outside an escape, or of an input or `system.*`, which
 * has no producer edge) is the plain `${root}`.
 */
function readSource(read: ConditionRead): string {
  const root = readRoot(read);
  if (read.escape === null || nodeProducer(read.path) === null) return `\${${root}}`;
  return `\${${root} :- null}`;
}

/** Return a `when:` read's root, its first `poolRootLength` names. */
function readRoot(read: ConditionRead): string {
  const names = read.path.split(".");
  return names.slice(0, poolRootLength(names)).join(".");
}

/**
 * Return one join source per co-skipping escape that reads a node output, in source order.
 *
 * A binding of such an escape skips only when all its producers skip; the join
 * `${<root1> | <root2> ...}` gives the gate that same required input group. Each producer's
 * first root is the one joined (`p.output` or an AGENT's `p.new_messages`), so a join of one
 * root reuses a plain read of it.
 *
 * @param reads One `when:`'s reference occurrences, in `conditionReads` order.
 * @returns One join source per such escape, each producer listed once in first-read order.
 */
function joinSources(reads: readonly ConditionRead[]): string[] {
  const roots = new Map<number, Map<string, string>>();
  for (const read of reads) {
    const producer = nodeProducer(read.path);
    if (read.escape === null || !read.coSkips || producer === null) continue;
    const seen = roots.get(read.escape) ?? new Map<string, string>();
    roots.set(read.escape, seen);
    if (!seen.has(producer)) seen.set(producer, readRoot(read));
  }
  return [...roots.values()].map((group) => `\${${[...group.values()].join(" | ")}}`);
}

/**
 * Allocate one `__rN` local per distinct read across all searched `when:`s.
 *
 * A read is a reference's root (`${root}`), a node-output root read inside a `:-` / `:?` / `|`
 * (`${root :- null}`, an optional edge), or, once per co-skipping escape that reads a node output,
 * a join input over its producers, which the `when:` doesn't read but whose required edges skip the
 * gate as that escape's binding would skip; its source is also its dedupe key. Per `when:`, its
 * reference reads are allocated before its join inputs.
 *
 * @throws LoadError at a `when:` that isn't text or doesn't parse (at `cases:` when it is missing).
 */
function searchedParams(desc: CaseDescriptor, diagnostics: LoadDiagnostics): SearchedParams {
  const sourceToLocal = new Map<string, string>();
  const localCaseIndex = new Map<string, number>();
  const caseLocals: string[][] = [];

  const allocate = (source: string, caseIndex: number): string => {
    let local = sourceToLocal.get(source);
    if (local === undefined) {
      local = `__r${sourceToLocal.size}`;
      sourceToLocal.set(source, local);
      localCaseIndex.set(local, caseIndex);
    }
    return local;
  };

  desc.cases.forEach((spec, caseIndex) => {
    const when = spec["when"];
    const whenRange = desc.whenRanges[caseIndex] ?? desc.fieldRanges.get(siteKey("field", "cases")) ?? desc.range;
    if (typeof when !== "string") {
      throw diagnostics.error(
        `case node ${reprText(desc.id)}: each searched case needs a string \`when:\` expression`,
        { range: whenRange },
      );
    }
    let tree: ExprNode;
    try {
      tree = parseCondition(when);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      throw diagnostics.error(
        `case node ${reprText(desc.id)}: \`when:\` ${reprText(when)} is not a valid boolean expression: ` +
          error.message,
        { range: whenRange, cause: error },
      );
    }
    // Parsing the whole value once covers both spellings, as `rewriteWhen` does.
    const reads = conditionReads(tree);
    const locals = reads.map((read) => allocate(readSource(read), caseIndex));
    for (const source of joinSources(reads)) allocate(source, caseIndex);
    caseLocals.push(locals);
  });
  return { sourceToLocal, localCaseIndex, caseLocals };
}

/**
 * Lower the searched form: one `__rN` input per distinct read, each `when:` rewritten over them.
 *
 * Reads are as `searchedParams` defines them. Each `__rN` is located at the `cases[N].when` it
 * was first seen in and labelled `cases[N].when`.
 *
 * @throws LoadError when a `when:` isn't text or doesn't parse (at that `when:`, else at `cases:`),
 *   or a case lacks its `then:` (at the `then:`, else at `cases:`).
 */
function desugarSearched(desc: CaseDescriptor, diagnostics: LoadDiagnostics): Lowering {
  const params = searchedParams(desc, diagnostics);
  const wiring = new Map<string, unknown>();
  for (const [source, local] of params.sourceToLocal) {
    const origin = desc.whenOrigins[params.localCaseIndex.get(local)!] ?? null;
    wiring.set(local, origin === null ? source : new Binding(source, { origin }));
  }

  const cases: Case[] = [];
  const handleTargets: [string, string][] = [];
  desc.cases.forEach((spec, index) => {
    const then = spec["then"] ?? null;
    if (then === null) {
      throw diagnostics.error(`case node ${reprText(desc.id)}: each searched case needs a \`then:\``, {
        range: desc.thenRanges[index] ?? desc.fieldRanges.get(siteKey("field", "cases")) ?? desc.range,
      });
    }
    // As in `desugarOn`: the branch check has made each `then:` and the `else:` a node name, and
    // `searchedParams` has made each `when:` text.
    const handle = then as string;
    cases.push(
      new Case(handle, {
        when: rewriteWhen(spec["when"] as string, params.caseLocals[index]!),
        origin: desc.whenOrigins[index] ?? null,
      }),
    );
    handleTargets.push([handle, handle]);
  });
  if (desc.else_ !== null) handleTargets.push([ELSE_HANDLE, desc.else_ as string]);

  const node = new CaseNode(desc.id, cases, { title: desc.nodeName });
  node.params = [...params.sourceToLocal.values()].map((local) => new Parameter(local));
  const paramRanges = new Map<string, SourceRange>();
  const paramLabels = new Map<string, string>();
  for (const [local, caseIndex] of params.localCaseIndex) {
    const range = desc.whenRanges[caseIndex] ?? null;
    if (range !== null) paramRanges.set(siteKey("input", local), range);
    paramLabels.set(local, `cases[${caseIndex}].when`);
  }
  return {
    node,
    wiring,
    dataEdges: dataEdges(desc.id, wiring),
    controlEdges: controlEdges(desc.id, handleTargets),
    paramRanges,
    paramLabels,
  };
}

/**
 * Rewrite the root of each reference occurrence in `when` to its local `__rN`.
 *
 * Works in either condition spelling (bare, or one whole `${...}`) and keeps the surrounding
 * expression, so the `when:` still evaluates lazily case by case. `rewriteConditionReads` walks
 * the same occurrences as `conditionReads` in `searchedParams`, so `locals[i]` is occurrence
 * `i`'s local.
 *
 * @param when The authored `when:`.
 * @param locals The local of each reference occurrence, in `conditionReads` order.
 * @returns The `when:` over the case's locals.
 */
function rewriteWhen(when: string, locals: readonly string[]): string {
  // Each local binds its read's root, so only the root is replaced; the steps after it stay as
  // written, `?.` included.
  return rewriteConditionReads(when, (i, read) => ({ text: locals[i]!, used: poolRootLength(read.path.split(".")) }));
}
