/**
 * Case exhaustiveness: the validate-phase check.
 *
 * Two coverage laws for `case` nodes:
 *
 * - Enum coverage, for every case: when an `on:` ref resolves to an ENUM node output or flow
 *   input, every enum tag must be covered by a case `when:` value or by a present `else:`. A
 *   missing tag with no `else:` is a `LoadError`.
 * - Value-producing cases: a case that produces a value (it is read through `${<case>.output}`, or
 *   is a branch target of such a case) must be exhaustive. It has an `else:`, or it is an `on:`
 *   form whose `on:` type is non-nullable and fully listed by its `when:` values: an enum with
 *   every tag, or a `bool` with both `true` and `false`. `when:` values compare as typed values,
 *   so the text `"true"` does not cover `true`. Its `on:` resolves against node outputs and flow
 *   inputs, under the case's narrowing view. Otherwise it is a `LoadError`. A routing-only case
 *   with no match runs no branch, so only enum coverage applies to it.
 *
 * One membership law for `on:`-form `when:` values (`checkCaseWhenValues`): where the `on:` type
 * is known, every `when:` value must be a possible member of it (compared exactly: no text
 * conversion, a bool is not a number), even when the case has an `else:`. A value that can never
 * equal the `on:` value is a `LoadError` naming the value, the type, and the fix. Two `when:`
 * values that are equal (typed equality, `onMatchKey`) are a `LoadError` at the later one, for
 * every `on:`-form case, since the later one never matches.
 *
 * This lives in the validate phase (not in `CaseDescriptor.build`) because it needs the flow's
 * `producers` map (each producer node's output contract), which only exists after every node is
 * built. The build lowers the case unconditionally; the loader runs these checks in the validate
 * phase, alongside the other cross-node checks.
 *
 * Knows about: `diagnostics` (`SourceRange`), `expr` (the ref scan), `nodes/contracts` (the
 * output-contract sum), `typesys` (`Type` and the strict value check), `compile/output_contracts`,
 * and the compose peers `helpers/errors`, `helpers/tags`, `helpers/yaml_words`,
 * `nodes/case`, `source`, `validate/narrowing` and `validate/typed_bindings`.
 * Never imports: build, the loader, or the runtime; nothing in the engine imports this back.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError, exprRefsOf } from "../../expr/index.ts";
import { isExactly, type OutputContract, ValueContract } from "../../nodes/contracts.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  type Type,
  TypeCheckError,
  ValueKind,
  writeAsText,
} from "../../typesys/values.ts";
import { LoadDiagnostics, sourceText } from "../helpers/errors.ts";
import { assertExactContracts, resolveOutputBindingContract } from "../../compile/output_contracts.ts";
import { valueFits, yamlQuoted } from "../helpers/tags.ts";
import { formerBooleanFix } from "../helpers/yaml_words.ts";
import type { NodeDescriptor } from "../nodes/base.ts";
import { CaseDescriptor, onMatchKey } from "../nodes/case.ts";
import { parseYamlLiteral, SourceSyntaxError } from "../source/index.ts";
import type { Narrowing } from "./narrowing.ts";
import { shownLiteral } from "./typed_bindings.ts";

/** A matchable `on:`-form `when:` value: text, a bool, an int or a float. */
type WhenValue = string | boolean | bigint | number;

/**
 * Check every case's enum coverage, and that each value-producing case is exhaustive.
 *
 * A known field projected from either arm of a partial record can drive enum coverage. With
 * `narrowing`, each case resolves its `on:` ref against its own view of the producers and flow
 * inputs, so a case inside a branch covers only the tags that can reach it, and enum coverage
 * applies to an `on:` over a flow input too. Errors are located at the case node.
 *
 * @param descriptors The flow's node descriptors by id.
 * @param producers Each producer node's exact or partial output contract by id.
 * @param options.narrowing Each node's narrowed producer and flow-input views. Without it, a case
 *   sees `producers` and no flow-input contracts.
 * @param options.valueCases The ids of the value-producing cases, which must have an `else:` or
 *   list every member of a non-nullable `Literal` or `bool` `on:` type.
 * @param options.diagnostics Source ownership for located errors.
 * @throws LoadError when a case leaves an enum tag uncovered with no `else:`, or a value-producing
 *   case is not exhaustive.
 * @throws TypeError when a producer contract is not exactly a `ValueContract` or a
 *   `PartialRecordContract`.
 *
 * @example
 * A case `route` with `on: ${cls.output}` over a `Literal(a, b, c)` producer, `when: a` and
 * `when: b`, and no `else:` fails with ``case node 'route' on ${cls.output}: non-exhaustive — enum
 * tag(s) ["c"] not covered by a case `when:` or an `else:` ``.
 */
export function checkCaseExhaustive(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  options: {
    narrowing?: Narrowing | null;
    valueCases?: ReadonlySet<string>;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const narrowing = options.narrowing ?? null;
  const valueCases = options.valueCases ?? new Set<string>();
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  assertExactContracts(producers);
  for (const desc of descriptors.values()) {
    if (!(desc instanceof CaseDescriptor)) continue;
    const [caseProducers, caseInputs] =
      narrowing !== null ? narrowing.forNode(desc.id) : [producers, new Map<string, ValueContract>()];
    if (valueCases.has(desc.id)) checkValueExhaustive(desc, caseProducers, caseInputs, diagnostics);
    checkExhaustive(desc, onCovered(desc), caseProducers, caseInputs, diagnostics);
  }
}

/**
 * Reject an `on:`-form `when:` value that repeats an earlier one or can never equal the `on:`
 * value.
 *
 * Two `when:` values are duplicates when they match the same `on:` values (typed equality: `1`
 * and `1.0` are equal, `1`, `true` and `"1"` are not); the later one never matches. A duplicate is
 * reported before any membership error in the same case, so a repeated value is reported as such
 * rather than as a non-member whose fix would repeat it.
 *
 * A `when:` value is a member of the `on:` type when the type accepts it as is: text is never
 * converted, a bool is not a number, an int fits a float, and `Literal` tags and numeric bounds
 * apply. The `on:` ref resolves against the unnarrowed producers and flow inputs first, then the
 * case's narrowed view, so a field only a branch's members have is checked too. A case whose `on:`
 * type is unknown or `object` gets no membership check (duplicates are still rejected).
 *
 * @param descriptors The level's node descriptors by id. Non-CASE and searched CASE descriptors
 *   are ignored.
 * @param producers Each producer node's output contract by id.
 * @param flowInputs The flow-input contracts by name.
 * @param options.narrowing Per-node narrowed views, tried when the unnarrowed `on:` type is
 *   unknown.
 * @param options.diagnostics Source ownership for located errors and the authored `when:` text.
 * @throws LoadError when two `when:` values are equal (typed equality), located at the later one;
 *   or when a `when:` value can never be a member of its case's `on:` type, located at the first
 *   such value in source order.
 *
 * @example
 * A case `route` with `on: ${input.n}` over `n: !int` and `when: "1"` fails with
 * ``case node 'route': `when: "1"` is text but `on: ${input.n}` is `int`; drop the quotes:
 * `when: 1` ``.
 */
export function checkCaseWhenValues(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { narrowing?: Narrowing | null; diagnostics?: LoadDiagnostics | null } = {},
): void {
  const narrowing = options.narrowing ?? null;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  for (const desc of descriptors.values()) {
    if (!(desc instanceof CaseDescriptor) || desc.on === null) continue;
    checkDuplicateWhens(desc, diagnostics);
    let typ = resolveOnType(desc.on, producers, flowInputs);
    if (typ === null && narrowing !== null) typ = resolveOnType(desc.on, ...narrowing.forNode(desc.id));
    if (typ === null || typ.kind === ValueKind.OBJECT) continue;
    for (const [index, spec] of desc.cases.entries()) {
      const range = whenRange(desc, index);
      const message = whenValueError(desc.on, spec["when"], typ, sourceText(diagnostics, range));
      if (message !== null) {
        throw diagnostics.error(`case node ${reprText(desc.id)}: ${message}`, { range: range ?? desc.range });
      }
    }
  }
}

/**
 * Reject an `on:`-form case's `when:` value that equals an earlier one.
 *
 * @param desc The `on:`-form case.
 * @param diagnostics Source ownership for located errors and the authored `when:` text.
 * @throws LoadError when two `when:` values match the same `on:` values (`onMatchKey`), located at
 *   the later one, the first such repeat in source order.
 */
function checkDuplicateWhens(desc: CaseDescriptor, diagnostics: LoadDiagnostics): void {
  const seen = new Map<string, number>();
  for (const [index, spec] of desc.cases.entries()) {
    const value = spec["when"];
    if (!VALUE_PHRASES.has(typeof value)) continue;
    const key = onMatchKey(value as WhenValue);
    const first = seen.get(key);
    if (first === undefined) {
      seen.set(key, index);
      continue;
    }
    const range = whenRange(desc, index);
    const firstRange = whenRange(desc, first);
    const message = duplicateWhenMessage(
      value,
      sourceText(diagnostics, range),
      desc.cases[first]!["when"],
      sourceText(diagnostics, firstRange),
      { firstRange, firstIndex: first },
    );
    throw diagnostics.error(`case node ${reprText(desc.id)}: ${message}`, { range: range ?? desc.range });
  }
}

/** Return the source range of a case's `index`-th `when:` value, when known. */
function whenRange(desc: CaseDescriptor, index: number): SourceRange | null {
  return desc.whenRanges[index] ?? null;
}

/**
 * Return the message for a `when:` value that repeats an earlier one. Exported for tests.
 *
 * @param value The later, repeated `when:` value.
 * @param src Its authored source text, when known.
 * @param firstValue The earlier `when:` value it equals.
 * @param firstSrc The earlier value's authored source text, when known.
 * @param options.firstRange The earlier value's source range, when known.
 * @param options.firstIndex The earlier value's index in `cases`, shown when its range is unknown.
 * @returns The message, without the `case node` prefix.
 */
export function duplicateWhenMessage(
  value: unknown,
  src: string | null,
  firstValue: unknown,
  firstSrc: string | null,
  options: { firstRange: SourceRange | null; firstIndex: number },
): string {
  const shown = authoredOrSpelling(value, src);
  const firstShown = authoredOrSpelling(firstValue, firstSrc);
  const { firstRange, firstIndex } = options;
  const where = firstRange !== null ? `first at line ${firstRange.startLine}` : `first at cases[${firstIndex}]`;
  return `\`when: ${shown}\` duplicates \`when: ${firstShown}\` (${where}), so it never matches; remove one`;
}

/**
 * Return a `when:` value's authored text, or its spelling when that text is unknown or starts with
 * an anchor, alias or tag (which is not part of the value's spelling).
 */
function authoredOrSpelling(value: unknown, src: string | null): string {
  if (src !== null && !/^[&*!]/.test(src)) return src;
  return spelling(value);
}

// The scalar kinds of each `when:` value's JS kind, for "this value has the right kind but is
// still not a member" (an out-of-bounds number).
const VALUE_KINDS: ReadonlyMap<string, readonly ValueKind[]> = new Map([
  // Text is how a date or datetime `on:` value is written, so wrong text is a bad value.
  ["string", [ValueKind.STRING, ValueKind.DATE, ValueKind.DATETIME]],
  ["boolean", [ValueKind.BOOLEAN]],
  ["bigint", [ValueKind.INTEGER, ValueKind.NUMBER]],
  ["number", [ValueKind.NUMBER]],
]);

const VALUE_PHRASES: ReadonlyMap<string, string> = new Map([
  ["string", "text"],
  ["boolean", "a bool"],
  ["bigint", "an int"],
  ["number", "a float"],
]);

/**
 * Return why one `when:` value can never equal an `on:` value of type `typ`.
 *
 * @param on The case's `on:` template, as authored.
 * @param value The authored `when:` value.
 * @param typ The resolved `on:` type.
 * @param src The authored source text of the `when:` scalar, when known.
 * @returns The message (without the `case node` prefix), or `null` when the value is a member or
 *   is not a matchable scalar (the build already rejects those).
 */
function whenValueError(on: string, value: unknown, typ: Type, src: string | null): string | null {
  const phrase = VALUE_PHRASES.get(typeof value);
  if (phrase === undefined) return null;
  let error: TypeCheckError;
  try {
    buildValueAs(typ, value);
    return null;
  } catch (caught) {
    if (!(caught instanceof TypeCheckError)) throw caught;
    error = caught;
  }
  const label = canonicalTypeLabel(typ);
  if (typeof value === "string" && typ.tags !== undefined) {
    const tags = [...typ.tags].sort(compareCodeUnits).map(textShown).join(", ");
    const shown = src || textShown(value);
    return `\`when: ${shown}\` is not a tag of \`on: ${on}\` (\`${label}\`); use one of: ${tags}`;
  }
  const shown = src || spelling(value);
  if (VALUE_KINDS.get(typeof value)!.includes(typ.kind)) {
    return `\`when: ${shown}\` is not a value of \`on: ${on}\` (\`${label}\`): ${error.message}`;
  }
  return `\`when: ${shown}\` is ${phrase} but \`on: ${on}\` is \`${label}\`${whenFix(value as WhenValue, typ, src)}`;
}

/**
 * Return the `; <fix>` suffix for a `when:` value of the wrong kind, or `""`.
 *
 * @param value The authored `when:` value.
 * @param typ The resolved `on:` type.
 * @param src The authored source text of the `when:` scalar, when known.
 * @returns A quote, write-as-an-int, write-the-boolean or drop-the-quotes hint, or `""` when none
 *   applies.
 */
function whenFix(value: WhenValue, typ: Type, src: string | null): string {
  if (typeof value !== "string") {
    // An anchor, alias or tag is not part of the value's spelling, so quoting the source would
    // change the text; the value's own spelling is quoted instead.
    const written = authoredOrSpelling(value, src);
    if (valueFits(typ, written)) return `; quote it: \`when: ${yamlQuoted(written)}\``;
    if (typeof value === "number" && Number.isInteger(value) && valueFits(typ, BigInt(value))) {
      return `; write it as an int: \`when: ${BigInt(value)}\``;
    }
    return "";
  }
  const former = formerBooleanFix(value, typ);
  if (former !== null) return `; ${former[1]}`;
  // Text written without quotes (`!!str 1`) has no quotes to drop.
  if (src !== null && !(src.startsWith("'") || src.startsWith('"'))) return "";
  const text = value.trim();
  const reading = yamlReading(text);
  if (reading !== null && valueFits(typ, reading)) return `; drop the quotes: \`when: ${text}\``;
  return "";
}

/**
 * Return the bool or number `text` reads as unquoted, else `null`.
 *
 * Only the values `on:` matching lowers as numbers or bools count, so a hint to drop the quotes
 * never suggests text that YAML still reads as text or `null`. An infinite number counts, since a
 * `when: .inf` lowers; the YAML reader refuses `.nan`.
 */
function yamlReading(text: string): boolean | bigint | number | null {
  let reading: unknown;
  try {
    reading = parseYamlLiteral(text);
  } catch (error) {
    if (error instanceof SourceSyntaxError) return null;
    throw error;
  }
  if (typeof reading === "boolean" || typeof reading === "bigint" || typeof reading === "number") return reading;
  return null;
}

/** Return `text` bare when YAML reads it back as the same text, else YAML-quoted. */
function textShown(text: string): string {
  let readsBack: boolean;
  try {
    readsBack = parseYamlLiteral(text) === text;
  } catch (error) {
    if (!(error instanceof SourceSyntaxError)) throw error;
    readsBack = false;
  }
  return readsBack ? text : yamlQuoted(text);
}

/** Return a `when:` value as YAML would spell it, for when its source text is unknown. */
function spelling(value: unknown): string {
  return typeof value === "string" ? textShown(value) : shownLiteral(value);
}

/**
 * Require a value-producing case to have an `else:` or to list every enum tag.
 *
 * @param desc The value-producing case.
 * @param producers The producer contracts the case sees.
 * @param inputs The flow-input contracts the case sees.
 * @param diagnostics Source ownership for located errors.
 * @throws LoadError when the case has no `else:` and its `on:` type is unknown, neither an enum
 *   nor a `bool`, nullable, or has a member no `when:` lists (or it is a searched case).
 */
function checkValueExhaustive(
  desc: CaseDescriptor,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  diagnostics: LoadDiagnostics,
): void {
  if (caseIsExhaustive(desc, producers, inputs)) return;
  throw diagnostics.error(
    `case node ${reprText(desc.id)} produces a value, so it must be exhaustive: ` +
      `add an \`else:\` (${notExhaustiveReason(desc, producers, inputs)})`,
    { range: desc.range },
  );
}

/** Return why a case that `caseIsExhaustive` rejected may match no branch. */
function notExhaustiveReason(
  desc: CaseDescriptor,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
): string {
  if (desc.on === null) return "a searched case needs an else:";
  const typ = resolveOnType(desc.on, producers, inputs);
  if (typ === null) return `the type of on: ${desc.on} is not known`;
  const covered = onCovered(desc);
  if (typ.kind === ValueKind.BOOLEAN && typ.tags === undefined) {
    if (typ.nullable) return `${desc.on} may be null`;
    const missing = (
      [
        ["true", true],
        ["false", false],
      ] as const
    )
      .filter(([, value]) => !covered.includes(value))
      .map(([name]) => name);
    return `bool value(s) [${missing.join(", ")}] not covered`;
  }
  if (typ.tags === undefined) return `on: ${desc.on} is ${canonicalTypeLabel(typ)}, not a Literal or bool`;
  if (typ.nullable) return `${desc.on} may be null`;
  const tags = coveredTags(covered);
  const missing = [...typ.tags].filter((tag) => !tags.has(tag)).sort(compareCodeUnits);
  return `enum tag(s) ${writeAsText(missing)} not covered`;
}

/**
 * Return whether a `case` always takes a branch when it runs.
 *
 * @param desc The `case` descriptor.
 * @param producers Each producer node's output contract by id, as the case sees them.
 * @param flowInputs The flow-input contracts by name, as the case sees them. Without them (`null`,
 *   the default) an `on:` over a flow input is never exhaustive.
 * @returns `true` when the case has an `else:`, or it is the `on:` form over a non-nullable
 *   `Literal` or `bool` node output or flow input whose every member is a `when:` value. A
 *   searched case without `else:` is never exhaustive.
 */
export function caseIsExhaustive(
  desc: CaseDescriptor,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract> | null = null,
): boolean {
  if (desc.else_ !== null) return true;
  if (desc.on === null) return false;
  const typ = resolveOnType(desc.on, producers, flowInputs ?? new Map());
  return typ !== null && covers(typ, onCovered(desc));
}

/**
 * Return whether the typed `when:` values `covered` list every member of `typ`.
 *
 * @param typ The resolved `on:` type.
 * @param covered The case's typed `when:` values.
 * @returns `true` when `typ` is non-nullable and is a `Literal` whose every tag is covered, or a
 *   `bool` with both `true` and `false` covered.
 */
function covers(typ: Type, covered: readonly unknown[]): boolean {
  if (typ.nullable) return false;
  if (typ.tags !== undefined) {
    const tags = coveredTags(covered);
    return [...typ.tags].every((tag) => tags.has(tag));
  }
  return typ.kind === ValueKind.BOOLEAN && covered.includes(true) && covered.includes(false);
}

/**
 * Return the ids of the CASE descriptors that always take a branch when they run.
 *
 * @param descriptors The level's node descriptors by id. Non-CASE descriptors are ignored.
 * @param producers Each producer node's output contract by id.
 * @param flowInputs The flow-input contracts by name (`null`, the default: none).
 * @param options.narrowing Per-node narrowed views; each CASE resolves its `on:` against its own
 *   view.
 * @returns The exhaustive CASE ids (see `caseIsExhaustive`).
 */
export function exhaustiveCases(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract> | null = null,
  options: { narrowing?: Narrowing | null } = {},
): Set<string> {
  const narrowing = options.narrowing ?? null;
  const exhaustive = new Set<string>();
  for (const desc of descriptors.values()) {
    if (!(desc instanceof CaseDescriptor)) continue;
    const [caseProducers, caseInputs] = narrowing !== null ? narrowing.forNode(desc.id) : [producers, flowInputs];
    if (caseIsExhaustive(desc, caseProducers, caseInputs)) exhaustive.add(desc.id);
  }
  return exhaustive;
}

/**
 * Return the typed values of an `on:`-form case's `when:`s (empty when searched).
 *
 * A list, not a set: `1n`, `1` and `true` are distinct `when:` values, and a list keeps each.
 */
function onCovered(desc: CaseDescriptor): unknown[] {
  if (desc.on === null) return [];
  return desc.cases.map((spec) => spec["when"]).filter((value) => value !== undefined && value !== null);
}

/** Return the text values among typed `when:` values, the only ones a tag can equal. */
function coveredTags(covered: readonly unknown[]): Set<string> {
  return new Set(covered.filter((value): value is string => typeof value === "string"));
}

/**
 * Return the `Type` an `on:` ref resolves to, else `null`.
 *
 * Resolves the binding through the shared output-contract projector, so an exact contract or a
 * known field of a partial record can supply the enum `Type`.
 *
 * @param onRef The case's `on:` template.
 * @param producers Each producer node's output contract by id.
 * @param flowInputs The flow-input contracts by name. With them an `${input.<name>[.<field>...]}`
 *   ref resolves too; without them (`null`, the default) only `${<id>.output[.<field>...]}` does.
 * @returns The resolved type, or `null` for an unsupported head, an opaque result, or a non-record
 *   step (lenient: no check).
 */
function resolveOnType(
  onRef: string,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract> | null = null,
): Type | null {
  let refs: string[];
  try {
    refs = exprRefsOf(onRef);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
  if (refs.length !== 1) return null;
  const parts = refs[0]!.split(".");
  const nodeOutput = parts.length >= 2 && parts[1] === "output";
  const flowInput = flowInputs !== null && parts.length >= 2 && parts[0] === "input";
  if (!(nodeOutput || flowInput)) return null;
  const contract = resolveOutputBindingContract(onRef, producers, flowInputs ?? new Map());
  return isExactly(contract, ValueContract) ? contract.type : null;
}

/**
 * Check enum exhaustiveness for the `on:` form.
 *
 * Resolves the `on:` ref to a `Type`; if it carries tags (an enum node output or flow input), every
 * tag must be a text `when:` value or be covered by a present `else:`. A non-enum or unresolved
 * `on:` is lenient (no check).
 *
 * @param desc The case descriptor.
 * @param covered The case's typed `when:` values.
 * @param producers The producer contracts the case sees.
 * @param inputs The flow-input contracts the case sees.
 * @param diagnostics Source ownership for located errors.
 * @throws LoadError when an enum tag is not covered and the case has no `else:`, located at the
 *   case.
 */
function checkExhaustive(
  desc: CaseDescriptor,
  covered: readonly unknown[],
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  diagnostics: LoadDiagnostics,
): void {
  // A searched case, or an `else:` that satisfies coverage.
  if (desc.on === null || desc.else_ !== null) return;
  const typ = resolveOnType(desc.on, producers, inputs);
  // Not a checked enum: lenient.
  if (typ === null || typ.tags === undefined) return;
  const tags = coveredTags(covered);
  const missing = [...typ.tags].filter((tag) => !tags.has(tag)).sort(compareCodeUnits);
  if (missing.length > 0) {
    throw diagnostics.error(
      `case node ${reprText(desc.id)} on ${desc.on}: non-exhaustive — enum tag(s) ` +
        `${writeAsText(missing)} not covered by a case \`when:\` or an \`else:\``,
      { range: desc.range },
    );
  }
}
