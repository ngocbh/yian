/**
 * Graph stage: the type-level half of a `loop`'s `'a -> 'a` contract (`checkLoopTypeContract`).
 *
 * `LoopDescriptor.build` checks only the carried field names, since the carried types depend on
 * the parent's producers and flow inputs. Once those are assembled, this pass checks that each
 * returned field keeps its carried type and fits the body parameter it is bound to next.
 *
 * Knows about: `compile/validation` (the type comparator), `compose/helpers` (shared
 * output-contract inference, tag advice, load diagnostics and the text form of a literal) and
 * `compose/nodes/loop` (a body's inputs as the carried values enter them, and the carried field
 * types).
 * Never imports: the loader or the compose package facade; only they import this back.
 */

import { typesCompatible } from "../../compile/validation.ts";
import { reprText, type SourceRange } from "../../diagnostics.ts";
import { conditionRefs, evaluateWhenRecord, ExpressionError } from "../../expr/expressions.ts";
import {
  Binding,
  bindingSource,
  isExactly,
  type OutputContract,
  PartialRecordContract,
  ValueContract,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { LoopNode } from "../../nodes/loop/node.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isList,
  isRecord,
  jsTypeName,
  setField,
  type Type,
  TypeCheckError,
  type Value,
  ValueKind,
  type ValueRecord,
} from "../../typesys/values.ts";
import { LoadDiagnostics, shownLiteral } from "../helpers/errors.ts";
import {
  collectConstructorNames,
  collectIsinstanceNames,
  constructorTypeFn,
  resolveConstructorType,
  resolveIsinstanceType,
} from "../helpers/isinstance_types.ts";
import {
  assertExactContracts,
  literalContract,
  literalValue,
  NOT_LITERAL,
  resolveOutputBindingContract,
} from "../../compile/output_contracts.ts";
import { anySourceHint, namedTypeLabel, suggestTag, tagText } from "../helpers/tags.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import { carriedFieldTypes, loopEntryInputs } from "../nodes/loop.ts";

/**
 * The type-level half of a `loop`'s `'a -> 'a` contract (a loader post-build pass).
 *
 * Run after `checkCallMapTypes`, it resolves each returned carried field's type from its bound
 * seed source (over the parent producer and flow-input contracts, as `checkCallMapTypes` does via
 * `resolveOutputBindingContract`) and checks it with `typesCompatible` against the body output's
 * same-named field type, read from the node's carried-record `outputContract` (the body's own
 * contract for a total loop, the merged returned-plus-seeded contract for a partial one). An exact
 * record gives those fields from its type; a partly typed record gives its independently proven
 * field contracts. A field outside the node's `returnedFields` (omitted by the body, so carried
 * over unchanged) is skipped. A literal seed is typed by its value, exactly, when the loop's
 * `while:` may be false on its seeds: the body then runs zero times and the loop outputs the
 * seeds. Under a `while:` true on its literal seeds, or `until:` or `times:`, the body runs at
 * least once, so a literal seed, like a coalesce or opaque seed source or an opaque body output
 * field, stays lenient (skipped); `checkCallMapTypes` checks every seed against the body's
 * declared input.
 *
 * Each returned field is bound to the body's parameter again on the next iteration (a named body's
 * declared input, or the parameter an inline body generates from its seed), so every statically
 * proven returned field must also fit that parameter, read as nullable where the loop's `while:`
 * proves the field not null. Over a typed seed, a nullable returned field whose non-null type fits
 * is left to the runtime input check.
 *
 * @param buildResults The built nodes keyed by node id.
 * @param producers Each producer node's output contract.
 * @param flowInputs The flow's declared input contracts.
 * @param flowWiring Each node's `param -> source` bindings.
 * @param options.inlineLoops The ids of loop nodes whose body is written inline; it selects the
 *   wording of the error for a returned field that does not fit its parameter (default none).
 * @param options.registry The flow's typedefs, used to spell a suggested seed tag by a typedef
 *   name and to type a `T(...)` constructor seed as `T`.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError when a returned field does not keep its carried type (with advice to tag the
 *   seed or narrow it with `isinstance` when it is `Any`), or the body returns a field that can't
 *   fit the parameter it is bound to next; located at the loop node's `input:` mapping range (else
 *   its node range), the second at the seed binding when it has a range.
 * @throws TypeError when a producer, flow input, or loop contract has an unexpected class.
 */
export function checkLoopTypeContract(
  buildResults: ReadonlyMap<string, BuildResult>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  options: {
    inlineLoops?: ReadonlySet<string>;
    registry?: TypeRegistry | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const { inlineLoops = new Set<string>(), registry = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const typeFn = registry !== null ? constructorTypeFn(registry) : null;
  assertExactContracts(producers, flowInputs);
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    if (!(node instanceof LoopNode)) continue;
    // The node's carried-record contract: the body's own contract for a total loop, or the merged
    // returned-plus-seeded contract for a partial loop.
    const loopContract = node.outputContract;
    if (
      loopContract !== null &&
      !isExactly(loopContract, ValueContract) &&
      !isExactly(loopContract, PartialRecordContract)
    ) {
      throw new TypeError(
        `node ${reprText(nodeId)} output_contract must be an exact ValueContract, PartialRecordContract, ` +
          `or null, got ${jsTypeName(loopContract)}`,
      );
    }
    const carriedFields = carriedFieldTypes(loopContract);
    // A non-record or absent carried contract was already caught by the field-name half in
    // `LoopDescriptor.build`. With no independently proven fields, nothing remains to compare.
    if (carriedFields.size === 0) continue;
    const returned = node.returnedFields;
    const wiring = flowWiring.get(nodeId) ?? new Map<string, unknown>();
    // A `while:` false on the seeds runs the body zero times, and the loop outputs its seeds; any
    // other loop runs the body at least once, so a literal seed never reaches its output.
    const literalsOut =
      node.predicateKind === "while" &&
      typeof node.predicate === "string" &&
      !holdsOnLiteralSeeds(node.predicate, wiring, registry ?? new Map());
    for (const [fieldName, src] of wiring) {
      if (returned !== null && !returned.includes(fieldName)) continue; // omitted: carried over
      // A returned field's carried type is the body's own field type.
      const bodyField = carriedFields.get(fieldName);
      if (bodyField === undefined) continue; // name mismatch caught in build
      let sourceContract = resolveOutputBindingContract(src, producers, flowInputs, { typeFn });
      const literal = sourceContract === null && literalsOut ? literalValue(bindingSource(src)) : NOT_LITERAL;
      // Each literal keeps its exact type, so a text seed fits a `Literal` field that names it.
      if (literal !== NOT_LITERAL) sourceContract = literalContract(literal, { exactLiterals: true });
      if (sourceContract === null) continue; // coalesce or opaque seed source, or a literal never output: lenient
      if (isExactly(sourceContract, PartialRecordContract)) continue; // no exact type to compare
      if (!isExactly(sourceContract, ValueContract)) {
        throw new TypeError(
          "resolved loop source contract must be ValueContract, PartialRecordContract, or null",
        );
      }
      const source = sourceContract.type;
      if (!typesCompatible(source, bodyField)) {
        const range: SourceRange | null = result.fieldRanges.get(siteKey("field", "inputs")) ?? result.nodeRange;
        const sourceText = literal === NOT_LITERAL ? "carried source is" : "literal seed is";
        const reason =
          literal === NOT_LITERAL
            ? "a returned field must keep its carried type"
            : "a `while:` false at the start outputs the seeds; a returned field must keep its carried type";
        let message =
          `loop node ${reprText(nodeId)}: carried field ${reprText(fieldName)} — body output is ` +
          `${reprText(canonicalTypeLabel(bodyField))}, ${sourceText} ` +
          `${reprText(canonicalTypeLabel(source))} (${reason})`;
        const hint = anySourceHint(fieldName, source, bodyField, { registry });
        if (hint !== null) message += `; ${hint}`;
        throw diagnostics.error(message, { range });
      }
    }
    checkReturnedFields(node, result, carriedFields, wiring, registry, diagnostics, {
      inline: inlineLoops.has(nodeId),
    });
  }
}

/**
 * Say whether a `while:` predicate is true on the loop's literal seeds, so the body runs at least
 * once. `false` when the predicate reads a field whose seed isn't a literal, or fails to evaluate.
 *
 * @param predicate The `while:` text.
 * @param wiring The loop's seed bindings keyed by carried field.
 * @param registry The flow's typedefs, which the predicate's `isinstance` and constructor names
 *   resolve against.
 */
function holdsOnLiteralSeeds(predicate: string, wiring: ReadonlyMap<string, unknown>, registry: TypeRegistry): boolean {
  const seeds: ValueRecord = {};
  for (const [name, src] of wiring) {
    const value = literalValue(bindingSource(src));
    // A literal seed binds a yian value as written.
    if (value !== NOT_LITERAL) setField(seeds, name, value as Value);
  }
  try {
    if (!conditionRefs(predicate).every((ref) => Object.hasOwn(seeds, ref.split(".", 1)[0]!))) return false;
    // The node's own type table is filled later in the load.
    const types = new Map<string, Type>();
    for (const name of collectIsinstanceNames(predicate, { condition: true })) {
      types.set(name, resolveIsinstanceType(name, registry));
    }
    for (const name of collectConstructorNames(predicate, { condition: true })) {
      types.set(name, resolveConstructorType(name, registry));
    }
    return evaluateWhenRecord(predicate, seeds, types);
  } catch (error) {
    if (error instanceof ExpressionError || error instanceof TypeCheckError) return false;
    throw error;
  }
}

/**
 * Require each returned field of a loop body to fit the parameter it is bound to next.
 *
 * The parameter is the body's declared input for a named body, or the one generated from the seed
 * for an inline body. A path the loop's `while:` proves not null is never null when the body
 * starts, so the parameter is read as nullable there and at each record above it (see
 * `loopEntryInputs`); the error still names the declared type. Otherwise, over a typed seed, a
 * nullable returned field whose non-null type fits the parameter is accepted; a null it returns
 * fails the next iteration's input check.
 *
 * @param node The loop node.
 * @param result The loop's build, for its source ranges.
 * @param carriedFields The statically proven carried field types; a returned field's entry is the
 *   body's own field type.
 * @param wiring The loop's seed bindings keyed by carried field.
 * @param registry The flow's typedefs, for the suggested types and tags.
 * @param diagnostics Factory for located load errors.
 * @param options.inline Whether the body is written inline, which selects the wording.
 * @throws LoadError when a returned field's type can't fit its parameter; the error names both
 *   types and says how to make them agree.
 */
function checkReturnedFields(
  node: LoopNode,
  result: BuildResult,
  carriedFields: ReadonlyMap<string, Type>,
  wiring: ReadonlyMap<string, unknown>,
  registry: TypeRegistry | null,
  diagnostics: LoadDiagnostics,
  options: { inline: boolean },
): void {
  const returned = node.returnedFields;
  const parameters = new Map<string, ValueContract>();
  for (const parameter of node.childInputs) {
    if (parameter.contract !== null) parameters.set(parameter.name, parameter.contract);
  }
  // The comparison reads the inputs as values enter the body; messages name the declared input,
  // which the body's start checks values against.
  const entryTypes = new Map<string, Type>();
  for (const parameter of loopEntryInputs(node)) {
    if (parameter.contract !== null) entryTypes.set(parameter.name, parameter.contract.type);
  }
  for (const [fieldName, seed] of wiring) {
    if (returned !== null && !returned.includes(fieldName)) continue; // never bound again
    const bodyField = carriedFields.get(fieldName);
    const parameter = parameters.get(fieldName);
    // Opaque on either side, or not a body input: nothing to compare.
    if (bodyField === undefined || parameter === undefined) continue;
    const entryType = entryTypes.get(fieldName)!;
    if (typesCompatible(bodyField, entryType)) continue;
    const typed = seed instanceof Binding && seed.contract !== null;
    const nonNullFits = typesCompatible(bodyField.replace({ nullable: false }), entryType);
    // The next iteration's input check rejects a null.
    if (typed && nonNullFits) continue;
    const returnedLabel = canonicalTypeLabel(bodyField);
    const parameterLabel = canonicalTypeLabel(parameter.type);
    const shownSeed = seedText(bindingSource(seed));
    let message: string;
    if (options.inline) {
      message =
        `loop node ${reprText(node.id)}: the body returns ${reprText(fieldName)} as ${returnedLabel}, ` +
        "but the carried field is " +
        (typed ? `declared ${parameterLabel}` : `typed ${parameterLabel} from its seed`) +
        "; " +
        inlineSeedAdvice(fieldName, bodyField, bindingSource(seed), shownSeed, registry);
    } else {
      message =
        `loop node ${reprText(node.id)}: the body ${reprText(node.flowId)} returns ${reprText(fieldName)} ` +
        `as ${returnedLabel}, but takes ${reprText(fieldName)} as ${parameterLabel}; `;
      if (nonNullFits) {
        // Only an untyped seed gets here, so tagging it is a real alternative.
        const nullable = parameter.type.replace({ nullable: true });
        const nullableLabel = tagText(nullable, { registry }) ?? canonicalTypeLabel(nullable);
        message +=
          `make the input ${nullableLabel}, or tag the seed to leave the check to runtime: ` +
          suggestTag(fieldName, parameter.type, shownSeed, { registry });
      } else if (bodyField.kind === ValueKind.ANY) {
        message += "narrow it in the body with isinstance in a case, or declare its producer's type";
      } else {
        message += "change the body's input or output type so they agree";
      }
    }
    throw diagnostics.error(message, {
      range:
        result.fieldRanges.get(siteKey("input", fieldName)) ??
        result.fieldRanges.get(siteKey("field", "inputs")) ??
        result.nodeRange,
    });
  }
}

/** Values a seed may be rewritten to, as written and as bound, tried in order. */
const EXAMPLE_SEEDS: readonly (readonly [string, unknown])[] = [
  ["0", 0n],
  ["false", false],
  ['""', ""],
  ["[]", []],
  ["{}", {}],
];

/**
 * Advise how to seed an inline loop's carried field so the body's returned type fits it.
 *
 * A tag on the seed only helps when load accepts the tagged seed: a template, or a literal the
 * body's type holds. Any other literal can never hold the body's type, so the advice names the
 * type instead and shows a seed of it that load accepts, when one is short to write.
 *
 * @param fieldName The carried field.
 * @param bodyField The type the body returns for the field.
 * @param source The seed's raw binding source.
 * @param shownSeed The seed as `seedText` renders it.
 * @param registry The flow's typedefs, to name the type and spell its tag.
 * @returns The advice clause of the load error.
 */
function inlineSeedAdvice(
  fieldName: string,
  bodyField: Type,
  source: unknown,
  shownSeed: string,
  registry: TypeRegistry | null,
): string {
  const literal = literalValue(source);
  if (literal === NOT_LITERAL || holds(bodyField, literal)) {
    return "declare the seed's type: " + suggestTag(fieldName, bodyField, shownSeed, { registry });
  }
  const label = namedTypeLabel(bodyField.replace({ nullable: false }), registry ?? new Map());
  let advice = `the seed ${shownSeed} is not of type ${label}: seed one that is`;
  // Without a tag spelling, `suggestTag` gives naming advice rather than a seed to show.
  const example =
    tagText(bodyField, { registry }) === null
      ? undefined
      : EXAMPLE_SEEDS.find(([, value]) => holds(bodyField, value));
  if (example !== undefined) advice += ", e.g. " + suggestTag(fieldName, bodyField, example[0], { registry });
  return advice;
}

/** Whether load accepts `value` written under a tag for `typ`, as it checks a tagged literal. */
function holds(typ: Type, value: unknown): boolean {
  try {
    buildValueAs(typ, value);
    return true;
  } catch (error) {
    if (error instanceof TypeCheckError) return false;
    throw error;
  }
}

/** Whether a seed's binding value is template or directive text rather than a literal. */
function isTemplateText(value: unknown): value is string {
  return typeof value === "string" && (value.includes("${") || value.startsWith("call("));
}

/**
 * Render a seed value as the right side of a typed binding.
 *
 * @param value The seed's binding value: template or directive text, or a literal.
 * @returns Template and directive text as written; a literal or a value template in its text form
 *   (a nested binding shown by its value).
 */
function seedText(value: unknown): string {
  if (isTemplateText(value)) return value;
  // The text form is YAML flow style, and a quoted string stays literal text.
  return shownLiteral(value instanceof ValueTemplate ? untypedTree(value.value) : value);
}

/** Return a template tree with each nested binding replaced by its value. */
function untypedTree(node: unknown): unknown {
  if (node instanceof Binding) return untypedTree(node.source);
  if (isRecord(node)) return Object.fromEntries(Object.entries(node).map(([key, item]) => [key, untypedTree(item)]));
  if (isList(node)) return node.map(untypedTree);
  return node;
}
