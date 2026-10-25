/**
 * Graph-stage: the type-level half of a `loop`'s `'a -> 'a` contract.
 *
 * A loader post-build pass (run after `checkCallMapTypes`). `LoopDescriptor.build` checked only
 * the carried field-name sets: it lacks the resolved carried-record types, which depend on the
 * parent producers and flow inputs assembled by the parent build. This pass resolves each returned
 * carried field's contract from its bound seed source (`${<id>.output}`/`${input.X}` over the
 * parent producer and flow-input contracts, via `resolveOutputBindingContract`), takes its type,
 * and checks it with `typesCompatible` against the body output's same-named field type. A field
 * the body omits carries over unchanged, so it has nothing to compare. The node's canonical output
 * contract already keeps independently proven body fields when no sound exact aggregate exists.
 * Its range anchor is `BuildResult.fieldRanges` at `siteKey("field", "inputs")` (the coarse
 * `input:` mapping span).
 *
 * Each proven returned field must also fit the body parameter it is bound to on the next iteration
 * (a named body's declared input, or the parameter an inline body generates from its seed), read
 * as nullable where the loop's `while:` proves the field not null; that error points at the seed
 * binding. A literal seed stays lenient here (its contract is not inferred); `checkCallMapTypes`
 * checks every seed against the body's declared input.
 *
 * Knows about: `compile/validation` (the type comparator), `compose/helpers` (shared
 * output-contract inference, tag advice and load diagnostics), `compose/nodes/loop` (a body's
 * inputs as the carried values enter them, and the carried field types) and
 * `compose/validate/typed_bindings` (the text form of a literal).
 * Never imports: the loader or the compose package facade; only they import this back.
 */

import { typesCompatible } from "../../compile/validation.ts";
import { reprText, type SourceRange } from "../../diagnostics.ts";
import {
  Binding,
  bindingSource,
  isExactly,
  type OutputContract,
  PartialRecordContract,
  ValueContract,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import type { LoopNode } from "../../nodes/loop/node.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { canonicalTypeLabel, isList, isRecord, jsTypeName, type Type, ValueKind } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { constructorTypeFn } from "../helpers/isinstance_types.ts";
import { assertExactContracts, resolveOutputBindingContract } from "../../compile/output_contracts.ts";
import { anySourceHint, suggestTag, tagText } from "../helpers/tags.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import { carriedFieldTypes, loopEntryInputs } from "../nodes/loop.ts";
import { shownLiteral } from "./typed_bindings.ts";

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
 * over unchanged) is skipped. A literal, coalesce or opaque seed source, or an opaque body output
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
    if (!result.node.isLoop) continue;
    // A loop is recognised by `isLoop`, as Python does, so a hand-built loop is checked too.
    const node = result.node as LoopNode;
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
          `or None, got ${jsTypeName(loopContract)}`,
      );
    }
    const carriedFields = carriedFieldTypes(loopContract);
    // A non-record or absent carried contract was already caught by the field-name half in
    // `LoopDescriptor.build`. With no independently proven fields, nothing remains to compare.
    if (carriedFields.size === 0) continue;
    const returned = node.returnedFields;
    const wiring = flowWiring.get(nodeId) ?? new Map<string, unknown>();
    for (const [fieldName, src] of wiring) {
      if (returned !== null && !returned.includes(fieldName)) continue; // omitted: carried over
      // A returned field's carried type is the body's own field type.
      const bodyField = carriedFields.get(fieldName);
      if (bodyField === undefined) continue; // name mismatch caught in build
      const sourceContract = resolveOutputBindingContract(src, producers, flowInputs, { typeFn });
      if (sourceContract === null) continue; // literal, coalesce or opaque seed source: lenient
      if (isExactly(sourceContract, PartialRecordContract)) continue; // no exact type to compare
      if (!isExactly(sourceContract, ValueContract)) {
        throw new TypeError(
          "resolved loop source contract must be ValueContract, PartialRecordContract, or None",
        );
      }
      const source = sourceContract.type;
      if (!typesCompatible(source, bodyField)) {
        const range: SourceRange | null = result.fieldRanges.get(siteKey("field", "inputs")) ?? result.nodeRange;
        let message =
          `loop node ${reprText(nodeId)}: carried field ${reprText(fieldName)} — body output is ` +
          `${reprText(canonicalTypeLabel(bodyField))}, carried source is ` +
          `${reprText(canonicalTypeLabel(source))} (a returned field must keep its carried type)`;
        const hint = anySourceHint(fieldName, source, bodyField, { registry });
        if (hint !== null) message += `; ${hint}`;
        throw diagnostics.error(message, { range });
      }
    }
    checkReturnedFields(result, carriedFields, wiring, registry, diagnostics, { inline: inlineLoops.has(nodeId) });
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
 * @param result The built loop node and its source ranges.
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
  result: BuildResult,
  carriedFields: ReadonlyMap<string, Type>,
  wiring: ReadonlyMap<string, unknown>,
  registry: TypeRegistry | null,
  diagnostics: LoadDiagnostics,
  options: { inline: boolean },
): void {
  const node = result.node as LoopNode;
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
        "; declare the seed's type: " +
        suggestTag(fieldName, bodyField, shownSeed, { registry });
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

/**
 * Render a seed value as the right side of a typed binding.
 *
 * @param value The seed's binding value: template or directive text, or a literal.
 * @returns Template and directive text as written; a literal or a value template in its text form
 *   (a nested binding shown by its value).
 */
function seedText(value: unknown): string {
  if (typeof value === "string" && (value.includes("${") || value.startsWith("call("))) return value;
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
