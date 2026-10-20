/**
 * Graph-stage: the cross-flow type check for `call`/`map` bindings and `loop` seeds.
 *
 * A loader post-build pass (run after the flow's leaves and child signatures are assembled). Each
 * binding's source contract is resolved from its reference over the parent producer and
 * flow-input contracts (a MAP's `${item...}` from the element type of `over:`), then its type is
 * checked with `typesCompatible` against the child input's declared type (a LOOP's body input for
 * a seed, read as nullable where the LOOP's `while:` proves the seed not null, since that seed
 * never enters the body as null). A literal, a constant operand of a `:-`/`|` escape, or each
 * element of a literal `over:` list read through `${item...}`, is validated by value instead, as
 * the child's start validates it: a CALL/MAP/LOOP edge is not a flow boundary, so text is never
 * converted to a number or bool across it. Its range anchor is the call, map, or loop node's
 * transient Compose `BuildResult.fieldRanges` entry for `siteKey("field", "inputs")` (the coarse
 * `input:` mapping span).
 *
 * Knows about: `compile/validation` (the type comparator), `expr` (the binding-null analysis and
 * escape parsing), `compose/helpers` (shared output-contract inference, the `Any` source hint and
 * load diagnostics), `compose/nodes/loop` (a LOOP body's inputs as its seeds enter them, and its
 * carried field types), and `compose/validate/typed_bindings` (literal and `${item}` helpers).
 * Never imports: the loader or the compose package facade; only they import this back.
 */

import { typesCompatible } from "../../compile/validation.ts";
import { reprText, type SourceRange } from "../../diagnostics.ts";
import { type ExprNode, Token, Tree } from "../../expr/grammar.ts";
import { ExpressionError, evalExpr, exprRefs } from "../../expr/index.ts";
import { bindingCoSkips, scanTemplate, Span } from "../../expr/template.ts";
import {
  Binding,
  bindingSource,
  isExactly,
  type OutputContract,
  Parameter,
  PartialRecordContract,
  ValueContract,
} from "../../nodes/contracts.ts";
import type { LoopNode } from "../../nodes/loop/node.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isList,
  isRecord,
  jsTypeName,
  TypeCheckError,
} from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import {
  directReferenceParts,
  literalValue,
  NOT_LITERAL,
  projectOutputContract,
  referenceContract,
  resolveOutputBindingContract,
  type TypeFn,
} from "../helpers/output_contracts.ts";
import { anySourceHint } from "../helpers/tags.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import { carriedFieldTypes, loopEntryInputs } from "../nodes/loop.ts";
import type { Narrowing } from "./narrowing.ts";
import { itemContract, shownLiteral } from "./typed_bindings.ts";

/**
 * Cross-flow type check for every `call`/`map` binding and `loop` seed (a loader pass).
 *
 * Each binding must fit the child input's declared type, read over the parent producer and
 * flow-input contracts as the node's `narrowing` view sees them when given: a reference by its
 * resolved contract under `typesCompatible` (a lone `${item...}` through the element type of a
 * proven `over:` list), a literal by value, and a `${ref :- fallback}`/`${ref :? message}` by its
 * de-nulled reference and its fallback. An unproven source, or an opaque or absent child input
 * (the reserved `over` key among them), stays lenient (skipped). Iterates the wiring (not the
 * params) so a mis-named binding still surfaces here when its child decl is absent rather than
 * being silently skipped. A LOOP's seeds are checked the same way against its body's inputs, read
 * as nullable at each path its `while:` proves not null (see `loopEntryInputs`); the error still
 * names the declared input. A seed only that relaxed reading admits must also fit the loop's
 * output, which the seeds are when the body never runs (see `breaksZeroIterationOutput`).
 *
 * @param buildResults Built nodes by id, with their transient ranges and labels.
 * @param producers Exact or partial node output contracts keyed by producer id.
 * @param flowInputs Exact flow input contracts keyed by input name.
 * @param flowWiring Final binding maps keyed by node id and parameter name.
 * @param options.narrowing The union narrowing each node reads against; `null` (the default)
 *   reads every node against the unchanged maps.
 * @param options.typeFn Types a `T(...)` constructor source as `T`.
 * @param options.registry The flow's typedefs, which spell the tag suggested for an `Any` source
 *   by a typedef name.
 * @param options.diagnostics Factory for located load errors.
 * @throws TypeError when a producer contract is neither an exact `ValueContract` nor a
 *   `PartialRecordContract`, or a flow-input contract is not an exact `ValueContract`.
 * @throws LoadError when a binding can't fill its child input, located at the node's
 *   Compose-owned `input:` mapping range, else its authored node range.
 */
export function checkCallMapTypes(
  buildResults: ReadonlyMap<string, BuildResult>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  options: {
    narrowing?: Narrowing | null;
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const { narrowing = null, typeFn = null, registry = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  for (const [name, contract] of producers) {
    if (!isExactly(contract, ValueContract) && !isExactly(contract, PartialRecordContract)) {
      throw new TypeError(
        "output producer contracts must be ValueContract or PartialRecordContract values, " +
          `got ${jsTypeName(contract)} for ${reprText(name)}`,
      );
    }
  }
  for (const [name, contract] of flowInputs) {
    if (!isExactly(contract, ValueContract)) {
      throw new TypeError(
        `flow input contracts must be exact ValueContract values, got ${jsTypeName(contract)} for ${reprText(name)}`,
      );
    }
  }
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    // CALL/MAP/LOOP carry `childInputs` to type-check against. A LOOP's seeds fill its body's
    // START like CALL arguments, so they get the same checks, except that a seed path its
    // `while:` proves not null never reaches the body when null, so the body input reads as
    // nullable there; `checkLoopTypeContract` also checks returned fields against the body
    // output. A leaf has no `childInputs`, and a zero-input child has none either, so skip.
    const childInputs = (node as { childInputs?: readonly Parameter[] }).childInputs;
    if (childInputs === undefined || childInputs.length === 0) continue;
    const kind = node.isLoop ? "loop" : "call";
    const childDecls = new Map(childInputs.map((decl) => [decl.name, decl]));
    // A loop is recognised by `isLoop`, as Python does, so a hand-built loop gets the relaxed
    // reading too.
    const entryDecls = node.isLoop
      ? new Map(loopEntryInputs(node as LoopNode).map((decl) => [decl.name, decl]))
      : childDecls;
    const [nodeProducers, nodeInputs] = narrowing !== null ? narrowing.forNode(nodeId) : [producers, flowInputs];
    const wiring = flowWiring.get(nodeId) ?? new Map<string, unknown>();
    const scope: Scope = {
      producers: nodeProducers,
      flowInputs: nodeInputs,
      item: itemContract(result, wiring, nodeProducers, nodeInputs),
      items: literalItems(result, wiring),
      typeFn,
      registry,
    };
    for (const [param, src] of wiring) {
      const decl = childDecls.get(param);
      // `over`, a name caught in build, or an opaque child input: lenient.
      if (decl === undefined || decl.contract === null) continue;
      const entry = entryDecls.get(param)!;
      let message = bindingMismatch(src, entry, scope);
      if (message !== null && entry !== decl) {
        // Name the declared input, which the body's start checks values against.
        message = bindingMismatch(src, decl, scope) ?? message;
      } else if (message === null && entry !== decl) {
        // When the `while:` is false on the seeds the body never runs and the loop outputs the
        // seeds as they are, so a seed only the `while:` admits must also fit the loop's output;
        // otherwise it is rejected as if there were no `while:`.
        const strict = bindingMismatch(src, decl, scope);
        if (
          strict !== null &&
          breaksZeroIterationOutput(node as LoopNode, src, decl, scope, producers, flowInputs)
        ) {
          message = strict;
        }
      }
      if (message !== null) {
        const range: SourceRange | null = result.fieldRanges.get(siteKey("field", "inputs")) ?? result.nodeRange;
        // An inline call's argument has a label naming it at its authored site.
        const subject =
          result.fieldLabels.get(`input ${reprText(param)}`) ||
          `${kind} node ${reprText(nodeId)}: binding ${reprText(param)}`;
        throw diagnostics.error(`${subject} — ${message}`, { range });
      }
    }
  }
}

/**
 * Return whether a loop seed must be rejected because the body may never run.
 *
 * With no iteration the loop outputs its seeds, so a seed for a field the body returns must fit
 * that field's carried type, the body's own; a seed it can't be shown to fit is rejected. A field
 * the body never returns is output as its seed whatever the iteration count, so its seed is never
 * rejected here. A seed whose resolved source type does not fit is left to
 * `checkLoopTypeContract`, which rejects the same pair next and names the carried field.
 *
 * @param node The built loop node.
 * @param src The seed binding, raw or a `Binding`.
 * @param decl The body input the seed fills, with a declared contract.
 * @param scope What the node's bindings read.
 * @param producers Node output contracts by producer id, as `checkLoopTypeContract` reads them.
 * @param flowInputs Flow input contracts by name, as `checkLoopTypeContract` reads them.
 * @returns Whether the seed must be rejected with its strict seed error.
 */
function breaksZeroIterationOutput(
  node: LoopNode,
  src: unknown,
  decl: Parameter,
  scope: Scope,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
): boolean {
  const returned = node.returnedFields;
  if (returned !== null && !returned.includes(decl.name)) return false;
  const carried = carriedFieldTypes(node.outputContract).get(decl.name);
  if (carried === undefined) return true;
  const source = resolveOutputBindingContract(src, producers, flowInputs, { typeFn: scope.typeFn });
  if (isExactly(source, ValueContract) && !typesCompatible(source.type, carried)) {
    // Its error says the returned field must keep its carried type, which is why this seed
    // fails; the seed error would blame the body input the `while:` guards.
    return false;
  }
  const carriedDecl = new Parameter(decl.name, {
    contract: new ValueContract(carried, decl.contract!.origins),
    required: decl.required,
    hasDefault: decl.hasDefault,
    default: decl.default,
    coerceStrings: decl.coerceStrings,
    origin: decl.origin,
    defaultOrigin: decl.defaultOrigin,
  });
  return bindingMismatch(src, carriedDecl, scope) !== null;
}

/** What a CALL/MAP/LOOP node's bindings read, for the load check. */
type Scope = Readonly<{
  /** Node output contracts by producer id. */
  producers: ReadonlyMap<string, OutputContract>;
  /** Flow input contracts by name. */
  flowInputs: ReadonlyMap<string, ValueContract>;
  /** The element contract `${item}` reads on a MAP over a proven list, else `null`. */
  item: ValueContract | null;
  /** The elements of a MAP's literal `over:` list, which `${item}` reads by value, else `null`. */
  items: readonly unknown[] | null;
  /** Types a `T(...)` constructor source as `T`. */
  typeFn: TypeFn | null;
  /** The flow's typedefs, which spell a suggested tag by a typedef name. */
  registry: TypeRegistry | null;
}>;

/**
 * Return the elements of a MAP's literal `over:` list, else `null`.
 *
 * @param result The built node.
 * @param wiring The node's binding map.
 * @returns The literal list's elements, or `null` when the node has no item scope or its `over:`
 *   is not a literal list.
 */
function literalItems(result: BuildResult, wiring: ReadonlyMap<string, unknown>): readonly unknown[] | null {
  const key = result.node.itemSourceKey;
  if (key === null || !wiring.has(key)) return null;
  const over = literalValue(bindingSource(wiring.get(key)));
  return isList(over) ? over : null;
}

/**
 * Return why one binding can't fill the child input `decl`, else `null`.
 *
 * A literal is validated by value with no string conversion, as the child's start validates it:
 * `"2026-01-01"` fits a `date` input but `"3"` does not fit an `int` input. A lone `${item...}`, a
 * `${ref :- fallback}` or `${ref :? message}` over a reference, and a `|` coalesce are checked
 * operand by operand (see `operandMismatch`). Any other source is checked by its resolved
 * contract.
 *
 * @param src The binding, raw or a `Binding`.
 * @param decl The child input, with a declared contract.
 * @param scope What the node's bindings read.
 * @returns The mismatch, without the node and binding names, or `null` when the binding fits or
 *   its type is unproven.
 */
function bindingMismatch(src: unknown, decl: Parameter, scope: Scope): string | null {
  const source = bindingSource(src);
  if (!(src instanceof Binding && src.contract !== null)) {
    const literal = literalValue(source);
    if (literal !== NOT_LITERAL) return valueMismatch("value", literal, decl);
    const tree = loneSpan(source);
    if (tree !== null) {
      const path = operandPath(tree);
      const kind = tree instanceof Tree ? tree.data : null;
      if (
        (path !== null && path.split(".")[0] === "item") ||
        ((kind === "default_expr" || kind === "required_expr") && operandPath((tree as Tree).children[0]!) !== null)
      ) {
        return operandMismatch(tree, decl, scope, { nonNull: false });
      }
      if (kind === "coalesce") {
        const message = operandMismatch(tree, decl, scope, { nonNull: false });
        if (message !== null) return message;
      }
    }
  }
  // A nullable source is fine for a non-nullable child input only when the binding guarantees a
  // non-null value: a non-null literal operand (`${a | 5}`) or text around a span. A present null
  // shadows the child's own `default:` (defaults fill only omitted inputs, never a bound null), so
  // it doesn't cover a null. `!bindingCoSkips` is exactly that non-null guarantee.
  return contractMismatch(
    resolveOutputBindingContract(src, scope.producers, scope.flowInputs, { typeFn: scope.typeFn }),
    decl,
    { nonNull: !bindingCoSkips(source), registry: scope.registry },
  );
}

/** Return the parsed expression of a source that is one whole `${...}`, else `null`. */
function loneSpan(source: unknown): ExprNode | null {
  if (typeof source !== "string") return null;
  let segments;
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
  const only = segments.length === 1 ? segments[0] : undefined;
  return only instanceof Span ? only.tree : null;
}

/** Return the dotted path of a plain reference operand (`a.b`), else `null`. */
function operandPath(node: ExprNode): string | null {
  const parts = directReferenceParts(node);
  return parts === null ? null : parts.join(".");
}

/**
 * Return why one operand of a lone `${...}` can't fill `decl`, else `null`.
 *
 * A reference is checked by its contract (`item...` through the element type of `over:`, or each
 * element of a literal `over:` by value). `head :- fallback` checks `head` without its null, then
 * `fallback`; `head :? message` checks `head` without its null; `a | b | c` checks every operand
 * but the last without its null. A constant is checked by value. A computed operand is left to the
 * child's start.
 *
 * @param node The parsed operand.
 * @param decl The child input, with a declared contract.
 * @param scope What the node's bindings read.
 * @param options.nonNull Whether a null from this operand falls through to another one, so only
 *   its non-null values reach the child.
 * @returns The mismatch, or `null` when it fits or is unproven.
 */
function operandMismatch(node: ExprNode, decl: Parameter, scope: Scope, options: { nonNull: boolean }): string | null {
  const { nonNull } = options;
  const path = operandPath(node);
  if (path !== null) return referenceMismatch(path, decl, scope, { nonNull });
  if (node instanceof Tree && node.data === "default_expr") {
    const [head, fallback] = node.children as [ExprNode, ExprNode];
    return (
      operandMismatch(head, decl, scope, { nonNull: true }) ?? operandMismatch(fallback, decl, scope, { nonNull })
    );
  }
  if (node instanceof Tree && node.data === "required_expr") {
    return operandMismatch(node.children[0]!, decl, scope, { nonNull: true });
  }
  if (node instanceof Tree && node.data === "coalesce") {
    const heads = node.children.slice(0, -1);
    for (const head of heads) {
      const message = operandMismatch(head, decl, scope, { nonNull: true });
      if (message !== null) return message;
    }
    return operandMismatch(node.children.at(-1)!, decl, scope, { nonNull });
  }
  if (exprRefs(node).length > 0) return null;
  let value: unknown;
  try {
    value = evalExpr(node, () => null);
  } catch (error) {
    if (error instanceof ExpressionError) return null; // a malformed constant fails when the binding is evaluated
    throw error;
  }
  if (value === null && nonNull) return null;
  // Only a single token was written as the value shown; a computed constant such as `2026-01-01`
  // (arithmetic here) gets no rewrite hint.
  const text = node instanceof Token && node.type === "NUMBER" ? node.value : null;
  return valueMismatch("default", value, decl, { hint: node instanceof Token, text });
}

/**
 * Return why the value one reference path reads can't fill `decl`, else `null`.
 *
 * @param path The dotted reference path.
 * @param decl The child input, with a declared contract.
 * @param scope What the node's bindings read.
 * @param options.nonNull Whether a null read here falls through to another operand.
 * @returns The mismatch, or `null` when it fits or is unproven.
 */
function referenceMismatch(path: string, decl: Parameter, scope: Scope, options: { nonNull: boolean }): string | null {
  const { nonNull } = options;
  const [head, ...fields] = path.split(".");
  if (head !== "item") {
    const contract = referenceContract(path, scope.producers, scope.flowInputs);
    return contractMismatch(contract, decl, { nonNull, registry: scope.registry });
  }
  if (scope.item !== null) {
    const contract = projectOutputContract(scope.item, fields);
    return contractMismatch(contract, decl, { nonNull, registry: scope.registry });
  }
  elements: for (const element of scope.items ?? []) {
    let value = element;
    for (const field of fields) {
      // A missing field fails when the element is bound.
      if (!isRecord(value) || !Object.hasOwn(value, field)) continue elements;
      value = value[field];
    }
    if (value === null && nonNull) continue;
    const message = valueMismatch("element", value, decl);
    if (message !== null) return message;
  }
  return null;
}

/**
 * Return why a source contract can't fill the child input `decl`, else `null`.
 *
 * An `Any` source gets advice to tag the binding or narrow it with `isinstance` when that can help
 * (see `anySourceHint`).
 *
 * @param contract The source's resolved contract.
 * @param decl The child input, with a declared contract.
 * @param options.nonNull Whether the binding guarantees a non-null value, so a nullable source is
 *   compared without its null.
 * @param options.registry The flow's typedefs, which spell a suggested tag by a typedef name.
 * @returns The mismatch, or `null` when it fits or is unproven (a `null` contract, or a whole
 *   partial record with no exact type).
 * @throws TypeError when `contract` is neither `null`, an exact `ValueContract` nor a
 *   `PartialRecordContract`.
 */
function contractMismatch(
  contract: OutputContract | null,
  decl: Parameter,
  options: { nonNull: boolean; registry?: TypeRegistry | null },
): string | null {
  const { nonNull, registry = null } = options;
  if (contract === null || isExactly(contract, PartialRecordContract)) return null;
  if (!isExactly(contract, ValueContract)) {
    throw new TypeError("resolved source contract must be ValueContract, PartialRecordContract, or null");
  }
  const childType = decl.contract!.type;
  let source = contract.type;
  if (source.nullable && !childType.nullable && nonNull) source = source.replace({ nullable: false });
  if (typesCompatible(source, childType)) return null;
  let message =
    `child expects ${reprText(canonicalTypeLabel(childType))}, source is ` +
    reprText(canonicalTypeLabel(source));
  const hint = anySourceHint(decl.name, source, childType, { registry });
  if (hint !== null) message += `; ${hint}`;
  return message;
}

/**
 * Return why a literal can't fill the child input `decl`, else `null`.
 *
 * @param what What the literal is in the binding (`value`, `default`, or `element`).
 * @param value The literal.
 * @param decl The child input, with a declared contract.
 * @param options.hint Whether to suggest a rewrite; off for a computed constant, whose text is not
 *   one literal (default `true`).
 * @param options.text The literal as written, when it differs from how `shownLiteral` renders the
 *   value (default `null`).
 * @returns The mismatch, with a hint when rewriting the literal would fix it, or `null` when the
 *   value fits.
 */
function valueMismatch(
  what: string,
  value: unknown,
  decl: Parameter,
  options: { hint?: boolean; text?: string | null } = {},
): string | null {
  const { hint = true, text = null } = options;
  const childType = decl.contract!.type;
  try {
    buildValueAs(childType, value);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    let message =
      `child expects ${reprText(canonicalTypeLabel(childType))}, but the ${what} ` +
      `${text || shownLiteral(value)} does not fit`;
    if (error.path.length > 0) message += ` (${error.message})`;
    const suggestion = hint ? literalHint(value, decl, text, { yaml: what !== "default" }) : null;
    return suggestion === null ? message : `${message}; ${suggestion}`;
  }
  return null;
}

/**
 * Suggest how to rewrite a literal that doesn't fit `decl`, else `null`; no rewrite is suggested
 * yet.
 *
 * @param value The literal that failed.
 * @param decl The child input, with a declared contract.
 * @param text The literal as written, if known.
 * @param options.yaml Whether YAML read the literal, rather than the `${...}` expression syntax.
 * @returns The hint, or `null` when rewriting the literal can't help.
 */
function literalHint(value: unknown, decl: Parameter, text: string | null, options: { yaml: boolean }): string | null {
  return null;
}
