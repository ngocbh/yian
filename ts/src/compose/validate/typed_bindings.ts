/**
 * Graph-stage: reject typed bindings whose value can never fit the declared type.
 *
 * A typed binding (`name: !Type value`, on a node input or a flow output) states the type its
 * value must have; the binder enforces it at runtime. A literal right side is already checked
 * against its type when the binding is split, where the authored text is at hand. This load pass
 * rejects the remaining bindings that could never pass the runtime check:
 *
 * - a string source with no `${...}` span (an escaped `$${x}`) is literal text, so it is checked
 *   exactly with the same write-boundary check the binder uses;
 * - a reference or expression is inferred from its raw source over the producer and flow-input
 *   contracts, and rejected only when that type is disjoint from the declared one. A narrower
 *   inferred type (an Optional source, a str into a Literal) loads and is left to the runtime
 *   check, as is an unproven source;
 * - a lone `${item}` or `${item.field}` read on a node that binds per item (MAP) is inferred from
 *   the element type of its `over` source, when that is a proven list;
 * - a value template is inferred from its structure, and each typed binding nested in it is
 *   checked the same way at its own path and origin.
 *
 * An AGENT's reserved `__llm_config` input is skipped: `llm_config_sources.ts` checks its tags at
 * the authored `llm_config` field (`typedValueError`), without CASE narrowing.
 *
 * `inferBindingType` returns the type one binding is proven to produce (a typed binding's
 * declared type, a literal's exact type, or a reference's inferred type), for passes that record
 * binding types rather than reject them. `localInputContracts` returns the proven contracts each
 * node-local name may have, for the passes that field-check what a node reads from its own inputs
 * (prompts).
 *
 * Node inputs are checked as soon as producer contracts exist, before the CALL, MAP, and LOOP
 * checks that compare the declared type with a child's parameters; flow outputs are checked once
 * their contracts are attached. A node input error locates at the binding's registered value
 * origin, else its `input:` entry range, else the node range; a flow output error at its
 * registered origin, else the `output:` range.
 *
 * Knows about: `compile/validation` (the type relation), `expr` (template scanning), `typesys`,
 * `nodes` (binding carriers, the LOOP node, the AGENT's reserved key), and `compose/helpers`
 * (output-contract inference and load diagnostics).
 * Never imports: the loader, the compose package facade, or the CALL/MAP check (`call_types.ts`,
 * for its literal and `${item}` helpers); they import this back.
 */

import type { FlowOutput } from "../../compile/model.ts";
import { typeRelation } from "../../compile/validation.ts";
import { type DiagnosticFrame, type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import { LLM_CONFIG_KEY } from "../../nodes/agent/llm_values.ts";
import {
  Binding,
  bindingSource,
  isExactly,
  type OutputContract,
  ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isList,
  isListKind,
  isRecord,
  type Type,
  TypeCheckError,
  type Value,
  valueRepr,
  writeAsJson,
} from "../../typesys/values.ts";
import { LoadDiagnostics, type LoadError } from "../helpers/errors.ts";
import {
  directReferencePath,
  literalType,
  literalValue,
  NOT_LITERAL,
  projectOutputContract,
  resolveOutputBindingContract,
  templateContract,
  type TypeFn,
} from "../helpers/output_contracts.ts";
import { namedTypeLabel, pathLabel } from "../helpers/tags.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import type { Narrowing } from "./narrowing.ts";

/** The proven contracts a node-local name's value has, and where it is bound from; frozen. */
export class LocalInput {
  /** A read is an error only when every alternative rejects it. */
  readonly alternatives: readonly OutputContract[];
  /**
   * The pool path the name is bound from when the binding's one `${...}` span reads exactly one
   * reference path (`${d.output}`, `${d.output :- null}`), else `null`; a member-only union hint
   * names it instead of the local name.
   */
  readonly source: string | null;

  /** Build the local input over a frozen copy of `alternatives`. */
  constructor(alternatives: readonly OutputContract[], source: string | null = null) {
    this.alternatives = Object.freeze([...alternatives]);
    this.source = source;
    Object.freeze(this);
  }
}

/** Resolves a binding's registered origin to its source frame, or `null` when it doesn't resolve. */
export type ResolveOrigin = (origin: OriginRef) => DiagnosticFrame | null;

/**
 * Reject every typed node input that can never fit its declared type.
 *
 * @param flowWiring Final binding maps keyed by node id and parameter name; typed entries are
 *   `Binding`s with a contract.
 * @param producers Exact or partial node output contracts keyed by producer id.
 * @param flowInputs Exact flow input contracts keyed by input name.
 * @param buildResults Built nodes with their transient value ranges, the fallback location for a
 *   node input.
 * @param options.narrowing The union narrowing each param reads against, from the CASE branches
 *   that route its node; `null` (the default) reads every param against the unchanged maps.
 * @param options.typeFn Resolves a constructor name to its type for inference.
 * @param options.registry The flow's typedefs, so a message names a declared record or union by
 *   its typedef.
 * @param options.resolveOrigin Resolves a binding's registered origin to its source frame, the
 *   preferred location.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError when a literal-text source does not fit its declared type, or a source's
 *   inferred type is disjoint from it. The input is named by its node's `BuildResult.siteName`, so
 *   an input a desugar copied onto a synth node is named as the node it was authored on.
 */
export function checkTypedInputs(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: {
    narrowing?: Narrowing | null;
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: ResolveOrigin | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const { narrowing = null, typeFn = null, registry = null, resolveOrigin = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  for (const [nodeId, wiring] of flowWiring) {
    const result = buildResults.get(nodeId) ?? null;
    const [nodeProducers, nodeInputs] = narrowing !== null ? narrowing.forNode(nodeId) : [producers, flowInputs];
    const item = itemContract(result, wiring, nodeProducers, nodeInputs, typeFn);
    for (const [name, binding] of wiring) {
      if (name === LLM_CONFIG_KEY) continue;
      const [paramProducers, paramInputs] =
        narrowing !== null ? narrowing.forParam(nodeId, name) : [nodeProducers, nodeInputs];
      for (const [path, typed] of typedSites(binding)) {
        const slot = pathLabel(name, path);
        const message = bindingError(typed, paramProducers, paramInputs, { item, typeFn, registry, slot });
        if (message === null) continue;
        let fallback: SourceRange | null = null;
        let where = `node ${reprText(nodeId)} input ${reprText(slot)}`;
        if (result !== null) {
          fallback = result.fieldRanges.get(siteKey("input", name)) ?? result.nodeRange;
          where = result.siteName(nodeId, `input ${reprText(slot)}`);
        }
        throw locatedError(`${where}: ${message}`, typed, fallback, resolveOrigin, diagnostics);
      }
    }
  }
}

/**
 * Return why a typed binding, or one nested in its value, can never fit its type.
 *
 * Nested bindings are checked first, innermost first, as `checkTypedInputs` checks a node input;
 * the contracts are used as given.
 *
 * @param binding A binding; untyped ones with no typed binding inside pass.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param options.typeFn Resolves a constructor name to its type.
 * @param options.registry The flow's typedefs, so the message names a declared record or union by
 *   its typedef.
 * @param options.slot The binding's slot as the message names a path inside it (default `""`).
 * @returns The first error message, without the binding's name, or `null`.
 */
export function typedValueError(
  binding: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { typeFn?: TypeFn | null; registry?: TypeRegistry | null; slot?: string } = {},
): string | null {
  const { typeFn = null, registry = null, slot = "" } = options;
  for (const [path, typed] of typedSites(binding)) {
    const message = bindingError(typed, producers, flowInputs, { typeFn, registry, slot: pathLabel(slot, path) });
    if (message !== null) return message;
  }
  return null;
}

/**
 * Reject every typed flow output that can never fit its declared type.
 *
 * @param flowOutputs The flow's outputs, typed ones carrying their declared contract on the
 *   binding.
 * @param producers Exact or partial node output contracts keyed by producer id.
 * @param flowInputs Exact flow input contracts keyed by input name.
 * @param options.valueMode Whether the flow has a single `output:` value, whose internal name the
 *   author never wrote, so errors name it `output` instead (default `false`).
 * @param options.outputsRange The authored `output:` range, the fallback location.
 * @param options.typeFn Resolves a constructor name to its type for inference.
 * @param options.registry The flow's typedefs, so a message names a declared record or union by
 *   its typedef.
 * @param options.resolveOrigin Resolves a binding's registered origin to its source frame, the
 *   preferred location.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError when a literal-text source does not fit its declared type, or a source's
 *   inferred type is disjoint from it.
 */
export function checkTypedOutputs(
  flowOutputs: readonly FlowOutput[],
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: {
    valueMode?: boolean;
    outputsRange?: SourceRange | null;
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: ResolveOrigin | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const { valueMode = false, outputsRange = null, typeFn = null, registry = null, resolveOrigin = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  for (const output of flowOutputs) {
    for (const [path, typed] of typedSites(output.binding)) {
      const slot = pathLabel(valueMode ? "output" : output.name, path);
      const message = bindingError(typed, producers, flowInputs, { typeFn, registry, slot });
      if (message === null) continue;
      const label = valueMode ? slot : `output ${reprText(slot)}`;
      throw locatedError(`${label}: ${message}`, typed, outputsRange, resolveOrigin, diagnostics);
    }
  }
}

/**
 * Return the type one binding is proven to produce, else `null`.
 *
 * A typed binding has its declared type; a literal has its exact type (a one-tag `Literal(...)` for
 * non-empty text, one-point bounds for a number); a reference or expression has the exact type
 * inferred over the producer and flow-input contracts; a value template has the exact type
 * inferred from its structure.
 *
 * @param binding A wiring binding, raw or a `Binding`.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param options.typeFn Resolves a constructor name to its type.
 * @returns The proven type, or `null` when it is unproven or only partial.
 */
export function inferBindingType(
  binding: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { typeFn?: TypeFn | null } = {},
): Type | null {
  const { typeFn = null } = options;
  if (binding instanceof Binding && binding.contract !== null) return binding.contract.type;
  const literal = literalValue(bindingSource(binding));
  if (literal !== NOT_LITERAL) return literalType(literal);
  const inferred = resolveOutputBindingContract(binding, producers, flowInputs, { typeFn });
  return isExactly(inferred, ValueContract) ? inferred.type : null;
}

/**
 * Build the load error, located at the binding's origin when it resolves.
 *
 * @param message The error message.
 * @param binding The rejected binding.
 * @param fallback The location used when the origin does not resolve.
 * @param resolveOrigin Resolves a registered origin to its source frame.
 * @param diagnostics Factory for located load errors.
 * @returns The located error.
 */
export function locatedError(
  message: string,
  binding: Binding,
  fallback: SourceRange | null,
  resolveOrigin: ResolveOrigin | null,
  diagnostics: LoadDiagnostics,
): LoadError {
  const frame = resolveOrigin !== null && binding.origin !== null ? resolveOrigin(binding.origin) : null;
  if (frame !== null) return diagnostics.error(message, { frames: [frame] });
  return diagnostics.error(message, { range: fallback });
}

/**
 * Return the exact element contract `${item}` reads on a per-item node, else `null`.
 *
 * @param result The built node; only a node with an item source key (MAP) has an item scope.
 * @param wiring The node's binding map, holding the item source under that key.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param typeFn Resolves a constructor name to its type.
 * @returns The element contract when the item source is an exact list with a known element type,
 *   else `null`.
 */
export function itemContract(
  result: BuildResult | null,
  wiring: ReadonlyMap<string, unknown>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  typeFn: TypeFn | null = null,
): ValueContract | null {
  const key = result !== null ? result.node.itemSourceKey : null;
  if (key === null || !wiring.has(key)) return null;
  const source = resolveOutputBindingContract(wiring.get(key), producers, flowInputs, { typeFn });
  if (!isExactly(source, ValueContract)) return null;
  if (!isListKind(source.type.kind) || source.type.element === undefined) return null;
  return new ValueContract(source.type.element);
}

/**
 * Return why one typed binding can never fit its declared type, else `null`.
 *
 * @param binding A wiring or output binding; untyped ones are skipped.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param options.item The element contract a lone `${item...}` reference reads, when the node
 *   binds per item over a proven list.
 * @param options.typeFn Resolves a constructor name to its type.
 * @param options.registry The flow's typedefs, so the message names a declared record or union by
 *   its typedef.
 * @param options.slot The binding's slot as the message names it (`p`, `p.decision`).
 * @returns The error message, without the binding's name, or `null` when the binding may fit.
 */
function bindingError(
  binding: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { item?: ValueContract | null; typeFn?: TypeFn | null; registry?: TypeRegistry | null; slot?: string } = {},
): string | null {
  const { item = null, typeFn = null, registry = null } = options;
  if (!(binding instanceof Binding) || binding.contract === null) return null;
  const declared = binding.contract.type;
  const label = namedTypeLabel(declared, registry ?? new Map());
  const literal = literalValue(binding.source);
  if (literal !== NOT_LITERAL) {
    try {
      buildValueAs(declared, literal);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      let message = `declared ${label}, but the value ${shownLiteral(literal)} does not fit`;
      if (error.path.length > 0) message += ` (${error.message})`;
      return message;
    }
    return null;
  }
  // Infer from the raw source: the Binding itself would resolve to its own declared contract and
  // always fit.
  const stringContract = (text: unknown): OutputContract | null =>
    itemReferenceContract(text, item) ?? resolveOutputBindingContract(text, producers, flowInputs, { typeFn });
  const inferred =
    binding.source instanceof ValueTemplate
      ? // Exact literals, so a wrong literal under a tag is disjoint from it.
        templateContract(binding.source, stringContract, { exactLiterals: true })
      : stringContract(binding.source);
  if (!isExactly(inferred, ValueContract)) return null; // unproven or partial: the runtime check covers it
  if (typeRelation(inferred.type, declared) !== "disjoint") return null;
  return `declared ${label}, but the value is ${canonicalTypeLabel(inferred.type)}`;
}

/**
 * Yield each typed binding nested in a binding's value, then the binding itself.
 *
 * Nested bindings come innermost first, each with its path in the value. A nested binding over a
 * container holding strings or bindings is yielded with that container as a value template, since
 * the binder evaluates it; the binding itself is yielded only when it is typed.
 *
 * @param binding A wiring or output binding.
 * @returns `[path, binding]` pairs.
 */
function* typedSites(binding: unknown): Generator<[ValuePath, Binding]> {
  const source = bindingSource(binding);
  if (source instanceof ValueTemplate) yield* nestedTypedSites(source.value, []);
  if (binding instanceof Binding && binding.contract !== null) yield [[], binding];
}

/** Yield the typed bindings in one template tree node, innermost first. */
function* nestedTypedSites(node: unknown, path: ValuePath): Generator<[ValuePath, Binding]> {
  if (node instanceof Binding) {
    yield* nestedTypedSites(node.source, path);
    if (node.contract === null) return;
    let source = node.source;
    if ((isRecord(source) || isList(source)) && hasTemplatePart(source)) source = new ValueTemplate(source);
    yield [path, new Binding(source, { contract: node.contract, origin: node.origin })];
  } else if (isRecord(node)) {
    for (const [key, item] of Object.entries(node)) yield* nestedTypedSites(item, [...path, key]);
  } else if (isList(node)) {
    for (const [index, item] of node.entries()) yield* nestedTypedSites(item, [...path, index]);
  }
}

/**
 * Return whether a template tree node holds a string or a typed binding.
 *
 * Such a container is evaluated by the binder, so it is inferred as a template rather than checked
 * as a literal.
 */
function hasTemplatePart(node: unknown): boolean {
  if (typeof node === "string" || node instanceof Binding) return true;
  if (isRecord(node)) return Object.values(node).some(hasTemplatePart);
  if (isList(node)) return node.some(hasTemplatePart);
  return false;
}

/**
 * Return the contract a lone `${item...}` reference reads, else `null`.
 *
 * @param source The binding's raw source.
 * @param item The node's element contract, or `null` when the node has no proven item scope.
 * @returns The element contract projected through the reference's fields, or `null` for any other
 *   source or an unknown field (reported by reference validation).
 */
function itemReferenceContract(source: unknown, item: ValueContract | null): OutputContract | null {
  if (item === null) return null;
  const path = directReferencePath(source);
  if (path === null) return null;
  const [head, ...fields] = path.split(".");
  if (head !== "item") return null;
  return projectOutputContract(item, fields);
}

/**
 * Render a literal as its text form: compact JSON (`true`, `null`, `"text"`, `[1,"a"]`).
 *
 * @param value The literal to render.
 * @returns `writeAsJson`'s text; `valueRepr` for anything it can't write.
 */
export function shownLiteral(value: unknown): string {
  try {
    return writeAsJson(value as Value);
  } catch (error) {
    if (error instanceof TypeError) return valueRepr(value);
    throw error;
  }
}
