/**
 * The post-parse conductor: assembled sections to a runnable `ComposedFlow`.
 *
 * `assemble` is the pipeline shared by the top-level flow and every in-file `defs:` callable. It
 * runs one phase function after another; each takes the `AssembleContext` and the records the
 * earlier phases returned, and returns its own:
 *
 *   1. `readSections`: extract inline bodies; stamp flow-assert origins; read the flow parameters;
 *      parse nodes (a tagged `nodes:` entry lowers to a CODE node returning its value) and stamp
 *      their origins; refuse a `uses:` alias that collides with a node id; bound written-value
 *      nesting; refuse a tagged node `output:`; split tagged inputs, `llm_config` and flow outputs
 *      into value + declared contract; check case branches.
 *   2. `desugarSections`: inline calls, adaptive questions, case call targets, then
 *      `${case.output}` to a coalesce over the case's branch targets.
 *   3. `buildNodes`: register inline bodies as private callables; build every node kind-blind
 *      (`desc.build(ctx)`; physical node-output contracts compile at their leaf-owned build
 *      position; a case lowers in its own build); stamp binding labels, wiring origins, input
 *      contracts and each node's merged `env`; collect the producers and wire each AGENT's
 *      `llm_config` values.
 *   4. `routeEdges`: infer data edges, append each case's own edges, copy each host's
 *      route/ordering edges and data gates onto the nodes lifted out of it, compute CASE
 *      narrowing.
 *   5. `checkTypes`: case `when:` values and impossible `when:`s, typed inputs (and TOOL input
 *      types), CALL/MAP types, the LOOP contract, case exhaustiveness, wiring parity.
 *   6. `checkReferences`: the node-local contracts, ordering edges and `SkipModel` the checks
 *      read; validate references, check node asserts, lower flow asserts.
 *   7. `finishOutputs`: attach flow-output origins, inline call origins and declared output
 *      contracts; check field reads; null outputs CASE routing may skip; check skipped arithmetic
 *      and typed outputs.
 *   8. `closeFlow`: synthesize the START_ID/END_ID boundary over all edges; check `llm_config`
 *      sources; fill each node's `isinstance` type table; check reserved sources (`over:`,
 *      `until:`).
 *   9. `checkGraph`: reject cycles and check if/else handles; then `new CompiledFlow`.
 *
 * One retained assertion item cursor supplies its carrier origin and Compose-only validation
 * range; only the origin crosses into the semantic assertion.
 *
 * Holds `ComposedFlow` (the compiled and validated flow the engine runs) plus the flow `output:`
 * parser and source-origin attachment. Ladder position: the last phase before the loader; it may
 * import every earlier phase (helpers, nodes, parse, desugar, edges, validate, build/boundary),
 * and the loader imports `assemble` and `ComposedFlow` back from here. It imports nothing from
 * the loader, which would be a back-edge.
 */

import { CompiledFlow, type Edge, FlowOutput } from "../../compile/model.ts";
import {
  compareCodeUnits,
  Diagnostic,
  DiagnosticFrame,
  Failure,
  type OriginRef,
  type OriginResolver,
  reprText,
  SourceRange,
} from "../../diagnostics.ts";
import type { Node } from "../../nodes/base.ts";
import {
  asBinding,
  type Assertion,
  Binding,
  isExactly,
  type OutputContract,
  Parameter,
  ValueContract,
  type ValuePath,
  type ValueTemplate,
} from "../../nodes/contracts.ts";
import { ASK } from "../../tools/context.ts";
import { messagesKey, NEW_MESSAGES } from "../../typesys/pool.ts";
import { newMessagesType, type TypeRegistry } from "../../typesys/types.ts";
import { isList, isListKind, isRecord, type Type, type Value, writeAsText } from "../../typesys/values.ts";
import { LoadDiagnostics, LoadError, rejectReservedName } from "../helpers/errors.ts";
import { constructorTypeFn } from "../helpers/isinstance_types.ts";
import { assertExactContracts, resolveOutputBindingContract, type TypeFn } from "../../compile/output_contracts.ts";
import {
  oldFormSuggestion,
  oldTypedText,
  QUOTED_STYLES,
  readTaggedValue,
  rejectIncludeIn,
  rejectTaggedOutput,
  suggestTag,
  yamlQuoted,
} from "../helpers/tags.ts";
import type { MemoryClass, MemoryTypeTable } from "../helpers/memory.ts";
import { readFlowInputs, readSourceParameters, readValueContract } from "../helpers/types.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import {
  BuildContext,
  type BindingContractReader,
  type BuildResult,
  type NodeDescriptor,
  type OriginPresenter,
} from "../nodes/base.ts";
import { normalizeNodeSignatureCursor, parseNodes } from "../parse/nodes.ts";
import type { SourceCatalogBuilder, SourceSitePath } from "../source/catalog.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";
import { AgentDescriptor, readLlmConfig } from "../nodes/agent.ts";
import { siteKey } from "../nodes/base.ts";
import { QUESTIONS_INPUT, ToolDescriptor } from "../nodes/tool.ts";
import { desugarAdaptiveQuestions } from "../desugar/adaptive_questions.ts";
import { type CaseRead, caseOutputReads, expandCaseOutputs } from "../desugar/case_outputs.ts";
import { desugarCaseCallTargets } from "../desugar/case_targets.ts";
import { extractInlineBodies, type InlineBody } from "../desugar/inline_bodies.ts";
import { desugarInlineCalls, synthIdGen } from "../desugar/inline_calls.ts";
import { isValueEntry, readValueEntry, type ValueEntryDescriptor } from "../desugar/value_entries.ts";
import { inferDataEdges } from "../edges/data.ts";
import { hostControlEdges } from "../edges/hosts.ts";
import { inferOrderingEdges } from "../edges/ordering.ts";
import { bindingProducers } from "../edges/producers.ts";
import { CallDescriptor } from "../nodes/call.ts";
import { CaseDescriptor } from "../nodes/case.ts";
import { LoopDescriptor } from "../nodes/loop.ts";
import { MapDescriptor } from "../nodes/map.ts";
import type { ScalarStyle } from "../source/common.ts";
import { synthesizeBoundaryGraph } from "./boundary.ts";
import {
  canonicalizeInlineType,
  explicitTypeHint,
  inferInlineBindingType,
  type ItemType,
  type ProducerType,
} from "./inline_inputs.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import { NodeKind } from "../../nodes/base.ts";
import { CaseNode } from "../../nodes/case/node.ts";
import { LoopNode } from "../../nodes/loop/node.ts";
import { TypeCheckError } from "../../typesys/values.ts";
import {
  collectConstructorNames,
  collectIsinstanceNames,
  expressionTrees,
  resolveConstructorType,
  resolveIsinstanceType,
} from "../helpers/isinstance_types.ts";
import { nth } from "../helpers/source.ts";
import {
  builtinArgumentErrors,
  builtinFieldErrors,
  constructorErrors,
  fieldReadErrors,
  subscriptErrors,
} from "../validate/constructors.ts";
import { rejectCycles } from "../validate/cycles.ts";
import { checkCaseHandles } from "../validate/handles.ts";
import { validateReferences } from "../validate/references.ts";
import { checkReservedSources } from "../validate/reserved_sources.ts";
import { stampToolInputTypes } from "../validate/tool_inputs.ts";
import { checkWrittenValues } from "../validate/value_nesting.ts";
import { checkWiringParity } from "../validate/wiring_parity.ts";
import { ExpressionError } from "../../expr/index.ts";
import { LLM_CONFIG_KEY, splitLayers, valueLeaves } from "../../nodes/agent/llm_values.ts";
import { AgentNode } from "../../nodes/agent/node.ts";
import { checkCallMapTypes } from "../validate/call_types.ts";
import { checkLlmConfigSources } from "../validate/llm_config_sources.ts";
import { checkLoopTypeContract } from "../validate/loop_types.ts";
import {
  checkTypedInputs,
  checkTypedOutputs,
  inferBindingType,
  type LocalInput,
  localInputContracts,
} from "../validate/typed_bindings.ts";
import { type LoweredAsserts, lowerFlowAsserts, validateNodeAsserts } from "../validate/asserts.ts";
import { checkCaseBranches } from "../validate/case_branches.ts";
import { checkCaseExhaustive, checkCaseWhenValues, exhaustiveCases } from "../validate/case_exhaustive.ts";
import { checkImpossibleWhens, computeNarrowing, type Narrowing } from "../validate/narrowing.ts";
import { checkFieldReads } from "../validate/field_reads.ts";
import { checkSkippedArithmetic, nullSkippedOutputs, type SearchedWhen, SkipModel } from "../validate/skips.ts";
import type { FlowSource } from "../../runtime/log.ts";

// Name for the single output of a bare whole-string `output:` binding. The value-mode EndNode
// returns the bare resolved value, so the name only labels the carrier.
const SINGLE_OUTPUT_NAME = "result";

const PRESENTATION_UNAVAILABLE_NOTE = "Source presentation was unavailable.";

/** Run `fn`, rethrowing a `SourceAccessError` from walking retained source as a located `LoadError`. */
function locateSourceAccess<T>(diagnostics: LoadDiagnostics, fn: () => T): T {
  try {
    return fn();
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
}

// Each `nodes:` cursor's node entries, normalized. A cursor never changes, so the passes of one
// assembly share one normalization per node.
const normalizedNodes = new WeakMap<SourceCursor, Map<string, SourceCursor>>();

/**
 * Return the value at `path` inside one authored node's normalized entry (`input:` read as
 * `inputs`), or the entry itself for an empty `path`.
 *
 * A text step selects a mapping key; an index step selects a sequence item the caller knows is
 * there. Don't call it for a value entry, which has no node body.
 *
 * @param nodesCursor Retained view of the flow's node mapping.
 * @param nodeId The node whose entry is read.
 * @param path The steps inside the entry.
 * @returns The cursor, or `null` when the node or a key along `path` is absent.
 * @throws SourceAccessError when the retained source can't be traversed.
 */
function authoredField(nodesCursor: SourceCursor, nodeId: string, ...path: (string | number)[]): SourceCursor | null {
  if (!nodesCursor.has(nodeId)) return null;
  let entries = normalizedNodes.get(nodesCursor);
  if (entries === undefined) {
    entries = new Map();
    normalizedNodes.set(nodesCursor, entries);
  }
  let cursor = entries.get(nodeId);
  if (cursor === undefined) {
    cursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
    entries.set(nodeId, cursor);
  }
  for (const step of path) {
    if (typeof step === "string" && !cursor.has(step)) return null;
    cursor = cursor.child(step);
  }
  return cursor;
}

/** One active descriptor build and the dependency edge that requested it; frozen. */
class BuildRequest {
  /** The node being built. */
  readonly nodeId: string;
  /** The requesting edge's label, or `null` for none. */
  readonly viaLabel: string | null;
  /** The requesting edge's range, or `null` for none. */
  readonly viaRange: SourceRange | null;

  /** Record one build of `nodeId` requested through the edge `viaLabel` at `viaRange`. */
  constructor(nodeId: string, viaLabel: string | null, viaRange: SourceRange | null) {
    this.nodeId = nodeId;
    this.viaLabel = viaLabel;
    this.viaRange = viaRange;
    Object.freeze(this);
  }
}

/** What the build cache reads of a descriptor: its build, and its range when it has one. */
type BuildableDescriptor = Pick<NodeDescriptor, "build" | "range">;

/**
 * Builds descriptors lazily once while retaining dependency-cycle provenance.
 *
 * A descriptor may request a later descriptor while deriving its own build product. Successful
 * products are memoized; failures are not. The final source-order walk therefore reuses
 * forward-built products without changing assembled node order.
 */
export class DescriptorBuildCache {
  readonly #descriptors: ReadonlyMap<string, BuildableDescriptor>;
  readonly #context: BuildContext;
  readonly #built = new Map<string, BuildResult>();
  readonly #building: BuildRequest[] = [];

  /** Create a cache building `descriptors` with `context`. */
  constructor(descriptors: ReadonlyMap<string, BuildableDescriptor>, context: BuildContext) {
    this.#descriptors = descriptors;
    this.#context = context;
  }

  /**
   * Build one descriptor, reusing success and rejecting an active dependency cycle.
   *
   * @param nodeId The descriptor to build.
   * @param viaLabel The requesting edge's label, used in a cycle's path.
   * @param viaRange The requesting edge's range, where a cycle it closes points.
   * @throws LoadError for a dependency cycle (`node build dependency cycle: a -> b -> a`), or the
   *   descriptor's own build failure, located at the descriptor when it has no frame.
   */
  buildOne(nodeId: string, viaLabel: string | null = null, viaRange: SourceRange | null = null): BuildResult {
    const built = this.#built.get(nodeId);
    if (built !== undefined) return built;

    const request = new BuildRequest(nodeId, viaLabel, viaRange);
    const repeatedAt = this.#building.findIndex((active) => active.nodeId === nodeId);
    if (repeatedAt !== -1) {
      const cycle = this.#building.slice(repeatedAt);
      const cycleEdges = [...cycle.slice(1), request];
      const path = cycleEdges.every((edge) => edge.viaLabel !== null)
        ? cycleEdges.map((edge) => edge.viaLabel!)
        : cycle.map((active) => active.nodeId);
      path.push(nodeId);
      const errorRange =
        request.viaRange ?? cycleEdges.find((edge) => edge.viaRange !== null)?.viaRange ?? null;
      throw this.#context.diagnostics.error(`node build dependency cycle: ${path.join(" -> ")}`, {
        range: errorRange,
      });
    }

    this.#building.push(request);
    try {
      const descriptor = this.#descriptors.get(nodeId)!;
      let result: BuildResult;
      try {
        result = descriptor.build(this.#context);
      } catch (error) {
        if (!(error instanceof LoadError) || error.diagnostic.frames.length > 0) throw error;
        if (descriptor.range === null) throw error;
        throw this.#context.diagnostics.prepend(error, { range: descriptor.range, cause: error });
      }
      this.#built.set(nodeId, result);
      return result;
    } finally {
      this.#building.pop();
    }
  }

  /** Return every successful build in descriptor insertion order. */
  buildInSourceOrder(): Map<string, BuildResult> {
    const results = new Map<string, BuildResult>();
    for (const nodeId of this.#descriptors.keys()) results.set(nodeId, this.buildOne(nodeId));
    return results;
  }
}

/**
 * A compiled, validated flow ready to run; frozen.
 *
 * Produced by `loadFlow` and consumed by `runFlow` and the engine. Immutable so it can be reused
 * across runs and resumes; the loader rebuilds it through the constructor to install its
 * resolver.
 */
export class ComposedFlow {
  /** The runnable IR (nodes, edges, outputs, wiring) the `FlowEngine` executes. */
  readonly compiled: CompiledFlow;
  /**
   * The flow's declared `version:`, or `null` if unversioned. Used to validate a
   * `uses: ref@<version>` pin against the resolved file.
   */
  readonly version: string | null;
  /**
   * The flow's declared `name:`: human-facing metadata (e.g. the CLI's interactive-prompt
   * header). `null` for a `defs:` child flow, which has no name.
   */
  readonly name: string | null;
  /** The flow's declared `description:`, or `null` if absent: human-facing metadata. */
  readonly description: string | null;
  /**
   * Source-neutral authority for every reachable origin in a flow returned by `loadFlow`.
   * `null` means the compiled graph is source-free. The private assembly pipeline leaves this
   * unset while one physical document is still being built; the loader installs the exact
   * immutable resolver before the flow is returned or cached.
   */
  readonly originResolver: OriginResolver | null;
  /**
   * Longest chain of `uses:` imports beneath this flow; 0 when it imports nothing or was built
   * without the loader.
   */
  readonly importDepth: number;
  /**
   * The flow files this flow was compiled from, which a run records and resume compiles again;
   * `null` for a flow built in code, or loaded over a child resume couldn't read back.
   */
  readonly source: FlowSource | null;
  /**
   * The host's own memory classes the flow was loaded with (`loadFlow`'s `memoryTypes`); empty
   * for a flow loaded again from its recorded source.
   */
  readonly memoryTypes: readonly MemoryClass[];

  /**
   * Wrap `compiled`; the metadata, resolver and source default to `null`, `importDepth` to 0,
   * and `memoryTypes` to none.
   */
  constructor(
    compiled: CompiledFlow,
    options: {
      version?: string | null;
      name?: string | null;
      description?: string | null;
      originResolver?: OriginResolver | null;
      importDepth?: number;
      source?: FlowSource | null;
      memoryTypes?: readonly MemoryClass[];
    } = {},
  ) {
    const {
      version = null,
      name = null,
      description = null,
      originResolver = null,
      importDepth = 0,
      source = null,
      memoryTypes = [],
    } = options;
    this.compiled = compiled;
    this.version = version;
    this.name = name;
    this.description = description;
    this.originResolver = originResolver;
    this.importDepth = importDepth;
    this.source = source;
    this.memoryTypes = memoryTypes;
    Object.freeze(this);
  }

  /**
   * The one canonical flow signature stored on the START node.
   *
   * @throws TypeError when the START params aren't an exact list of exact `Parameter` values.
   * @throws RangeError when the compiled flow has no START node.
   */
  get input(): readonly Parameter[] {
    const start = this.compiled.nodes.get(this.compiled.startId);
    if (start === undefined) {
      throw new RangeError(`compiled flow has no START node ${reprText(this.compiled.startId)}`);
    }
    const parameters = start.params;
    if (!isList(parameters)) throw new TypeError("compiled START params must be an exact list");
    for (const parameter of parameters) {
      if (!isExactly(parameter, Parameter)) {
        throw new TypeError("compiled START params must contain exact Parameter values");
      }
    }
    return parameters;
  }

  /**
   * Resolve one semantic failure through this flow's source authority.
   *
   * Resolution is presentation-only. A missing resolver, or one that throws or answers an origin
   * with anything but a `DiagnosticFrame`, produces an unlocated diagnostic without changing the
   * semantic failure fields. Each frame is kept as the resolver returns it.
   *
   * @param failure Valid source-neutral failure whose trace is ordered outermost to innermost.
   * @returns The unchanged semantic fields plus every resolved frame in trace order, or no
   *   frames and one generic note when presentation is unavailable.
   * @throws TypeError when `failure` is not a `Failure`.
   */
  diagnose(failure: Failure): Diagnostic {
    if (!(failure instanceof Failure)) throw new TypeError("diagnose requires a Failure");

    const diagnostic = new Diagnostic({
      code: failure.code,
      message: failure.message,
      nodeId: failure.nodeId,
      causeType: failure.causeType,
      traceback: failure.traceback,
    });
    if (failure.trace.length === 0) return diagnostic;

    const unavailable = new Diagnostic({ ...diagnostic, notes: [PRESENTATION_UNAVAILABLE_NOTE] });
    const resolver = this.originResolver;
    if (resolver === null) return unavailable;

    const frames: DiagnosticFrame[] = [];
    for (const origin of failure.trace) {
      try {
        const frame = resolver.resolve(origin);
        if (!(frame instanceof DiagnosticFrame)) return unavailable;
        frames.push(frame);
      } catch {
        return unavailable;
      }
    }

    return new Diagnostic({ ...diagnostic, frames });
  }
}

/** Structural seam for registering a lazy compiler-owned child callable. */
export interface PrivateRegistrar {
  /** Register one private callable builder under its generated identity. */
  registerPrivate(flowId: string, owner: readonly string[], label: string, builder: () => ComposedFlow): void;
}

/** File-owned callable scope used by assembly and anonymous child lowering. */
export interface CallableScope extends PrivateRegistrar {
  /** The one mutable origin registry owned by this source file. */
  readonly catalogBuilder: SourceCatalogBuilder;
  /** How this file's flow inputs read memory types beyond the built-ins. */
  readonly memoryTypes: MemoryTypeTable;
  /** Whether `flowId` names a `defs:` callable or a `uses:` alias, which `call(...)` can run. */
  has(flowId: string): boolean;
  /** Resolve one named or compiler-owned callable in this file scope. */
  resolve(flowId: string): ComposedFlow;
}

/**
 * Associate retained inline bodies with descriptors and register lazy builders.
 *
 * Every association is validated before the registrar is touched. The returned map replaces only
 * matching frozen descriptors with copies carrying an author-facing child label; the input
 * descriptors remain unchanged. Child construction stays lazy until the registrar invokes a
 * captured zero-argument builder.
 *
 * @param descriptors The final node descriptors by id.
 * @param bodies The retained inline bodies by node id.
 * @param options.registrar Where each body's lazy builder is registered.
 * @param options.buildChild Builds one body's child flow under its labelled descriptor.
 * @returns A new descriptor map with each inline composite labelled.
 * @throws RangeError, a compiler bug, when a body has no descriptor, its descriptor is not a
 *   CALL, MAP or LOOP, or the descriptor does not call the body's private id: body extraction
 *   gives every body such a descriptor.
 */
export function registerInlineBodies(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  bodies: ReadonlyMap<string, InlineBody>,
  options: {
    registrar: PrivateRegistrar;
    buildChild: (body: InlineBody, descriptor: CallDescriptor | LoopDescriptor) => ComposedFlow;
  },
): Map<string, NodeDescriptor> {
  const { registrar, buildChild } = options;
  const associated = new Map(descriptors);
  const registrations: [InlineBody, CallDescriptor | LoopDescriptor, string][] = [];

  for (const body of bodies.values()) {
    const label = `inline body of node ${reprText(body.nodeId)}`;
    const descriptor = descriptors.get(body.nodeId);
    if (descriptor === undefined) {
      throw new RangeError(`${label}: no matching composite node descriptor`);
    }
    if (!(descriptor instanceof CallDescriptor || descriptor instanceof LoopDescriptor)) {
      throw new RangeError(`${label}: final node descriptor is not CALL, MAP, or LOOP`);
    }
    if (descriptor.call !== body.flowId) {
      throw new RangeError(`${label}: final node descriptor does not target its retained inline body`);
    }

    const labelledDescriptor = descriptor.replace({ childLabel: label });
    associated.set(body.nodeId, labelledDescriptor);
    registrations.push([body, labelledDescriptor, label]);
  }

  for (const [body, labelledDescriptor, label] of registrations) {
    registrar.registerPrivate(body.flowId, body.owner, label, () => buildChild(body, labelledDescriptor));
  }

  return associated;
}

/**
 * Align one retained assertion sequence with its semantic source order.
 *
 * @param assertions Raw or carrierized assertion values in authored order.
 * @param assertsCursor Retained authored sequence, or `null` for a generated or source-free list.
 * @param diagnostics Factory for located load errors.
 * @returns One item cursor per assertion. Source-free lists contain `null` entries.
 * @throws RangeError, a compiler bug, when the retained sequence does not align with the semantic
 *   list.
 * @throws LoadError when the retained sequence cannot be traversed.
 */
export function assertionCursors(
  assertions: readonly unknown[],
  assertsCursor: SourceCursor | null,
  diagnostics: LoadDiagnostics,
): readonly (SourceCursor | null)[] {
  if (assertsCursor === null) return assertions.map(() => null);
  const cursors = locateSourceAccess(diagnostics, () => assertsCursor.children());
  if (cursors.length !== assertions.length) {
    throw new RangeError("retained assertion cursor does not align with semantic assertions");
  }
  return cursors;
}

/**
 * Register one authored assertion item without exposing source to consumers.
 *
 * @param cursor Authored list-item cursor, or `null` for a source-free assertion.
 * @param sitePath Complete lexical semantic path for the assertion occurrence.
 * @param catalogBuilder File-owned origin catalog builder shared by all lexical children.
 * @param diagnostics Factory for located load errors.
 * @returns Opaque registered origin, or `null` when no physical site exists.
 * @throws LoadError when the catalog site collides with another registration.
 */
export function registerAssertionOrigin(
  cursor: SourceCursor | null,
  sitePath: SourceSitePath,
  catalogBuilder: SourceCatalogBuilder,
  diagnostics: LoadDiagnostics,
): OriginRef | null {
  if (cursor === null) return null;
  return locateSourceAccess(diagnostics, () => catalogBuilder.register(sitePath, cursor));
}

/**
 * Copy the load-error names desugars gave generated params into each node's `bindingLabels`.
 *
 * A CASE `on:` or `cases[i].when` param takes its `wiringLabels` site name; a value entry's
 * input, an inline `call(...)` argument, and an adaptive composer's copied input take their
 * `fieldLabels` phrase. Other params keep the plain `input '<param>'` wording.
 *
 * @param buildResults Built nodes keyed by final descriptor id; each node's `bindingLabels` is
 *   replaced by a copy holding the labels.
 */
export function stampBindingLabels(buildResults: ReadonlyMap<string, BuildResult>): void {
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    const labels = new Map(node.bindingLabels);
    for (const parameter of node.params ?? []) {
      const name = parameter.name;
      const wiringLabel = node.wiringLabels.get(name);
      const fieldLabel = result.fieldLabels.get(`input ${reprText(name)}`);
      if (wiringLabel !== undefined) {
        // `on:` names the YAML key; a runtime message adds its own colon.
        const site = wiringLabel.endsWith(":") ? wiringLabel.slice(0, -1) : wiringLabel;
        labels.set(name, result.siteName(nodeId, site));
      } else if (fieldLabel !== undefined) {
        labels.set(name, fieldLabel);
      }
    }
    node.bindingLabels = labels;
  }
}

/** Return `binding` (a raw source or a `Binding`) as a `Binding` carrying `origin`. */
function withOrigin(binding: unknown, origin: OriginRef): Binding {
  const carrier = asBinding(binding);
  return new Binding(carrier.source, { origin, contract: carrier.contract });
}

/**
 * Attach authored node-wiring sites to their canonical carriers.
 *
 * Desugaring has already produced each final binding source, while `nodesCursor` still retains
 * the original occurrence. Each ordinary input entry contributes two meanings: its key owns the
 * formal `Parameter` and its value owns the flow `Binding`; a value entry's one input is the
 * entry itself, so a runtime type failure points at it. A reserved MAP `over` or WAIT `until`
 * field contributes only a flow-owned `Binding` origin. For names absent from the retained
 * cursor, Compose mints no origin and preserves any existing carrier origin.
 *
 * @param descriptors Final descriptors keyed by node id after every graph desugar.
 * @param nodesCursor Retained source view of the flow's authored node mapping.
 * @param nodes Built runtime nodes keyed by final descriptor id.
 * @param flowWiring Mutable final binding maps keyed by node id and wiring key.
 * @param options.owner Complete lexical owner path used to namespace catalog sites.
 * @param options.catalogBuilder File-owned origin catalog builder shared by all lexical children.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError when a retained source cursor cannot be traversed.
 */
export function stampNodeWiringOrigins(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  nodesCursor: SourceCursor,
  nodes: ReadonlyMap<string, Node>,
  flowWiring: ReadonlyMap<string, Map<string, unknown>>,
  options: { owner: readonly string[]; catalogBuilder: SourceCatalogBuilder; diagnostics: LoadDiagnostics },
): void {
  const { owner, catalogBuilder, diagnostics } = options;
  for (const [nodeId, descriptor] of descriptors) {
    const node = nodes.get(nodeId);
    const wiring = flowWiring.get(nodeId);
    if (node === undefined || wiring === undefined) continue;

    locateSourceAccess(diagnostics, () => {
      if (!nodesCursor.has(nodeId)) return;
      let inputCursors: Map<string, SourceCursor>;
      if (isValueEntry(descriptor)) {
        inputCursors = new Map([[nodeId, nodesCursor.child(nodeId)]]);
      } else {
        const nodeCursor = authoredField(nodesCursor, nodeId)!;

        for (const key of [...node.reservedWiringKeys()].sort(compareCodeUnits)) {
          if (!wiring.has(key) || !nodeCursor.has(key)) continue;
          const bindingOrigin = catalogBuilder.register(
            [...owner, "node", nodeId, "reserved", key, "binding"],
            nodeCursor.child(key),
          );
          if (bindingOrigin !== null) wiring.set(key, withOrigin(wiring.get(key), bindingOrigin));
        }

        const inputsCursor = authoredField(nodesCursor, nodeId, "inputs");
        if (Object.keys(descriptor.inputs).length === 0 || inputsCursor === null) return;
        inputCursors = new Map();
        for (const name of Object.keys(descriptor.inputs)) {
          if (inputsCursor.has(name)) inputCursors.set(name, inputsCursor.child(name));
        }
      }
      if (node.params === null) return;
      const parameterIndexes = new Map(node.params.map((parameter, index) => [parameter.name, index]));
      const parameters = [...node.params];
      let parametersChanged = false;
      for (const [name, entryCursor] of inputCursors) {
        const parameterIndex = parameterIndexes.get(name);
        if (parameterIndex === undefined || !wiring.has(name)) continue;
        const parameterOrigin = catalogBuilder.register(
          [...owner, "node", nodeId, "input", name, "parameter"],
          entryCursor,
          "key",
        );
        const bindingOrigin = catalogBuilder.register(
          [...owner, "node", nodeId, "input", name, "binding"],
          entryCursor,
        );

        if (parameterOrigin !== null) {
          const parameter = parameters[parameterIndex]!;
          parameters[parameterIndex] = parameter.replace({ origin: parameterOrigin });
          parametersChanged = true;
        }
        if (bindingOrigin !== null) wiring.set(name, withOrigin(wiring.get(name), bindingOrigin));
      }
      if (parametersChanged) node.params = parameters;
    });
  }
}

/**
 * Split one binding value into its value and declared contract.
 *
 * A tagged value (`!int ${size.output.qty}`) is read through its tag. Untagged plain text in the
 * old `Type = value` form is an error that shows the tag; a quoted or block value is text. Every
 * authored value is read with the spans and tags inside it, so a container holding them comes
 * back as a `ValueTemplate`, typed or not.
 *
 * @param value The authored binding value.
 * @param cursor Retained view of the value, or `null` for a value with no authored occurrence.
 * @param options.bindingLabel The binding's key name, used in error messages.
 * @param options.sitePath Catalog site prefix for this binding.
 * @param options.registry The file's typedefs the type text resolves against.
 * @param options.catalogBuilder File-owned origin catalog builder.
 * @param options.diagnostics Factory for located load errors.
 * @returns The binding value and its declared contract (`null` for an untyped mapping or list),
 *   or `null` when the value is untyped and kept as it is: not a mapping or list, or not the
 *   value its cursor shows.
 * @throws LoadError when a tag is malformed, has no value after it, or does not fit its tagged
 *   literal, or the value is in the old `Type = value` form.
 */
function splitTypedValue(
  value: unknown,
  cursor: SourceCursor | null,
  options: {
    bindingLabel: string;
    sitePath: SourceSitePath;
    registry: TypeRegistry;
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
  },
): [value: unknown, contract: ValueContract | null] | null {
  const { bindingLabel, sitePath, registry, catalogBuilder, diagnostics } = options;
  const readOptions = {
    label: bindingLabel,
    sitePath,
    catalogBuilder,
    diagnostics,
    ledger: catalogBuilder.consumedTags,
  };
  if (cursor !== null && cursor.tag !== null) {
    const slot = readTaggedValue(cursor, registry, readOptions);
    // A top-level tag always gives the slot its contract.
    const contract = slot.contract!;
    if (!slot.hasValue) {
      const spelled = suggestTag(bindingLabel, contract.type, "${...}", { registry, typeText: cursor.tag });
      throw diagnostics.error(`${reprText(bindingLabel)}: a binding needs a value: ${spelled}`, {
        range: cursor.range,
      });
    }
    return [slot.value, contract];
  }
  const style = cursor !== null ? cursor.scalarStyle : null;
  const old = QUOTED_STYLES.has(style) ? null : oldTypedText(value, registry);
  if (old !== null) {
    const [typ, spelled, right] = old;
    const suggestion = oldFormSuggestion(bindingLabel, typ, spelled, right, {
      registry,
      placeholder: "${...}",
    });
    // Builtin names such as `date` or `file` are also words, so the text may be meant literally.
    throw diagnostics.error(
      `binding ${reprText(bindingLabel)}: write the type as a tag: ${suggestion}. ` +
        `If you meant the text, quote it: \`${bindingLabel}: ${yamlQuoted((value as string).trim())}\``,
      { range: cursor !== null ? cursor.range : null },
    );
  }
  // Only an authored mapping or list can hold nested spans or tags; a string stays as written,
  // and a carrier or a value replaced after parsing is not what the cursor shows, so it is kept
  // as given.
  if (cursor === null || !(isRecord(value) || isList(value)) || !cursor.holds(value as Value)) return null;
  return [readTaggedValue(cursor, registry, readOptions).value, null];
}

/**
 * Split each typed `input:` binding into its value and declared contract.
 *
 * A binding tagged `qty: !int ${size.output.qty}` keeps `${size.output.qty}` as its value, and
 * the compiled `int` contract lands in the descriptor's `inputTypes` under the parameter name;
 * the retained cursor locates both errors and the contract origin.
 *
 * @param descriptors Located descriptors keyed by node id.
 * @param nodesCursor Retained view of the flow's node mapping.
 * @param options.registry The file's typedefs the type text resolves against.
 * @param options.owner Complete lexical owner path for catalog sites.
 * @param options.catalogBuilder File-owned origin catalog builder.
 * @param options.diagnostics Factory for located load errors.
 * @returns The descriptors, with typed inputs split; a value entry, already read through its
 *   tag, is returned unchanged.
 * @throws LoadError when a typed binding's type or value is malformed, a binding is in the old
 *   `Type = value` form, or a retained cursor cannot be traversed.
 */
export function splitTypedInputs(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  nodesCursor: SourceCursor,
  options: {
    registry: TypeRegistry;
    owner: readonly string[];
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
  },
): Map<string, NodeDescriptor> {
  const { registry, owner, catalogBuilder, diagnostics } = options;
  const split = new Map<string, NodeDescriptor>();
  for (const [nodeId, descriptor] of descriptors) {
    // A value entry's one input was read through its tag when it was parsed.
    if (Object.keys(descriptor.inputs).length === 0 || isValueEntry(descriptor)) {
      split.set(nodeId, descriptor);
      continue;
    }
    const inputs = { ...descriptor.inputs };
    const inputTypes = new Map(descriptor.inputTypes);
    const changed = locateSourceAccess(diagnostics, () => {
      let changed = false;
      const inputsCursor = authoredField(nodesCursor, nodeId, "inputs");
      for (const [name, value] of Object.entries(descriptor.inputs)) {
        const cursor = inputsCursor !== null && inputsCursor.has(name) ? inputsCursor.child(name) : null;
        const typed = splitTypedValue(value, cursor, {
          bindingLabel: name,
          sitePath: [...owner, "node", nodeId, "input", name],
          registry,
          catalogBuilder,
          diagnostics,
        });
        if (typed === null) continue;
        const [splitValue, contract] = typed;
        inputs[name] = splitValue;
        if (contract !== null) inputTypes.set(name, contract);
        changed = true;
      }
      return changed;
    });
    split.set(nodeId, changed ? descriptor.replace({ inputs, inputTypes }) : descriptor);
  }
  return split;
}

/**
 * Read each AGENT's and adaptive asking node's `llm_config:` with the values and tags in it.
 *
 * An AGENT's own layer and the `adaptive_questions.llm_config` of a TOOL `ask` are read. A layer
 * holding values becomes a `ValueTemplate` (a `Binding` for a whole record written as one value);
 * a layer without values stays a plain record. The value range of each key in a mapping layer,
 * literal or not, lands in the descriptor's `fieldRanges` under
 * `siteKey("llm_config", "<dotted.path>")`, so its errors point at the field; the adaptive desugar
 * copies both to the composer agent.
 *
 * @param descriptors Located descriptors keyed by node id.
 * @param nodesCursor Retained view of the flow's node mapping.
 * @param options.registry The file's typedefs the tags resolve against.
 * @param options.owner Complete lexical owner path for catalog sites.
 * @param options.catalogBuilder File-owned origin catalog builder.
 * @param options.diagnostics Factory for located load errors.
 * @returns The descriptors, each layer read.
 * @throws LoadError when a tag in a layer is malformed or misplaced, or a retained cursor can't be
 *   traversed.
 */
function splitTypedLlmConfig(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  nodesCursor: SourceCursor,
  options: {
    registry: TypeRegistry;
    owner: readonly string[];
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
  },
): Map<string, NodeDescriptor> {
  const { registry, owner, catalogBuilder, diagnostics } = options;
  const split = new Map(descriptors);
  locateSourceAccess(diagnostics, () => {
    for (const [nodeId, descriptor] of descriptors) {
      let path: readonly string[];
      let written: unknown;
      let prefix: string;
      if (descriptor instanceof AgentDescriptor) {
        path = ["llm_config"];
        written = descriptor.llmConfig;
        prefix = `node ${reprText(nodeId)}: llm_config`;
      } else if (
        descriptor instanceof ToolDescriptor &&
        descriptor.toolId === ASK &&
        isRecord(descriptor.adaptiveQuestions)
      ) {
        path = ["adaptive_questions", "llm_config"];
        written = descriptor.adaptiveQuestions["llm_config"];
        prefix = `node ${reprText(nodeId)} adaptive_questions.llm_config`;
      } else {
        continue;
      }
      const cursor = authoredField(nodesCursor, nodeId, ...path);
      if (cursor === null || !cursor.holds(written as Value)) continue;
      const [layer, ranges] = readLlmConfig(cursor, registry, {
        prefix,
        sitePath: [...owner, "node", nodeId, ...path],
        catalogBuilder,
        diagnostics,
      });
      const fieldRanges = new Map(descriptor.fieldRanges);
      for (const [key, range] of ranges) fieldRanges.set(siteKey("llm_config", key), range);
      split.set(
        nodeId,
        descriptor instanceof AgentDescriptor
          ? descriptor.replace({ llmConfig: layer, fieldRanges })
          : descriptor.replace({
              adaptiveQuestions: { ...(descriptor.adaptiveQuestions as Record<string, unknown>), llm_config: layer },
              fieldRanges,
            }),
      );
    }
  });
  return split;
}

/**
 * Wire each AGENT's `llm_config` values as its reserved `__llm_config` input.
 *
 * Each value in an agent's own layer, and in the flow's layer, gets its static type, read over the
 * flow's contracts as written (no CASE narrowing applies to a config). An agent gets the key when
 * a value in its own layer, or one in the flow's layer that its own layer doesn't always
 * override, can reach its config. A higher value overrides only when its type can't be null. The
 * key carries both layers whole (`splitLayers`), so edges, reference checks, skips and child
 * re-namespacing treat the values as any other binding, and a dropped value adds no edge.
 *
 * The agent's runtime binding errors about the key name its `llm_config` field (an adaptive
 * composer's, the ask's `adaptive_questions.llm_config`).
 *
 * @param buildResults Built nodes keyed by final descriptor id.
 * @param flowWiring Mutable binding maps keyed by node id.
 * @param flowLlmConfig The flow's authored `llm_config:` layer: a record, a `ValueTemplate` or a
 *   `Binding`.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param options.typeFn Resolves a constructor name to its type for inference.
 * @returns The static type of each value in the flow's layer by its field path; `null` when it is
 *   not proven.
 */
function attachLlmConfigWiring(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowWiring: ReadonlyMap<string, Map<string, unknown>>,
  flowLlmConfig: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { typeFn?: TypeFn | null } = {},
): (readonly [ValuePath, Type | null])[] {
  const { typeFn = null } = options;
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    if (!(node instanceof AgentNode)) continue;
    node.llmValueTypes = llmValueTypes(node.ownLlmConfig, producers, flowInputs, typeFn);
    const carried = splitLayers(node.ownLlmConfig, flowLlmConfig, node.llmInherit, node.llmNullable());
    if (carried !== null) {
      flowWiring.get(nodeId)!.set(LLM_CONFIG_KEY, carried);
      node.wiresLlmConfig = true;
      node.bindingLabels = new Map([
        ...node.bindingLabels,
        [LLM_CONFIG_KEY, result.siteName(nodeId, node.reservedKeyLabel(LLM_CONFIG_KEY))],
      ]);
    }
  }
  return llmValueTypes(flowLlmConfig, producers, flowInputs, typeFn);
}

/** Return the static type of each value in an `llm_config` layer, by field path. */
function llmValueTypes(
  layer: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  typeFn: TypeFn | null,
): (readonly [ValuePath, Type | null])[] {
  return valueLeaves(layer).map(([path, leaf]) => {
    try {
      return [path, inferBindingType(leaf, producers, flowInputs, { typeFn })] as const;
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      return [path, null] as const; // the reference pass reports it, located
    }
  });
}

/**
 * Return the authored YAML scalar style of one node binding, or of the leaf at `path` inside it.
 *
 * A value entry's one binding is the entry itself, so its style is the entry's.
 *
 * @param nodesCursor Retained view of the flow's node mapping.
 * @param nodeId The node that owns the binding.
 * @param field The binding field (`inputs`, or a scalar field such as `over`).
 * @param key The binding name in a mapping field; equals `field` for a scalar one.
 * @param path The leaf's path inside the binding; `[]` for the binding itself.
 * @param options.valueEntries The ids of the flow's value entries.
 * @param options.diagnostics Factory for located load errors.
 * @returns The style, or `null` for a synthesized node, binding or leaf.
 * @throws LoadError when the retained cursor cannot be traversed.
 */
function bindingStyle(
  nodesCursor: SourceCursor,
  nodeId: string,
  field: string,
  key: string,
  path: ValuePath,
  options: { valueEntries: ReadonlySet<string>; diagnostics: LoadDiagnostics },
): ScalarStyle | null {
  const { valueEntries, diagnostics } = options;
  return locateSourceAccess(diagnostics, () => {
    if (!nodesCursor.has(nodeId)) return null;
    if (valueEntries.has(nodeId)) {
      if (field !== "inputs" || key !== nodeId) return null;
      return styleAlong(nodesCursor.child(nodeId), path);
    }
    const cursor = authoredField(nodesCursor, nodeId, ...(key === field ? [field] : [field, key]));
    return cursor === null ? null : styleAlong(cursor, path);
  });
}

/**
 * Return the YAML scalar style of the value at `path` under `cursor`, or `null` when the path
 * leaves the authored value.
 */
function styleAlong(cursor: SourceCursor, path: ValuePath): ScalarStyle | null {
  for (const part of path) {
    const held =
      typeof part === "string" ? cursor.has(part) : Array.isArray(cursor.value) && part < cursor.value.length;
    if (!held) return null;
    cursor = cursor.child(part);
  }
  return cursor.scalarStyle;
}

/**
 * Return the authored YAML scalar style of one flow output binding, or of the leaf at `path`
 * inside it.
 *
 * @param outputsCursor Retained authored `output:` value.
 * @param key The output's record field name, or `null` for a single output.
 * @param path The leaf's path inside the binding; `[]` for the binding itself.
 * @param options.diagnostics Factory for located load errors.
 * @returns The style, or `null` for a generated section or leaf.
 * @throws LoadError when the retained cursor cannot be traversed.
 */
function outputStyle(
  outputsCursor: SourceCursor | null,
  key: string | null,
  path: ValuePath,
  options: { diagnostics: LoadDiagnostics },
): ScalarStyle | null {
  const { diagnostics } = options;
  if (outputsCursor === null) return null;
  return locateSourceAccess(diagnostics, () => {
    if (key === null) return styleAlong(outputsCursor, path);
    if (!outputsCursor.has(key)) return null;
    return styleAlong(outputsCursor.child(key), path);
  });
}

/**
 * Return the authored `output:` of one node, or `null` when it has none.
 *
 * @param nodesCursor Retained view of the flow's node mapping.
 * @param nodeId The case node's id.
 * @param options.diagnostics Factory for located load errors.
 * @returns The `output:` value, or `null` for a synthesized node or a node without `output:`.
 * @throws LoadError when the retained cursor cannot be traversed.
 */
function caseOutputCursor(
  nodesCursor: SourceCursor,
  nodeId: string,
  options: { diagnostics: LoadDiagnostics },
): SourceCursor | null {
  const { diagnostics } = options;
  return locateSourceAccess(diagnostics, () => authoredField(nodesCursor, nodeId, "outputs"));
}

/**
 * Put each declared input type on its node's final wiring binding.
 *
 * Desugaring may rewrite a typed binding's value (an inline `call(...)` becomes a synthetic
 * node's output, a `${case.output}` becomes a coalesce), so the contract is keyed by parameter
 * name rather than by value.
 *
 * @param descriptors Final descriptors keyed by node id.
 * @param flowWiring Mutable final binding maps keyed by node id.
 * @throws RangeError, a compiler bug, when a node has a declared input type but no wiring binding
 *   by that name.
 */
export function attachInputContracts(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  flowWiring: ReadonlyMap<string, Map<string, unknown>>,
): void {
  for (const [nodeId, descriptor] of descriptors) {
    const wiring = flowWiring.get(nodeId);
    if (wiring === undefined) continue;
    for (const [name, contract] of descriptor.inputTypes) {
      // Every kind keeps its authored inputs in its wiring, so a declared type with no binding
      // to land on would be silently unchecked at runtime.
      if (!wiring.has(name)) {
        throw new RangeError(`node ${reprText(nodeId)}: typed input ${reprText(name)} has no wiring binding`);
      }
      const binding = asBinding(wiring.get(name));
      wiring.set(name, new Binding(binding.source, { origin: binding.origin, contract }));
    }
  }
}

/**
 * Split each typed flow output into its value and declared contract.
 *
 * A record field written `decision: !Decision ${d.output}` keeps `${d.output}` as its value; a
 * bare `output: !Size ${size.output}` is the single output named `result`. A tagged whole section
 * is that single output even when its value is a mapping (`output: !T {a: 1}`), so it comes back
 * as a `Binding` carrier rather than a record. Recognition and locations follow the retained
 * cursor exactly as for node inputs.
 *
 * An untyped output needs no check that its literals are JSON values: the YAML
 * reader keeps a date as text and refuses `.nan`, a float too large to read, `!!set`, `!!binary`,
 * `!!omap` and `!!pairs`, and the source layer refuses a non-text record key.
 *
 * @param outputsSection The raw `output:` section: a record, a single value, or `null`.
 * @param outputsCursor Retained authored `output:` value, or `null` for a generated section.
 * @param options.registry The file's typedefs the type text resolves against.
 * @param options.owner Complete lexical owner path for catalog sites.
 * @param options.catalogBuilder File-owned origin catalog builder.
 * @param options.diagnostics Factory for located load errors.
 * @returns The section with typed values split (a `Binding` for a tagged whole section), and the
 *   declared contracts keyed by output name.
 * @throws LoadError when a typed output's type or value is malformed, an output is in the old
 *   `Type = value` form, or the retained cursor cannot be traversed.
 */
function splitTypedOutputs(
  outputsSection: unknown,
  outputsCursor: SourceCursor | null,
  options: {
    registry: TypeRegistry;
    owner: readonly string[];
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
  },
): [outputsSection: unknown, outputTypes: Map<string, ValueContract>] {
  const { registry, owner, catalogBuilder, diagnostics } = options;
  /** Split one output value at its positional catalog site. */
  const splitOne = (value: unknown, cursor: SourceCursor | null, bindingLabel: string, index: number) =>
    splitTypedValue(value, cursor, {
      bindingLabel,
      sitePath: [...owner, "output", index],
      registry,
      catalogBuilder,
      diagnostics,
    });

  return locateSourceAccess(diagnostics, () => {
    if (outputsCursor?.tag != null) {
      // A tagged whole section is one typed value, even a mapping: the carrier is not a record,
      // so every later pass takes it as the single output. A top-level tag always gives the
      // split its contract.
      const [value, contract] = splitOne(outputsSection, outputsCursor, "output", 0)!;
      return [new Binding(value, { contract }), new Map([[SINGLE_OUTPUT_NAME, contract!]])];
    }
    if (outputsSection === null) return [null, new Map()];
    if (!isRecord(outputsSection)) {
      const typed = splitOne(outputsSection, outputsCursor, "output", 0);
      if (typed === null) return [outputsSection, new Map()];
      const [value, contract] = typed;
      if (contract === null) return [value, new Map()];
      return [value, new Map([[SINGLE_OUTPUT_NAME, contract]])];
    }
    const split: Record<string, unknown> = { ...outputsSection };
    const outputTypes = new Map<string, ValueContract>();
    for (const [index, [key, value]] of Object.entries(outputsSection).entries()) {
      const cursor = outputsCursor !== null && outputsCursor.has(key) ? outputsCursor.child(key) : null;
      const typed = splitOne(value, cursor, key, index);
      if (typed === null) continue;
      const [splitValue, contract] = typed;
      split[key] = splitValue;
      if (contract !== null) outputTypes.set(key, contract);
    }
    return [split, outputTypes];
  });
}

/**
 * Read the flow `output:` section into carriers for the boundary EndNode.
 *
 * A name to binding map becomes one `FlowOutput` per entry (a record-mode END returns a record
 * keyed by every name, `{}` for an empty map); a bare whole-string binding (`${note.output}`) or
 * a tagged section's `Binding` carrier is one output named `result` (a value-mode END returns the
 * bare value). A `null` section declares no outputs (a value-mode END returns `null`).
 *
 * @param outputs The raw `output:` section: a name to binding map, a single binding string, a
 *   tagged single output's `Binding`, or `null`.
 */
export function flowOutputs(outputs: unknown): FlowOutput[] {
  if (outputs === null) return [];
  if (isRecord(outputs)) {
    return Object.entries(outputs).map(([name, value]) => new FlowOutput(name, value));
  }
  return [new FlowOutput(SINGLE_OUTPUT_NAME, outputs)];
}

/**
 * Attach authored output values to their final canonical bindings.
 *
 * Desugaring has already produced each final binding source, while the cursor retains the
 * original occurrence. Mapping keys identify their effective raw source children, but only
 * values receive origins. A generated or source-free output has no cursor; an explicitly
 * originated binding remains untouched.
 *
 * @param outputs Final normalized output carriers in section order.
 * @param outputsSection Final post-desugar output values used to align mapping keys.
 * @param outputsCursor Retained authored `output:` value, or `null` for generated output.
 * @param options.owner Complete lexical owner path used to namespace output sites.
 * @param options.catalogBuilder File-owned origin catalog builder shared by all lexical children.
 * @param options.diagnostics Factory for located load errors.
 * @returns Output carriers whose physically authored bindings own their origin.
 * @throws RangeError when the normalized carriers do not align with the section.
 * @throws LoadError when the retained cursor cannot be traversed or a catalog site collides with
 *   another registration.
 */
export function stampFlowOutputOrigins(
  outputs: FlowOutput[],
  outputsSection: unknown,
  outputsCursor: SourceCursor | null,
  options: { owner: readonly string[]; catalogBuilder: SourceCatalogBuilder; diagnostics: LoadDiagnostics },
): FlowOutput[] {
  const { owner, catalogBuilder, diagnostics } = options;
  if (outputsCursor === null || outputs.length === 0) return outputs;

  let cursors: SourceCursor[];
  if (isRecord(outputsSection)) {
    const rawKeys = Object.keys(outputsSection);
    if (rawKeys.length !== outputs.length) {
      throw new RangeError("flow output carriers do not align with their section");
    }
    cursors = locateSourceAccess(diagnostics, () => rawKeys.map((key) => outputsCursor.child(key)));
  } else {
    if (outputs.length !== 1) throw new RangeError("a bare flow output requires exactly one carrier");
    cursors = [outputsCursor];
  }

  return outputs.map((output, index) => {
    if (output.binding.origin !== null) return output;
    const origin = locateSourceAccess(diagnostics, () =>
      catalogBuilder.register([...owner, "output", index, "binding"], cursors[index]!),
    );
    return origin === null ? output : new FlowOutput(output.name, withOrigin(output.binding, origin));
  });
}

/**
 * Locate each inline `call(...)` node and its arguments at the binding it was written in.
 *
 * A synth call node has no authored entry of its own, so its argument bindings and the node itself
 * take the origin of the authored binding named by its `callSite`: a failing argument then points
 * at that binding, and the child flow's failures get it as their outer frame. A node input, MAP
 * `over:`, flow output, or flow assert already owns an origin, which the call shares rather than
 * registering a second site, because the argument sits inside that authored scalar. A CASE
 * `then:`/`else:` target owns no binding origin, so its site is registered once here; a nested
 * call reuses it. An argument or node that already has an origin keeps it.
 *
 * @param descriptors Final descriptors keyed by node id after every graph desugar.
 * @param nodesCursor Retained source view of the flow's authored node mapping.
 * @param nodes Built runtime nodes keyed by final descriptor id; a synth call node's `origin` is
 *   set in place.
 * @param flowWiring Mutable final binding maps keyed by node id and wiring key, with the authored
 *   node bindings already originated.
 * @param flowOutputs Final output carriers, with their authored origins already stamped.
 * @param flowAsserts The flow's `asserts:` carriers in authored order, or `null`.
 * @param options.owner Complete lexical owner path used to namespace catalog sites.
 * @param options.catalogBuilder File-owned origin catalog builder shared by all lexical children.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError when a retained source cursor cannot be traversed or a catalog site collides
 *   with another registration.
 */
function stampInlineCallOrigins(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  nodesCursor: SourceCursor,
  nodes: ReadonlyMap<string, Node>,
  flowWiring: ReadonlyMap<string, Map<string, unknown>>,
  flowOutputs: readonly FlowOutput[],
  flowAsserts: readonly Assertion[] | null,
  options: { owner: readonly string[]; catalogBuilder: SourceCatalogBuilder; diagnostics: LoadDiagnostics },
): void {
  const { owner, catalogBuilder, diagnostics } = options;

  /** Return the origin already stamped on one node binding, if any. */
  const wiringOrigin = (host: string, key: string): OriginRef | null => {
    const source = flowWiring.get(host)?.get(key);
    return source === undefined ? null : asBinding(source).origin;
  };

  /** Register and return the origin of one CASE `then:` or `else:` branch. */
  const branchOrigin = (address: readonly (string | number | null)[]): OriginRef | null => {
    const caseId = address[1] as string;
    let cursor: SourceCursor | null;
    let path: SourceSitePath;
    if (address[0] === "else") {
      cursor = authoredField(nodesCursor, caseId, "else");
      path = [...owner, "node", caseId, "field", "else"];
    } else {
      const index = address[2] as number;
      cursor = authoredField(nodesCursor, caseId, "cases", index, "then");
      path = [...owner, "node", caseId, "case", index, "then"];
    }
    return cursor === null ? null : catalogBuilder.register(path, cursor);
  };

  /** Return the origin of the authored binding at one `callSite` address. */
  const siteOrigin = (address: readonly (string | number | null)[]): OriginRef | null => {
    const kind = address[0];
    if (kind === "input" || kind === "field") return wiringOrigin(address[1] as string, address[2] as string);
    if (kind === "output") {
      const name = address[1] ?? SINGLE_OUTPUT_NAME;
      return flowOutputs.find((flowOutput) => flowOutput.name === name)?.binding.origin ?? null;
    }
    if (kind === "assert") {
      const index = address[1] as number;
      if (flowAsserts === null || index >= flowAsserts.length) return null;
      return flowAsserts[index]!.origin;
    }
    return branchOrigin(address);
  };

  // The calls of one branch share its address; registering it twice would collide. Keyed by the
  // address's JSON, since arrays don't compare by value.
  const origins = new Map<string, OriginRef | null>();
  for (const [callId, descriptor] of descriptors) {
    if (!(descriptor instanceof CallDescriptor) || descriptor.callSite === null) continue;
    const address = descriptor.callSite;
    const addressKey = JSON.stringify(address);
    if (!origins.has(addressKey)) {
      locateSourceAccess(diagnostics, () => origins.set(addressKey, siteOrigin(address)));
    }
    const origin = origins.get(addressKey)!;
    if (origin === null) continue;
    const wiring = flowWiring.get(callId);
    if (wiring !== undefined) {
      for (const [key, source] of wiring) {
        if (asBinding(source).origin === null) wiring.set(key, withOrigin(source, origin));
      }
    }
    const node = nodes.get(callId);
    if (node !== undefined && node.origin === null) node.origin = origin;
  }
}

/**
 * Return a site's own range, else its node field's range, else the node's range.
 *
 * @param item The site's own range, `null` when it has none.
 * @param fieldRanges The node's field ranges, keyed like `BuildResult.fieldRanges`.
 * @param field The node field holding the site, e.g. `"asserts"`.
 * @param nodeRange The node's range.
 * @returns The narrowest range available.
 */
function siteRange(
  item: SourceRange | null | undefined,
  fieldRanges: ReadonlyMap<string, SourceRange>,
  field: string,
  nodeRange: SourceRange | null,
): SourceRange | null {
  return item ?? fieldRanges.get(siteKey("field", field)) ?? nodeRange;
}

/** The `isinstance` type names and the constructor names one expression site uses. */
type TypeNames = [isinstanceNames: string[], constructors: string[]];

/**
 * Set each node's `types` to the type names its `isinstance` calls and constructors use.
 *
 * A node evaluates its wiring sources, its asserts, a CASE's `when:`s and a LOOP's predicate with
 * its own table. Names resolve against this flow's `registry`; a constructor name must be a
 * `typedefs:` name.
 *
 * @param nodes Every node of the flow, START/END included, keyed by id. Mutated in place.
 * @param flowWiring Each node's `param -> source` wiring.
 * @param buildResults The body nodes' build results, for their field ranges and input site names.
 * @param descriptors The body nodes' descriptors, for their authored asserts and ranges.
 * @param flowAsserts The flow's authored `asserts:` carriers, aligned with `flowAssertionRanges`.
 * @param flowAssertionRanges Each flow assert's range.
 * @param outputRange One flow output's authored range, else the whole `output:` range.
 * @param registry The flow's `typedefs:` registry.
 * @param diagnostics The flow document's diagnostic factory, for located errors.
 * @throws LoadError at the offending site for an unknown type or constructor name, or for
 *   `isinstance` or a constructor in a reserved wiring key (WAIT `until:`, MAP `over:`).
 */
function fillTypeTables(
  nodes: ReadonlyMap<string, Node>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  buildResults: ReadonlyMap<string, BuildResult>,
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  flowAsserts: readonly Assertion[],
  flowAssertionRanges: readonly (SourceRange | null)[],
  outputRange: (name: unknown) => SourceRange | null,
  registry: TypeRegistry,
  diagnostics: LoadDiagnostics,
): void {
  const resolved = new Map<string, Type>();

  /** Resolve `names`, throwing at `range` for an unknown one. */
  const table = (names: TypeNames, where: string, range: SourceRange | null): Map<string, Type> => {
    const [isinstanceNames, constructors] = names;
    for (const name of isinstanceNames) {
      if (resolved.has(name)) continue;
      try {
        resolved.set(name, resolveIsinstanceType(name, registry));
      } catch (error) {
        if (!(error instanceof TypeCheckError)) throw error;
        throw diagnostics.error(`${where}: ${error.message}`, { range, cause: error });
      }
    }
    for (const name of constructors) {
      // A typedef name resolves the same for isinstance and constructors.
      if (resolved.has(name) && registry.has(name)) continue;
      try {
        resolved.set(name, resolveConstructorType(name, registry));
      } catch (error) {
        if (!(error instanceof TypeCheckError)) throw error;
        throw diagnostics.error(`${where}: ${error.message}`, { range, cause: error });
      }
    }
    return new Map([...isinstanceNames, ...constructors].map((name) => [name, resolved.get(name)!]));
  };

  /**
   * Resolve `names` as `table` does, then reject a constructor call in `source` that can never
   * succeed whatever the references hold, a field read on a builtin's result its type can't
   * have, or a builtin argument or subscript of a kind it never takes, such as a literal.
   *
   * Only for condition sites: `validateReferences` already checked every binding's constructor
   * calls and builtin field reads, with the references typed. Every
   * caller parses its source as a condition.
   */
  const checked = (names: TypeNames, source: unknown, where: string, range: SourceRange | null): Map<string, Type> => {
    const resolvedNames = table(names, where, range);
    let trees: ExprNode[];
    try {
      trees = expressionTrees(source, { condition: true });
    } catch (error) {
      // An earlier pass reports a source that won't parse.
      if (error instanceof ExpressionError) return resolvedNames;
      throw error;
    }
    for (const tree of trees) {
      const problems = names[1].length > 0 ? constructorErrors(tree, registry) : [];
      problems.push(
        ...builtinFieldErrors(tree),
        ...builtinArgumentErrors(tree),
        ...subscriptErrors(tree),
        ...fieldReadErrors(tree),
      );
      if (problems.length > 0) throw diagnostics.error(`${where}: ${problems[0]}`, { range });
    }
    return resolvedNames;
  };

  /** The `isinstance` and constructor names `source` uses; earlier passes report a source that won't parse. */
  const namesIn = (source: unknown, { condition = false }: { condition?: boolean } = {}): TypeNames => {
    try {
      return [collectIsinstanceNames(source, { condition }), collectConstructorNames(source, { condition })];
    } catch (error) {
      if (error instanceof ExpressionError) return [[], []];
      throw error;
    }
  };

  // Authored flow asserts first, so an unknown name is located at its own assert. The lowered
  // START/END asserts below use the same names.
  flowAsserts.forEach((assertion, index) => {
    const where = `flow assert ${reprText(assertion.display)}`;
    const names = namesIn(assertion.expression, { condition: true });
    checked(names, assertion.expression, where, flowAssertionRanges[index] ?? null);
  });

  for (const [nodeId, node] of nodes) {
    const result = buildResults.get(nodeId);
    const desc = descriptors.get(nodeId);
    const fieldRanges = result?.fieldRanges ?? new Map<string, SourceRange>();
    const nodeRange = result?.nodeRange ?? null;
    const types = new Map<string, Type>();
    const reserved = node.reservedWiringKeys();
    for (const [param, source] of flowWiring.get(nodeId) ?? []) {
      let where: string;
      let range: SourceRange | null;
      if (node.kind === NodeKind.END) {
        // END's wiring carries the flow outputs.
        where = node.bindingLabels.get(param) ?? `flow output ${reprText(param)}`;
        range = outputRange(param);
      } else {
        // A synthesized param's label may end in ':' (`on:`); the message adds one.
        const label = node.wiringRefLabel(param);
        const site = label.endsWith(":") ? label.slice(0, -1) : label;
        where = result !== undefined ? result.siteName(nodeId, site) : `node ${reprText(nodeId)} ${site}`;
        const key = reserved.has(param) ? siteKey("field", param) : siteKey("input", param);
        range = siteRange(fieldRanges.get(key), fieldRanges, "inputs", nodeRange);
      }
      const names = namesIn(source);
      if (reserved.has(param) && (names[0].length > 0 || names[1].length > 0)) {
        const used = names[0].length > 0 ? "isinstance" : `${names[1][0]}(...)`;
        throw diagnostics.error(
          `node ${reprText(nodeId)}: ${used} is not supported in a ${node.kind} node's ${param}:`,
          { range },
        );
      }
      for (const [name, typ] of table(names, where, range)) types.set(name, typ);
    }
    // Not every descriptor has `asserts` (MAP and WAIT don't), and a stand-in's may be null.
    const authored =
      (desc !== undefined && "asserts" in desc ? (desc.asserts as readonly string[] | null) : null) ?? [];
    const assertRanges = desc?.assertRanges ?? [];
    authored.forEach((expression, index) => {
      const where = `node ${reprText(nodeId)} assert ${reprText(expression)}`;
      const range = siteRange(nth(assertRanges, index), fieldRanges, "asserts", nodeRange);
      const names = namesIn(expression, { condition: true });
      for (const [name, typ] of checked(names, expression, where, range)) types.set(name, typ);
    });
    if (node.kind === NodeKind.START || node.kind === NodeKind.END) {
      // START/END evaluate the lowered flow asserts, whose names were located above.
      for (const assertion of [...node.preAsserts, ...node.postAsserts]) {
        const names = namesIn(assertion.expression, { condition: true });
        for (const [name, typ] of table(names, `flow assert ${reprText(assertion.display)}`, null)) {
          types.set(name, typ);
        }
      }
    }
    if (node instanceof CaseNode) {
      const whenRanges = desc instanceof CaseDescriptor ? desc.whenRanges : [];
      node.cases.forEach((branch, index) => {
        const where = `case node ${reprText(nodeId)} cases[${index}].when`;
        const range = siteRange(nth(whenRanges, index), fieldRanges, "cases", nodeRange);
        const names = namesIn(branch.when, { condition: true });
        for (const [name, typ] of checked(names, branch.when, where, range)) types.set(name, typ);
      });
    }
    if (node instanceof LoopNode && typeof node.predicate === "string") {
      const keyword = node.predicateKind;
      const where = `loop node ${reprText(nodeId)} ${keyword}`;
      const range = siteRange(null, fieldRanges, keyword, nodeRange);
      const names = namesIn(node.predicate, { condition: true });
      for (const [name, typ] of checked(names, node.predicate, where, range)) types.set(name, typ);
    }
    node.types = types;
  }
}

/**
 * Return the ids of the cases that produce a value.
 *
 * A case produces a value when it is in `read`; a case that is a `then:`/`else:` target of a
 * value-producing case produces one too.
 *
 * @param descriptors Node descriptors keyed by node id.
 * @param read The case ids some binding reads as `${<case>.output}`.
 * @returns The value-producing case ids.
 */
function valueCases(descriptors: ReadonlyMap<string, NodeDescriptor>, read: ReadonlySet<string>): ReadonlySet<string> {
  const cases = new Map<string, CaseDescriptor>();
  for (const [nodeId, d] of descriptors) if (d instanceof CaseDescriptor) cases.set(nodeId, d);
  const pending = [...cases.keys()].filter((nodeId) => read.has(nodeId));
  const found = new Set<string>();
  while (pending.length > 0) {
    const nodeId = pending.pop()!;
    if (found.has(nodeId)) continue;
    found.add(nodeId);
    const desc = cases.get(nodeId)!;
    for (const target of [...desc.cases.map((spec) => spec["then"]), desc.else_]) {
      if (typeof target === "string" && cases.has(target)) pending.push(target);
    }
  }
  return found;
}

/** The options of one assembly: the parsed sections and the scope it reads. */
export interface AssembleOptions {
  /** The authored `input:` mapping that still needs parsing; exclusive with `inputParameters`. */
  inputsSection?: Readonly<Record<string, unknown>> | null;
  /** Parameters an earlier compiler pass already resolved; exclusive with `inputsSection`. */
  inputParameters?: readonly Parameter[] | null;
  /** Retained occurrence of the authored `input:` mapping, or `null`. */
  inputsCursor?: SourceCursor | null;
  /** Retained view of the flow's `nodes:` mapping; owns diagnostic locations. */
  nodesCursor: SourceCursor;
  /** The raw `output:` section. */
  outputsSection: unknown;
  /** The raw flow `asserts:` items in authored order. */
  assertsSection: readonly unknown[];
  /** The file's typedefs. */
  registry: TypeRegistry;
  /** Owns composite call resolution (defs first) and private child registration. */
  scope: CallableScope;
  /** This flow's complete lexical path; seeds stable identities for anonymous children. */
  owner: readonly string[];
  /** The flow's `uses:` aliases, each resolved at load. */
  usesAliases?: readonly string[];
  /** The flow's declared `version:`. */
  version?: string | null;
  /** The flow's declared `name:`. */
  name?: string | null;
  /** The flow's declared `description:`. */
  description?: string | null;
  /** The flow-level `llm_config:`. */
  flowLlmConfig?: Readonly<Record<string, unknown>> | ValueTemplate | Binding | null;
  /** The flow-level `env:` default, merged under each node's own `env:`. */
  flowEnv?: Readonly<Record<string, unknown>> | null;
  /** Authored flow-input declaration ranges by input name, for source-free type compilation. */
  boundaryFieldRanges?: ReadonlyMap<string, SourceRange | null> | null;
  /** Retained flow `asserts:` sequence; the single authority for each assertion's origin and range. */
  assertsCursor?: SourceCursor | null;
  /** The retained authored `output:` range, when one exists. */
  outputsRange?: SourceRange | null;
  /** Retained occurrence of each physically authored output binding. */
  outputsCursor?: SourceCursor | null;
  /** The authored or generated semantic identity. */
  flowId?: string | null;
  /**
   * The directory the flow's CODE references resolve against, its inline bodies' too; `null` is
   * the current directory.
   */
  baseDir?: string | null;
}

/** What every phase of one assembly reads: its diagnostics, file scope, typedefs and lexical owner. */
interface AssembleContext {
  /** Factory for located load errors, labelled by the flow's document. */
  readonly diagnostics: LoadDiagnostics;
  /** Owns composite call resolution (defs first) and private child registration. */
  readonly scope: CallableScope;
  /** The file's one origin registry, `scope.catalogBuilder`. */
  readonly catalogBuilder: SourceCatalogBuilder;
  /** A bound `catalogBuilder.resolveRegistered`, which every pass presents origins through. */
  readonly resolveOrigin: OriginPresenter;
  /** The file's typedefs. */
  readonly registry: TypeRegistry;
  /** Static inference types a `T(...)` constructor by this flow's own typedefs. */
  readonly constructorType: TypeFn;
  /** This flow's complete lexical path. */
  readonly owner: readonly string[];
  /** The directory the flow's CODE references resolve against; `null` is the current directory. */
  readonly baseDir: string | null;
}

/** The flow's parameters and the views of them the passes read. */
interface FlowParameters {
  /** The parameters in declaration order. */
  readonly inputs: readonly Parameter[];
  /** Each typed parameter's contract by name. */
  readonly contracts: ReadonlyMap<string, ValueContract>;
  /** Each typed parameter's type by name. */
  readonly types: ReadonlyMap<string, Type>;
  /** Every parameter name. */
  readonly names: ReadonlySet<string>;
}

/** The flow's `output:` section and where it was written. */
interface OutputSection {
  /** The section with tagged bindings split to their values. */
  readonly section: unknown;
  /** Each tagged output's declared contract by output name. */
  readonly types: ReadonlyMap<string, ValueContract>;
  /** Retained occurrence of each physically authored output binding. */
  readonly cursor: SourceCursor | null;
  /** The whole `output:` range, when one exists. */
  readonly range: SourceRange | null;
  /** One flow output's authored range, else the whole `output:` range. */
  readonly rangeOf: (outputName: unknown) => SourceRange | null;
}

/** What the read phase found in the authored sections. */
interface ReadSections {
  /** The `nodes:` mapping with each inline body masked to a call of its private id. */
  readonly nodesCursor: SourceCursor;
  /** The unmasked `nodes:` mapping, which owns assertion provenance. */
  readonly authoredNodesCursor: SourceCursor;
  /** The retained inline bodies by node id. */
  readonly inlineBodies: ReadonlyMap<string, InlineBody>;
  /** Each authored flow assertion's range, `null` when source-free. */
  readonly flowAssertionRanges: readonly (SourceRange | null)[];
  /** Return the registered origin for one authored flow assertion. */
  readonly flowAssertionOriginAt: (index: number) => OriginRef | null;
  /** The flow's parameters. */
  readonly params: FlowParameters;
  /** The located node descriptors, tagged inputs and `llm_config` split, before any desugar. */
  readonly descriptors: ReadonlyMap<string, NodeDescriptor>;
  /** The ids of the tagged `nodes:` entries, which have no node body to re-read. */
  readonly valueEntries: ReadonlySet<string>;
  /** The flow's `output:` section. */
  readonly outputs: OutputSection;
}

/** The sections after every desugar. */
interface DesugaredSections {
  /** Descriptors with the synth nodes the desugars added. */
  readonly descriptors: ReadonlyMap<string, NodeDescriptor>;
  /** The `output:` section with inline calls and case reads rewritten. */
  readonly outputsSection: unknown;
  /** The flow assertions with inline calls rewritten. */
  readonly assertsSection: Assertion[];
  /** The flow `llm_config:` with case reads rewritten. */
  readonly flowLlmConfig: Readonly<Record<string, unknown>> | ValueTemplate | Binding | null;
  /** How errors about the flow `llm_config:` name it. */
  readonly flowLlmConfigLabel: string;
  /** The cases whose value is used; each must be exhaustive. */
  readonly valueCaseIds: ReadonlySet<string>;
  /** Each expanded case read as written, so an error names the read. */
  readonly caseSpellings: ReadonlyMap<string, CaseRead>;
}

/** The built nodes and their flow-owned wiring. */
interface BuiltNodes {
  /** The final descriptors, each inline composite labelled. */
  readonly descriptors: ReadonlyMap<string, NodeDescriptor>;
  /** Each node's build result by id. */
  readonly buildResults: ReadonlyMap<string, BuildResult>;
  /** Each built node by id. */
  readonly nodesBuilt: ReadonlyMap<string, Node>;
  /** `flowWiring[nodeId][param] -> source`, an entry for every node. */
  readonly flowWiring: Map<string, Map<string, unknown>>;
  /** Each case's own data and control edges. */
  readonly caseEdges: readonly Edge[];
  /** Each node's output contract, and each AGENT's new messages under `<id>.new_messages`. */
  readonly producers: ReadonlyMap<string, OutputContract>;
  /** The static type of each value in the flow's `llm_config` layer, by field path. */
  readonly flowLlmValueTypes: (readonly [ValuePath, Type | null])[];
}

/** The routed edges and the CASE narrowing they imply. */
interface RoutedEdges {
  /** Data, case and host edges, in that order. */
  readonly edges: readonly Edge[];
  /** The edges copied from hosts onto the nodes lifted out of them. */
  readonly hostEdges: readonly Edge[];
  /** What each node may assume about union values from the CASE branches that route it. */
  readonly narrowing: Narrowing;
}

/** What the reference pass computed for the passes after it. */
interface CheckedReferences {
  /** The proven contracts of each node-local name, narrowed. */
  readonly localContracts: Map<string, Map<string, LocalInput>>;
  /** The same, unnarrowed. */
  readonly baseLocals: Map<string, Map<string, LocalInput>>;
  /** The routed edges plus the ordering edges. */
  readonly structural: readonly Edge[];
  /** Which nodes may be skipped while a reader runs. */
  readonly skips: SkipModel;
  /** Each searched CASE's authored `when:`s. */
  readonly searchedWhens: ReadonlyMap<string, SearchedWhen[]>;
  /** The flow asserts lowered onto START/END. */
  readonly lowered: LoweredAsserts;
}

/** The whole graph, START_ID/END_ID included. */
interface ClosedFlow {
  /** Every node by id. */
  readonly nodes: Map<string, Node>;
  /** Every edge. */
  readonly edges: Edge[];
  /** Every node's wiring by id. */
  readonly wiring: Map<string, Map<string, unknown>>;
}

/** Read the flow's parameters from the authored `input:` mapping or the resolved parameters. */
function readFlowParameters(options: AssembleOptions, ctx: AssembleContext): FlowParameters {
  const { inputsSection = null, inputParameters = null, inputsCursor = null, boundaryFieldRanges = null } = options;
  const { registry, owner, catalogBuilder, diagnostics } = ctx;
  let inputs: Parameter[];
  if (inputsSection !== null) {
    if (inputsCursor !== null) {
      inputs = readSourceParameters(inputsCursor.withValue(inputsSection as Value), registry, {
        sitePath: owner,
        catalogBuilder,
        diagnostics,
        memoryTypes: ctx.scope.memoryTypes,
      });
    } else {
      const inputRanges = new Map(
        Object.keys(inputsSection).map((name) => [name, boundaryFieldRanges?.get(name) ?? null]),
      );
      inputs = readFlowInputs(inputsSection, registry, inputRanges, { diagnostics });
    }
  } else {
    inputs = [...inputParameters!];
  }
  const contracts = new Map(
    inputs.flatMap((parameter) => (parameter.contract === null ? [] : [[parameter.name, parameter.contract] as const])),
  );
  assertExactContracts(new Map(), contracts);
  const types = new Map([...contracts].map(([inputName, contract]) => [inputName, contract.type]));
  return { inputs, contracts, types, names: new Set(inputs.map((parameter) => parameter.name)) };
}

/**
 * Read the authored sections: extract inline bodies, stamp flow-assert origins, read the
 * parameters, parse and locate the nodes, refuse a colliding `uses:` alias, bound written-value
 * nesting, refuse a tagged node `output:`, split tagged values from their contracts, and check case
 * branches.
 */
function readSections(options: AssembleOptions, ctx: AssembleContext): ReadSections {
  const { usesAliases = [], assertsCursor = null, outputsCursor = null } = options;
  const { registry, scope, owner, catalogBuilder, diagnostics } = ctx;
  const authoredNodesCursor = options.nodesCursor;
  const outputsRange = options.outputsRange ?? outputsCursor?.range ?? null;

  // Inline-body extraction masks child-only fields for descriptor parsing. Assertion provenance
  // still belongs to the unmodified authored occurrence tree.
  const [nodesCursor, inlineBodies] = locateSourceAccess(diagnostics, () =>
    extractInlineBodies(authoredNodesCursor, { owner }),
  );

  rejectIncludeIn(assertsCursor, "asserts", "assertions", diagnostics);
  const flowAssertionCursors = assertionCursors(options.assertsSection, assertsCursor, diagnostics);
  const flowAssertionRanges = flowAssertionCursors.map((cursor) => (cursor !== null ? cursor.range : null));
  const flowAssertionOriginAt = (index: number): OriginRef | null =>
    registerAssertionOrigin(flowAssertionCursors[index]!, [...owner, "assert", index], catalogBuilder, diagnostics);

  const params = readFlowParameters(options, ctx);

  /** Lower one tagged `nodes:` entry after checking its nesting. */
  const readValue = (nodeId: string, cursor: SourceCursor): ValueEntryDescriptor => {
    // First: the tag read walks the value recursively.
    checkWrittenValue(cursor.value, `value entry ${reprText(nodeId)}`, cursor.range ?? cursor.keyRange, diagnostics);
    return readValueEntry(nodeId, cursor, { registry, owner, catalogBuilder, diagnostics });
  };

  const parsed = parseNodes(nodesCursor, { readValue });
  // Value entries have no node body, so every pass below that re-reads a node's body from
  // `nodesCursor` skips them or reads the entry itself.
  const valueEntries: ReadonlySet<string> = new Set(
    [...parsed].filter(([, descriptor]) => isValueEntry(descriptor)).map(([nodeId]) => nodeId),
  );
  const located = locateSourceAccess(diagnostics, () => {
    const locatedDescriptors = new Map<string, NodeDescriptor>();
    for (const [nodeId, descriptor] of parsed) {
      const authoredCursor = nodesCursor.child(nodeId);
      const nodePath = [...owner, "node", nodeId];
      const withOrigin = descriptor.replace({ origin: catalogBuilder.register(nodePath, authoredCursor, "key") });
      if (valueEntries.has(nodeId)) {
        locatedDescriptors.set(nodeId, withOrigin);
        continue;
      }
      locatedDescriptors.set(
        nodeId,
        withOrigin.withSourceOrigins(authoredField(nodesCursor, nodeId)!, { sitePath: nodePath, catalogBuilder }),
      );
    }
    return locatedDescriptors;
  });

  // A `uses:` alias shares the callable namespace, so it must not collide with a node id (the
  // resolver only sees a bare name; the author conflates them). The loader already checks the
  // authored node ids before it resolves aliases; this runs before any desugar, so it repeats
  // that check for a caller of `assemble` that skips the loader.
  const collide = [...new Set(usesAliases)].filter((alias) => located.has(alias)).sort(compareCodeUnits);
  if (collide.length > 0) throw diagnostics.error(`uses: alias ${writeAsText(collide)} collides with a node id`);

  // Every alias is resolved, so a declared-but-uncalled broken/`hub:`/missing target is loud at
  // load. The loader already resolved them in source order before the defs, so this only hits
  // its cache.
  for (const alias of usesAliases) scope.resolve(alias);

  // A record section's keys name END's inputs, beside the ones yian generates for asserts.
  if (isRecord(options.outputsSection) && outputsCursor?.tag == null) {
    for (const name of Object.keys(options.outputsSection)) {
      const range = locateSourceAccess(diagnostics, () => outputsCursor?.keyRangeOf(name) ?? null);
      rejectReservedName("flow output", name, range ?? outputsRange, diagnostics);
    }
  }

  /** One flow output's authored range, else the whole `output:` range. */
  const outputRange = (outputName: unknown): SourceRange | null => {
    try {
      if (outputsCursor !== null && typeof outputName === "string" && outputsCursor.has(outputName)) {
        return outputsCursor.child(outputName).range;
      }
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      // Swallowed: an `output:` at a recursive alias can't be walked, so the whole range stands in.
    }
    return outputsRange;
  };

  // Bound a written-out value's nesting before the passes below walk it recursively; YAML
  // aliases can make it deep while the text is shallow.
  checkWrittenValues(located, options.outputsSection, {
    outputsTagged: outputsCursor !== null && outputsCursor.tag !== null,
    outputRange,
    outputsRange,
    diagnostics,
  });

  // A node's `output:` is a type declaration; a tag there is rejected before a declaration with
  // nothing after its tag reads as no output at all. A case has no `output:`; the case-branch
  // check below rejects it, tagged or not, with the rewrite.
  locateSourceAccess(diagnostics, () => {
    for (const [nodeId, descriptor] of located) {
      if (descriptor.kind === "case") continue;
      if (valueEntries.has(nodeId)) continue;
      const outputCursor = authoredField(nodesCursor, nodeId, "outputs");
      if (outputCursor !== null) rejectTaggedOutput(outputCursor, diagnostics);
    }
  });

  // Split tagged (`name: !Type value`) input and output bindings before any desugar reads their
  // values, so an inline `call(...)` or `${case.output}` in the value desugars as usual.
  const splitOptions = { registry, owner, catalogBuilder, diagnostics };
  const typedInputs = splitTypedInputs(located, nodesCursor, splitOptions);
  const descriptors = splitTypedLlmConfig(typedInputs, nodesCursor, splitOptions);
  const [outputsSection, outputTypes] = splitTypedOutputs(options.outputsSection, outputsCursor, splitOptions);

  // A case branch names a node or calls a flow, and a case has no `output:`. Runs BEFORE the
  // inline-call desugar so a branch's text is still as authored.
  checkCaseBranches(descriptors, {
    registry,
    isCallable: (name) => scope.has(name),
    inputTypes: params.types,
    outputAt: (nodeId) => caseOutputCursor(nodesCursor, nodeId, { diagnostics }),
    diagnostics,
  });

  return {
    nodesCursor,
    authoredNodesCursor,
    inlineBodies,
    flowAssertionRanges,
    flowAssertionOriginAt,
    params,
    descriptors,
    valueEntries,
    outputs: {
      section: outputsSection,
      types: outputTypes,
      cursor: outputsCursor,
      range: outputsRange,
      rangeOf: outputRange,
    },
  };
}

/**
 * Desugar the read sections: inline calls, adaptive questions, case call targets, then
 * `${case.output}` to a coalesce over the case's branch targets.
 */
function desugarSections(
  read: ReadSections,
  assertsSection: readonly unknown[],
  flowLlmConfig: DesugaredSections["flowLlmConfig"],
  ctx: AssembleContext,
): DesugaredSections {
  const { registry, owner, resolveOrigin, diagnostics } = ctx;
  const { nodesCursor, valueEntries, outputs } = read;
  const flowAssertionRangeAt = (index: number): SourceRange | null => read.flowAssertionRanges[index]!;

  // Desugar each whole-value `call(<flow>, ...)` binding: it becomes a synth `call` node
  // (CallDescriptor, over=null) and its host binding is rewritten to `${<synth>.output}`. Pure
  // sugar — the rest of the pipeline treats the synth nodes as ordinary `call` nodes
  // (build/edges/validate/DAG run unchanged). The retired `${ f(...) }` form is refused here.
  const mint = synthIdGen(); // ONE synth-id minter shared by both inline-call desugars (no collision)
  const [withCalls, callOutputs, asserts] = desugarInlineCalls(read.descriptors, outputs.section, {
    assertsSection,
    outputsRange: outputs.range,
    assertionRangeAt: flowAssertionRangeAt,
    assertionOriginAt: read.flowAssertionOriginAt,
    bindingStyleAt: (nodeId, field, key, path) =>
      bindingStyle(nodesCursor, nodeId, field, key, path, { valueEntries, diagnostics }),
    outputStyleAt: (key, path) => outputStyle(outputs.cursor, key, path, { diagnostics }),
    typedOutputs:
      outputs.types.size > 0
        ? new Set<string | null>(isRecord(outputs.section) ? outputs.types.keys() : [null])
        : new Set(),
    nextId: mint,
    typeNames: new Set(registry.keys()),
    isCallable: (name) => ctx.scope.has(name),
    diagnostics,
  });

  // Lower each `ask` node's `adaptive_questions:` block into a synth composer agent + the
  // rewritten ask. A descriptor-level pass (the `desugarInlineCalls` precedent above): runs
  // BEFORE the build loop so the synth agent flows through build + prompt-scope validation + ref
  // wiring like any author-written agent.
  const withComposers = desugarAdaptiveQuestions(withCalls, { diagnostics });

  // `then:/else: call(...)`: a `case` branch target that is a call directive becomes a synth
  // `call` node, the then:/else: rewritten to its id. Runs BEFORE expandCaseOutputs (so the synth args +
  // the rewritten then-targets are visible to it) and shares `mint`.
  const withTargets = desugarCaseCallTargets(withComposers, mint, { diagnostics });

  // The cases whose value is used, taken before the expansion below rewrites the reads; each
  // must be exhaustive (checked once producer types are known).
  const valueCaseIds = valueCases(withTargets, caseOutputReads(withTargets, callOutputs, flowLlmConfig));

  // Expand `${<case>.output}` value refs to a coalesce over the case's branch targets. Runs after
  // the inline-call desugar (so synth-node bindings are covered too) and before the build loop —
  // downstream sees only ordinary branch-target refs.
  const flowLlmConfigLabel = owner[0] === "def" ? `defs entry ${reprText(owner[1]!)} llm_config` : "flow llm_config";
  // Each expanded case read, so a reference or field-read error names the read as written.
  const caseSpellings = new Map<string, CaseRead>();
  const [descriptors, outputsSection, expandedLlmConfig] = expandCaseOutputs(withTargets, callOutputs, asserts, {
    outputsRange: outputs.range,
    assertionRangeAt: flowAssertionRangeAt,
    flowLlmConfig,
    flowLlmConfigLabel,
    resolveOrigin,
    diagnostics,
    spellings: caseSpellings,
  });

  return {
    descriptors,
    outputsSection,
    assertsSection: asserts,
    flowLlmConfig: expandedLlmConfig as DesugaredSections["flowLlmConfig"],
    flowLlmConfigLabel,
    valueCaseIds,
    caseSpellings,
  };
}

/**
 * Register the inline bodies, then build every node kind-blind and stamp what the flow owns of
 * it: binding labels, wiring origins, input contracts, its merged `env` and its `llm_config` wiring.
 */
function buildNodes(
  read: ReadSections,
  desugared: DesugaredSections,
  flowEnv: Readonly<Record<string, unknown>> | null,
  ctx: AssembleContext,
): BuiltNodes {
  const { registry, scope, owner, catalogBuilder, resolveOrigin, constructorType, baseDir, diagnostics } = ctx;
  const { nodesCursor, valueEntries, params } = read;

  /**
   * Compile one physical node-output declaration at its build position.
   *
   * @param nodeId Node whose normalized declaration should be selected.
   * @param outputSpec Semantic type shape already retained by its descriptor.
   * @returns Located contract, or `null` for a virtual site.
   * @throws LoadError when source traversal or type compilation fails.
   */
  const readAuthoredNodeOutput = (nodeId: string, outputSpec: unknown): ValueContract | null => {
    return locateSourceAccess(diagnostics, () => {
      if (valueEntries.has(nodeId)) return null;
      const outputCursor = authoredField(nodesCursor, nodeId, "outputs");
      if (outputCursor === null || outputCursor.range === null) return null;
      // `withValue` refuses a non-YAML value itself.
      return readValueContract(outputCursor.withValue(outputSpec as Value), registry, {
        sitePath: [...owner, "node", nodeId, "output", "contract"],
        catalogBuilder,
        diagnostics,
      });
    });
  };

  /** Compile one physical node's variables through its normalized cursor. */
  const readAuthoredNodeVariables = (nodeId: string, declarations: unknown): Parameter[] | null => {
    return locateSourceAccess(diagnostics, () => {
      if (valueEntries.has(nodeId)) return null;
      const variablesCursor = authoredField(nodesCursor, nodeId, "variables");
      if (variablesCursor === null || variablesCursor.range === null) return null;
      return readSourceParameters(variablesCursor.withValue(declarations as Value), registry, {
        sitePath: [...owner, "node", nodeId],
        catalogBuilder,
        diagnostics,
        fieldName: "variables",
        declarationLabel: "variable",
        requireDefault: true,
      });
    });
  };

  // The inline-child builder and the binding-contract reader read it lazily; it is set just
  // before the build.
  let descriptorCache: DescriptorBuildCache | null = null;

  /** Assemble one retained body lazily in this file's callable scope. */
  const buildInlineChild = (body: InlineBody, descriptor: CallDescriptor | LoopDescriptor): ComposedFlow => {
    /** Resolve one parent producer through this assembly's build cache. */
    const producerType: ProducerType = (producerKey, bindingLabel, bindingRange) => {
      // `messagesKey(id)` names an AGENT's new messages.
      const messages = producerKey.endsWith(`.${NEW_MESSAGES}`);
      const producerId = messages ? producerKey.slice(0, -`.${NEW_MESSAGES}`.length) : producerKey;
      if (!desugared.descriptors.has(producerId)) return null;

      const cache = descriptorCache;
      if (cache === null) throw new RangeError("descriptor cache was not initialized before inline child resolution");
      const node = cache.buildOne(producerId, bindingLabel, bindingRange).node;
      if (messages) {
        if (node.producesMessages) return new ValueContract(newMessagesType());
        throw diagnostics.error(
          `cannot infer type for inline binding ${reprText(bindingLabel)}: node ${reprText(producerId)} ` +
            `has no ${NEW_MESSAGES}, which only an AGENT node has`,
          { range: bindingRange },
        );
      }
      if (node.outputContract === null) return null;
      assertExactContracts(new Map([[producerId, node.outputContract]]));
      return node.outputContract;
    };

    let itemType: ItemType | null = null;
    if (descriptor instanceof MapDescriptor) {
      let cachedItemType: Type | null = null;

      /** Prove the MAP element type once, only when an item walk requests it. */
      itemType = (bindingLabel, bindingRange) => {
        if (cachedItemType !== null) return cachedItemType;

        const overType = inferInlineBindingType(descriptor.over, {
          parentInputTypes: params.types,
          producerType,
          typeFn: constructorType,
          bindingLabel,
          bindingRange,
          diagnostics,
        });
        if (!isListKind(overType.kind)) {
          throw diagnostics.error(
            `cannot infer type for inline binding ${reprText(bindingLabel)}: ` +
              "MAP `over:` must have a proven list element type; " +
              explicitTypeHint(bindingLabel),
            { range: bindingRange },
          );
        }
        // Inference proves a list kind only with its element type.
        cachedItemType = overType.element!;
        return cachedItemType;
      };
    }

    const childInputs: Parameter[] = [];
    for (const [key, value] of Object.entries(descriptor.inputs)) {
      const bindingLabel = `${descriptor.id}.input.${key}`;
      const bindingRange = body.inputRanges.get(key) ?? null;
      const declared = descriptor.inputTypes.get(key);
      // A typed binding states the parameter's type, and the binder checks the value against it at
      // runtime, so its value need not be inferable.
      const parameterType =
        declared !== undefined
          ? declared.type
          : inferInlineBindingType(value, {
              parentInputTypes: params.types,
              producerType,
              typeFn: constructorType,
              itemType,
              bindingLabel,
              bindingRange,
              diagnostics,
            });
      const canonical = canonicalizeInlineType(parameterType);
      childInputs.push(
        new Parameter(key, {
          contract: new ValueContract(canonical),
          required: !canonical.nullable,
          coerceStrings: true,
        }),
      );
    }

    return assemble({
      inputParameters: childInputs,
      nodesCursor: body.nodesCursor,
      outputsSection: body.output,
      assertsSection: [],
      registry,
      scope,
      owner: body.owner,
      flowLlmConfig: {},
      flowEnv: {},
      boundaryFieldRanges: new Map(body.inputRanges),
      outputsRange: body.outputRange,
      outputsCursor: body.outputCursor,
      flowId: body.flowId,
      baseDir,
    });
  };

  /** Resolve one parent binding's contract, building referenced producers on demand. */
  const readBindingContract: BindingContractReader = (bindingLabel, binding, bindingRange) => {
    const cache = descriptorCache;
    if (cache === null) throw new RangeError("descriptor cache was not initialized before binding contract resolution");
    const producers = new Map<string, OutputContract>();
    for (const producerId of new Set(bindingProducers(binding))) {
      if (!desugared.descriptors.has(producerId)) continue;
      const node = cache.buildOne(producerId, bindingLabel, bindingRange).node;
      if (node.outputContract !== null) producers.set(producerId, node.outputContract);
      if (node.producesMessages) producers.set(messagesKey(producerId), new ValueContract(newMessagesType()));
    }
    return resolveOutputBindingContract(binding, producers, params.contracts, { typeFn: constructorType });
  };

  const descriptors = registerInlineBodies(desugared.descriptors, read.inlineBodies, {
    registrar: scope,
    buildChild: buildInlineChild,
  });

  // Build EVERY node kind-blind through its own descriptor's `build(ctx)` — leaf, call/map (a
  // map rides the call descriptor, no branch), loop, AND case (which lowers in its own `build`,
  // returning its data+control edges in `BuildResult.edges`). The conductor never branches on the
  // descriptor kind. Flow-owned wiring (the split): `flowWiring[nodeId][param] -> source`, an
  // empty entry for every node so the key-parity check ranges over all nodes.
  const ctxBase = new BuildContext({
    registry,
    resolver: (childId) => scope.resolve(childId),
    // Inline CODE receives the owning document label only as an opaque traceback file name;
    // authored locations come from semantic origins. Inert for reference-mode CODE.
    sourceFile: diagnostics.label || "<inline-code>",
    resolveOrigin,
    diagnostics,
    outputContractReader: readAuthoredNodeOutput,
    variableDeclarationReader: readAuthoredNodeVariables,
    bindingContractReader: readBindingContract,
    baseDir,
    flowInputs: params.types,
    memoryTypes: scope.memoryTypes,
  });
  descriptorCache = new DescriptorBuildCache(descriptors, ctxBase);
  const buildResults = descriptorCache.buildInSourceOrder();
  const nodesBuilt = new Map([...buildResults].map(([nodeId, result]) => [nodeId, result.node]));
  const flowWiring = new Map([...buildResults].map(([nodeId, result]) => [nodeId, result.wiring]));
  stampBindingLabels(buildResults);
  stampNodeWiringOrigins(descriptors, nodesCursor, nodesBuilt, flowWiring, { owner, catalogBuilder, diagnostics });
  attachInputContracts(descriptors, flowWiring);
  // A case's own data+control edges; every other kind returns an empty list.
  const caseEdges: Edge[] = [...buildResults.values()].flatMap((result) => result.edges);
  // Stamp each built node's effective static config: the flow-level `env:` default merged under
  // the node's own `env:` (node override wins), flow-local (a called child flow has its own flow
  // env). The engine core never reads `Node.env` — only the node's `forward()` path does.
  const baseEnv = flowEnv ?? {};
  for (const [nodeId, node] of nodesBuilt) node.env = { ...baseEnv, ...descriptors.get(nodeId)!.env };
  // producers: each built node's declared output contract (drives the case `on:` enum
  // exhaustiveness + the dotted-field walk in ref/assert validation), and each AGENT's new
  // messages under their pool name `<id>.new_messages` (ids hold no `.`). A case node has no
  // contract, so it is naturally excluded.
  const producers = new Map<string, OutputContract>();
  for (const [nodeId, node] of nodesBuilt) {
    if (node.outputContract !== null) producers.set(nodeId, node.outputContract);
  }
  assertExactContracts(producers);
  for (const [nodeId, node] of nodesBuilt) {
    if (node.producesMessages) producers.set(messagesKey(nodeId), new ValueContract(newMessagesType()));
  }

  const flowLlmValueTypes = attachLlmConfigWiring(
    buildResults,
    flowWiring,
    desugared.flowLlmConfig ?? {},
    producers,
    params.contracts,
    { typeFn: constructorType },
  );

  return { descriptors, buildResults, nodesBuilt, flowWiring, caseEdges, producers, flowLlmValueTypes };
}

/**
 * Infer the data edges, append each case's own edges, copy each host's route and ordering edges
 * and data gates onto the nodes lifted out of it, and compute CASE narrowing.
 */
function routeEdges(built: BuiltNodes, params: FlowParameters, ctx: AssembleContext): RoutedEdges {
  const { descriptors, flowWiring, caseEdges, producers } = built;
  // Data edges — the projection of the flow wiring (leaf + WAIT + call/map); a case owns its own
  // data+control edges (`caseEdges`, from each case descriptor's build), appended below.
  const dataEdges = inferDataEdges(descriptors, flowWiring);
  // A node lifted out of an authored node (an inline `call(...)`, an adaptive composer) runs only
  // on its host's routes, after its host's ordering, and once its host's other data inputs
  // settle, skipping when they would skip the host. So it gets copies of the routes and ordering
  // plus data gates here, before narrowing and the skip model read the graph.
  const hostEdges = hostControlEdges(descriptors, [...dataEdges, ...caseEdges]);
  // What each node may assume about union values from the CASE branches that route it.
  const narrowing = computeNarrowing(
    descriptors,
    [...dataEdges, ...caseEdges, ...hostEdges],
    producers,
    params.contracts,
    { registry: ctx.registry },
  );
  return { edges: [...dataEdges, ...caseEdges, ...hostEdges], hostEdges, narrowing };
}

/**
 * Check types over the built flow: case `when:` values and impossible `when:`s, typed inputs (and
 * stamp TOOL input types), CALL/MAP types, the LOOP contract, case exhaustiveness, wiring parity.
 */
function checkTypes(
  built: BuiltNodes,
  params: FlowParameters,
  narrowing: Narrowing,
  valueCaseIds: ReadonlySet<string>,
  ctx: AssembleContext,
): void {
  const { registry, resolveOrigin, constructorType, diagnostics } = ctx;
  const { descriptors, buildResults, flowWiring, producers } = built;
  const flowInputContracts = params.contracts;
  // Runs before every pass that reads narrowing, so a typo'd tag is reported first.
  checkCaseWhenValues(descriptors, producers, flowInputContracts, { narrowing, diagnostics });
  // A searched `when:` check that can never be true, before the reads it makes dead are left
  // unchecked by the reference pass.
  checkImpossibleWhens(descriptors, producers, flowInputContracts, narrowing, { registry, diagnostics });

  // Typed node inputs that can never fit their declared type. The CALL, MAP, and LOOP checks
  // below take a typed binding's declared type as its source, so a value that cannot fit that
  // type must be reported here first, against the binding itself.
  checkTypedInputs(flowWiring, producers, flowInputContracts, buildResults, {
    narrowing,
    typeFn: constructorType,
    registry,
    resolveOrigin,
    diagnostics,
  });
  // The run-start tool check compares these with the host tool's declared input types.
  stampToolInputTypes(flowWiring, producers, flowInputContracts, buildResults, {
    narrowing,
    typeFn: constructorType,
  });
  // cross-flow type check: each call/map binding's and loop seed's source type vs the child
  // input type (needs producers, so it runs after all nodes are built).
  checkCallMapTypes(buildResults, producers, flowInputContracts, flowWiring, {
    narrowing,
    typeFn: constructorType,
    registry,
    diagnostics,
  });
  // loop `'a -> 'a` type contract (types): each field the body returns is compared with its
  // carried seed-source Type (needs producers, so it runs alongside the cross-flow call/map type
  // check above). Each returned field must also fit the body parameter it is bound to on the
  // next turn: a named body's declared input, or the parameter an inline body generates from its
  // seed (`inlineLoops` only selects the wording).
  const inlineLoops = new Set(
    [...descriptors].filter(([, d]) => d instanceof LoopDescriptor && d.childLabel !== null).map(([id]) => id),
  );
  checkLoopTypeContract(buildResults, producers, flowInputContracts, flowWiring, {
    inlineLoops,
    registry,
    diagnostics,
  });
  // Case `on:` enum exhaustiveness needs the assembled producer types, so it runs after the
  // cross-flow type contracts while preserving validation error order.
  checkCaseExhaustive(descriptors, producers, { narrowing, valueCases: valueCaseIds, diagnostics });

  // Wiring/params key parity over EVERY node (catches an orphan/missing source or a node absent
  // from the wiring). Runs once the wiring is complete.
  checkWiringParity(buildResults, flowWiring);
}

/**
 * Validate references and asserts, with the node-local contracts, ordering edges and skip model
 * they read: validate references, check node asserts, lower flow asserts.
 */
function checkReferences(
  read: ReadSections,
  desugared: DesugaredSections,
  built: BuiltNodes,
  routed: RoutedEdges,
  ctx: AssembleContext,
): CheckedReferences {
  const { registry, owner, catalogBuilder, resolveOrigin, constructorType, diagnostics } = ctx;
  const { descriptors, buildResults, nodesBuilt, flowWiring, producers } = built;
  const { narrowing } = routed;
  const { params, valueEntries, authoredNodesCursor } = read;

  // The proven contracts of each node-local name, for the field reads a node makes of its own
  // inputs (prompts and node asserts).
  const localContracts = localInputContracts(flowWiring, producers, params.contracts, buildResults, {
    narrowing,
    typeFn: constructorType,
  });
  // The same, unnarrowed: a local the routing narrows from an untyped value is still untyped at run
  // time.
  const baseLocals = localInputContracts(flowWiring, producers, params.contracts, buildResults, {
    typeFn: constructorType,
  });

  // The ordering edges and the skip model live in this phase, not in edge routing, because
  // keeping the old error order puts an ordering error after the type checks and before references.
  // Run-ordering edges: depends_on (co-skip) / runs_after (pure order). Folded in BEFORE root
  // synthesis (`closeFlow`) so an ordering-only dependent is demoted from root, and into the cycle
  // check (`checkGraph`) so an ordering cycle is loud.
  const orderingEdges = inferOrderingEdges(descriptors, new Set(descriptors.keys()), { diagnostics });

  // Which nodes may be skipped while a reader runs: the field-read checks need it, so it is built
  // before them, over the structural edges (data, case, host and ordering).
  const structural = [...routed.edges, ...orderingEdges];
  const skips = new SkipModel(
    new Set(descriptors.keys()),
    structural,
    exhaustiveCases(descriptors, producers, params.contracts, { narrowing }),
  );

  // Each searched CASE's authored `when:`s: their reads are checked as written, since the wiring
  // binds only the roots they read.
  const searchedWhens = new Map<string, SearchedWhen[]>();
  for (const [nodeId, desc] of descriptors) {
    if (!(desc instanceof CaseDescriptor) || desc.on !== null) continue;
    searchedWhens.set(
      nodeId,
      desc.cases.map((case_, index) => [
        index,
        case_["when"] ?? null,
        desc.whenOrigins[index] ?? null,
        desc.whenRanges[index] ?? null,
      ]),
    );
  }

  // Reference + assert validation over the built flow. Errors locate at the node's line (a bad
  // node-input ref) or the `output:` section line (a dangling flow-output ref).
  validateReferences(buildResults, params.names, producers, desugared.outputsSection, flowWiring, {
    inputContracts: params.contracts,
    outputsRange: read.outputs.range,
    narrowing,
    localContracts,
    registry,
    resolveOrigin,
    diagnostics,
    searchedWhens,
    caseSpellings: desugared.caseSpellings,
  });

  /** Return the origin and range from one authored assertion cursor. */
  const nodeAssertionSiteAt = (
    nodeId: string,
    index: number,
  ): readonly [OriginRef | null, SourceRange | null] | null => {
    const cursor = locateSourceAccess(diagnostics, () => {
      if (valueEntries.has(nodeId)) return null;
      return authoredField(authoredNodesCursor, nodeId, "asserts", index);
    });
    if (cursor === null) return null;
    return [
      registerAssertionOrigin(cursor, [...owner, "node", nodeId, "assert", index], catalogBuilder, diagnostics),
      cursor.range,
    ];
  };

  // Node-local `asserts:` (per-node contract): validate node-locally + classify pre/post + stamp
  // onto each built leaf/call node. Shared by defs (a def's internal nodes too).
  validateNodeAsserts(nodesBuilt, descriptors, {
    assertionSiteAt: nodeAssertionSiteAt,
    typeNames: new Set(registry.keys()),
    isCallable: (name) => ctx.scope.has(name),
    localContracts,
    partialUnionNodes: narrowing.unordered,
    skips,
    baseLocals,
    registry,
    diagnostics,
  });
  // Lower the flow-level `asserts:` into node-local asserts stamped on the synthesized START/END.
  // Fires uniformly via eval_node — a cross-node post assert's `${producer.output}` becomes a
  // real END input edge, so it survives child cloning. `flowAssertionRanges` comes from the same
  // item cursors as each origin and stays aligned 1:1 with the authored assertions, so lowered
  // START/END asserts and bad-ref errors locate by index — never by non-unique text. Source-free
  // assertions carry `null`.
  const lowered = lowerFlowAsserts(
    desugared.assertsSection,
    params.names,
    new Set([
      ...descriptors.keys(),
      ...[...nodesBuilt].filter(([, node]) => node.producesMessages).map(([nodeId]) => messagesKey(nodeId)),
    ]),
    producers,
    read.flowAssertionRanges,
    { inputContracts: params.contracts, registry, skips, diagnostics },
  );

  return { localContracts, baseLocals, structural, skips, searchedWhens, lowered };
}

/**
 * Attach flow-output origins, inline call origins and declared output contracts; check field
 * reads; null the outputs CASE routing may skip; check skipped arithmetic and typed outputs.
 *
 * @returns The flow outputs, carrying their origins and contracts.
 */
function finishOutputs(
  read: ReadSections,
  desugared: DesugaredSections,
  built: BuiltNodes,
  narrowing: Narrowing,
  checked: CheckedReferences,
  ctx: AssembleContext,
): FlowOutput[] {
  const { registry, owner, catalogBuilder, resolveOrigin, constructorType, diagnostics } = ctx;
  const { descriptors, buildResults, nodesBuilt, flowWiring, producers } = built;
  const { outputsSection, assertsSection } = desugared;
  const { skips } = checked;
  const outputsRange = read.outputs.range;

  let outputs = stampFlowOutputOrigins(flowOutputs(outputsSection), outputsSection, read.outputs.cursor, {
    owner,
    catalogBuilder,
    diagnostics,
  });
  stampInlineCallOrigins(descriptors, read.nodesCursor, nodesBuilt, flowWiring, outputs, assertsSection, {
    owner,
    catalogBuilder,
    diagnostics,
  });
  outputs = outputs.map((output) => {
    const contract = read.outputs.types.get(output.name);
    if (contract === undefined) return output;
    return new FlowOutput(output.name, new Binding(output.binding.source, { origin: output.binding.origin, contract }));
  });
  const valueMode = outputsSection !== null && !isRecord(outputsSection);
  // A `.` step through a value that may be null (an `Optional`, or a producer that may be skipped
  // while the reader runs) is a load error; it runs before the outputs get their null defaults.
  checkFieldReads(
    {
      flowWiring,
      buildResults,
      narrowing,
      skips,
      registry,
      typeFn: constructorType,
      outputs,
      valueMode,
      outputsRange,
      localContracts: checked.localContracts,
      baseLocals: checked.baseLocals,
      searchedWhens: checked.searchedWhens,
      caseSpellings: desugared.caseSpellings,
      // An inline body's lexical owner ends `node, <id>, inline`.
      outputsOwner: owner.at(-1) === "inline" && owner.at(-3) === "node" ? `inline body of node ${reprText(owner.at(-2)!)}` : null,
    },
    { resolveOrigin, diagnostics },
  );
  // An output that reads only nodes CASE routing may skip would leave END unreachable; the
  // explicit null default makes its END edge optional, so the run returns null.
  outputs = nullSkippedOutputs(outputs, skips);
  // Arithmetic on such a value would fail at runtime with null; ask for a `:-` default.
  checkSkippedArithmetic(flowWiring, outputs, skips, buildResults, {
    registry,
    valueMode,
    outputsRange,
    resolveOrigin,
    diagnostics,
    searchedWhens: checked.searchedWhens,
  });
  // Typed flow outputs that can never fit their declared type (needs the output contracts
  // attached above).
  checkTypedOutputs(outputs, producers, read.params.contracts, {
    valueMode,
    outputsRange,
    typeFn: constructorType,
    registry,
    resolveOrigin,
    diagnostics,
  });
  return outputs;
}

/**
 * Synthesize the START_ID/END_ID boundary over all edges, check `llm_config` sources, fill each
 * node's `isinstance` type table and check reserved sources (`over:`, `until:`).
 */
function closeFlow(
  read: ReadSections,
  desugared: DesugaredSections,
  built: BuiltNodes,
  narrowing: Narrowing,
  checked: CheckedReferences,
  outputs: readonly FlowOutput[],
  ctx: AssembleContext,
): ClosedFlow {
  const { registry, resolveOrigin, constructorType, diagnostics } = ctx;
  const { descriptors, buildResults, nodesBuilt, flowWiring, producers } = built;
  const flowInputContracts = read.params.contracts;

  // The START_ID/END_ID boundary nodes, their edges and wiring, over the full structural edge set.
  const [boundaryNodes, boundaryEdges, boundaryWiring] = synthesizeBoundaryGraph(
    read.params.inputs,
    outputs,
    new Set(descriptors.keys()),
    checked.structural,
    checked.lowered,
    { record: isRecord(desugared.outputsSection) },
  );
  const nodes = new Map([...nodesBuilt, ...boundaryNodes]);
  const wiring = new Map([...flowWiring, ...boundaryWiring]);
  const edges: Edge[] = [...checked.structural, ...boundaryEdges];

  // Before the type tables, so a constructor in a config value gets the config's message.
  checkLlmConfigSources(
    buildResults,
    desugared.flowLlmConfig ?? {},
    desugared.flowLlmConfigLabel,
    producers,
    flowInputContracts,
    { typeFn: constructorType, registry, resolveOrigin, diagnostics },
  );
  // Type tables for `isinstance`, once START/END carry the lowered flow asserts and outputs.
  fillTypeTables(
    nodes,
    wiring,
    buildResults,
    descriptors,
    desugared.assertsSection,
    read.flowAssertionRanges,
    read.outputs.rangeOf,
    registry,
    diagnostics,
  );
  // After the type tables, so `isinstance` in `over:`/`until:` keeps its own error.
  checkReservedSources(buildResults, wiring, producers, flowInputContracts, {
    narrowing,
    typeFn: constructorType,
    registry,
    resolveOrigin,
    diagnostics,
  });
  return { nodes, edges, wiring };
}

/** Reject cycles and check each case's if/else handles over the closed graph. */
function checkGraph(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  closed: ClosedFlow,
  hostEdges: readonly Edge[],
  ctx: AssembleContext,
): void {
  const { diagnostics } = ctx;
  const { nodes, edges } = closed;
  // A cycle spans nodes; every actual member contributes its authored range. START_ID/END_ID are
  // acyclic by construction (START_ID has no incoming, END_ID has no outgoing).
  const nodeRanges = new Map<string, SourceRange>();
  for (const [nodeId, descriptor] of descriptors) {
    if (descriptor.range !== null) nodeRanges.set(nodeId, descriptor.range);
  }
  // A synth node whose text all lives under its host (it has `fieldLabels`) is reported as that
  // host only when its generated wire into the host alone closes the cycle (an adaptive
  // composer); otherwise, as for an inline call, a note names its host.
  const synthHosts = new Map<string, string>();
  for (const [nodeId, descriptor] of descriptors) {
    if (descriptor.host !== null && descriptor.fieldLabels.size > 0) synthHosts.set(nodeId, descriptor.host);
  }
  const synthWires = new Set(
    edges
      .filter((edge) => synthHosts.get(edge.from) === edge.to && edge.inputGroup === QUESTIONS_INPUT)
      .map((edge) => edge.id),
  );
  rejectCycles(edges, new Set(descriptors.keys()), {
    nodeRanges,
    diagnostics,
    gateIds: new Set(hostEdges.filter((edge) => edge.inputGroup).map((edge) => edge.id)),
    synthHosts,
    synthWires,
  });
  checkCaseHandles(nodes, edges, { diagnostics });
}

/**
 * Assemble parsed sections into a `ComposedFlow`: the post-parse pipeline shared by the top-level
 * flow and every in-file def. The module docstring lists its phases in order.
 *
 * Exactly one input representation is required: `inputsSection` is the authored mapping that
 * still needs parsing, while `inputParameters` carries parameters already resolved by an earlier
 * compiler pass. `inputsCursor` is the optional retained occurrence for the authored mapping;
 * when present, input parameters and their recursive contracts compile opaque origins from it.
 * A Compose-owned callable in `BuildContext` compiles a physical node `output:` cursor into its
 * canonical contract only when the leaf reaches its established output-contract step;
 * source-free builds use the descriptor's ordinary type reader. Retained ranges and
 * `nodesCursor` own diagnostic locations; after the common header stamp each descriptor may
 * immutably attach kind-owned opaque origins. `scope.catalogBuilder` is the one registry shared
 * by every assembly originating from this physical document, so root, definitions, and inline
 * bodies mint collision-free local origins under `owner`. After kind-owned attachment, CODE may
 * present a syntax failure through the builder's registered-site lookup without freezing the
 * catalog. `assemble` performs one construction attempt; retriable callable attempts run through
 * the scope's `resolve`, while a failed top-level load discards its scope.
 *
 * @throws RangeError when not exactly one input representation is given, or resolved parameters
 *   come with an inputs cursor.
 * @throws TypeError when a flow input or producer contract has the wrong class.
 * @throws LoadError for an authored mistake any pass finds.
 */
export function assemble(options: AssembleOptions): ComposedFlow {
  const { inputsSection = null, inputParameters = null, inputsCursor = null } = options;
  if ((inputsSection === null) === (inputParameters === null)) {
    throw new RangeError("assemble requires exactly one of inputsSection or inputParameters");
  }
  if (inputParameters !== null && inputsCursor !== null) {
    throw new RangeError("pre-resolved input parameters cannot carry an inputs cursor");
  }

  const { scope, registry } = options;
  const catalogBuilder = scope.catalogBuilder;
  const ctx: AssembleContext = {
    diagnostics: LoadDiagnostics.fromCursor(options.nodesCursor),
    scope,
    catalogBuilder,
    resolveOrigin: (origin) => catalogBuilder.resolveRegistered(origin),
    registry,
    constructorType: constructorTypeFn(registry),
    owner: options.owner,
    baseDir: options.baseDir ?? null,
  };

  const read = readSections(options, ctx);
  const desugared = desugarSections(read, options.assertsSection, options.flowLlmConfig ?? null, ctx);
  const built = buildNodes(read, desugared, options.flowEnv ?? null, ctx);
  const routed = routeEdges(built, read.params, ctx);
  checkTypes(built, read.params, routed.narrowing, desugared.valueCaseIds, ctx);
  const checked = checkReferences(read, desugared, built, routed, ctx);
  const outputs = finishOutputs(read, desugared, built, routed.narrowing, checked, ctx);
  const closed = closeFlow(read, desugared, built, routed.narrowing, checked, outputs, ctx);
  checkGraph(built.descriptors, closed, routed.hostEdges, ctx);

  const compiled = new CompiledFlow(closed.nodes, closed.edges, {
    outputs,
    wiring: closed.wiring,
    flowLlmConfig: desugared.flowLlmConfig ?? {},
    flowId: options.flowId ?? null,
  });
  compiled.flowLlmValueTypes = built.flowLlmValueTypes;
  return new ComposedFlow(compiled, {
    version: options.version ?? null,
    name: options.name ?? null,
    description: options.description ?? null,
  });
}
