/**
 * The post-parse conductor: assembled sections to a runnable `ComposedFlow`.
 *
 * `assemble` is the pipeline shared by the top-level flow and every in-file `defs:` callable:
 *
 *   flow parameters + authored parameter/contract/default origins -> parse nodes, lowering each
 *   tagged `nodes:` entry to a CODE node -> build every node kind-blind (`desc.build(ctx)`;
 *   physical node-output contracts compile at their existing leaf-owned build position; a case
 *   lowers in its own build) -> cross-flow type check + case exhaustiveness -> infer data edges +
 *   append each case's own edges + copy each host's route/ordering edges and data gates onto the
 *   nodes lifted out of it -> validate references + classify asserts -> attach authored
 *   assertion/flow-output binding origins -> synthesize the START_ID/END_ID boundary over all
 *   edges -> resolve each node's `isinstance` type table -> reject cycles + check if/else
 *   handles -> `CompiledFlow.fromParts`. One retained assertion item cursor supplies its carrier
 *   origin and Compose-only validation range; only the origin crosses into the semantic
 *   assertion.
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
import { DeclSite, MemoryDecl } from "../../memory/contracts.ts";
import type { Node } from "../../nodes/base.ts";
import {
  asBinding,
  type Assertion,
  Binding,
  isExactly,
  type OutputContract,
  Parameter,
  PartialRecordContract,
  ValueContract,
  type ValuePath,
  type ValueTemplate,
} from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isList, isListKind, isRecord, jsTypeName, type Type, type Value, writeAsText } from "../../typesys/values.ts";
import { LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import { constructorTypeFn } from "../helpers/isinstance_types.ts";
import { resolveOutputBindingContract, type TypeFn } from "../helpers/output_contracts.ts";
import {
  oldFormSuggestion,
  oldTypedText,
  QUOTED_STYLES,
  readTaggedValue,
  rejectTaggedOutput,
  suggestTag,
  yamlQuoted,
} from "../helpers/tags.ts";
import { readFlowInputs, readSourceFlowInputs, readSourceParameters, readValueContract } from "../helpers/types.ts";
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
// T3-15 imports
import { AgentDescriptor, readLlmConfig } from "../nodes/agent.ts";
import { siteKey } from "../nodes/base.ts";
import { HumanInputDescriptor } from "../nodes/human_input.ts";
// T3-17 imports
import { desugarAdaptiveQuestions, QPARAM } from "../desugar/adaptive_questions.ts";
import { caseOutputReads, expandCaseOutputs } from "../desugar/case_outputs.ts";
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
// T3-18 imports
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
import { builtinFieldErrors, constructorErrors } from "../validate/constructors.ts";
import { rejectCycles } from "../validate/cycles.ts";
import { checkCaseHandles } from "../validate/handles.ts";
import { validateHumanQuestions } from "../validate/human_questions.ts";
import { validateReferences } from "../validate/references.ts";
import { checkReservedSources } from "../validate/reserved_sources.ts";
import { stampToolInputTypes } from "../validate/tool_inputs.ts";
import { checkWrittenValues } from "../validate/value_nesting.ts";
import { checkWiringParity } from "../validate/wiring_parity.ts";
// T3-19 imports
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
  localInputContracts,
} from "../validate/typed_bindings.ts";
// T3-20 imports
import { lowerFlowAsserts, validateNodeAsserts } from "../validate/asserts.ts";
import { checkCaseBranches } from "../validate/case_branches.ts";
import { checkCaseExhaustive, checkCaseWhenValues, exhaustiveCases } from "../validate/case_exhaustive.ts";
import { checkImpossibleWhens, computeNarrowing } from "../validate/narrowing.ts";
import { checkSkippedArithmetic, nullSkippedOutputs, type SearchedWhen, SkipModel } from "../validate/skips.ts";

// Name for the single output of a bare whole-string `output:` binding. The value-mode EndNode
// returns the bare resolved value, so the name only labels the carrier.
const SINGLE_OUTPUT_NAME = "result";

const PRESENTATION_UNAVAILABLE_NOTE = "Source presentation was unavailable.";

/** One active descriptor build and the dependency edge that requested it; frozen. */
export class BuildRequest {
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

  /** Wrap `compiled`; the metadata and resolver default to `null`, `importDepth` to 0. */
  constructor(
    compiled: CompiledFlow,
    options: {
      version?: string | null;
      name?: string | null;
      description?: string | null;
      originResolver?: OriginResolver | null;
      importDepth?: number;
    } = {},
  ) {
    const {
      version = null,
      name = null,
      description = null,
      originResolver = null,
      importDepth = 0,
    } = options;
    this.compiled = compiled;
    this.version = version;
    this.name = name;
    this.description = description;
    this.originResolver = originResolver;
    this.importDepth = importDepth;
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
   * Resolution is presentation-only. A missing, incomplete, malformed, or failing resolver
   * produces an unlocated diagnostic without changing the semantic failure fields.
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
        const frame: unknown = resolver.resolve(origin);
        if (!(frame instanceof DiagnosticFrame)) return unavailable;

        const { label, text, range, note } = frame;
        if (
          typeof label !== "string" ||
          typeof text !== "string" ||
          (note !== null && typeof note !== "string") ||
          !(range instanceof SourceRange)
        ) {
          return unavailable;
        }

        const { startLine, startCol, endLine, endCol } = range;
        const coordinates = [startLine, startCol, endLine, endCol];
        if (coordinates.some((coordinate) => !Number.isInteger(coordinate) || coordinate <= 0)) {
          return unavailable;
        }
        if (endLine < startLine || (endLine === startLine && endCol < startCol)) return unavailable;

        frames.push(
          new DiagnosticFrame({
            label,
            text,
            range: new SourceRange(startLine, startCol, endLine, endCol),
            note,
          }),
        );
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
  /** Resolve one named or compiler-owned callable in this file scope. */
  resolve(flowId: string, version?: string | null): ComposedFlow;
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
 * @param options.diagnostics Factory for located load errors (default: from the first body's
 *   `nodes:` cursor).
 * @param options.buildChild Builds one body's child flow under its labelled descriptor.
 * @returns A new descriptor map with each inline composite labelled.
 * @throws LoadError when a body has no descriptor, its descriptor is not a CALL, MAP or LOOP, or
 *   the descriptor does not call the body's private id.
 */
export function registerInlineBodies(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  bodies: ReadonlyMap<string, InlineBody>,
  options: {
    registrar: PrivateRegistrar;
    diagnostics?: LoadDiagnostics | null;
    buildChild: (body: InlineBody, descriptor: CallDescriptor | LoopDescriptor) => ComposedFlow;
  },
): Map<string, NodeDescriptor> {
  const { registrar, buildChild } = options;
  const diagnostics =
    options.diagnostics ?? LoadDiagnostics.fromCursor(bodies.values().next().value?.nodesCursor ?? null);
  const associated = new Map(descriptors);
  const registrations: [InlineBody, CallDescriptor | LoopDescriptor, string][] = [];

  for (const body of bodies.values()) {
    const label = `inline body of node ${reprText(body.nodeId)}`;
    const descriptor = descriptors.get(body.nodeId);
    if (descriptor === undefined) {
      throw diagnostics.error(`${label}: no matching composite node descriptor`, { range: body.nodesCursor.range });
    }
    if (!(descriptor instanceof CallDescriptor || descriptor instanceof LoopDescriptor)) {
      throw diagnostics.error(`${label}: final node descriptor is not CALL, MAP, or LOOP`, {
        range: body.nodesCursor.range,
      });
    }
    if (descriptor.call !== body.flowId) {
      throw diagnostics.error(`${label}: final node descriptor does not target its retained inline body`, {
        range: body.nodesCursor.range,
      });
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
 * @throws LoadError when the retained sequence does not align with the semantic list.
 */
export function assertionCursors(
  assertions: readonly unknown[],
  assertsCursor: SourceCursor | null,
  diagnostics: LoadDiagnostics,
): readonly (SourceCursor | null)[] {
  if (assertsCursor === null) return assertions.map(() => null);
  let cursors: readonly SourceCursor[];
  try {
    cursors = assertsCursor.children();
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
  if (cursors.length !== assertions.length) {
    throw diagnostics.error("retained assertion cursor does not align with semantic assertions", {
      range: assertsCursor.range,
    });
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
  try {
    return catalogBuilder.register(sitePath, cursor);
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
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

    try {
      if (!nodesCursor.has(nodeId)) continue;
      let inputCursors: Map<string, SourceCursor>;
      if (isValueEntry(descriptor)) {
        inputCursors = new Map([[nodeId, nodesCursor.child(nodeId)]]);
      } else {
        const nodeCursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));

        for (const key of [...node.reservedWiringKeys()].sort(compareCodeUnits)) {
          if (!wiring.has(key) || !nodeCursor.has(key)) continue;
          const bindingOrigin = catalogBuilder.register(
            [...owner, "node", nodeId, "reserved", key, "binding"],
            nodeCursor.child(key),
          );
          if (bindingOrigin !== null) wiring.set(key, withOrigin(wiring.get(key), bindingOrigin));
        }

        if (Object.keys(descriptor.inputs).length === 0 || !nodeCursor.has("inputs")) continue;
        const inputsCursor = nodeCursor.child("inputs");
        inputCursors = new Map();
        for (const name of Object.keys(descriptor.inputs)) {
          if (inputsCursor.has(name)) inputCursors.set(name, inputsCursor.child(name));
        }
      }
      if (node.params === null) continue;
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
          parameters[parameterIndex] = new Parameter(parameter.name, {
            contract: parameter.contract,
            required: parameter.required,
            hasDefault: parameter.hasDefault,
            default: parameter.default,
            coerceStrings: parameter.coerceStrings,
            origin: parameterOrigin,
            defaultOrigin: parameter.defaultOrigin,
          });
          parametersChanged = true;
        }
        if (bindingOrigin !== null) wiring.set(name, withOrigin(wiring.get(name), bindingOrigin));
      }
      if (parametersChanged) node.params = parameters;
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      throw diagnostics.error(error.message, { range: error.range, cause: error });
    }
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
    let changed = false;
    try {
      let inputsCursor: SourceCursor | null = null;
      if (nodesCursor.has(nodeId)) {
        const nodeCursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
        if (nodeCursor.has("inputs")) inputsCursor = nodeCursor.child("inputs");
      }
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
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      throw diagnostics.error(error.message, { range: error.range, cause: error });
    }
    split.set(nodeId, changed ? descriptor.replace({ inputs, inputTypes }) : descriptor);
  }
  return split;
}

/**
 * Read each AGENT's and adaptive gate's `llm_config:` with the values and tags in it.
 *
 * An AGENT's own layer and an adaptive gate's `adaptive_questions.llm_config` are read. A layer
 * holding values becomes a `ValueTemplate` (a `Binding` for a whole record written as one value),
 * and the value range of each key in it lands in the descriptor's `fieldRanges` under
 * `siteKey("llm_config", "<dotted.path>")`, so its errors point at the field; the adaptive desugar
 * copies both to the composer agent. A layer without values stays a plain record.
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
  try {
    for (const [nodeId, descriptor] of descriptors) {
      if (!nodesCursor.has(nodeId)) continue;
      let path: readonly string[];
      let written: unknown;
      let prefix: string;
      if (descriptor instanceof AgentDescriptor) {
        path = ["llm_config"];
        written = descriptor.llmConfig;
        prefix = `node ${reprText(nodeId)}: llm_config`;
      } else if (descriptor instanceof HumanInputDescriptor && isRecord(descriptor.adaptiveQuestions)) {
        path = ["adaptive_questions", "llm_config"];
        written = descriptor.adaptiveQuestions["llm_config"];
        prefix = `node ${reprText(nodeId)} adaptive_questions.llm_config`;
      } else {
        continue;
      }
      let cursor: SourceCursor | null = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
      for (const key of path) {
        cursor = cursor.has(key) ? cursor.child(key) : null;
        if (cursor === null) break;
      }
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
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
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
 * composer's, the gate's `adaptive_questions.llm_config`).
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
 * Return the authored YAML scalar style of one node binding.
 *
 * A value entry's one binding is the entry itself, so its style is the entry's.
 *
 * @param nodesCursor Retained view of the flow's node mapping.
 * @param nodeId The node that owns the binding.
 * @param field The binding field (`inputs`, or a scalar field such as `over`).
 * @param key The binding name in a mapping field; equals `field` for a scalar one.
 * @param options.valueEntries The ids of the flow's value entries.
 * @param options.diagnostics Factory for located load errors.
 * @returns The style, or `null` for a synthesized node or binding.
 * @throws LoadError when the retained cursor cannot be traversed.
 */
function bindingStyle(
  nodesCursor: SourceCursor,
  nodeId: string,
  field: string,
  key: string,
  options: { valueEntries: ReadonlySet<string>; diagnostics: LoadDiagnostics },
): ScalarStyle | null {
  const { valueEntries, diagnostics } = options;
  try {
    if (!nodesCursor.has(nodeId)) return null;
    if (valueEntries.has(nodeId)) {
      if (field !== "inputs" || key !== nodeId) return null;
      return nodesCursor.child(nodeId).scalarStyle;
    }
    let cursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
    if (!cursor.has(field)) return null;
    cursor = cursor.child(field);
    if (key !== field) {
      if (!cursor.has(key)) return null;
      cursor = cursor.child(key);
    }
    return cursor.scalarStyle;
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
}

/**
 * Return the authored YAML scalar style of one flow output binding.
 *
 * @param outputsCursor Retained authored `output:` value.
 * @param key The output's record field name, or `null` for a single output.
 * @param options.diagnostics Factory for located load errors.
 * @returns The style, or `null` for a generated section.
 * @throws LoadError when the retained cursor cannot be traversed.
 */
function outputStyle(
  outputsCursor: SourceCursor | null,
  key: string | null,
  options: { diagnostics: LoadDiagnostics },
): ScalarStyle | null {
  const { diagnostics } = options;
  if (outputsCursor === null) return null;
  try {
    if (key === null) return outputsCursor.scalarStyle;
    if (!outputsCursor.has(key)) return null;
    return outputsCursor.child(key).scalarStyle;
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
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
  try {
    if (!nodesCursor.has(nodeId)) return null;
    const cursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
    return cursor.has("outputs") ? cursor.child("outputs") : null;
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
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
 * @throws Error when a node has a declared input type but no wiring binding by that name.
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
        throw new Error(`node ${reprText(nodeId)}: typed input ${reprText(name)} has no wiring binding`);
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
 * Unlike Python's, an untyped output needs no check that its literals are JSON values: the YAML
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

  try {
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
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
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
    try {
      cursors = rawKeys.map((key) => outputsCursor.child(key));
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      throw diagnostics.error(error.message, { range: error.range, cause: error });
    }
  } else {
    if (outputs.length !== 1) throw new RangeError("a bare flow output requires exactly one carrier");
    cursors = [outputsCursor];
  }

  return outputs.map((output, index) => {
    if (output.binding.origin !== null) return output;
    let origin: OriginRef | null;
    try {
      origin = catalogBuilder.register([...owner, "output", index, "binding"], cursors[index]!);
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      throw diagnostics.error(error.message, { range: error.range, cause: error });
    }
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
    if (!nodesCursor.has(caseId)) return null;
    let cursor = normalizeNodeSignatureCursor(nodesCursor.child(caseId));
    let path: SourceSitePath;
    if (address[0] === "else") {
      if (!cursor.has("else")) return null;
      path = [...owner, "node", caseId, "field", "else"];
      cursor = cursor.child("else");
    } else {
      const index = address[2] as number;
      if (!cursor.has("cases")) return null;
      const item = cursor.child("cases").child(index);
      if (!item.has("then")) return null;
      path = [...owner, "node", caseId, "case", index, "then"];
      cursor = item.child("then");
    }
    return catalogBuilder.register(path, cursor);
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
      try {
        origins.set(addressKey, siteOrigin(address));
      } catch (error) {
        if (!(error instanceof SourceAccessError)) throw error;
        throw diagnostics.error(error.message, { range: error.range, cause: error });
      }
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
 * @param outputsCursor The authored `output:` value; a mapping locates each named output.
 * @param outputsRange The whole `output:` range, used when an output has no range of its own.
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
  outputsCursor: SourceCursor | null,
  outputsRange: SourceRange | null,
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

  /** One flow output's authored range, else the whole `output:` range. */
  const outputRange = (name: string): SourceRange | null => {
    try {
      if (outputsCursor !== null && outputsCursor.has(name)) return outputsCursor.child(name).range;
    } catch (error) {
      // A bare `output:` value has no named children.
      if (!(error instanceof SourceAccessError)) throw error;
    }
    return outputsRange;
  };

  /**
   * Resolve `names` as `table` does, then reject a constructor call in `source` that can never
   * succeed whatever the references hold, or a field read on a builtin's result its type can't
   * have.
   *
   * Only for condition sites: `validateReferences` already checked every binding's constructor
   * calls and builtin field reads, with the references typed. Python's `condition` flag is dropped:
   * every caller parses its source as a condition.
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
      problems.push(...builtinFieldErrors(tree));
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

/** Python's `_assemble` keyword arguments: the parsed sections and the scope one assembly reads. */
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
  /** The flow-owned memory declarations. */
  memories?: readonly MemoryDecl[];
  /** Retained occurrence of the `memories:` declarations, or `null`. */
  memoriesCursor?: SourceCursor | null;
  /**
   * The directory the flow's CODE references resolve against, its inline bodies' too; `null` is
   * the current directory.
   */
  baseDir?: string | null;
}

/**
 * Assemble parsed sections into a `ComposedFlow`: the post-parse pipeline shared by the top-level
 * flow and every in-file def.
 *
 *   flow parameters -> parse nodes (a tagged entry lowers to a CODE node returning its value) ->
 *   split tagged node inputs and flow outputs into value + declared contract -> check case
 *   branches (a node or a call; a value or a case `output:` is a load error showing the value
 *   entry) -> desugar (inline calls, adaptive questions, case call targets, `${case.output}` to a
 *   coalesce over the case's branch targets) -> build every node kind-blind (`desc.build(ctx)`;
 *   physical node-output contracts compile at their existing leaf-owned position; a case lowers
 *   in its own build) -> attach declared input contracts to the node wiring -> cross-flow type
 *   check + case exhaustiveness -> infer data edges + append each case's own edges + copy each
 *   host's route/ordering edges and data gates onto the nodes lifted out of it -> validate
 *   references + classify asserts -> attach authored flow-output binding origins + declared
 *   output contracts -> synthesize roots/terminals -> resolve `isinstance` type tables -> reject
 *   cycles + check if/else handles -> `CompiledFlow.fromParts`.
 *
 * Exactly one input representation is required: `inputsSection` is the authored mapping that
 * still needs parsing, while `inputParameters` carries parameters already resolved by an earlier
 * compiler pass. `inputsCursor` is the optional retained occurrence for the authored mapping;
 * when present, input parameters and their recursive contracts compile opaque origins from it.
 * A Compose-owned callable in `BuildContext` compiles a physical node `output:` cursor into its
 * canonical contract only when the leaf reaches its established output-contract step;
 * source-free builds use the descriptor's ordinary type reader. Retained ranges and
 * `nodesCursor` own diagnostic locations; after the common header stamp each descriptor may
 * immutably attach kind-owned opaque origins. `memories` and its optional cursor carry the
 * flow-owned declarations; each is restamped with this flow's `DeclSite` and gets one origin per
 * entry. `scope.catalogBuilder` is the one registry shared by every assembly originating from
 * this physical document, so root, definitions, and inline bodies mint collision-free local
 * origins under `owner`. After kind-owned attachment, CODE may present a syntax failure through
 * the builder's registered-site lookup without freezing the catalog. `assemble` performs one
 * construction attempt; retriable callable attempts run through the scope's `resolve`, while a
 * failed top-level load discards its scope.
 *
 * @throws RangeError when not exactly one input representation is given, or resolved parameters
 *   come with an inputs cursor.
 * @throws TypeError when `memories` isn't an array of exact `MemoryDecl` values, or a flow input
 *   or producer contract has the wrong class.
 * @throws LoadError for an authored mistake any pass finds.
 */
export function assemble(options: AssembleOptions): ComposedFlow {
  const { inputsSection = null, inputParameters = null, inputsCursor = null } = options;
  const { registry, scope, owner, usesAliases = [], flowLlmConfig = null, flowEnv = null } = options;
  const { boundaryFieldRanges = null, assertsCursor = null, outputsCursor = null } = options;
  const { version = null, name = null, description = null, flowId = null, memoriesCursor = null } = options;
  const { baseDir = null } = options;
  let { nodesCursor, outputsSection, outputsRange = null, memories = [] } = options;

  if ((inputsSection === null) === (inputParameters === null)) {
    throw new RangeError("assemble requires exactly one of inputsSection or inputParameters");
  }
  if (inputParameters !== null && inputsCursor !== null) {
    throw new RangeError("pre-resolved input parameters cannot carry an inputs cursor");
  }

  const diagnostics = LoadDiagnostics.fromCursor(nodesCursor);

  if (outputsRange === null && outputsCursor !== null) outputsRange = outputsCursor.range;

  // Inline-body extraction masks child-only fields for descriptor parsing. Assertion provenance
  // still belongs to the unmodified authored occurrence tree.
  const authoredNodesCursor = nodesCursor;
  let inlineBodies: Map<string, InlineBody>;
  try {
    [nodesCursor, inlineBodies] = extractInlineBodies(nodesCursor, { owner });
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }

  const catalogBuilder = scope.catalogBuilder;
  // A bound `catalogBuilder.resolveRegistered`, which every pass below presents origins through.
  const resolveOrigin: OriginPresenter = (origin) => catalogBuilder.resolveRegistered(origin);
  if (!Array.isArray(memories) || memories.some((decl) => !isExactly(decl, MemoryDecl))) {
    throw new TypeError("memories must be an array of exact MemoryDecl values");
  }
  if (memories.length > 0) {
    const stamped: MemoryDecl[] = [];
    for (const decl of memories) {
      let origin = decl.origin;
      if (memoriesCursor !== null) {
        const path = [...owner, "memories", decl.name];
        if (memoriesCursor.has(decl.name)) {
          origin = catalogBuilder.register(path, memoriesCursor.child(decl.name), "key");
        } else {
          // sugar: the whole scalar value is the one entry
          origin = catalogBuilder.register(path, memoriesCursor);
        }
      }
      stamped.push(new MemoryDecl(decl.name, decl.type, decl.config, new DeclSite([], owner, null), origin));
    }
    memories = stamped;
  }
  const flowAssertionCursors = assertionCursors(options.assertsSection, assertsCursor, diagnostics);
  const flowAssertionRanges = flowAssertionCursors.map((cursor) => (cursor !== null ? cursor.range : null));

  /** Return the registered origin for one authored flow assertion. */
  const flowAssertionOriginAt = (index: number): OriginRef | null =>
    registerAssertionOrigin(flowAssertionCursors[index]!, [...owner, "assert", index], catalogBuilder, diagnostics);

  /** Return one authored flow assertion's retained positional range. */
  const flowAssertionRangeAt = (index: number): SourceRange | null => flowAssertionRanges[index]!;

  // These readers run only during the build below, after `valueEntries` exists.
  /** Return the origin and range from one authored assertion cursor. */
  const nodeAssertionSiteAt = (
    nodeId: string,
    index: number,
  ): readonly [OriginRef | null, SourceRange | null] | null => {
    let cursor: SourceCursor;
    try {
      if (valueEntries.has(nodeId) || !authoredNodesCursor.has(nodeId)) return null;
      const nodeCursor = normalizeNodeSignatureCursor(authoredNodesCursor.child(nodeId));
      if (!nodeCursor.has("asserts")) return null;
      cursor = nodeCursor.child("asserts").child(index);
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      throw diagnostics.error(error.message, { range: error.range, cause: error });
    }
    return [
      registerAssertionOrigin(cursor, [...owner, "node", nodeId, "assert", index], catalogBuilder, diagnostics),
      cursor.range,
    ];
  };

  /**
   * Compile one physical node-output declaration at its build position.
   *
   * @param nodeId Node whose normalized declaration should be selected.
   * @param outputSpec Semantic type shape already retained by its descriptor.
   * @returns Located contract, or `null` for a virtual site.
   * @throws LoadError when source traversal or type compilation fails.
   */
  const readAuthoredNodeOutput = (nodeId: string, outputSpec: unknown): ValueContract | null => {
    try {
      if (valueEntries.has(nodeId) || !nodesCursor.has(nodeId)) return null;
      const nodeCursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
      if (!nodeCursor.has("outputs")) return null;
      const outputCursor = nodeCursor.child("outputs");
      if (outputCursor.range === null) return null;
      // `withValue` refuses a non-YAML value itself.
      return readValueContract(outputCursor.withValue(outputSpec as Value), registry, {
        sitePath: [...owner, "node", nodeId, "output", "contract"],
        catalogBuilder,
        diagnostics,
      });
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      throw diagnostics.error(error.message, { range: error.range, cause: error });
    }
  };

  /** Compile one physical node's variables through its normalized cursor. */
  const readAuthoredNodeVariables = (nodeId: string, declarations: unknown): Parameter[] | null => {
    try {
      if (valueEntries.has(nodeId) || !nodesCursor.has(nodeId)) return null;
      const nodeCursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
      if (!nodeCursor.has("variables")) return null;
      const variablesCursor = nodeCursor.child("variables");
      if (variablesCursor.range === null) return null;
      return readSourceParameters(variablesCursor.withValue(declarations as Value), registry, {
        sitePath: [...owner, "node", nodeId],
        catalogBuilder,
        diagnostics,
        fieldName: "variables",
        declarationLabel: "variable",
        requireDefault: true,
      });
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      throw diagnostics.error(error.message, { range: error.range, cause: error });
    }
  };

  // Inline CODE receives the owning document label only as an opaque traceback file name;
  // authored locations come from semantic origins. Inert for reference-mode CODE.
  const flowLabel = diagnostics.label || "<inline-code>";
  let inputs: Parameter[];
  if (inputsSection !== null) {
    if (inputsCursor !== null) {
      inputs = readSourceFlowInputs(inputsCursor.withValue(inputsSection as Value), registry, {
        sitePath: owner,
        catalogBuilder,
        diagnostics,
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
  const flowInputContracts = new Map<string, ValueContract>();
  for (const parameter of inputs) {
    const contract = parameter.contract;
    if (contract === null) continue;
    if (!isExactly(contract, ValueContract)) {
      throw new TypeError(
        "flow input contracts must be exact ValueContract values, " +
          `got ${jsTypeName(contract)} for ${reprText(parameter.name)}`,
      );
    }
    flowInputContracts.set(parameter.name, contract);
  }
  const flowInputTypes = new Map([...flowInputContracts].map(([inputName, contract]) => [inputName, contract.type]));
  // Static inference types a `T(...)` constructor by this flow's own typedefs.
  const constructorType = constructorTypeFn(registry);
  const flowInputs = new Set(inputs.map((parameter) => parameter.name));

  /** Lower one tagged `nodes:` entry after checking its nesting. */
  const readValue = (nodeId: string, cursor: SourceCursor): ValueEntryDescriptor => {
    // First: the tag read walks the value recursively.
    checkWrittenValue(cursor.value, `value entry ${reprText(nodeId)}`, cursor.range ?? cursor.keyRange, diagnostics);
    return readValueEntry(nodeId, cursor, { registry, owner, catalogBuilder, diagnostics });
  };

  let descriptors = parseNodes(nodesCursor, { readValue });
  // Value entries have no node body, so every pass below that re-reads a node's body from
  // `nodesCursor` skips them or reads the entry itself.
  const valueEntries: ReadonlySet<string> = new Set(
    [...descriptors].filter(([, descriptor]) => isValueEntry(descriptor)).map(([nodeId]) => nodeId),
  );
  try {
    const locatedDescriptors = new Map<string, NodeDescriptor>();
    for (const [nodeId, descriptor] of descriptors) {
      const authoredCursor = nodesCursor.child(nodeId);
      const nodePath = [...owner, "node", nodeId];
      const located = descriptor.replace({ origin: catalogBuilder.register(nodePath, authoredCursor, "key") });
      if (valueEntries.has(nodeId)) {
        locatedDescriptors.set(nodeId, located);
        continue;
      }
      locatedDescriptors.set(
        nodeId,
        located.withSourceOrigins(normalizeNodeSignatureCursor(authoredCursor), { sitePath: nodePath, catalogBuilder }),
      );
    }
    descriptors = locatedDescriptors;
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }

  // A `uses:` alias shares the callable namespace, so it must not collide with a node id (the
  // resolver only sees a bare name; the author conflates them). The loader checks the authored
  // node ids before it resolves aliases; this covers the assembled ids.
  const collide = [...new Set(usesAliases)].filter((alias) => descriptors.has(alias)).sort(compareCodeUnits);
  if (collide.length > 0) throw diagnostics.error(`uses: alias ${writeAsText(collide)} collides with a node id`);

  // Every alias is resolved, so a declared-but-uncalled broken/`hub:`/missing target is loud at
  // load. The loader already resolved them in source order before the defs, so this only hits
  // its cache.
  for (const alias of usesAliases) scope.resolve(alias);

  /** One flow output's authored range, else the whole `output:` range. */
  const outputRange = (outputName: unknown): SourceRange | null => {
    try {
      if (outputsCursor !== null && typeof outputName === "string" && outputsCursor.has(outputName)) {
        return outputsCursor.child(outputName).range;
      }
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
      // Swallowed: a bare `output:` value has no named children, so the whole range stands in.
    }
    return outputsRange;
  };

  // Bound a written-out value's nesting before the passes below walk it recursively; YAML
  // aliases can make it deep while the text is shallow.
  checkWrittenValues(descriptors, outputsSection, {
    outputsTagged: outputsCursor !== null && outputsCursor.tag !== null,
    outputRange,
    outputsRange,
    diagnostics,
  });

  // A node's `output:` is a type declaration; a tag there is rejected before a declaration with
  // nothing after its tag reads as no output at all. A case has no `output:`; the case-branch
  // check below rejects it, tagged or not, with the rewrite.
  try {
    for (const [nodeId, descriptor] of descriptors) {
      if (descriptor.kind === "case") continue;
      if (!valueEntries.has(nodeId) && nodesCursor.has(nodeId)) {
        const nodeCursor = normalizeNodeSignatureCursor(nodesCursor.child(nodeId));
        if (nodeCursor.has("outputs")) rejectTaggedOutput(nodeCursor.child("outputs"), diagnostics);
      }
    }
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }

  // Split tagged (`name: !Type value`) input and output bindings before any desugar reads their
  // values, so an inline `call(...)` or `${case.output}` in the value desugars as usual.
  descriptors = splitTypedInputs(descriptors, nodesCursor, { registry, owner, catalogBuilder, diagnostics });
  descriptors = splitTypedLlmConfig(descriptors, nodesCursor, { registry, owner, catalogBuilder, diagnostics });
  let outputTypes: Map<string, ValueContract>;
  [outputsSection, outputTypes] = splitTypedOutputs(outputsSection, outputsCursor, {
    registry,
    owner,
    catalogBuilder,
    diagnostics,
  });

  // A case branch names a node or calls a flow, and a case has no `output:`. Runs BEFORE the
  // inline-call desugar so a branch's text is still as authored.
  checkCaseBranches(descriptors, {
    registry,
    inputTypes: flowInputTypes,
    outputAt: (nodeId) => caseOutputCursor(nodesCursor, nodeId, { diagnostics }),
    diagnostics,
  });

  // Desugar inline `${ f(...) }` call expressions: each becomes a synth `call` node
  // (CallDescriptor, over=null) and its host binding is rewritten to `${<synth>.output}`. Pure
  // sugar — the rest of the pipeline treats the synth nodes as ordinary `call` nodes
  // (build/edges/validate/DAG run unchanged).
  const mint = synthIdGen(); // ONE synth-id minter shared by both inline-call desugars (no collision)
  let assertsSection: Assertion[];
  [descriptors, outputsSection, assertsSection] = desugarInlineCalls(descriptors, outputsSection, {
    assertsSection: options.assertsSection,
    outputsRange,
    assertionRangeAt: flowAssertionRangeAt,
    assertionOriginAt: flowAssertionOriginAt,
    bindingStyleAt: (nodeId, field, key) =>
      bindingStyle(nodesCursor, nodeId, field, key, { valueEntries, diagnostics }),
    outputStyleAt: (key) => outputStyle(outputsCursor, key, { diagnostics }),
    typedOutputs:
      outputTypes.size > 0
        ? new Set<string | null>(isRecord(outputsSection) ? outputTypes.keys() : [null])
        : new Set(),
    nextId: mint,
    typeNames: new Set(registry.keys()),
    diagnostics,
  });

  // Lower each `human_input` `adaptive_questions:` block into a synth composer agent + the
  // rewritten gate. A descriptor-level pass (the `desugarInlineCalls` precedent above): runs
  // BEFORE the build loop so the synth agent flows through build + prompt-scope validation + ref
  // wiring like any author-written agent.
  descriptors = desugarAdaptiveQuestions(descriptors, { diagnostics });

  // `then:/else: ${call}`: a `case` branch target that is an inline call becomes a synth `call`
  // node, the then:/else: rewritten to its id. Runs BEFORE expandCaseOutputs (so the synth args +
  // the rewritten then-targets are visible to it) and shares `mint`.
  descriptors = desugarCaseCallTargets(descriptors, mint, { diagnostics });

  // The cases whose value is used, taken before the expansion below rewrites the reads; each
  // must be exhaustive (checked once producer types are known).
  const valueCaseIds = valueCases(descriptors, caseOutputReads(descriptors, outputsSection));

  // Expand `${<case>.output}` value refs to a coalesce over the case's branch targets. Runs after
  // the inline-call desugar (so synth-node bindings are covered too) and before the build loop —
  // downstream sees only ordinary branch-target refs.
  [descriptors, outputsSection] = expandCaseOutputs(descriptors, outputsSection, assertsSection, {
    outputsRange,
    assertionRangeAt: flowAssertionRangeAt,
    diagnostics,
  });

  // The inline-child builder and the binding-contract reader read it lazily; it is set just
  // before the build.
  let descriptorCache: DescriptorBuildCache | null = null;

  /** Assemble one retained body lazily in this file's callable scope. */
  const buildInlineChild = (body: InlineBody, descriptor: CallDescriptor | LoopDescriptor): ComposedFlow => {
    /** Resolve one parent producer through this assembly's build cache. */
    const producerType: ProducerType = (producerId, bindingLabel, bindingRange) => {
      const unavailableType = (): LoadError =>
        diagnostics.error(
          `cannot infer type for inline binding ${reprText(bindingLabel)}: ` +
            `producer ${reprText(producerId)} has no available static type; ` +
            explicitTypeHint(bindingLabel),
          { range: bindingRange },
        );

      if (!descriptors.has(producerId)) throw unavailableType();

      const cache = descriptorCache;
      if (cache === null) throw new Error("descriptor cache was not initialized before inline child resolution");
      const outputContract = cache.buildOne(producerId, bindingLabel, bindingRange).node.outputContract;
      if (outputContract === null) throw unavailableType();
      if (!isExactly(outputContract, ValueContract) && !isExactly(outputContract, PartialRecordContract)) {
        throw new TypeError(
          "output producer contracts must be ValueContract or PartialRecordContract values, " +
            `got ${jsTypeName(outputContract)} for ${reprText(producerId)}`,
        );
      }
      return outputContract;
    };

    let itemType: ItemType | null = null;
    if (descriptor instanceof MapDescriptor) {
      let cachedItemType: Type | null = null;

      /** Prove the MAP element type once, only when an item walk requests it. */
      itemType = (bindingLabel, bindingRange) => {
        if (cachedItemType !== null) return cachedItemType;

        const overType = inferInlineBindingType(descriptor.over, {
          parentInputTypes: flowInputTypes,
          producerType,
          typeFn: constructorType,
          bindingLabel,
          bindingRange,
          diagnostics,
        });
        if (!isListKind(overType.kind) || overType.element === undefined) {
          throw diagnostics.error(
            `cannot infer type for inline binding ${reprText(bindingLabel)}: ` +
              "MAP `over:` must have a proven list element type; " +
              explicitTypeHint(bindingLabel),
            { range: bindingRange },
          );
        }
        cachedItemType = overType.element;
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
              parentInputTypes: flowInputTypes,
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
    if (cache === null) throw new Error("descriptor cache was not initialized before binding contract resolution");
    const producers = new Map<string, OutputContract>();
    for (const producerId of new Set(bindingProducers(binding))) {
      if (!descriptors.has(producerId)) continue;
      const contract = cache.buildOne(producerId, bindingLabel, bindingRange).node.outputContract;
      if (contract !== null) producers.set(producerId, contract);
    }
    return resolveOutputBindingContract(binding, producers, flowInputContracts, { typeFn: constructorType });
  };

  descriptors = registerInlineBodies(descriptors, inlineBodies, {
    registrar: scope,
    diagnostics,
    buildChild: buildInlineChild,
  });

  // Build EVERY node kind-blind through its own descriptor's `build(ctx)` — leaf, call/map (a
  // map rides the call descriptor, no branch), loop, AND case (which lowers in its own `build`,
  // returning its data+control edges in `BuildResult.edges`). The conductor never branches on the
  // descriptor kind. Flow-owned wiring (the split): `flowWiring[nodeId][param] -> source`, an
  // empty entry for every node so the key-parity check ranges over all nodes.
  const ctxBase = new BuildContext({
    registry,
    resolver: (childId, childVersion) => scope.resolve(childId, childVersion),
    sourceFile: flowLabel,
    resolveOrigin,
    diagnostics,
    outputContractReader: readAuthoredNodeOutput,
    variableDeclarationReader: readAuthoredNodeVariables,
    bindingContractReader: readBindingContract,
    owner,
    baseDir,
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
  // exhaustiveness + the dotted-field walk in ref/assert validation). A case node has no
  // contract, so it is naturally excluded.
  const producers = new Map<string, OutputContract>();
  for (const [nodeId, node] of nodesBuilt) {
    const contract = node.outputContract;
    if (contract === null) continue;
    if (!isExactly(contract, ValueContract) && !isExactly(contract, PartialRecordContract)) {
      throw new TypeError(
        "output producer contracts must be ValueContract or PartialRecordContract values, " +
          `got ${jsTypeName(contract)} for ${reprText(nodeId)}`,
      );
    }
    producers.set(nodeId, contract);
  }

  const flowLlmValueTypes = attachLlmConfigWiring(
    buildResults,
    flowWiring,
    flowLlmConfig ?? {},
    producers,
    flowInputContracts,
    { typeFn: constructorType },
  );

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
    flowInputContracts,
    { registry },
  );
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

  const nodes = new Map(nodesBuilt);

  // Wiring/params key parity over EVERY node (catches an orphan/missing source or a node absent
  // from the wiring). Runs once the wiring is complete.
  checkWiringParity(buildResults, flowWiring, { diagnostics });

  const reconciled = [...dataEdges, ...caseEdges, ...hostEdges];

  // The proven contracts of each node-local name, for the field reads a node makes of its own
  // inputs (prompts and node asserts).
  const localContracts = localInputContracts(flowWiring, producers, flowInputContracts, buildResults, {
    narrowing,
    typeFn: constructorType,
  });

  // Reference + assert validation over the built flow. Errors locate at the node's line (a bad
  // node-input ref) or the `output:` section line (a dangling flow-output ref).
  validateReferences(buildResults, flowInputs, producers, outputsSection, flowWiring, {
    inputContracts: flowInputContracts,
    outputsRange,
    narrowing,
    localContracts,
    registry,
    resolveOrigin,
    diagnostics,
  });
  // human_input gates: structurally check a literal `questions:` list at load, and pin a
  // ref-form `questions: ${name}` to a declared input (the prompt-scope rule).
  validateHumanQuestions(buildResults, { diagnostics });
  // Node-local `asserts:` (per-node contract): validate node-locally + classify pre/post + stamp
  // onto each built leaf/call node. Shared by defs (a def's internal nodes too).
  validateNodeAsserts(nodesBuilt, descriptors, {
    assertionSiteAt: nodeAssertionSiteAt,
    typeNames: new Set(registry.keys()),
    localContracts,
    partialUnionNodes: narrowing.unordered,
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
    assertsSection,
    flowInputs,
    new Set(descriptors.keys()),
    producers,
    flowAssertionRanges,
    { inputContracts: flowInputContracts, registry, diagnostics },
  );

  // Run-ordering edges: depends_on (co-skip) / runs_after (pure order). Folded in BEFORE root
  // synthesis so an ordering-only dependent is demoted from root, and into the cycle check below
  // so an ordering cycle is loud.
  const nodeIds = new Set(descriptors.keys());
  const orderingEdges = inferOrderingEdges(descriptors, nodeIds, { diagnostics });

  // DAG assembly: synthesize the START_ID/END_ID boundary NODES + their edges over the FULL
  // reconciled body-edge set (the __start__/__end__ sentinels retire; the strings persist as the
  // boundary node ids). synthesizeBoundaryGraph mints the END_ID producer edges (one per producer
  // per output, keyed by output name, with the coalesce/optional stance) + the START_ID root
  // edges + (0-output) the bare START_ID->END_ID root edge, and returns parity-clean
  // START_ID/END_ID wiring.
  let outputs = stampFlowOutputOrigins(flowOutputs(outputsSection), outputsSection, outputsCursor, {
    owner,
    catalogBuilder,
    diagnostics,
  });
  stampInlineCallOrigins(descriptors, nodesCursor, nodesBuilt, flowWiring, outputs, assertsSection, {
    owner,
    catalogBuilder,
    diagnostics,
  });
  outputs = outputs.map((output) => {
    const contract = outputTypes.get(output.name);
    if (contract === undefined) return output;
    return new FlowOutput(output.name, new Binding(output.binding.source, { origin: output.binding.origin, contract }));
  });
  // An output that reads only nodes CASE routing may skip would leave END unreachable; the
  // explicit null default makes its END edge optional, so the run returns null.
  const structural = [...reconciled, ...orderingEdges];
  const skips = new SkipModel(
    new Set(descriptors.keys()),
    structural,
    exhaustiveCases(descriptors, producers, flowInputContracts, { narrowing }),
  );
  outputs = nullSkippedOutputs(outputs, skips);
  const valueMode = outputsSection !== null && !isRecord(outputsSection);
  // Arithmetic on such a value would fail at runtime with null; ask for a `:-` default. A
  // searched `when:` is checked as authored: its wiring holds only its `__rN` reads.
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
  checkSkippedArithmetic(flowWiring, outputs, skips, buildResults, {
    valueMode,
    outputsRange,
    resolveOrigin,
    diagnostics,
    searchedWhens,
  });
  // Typed flow outputs that can never fit their declared type (needs the output contracts
  // attached above).
  checkTypedOutputs(outputs, producers, flowInputContracts, {
    valueMode,
    outputsRange,
    typeFn: constructorType,
    registry,
    resolveOrigin,
    diagnostics,
  });
  const [boundaryNodes, boundaryEdges, boundaryWiring] = synthesizeBoundaryGraph(
    inputs,
    outputs,
    nodeIds,
    structural,
    lowered,
    { record: isRecord(outputsSection) },
  );
  for (const [nodeId, node] of boundaryNodes) nodes.set(nodeId, node);
  for (const [nodeId, wiring] of boundaryWiring) flowWiring.set(nodeId, wiring);
  const edges: Edge[] = [...structural, ...boundaryEdges];

  // Before the type tables, so a constructor in a config value gets the config's message.
  checkLlmConfigSources(
    buildResults,
    flowLlmConfig ?? {},
    owner[0] === "def" ? `defs entry ${reprText(owner[1]!)} llm_config` : "flow llm_config",
    producers,
    flowInputContracts,
    { typeFn: constructorType, registry, resolveOrigin, diagnostics },
  );
  // Type tables for `isinstance`, once START/END carry the lowered flow asserts and outputs.
  fillTypeTables(
    nodes,
    flowWiring,
    buildResults,
    descriptors,
    assertsSection,
    flowAssertionRanges,
    outputsCursor,
    outputsRange,
    registry,
    diagnostics,
  );
  // After the type tables, so `isinstance` in `over:`/`until:` keeps its own error.
  checkReservedSources(buildResults, flowWiring, producers, flowInputContracts, {
    narrowing,
    typeFn: constructorType,
    registry,
    resolveOrigin,
    diagnostics,
  });

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
    edges.filter((edge) => synthHosts.get(edge.from) === edge.to && edge.inputGroup === QPARAM).map((edge) => edge.id),
  );
  rejectCycles(edges, nodeIds, {
    nodeRanges,
    diagnostics,
    gateIds: new Set(hostEdges.filter((edge) => edge.inputGroup).map((edge) => edge.id)),
    synthHosts,
    synthWires,
  });
  checkCaseHandles(nodes, edges, { diagnostics });

  const compiled = CompiledFlow.fromParts(nodes, edges, {
    outputs,
    wiring: flowWiring,
    flowLlmConfig: flowLlmConfig ?? {},
    flowId,
    memories,
  });
  compiled.flowLlmValueTypes = flowLlmValueTypes;
  return new ComposedFlow(compiled, { version, name, description });
}

// ---------- Passes not ported yet ----------
//
// Every placeholder, by owner. Kind 1 is a stub in its owner's sub-block below: the owner deletes
// the stub and imports the port under its import slot, keeping the sub-block header. A local type
// alias is replaced by a `type` import of the same name under that slot, never just deleted.
// `TypeFn` is imported from compose/helpers/output_contracts.ts.
// Kind 2 is a pipeline helper stubbed at its Python position: the owner ports its body in place,
// adding the private helpers named beside it. Kind 3 is a `const` inside `assemble`: the owner
// replaces that line. Until then a placeholder returns nothing, except as noted; "throws" means
// `Error("<python name> is not ported yet (T3-xx)")`.

// T3-15: compose/nodes/{agent,code,tool,human_input,wait}.ts

// T3-17: compose/desugar/*.ts, compose/edges/*.ts, compose/build/{boundary,inline_inputs}.ts

// T3-18: compose/validate/{value_nesting,tool_inputs,wiring_parity,references,human_questions,
// reserved_sources,cycles,handles}.ts

// T3-19: compose/validate/{typed_bindings,call_types,loop_types,llm_config_sources}.ts

// T3-20: compose/validate/{case_branches,narrowing,case_exhaustive,asserts,skips}.ts
