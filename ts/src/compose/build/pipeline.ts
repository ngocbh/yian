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
import type { Node } from "../../nodes/base.ts";
import {
  asBinding,
  type Assertion,
  Binding,
  isExactly,
  type OutputContract,
  Parameter,
  type ValueContract,
  type ValuePath,
} from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isList, isRecord, type Type, type Value } from "../../typesys/values.ts";
import { type LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import {
  oldFormSuggestion,
  oldTypedText,
  QUOTED_STYLES,
  readTaggedValue,
  suggestTag,
  yamlQuoted,
} from "../helpers/tags.ts";
import type { BuildContext, BuildResult, NodeDescriptor, OriginPresenter } from "../nodes/base.ts";
import { normalizeNodeSignatureCursor } from "../parse/nodes.ts";
import type { SourceCatalogBuilder, SourceSitePath } from "../source/catalog.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";
// T3-13 imports
// T3-15 imports
// T3-17 imports
// T3-18 imports
// T3-19 imports
// T3-20 imports

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

/** Python's `compose.build.pipeline._register_inline_bodies`; not ported yet. */
function registerInlineBodies(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  bodies: ReadonlyMap<string, InlineBody>,
  options: {
    registrar: PrivateRegistrar;
    diagnostics?: LoadDiagnostics | null;
    buildChild: (body: InlineBody, descriptor: NodeDescriptor) => ComposedFlow;
  },
): Map<string, NodeDescriptor> {
  return new Map(descriptors);
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

/** Python's `compose.build.pipeline._split_typed_llm_config`; not ported yet. */
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
  return new Map(descriptors);
}

/** Python's `compose.build.pipeline._attach_llm_config_wiring`; not ported yet. */
function attachLlmConfigWiring(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowWiring: ReadonlyMap<string, Map<string, unknown>>,
  flowLlmConfig: unknown,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { typeFn?: TypeFn | null } = {},
): (readonly [ValuePath, Type | null])[] {
  return [];
}

/** Python's `compose.build.pipeline._binding_style`; not ported yet. */
function bindingStyle(
  nodesCursor: SourceCursor,
  nodeId: string,
  field: string,
  key: string,
  options: { valueEntries: ReadonlySet<string>; diagnostics: LoadDiagnostics },
): string | null {
  return null;
}

/** Python's `compose.build.pipeline._output_style`; not ported yet. */
function outputStyle(
  outputsCursor: SourceCursor | null,
  key: unknown,
  options: { diagnostics: LoadDiagnostics },
): string | null {
  return null;
}

/** Python's `compose.build.pipeline._case_output_cursor`; not ported yet. */
function caseOutputCursor(
  nodesCursor: SourceCursor,
  nodeId: string,
  options: { diagnostics: LoadDiagnostics },
): SourceCursor | null {
  return null;
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

/** Python's `compose.build.pipeline._split_typed_outputs`; not ported yet. */
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
  return [outputsSection, new Map()];
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

/** Python's `compose.build.pipeline._stamp_inline_call_origins`; not ported yet. */
function stampInlineCallOrigins(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  nodesCursor: SourceCursor,
  nodes: ReadonlyMap<string, Node>,
  flowWiring: ReadonlyMap<string, Map<string, unknown>>,
  flowOutputs: readonly FlowOutput[],
  flowAsserts: readonly Assertion[] | null,
  options: { owner: readonly string[]; catalogBuilder: SourceCatalogBuilder; diagnostics: LoadDiagnostics },
): void {}

/** Python's `compose.build.pipeline._fill_type_tables`; not ported yet. */
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
): void {}

/** Python's `compose.build.pipeline._value_cases`; not ported yet. */
function valueCases(descriptors: ReadonlyMap<string, NodeDescriptor>, read: ReadonlySet<string>): ReadonlySet<string> {
  return new Set();
}

// ---------- Passes not ported yet ----------
//
// Every placeholder, by owner. Kind 1 is a stub in its owner's sub-block below: the owner deletes
// the stub (and its sub-block's local type aliases) and imports the port under its import slot,
// keeping the sub-block header. Kind 2 is a pipeline helper stubbed at its Python position: the
// owner ports its body in place, adding the private helpers named beside it. Kind 3 is a `const`
// inside `assemble`: the owner replaces that line. Until then a placeholder returns nothing,
// except as noted; "throws" means `Error("<python name> is not ported yet (T3-xx)")`.
//
//   T3-13  constructorTypeFn (1): returns `() => null`. T3-13's merge or T3-17, whichever edits
//          this file first, swaps it for the import, with `TypeFn`.
//   T3-15  splitTypedLlmConfig (2): returns the descriptors.
//   T3-17  extractInlineBodies (1): returns `[nodesCursor, new Map()]`.
//          readValueEntry (1): throws.
//          isValueEntry (1): returns `false`.
//          splitTypedOutputs (2, with _check_output_names, _reject_non_json_literals and
//            _raise_if_non_json_literal): returns `[outputsSection, new Map()]`.
//          synthIdGen (1): returns a minter that throws.
//          desugarInlineCalls (1): returns `[descriptors, outputsSection, assertsSection]`;
//            bindingStyle, outputStyle (2): return `null`.
//          desugarAdaptiveQuestions, desugarCaseCallTargets (1): return the descriptors.
//          caseOutputReads (1), valueCases (2): return `new Set()`.
//          expandCaseOutputs (1): returns `[descriptors, outputsSection]`.
//          buildInlineChild (3, `(body: InlineBody, descriptor: NodeDescriptor) => ComposedFlow`):
//            throws.
//          readBindingContract (3, `BindingContractReader | null`): `null`.
//          registerInlineBodies (2): returns the descriptors.
//          inferDataEdges, hostControlEdges, inferOrderingEdges (1): return `[]`.
//          stampInlineCallOrigins (2).
//          synthesizeBoundaryGraph (1): returns `[new Map(), [], new Map()]`, so
//            `ComposedFlow.input` throws until it is ported.
//          synthWires (3, `ReadonlySet<string>`, reads `QPARAM`): `new Set()`.
//   T3-18  checkWrittenValues, stampToolInputTypes, checkWiringParity, validateReferences,
//          validateHumanQuestions, checkReservedSources, rejectCycles, checkCaseHandles (1).
//          fillTypeTables (2, with _site_range).
//   T3-19  attachLlmConfigWiring (2, with _llm_value_types): returns `[]`.
//          checkTypedInputs, checkCallMapTypes, checkLoopTypeContract, checkTypedOutputs,
//          checkLlmConfigSources (1).
//          inlineLoops (3, `ReadonlySet<string>`): `new Set()`.
//          localInputContracts (1): returns `new Map()`.
//   T3-20  checkCaseBranches (1); caseOutputCursor (2): returns `null`.
//          computeNarrowing (1): returns `{ unordered: new Set() }`.
//          checkCaseWhenValues, checkImpossibleWhens, checkCaseExhaustive, validateNodeAsserts,
//          checkSkippedArithmetic (1).
//          lowerFlowAsserts (1): returns `null`.
//          exhaustiveCases (1): returns `new Set()`; skipModel (1, Python's `SkipModel` class):
//            returns `null`; nullSkippedOutputs (1): returns the outputs.
//          searchedWhens (3, `ReadonlyMap<string, readonly SearchedWhen[]>`): `new Map()`.

// T3-13: compose/helpers/{isinstance_types,output_contracts}.ts

/** Python's `compose.helpers.output_contracts.TypeFn`: the type a constructor name builds. */
type TypeFn = (name: string) => Type | null;

/** Python's `compose.helpers.isinstance_types.constructor_type_fn`; not ported yet. */
function constructorTypeFn(registry: TypeRegistry): TypeFn {
  return () => null;
}

// T3-15: compose/nodes/{agent,code,tool,human_input,wait}.ts

// T3-17: compose/desugar/*.ts, compose/edges/*.ts, compose/build/{boundary,inline_inputs}.ts

type InlineBody = unknown;
type ValueEntryDescriptor = NodeDescriptor;

/** Python's `compose.desugar.inline_bodies.extract_inline_bodies`; not ported yet. */
function extractInlineBodies(
  nodesCursor: SourceCursor,
  options: { owner: readonly string[] },
): [nodesCursor: SourceCursor, inlineBodies: Map<string, InlineBody>] {
  return [nodesCursor, new Map()];
}

/** Python's `compose.desugar.value_entries.read_value_entry`; not ported yet. */
function readValueEntry(
  nodeId: string,
  cursor: SourceCursor,
  options: {
    registry: TypeRegistry;
    owner: readonly string[];
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
  },
): ValueEntryDescriptor {
  throw new Error("read_value_entry is not ported yet (T3-17)");
}

/** Python's `isinstance(descriptor, compose.desugar.value_entries.ValueEntryDescriptor)`; not ported yet. */
function isValueEntry(descriptor: NodeDescriptor): boolean {
  return false;
}

/** Python's `compose.desugar.inline_calls._synth_id_gen`; not ported yet. */
function synthIdGen(): () => string {
  return () => {
    throw new Error("_synth_id_gen is not ported yet (T3-17)");
  };
}

/** Python's `compose.desugar.inline_calls.desugar_inline_calls`; not ported yet. */
function desugarInlineCalls(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
  options: {
    assertsSection?: readonly unknown[] | null;
    outputsRange?: SourceRange | null;
    assertionRangeAt?: ((index: number) => SourceRange | null) | null;
    assertionOriginAt?: ((index: number) => OriginRef | null) | null;
    bindingStyleAt?: ((nodeId: string, field: string, key: string) => string | null) | null;
    outputStyleAt?: ((key: unknown) => string | null) | null;
    typedOutputs?: ReadonlySet<string | null>;
    nextId?: (() => string) | null;
    typeNames?: ReadonlySet<string>;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): [descriptors: Map<string, NodeDescriptor>, outputsSection: unknown, assertsSection: Assertion[]] {
  return [new Map(descriptors), outputsSection, (options.assertsSection ?? []) as Assertion[]];
}

/** Python's `compose.desugar.adaptive_questions.desugar_adaptive_questions`; not ported yet. */
function desugarAdaptiveQuestions(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Map<string, NodeDescriptor> {
  return new Map(descriptors);
}

/** Python's `compose.desugar.case_targets.desugar_case_call_targets`; not ported yet. */
function desugarCaseCallTargets(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  mint: () => string,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Map<string, NodeDescriptor> {
  return new Map(descriptors);
}

/** Python's `compose.desugar.case_outputs.case_output_reads`; not ported yet. */
function caseOutputReads(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
): ReadonlySet<string> {
  return new Set();
}

/** Python's `compose.desugar.case_outputs.expand_case_outputs`; not ported yet. */
function expandCaseOutputs(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
  assertsSection: readonly Assertion[],
  options: {
    outputsRange?: SourceRange | null;
    assertionRangeAt?: ((index: number) => SourceRange | null) | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): [descriptors: Map<string, NodeDescriptor>, outputsSection: unknown] {
  return [new Map(descriptors), outputsSection];
}

/** Python's `compose.edges.data.infer_data_edges`; not ported yet. */
function inferDataEdges(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
): Edge[] {
  return [];
}

/** Python's `compose.edges.hosts.host_control_edges`; not ported yet. */
function hostControlEdges(descriptors: ReadonlyMap<string, NodeDescriptor>, edges: readonly Edge[]): Edge[] {
  return [];
}

/** Python's `compose.edges.ordering.infer_ordering_edges`; not ported yet. */
function inferOrderingEdges(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  nodeIds: ReadonlySet<string>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Edge[] {
  return [];
}

/** Python's `compose.build.boundary.synthesize_boundary_graph`; not ported yet. */
function synthesizeBoundaryGraph(
  inputParameters: readonly Parameter[],
  outputs: readonly FlowOutput[],
  nodeIds: ReadonlySet<string>,
  bodyEdges: readonly Edge[],
  options: { lowered?: LoweredAsserts | null; record: boolean },
): [nodes: Map<string, Node>, edges: Edge[], wiring: Map<string, Map<string, unknown>>] {
  return [new Map(), [], new Map()];
}

// T3-18: compose/validate/{value_nesting,tool_inputs,wiring_parity,references,human_questions,
// reserved_sources,cycles,handles}.ts

/** Python's `compose.validate.value_nesting.check_written_values`; not ported yet. */
function checkWrittenValues(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  outputsSection: unknown,
  options: {
    outputsTagged: boolean;
    outputRange: (name: unknown) => SourceRange | null;
    outputsRange: SourceRange | null;
    diagnostics: LoadDiagnostics;
  },
): void {}

/** Python's `compose.validate.tool_inputs.stamp_tool_input_types`; not ported yet. */
function stampToolInputTypes(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: { narrowing?: Narrowing | null; typeFn?: TypeFn | null } = {},
): void {}

/** Python's `compose.validate.wiring_parity.check_wiring_parity`; not ported yet. */
function checkWiringParity(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): void {}

/** Python's `compose.validate.references.validate_references`; not ported yet. */
function validateReferences(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowInputs: ReadonlySet<string>,
  producers: ReadonlyMap<string, OutputContract>,
  outputs: unknown,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  options: {
    inputContracts?: ReadonlyMap<string, ValueContract> | null;
    outputsRange?: SourceRange | null;
    narrowing?: Narrowing | null;
    localContracts?: ReadonlyMap<string, ReadonlyMap<string, LocalInput>> | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

/** Python's `compose.validate.human_questions.validate_human_questions`; not ported yet. */
function validateHumanQuestions(
  buildResults: ReadonlyMap<string, BuildResult>,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): void {}

/** Python's `compose.validate.reserved_sources.check_reserved_sources`; not ported yet. */
function checkReservedSources(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: {
    narrowing?: Narrowing | null;
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

/** Python's `compose.validate.cycles.reject_cycles`; not ported yet. */
function rejectCycles(
  edges: readonly Edge[],
  nodeIds: ReadonlySet<string>,
  options: {
    nodeRanges?: ReadonlyMap<string, SourceRange> | null;
    diagnostics?: LoadDiagnostics | null;
    gateIds?: ReadonlySet<string>;
    synthHosts?: ReadonlyMap<string, string> | null;
    synthWires?: ReadonlySet<string>;
  } = {},
): void {}

/** Python's `compose.validate.handles.check_case_handles`; not ported yet. */
function checkCaseHandles(
  nodes: ReadonlyMap<string, Node>,
  edges: readonly Edge[],
  options: { diagnostics?: LoadDiagnostics | null } = {},
): void {}

// T3-19: compose/validate/{typed_bindings,call_types,loop_types,llm_config_sources}.ts

type LocalInput = unknown;

/** Python's `compose.validate.typed_bindings.check_typed_inputs`; not ported yet. */
function checkTypedInputs(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: {
    narrowing?: Narrowing | null;
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

/** Python's `compose.validate.call_types.check_call_map_types`; not ported yet. */
function checkCallMapTypes(
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
): void {}

/** Python's `compose.validate.loop_types.check_loop_type_contract`; not ported yet. */
function checkLoopTypeContract(
  buildResults: ReadonlyMap<string, BuildResult>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  options: {
    inlineLoops?: ReadonlySet<string>;
    registry?: TypeRegistry | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

/** Python's `compose.validate.typed_bindings.local_input_contracts`; not ported yet. */
function localInputContracts(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: { narrowing?: Narrowing | null; typeFn?: TypeFn | null } = {},
): Map<string, Map<string, LocalInput>> {
  return new Map();
}

/** Python's `compose.validate.typed_bindings.check_typed_outputs`; not ported yet. */
function checkTypedOutputs(
  flowOutputs: readonly FlowOutput[],
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: {
    valueMode?: boolean;
    outputsRange?: SourceRange | null;
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

/** Python's `compose.validate.llm_config_sources.check_llm_config_sources`; not ported yet. */
function checkLlmConfigSources(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowLlmConfig: unknown,
  flowLabel: string,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: {
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

// T3-20: compose/validate/{case_branches,narrowing,case_exhaustive,asserts,skips}.ts

/** What each node may assume about union values from the CASE branches that route it. */
interface Narrowing {
  /** Nodes a topological order could not reach, on or after a cycle; they get no facts. */
  readonly unordered: ReadonlySet<string>;
}
type LoweredAsserts = unknown;
type SkipModel = unknown;
/** One searched `when:` as authored: its index, text, origin and range. */
type SearchedWhen = readonly [index: number, when: unknown, origin: OriginRef | null, range: SourceRange | null];

/** Python's `compose.validate.case_branches.check_case_branches`; not ported yet. */
function checkCaseBranches(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: {
    registry: TypeRegistry;
    inputTypes: ReadonlyMap<string, Type>;
    outputAt: (nodeId: string) => SourceCursor | null;
    diagnostics: LoadDiagnostics;
  },
): void {}

/** Python's `compose.validate.narrowing.compute_narrowing`; not ported yet. */
function computeNarrowing(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  edges: readonly Edge[],
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  options: { registry?: TypeRegistry | null } = {},
): Narrowing {
  return { unordered: new Set() };
}

/** Python's `compose.validate.case_exhaustive.check_case_when_values`; not ported yet. */
function checkCaseWhenValues(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: { narrowing?: Narrowing | null; diagnostics?: LoadDiagnostics | null } = {},
): void {}

/** Python's `compose.validate.narrowing.check_impossible_whens`; not ported yet. */
function checkImpossibleWhens(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  narrowing: Narrowing,
  options: { registry?: TypeRegistry | null; diagnostics?: LoadDiagnostics | null } = {},
): void {}

/** Python's `compose.validate.case_exhaustive.check_case_exhaustive`; not ported yet. */
function checkCaseExhaustive(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  options: {
    narrowing?: Narrowing | null;
    valueCases?: ReadonlySet<string>;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

/** Python's `compose.validate.asserts.validate_node_asserts`; not ported yet. */
function validateNodeAsserts(
  nodes: ReadonlyMap<string, Node>,
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  options: {
    assertionSiteAt?:
      | ((nodeId: string, index: number) => readonly [OriginRef | null, SourceRange | null] | null)
      | null;
    typeNames?: ReadonlySet<string>;
    localContracts?: ReadonlyMap<string, ReadonlyMap<string, LocalInput>> | null;
    partialUnionNodes?: ReadonlySet<string>;
    registry?: TypeRegistry | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {}

/** Python's `compose.validate.asserts.lower_flow_asserts`; not ported yet. */
function lowerFlowAsserts(
  assertList: readonly Assertion[],
  flowInputs: ReadonlySet<string>,
  validTargets: ReadonlySet<string>,
  producers: ReadonlyMap<string, OutputContract>,
  assertRanges: readonly (SourceRange | null)[] | null = null,
  options: {
    inputContracts?: ReadonlyMap<string, ValueContract> | null;
    registry?: TypeRegistry | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): LoweredAsserts {
  return null;
}

/** Python's `compose.validate.case_exhaustive.exhaustive_cases`; not ported yet. */
function exhaustiveCases(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract> | null = null,
  options: { narrowing?: Narrowing | null } = {},
): Set<string> {
  return new Set();
}

/** Python's `compose.validate.skips.SkipModel` constructor; not ported yet. */
function skipModel(
  nodeIds: Iterable<string>,
  edges: Iterable<Edge>,
  exhaustive: ReadonlySet<string>,
  options: { limit?: number | null } = {},
): SkipModel {
  return null;
}

/** Python's `compose.validate.skips.null_skipped_outputs`; not ported yet. */
function nullSkippedOutputs(flowOutputs: FlowOutput[], model: SkipModel): FlowOutput[] {
  return flowOutputs;
}

/** Python's `compose.validate.skips.check_skipped_arithmetic`; not ported yet. */
function checkSkippedArithmetic(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  flowOutputs: readonly FlowOutput[],
  model: SkipModel,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: {
    valueMode?: boolean;
    outputsRange?: SourceRange | null;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics | null;
    searchedWhens?: ReadonlyMap<string, readonly SearchedWhen[]> | null;
  } = {},
): void {}
