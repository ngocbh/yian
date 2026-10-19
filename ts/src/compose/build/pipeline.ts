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

import { CompiledFlow, FlowOutput } from "../../compile/model.ts";
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
import { asBinding, Binding, isExactly, Parameter, type ValueContract } from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isList, isRecord, type Value } from "../../typesys/values.ts";
import { type LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import {
  oldFormSuggestion,
  oldTypedText,
  QUOTED_STYLES,
  readTaggedValue,
  suggestTag,
  yamlQuoted,
} from "../helpers/tags.ts";
import type { BuildContext, BuildResult, NodeDescriptor } from "../nodes/base.ts";
import { normalizeNodeSignatureCursor } from "../parse/nodes.ts";
import type { SourceCatalogBuilder, SourceSitePath } from "../source/catalog.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";

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
export type BuildableDescriptor = Pick<NodeDescriptor, "build"> & { readonly range?: SourceRange | null };

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
        const descriptorRange = descriptor.range ?? null;
        if (descriptorRange === null) throw error;
        throw this.#context.diagnostics.prepend(error, { range: descriptorRange, cause: error });
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

        const coordinates = [range.startLine, range.startCol, range.endLine, range.endCol];
        if (coordinates.some((coordinate) => !Number.isInteger(coordinate) || coordinate <= 0)) {
          return unavailable;
        }
        const [startLine, startCol, endLine, endCol] = coordinates as [number, number, number, number];
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
export function splitTypedValue(
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

// ---------- Passes not ported yet ----------
//
// Each sub-block holds the stand-ins for one owner's passes, under a header naming the owner and
// its files. The owner's change deletes its stubs and imports the ported passes; the call sites
// keep their shape.

// T3-17: compose/desugar/value_entries.ts

/** Whether `descriptor` is a value entry's; none is until value entries are ported. */
function isValueEntry(_descriptor: NodeDescriptor): boolean {
  return false;
}
