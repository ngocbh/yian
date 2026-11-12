/**
 * The top-level loader: Compose YAML text to a runnable `ComposedFlow`.
 *
 * The file-I/O front door: own one source document, build the typedefs registry, compose the
 * call resolver, resolve every `uses:` import, load every `defs:` callable, and hand the parsed
 * sections to the post-parse conductor (`build/pipeline.ts`'s `assemble`) which returns the
 * `ComposedFlow`. This module owns the source projections of the root and of both definition forms
 * (a single-node def projects onto its one synthesized node), resolver composition
 * (`makeFileResolver`, `CallableScope`) and the defs loaders (`loadDef`, `loadSingleNodeDef`), not
 * the assembly pipeline itself.
 *
 * A `call:` resolves a compiler-owned inline body, then an in-file `defs:` callable (built by the
 * same `assemble`), then a `uses:` alias resolved through the injected `childResolver`
 * (`(path, version) -> ComposedFlow`, sync or async). Every alias resolves, in source order, before
 * any def or the root assembles, so assembly itself stays synchronous and reads each import from
 * the scope's memo. A recursive or mutually recursive def is refused. The loader derives the
 * callable's signature and bakes its compiled flow onto the built `CallNode` so `run` drives the
 * embedded child.
 *
 * Source resolution follows two cache boundaries. A physical-file cache entry is canonical and
 * complete. `CallableScope` validates it, copies and rebases its compiled origins once per lexical
 * `uses:` alias, and reuses that alias view at every callsite. After the root is assembled, this
 * module freezes the one document catalog, filters local and alias authorities to the exact
 * reachable keys, and installs `ComposedFlow.originResolver`. Source-free graphs carry `null`;
 * private `assemble` products keep the resolver unset.
 *
 * Before a file is parsed for good, the file each `!include` names is read: from disk, against the
 * including file's folder, or from a recorded source, and for an `!include(T)` whose `T` isn't
 * text read as one plain YAML value. The including file records where each path went.
 * After a file assembles, its CODE nodes are checked (each def's in written order, then the
 * root's) against the file's base directory, and inside the load's `codeRoot` when it has one.
 * Only the root load sees the fully assembled graph, so `loadFlow` then refuses an `llm_config`
 * value a child flow's agent would inherit before it is known.
 *
 * A flow's chain of `uses:` imports is at most `current().maxCallNesting` files deep, the call
 * nesting limit, counted apart from call depth; each loaded flow records its own `importDepth`.
 *
 * Imports flow down only: this module composes the compose package's own components plus
 * `compile`, `defaults`, `diagnostics`, `llm_clients`, `nodes`, `platform` and `typesys`, and the
 * `FlowSource` type of `runtime/log`; nothing in the engine imports it back.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import fsp from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve as resolvePath } from "node:path";

import { fieldLabel, findInheritedLlmValue, inheritedValueFix } from "../compile/llm_cascade.ts";
import { type CompiledFlow, type Flow, nodeChild } from "../compile/model.ts";
import { iterFlowOrigins, rebaseFlowOrigins } from "../compile/origins.ts";
import { current, type Env, processEnv, withEnv } from "../defaults.ts";
import {
  compareCodeUnits,
  DiagnosticFrame,
  OriginRef,
  type OriginResolver,
  ResolverTable,
  reprText,
  type SourceRange,
} from "../diagnostics.ts";
import type { Node } from "../nodes/base.ts";
import { Binding, type ValuePath, ValueTemplate } from "../nodes/contracts.ts";
import { isSystemError, leavesRoot, readRegularFile, resolvedPath, statOrNull } from "../platform.ts";
import { YianTypeError } from "../typesys/errors.ts";
import { readTypedefs, typeFor, typeName, type TypeRegistry, withMemoryTypes } from "../typesys/types.ts";
import {
  isList,
  isRecord,
  jsTypeName,
  TypeCheckError,
  ValueKind,
  type Value,
  type ValueRecord,
  valueRepr,
  writeAsText,
} from "../typesys/values.ts";
import type { FlowSource } from "../runtime/log.ts";
import { assemble, type AssembleOptions, type CallableScope as AssemblyScope, ComposedFlow } from "./build/pipeline.ts";
import { INLINE_FLOW_PREFIX } from "./desugar/inline_bodies.ts";
import { LoadDiagnostics, LoadError, rejectRemovedMemoryKey } from "./helpers/errors.ts";
import { type MemoryClass, type MemoryTypeTable, memoryTypeTools, NO_MEMORY_TYPES } from "./helpers/memory.ts";
import { boundaryFieldRanges, pathText } from "./helpers/source.ts";
import { includeFile, rejectUnreadTags } from "./helpers/tags.ts";
import { parseUsesRef } from "./helpers/uses.ts";
import { checkEnvValues } from "./helpers/value_nesting.ts";
import { maxToolIterationsError, readLlmConfig, validateLlmConfig } from "./nodes/agent.ts";
import { notTextError, valueTypeError } from "./nodes/base.ts";
import { KINDS, REMOVED_KINDS } from "./nodes/index.ts";
import { checkFlowCode, type CodeCheckOptions } from "./nodes/code.ts";
import { callNestingScope } from "./nodes/resolver.ts";
import {
  flowNodesCursor,
  keyProblem,
  parseFile,
  parseSourceDocument,
  signatureNodeBody,
  signatureParams,
} from "./parse/file.ts";
import { scopeOrigin, SourceCatalog, SourceCatalogBuilder } from "./source/catalog.ts";
import {
  includeExample,
  includeTag,
  SourceAccessError,
  SourceCursor,
  SourceDocument,
  SourceSyntaxError,
} from "./source/index.ts";

/**
 * Resolves a `uses:` ref to its flow: `path` is the ref's flow path and `version` its
 * `@<version>` (`null` when unpinned). It may return the flow or a promise of it; a pinned ref's
 * flow must declare that `version`, or the load fails.
 */
export type ChildResolver = (path: string, version: string | null) => ComposedFlow | Promise<ComposedFlow>;

const UNLOCATED_DIAGNOSTICS = new LoadDiagnostics();

// The fatal decoder refuses a file that isn't UTF-8; a BOM stays in the text, and the reader
// turns `\r\n` and a lone `\r` into `\n`.
const UTF8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/** Say whether a file-system error means the path doesn't exist. */
function isMissingPath(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException).code;
  return code === "ENOENT" || code === "ENOTDIR" || code === "ELOOP";
}

/** Return how a message shows a thrown value: an error's message, anything else as text. */
function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Return `flow` with some of its fields replaced, as a new frozen `ComposedFlow`. */
function replaceFlow(
  flow: ComposedFlow,
  changes: {
    compiled?: CompiledFlow;
    originResolver?: OriginResolver | null;
    importDepth?: number;
    source?: FlowSource | null;
    memoryTypes?: readonly MemoryClass[];
  },
): ComposedFlow {
  return new ComposedFlow(changes.compiled ?? flow.compiled, {
    version: flow.version,
    name: flow.name,
    description: flow.description,
    originResolver: changes.originResolver === undefined ? flow.originResolver : changes.originResolver,
    importDepth: changes.importDepth === undefined ? flow.importDepth : changes.importDepth,
    source: changes.source === undefined ? flow.source : changes.source,
    memoryTypes: changes.memoryTypes ?? flow.memoryTypes,
  });
}

/**
 * Restore normalized node signatures to the singular authored keys.
 *
 * The normalized `ComposeFile` form uses `inputs`/`outputs`. A spanless def fallback must look
 * like raw YAML before inline-body extraction, so only each direct node body's signature keys are
 * restored. The walk follows an actual nested `nodes:` graph and never descends into other
 * literal mappings or lists.
 */
function restoreNodeSurface(nodes: Value): Value {
  if (!isRecord(nodes)) return nodes;
  return Object.fromEntries(
    Object.entries(nodes).map(([nodeId, rawBody]) => {
      if (!isRecord(rawBody)) return [nodeId, rawBody];
      // A spread defines a `__proto__` key as data.
      const body: ValueRecord = { ...rawBody };
      for (const [plural, singular] of [
        ["inputs", "input"],
        ["outputs", "output"],
      ] as const) {
        if (Object.hasOwn(body, plural)) {
          const value = body[plural]!;
          delete body[plural];
          body[singular] = value;
        }
      }
      if (isRecord(body["nodes"])) body["nodes"] = restoreNodeSurface(body["nodes"]);
      return [nodeId, body];
    }),
  );
}

/** Return the authored def cursor's `nodes:`, or a surface-restored virtual fallback. Exported for tests. */
export function defAssemblyCursor(defCursor: SourceCursor | null, normalizedNodes: Value): SourceCursor {
  if (defCursor !== null) {
    try {
      return defCursor.child("nodes");
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
    }
  }
  return SourceCursor.virtual(restoreNodeSurface(normalizedNodes));
}

/** Return one existing child source, or `null` for a virtual fallback. */
function sourceChild(cursor: SourceCursor | null, key: string): SourceCursor | null {
  if (cursor === null) return null;
  try {
    return cursor.has(key) ? cursor.child(key) : null;
  } catch (error) {
    if (error instanceof SourceAccessError) return null;
    throw error;
  }
}

/**
 * Project a flat def into one source-aligned synthetic node mapping, `{<name>: <node body>}`.
 *
 * The node body is the def's body without its signature, plus one generated `p: ${input.p}`
 * binding per parameter and the def's `output:` type. Each projected entry keeps the authored
 * source it came from: a generated binding points at its parameter's declaration. Exported for
 * tests.
 */
export function singleNodeDefCursor(name: string, body: ValueRecord, defCursor: SourceCursor | null): SourceCursor {
  const paramTypes = isRecord(body["inputs"]) ? body["inputs"] : {};
  const nodeBody = signatureNodeBody(body, paramTypes, { input: "inputs", output: "outputs" });
  const generatedInputs = nodeBody["input"] as ValueRecord;

  if (defCursor === null) return SourceCursor.virtual(Object.fromEntries([[name, nodeBody]]));

  const bodyEntries = new Map<string, readonly [Value, SourceCursor | null]>(
    Object.entries(nodeBody)
      .filter(([key]) => key !== "input" && key !== "output")
      .map(([key, value]) => [key, [value, sourceChild(defCursor, key)]]),
  );

  const inputSource = sourceChild(defCursor, "input");
  const inputCursor =
    inputSource !== null && isRecord(inputSource._valueView())
      ? inputSource.projectMapping(
          new Map(
            Object.entries(generatedInputs).map(([param, binding]) => [
              param,
              [binding, sourceChild(inputSource, param)],
            ]),
          ),
        )
      : null;
  bodyEntries.set("input", [generatedInputs, inputCursor]);

  if (Object.hasOwn(nodeBody, "output")) {
    bodyEntries.set("output", [nodeBody["output"]!, sourceChild(defCursor, "output")]);
  }

  const bodyCursor = defCursor.projectMapping(bodyEntries);
  return defCursor.projectMapping(new Map([[name, [nodeBody, bodyCursor]]]));
}

// A `defs:` child flow is either a multi-node body or a flat single-node `kind:` body. Both share
// the file's metadata and typedefs rather than declaring `id`/`name`/`typedefs` themselves.
// `asserts:` are allowed: the call and map child seam enforces a def's asserts against the child
// pool (boundary before the child runs, post after it ends). `env:` is the def's own flow-level
// default for its nodes; a parent's `env:` doesn't reach a called child.
const DEF_ALLOWED: ReadonlySet<string> = new Set([
  "inputs",
  "nodes",
  "outputs",
  "asserts",
  "llm_config",
  "env",
]);
// The kinds a single-node def may use, sorted: a case routes to other nodes, which a one-node
// def has none of.
const SINGLE_NODE_DEF_KINDS: readonly string[] = [...KINDS.keys()]
  .filter((kind) => kind !== "case")
  .sort(compareCodeUnits);
// The authored spellings of `DEF_ALLOWED`, sorted, for error messages.
const DEF_SURFACE_KEYS: readonly string[] = [...DEF_ALLOWED]
  .map((key) => (key === "inputs" ? "input" : key === "outputs" ? "output" : key))
  .sort(compareCodeUnits);

/**
 * Validate the top-level `system:` section's shape. Only `paths:` (a list of strings) is allowed:
 * the resolution search path, not the strict `${system.X}` ambient namespace (a
 * `${system.paths}` reference stays a compile error).
 *
 * @throws LoadError for an unknown key, at its key, or `paths:` that isn't a list of strings.
 */
function validateSystem(system: ValueRecord, cursor: SourceCursor | null = null): void {
  const extra = Object.keys(system)
    .filter((key) => key !== "paths")
    .sort(compareCodeUnits);
  if (extra.length > 0) {
    throw LoadDiagnostics.fromCursor(cursor).error(`system: unknown key(s) ${writeAsText(extra)} (allowed: paths)`, {
      range: cursor !== null ? cursor.keyRangeOf(extra[0]!) : null,
    });
  }
  const paths = Object.hasOwn(system, "paths") ? system["paths"] : [];
  if (!isList(paths) || !paths.every((path) => typeof path === "string")) {
    throw LoadDiagnostics.fromCursor(cursor).error("system: paths must be a list of strings", {
      range: cursor !== null ? cursor.rangeOf("paths") : null,
    });
  }
}

/**
 * Reject file-scoped callable declarations in the inline-flow namespace, at the offending
 * declaration's key.
 *
 * @throws LoadError when a `defs:` name or `uses:` alias starts with `INLINE_FLOW_PREFIX`.
 */
function rejectAuthoredPrivateCallableNames(root: SourceCursor, defs: ValueRecord, uses: Record<string, string>): void {
  for (const [section, declarations] of [
    ["defs", defs],
    ["uses", uses],
  ] as const) {
    const offender = Object.keys(declarations).find((name) => name.startsWith(INLINE_FLOW_PREFIX));
    if (offender === undefined) continue;
    const declarationRange = root.has(section) ? root.child(section).keyRangeOf(offender) : null;
    const noun = section === "defs" ? "callable names" : "aliases";
    throw LoadDiagnostics.fromCursor(root).error(
      `${section}: ${noun} with prefix ${reprText(INLINE_FLOW_PREFIX)} are reserved ` +
        "for compiler-generated inline child flows",
      { range: declarationRange },
    );
  }
}

/**
 * A file a load read from disk, a flow or a file one includes: its text, and the path each `uses:`
 * and `!include` path in it resolved to.
 */
interface ReadFile {
  readonly text: string;
  /**
   * Each `uses:` ref written in the file (`usesRef`) and each `!include` path as
   * `include:<path>`, mapped to the real path it resolved to.
   */
  readonly refs: Map<string, string>;
}

/** The `refs` key prefix of an `!include` path; a local `uses:` ref never has a scheme. */
const INCLUDE_REF = "include:";

/** Return a `uses:` ref as written: its path, and `@<version>` when pinned. */
function usesRef(path: string, version: string | null): string {
  return version === null ? path : `${path}@${version}`;
}

/** Cross-file cycle state, successful compiled-flow reuse and the files read. */
export class LoadCtx {
  /** The real paths (or recorded keys) of the files being loaded. */
  readonly loading = new Set<string>();
  /** Each loaded file's flow, by real path (or recorded key). */
  readonly flows = new Map<string, ComposedFlow>();
  /** Each file read from disk, by real path. */
  readonly read = new Map<string, ReadFile>();
  /** The real path every CODE reference in the import graph must resolve inside, or `null`. */
  readonly codeRoot: string | null;
  /** The tool names of each host memory type the load was given, by contract type. */
  readonly memoryTools: ReadonlyMap<string, readonly string[]>;
  /** Whether a flow input's unknown tag reads as a memory type, as for a recorded flow. */
  readonly opaqueMemoryTags: boolean;

  constructor(
    codeRoot: string | null = null,
    options: { memoryTools?: ReadonlyMap<string, readonly string[]>; opaqueMemoryTags?: boolean } = {},
  ) {
    this.codeRoot = codeRoot;
    this.memoryTools = options.memoryTools ?? new Map();
    this.opaqueMemoryTags = options.opaqueMemoryTags ?? false;
  }

  /**
   * Return the flow loaded for `key`, loading it with `load` the first time.
   *
   * @param key The file's real path (or recorded key).
   * @param load Loads the file's flow.
   * @throws LoadError when `key` is already being loaded: a reference cycle across files.
   */
  async loadOnce(key: string, load: () => Promise<ComposedFlow>): Promise<ComposedFlow> {
    const cached = this.flows.get(key);
    if (cached !== undefined) return cached;
    if (this.loading.has(key)) throw UNLOCATED_DIAGNOSTICS.error(`flow-reference cycle across files: ${key}`);
    this.loading.add(key);
    let child: ComposedFlow;
    try {
      child = await load();
    } finally {
      this.loading.delete(key);
    }
    this.flows.set(key, child);
    return child;
  }
}

/**
 * Return a fresh `LoadCtx` whose `codeRoot` is `codeRoot` made absolute and real, or `null`, and
 * that knows the host memory types `memoryTools`.
 */
async function newLoadCtx(
  codeRoot: string | null,
  memoryTools: ReadonlyMap<string, readonly string[]>,
): Promise<LoadCtx> {
  return new LoadCtx(codeRoot === null ? null : await realOr(resolvePath(codeRoot)), { memoryTools });
}

// `uses:` imports being loaded around the current one. It spans files and `loadFlow` never resets
// it, so a custom `childResolver` that calls `loadFlow` is bounded too. Each import runs in its
// own store (`run`, never `enterWith`), so concurrent loads keep separate counts.
const ACTIVE_IMPORTS = new AsyncLocalStorage<number>();

/** Return a `system.paths` entry: a relative one joined onto the file's own directory. */
function joinPath(base: string, path: string): string {
  return pathText(isAbsolute(path) ? path : `${base}/${path}`);
}

/**
 * Throw if a `uses:` ref pins a version the resolved file doesn't declare.
 *
 * @param child The resolved child flow whose declared `version` is compared.
 * @param path The `uses:` path, for the message.
 * @param version The pinned `@<version>`; `null` means unpinned (no check).
 * @throws LoadError when the pinned version differs from the child's.
 */
function checkVersion(child: ComposedFlow, path: string, version: string | null): void {
  if (version !== null && child.version !== version) {
    throw UNLOCATED_DIAGNOSTICS.error(
      `uses: ref pinned ${path}@${version} but the resolved file declares ` +
        `version: ${valueRepr(child.version)} (file-per-version: name a file that declares ${valueRepr(version)})`,
    );
  }
}

/**
 * Return the distinct reachable origins of `compiled`, the first exact origin object reached for
 * each opaque key, in structural visit order.
 *
 * @throws TypeError or RangeError for a malformed graph or origin key.
 */
function distinctOrigins(compiled: CompiledFlow): OriginRef[] {
  const byKey = new Map<string, OriginRef>();
  for (const origin of iterFlowOrigins(compiled)) {
    const canonical = new OriginRef(origin.key);
    if (!byKey.has(canonical.key)) byKey.set(canonical.key, origin);
  }
  return [...byKey.values()];
}

/**
 * Validate one custom or cached child before lexical rebasing.
 *
 * @param alias The authored `uses:` alias selecting the child.
 * @param child The value the external child resolver returned.
 * @returns The first exact reachable child origin for each distinct opaque key.
 * @throws LoadError when the value isn't a `ComposedFlow`, resolver presence doesn't match
 *   provenance, or a reachable origin lacks a concrete frame.
 */
function validateExternalChild(alias: string, child: unknown): OriginRef[] {
  const validated = child instanceof ComposedFlow ? validatedChildren.get(child) : undefined;
  if (validated !== undefined) return validated;
  if (!(child instanceof ComposedFlow)) {
    throw UNLOCATED_DIAGNOSTICS.error(
      `uses: alias ${reprText(alias)} resolver returned ${jsTypeName(child)}, expected ComposedFlow`,
    );
  }
  let origins: OriginRef[];
  try {
    origins = distinctOrigins(child.compiled);
  } catch (error) {
    throw UNLOCATED_DIAGNOSTICS.error(
      `uses: alias ${reprText(alias)} returned an invalid compiled origin graph: ${errorText(error)}`,
      { cause: error },
    );
  }

  const resolver = child.originResolver;
  if (origins.length === 0) {
    if (resolver !== null) {
      throw UNLOCATED_DIAGNOSTICS.error(
        `uses: alias ${reprText(alias)} returned a source-free child with an origin resolver`,
      );
    }
    return origins;
  }
  if (resolver === null) {
    throw UNLOCATED_DIAGNOSTICS.error(
      `uses: alias ${reprText(alias)} returned an originful child without an origin resolver`,
    );
  }
  let resolve: unknown;
  try {
    resolve = resolver.resolve;
  } catch (error) {
    throw UNLOCATED_DIAGNOSTICS.error(
      `uses: alias ${reprText(alias)} child origin resolver could not expose resolve: ${errorText(error)}`,
      { cause: error },
    );
  }
  if (typeof resolve !== "function") {
    throw UNLOCATED_DIAGNOSTICS.error(
      `uses: alias ${reprText(alias)} child origin resolver does not implement resolve`,
    );
  }

  for (const origin of origins) {
    checkFrame(
      () => resolve.call(resolver, origin),
      () => `uses: alias ${reprText(alias)} child origin resolver`,
      () => reprText(origin.key),
      UNLOCATED_DIAGNOSTICS,
    );
  }
  validatedChildren.set(child, origins);
  return origins;
}

/**
 * The children `validateExternalChild` passed, with their origins: a flow is immutable, so a child
 * imported under several aliases, or at several levels of a diamond, is walked once.
 */
const validatedChildren = new WeakMap<ComposedFlow, OriginRef[]>();

/**
 * Throw unless `resolve` returns a `DiagnosticFrame`.
 *
 * @param resolve Resolves one origin through a resolver.
 * @param subject The resolver, as messages name it; built only for a failure, since a load that
 *   passes checks every origin.
 * @param target The origin, as messages name it; built only for a failure.
 * @param diagnostics Factory for the load error.
 * @throws LoadError when `resolve` throws or returns anything but a frame.
 */
function checkFrame(
  resolve: () => unknown,
  subject: () => string,
  target: () => string,
  diagnostics: LoadDiagnostics,
): void {
  let frame: unknown;
  try {
    frame = resolve();
  } catch (error) {
    throw diagnostics.error(`${subject()} failed for ${target()}: ${errorText(error)}`, { cause: error });
  }
  if (!(frame instanceof DiagnosticFrame)) {
    const detail = frame === null ? "no frame" : jsTypeName(frame);
    throw diagnostics.error(`${subject()} returned ${detail} for ${target()}`);
  }
}

/** Each flow's node count with its static children expanded (`expandedNodes`). */
const expandedCounts = new WeakMap<Flow, number>();

/**
 * Return how many nodes `flow` expands to: its own nodes and, for each with a static child (a
 * CALL, MAP or LOOP body), that child's count, so each call of a def or `uses:` flow counts as a
 * copy and a MAP or LOOP body counts once however many items it runs. Every CASE branch counts.
 * Cached per flow, so the count is linear in the distinct flows however large it is.
 */
function expandedNodes(flow: Flow): number {
  let count = expandedCounts.get(flow);
  if (count === undefined) {
    count = 0;
    for (const node of flow.nodes.values()) {
      const child = nodeChild(node);
      count += 1 + (child === null ? 0 : expandedNodes(child));
    }
    expandedCounts.set(flow, count);
  }
  return count;
}

/**
 * Return the first node of `flow`, in node order, at which its expanded count passes `limit`,
 * looking inside a child that passes it alone; `null` when `flow` stays within it.
 */
function nodePastLimit(flow: Flow, limit: number): Node | null {
  let count = 0;
  for (const node of flow.nodes.values()) {
    const child = nodeChild(node);
    const size = child === null ? 0 : expandedNodes(child);
    if (child !== null && size > limit) return nodePastLimit(child, limit) ?? node;
    count += 1 + size;
    if (count > limit) return node;
  }
  return null;
}

/** The message for a flow that expands to more than `limit` nodes. */
function tooManyNodes(limit: number): string {
  return (
    `this flow expands to more than ${limit} nodes, counting each call of a def or uses: flow as a ` +
    `copy; the limit for a flow is ${limit} (YIAN_MAX_FLOW_NODES)`
  );
}

/**
 * Return one validated child copy scoped under its lexical alias.
 *
 * @param alias The authored `uses:` alias owning the returned view.
 * @param child The canonical `ComposedFlow` the host resolver returned.
 * @returns An independent alias-scoped child with an exact route table, or `originResolver` null
 *   when its compiled graph is source-free.
 * @throws LoadError when validating or rebasing the child fails.
 */
function rebaseExternalChild(alias: string, child: unknown): ComposedFlow {
  const origins = validateExternalChild(alias, child);
  const flow = child as ComposedFlow;
  // Before the copy, which would cost what the limit bounds.
  const limit = current().maxFlowNodes;
  if (expandedNodes(flow.compiled) > limit) {
    throw UNLOCATED_DIAGNOSTICS.error(`uses: alias ${reprText(alias)}: ${tooManyNodes(limit)}`);
  }
  let compiled: CompiledFlow;
  let table: ResolverTable;
  try {
    [compiled, table] = rebaseFlowOrigins(flow.compiled, (origin) => scopeOrigin(alias, origin), {
      resolver: flow.originResolver,
    });
  } catch (error) {
    throw UNLOCATED_DIAGNOSTICS.error(`uses: alias ${reprText(alias)} could not rebase child: ${errorText(error)}`, {
      cause: error,
    });
  }
  if (table.size !== origins.length) {
    throw UNLOCATED_DIAGNOSTICS.error(`uses: alias ${reprText(alias)} produced an incomplete rebased origin table`);
  }
  return replaceFlow(flow, { compiled, originResolver: origins.length > 0 ? table : null });
}

/**
 * Install one exact resolver over a fully assembled physical file.
 *
 * @param staged The root assembly, its graph complete and its resolver still unset.
 * @param localCatalog The frozen authority for this document's sites.
 * @param externalTables The alias-scoped child authorities, including eagerly loaded aliases that
 *   may be unreachable.
 * @param diagnostics Factory for the load errors (default: unlocated).
 * @returns A copy carrying exactly one route per reachable origin, or `originResolver` null when
 *   the root graph is source-free.
 * @throws LoadError for a malformed graph, a missing or overlapping route, or a selected authority
 *   that can't produce a `DiagnosticFrame`.
 */
export function finalizeComposedFlow(
  staged: ComposedFlow,
  localCatalog: SourceCatalog,
  externalTables: readonly ResolverTable[],
  diagnostics: LoadDiagnostics = UNLOCATED_DIAGNOSTICS,
): ComposedFlow {
  let origins: OriginRef[];
  try {
    origins = distinctOrigins(staged.compiled);
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    throw diagnostics.error(`invalid compiled origin graph: ${error.message}`, { cause: error });
  }
  if (origins.length === 0) return replaceFlow(staged, { originResolver: null });

  const entries: [OriginRef, OriginResolver, OriginRef][] = [];
  for (const origin of origins) {
    const authorities: OriginResolver[] = [];
    if (localCatalog.has(origin)) authorities.push(localCatalog);
    authorities.push(...externalTables.filter((table) => table.has(origin)));
    if (authorities.length === 0) {
      throw diagnostics.error(`reachable origin ${reprText(origin.key)} has no resolver route`);
    }
    if (authorities.length !== 1) {
      throw diagnostics.error(`reachable origin ${reprText(origin.key)} has multiple resolver routes`);
    }
    const authority = authorities[0]!;
    checkFrame(
      () => authority.resolve(origin),
      () => "resolver route",
      () => `origin ${reprText(origin.key)}`,
      diagnostics,
    );
    entries.push([origin, authority, origin]);
  }
  return replaceFlow(staged, { originResolver: ResolverTable.fromEntries(entries) });
}

/**
 * Build the default local file resolver for `uses:` references.
 *
 * The returned resolver finds a ref's `<dir>/<path>.yaml` on the search path, the first match
 * winning; `@<version>` is never part of the file name: the importer checks the resolved file's
 * declared `version:` against the pin. Children load recursively, so their own
 * `uses:` and `system.paths` re-root at the child's directory. A `hub:` or other scheme ref never
 * reaches here.
 *
 * @param searchPaths Directories to search, in order.
 * @param ctx The load context shared by the whole import graph: a cross-file cycle guard and a
 *   diamond-reuse cache, keyed by real path, and the files read.
 * @param from The importing file, which records the path each of its refs resolved to; `null`
 *   records nothing.
 * @returns The resolver; it rejects with a `LoadError` for a missing target or a cross-file
 *   reference cycle.
 */
export function makeFileResolver(
  searchPaths: readonly string[],
  ctx: LoadCtx,
  from: ReadFile | null = null,
): ChildResolver {
  return async (path: string, version: string | null = null): Promise<ComposedFlow> => {
    const name = `${path}.yaml`;
    // A folder, FIFO or device named like the flow is no flow, so the search goes on past it.
    const notFiles: string[] = [];
    for (const dir of searchPaths) {
      // Shown with the search folder as given, never the real path.
      const shown = `${dir}/${name}`;
      let real: string;
      try {
        real = await fsp.realpath(resolvePath(dir, name));
      } catch (error) {
        // A search directory or file that doesn't exist is skipped; any other failure is loud.
        if (isMissingPath(error)) continue;
        throw error;
      }
      if (!(await fsp.stat(real)).isFile()) {
        notFiles.push(shown);
        continue;
      }
      from?.refs.set(usesRef(path, version), real);
      return ctx.loadOnce(real, async () => {
        // Kept as decoded, as an `!include` of the same file keeps it, whichever reads it first.
        let file = ctx.read.get(real);
        if (file === undefined) {
          file = { text: await readChildText(real, shown), refs: new Map() };
          ctx.read.set(real, file);
        }
        const own = dirname(real);
        const text = flowText(file.text);
        const document = await parseIncluding(text, basename(real), diskIncludes(own, ctx, file), ctx.memoryTools.keys());
        return loadFlowFile(document, null, [own], ctx, own, file);
      });
    }
    let message = `flow ${reprText(path)} not found on search path ${writeAsText([...searchPaths])} (looked for ${reprText(name)})`;
    for (const shown of notFiles) message += `; ${reprText(shown)} is not a file`;
    const yml = searchPaths.some((dir) => statOrNull(resolvePath(dir, `${path}.yml`))?.isFile() === true);
    throw UNLOCATED_DIAGNOSTICS.error(message + usesRefHint(path, yml));
  };
}

/**
 * Return the hint a `uses:` ref's not-found error ends with, else `""`: a ref written with its
 * extension (`./child.yaml`) drops it, and a flow saved as `.yml` (`ymlExists`) isn't imported.
 */
export function usesRefHint(path: string, ymlExists: boolean): string {
  if (/\.ya?ml$/.test(path)) return "; write a uses: ref without its extension";
  return ymlExists ? "; uses: imports only .yaml files" : "";
}

/**
 * Read the `uses:` child file at `real` as UTF-8 text, kept as decoded.
 *
 * @throws Error `can't read <shown>: not UTF-8 text`, or `can't read <shown> (<code>)` for a
 *   system error, naming the file as `shown` rather than by its real path.
 */
async function readChildText(real: string, shown: string): Promise<string> {
  let bytes: Uint8Array;
  try {
    bytes = await fsp.readFile(real);
  } catch (error) {
    if (!isSystemError(error)) throw error;
    throw new Error(`can't read ${shown} (${error.code})`, { cause: error });
  }
  try {
    return UTF8.decode(bytes);
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    throw new Error(`can't read ${shown}: not UTF-8 text`, { cause: error });
  }
}

/** Return a flow file's text, as decoded, with its line endings as LF, as a flow reads it. */
function flowText(text: string): string {
  return text.replace(/\r\n?/g, "\n");
}

/** The options `loadFlow` takes. */
interface LoadOptions {
  childResolver?: ChildResolver | null;
  searchPaths?: readonly string[] | null;
  sourceLabel?: string;
  baseDir?: string | null;
  codeRoot?: string | null;
  env?: Env;
  memoryTypes?: readonly MemoryClass[];
}

/**
 * Load Compose YAML text into a runnable `ComposedFlow`.
 *
 * Parses, compiles and validates the flow in one pass. A `call:` resolves an inline body, an
 * in-file `defs:` callable, or a `uses:` alias (external, with the ref's `@<version>`); any other
 * name throws. A flow whose calls all reach in-file defs loads without a resolver. With neither
 * `searchPaths` nor `childResolver`, a `uses:` alias throws: external resolution is additive.
 *
 * @param text The Compose-shaped YAML source of the flow.
 * @param options.childResolver An explicit resolver for external flows (a fake, or a future
 *   marketplace). When given, it replaces the local file resolver: `searchPaths` and the flow's
 *   `system: paths:` are not searched.
 * @param options.searchPaths Directories the default local file resolver searches for `uses:`
 *   targets (`<dir>/<path>.yaml`). Pass the flow file's own directory first; the list is extended
 *   by the flow's `system: paths:`.
 * @param options.sourceLabel The name load diagnostics give this text (default `<flow>`).
 * @param options.baseDir The directory a CODE node's `./file.ts:fn` reference and an `!include`
 *   path resolve against (default: the first search path, else the current directory). A file
 *   read through the search path resolves its own against its own directory.
 * @param options.codeRoot A directory every CODE reference's file must resolve inside (a relative
 *   one against the current directory); symlinks are resolved on both sides. A reference outside
 *   it is a `LoadError` at its `code:` value. The files read through the search path are checked
 *   too, and so is each flow a `childResolver` returns, however the host loaded it. An inline
 *   `code:` body is not confined. Each file an `!include` reads must resolve inside it too, or the
 *   load fails at the `!include`. `null` or absent confines nothing. The flow's `source` keeps the root's
 *   real path, so a run records it and a resume confines the recorded files to it again.
 * @param options.env The environment this load reads the `YIAN_*` values from; default
 *   `processEnv`, which keeps the values in effect, so a `childResolver` that calls `loadFlow`
 *   without `env` loads under the caller's.
 * @param options.memoryTypes The host's own memory classes, so a flow input can name each by its
 *   contract's type (`j: !TradeJournal`) and a grant is checked against its tools. A class that
 *   implements a built-in contract needs no entry. The flow keeps the list in `memoryTypes`.
 * @returns The compiled, validated flow. Its `source` holds `text` under `sourceLabel` and every
 *   file the load read, with where each `uses:` and `!include` path resolved (see `FlowSource`);
 *   it is `null` when a `childResolver` returned a flow without a source (one built in code).
 * @throws LoadError (as a rejection) for any malformed or inconsistent flow, located at the
 *   offending line when source provenance is available: among them an `llm_config` value a child
 *   flow's agent would inherit before it is known.
 * @throws DefaultsError when a `YIAN_<NAME>` value in `env` isn't accepted.
 * @throws TypeError when `sourceLabel` isn't text, or `memoryTypes` isn't a list of memory
 *   classes with contracts a tag can name.
 */
export async function loadFlow(text: string, options: LoadOptions = {}): Promise<ComposedFlow> {
  const { env = processEnv, ...rest } = options;
  // `withEnv` reads `env` first, so a bad YIAN_<NAME> value fails before any parsing.
  return withEnv(env, () => loadText(text, rest));
}

/** Load `text` as `loadFlow` does, under the `YIAN_*` values already in effect. */
async function loadText(text: string, options: Omit<LoadOptions, "env">): Promise<ComposedFlow> {
  const { childResolver = null, searchPaths = null, sourceLabel = "<flow>", baseDir = null, codeRoot = null } = options;
  const { memoryTypes = [] } = options;
  if (typeof sourceLabel !== "string") throw new YianTypeError("source_label must be a string");
  const memoryTools = memoryTypeTools(memoryTypes);
  // Absolute, so a CODE reference doesn't depend on the cwd when the node runs.
  const ownDir = resolvePath(baseDir ?? searchPaths?.[0] ?? process.cwd());
  const ctx = await newLoadCtx(codeRoot, memoryTools);
  const rootFile: ReadFile = { text, refs: new Map() };
  const document = await parseIncluding(text, sourceLabel, diskIncludes(ownDir, ctx, rootFile), ctx.memoryTools.keys());
  const hosted = new Map<string, unknown>();
  const resolver: ChildResolver | null =
    childResolver === null
      ? null
      : async (path, version) => {
          const child = await childResolver(path, version);
          hosted.set(usesRef(path, version), child);
          return child;
        };
  const loaded = await loadFlowFile(document, resolver, searchPaths, ctx, ownDir, rootFile);
  if (ctx.codeRoot !== null) {
    await checkHostedCode(hosted, document, ctx.codeRoot);
    checkHostedIncludes(hosted, ctx.codeRoot);
  }
  const flow = replaceFlow(loaded, {
    source: await flowSource(ctx, rootFile, hosted, sourceLabel, ownDir),
    memoryTypes: Object.freeze([...memoryTypes]),
  });
  return checkRoot(flow);
}

/**
 * Check each CODE reference of every flow a host resolver returned, its own children included,
 * against `codeRoot`: the host may have loaded it under no root, and a run records the root for
 * all of the flow's files, so the loaded flow and a load from its recorded source agree.
 *
 * @throws LoadError at the first CODE node whose file resolves outside `codeRoot`, or can't be read.
 */
async function checkHostedCode(
  hosted: ReadonlyMap<string, unknown>,
  document: SourceDocument,
  codeRoot: string,
): Promise<void> {
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const seen = new Set<object>();
  for (const child of hosted.values()) {
    if (!(child instanceof ComposedFlow)) continue;
    const resolver = child.originResolver;
    const resolveOrigin = resolver === null ? null : (origin: OriginRef) => resolver.resolve(origin);
    await checkFlowCode(child.compiled, { diagnostics, resolveOrigin, seen, codeRoot });
  }
}

/**
 * Throw at the first file a host child's source includes from outside `codeRoot`: the host may
 * have loaded it under no root, and a load from the run's recorded source holds every included
 * file to the root.
 *
 * @throws LoadError naming the including file and the path as written.
 */
function checkHostedIncludes(hosted: ReadonlyMap<string, unknown>, codeRoot: string): void {
  for (const child of hosted.values()) {
    if (!(child instanceof ComposedFlow) || child.source === null) continue;
    const { dir, files } = child.source;
    for (const [key, file] of Object.entries(files)) {
      for (const [ref, target] of Object.entries(file.refs)) {
        if (!ref.startsWith(INCLUDE_REF) || !leavesRoot(relative(codeRoot, resolvePath(dir, target)))) continue;
        const file = includeFile(ref.slice(INCLUDE_REF.length));
        throw UNLOCATED_DIAGNOSTICS.error(`${basename(key)}: ${outsideRoot(file)}`);
      }
    }
  }
}

/**
 * Reads the file an `!include` path names; `fail` makes the error, at the value, for one it can't,
 * and `file` is how its messages name the file (`includeFile`).
 */
type IncludeReader = (path: string, fail: (message: string) => LoadError, file: string) => Promise<string>;

/** The message for an included file, named as `includeFile` names it, outside the code root. */
function outsideRoot(file: string): string {
  return `${file} is outside the code root; include a file inside it`;
}

/** The message for an included file, named as `includeFile` names it, over the cap `limit`. */
function tooLarge(file: string, limit: number): string {
  return `${file} is larger than ${limit} bytes (YIAN_MAX_INCLUDE_BYTES); shorten it or raise the cap`;
}

/**
 * Parse `text` as `parseSourceDocument` does, each `!include` value reading the text `read` gives
 * its path. An `!include(T)` value reads that text too when `T` is text (`readsText`, against the
 * file's typedefs and the memory types named `memoryTypes`), else the value the text holds as YAML
 * (`readIncludedValue`). Each distinct path is read once, in document order.
 *
 * @throws LoadError at the first `!include` with an empty path, a path holding a NUL, a file
 *   `read` refuses, or a file `readIncludedValue` refuses.
 */
async function parseIncluding(
  text: string,
  label: string,
  read: IncludeReader,
  memoryTypes: Iterable<string>,
): Promise<SourceDocument> {
  const document = parseSourceDocument(text, label);
  if (document.includes.length === 0) return document;
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const textual = readsText(document, memoryTypes);
  const texts = new Map<string, string>();
  const values = new Map<string, SourceDocument>();
  const textTypes = new Set<string>();
  for (const { path, range, type } of document.includes) {
    const fail = (message: string) => diagnostics.error(message, { range });
    const file = includeFile(path, type);
    if (!texts.has(path)) {
      if (path === "") throw fail(`${includeTag(type)} needs a file path, e.g. ${includeExample(type)}`);
      // No file system takes it; Node and Bun throw a TypeError naming the absolute path.
      if (path.includes("\0")) throw fail(`${file} holds a NUL character; write the path without it`);
      texts.set(path, await read(path, fail, file));
    }
    if (type === null) continue;
    if (textual(type)) textTypes.add(type);
    else if (!values.has(path)) values.set(path, readIncludedValue(texts.get(path)!, path, file, fail));
  }
  return parseSourceDocument(text, label, texts, { values, textTypes });
}

/**
 * Return whether an `!include(T)` in `document` reads its file's text: whether `T` names, against
 * the file's typedefs, a type whose every value but null is text, such as `str`, a typedef naming
 * it, `Optional(str)` or a `Literal`. A `T` that names no type, or typedefs that don't read, read
 * text too, so the load reports the type, not the file.
 */
function readsText(document: SourceDocument, memoryTypes: Iterable<string>): (type: string) => boolean {
  let registry: TypeRegistry | null = null;
  try {
    const root = document.root;
    const typedefs = root.has("typedefs") ? root.get("typedefs") : null;
    registry = readTypedefs(isRecord(typedefs) ? typedefs : null, memoryTypes);
  } catch (error) {
    if (!(error instanceof TypeCheckError || error instanceof SourceAccessError)) throw error;
  }
  return (type) => {
    if (registry === null) return true;
    try {
      return typeFor(type, registry).kind === ValueKind.STRING;
    } catch (error) {
      if (error instanceof TypeCheckError) return true;
      throw error;
    }
  };
}

/**
 * Read an included file's `text` as one YAML value (`SourceDocument.parseData`), labelled with its
 * `path` as written.
 *
 * @throws LoadError, made by `fail` and naming the file as `file` and the line and column in it,
 *   for text that isn't one YAML document, or holds a tag, an anchor, an alias, a `<<:` merge key,
 *   a key YAML reads as non-text or a repeated key.
 */
function readIncludedValue(
  text: string,
  path: string,
  file: string,
  fail: (message: string) => LoadError,
): SourceDocument {
  const at = (message: string, range: SourceRange | null): LoadError =>
    fail(`${file}${range === null ? "" : `, line ${range.startLine}, column ${range.startCol}`}: ${message}`);
  let value: SourceDocument;
  try {
    value = SourceDocument.parseData(text, path);
  } catch (error) {
    if (!(error instanceof SourceSyntaxError)) throw error;
    throw at(error.message, error.range);
  }
  const problem = keyProblem(value);
  if (problem !== null) throw at(problem.message, problem.range);
  return value;
}

/**
 * Return the reader of the files `!include` paths name, resolved against `dir`. A file must
 * resolve, symlinks followed, inside `ctx.codeRoot` when it has one, which is checked first and
 * named without an absolute path; be a regular file, read through one descriptor that doesn't
 * wait for a FIFO's writer; hold at most `YIAN_MAX_INCLUDE_BYTES`, of which at most one byte more
 * is read; and be UTF-8 text, kept as decoded. Under a code root, the path must then still resolve
 * inside it, to the file read. The file is kept in `ctx.read` by its real path, which `file.refs`
 * records under `include:<path>`.
 */
function diskIncludes(dir: string, ctx: LoadCtx, file: ReadFile): IncludeReader {
  return async (path, fail, named) => {
    const { codeRoot } = ctx;
    const target = resolvePath(dir, path);
    try {
      let real: string;
      try {
        real = await fsp.realpath(target);
      } catch (error) {
        if (codeRoot !== null && leavesRoot(relative(codeRoot, resolvedPath(target)))) throw fail(outsideRoot(named));
        if (!isMissingPath(error)) throw error;
        const where = codeRoot === null ? ` in ${reprText(dir)}` : "";
        throw fail(`${named} does not exist${where}; its path is relative to the including file's folder`);
      }
      if (codeRoot !== null && leavesRoot(relative(codeRoot, real))) throw fail(outsideRoot(named));
      const limit = current().maxIncludeBytes;
      const opened = await readRegularFile(real, limit);
      if (opened === null) throw fail(`${named} is not a regular file; include a text file`);
      if (opened.bytes.length > limit) throw fail(tooLarge(named, limit));
      if (codeRoot !== null) {
        // A folder on the path swapped for a symlink after the check opens another file, so the
        // path must still resolve inside the root, to the file that was read.
        const again = await fsp.realpath(target);
        if (leavesRoot(relative(codeRoot, again))) throw fail(outsideRoot(named));
        const now = await fsp.stat(again, { bigint: true });
        if (now.dev !== opened.dev || now.ino !== opened.ino) {
          throw fail(`${named} changed while it was read; load the flow again`);
        }
      }
      let read = ctx.read.get(real);
      if (read === undefined) {
        let text: string;
        try {
          text = UTF8.decode(opened.bytes);
        } catch (error) {
          if (!(error instanceof TypeError)) throw error;
          throw fail(`${named} is not UTF-8 text; save it as UTF-8`);
        }
        // Kept as decoded: `!include(str)` reads it exactly, and `!include` reads its line endings as LF.
        read = { text, refs: new Map() };
        ctx.read.set(real, read);
      }
      file.refs.set(INCLUDE_REF + path, real);
      return read.text;
    } catch (error) {
      // A system error's message holds absolute paths.
      if (!isSystemError(error)) throw error;
      throw fail(`${named} can't be read (${error.code}); include a text file you can read`);
    }
  };
}

/**
 * Return the reader of the files `!include` paths in `source`'s file `from` name, through its
 * `refs`, never from disk. As when it was read from disk, a file's key must be inside
 * `source.codeRoot` when it has one, and its text at most `YIAN_MAX_INCLUDE_BYTES`.
 */
function sourceIncludes(source: FlowSource, from: string): IncludeReader {
  return async (path, fail, named) => {
    const refs = source.files[from]!.refs;
    const ref = INCLUDE_REF + path;
    // A host may build a source by hand, its ref naming no file.
    if (!Object.hasOwn(refs, ref) || !Object.hasOwn(source.files, refs[ref]!)) {
      throw fail(
        `${named} is not among the recorded files; record it under the including file's ` + `\`${ref}\` ref`,
      );
    }
    const key = refs[ref]!;
    if (source.codeRoot !== null && leavesRoot(relative(source.codeRoot, resolvePath(source.dir, key)))) {
      throw fail(outsideRoot(named));
    }
    const text = source.files[key]!.text;
    const limit = current().maxIncludeBytes;
    if (new TextEncoder().encode(text).length > limit) throw fail(tooLarge(named, limit));
    return text;
  };
}

/**
 * Compile the flow files `source` holds, as a run recorded them, without reading them from disk:
 * the load starts at `source.root` and resolves each `uses:` ref and `!include` path through the
 * file's `refs`. Each file is labelled with its key's file name, and its CODE references resolve
 * on disk in its key's folder below `source.dir`, inside `source.codeRoot` when it has one (a host
 * child's recorded file included), as `loadFlow`'s `codeRoot`; each included file's key must be
 * inside it too. A flow input's tag naming a type no file declares reads as a memory type: the
 * run's start checked its memories, so the host need not pass its memory classes again, and the
 * fingerprint is the same.
 *
 * @param source A loaded flow's `source`, or a run's recorded `flow`.
 * @param options.env The environment the load reads the `YIAN_*` values from, as `loadFlow`'s.
 * @returns The flow, whose `source` is `source`.
 * @throws LoadError for a flow that doesn't load, a `uses:` ref or `!include` path its file's
 *   `refs` lacks, or a CODE reference or included file outside `source.codeRoot`.
 * @throws DefaultsError when a `YIAN_<NAME>` value in `env` isn't accepted.
 */
export async function loadFlowSource(
  source: FlowSource,
  options: { env?: Env } = {},
): Promise<ComposedFlow> {
  const { env = processEnv } = options;
  return withEnv(env, async () => {
    // Already a real path, as the load that made the source resolved it.
    const ctx = new LoadCtx(source.codeRoot, { opaqueMemoryTags: true });
    const loaded = await loadSourceFile(source, source.root, ctx);
    return checkRoot(replaceFlow(loaded, { source }));
  });
}

/** Load `source`'s file `key`, its refs resolved through `source` too. */
async function loadSourceFile(source: FlowSource, key: string, ctx: LoadCtx): Promise<ComposedFlow> {
  const read = sourceIncludes(source, key);
  const document = await parseIncluding(flowText(source.files[key]!.text), basename(key), read, ctx.memoryTools.keys());
  return loadFlowFile(document, sourceResolver(source, key, ctx), null, ctx, resolvePath(source.dir, dirname(key)));
}

/** Return a resolver for the `uses:` refs written in `source`'s file `from`, through its `refs`. */
function sourceResolver(source: FlowSource, from: string, ctx: LoadCtx): ChildResolver {
  return async (path: string, version: string | null = null): Promise<ComposedFlow> => {
    const refs = source.files[from]!.refs;
    const ref = usesRef(path, version);
    if (!Object.hasOwn(refs, ref)) {
      throw UNLOCATED_DIAGNOSTICS.error(`flow ${reprText(ref)} is not among the recorded flow files`);
    }
    const key = refs[ref]!;
    return ctx.loadOnce(key, () => loadSourceFile(source, key, ctx));
  };
}

/** Run the checks only a root load makes on `flow`, and return it. */
function checkRoot(flow: ComposedFlow): ComposedFlow {
  // Only the root load sees the fully assembled graph, imports included; children loaded for
  // uses: are checked as part of their importer.
  // Which flow is the run's root, and so which inputs are known at run start, is decided only
  // here.
  refuseInheritedLlmValues(flow);
  return flow;
}

/** A flow file in a source graph: its real path, its text, and the file each `uses:` ref in it went to. */
interface SourceNode {
  readonly path: string;
  readonly text: string;
  readonly refs: Map<string, SourceNode>;
}

/**
 * Return the source of a loaded root: `rootFile`, every file the disk resolver read, and each host
 * child's own source, each keyed by its real path relative to the real `ownDir` (see `FlowSource`).
 * The root's path is `<real ownDir>/<file name of sourceLabel>`, as is any source's root; a label
 * with no file name (`""`, `.`, `..`, `/`) gives `<flow>`.
 *
 * @returns The source, or `null` when a host child has none (it was built in code).
 */
async function flowSource(
  ctx: LoadCtx,
  rootFile: ReadFile,
  hosted: ReadonlyMap<string, unknown>,
  sourceLabel: string,
  ownDir: string,
): Promise<FlowSource | null> {
  const dir = await realOr(ownDir);
  const fromDisk = new Map<string, SourceNode>();
  const diskNode = (path: string, file: ReadFile): SourceNode => {
    let node = fromDisk.get(path);
    if (node === undefined) {
      node = { path, text: file.text, refs: new Map() };
      fromDisk.set(path, node);
      for (const [ref, real] of file.refs) node.refs.set(ref, diskNode(real, ctx.read.get(real)!));
    }
    return node;
  };
  // A label with no file name (`""`, `.`, `..`, `/`) would key the root at the folder itself.
  const name = basename(sourceLabel);
  const rootName = name === "" || name === "." || name === ".." ? "<flow>" : name;
  // Built apart from the disk files: a file on disk may share the root's path with other text.
  const root: SourceNode = { path: join(dir, rootName), text: rootFile.text, refs: new Map() };
  for (const [ref, real] of rootFile.refs) root.refs.set(ref, diskNode(real, ctx.read.get(real)!));
  for (const [ref, child] of hosted) {
    if (!(child instanceof ComposedFlow) || child.source === null) return null;
    root.refs.set(ref, sourceNodes(child.source));
  }
  return sourceOf(root, dir, ctx.codeRoot);
}

/** Return the graph of `source`'s files, from its root, each at its path below `source.dir`. */
function sourceNodes(source: FlowSource): SourceNode {
  const nodes = new Map<string, SourceNode>();
  const node = (key: string): SourceNode => {
    let found = nodes.get(key);
    if (found === undefined) {
      const file = source.files[key]!;
      found = { path: resolvePath(source.dir, key), text: file.text, refs: new Map() };
      nodes.set(key, found);
      for (const [ref, target] of Object.entries(file.refs)) found.refs.set(ref, node(target));
    }
    return found;
  };
  return node(source.root);
}

/**
 * Key each file of the graph at `root` by its path relative to `dir`, and return the source, whose
 * code root is `codeRoot`. Two different files claiming one path (texts a host gave) get keys of
 * their own, `<key>~1`, ...
 */
function sourceOf(root: SourceNode, dir: string, codeRoot: string | null): FlowSource {
  // Two nodes are one file when their paths, texts and the files their refs reach are; each
  // distinct file gets a number, so a long chain compares in linear time.
  const ids = new Map<string, number>();
  const nodeIds = new Map<SourceNode, number>();
  const signature = (node: SourceNode): number => {
    let found = nodeIds.get(node);
    if (found === undefined) {
      // The ref graph has no cycle: the load refused one.
      const refs = [...node.refs].sort(([a], [b]) => compareCodeUnits(a, b)).map(([ref, to]) => [ref, signature(to)]);
      const text = JSON.stringify([node.path, node.text, refs]);
      found = ids.get(text) ?? ids.size;
      ids.set(text, found);
      nodeIds.set(node, found);
    }
    return found;
  };
  const keys = new Map<number, string>();
  const used = new Set<string>();
  const files: [string, FlowSource["files"][string]][] = [];
  const visit = (node: SourceNode): string => {
    const sign = signature(node);
    const known = keys.get(sign);
    if (known !== undefined) return known;
    const wanted = relative(dir, node.path);
    let key = wanted;
    for (let n = 1; used.has(key); n += 1) key = `${wanted}~${n}`;
    keys.set(sign, key);
    used.add(key);
    const refs = [...node.refs]
      .sort(([a], [b]) => compareCodeUnits(a, b))
      .map(([ref, to]): [string, string] => [ref, visit(to)]);
    // `Object.fromEntries` defines own data properties, so a `__proto__` key stays a key.
    files.push([key, Object.freeze({ text: node.text, refs: Object.freeze(Object.fromEntries(refs)) })]);
    return key;
  };
  const rootKey = visit(root);
  return Object.freeze({ root: rootKey, dir, codeRoot, files: Object.freeze(Object.fromEntries(files)) });
}

/** Return `path`'s real path, or `path` itself when it doesn't exist. */
async function realOr(path: string): Promise<string> {
  try {
    return await fsp.realpath(path);
  } catch (error) {
    if (isMissingPath(error)) return path;
    throw error;
  }
}

/** Say whether two frames sit in one file: the same label and the same text, since labels may repeat. */
function sameFile(a: DiagnosticFrame, b: DiagnosticFrame): boolean {
  return a.label === b.label && a.text === b.text;
}

/** Return `origin`'s frame through `resolver`, or `null` when either is missing. */
function frameOf(resolver: OriginResolver | null, origin: OriginRef | null): DiagnosticFrame | null {
  return origin === null || resolver === null ? null : resolver.resolve(origin);
}

/**
 * Refuse an `llm_config` value a child flow's agent would inherit before it is known.
 *
 * Only the root flow's values that read just its inputs are known when the run starts; any other
 * value in a flow's layer is read when that flow runs, so a child flow's agent can't inherit it
 * (`findInheritedLlmValue`). The error is located at the value and names the agent and the fix; a
 * value in an imported file names the `uses:` alias and the node that calls it. A value or
 * callsite with no source location names the callee as the callsite does, without saying whether
 * it is an import or a `defs:` entry.
 *
 * @param flow The loaded root flow, imports included.
 * @throws LoadError when a child flow's agent would inherit such a value.
 */
function refuseInheritedLlmValues(flow: ComposedFlow): void {
  const found = findInheritedLlmValue(flow.compiled);
  if (found === null) return;
  const frame = (origin: OriginRef | null) => frameOf(flow.originResolver, origin);
  const valueFrame = frame(found.origin);
  const field = fieldLabel(found.field);
  let subject: string;
  if (found.callsite === null) {
    subject = `flow ${field}`;
  } else {
    const name = reprText(found.callsite.flowId);
    const called = `called by node ${reprText(found.owner.join("/"))}`;
    const callsiteFrame = frame(found.callsite.origin);
    if (valueFrame === null || callsiteFrame === null) {
      subject = `${field} of the flow ${name} (${called})`;
    } else if (!sameFile(valueFrame, callsiteFrame)) {
      subject = `${field} of the flow imported through uses: ${name} (${called})`;
    } else {
      subject = `defs entry ${name} ${field} (${called})`;
    }
  }
  const agent = found.gate ?? `agent ${reprText(found.agent.join("/"))} in a child flow`;
  throw UNLOCATED_DIAGNOSTICS.error(
    `${subject}: ${agent} would inherit this value, which is read only when the flow ` +
      `runs; ${inheritedValueFix(found.field)}`,
    { frames: valueFrame === null ? [] : [valueFrame] },
  );
}

/**
 * @internal Read a flow- or def-level `llm_config` with its values and tags, and check it.
 *
 * The layer is read through its tags and checked the way an AGENT node's own is
 * (`validateLlmConfig`): a layer holding `${...}` values becomes a `ValueTemplate`, or a `Binding`
 * for a whole record written as one value; otherwise it must be a mapping of `LLMConfig` fields
 * without the fields only the host sets. `inherit` is an AGENT-only key.
 *
 * @param llmConfig The authored layer; `null` means absent.
 * @param where The layer's owner in messages, such as `flow`.
 * @param options.registry The file's typedefs the tags resolve against.
 * @param options.sitePath Catalog site prefix for the layer.
 * @param options.catalogBuilder File-owned origin catalog builder.
 * @param options.diagnostics Factory for located load errors.
 * @param options.cursor The authored flow or def body, when retained.
 * @returns The layer: a record, a `ValueTemplate` or a `Binding`; `{}` when absent.
 * @throws LoadError when the layer is malformed or sets a field a flow file may not set.
 */
export function readAuthoredLlmConfig(
  llmConfig: unknown,
  where: string,
  options: {
    registry: TypeRegistry;
    sitePath: readonly (string | number)[];
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
    cursor: SourceCursor | null;
  },
): Readonly<Record<string, unknown>> | ValueTemplate | Binding {
  const { registry, sitePath, catalogBuilder, diagnostics, cursor } = options;
  // `llm_config: !T` with nothing after the tag reads as null; the reader rejects it.
  const tagged = cursor !== null && cursor.has("llm_config") && cursor.child("llm_config").tag !== null;
  if (llmConfig === null && !tagged) return {};
  let sourceRange: SourceRange | null = null;
  let ranges = new Map<string, SourceRange>();
  if (cursor !== null && cursor.has("llm_config")) {
    sourceRange = cursor.rangeOf("llm_config");
    const layerCursor = cursor.child("llm_config");
    if (layerCursor.holds(llmConfig as Value)) {
      try {
        [llmConfig, ranges] = readLlmConfig(layerCursor, registry, {
          prefix: `${where}: llm_config`,
          sitePath,
          catalogBuilder,
          diagnostics,
        });
      } catch (error) {
        if (!(error instanceof SourceAccessError)) throw error;
        throw diagnostics.error(error.message, { range: error.range, cause: error });
      }
    }
  }
  return validateLlmConfig(llmConfig, `${where}: llm_config`, sourceRange, diagnostics, ranges, {
    allowInherit: false,
  })[0];
}

/**
 * Refuse a flow or `defs:` entry `env:` value too deep or holding itself, or an invalid
 * `max_tool_iterations`.
 *
 * The nesting and recursive alias checks run first, so a deep `max_tool_iterations` isn't shown in
 * full; a `max_tool_iterations` default must be one every AGENT node would accept when it runs.
 *
 * @param env The `env:` mapping.
 * @param prefix The message prefix: `flow ` or `defs entry '<name>': `.
 * @param envCursor The authored `env:` mapping, or `null` when there is none.
 * @param diagnostics The file's diagnostics.
 * @throws LoadError at the offending entry.
 */
function validateEnv(
  env: ValueRecord,
  prefix: string,
  envCursor: SourceCursor | null,
  diagnostics: LoadDiagnostics,
): void {
  checkEnvValues(env, prefix, envCursor, diagnostics);
  const maxToolIterations = maxToolIterationsError(env);
  if (maxToolIterations !== null) {
    throw diagnostics.error(prefix + maxToolIterations, {
      range: envCursor?.rangeOf("max_tool_iterations") ?? null,
    });
  }
}

/**
 * Return the source range of the `typedefs:` entry or field a typedef error names.
 *
 * @param root The flow document's root cursor.
 * @param path The error's `[typedef]` or `[typedef, field]` path; empty for a block error.
 * @returns The field's value range, else the typedef's key range, else the `typedefs:` key range;
 *   `null` when the flow has no `typedefs:` section.
 */
function typedefErrorRange(root: SourceCursor, path: ValuePath): SourceRange | null {
  if (!root.has("typedefs")) return null;
  const typedefs = root.child("typedefs");
  const [name, field] = path;
  if (typeof name !== "string" || !typedefs.has(name)) return root.keyRangeOf("typedefs");
  const entry = typedefs.child(name);
  if (typeof field === "string" && entry.has(field)) return entry.rangeOf(field);
  return typedefs.keyRangeOf(name);
}

/**
 * Reject text `!include` read as a typedef or a typedef's field: each is a type.
 *
 * @throws LoadError at the first such value.
 */
function rejectIncludedTypedefs(root: SourceCursor, diagnostics: LoadDiagnostics): void {
  try {
    const typedefs = sourceChild(root, "typedefs");
    if (typedefs === null || !isRecord(typedefs._valueView())) return;
    for (const typedef of typedefs.children()) {
      const fields = isRecord(typedef._valueView()) ? typedef.children() : [];
      const found = [typedef, ...fields].find((cursor) => cursor.included);
      if (found === undefined) continue;
      const name = reprText(typedef.keyText!);
      throw diagnostics.error(
        `typedef ${name}: a type definition is written as a type, which \`!include\` can't give`,
        { range: found.range },
      );
    }
  } catch (error) {
    // A typedef at a recursive alias can't be walked; reading the typedefs reports it.
    if (!(error instanceof SourceAccessError)) throw error;
  }
}

/**
 * Load one parsed flow file: its checks, its `uses:` imports, its `defs:`, then its assembly.
 *
 * Every step before the imports is synchronous; each alias then resolves, in source order, and
 * the assembly runs inside one synchronous call nesting scope, so the file counts its own calls.
 * The CODE nodes are checked after assembly: each def's in written order, then the root's.
 *
 * @param document The file's parsed document.
 * @param childResolver The explicit external resolver, or `null` for the local file resolver.
 * @param searchPaths The file's own directory first, then other search directories; `null` makes
 *   `system.paths` inert.
 * @param ctx The load context shared by the whole import graph.
 * @param baseDir The directory the file's CODE references resolve against.
 * @param file The file as read, which records where the local file resolver sent each of its
 *   `uses:` paths; `null` records nothing.
 *
 * Exported for tests; hosts call `loadFlow`.
 */
export async function loadFlowFile(
  document: SourceDocument,
  childResolver: ChildResolver | null,
  searchPaths: readonly string[] | null,
  ctx: LoadCtx,
  baseDir: string,
  file: ReadFile | null = null,
): Promise<ComposedFlow> {
  const root = document.root;
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const f = parseFile(document);
  rejectAuthoredPrivateCallableNames(root, f.defs, f.uses);
  validateEnv(f.env, "flow ", root.has("env") ? root.child("env") : null, diagnostics);
  validateSystem(f.system, root.has("system") ? root.child("system") : null);

  // Typedefs are read as plain values, so a tag or included text in them is rejected before they are.
  rejectUnreadTags(document, new Set(), diagnostics, { typedefsOnly: true });
  rejectIncludedTypedefs(root, diagnostics);
  let registry: TypeRegistry;
  try {
    registry = readTypedefs(f.typedefs, ctx.memoryTools.keys());
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw diagnostics.error(`bad typedefs: ${error.message}`, {
      range: typedefErrorRange(root, error.path),
      cause: error,
    });
  }
  // A host memory type the file also declares as a typedef keeps the typedef's meaning, and a
  // flow input naming it is refused (`inputTagRegistry`); an unused one loads, since a host passes
  // its memory types to every flow it loads.
  const clashes = new Set([...ctx.memoryTools.keys()].filter((name) => registry.has(name)));
  registry = withMemoryTypes(registry, ctx.memoryTools.keys());
  const memoryTypes: MemoryTypeTable = { tools: ctx.memoryTools, opaque: ctx.opaqueMemoryTags, clashes };

  // `system.paths` is honored only when the file's own directory is known: it was loaded with
  // search paths, whose first entry is that directory. Without them `system.paths` is inert, so
  // no resolver exists and a `uses:` alias is loud.
  let effective: string[] = [];
  if (searchPaths !== null && searchPaths.length > 0) {
    const own = searchPaths[0]!;
    const paths = (f.system["paths"] ?? []) as string[];
    effective = [...searchPaths.map(pathText), ...paths.map((path) => joinPath(own, path))];
  }

  const external = childResolver ?? (effective.length > 0 ? makeFileResolver(effective, ctx, file) : null);

  const scope = new CallableScope(f.defs, registry, external, f.uses, {
    defsCursor: root.has("defs") ? root.child("defs") : null,
    usesCursor: root.has("uses") ? root.child("uses") : null,
    memoryTypes,
    diagnostics,
    baseDir,
  });
  // Read after the typedefs, which its tags resolve against.
  const flowLlmConfig = readAuthoredLlmConfig(f.llmConfig, "flow", {
    registry,
    sitePath: ["root", "llm_config"],
    catalogBuilder: scope.catalogBuilder,
    diagnostics,
    cursor: root,
  });
  // A callable named like a node reads as that node in `call:`, so neither section may take a node
  // id. A collision is reported before a colliding callable's broken body or target.
  for (const [section, noun, names] of [
    ["uses", "alias", Object.keys(f.uses)],
    ["defs", "name", Object.keys(f.defs)],
  ] as const) {
    const collide = names.filter((name) => Object.hasOwn(f.nodes, name)).sort(compareCodeUnits);
    if (collide.length === 0) continue;
    throw diagnostics.error(`${section}: ${noun} ${writeAsText(collide)} collides with a node id`, {
      range: root.has(section) ? root.child(section).keyRangeOf(collide[0]!) : null,
    });
  }
  // Imports load before anything assembles, so assembly reads every alias from the memo.
  for (const alias of Object.keys(f.uses)) await scope.resolveAlias(alias);

  const staged = callNestingScope(() => {
    scope.eagerLoadDefs();
    return assemble(
      rootAssembleOptions(document, f, {
        registry,
        scope,
        usesAliases: Object.keys(f.uses),
        flowLlmConfig,
        baseDir,
      }),
    );
  });
  const code: CodeCheckOptions = {
    diagnostics,
    resolveOrigin: (origin) => scope.catalogBuilder.resolveRegistered(origin),
    // Shared so the root skips the defs its calls bake in, already checked. A `uses:` child was
    // checked in its own load, so its flow starts out seen.
    seen: new Set(Object.keys(f.uses).map((alias) => scope.resolve(alias).compiled)),
    codeRoot: ctx.codeRoot,
  };
  // Before the walks that cost per expanded node: finalizing the origins, the CODE check, and
  // the root's memory layout.
  const limit = current().maxFlowNodes;
  const past = nodePastLimit(staged.compiled, limit);
  if (past !== null) {
    const frame = past.origin === null ? null : scope.catalogBuilder.resolveRegistered(past.origin);
    throw diagnostics.error(`node ${reprText(past.id)}: ${tooManyNodes(limit)}`, {
      frames: frame === null ? [] : [frame],
    });
  }
  await scope.checkDefsCode(code);
  await checkFlowCode(staged.compiled, code);
  // Defs, inline bodies and the root are assembled now, so every slot has read its tags; a
  // `uses:` child checks its own document in its own load.
  rejectUnreadTags(document, scope.catalogBuilder.consumedTags, diagnostics, { registry });
  return replaceFlow(scope.finalize(staged), { importDepth: scope.importDepth });
}

/**
 * @internal Return the `assemble` options for `document`'s root flow: the sections of `f`, its
 * parsed file, their cursors and the flow's metadata, plus `rest`. Only a full flow (one with
 * `nodes:`) locates its `asserts:` and `output:`: a compact flow lends its
 * top-level `output:` to its one synthesized node.
 */
export function rootAssembleOptions(
  document: SourceDocument,
  f: ReturnType<typeof parseFile>,
  rest: Pick<AssembleOptions, "registry" | "scope" | "usesAliases" | "flowLlmConfig" | "baseDir">,
): AssembleOptions {
  const root = document.root;
  const fullFlowChild = (key: string) => (root.has("nodes") && root.has(key) ? root.child(key) : null);
  return {
    ...rest,
    inputsSection: f.inputs,
    inputsCursor: sourceChild(root, "input"),
    nodesCursor: flowNodesCursor(document, f.nodes),
    outputsSection: f.outputs,
    assertsSection: f.asserts,
    owner: ["root"],
    version: f.version,
    name: f.name,
    description: f.description,
    flowEnv: f.env,
    boundaryFieldRanges: boundaryFieldRanges(root),
    assertsCursor: fullFlowChild("asserts"),
    outputsCursor: fullFlowChild("output"),
    flowId: f.id,
  };
}

/**
 * One compiler-owned callable declaration awaiting lazy construction; frozen.
 *
 * `owner` is the complete lexical path that determines generated identity, `label` its
 * author-facing name, and `builder` constructs the child flow when resolution first reaches it.
 */
class PrivateCallable {
  readonly owner: readonly string[];
  readonly label: string;
  readonly builder: () => ComposedFlow;

  /** Record one declaration. */
  constructor(owner: readonly string[], label: string, builder: () => ComposedFlow) {
    this.owner = Object.freeze([...owner]);
    this.label = label;
    this.builder = builder;
    Object.freeze(this);
  }
}

/**
 * The file-local namespace and cache for callable resolution.
 *
 * Compiler-owned callables resolve first, then authored `defs:` and `uses:`. A bare name in
 * neither authored mapping is rejected instead of reaching the external resolver. Private
 * declarations, local defs and external aliases share one memo table. The in-progress set makes
 * recursive private and def construction loud; cross-file recursion is outside this scope and
 * belongs to the selected external resolver. A failed callable construction is never retried: its
 * error aborts the load, so what it added to the memo and the source catalog is never read.
 *
 * Aliases resolve asynchronously, through `resolveAlias`, before assembly; the synchronous
 * `resolve` assembly calls only reads them from the memo.
 */
export class CallableScope implements AssemblyScope {
  readonly #defsSection: ValueRecord;
  readonly #registry: TypeRegistry;
  readonly #external: ChildResolver | null;
  readonly #uses: Readonly<Record<string, string>>;
  readonly #defsCursor: SourceCursor | null;
  readonly #usesCursor: SourceCursor | null;
  readonly #diagnostics: LoadDiagnostics;
  readonly #catalogBuilder: SourceCatalogBuilder;
  readonly #baseDir: string | null;
  readonly #memoryTypes: MemoryTypeTable;
  readonly #private = new Map<string, PrivateCallable>();
  readonly #built = new Map<string, ComposedFlow>();
  readonly #inProgress = new Set<string>();
  #importDepth = 0;

  /**
   * Create the scope for one file.
   *
   * @param defsSection In-file callable name to raw `defs:` body, in authored order.
   * @param registry The file's typedefs, shared by every local def.
   * @param external The resolver for refs declared through `uses:`; `null` disables them.
   * @param uses Callable alias to authored external reference.
   * @param options.defsCursor Cursor over the authored `defs:` mapping, or `null`.
   * @param options.usesCursor Cursor over the authored `uses:` mapping, which locates alias
   *   failures, or `null`.
   * @param options.catalogBuilder The document-local registry shared by the root and inline
   *   bodies (default: a new one).
   * @param options.diagnostics The document's load-diagnostic authority (default: from the first
   *   retained declaration cursor, or source-free).
   * @param options.baseDir The directory the file's CODE references resolve against (default
   *   `null`, the current directory).
   * @param options.memoryTypes How the file's flow inputs read memory types beyond the
   *   built-ins (default: none).
   * @throws LoadError when a name is declared in both `defs:` and `uses:`.
   */
  constructor(
    defsSection: ValueRecord,
    registry: TypeRegistry,
    external: ChildResolver | null,
    uses: Readonly<Record<string, string>>,
    options: {
      defsCursor?: SourceCursor | null;
      usesCursor?: SourceCursor | null;
      catalogBuilder?: SourceCatalogBuilder | null;
      diagnostics?: LoadDiagnostics | null;
      baseDir?: string | null;
      memoryTypes?: MemoryTypeTable;
    } = {},
  ) {
    const { defsCursor = null, usesCursor = null, catalogBuilder = null, diagnostics = null, baseDir = null } = options;
    this.#memoryTypes = options.memoryTypes ?? NO_MEMORY_TYPES;
    const ownedDiagnostics = diagnostics ?? LoadDiagnostics.fromCursor(defsCursor ?? usesCursor);
    const both = Object.keys(defsSection)
      .filter((name) => Object.hasOwn(uses, name))
      .sort(compareCodeUnits);
    if (both.length > 0) {
      const collision = sourceChild(defsCursor, both[0]!);
      throw ownedDiagnostics.error(
        `name(s) ${writeAsText(both)} declared in both defs: and uses: (a callable has one namespace)`,
        { range: collision !== null ? collision.range : null },
      );
    }
    this.#defsSection = defsSection;
    this.#registry = registry;
    this.#external = external;
    this.#uses = uses;
    this.#defsCursor = defsCursor;
    this.#usesCursor = usesCursor;
    this.#diagnostics = ownedDiagnostics;
    this.#catalogBuilder = catalogBuilder ?? new SourceCatalogBuilder();
    this.#baseDir = baseDir;
  }

  /** The longest chain of `uses:` imports among the aliases resolved so far. */
  get importDepth(): number {
    return this.#importDepth;
  }

  /** This physical file's mutable origin registry. */
  get catalogBuilder(): SourceCatalogBuilder {
    return this.#catalogBuilder;
  }

  /** This physical file's load-diagnostic authority. */
  get diagnostics(): LoadDiagnostics {
    return this.#diagnostics;
  }

  /** The directory this file's CODE references resolve against; `null` is the current directory. */
  get baseDir(): string | null {
    return this.#baseDir;
  }

  /** How this file's flow inputs read memory types beyond the built-ins. */
  get memoryTypes(): MemoryTypeTable {
    return this.#memoryTypes;
  }

  /** Return the range of alias `alias`'s ref, or `null`. */
  #aliasRange(alias: string): SourceRange | null {
    return sourceChild(this.#usesCursor, alias)?.range ?? null;
  }

  /**
   * Install the exact resolver for one completed physical-file root.
   *
   * @param staged The root graph assembled with its resolver unset.
   * @returns The final public and cached value, with exact reachable routes.
   * @throws LoadError when an alias is missing its rebased table, or the final reachable-origin
   *   ownership is incomplete or ambiguous.
   */
  finalize(staged: ComposedFlow): ComposedFlow {
    const tables: ResolverTable[] = [];
    for (const alias of Object.keys(this.#uses)) {
      const child = this.#built.get(alias);
      if (child === undefined) {
        throw this.#diagnostics.error(`uses: alias ${reprText(alias)} was not resolved before finalization`, {
          range: this.#aliasRange(alias),
        });
      }
      const resolver = child.originResolver;
      if (resolver === null) continue;
      if (!(resolver instanceof ResolverTable)) {
        throw this.#diagnostics.error(`uses: alias ${reprText(alias)} did not retain a rebased resolver table`, {
          range: this.#aliasRange(alias),
        });
      }
      tables.push(resolver);
    }
    return finalizeComposedFlow(staged, this.#catalogBuilder.freeze(), tables, this.#diagnostics);
  }

  /** Build callable `flowId`, marked in progress while it builds, and cache it. */
  #buildOnce(flowId: string, builder: () => ComposedFlow): ComposedFlow {
    this.#inProgress.add(flowId);
    let loaded: ComposedFlow;
    try {
      loaded = builder();
    } finally {
      this.#inProgress.delete(flowId);
    }
    this.#built.set(flowId, loaded);
    return loaded;
  }

  /**
   * Register one lazy compiler-owned callable in this file scope.
   *
   * Registering the same generated identity again for the same complete lexical owner is
   * idempotent: the first label and builder stay. The messages name only the friendly label, the
   * owner and the reserved prefix, never the opaque generated identifier.
   *
   * @throws LoadError when `flowId` lacks the reserved prefix, or collides with another owner's
   *   callable or an authored name.
   */
  registerPrivate(flowId: string, owner: readonly string[], label: string, builder: () => ComposedFlow): void {
    if (!flowId.startsWith(INLINE_FLOW_PREFIX)) {
      throw this.#diagnostics.error(
        `compiler-generated callable ${reprText(label)} owned by ${writeAsText([...owner])} must use ` +
          `the reserved prefix ${reprText(INLINE_FLOW_PREFIX)}`,
      );
    }

    const existing = this.#private.get(flowId);
    if (existing !== undefined) {
      if (existing.owner.length === owner.length && existing.owner.every((step, i) => step === owner[i])) return;
      throw this.#diagnostics.error(
        "compiler-generated callable collision in reserved prefix " +
          `${reprText(INLINE_FLOW_PREFIX)}: ${reprText(existing.label)} owned by ` +
          `${writeAsText([...existing.owner])} conflicts with ${reprText(label)} owned by ${writeAsText([...owner])}`,
      );
    }

    const authoredSection = Object.hasOwn(this.#defsSection, flowId)
      ? "defs"
      : Object.hasOwn(this.#uses, flowId)
        ? "uses"
        : null;
    if (authoredSection !== null) {
      throw this.#diagnostics.error(
        "compiler-generated callable collision in reserved prefix " +
          `${reprText(INLINE_FLOW_PREFIX)}: ${reprText(label)} owned by ${writeAsText([...owner])} conflicts ` +
          `with an authored ${authoredSection}: name`,
      );
    }

    this.#private.set(flowId, new PrivateCallable(owner, label, builder));
  }

  /** Whether `flowId` names a `defs:` callable or a `uses:` alias, which `call(...)` can run. */
  has(flowId: string): boolean {
    return Object.hasOwn(this.#defsSection, flowId) || Object.hasOwn(this.#uses, flowId);
  }

  /**
   * Resolve one callable once, synchronously, for assembly.
   *
   * A `uses:` alias is read from the memo `resolveAlias` filled; this never loads a file.
   *
   * @param flowId The callable name: a compiler-generated callable, a `defs:` entry or a `uses:`
   *   alias.
   * @returns The built callable, cached for later calls.
   * @throws LoadError when the callable is unknown, recursive, fails to build, or is an alias that
   *   wasn't resolved first.
   */
  resolve(flowId: string): ComposedFlow {
    const built = this.#built.get(flowId);
    if (built !== undefined) return built;
    const privateCallable = this.#private.get(flowId);
    if (privateCallable !== undefined) {
      if (this.#inProgress.has(flowId)) {
        throw this.#diagnostics.error(
          "recursive compiler-generated callable in reserved prefix " +
            `${reprText(INLINE_FLOW_PREFIX)}: ${reprText(privateCallable.label)} owned by ` +
            `${writeAsText([...privateCallable.owner])} is defined in terms of itself ` +
            "(the baked call graph must be finite)",
        );
      }
      return this.#buildOnce(flowId, privateCallable.builder);
    }
    if (Object.hasOwn(this.#defsSection, flowId)) {
      const defCursor = sourceChild(this.#defsCursor, flowId);
      if (this.#inProgress.has(flowId)) {
        throw this.#diagnostics.error(
          `recursive defs reference: callable ${reprText(flowId)} is defined in terms ` +
            "of itself (the baked call graph must be finite)",
          { range: defCursor?.range ?? null },
        );
      }
      return this.#buildOnce(flowId, () =>
        loadDef(flowId, this.#defsSection[flowId]!, this.#registry, this, defCursor),
      );
    }
    if (Object.hasOwn(this.#uses, flowId)) {
      throw this.#diagnostics.error(`uses: alias ${reprText(flowId)} was not resolved before assembly`, {
        range: this.#aliasRange(flowId),
      });
    }
    if (flowId.startsWith(INLINE_FLOW_PREFIX)) {
      throw this.#diagnostics.error(
        "compiler-generated callable with reserved prefix " +
          `${reprText(INLINE_FLOW_PREFIX)} was not registered in this file scope`,
      );
    }
    throw this.#diagnostics.error(
      `callable ${reprText(flowId)} is neither an in-file defs: callable nor a uses: ` +
        "alias — external flows are reachable only through a uses: alias " +
        `(declare \`uses: {${flowId}: <ref>}\`)`,
    );
  }

  /**
   * Resolve one `uses:` alias once through the external resolver.
   *
   * An alias whose import would make the chain of `uses:` files deeper than
   * `current().maxCallNesting` is refused: before loading when the files already being imported
   * reach the limit, and after loading by the `importDepth` the resolved flow carries, so a cached
   * or memoized file is held to the same rule.
   *
   * @param alias A declared `uses:` alias.
   * @returns The alias's rebased child, cached for later calls.
   * @throws LoadError (as a rejection), located at the alias, when its ref is malformed or has a
   *   scheme, no resolver was given, the resolver fails or returns a bad flow, the flow declares
   *   another `version` than the ref pins, or the import chain is past the limit.
   */
  async resolveAlias(alias: string): Promise<ComposedFlow> {
    const built = this.#built.get(alias);
    if (built !== undefined) return built;
    if (!Object.hasOwn(this.#uses, alias)) throw new Error(`${reprText(alias)} is not a uses: alias`);
    const authored = this.#uses[alias]!;
    const where = `callable ${reprText(alias)} (uses: ${reprText(authored)})`;
    let loaded: ComposedFlow;
    try {
      const ref = parseUsesRef(authored, { diagnostics: this.#diagnostics });
      if (ref.scheme === "hub") {
        throw this.#diagnostics.error(`${where}: marketplace not supported yet (hub: scheme)`);
      }
      if (ref.scheme !== null) {
        throw this.#diagnostics.error(
          `${where}: unknown scheme ${reprText(ref.scheme)}: (recognized: a local ref, or hub:)`,
        );
      }
      const external = this.#external;
      if (external === null) {
        throw this.#diagnostics.error(
          `${where} needs a search path or child_resolver to resolve (load with search_paths=[...])`,
        );
      }
      const limit = current().maxCallNesting;
      const active = ACTIVE_IMPORTS.getStore() ?? 0;
      // Checked before loading: loading the file would recurse another level.
      if (active >= limit) throw this.#importTooDeep(alias, active + 1, limit);
      let child: unknown;
      try {
        child = await ACTIVE_IMPORTS.run(active + 1, () => external(ref.path, ref.version));
      } catch (error) {
        if (error instanceof LoadError) throw error;
        // A resolver's own failure is a load error.
        throw this.#diagnostics.error(`${where}: could not resolve: ${errorText(error)}`, { cause: error });
      }
      loaded = rebaseExternalChild(alias, child);
      // Checked here for every resolver, a host's too, so a load and a resume apply one rule.
      checkVersion(loaded, ref.path, ref.version);
      const depth: unknown = loaded.importDepth;
      if (typeof depth !== "number" || !Number.isInteger(depth) || depth < 0) {
        throw this.#diagnostics.error(
          `${where}: resolved flow has import_depth ${valueRepr(depth)}; expected a non-negative int`,
        );
      }
      // A file loaded earlier (cached or memoized) never ran on this count.
      if (active + 1 + depth > limit) throw this.#importTooDeep(alias, active + 1 + depth, limit);
      this.#importDepth = Math.max(this.#importDepth, 1 + depth);
    } catch (error) {
      if (!(error instanceof LoadError)) throw error;
      throw this.#diagnostics.prepend(error, {
        range: this.#aliasRange(alias),
        note: `uses alias ${reprText(alias)}`,
        cause: error,
      });
    }
    this.#built.set(alias, loaded);
    return loaded;
  }

  /** Return the error for alias `alias`, whose import makes the chain `levels` deep. */
  #importTooDeep(alias: string, levels: number, limit: number): LoadError {
    return this.#diagnostics.error(
      `callable ${reprText(alias)} (uses: ${reprText(this.#uses[alias]!)}): importing it makes a chain ` +
        `of uses: files ${levels} deep, past the call nesting limit of ${limit} (a ` +
        "chain of uses: imports is held to the same limit as nested call, map, and " +
        "loop bodies); flatten the chain",
    );
  }

  /** Build every local def in authored order before the parent flow. */
  eagerLoadDefs(): void {
    for (const name of Object.keys(this.#defsSection)) this.resolve(name);
  }

  /**
   * Check the CODE nodes of every def, in written order, each in its own frame.
   *
   * A def no node calls isn't reachable from the root graph, so each def is checked here. A def
   * calling another checks the callee inside its own frame, as building it does, so a callee's
   * failure carries both prefixes and both frames, outer first. A def is checked once. A flow
   * already in `options.seen` is skipped; the loader seeds it with each `uses:` child, which was
   * checked in its own load.
   *
   * @param options The CODE check options; their `seen` set records each checked flow.
   * @throws LoadError (as a rejection), prefixed `defs entry '<name>': ` and located at the def,
   *   at the first CODE node whose check fails.
   */
  async checkDefsCode(options: CodeCheckOptions): Promise<void> {
    // A callsite's child flow is the built def's own `compiled`, so identity names the def.
    const defNames = new Map<object, string>();
    for (const [name, flow] of this.#built) {
      if (Object.hasOwn(this.#defsSection, name)) defNames.set(flow.compiled, name);
    }
    const defOptions: CodeCheckOptions = {
      ...options,
      seen: options.seen ?? new Set<object>(),
      child: (flow, visit) => {
        const name = defNames.get(flow);
        return name === undefined ? visit(flow) : checkDef(name);
      },
    };
    const checkDef = async (name: string): Promise<void> => {
      try {
        await checkFlowCode(this.#built.get(name)!.compiled, defOptions);
      } catch (error) {
        if (!(error instanceof LoadError)) throw error;
        throw this.#diagnostics.prepend(error, {
          range: sourceChild(this.#defsCursor, name)?.range ?? null,
          message: `defs entry ${reprText(name)}: ${error.message}`,
          note: `definition ${reprText(name)}`,
          cause: error,
        });
      }
    };
    for (const name of Object.keys(this.#defsSection)) await checkDef(name);
  }
}

/**
 * Reject a multi-node `defs:` entry's `asserts:` that isn't a list of strings.
 *
 * A def body isn't checked by the `ComposeFile` schema, so its `asserts:` is checked here, in the
 * same words as a node's.
 *
 * @param name The `defs:` entry name, for the message.
 * @param asserts The authored `asserts:` value; `null` means absent.
 * @param defCursor The authored def body, when retained.
 * @param diagnostics Factory for located load errors.
 * @throws LoadError at the field when `asserts:` is set but isn't a list, and at the item for the
 *   first item YAML read as non-text.
 */
function checkDefAsserts(
  name: string,
  asserts: Value,
  defCursor: SourceCursor | null,
  diagnostics: LoadDiagnostics,
): void {
  if (asserts === null) return;
  const assertsCursor = sourceChild(defCursor, "asserts");
  const fieldRange = assertsCursor?.range ?? null;
  if (!isList(asserts)) {
    throw diagnostics.error(`defs entry ${reprText(name)}: \`asserts:\` must be a list of strings`, {
      range: fieldRange,
    });
  }
  for (const [i, item] of asserts.entries()) {
    if (typeof item === "string") continue;
    let itemRange: SourceRange | null = null;
    // `sourceChild` selects mapping keys only, so a sequence item is read through `child`.
    if (assertsCursor !== null) {
      try {
        itemRange = assertsCursor.child(i).range;
      } catch (error) {
        if (!(error instanceof SourceAccessError)) throw error;
      }
    }
    throw diagnostics.error(`defs entry ${reprText(name)}: ` + notTextError(`asserts[${i}]`, item), {
      range: itemRange ?? fieldRange,
    });
  }
}

/**
 * Build one in-file `defs:` entry into a `ComposedFlow`, an inline child flow.
 *
 * Takes a multi-node child flow (`{input?, nodes, output?, asserts?}`) or a flat single-node body
 * (`{kind, ...}`). Both share the file's typedefs and callable scope. `defCursor` is the authored
 * body of this occurrence: it keeps the absolute locations assembly and its diagnostics use.
 *
 * @param name The `defs:` entry name.
 * @param body The entry's normalized body.
 * @param registry The file's typedefs.
 * @param scope The file's callable scope.
 * @param defCursor The authored body, or `null` for a virtual fallback.
 * @returns The built child flow.
 * @throws LoadError, located and prefixed `defs entry '<name>': `, when the body isn't a mapping,
 *   has no `nodes:` or `kind:`, sets an unknown or renamed field, a malformed `llm_config:`,
 *   `env:` or `asserts:`, or fails to assemble.
 */
function loadDef(
  name: string,
  body: Value,
  registry: TypeRegistry,
  scope: CallableScope,
  defCursor: SourceCursor | null = null,
): ComposedFlow {
  const diagnostics = scope.diagnostics;
  const where = `defs entry ${reprText(name)}`;
  if (!isRecord(body)) {
    throw diagnostics.error(`${where}: body must be a mapping, got ${typeName(body)}`, {
      range: defCursor?.range ?? null,
    });
  }
  if (!Object.hasOwn(body, "nodes")) {
    if (Object.hasOwn(body, "kind")) return loadSingleNodeDef(name, body, registry, scope, defCursor);
    throw diagnostics.error(
      `${where}: a child flow needs a \`nodes:\` body (input?/nodes/output?) OR a single-node \`kind:\` form`,
      { range: defCursor?.range ?? null },
    );
  }
  rejectRemovedMemoryKey((key) => Object.hasOwn(body, key), defCursor, diagnostics, `${where}: `);
  const extra = Object.keys(body)
    .filter((key) => !DEF_ALLOWED.has(key))
    .sort(compareCodeUnits);
  if (extra.length > 0) {
    throw diagnostics.error(
      `${where}: unknown field(s) ${writeAsText(extra)} (allowed: ${DEF_SURFACE_KEYS.join(", ")})`,
      { range: defCursor?.keyRangeOf(extra[0]!) ?? null },
    );
  }
  const field = (key: string): Value => (Object.hasOwn(body, key) ? body[key]! : null);
  const defLlmConfig = readAuthoredLlmConfig(field("llm_config"), where, {
    registry,
    sitePath: ["def", name, "llm_config"],
    catalogBuilder: scope.catalogBuilder,
    diagnostics,
    cursor: defCursor,
  });
  const defEnv = field("env");
  if (defEnv !== null && !isRecord(defEnv)) {
    throw diagnostics.error(`${where}: \`env:\` must be a mapping with string keys`, {
      range: defCursor?.rangeOf("env") ?? null,
    });
  }
  if (defEnv !== null) validateEnv(defEnv, `${where}: `, defCursor?.child("env") ?? null, diagnostics);
  checkDefAsserts(name, field("asserts"), defCursor, diagnostics);
  // Like a full flow's, a written `input:` must be a mapping, so an empty one is refused.
  const inputs = Object.hasOwn(body, "inputs") ? body["inputs"]! : {};
  if (!isRecord(inputs)) {
    throw diagnostics.error(`${where}: ${valueTypeError("input", "a mapping", inputs, "write a mapping")}`, {
      range: defCursor?.rangeOf("input") ?? null,
    });
  }
  const nodesCursor = defAssemblyCursor(defCursor, body["nodes"]!);
  try {
    const asserts = field("asserts");
    return assemble({
      inputsSection: inputs,
      inputsCursor: sourceChild(defCursor, "input"),
      nodesCursor,
      outputsSection: field("outputs"),
      // Enforced at the child seam.
      assertsSection: isList(asserts) ? asserts : [],
      registry,
      scope,
      owner: ["def", name],
      flowLlmConfig: defLlmConfig,
      flowEnv: defEnv ?? {},
      boundaryFieldRanges: defCursor !== null ? boundaryFieldRanges(defCursor) : null,
      assertsCursor: sourceChild(defCursor, "asserts"),
      outputsCursor: sourceChild(defCursor, "output"),
      flowId: name,
      baseDir: scope.baseDir,
    });
  } catch (error) {
    if (!(error instanceof LoadError)) throw error;
    throw diagnostics.prepend(error, {
      range: defCursor?.range ?? null,
      message: `${where}: ${error.message}`,
      note: `definition ${reprText(name)}`,
      cause: error,
    });
  }
}

/**
 * Build a single-node `defs:` callable: `let f params = <one node over params>`.
 *
 * The flat body is `{kind, <logic fields>, input?, output?}`. `input:` declares the def's
 * parameters (name to type, like a flow signature), each bound by name into the single node
 * (`p: ${input.p}`); `output:` is the node's output type, the codomain. Everything else is the
 * node body (the kind, its logic fields and any node-local `asserts:`). It desugars to a one-node
 * child flow assembled as the multi-node form is. The synthetic node projects its header, logic,
 * generated bindings and output type onto the authored def cursor without changing that source.
 *
 * @param name The `defs:` entry name, which also names the node.
 * @param body The entry's normalized body.
 * @param registry The file's typedefs.
 * @param scope The file's callable scope.
 * @param defCursor The authored body, or `null` for a virtual fallback.
 * @returns The built child flow.
 * @throws LoadError, located and prefixed `defs entry '<name>': `, when `kind:` is non-text,
 *   unknown or `case`, `input:` isn't a param map, a parameter is named `inf`, or the node fails
 *   to assemble.
 */
function loadSingleNodeDef(
  name: string,
  body: ValueRecord,
  registry: TypeRegistry,
  scope: CallableScope,
  defCursor: SourceCursor | null = null,
): ComposedFlow {
  const diagnostics = scope.diagnostics;
  const where = `defs entry ${reprText(name)}`;
  // Framed like an error from assembling the node: the definition, then the `kind:` value.
  const kindError = (message: string): LoadError =>
    diagnostics.prepend(diagnostics.error(message, { range: defCursor?.rangeOf("kind") ?? null }), {
      range: defCursor?.range ?? null,
      message: `${where}: ${message}`,
      note: `definition ${reprText(name)}`,
    });
  const kind = body["kind"]!;
  const allowed = SINGLE_NODE_DEF_KINDS.join(", ");
  if (typeof kind !== "string") {
    throw kindError(`node ${reprText(name)}: ${notTextError("kind", kind, `write one of ${allowed}`)}`);
  }
  if (kind === "case") {
    throw kindError(
      `kind 'case' is not allowed in a single-node def (allowed: ${allowed}); a case routes to other ` +
        "nodes, which a one-node def has none of — write a full `nodes:` body instead",
    );
  }
  // A removed kind keeps the node parse's message, which names its replacement.
  if (!KINDS.has(kind) && !REMOVED_KINDS.has(kind)) {
    throw kindError(`node ${reprText(name)}: unknown kind ${reprText(kind)} (allowed: ${allowed})`);
  }
  const inputCursor = sourceChild(defCursor, "input");
  const paramTypes = signatureParams(Object.hasOwn(body, "inputs") ? body["inputs"]! : undefined, {
    where,
    inputLabel: "single-node `input:`",
    inputCursor,
    fallbackRange: defCursor?.range ?? null,
    diagnostics,
  });
  const nodesCursor = singleNodeDefCursor(name, body, defCursor);
  try {
    return assemble({
      inputsSection: paramTypes,
      inputsCursor: inputCursor,
      nodesCursor,
      // A single value re-export, node first.
      outputsSection: `\${${name}.output}`,
      assertsSection: [],
      registry,
      scope,
      owner: ["def", name],
      boundaryFieldRanges: defCursor !== null ? boundaryFieldRanges(defCursor) : null,
      outputsRange: sourceChild(defCursor, "output")?.range ?? null,
      flowId: name,
      baseDir: scope.baseDir,
    });
  } catch (error) {
    if (!(error instanceof LoadError)) throw error;
    throw diagnostics.prepend(error, {
      range: defCursor?.range ?? null,
      message: `${where}: ${error.message}`,
      note: `definition ${reprText(name)}`,
      cause: error,
    });
  }
}
