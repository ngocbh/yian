/**
 * The top-level loader: Compose YAML text to a runnable `ComposedFlow`.
 *
 * The file-I/O front door: own one source document, build the typedefs registry, compose the
 * call resolver, resolve every `uses:` import, and hand the parsed sections to the post-parse
 * conductor (`build/pipeline.ts`'s `assemble`) which returns the `ComposedFlow`. This module owns
 * the root's source projections and resolver composition (`makeFileResolver`, `CallableScope`),
 * not the assembly pipeline itself.
 *
 * A `call:` resolves a compiler-owned inline body, then a `uses:` alias resolved through the
 * injected `childResolver` (`(path, version) -> ComposedFlow`, sync or async). Every alias
 * resolves, in source order, before assembly starts, so assembly itself stays synchronous and
 * reads each import from the scope's memo. The loader derives the callable's signature and bakes
 * its compiled flow onto the built `CallNode` so `run` drives the embedded child.
 *
 * Source resolution follows two cache boundaries. A physical-file cache entry is canonical and
 * complete. `CallableScope` validates it, copies and rebases its compiled origins once per lexical
 * `uses:` alias, and reuses that alias view at every callsite. After the root is assembled, this
 * module freezes the one document catalog, filters local and alias authorities to the exact
 * reachable keys, and installs `ComposedFlow.originResolver`. Source-free graphs carry `null`;
 * private `assemble` products keep the resolver unset.
 *
 * A flow's chain of `uses:` imports is at most `current().maxCallNesting` files deep, the call
 * nesting limit, counted apart from call depth; each loaded flow records its own `importDepth`.
 *
 * Imports flow down only: this module composes the compose package's own components plus
 * `typesys` and `compile`; nothing in the engine imports it back.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import fsp from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve as resolvePath } from "node:path";

import { scopeMemoryDeclarations } from "../compile/memory_layout.ts";
import type { CompiledFlow } from "../compile/model.ts";
import { iterFlowOrigins, rebaseFlowOrigins } from "../compile/origins.ts";
import { current } from "../defaults.ts";
import {
  compareCodeUnits,
  DiagnosticFrame,
  OriginRef,
  type OriginResolver,
  ResolverTable,
  reprText,
  type SourceRange,
} from "../diagnostics.ts";
import { hostOnlyFieldError, LLMConfig } from "../llm_clients/config.ts";
import { ValidationError } from "../llm_clients/provider_blocks.ts";
import { isMaxToolIterations } from "../nodes/agent/node.ts";
import { Binding, type ValuePath, ValueTemplate } from "../nodes/contracts.ts";
import { readTypedefs, type TypeRegistry } from "../typesys/types.ts";
import {
  isList,
  isRecord,
  jsTypeName,
  TypeCheckError,
  type Value,
  type ValueRecord,
  valueRepr,
  writeAsText,
} from "../typesys/values.ts";
import { assemble, type CallableScope as AssemblyScope, ComposedFlow } from "./build/pipeline.ts";
import { INLINE_FLOW_PREFIX } from "./desugar/inline_bodies.ts";
import { LoadDiagnostics, LoadError } from "./helpers/errors.ts";
import { boundaryFieldRanges } from "./helpers/source.ts";
import { rejectUnreadTags } from "./helpers/tags.ts";
import { parseUsesRef } from "./helpers/uses.ts";
import { checkEnvValues } from "./helpers/value_nesting.ts";
import { checkLlmConfigValue, readLlmConfig } from "./nodes/agent.ts";
import { notTextDetail, notTextError } from "./nodes/base.ts";
import { callNestingScope } from "./nodes/resolver.ts";
import { flowNodesCursor, parseFile, parseSourceDocument } from "./parse/file.ts";
import { scopeOrigin, SourceCatalog, SourceCatalogBuilder } from "./source/catalog.ts";
import { SourceAccessError, type SourceCursor, type SourceDocument } from "./source/index.ts";

/**
 * Resolves a `uses:` ref to its flow: `path` is the ref's flow path and `version` its
 * `@<version>` (`null` when unpinned). It may return the flow or a promise of it.
 */
export type ChildResolver = (path: string, version: string | null) => ComposedFlow | Promise<ComposedFlow>;

const UNLOCATED_DIAGNOSTICS = new LoadDiagnostics();

// The fatal decoder refuses a file that isn't UTF-8; a BOM stays in the text, as Python reads it.
const UTF8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/** Return how a message shows a thrown value: an error's message, anything else as text. */
function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Return `flow` with some of its fields replaced, as a new frozen `ComposedFlow`. */
function replaceFlow(
  flow: ComposedFlow,
  changes: { compiled?: CompiledFlow; originResolver?: OriginResolver | null; importDepth?: number },
): ComposedFlow {
  return new ComposedFlow(changes.compiled ?? flow.compiled, {
    version: flow.version,
    name: flow.name,
    description: flow.description,
    originResolver: changes.originResolver === undefined ? flow.originResolver : changes.originResolver,
    importDepth: changes.importDepth === undefined ? flow.importDepth : changes.importDepth,
  });
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

/** Cross-file cycle state and successful compiled-flow reuse, by real path. */
export class LoadCtx {
  /** The real paths of the files being loaded. */
  readonly loading = new Set<string>();
  /** Each loaded file's flow, by real path. */
  readonly flows = new Map<string, ComposedFlow>();
}

// `uses:` imports being loaded around the current one. It spans files and `loadFlow` never resets
// it, so a custom `childResolver` that calls `loadFlow` is bounded too. Each import runs in its
// own store (`run`, never `enterWith`), so concurrent loads keep separate counts.
const ACTIVE_IMPORTS = new AsyncLocalStorage<number>();

/** Return a `system.paths` entry: a relative one joined onto the file's own directory. */
function joinPath(base: string, path: string): string {
  return isAbsolute(path) ? path : join(base, path);
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
    let frame: unknown;
    try {
      frame = resolve.call(resolver, origin);
    } catch (error) {
      throw UNLOCATED_DIAGNOSTICS.error(
        `uses: alias ${reprText(alias)} child origin resolver failed for ${reprText(origin.key)}: ${errorText(error)}`,
        { cause: error },
      );
    }
    if (!(frame instanceof DiagnosticFrame)) {
      const detail = frame === null ? "no frame" : jsTypeName(frame);
      throw UNLOCATED_DIAGNOSTICS.error(
        `uses: alias ${reprText(alias)} child origin resolver returned ${detail} for ${reprText(origin.key)}`,
      );
    }
  }
  return origins;
}

/**
 * Return one validated child copy scoped under its lexical alias. Every reachable memory
 * declaration in the copy gains `alias` as its outermost scope segment.
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
  let compiled: CompiledFlow;
  let table: ResolverTable;
  try {
    [compiled, table] = rebaseFlowOrigins(flow.compiled, (origin) => scopeOrigin(alias, origin), {
      resolver: flow.originResolver,
    });
    // The rebase returned an independent copy, so the scope rewrite can't reach the cached
    // canonical child.
    scopeMemoryDeclarations(compiled, alias);
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
    let frame: unknown;
    try {
      frame = authority.resolve(origin);
    } catch (error) {
      throw diagnostics.error(`resolver route failed for origin ${reprText(origin.key)}: ${errorText(error)}`, {
        cause: error,
      });
    }
    if (!(frame instanceof DiagnosticFrame)) {
      const detail = frame === null ? "no frame" : jsTypeName(frame);
      throw diagnostics.error(`resolver route returned ${detail} for origin ${reprText(origin.key)}`);
    }
    entries.push([origin, authority, origin]);
  }
  return replaceFlow(staged, { originResolver: ResolverTable.fromEntries(entries) });
}

/**
 * Build the default local file resolver for `uses:` references.
 *
 * The returned resolver finds a ref's `<dir>/<path>.yaml` on the search path, the first match
 * winning; `@<version>` is never part of the file name but a guard: the resolved file is loaded,
 * then its declared `version:` must equal the pinned one. Children load recursively, so their own
 * `uses:` and `system.paths` re-root at the child's directory. A `hub:` or other scheme ref never
 * reaches here.
 *
 * @param searchPaths Directories to search, in order.
 * @param ctx The load context shared by the whole import graph: a cross-file cycle guard and a
 *   diamond-reuse cache, keyed by real path.
 * @returns The resolver; it rejects with a `LoadError` for a missing target, a cross-file
 *   reference cycle, or a pinned-version mismatch.
 */
export function makeFileResolver(searchPaths: readonly string[], ctx: LoadCtx): ChildResolver {
  return async (path: string, version: string | null = null): Promise<ComposedFlow> => {
    for (const dir of searchPaths) {
      let real: string;
      try {
        real = await fsp.realpath(resolvePath(dir, path + ".yaml"));
      } catch {
        continue; // a search directory or file that doesn't exist is skipped
      }
      const cached = ctx.flows.get(real);
      if (cached !== undefined) {
        checkVersion(cached, path, version);
        return cached;
      }
      if (ctx.loading.has(real)) {
        throw UNLOCATED_DIAGNOSTICS.error(`flow-reference cycle across files: ${real}`);
      }
      ctx.loading.add(real);
      let child: ComposedFlow;
      try {
        const text = UTF8.decode(await fsp.readFile(real));
        child = await loadFlowFile(parseSourceDocument(text, basename(real)), null, [dirname(real)], ctx);
      } finally {
        ctx.loading.delete(real);
      }
      ctx.flows.set(real, child);
      checkVersion(child, path, version);
      return child;
    }
    throw UNLOCATED_DIAGNOSTICS.error(
      `flow ${reprText(path)} not found on search path ${writeAsText([...searchPaths])}`,
    );
  };
}

/**
 * Load Compose YAML text into a runnable `ComposedFlow`.
 *
 * Parses, compiles and validates the flow in one pass. A `call:` resolves an inline body or a
 * `uses:` alias (external, with the ref's `@<version>`); any other name throws. With neither
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
 * @returns The compiled, validated flow.
 * @throws LoadError (as a rejection) for any malformed or inconsistent flow, located at the
 *   offending line when source provenance is available.
 * @throws DefaultsError when a `YIAN_<NAME>` variable holds a value its entry doesn't accept.
 * @throws TypeError when `sourceLabel` isn't text.
 */
export async function loadFlow(
  text: string,
  options: { childResolver?: ChildResolver | null; searchPaths?: readonly string[] | null; sourceLabel?: string } = {},
): Promise<ComposedFlow> {
  const { childResolver = null, searchPaths = null, sourceLabel = "<flow>" } = options;
  // A bad YIAN_<NAME> value fails here, before any parsing.
  current();
  if (typeof sourceLabel !== "string") throw new TypeError("source_label must be a string");
  const document = parseSourceDocument(text, sourceLabel);
  return loadFlowFile(document, childResolver, searchPaths, new LoadCtx());
}

/**
 * Read a flow- or def-level `llm_config` with its values and tags, and check it.
 *
 * The layer is read through its tags the way an AGENT node's own is: a layer holding `${...}`
 * values becomes a `ValueTemplate`, or a `Binding` for a whole record written as one value, and is
 * checked by `checkLlmConfigValue`. Otherwise it must be a mapping of `LLMConfig` fields without
 * `api_key`, `organization`, `project`, `headers`, or `base_url` other than beside
 * `provider: azure`, which only the host sets. A text field YAML read as non-text (such as
 * `model: 5`) is named by `notTextError`; any other violation keeps the `LLMConfig` message.
 * `inherit` is an AGENT-only key.
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
function readAuthoredLlmConfig(
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
  if (llmConfig instanceof ValueTemplate || llmConfig instanceof Binding) {
    return checkLlmConfigValue(llmConfig, `${where}: llm_config`, sourceRange, diagnostics, ranges, {
      allowInherit: false,
    })[0];
  }
  if (!isRecord(llmConfig)) throw diagnostics.error(`${where}: llm_config must be a mapping`, { range: sourceRange });
  const message = hostOnlyFieldError(llmConfig);
  if (message !== null) throw diagnostics.error(`${where}: ${message}`, { range: sourceRange });
  try {
    new LLMConfig(llmConfig);
  } catch (error) {
    if (error instanceof ValidationError) {
      const detail = notTextDetail(error, llmConfig);
      throw diagnostics.error(`${where}: llm_config: ` + (detail !== null ? notTextError(...detail) : error.message), {
        range: sourceRange,
        cause: error,
      });
    }
    throw diagnostics.error(`${where}: llm_config: ${errorText(error)}`, { range: sourceRange, cause: error });
  }
  return llmConfig;
}

/**
 * Refuse a flow `env:` value too deep or holding itself, or an invalid `max_tool_iterations`.
 *
 * The nesting and recursive alias checks run first, so a deep `max_tool_iterations` isn't shown in
 * full; a `max_tool_iterations` default must be one every AGENT node would accept when it runs.
 *
 * @throws LoadError at the offending entry.
 */
function validateFlowEnv(env: ValueRecord, options: { diagnostics: LoadDiagnostics; cursor: SourceCursor }): void {
  const { diagnostics, cursor } = options;
  checkEnvValues(env, "flow ", cursor.has("env") ? cursor.child("env") : null, diagnostics);
  if (Object.hasOwn(env, "max_tool_iterations") && !isMaxToolIterations(env["max_tool_iterations"])) {
    throw diagnostics.error(
      "flow `env.max_tool_iterations` must be a positive int or -1 (no cap), " +
        `got ${valueRepr(env["max_tool_iterations"])}`,
      { range: cursor.child("env").rangeOf("max_tool_iterations") },
    );
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
 * Load one parsed flow file: its checks, its `uses:` imports, then its assembly.
 *
 * Every step before the imports is synchronous; each alias then resolves, in source order, and
 * the rest runs inside one synchronous call nesting scope, so the file counts its own calls.
 *
 * @param document The file's parsed document.
 * @param childResolver The explicit external resolver, or `null` for the local file resolver.
 * @param searchPaths The file's own directory first, then other search directories; `null` makes
 *   `system.paths` inert.
 * @param ctx The load context shared by the whole import graph.
 */
async function loadFlowFile(
  document: SourceDocument,
  childResolver: ChildResolver | null,
  searchPaths: readonly string[] | null,
  ctx: LoadCtx,
): Promise<ComposedFlow> {
  const root = document.root;
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const f = parseFile(document);
  rejectAuthoredPrivateCallableNames(root, f.defs, f.uses);
  validateFlowEnv(f.env, { diagnostics, cursor: root });
  validateSystem(f.system, root.has("system") ? root.child("system") : null);

  // Typedefs are read as plain values, so a tag in them is rejected before they are.
  rejectUnreadTags(document, new Set(), diagnostics, { typedefsOnly: true });
  let registry: TypeRegistry;
  try {
    registry = readTypedefs(f.typedefs);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw diagnostics.error(`bad typedefs: ${error.message}`, {
      range: typedefErrorRange(root, error.path),
      cause: error,
    });
  }

  // `system.paths` is honored only when the file's own directory is known: it was loaded with
  // search paths, whose first entry is that directory. Without them `system.paths` is inert, so
  // no resolver exists and a `uses:` alias is loud.
  let effective: string[] = [];
  if (searchPaths !== null && searchPaths.length > 0) {
    const own = searchPaths[0]!;
    const paths = (f.system["paths"] ?? []) as string[];
    effective = [...searchPaths, ...paths.map((path) => joinPath(own, path))];
  }

  let external: ChildResolver | null = null;
  if (childResolver !== null) external = childResolver;
  else if (effective.length > 0) external = makeFileResolver(effective, ctx);

  const scope = new CallableScope(f.defs, registry, external, f.uses, {
    defsCursor: root.has("defs") ? root.child("defs") : null,
    usesCursor: root.has("uses") ? root.child("uses") : null,
    diagnostics,
  });
  // Read after the typedefs, which its tags resolve against.
  const flowLlmConfig = readAuthoredLlmConfig(f.llmConfig, "flow", {
    registry,
    sitePath: ["root", "llm_config"],
    catalogBuilder: scope.catalogBuilder,
    diagnostics,
    cursor: root,
  });
  // A collision is reported before a colliding alias's broken target.
  const collide = Object.keys(f.uses)
    .filter((alias) => Object.hasOwn(f.nodes, alias))
    .sort(compareCodeUnits);
  if (collide.length > 0) throw diagnostics.error(`uses: alias ${writeAsText(collide)} collides with a node id`);
  // Imports load before anything assembles, so assembly reads every alias from the memo.
  for (const alias of Object.keys(f.uses)) await scope.resolveAlias(alias);

  return callNestingScope(() => {
    scope.eagerLoadDefs();
    const staged = assemble({
      inputsSection: f.inputs,
      inputsCursor: sourceChild(root, "input"),
      nodesCursor: flowNodesCursor(document, f.nodes),
      outputsSection: f.outputs,
      assertsSection: f.asserts,
      registry,
      scope,
      owner: ["root"],
      usesAliases: Object.keys(f.uses),
      version: f.version,
      name: f.name,
      description: f.description,
      flowLlmConfig,
      flowEnv: f.env,
      boundaryFieldRanges: boundaryFieldRanges(root),
      assertsCursor: root.has("nodes") && root.has("asserts") ? root.child("asserts") : null,
      outputsCursor: root.has("nodes") && root.has("output") ? root.child("output") : null,
      flowId: f.id,
    });
    // The root and its inline bodies are assembled now, so every slot has read its tags; a
    // `uses:` child checks its own document in its own load.
    rejectUnreadTags(document, scope.catalogBuilder.consumedTags, diagnostics);
    return replaceFlow(scope.finalize(staged), { importDepth: scope.importDepth });
  });
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
 * Compiler-owned callables resolve first, then authored `uses:` aliases. A bare name in no
 * authored mapping is rejected instead of reaching the external resolver. Private declarations
 * and external aliases share one memo table. The in-progress set makes recursive private
 * construction loud; cross-file recursion is outside this scope and belongs to the selected
 * external resolver. A failed callable construction drops its new private declarations, nested
 * cache entries and source-catalog sites together, so a later retry can't see a stale partial
 * attempt.
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
    } = {},
  ) {
    const { defsCursor = null, usesCursor = null, catalogBuilder = null, diagnostics = null } = options;
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

  /**
   * Build one callable without caching origins from a failed attempt: when `builder` throws, the
   * memo entries and private declarations it added are dropped with the catalog sites the
   * builder's transaction rolls back.
   */
  #buildTransactionally(builder: () => ComposedFlow): ComposedFlow {
    const cachedBefore = new Set(this.#built.keys());
    const privateBefore = new Set(this.#private.keys());
    try {
      return this.#catalogBuilder.transaction(builder);
    } catch (error) {
      for (const flowId of [...this.#built.keys()]) if (!cachedBefore.has(flowId)) this.#built.delete(flowId);
      for (const flowId of [...this.#private.keys()]) if (!privateBefore.has(flowId)) this.#private.delete(flowId);
      throw error;
    }
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

    let authoredSection: string | null = null;
    if (Object.hasOwn(this.#defsSection, flowId)) authoredSection = "defs";
    else if (Object.hasOwn(this.#uses, flowId)) authoredSection = "uses";
    if (authoredSection !== null) {
      throw this.#diagnostics.error(
        "compiler-generated callable collision in reserved prefix " +
          `${reprText(INLINE_FLOW_PREFIX)}: ${reprText(label)} owned by ${writeAsText([...owner])} conflicts ` +
          `with an authored ${authoredSection}: name`,
      );
    }

    this.#private.set(flowId, new PrivateCallable(owner, label, builder));
  }

  /**
   * Resolve one callable once, synchronously, for assembly.
   *
   * A `uses:` alias is read from the memo `resolveAlias` filled; this never loads a file.
   *
   * @param flowId The callable name: a compiler-generated callable or a `uses:` alias.
   * @param _version Unused; a `uses:` alias takes its version from its own ref.
   * @returns The built callable, cached for later calls.
   * @throws LoadError when the callable is unknown, recursive, fails to build, or is an alias that
   *   wasn't resolved first.
   */
  resolve(flowId: string, _version: string | null = null): ComposedFlow {
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
      this.#inProgress.add(flowId);
      let loaded: ComposedFlow;
      try {
        loaded = this.#buildTransactionally(privateCallable.builder);
      } finally {
        this.#inProgress.delete(flowId);
      }
      this.#built.set(flowId, loaded);
      return loaded;
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
   * @param alias The `uses:` alias.
   * @returns The alias's rebased child, cached for later calls.
   * @throws LoadError (as a rejection), located at the alias, when its ref is malformed or has a
   *   scheme, no resolver was given, the resolver fails or returns a bad flow, or the import chain
   *   is past the limit.
   */
  async resolveAlias(alias: string): Promise<ComposedFlow> {
    const built = this.#built.get(alias);
    if (built !== undefined) return built;
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
}
