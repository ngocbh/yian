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
 * After a file assembles, its CODE nodes are checked (each def's in written order, then the
 * root's) against the file's base directory. Only the root load sees the fully assembled graph, so
 * `loadFlow` then resolves its memory layout and refuses an `llm_config` value a child flow's agent
 * would inherit before it is known.
 *
 * A flow's chain of `uses:` imports is at most `current().maxCallNesting` files deep, the call
 * nesting limit, counted apart from call depth; each loaded flow records its own `importDepth`.
 *
 * Imports flow down only: this module composes the compose package's own components plus
 * `typesys` and `compile`; nothing in the engine imports it back.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import fsp from "node:fs/promises";
import { basename, dirname, isAbsolute, resolve as resolvePath } from "node:path";

import { fieldLabel, findInheritedLlmValue, inheritedValueFix } from "../compile/llm_cascade.ts";
import { resolveMemoryLayout, scopeMemoryDeclarations } from "../compile/memory_layout.ts";
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
import { DeclSite, MemorySetupError } from "../memory/contracts.ts";
import { isMaxToolIterations } from "../nodes/agent/node.ts";
import type { Node } from "../nodes/base.ts";
import { CodeNode } from "../nodes/code/node.ts";
import { Binding, type ValuePath, ValueTemplate } from "../nodes/contracts.ts";
import { pythonTypeName, readTypedefs, type TypeRegistry } from "../typesys/types.ts";
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
import { parseMemoryDeclarations } from "./helpers/memory.ts";
import { boundaryFieldRanges } from "./helpers/source.ts";
import { rejectUnreadTags } from "./helpers/tags.ts";
import { parseUsesRef } from "./helpers/uses.ts";
import { checkEnvValues } from "./helpers/value_nesting.ts";
import { checkLlmConfigValue, readLlmConfig } from "./nodes/agent.ts";
import { notTextDetail, notTextError, rejectInfinityName } from "./nodes/base.ts";
import { type CheckedFlow, checkCodeNode, checkFlowCode, type CodeCheckOptions } from "./nodes/code.ts";
import { callNestingScope } from "./nodes/resolver.ts";
import { flowNodesCursor, isEmptyValue, parseFile, parseSourceDocument } from "./parse/file.ts";
import { scopeOrigin, SourceCatalog, SourceCatalogBuilder } from "./source/catalog.ts";
import { SourceAccessError, SourceCursor, type SourceDocument } from "./source/index.ts";

/**
 * Resolves a `uses:` ref to its flow: `path` is the ref's flow path and `version` its
 * `@<version>` (`null` when unpinned). It may return the flow or a promise of it.
 */
export type ChildResolver = (path: string, version: string | null) => ComposedFlow | Promise<ComposedFlow>;

const UNLOCATED_DIAGNOSTICS = new LoadDiagnostics();

// The fatal decoder refuses a file that isn't UTF-8; a BOM stays in the text, and the reader
// turns `\r\n` and a lone `\r` into `\n`, as Python's `Path.read_text()` reads it.
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

/** Return the authored def cursor's `nodes:`, or a surface-restored virtual fallback. */
function defAssemblyCursor(defCursor: SourceCursor | null, normalizedNodes: Value): SourceCursor {
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
 * source it came from: a generated binding points at its parameter's declaration.
 */
function singleNodeDefCursor(name: string, body: ValueRecord, defCursor: SourceCursor | null): SourceCursor {
  const paramTypes = isRecord(body["inputs"]) ? body["inputs"] : {};
  const generatedInputs: ValueRecord = Object.fromEntries(
    Object.keys(paramTypes).map((param) => [param, `\${input.${param}}`]),
  );
  const nodeBody: ValueRecord = Object.fromEntries(
    Object.entries(body).filter(([key]) => key !== "inputs" && key !== "outputs"),
  );
  nodeBody["input"] = generatedInputs;
  if (Object.hasOwn(body, "outputs")) nodeBody["output"] = body["outputs"]!;

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
  "long_term_memory",
]);
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

/**
 * Return `path` spelled as Python's `PurePosixPath` spells it: repeated slashes merged, `.`
 * segments and a trailing slash dropped, `..` kept.
 */
function pathText(path: string): string {
  const root = path.startsWith("/") ? "/" : "";
  return root + path.split("/").filter((part) => part !== "" && part !== ".").join("/") || ".";
}

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
      } catch (error) {
        // A search directory or file that doesn't exist is skipped; any other failure is loud.
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "ENOENT" || code === "ENOTDIR" || code === "ELOOP") continue;
        throw error;
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
        const text = UTF8.decode(await fsp.readFile(real)).replace(/\r\n?/g, "\n");
        const own = dirname(real);
        child = await loadFlowFile(parseSourceDocument(text, basename(real)), null, [own], ctx, own);
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
 * @param options.baseDir The directory a CODE node's `./file.ts:fn` reference resolves against
 *   (default: the first search path, else the current directory). A file read through the search
 *   path resolves its own references against its own directory.
 * @returns The compiled, validated flow.
 * @throws LoadError (as a rejection) for any malformed or inconsistent flow, located at the
 *   offending line when source provenance is available: among them a memory layout that breaks a
 *   rule, and an `llm_config` value a child flow's agent would inherit before it is known.
 * @throws DefaultsError when a `YIAN_<NAME>` variable holds a value its entry doesn't accept.
 * @throws TypeError when `sourceLabel` isn't text.
 */
export async function loadFlow(
  text: string,
  options: {
    childResolver?: ChildResolver | null;
    searchPaths?: readonly string[] | null;
    sourceLabel?: string;
    baseDir?: string | null;
  } = {},
): Promise<ComposedFlow> {
  const { childResolver = null, searchPaths = null, sourceLabel = "<flow>", baseDir = null } = options;
  // A bad YIAN_<NAME> value fails here, before any parsing.
  current();
  if (typeof sourceLabel !== "string") throw new TypeError("source_label must be a string");
  const document = parseSourceDocument(text, sourceLabel);
  const ownDir = baseDir ?? searchPaths?.[0] ?? process.cwd();
  const flow = await loadFlowFile(document, childResolver, searchPaths, new LoadCtx(), ownDir);
  // Only the root load sees the fully assembled graph, imports included; children loaded for
  // uses: are checked as part of their importer.
  try {
    resolveMemoryLayout(flow.compiled);
  } catch (error) {
    if (!(error instanceof MemorySetupError)) throw error;
    throw memorySetupLoadError(flow, error);
  }
  // Which flow is the run's root, and so which inputs are known at run start, is decided only
  // here.
  refuseInheritedLlmValues(flow);
  return flow;
}

/**
 * Locate one memory layout failure at its first declaration site.
 *
 * @param flow The loaded root, whose resolver maps origins to source.
 * @param error The layout failure with its origins and notes.
 * @returns A `LoadError` framed at `error.origin`. Its notes are the failure's own notes, then
 *   one located note per related origin whose location differs from the primary frame and from
 *   every earlier note.
 */
export function memorySetupLoadError(
  flow: { readonly originResolver: OriginResolver | null },
  error: MemorySetupError,
): LoadError {
  const resolver = flow.originResolver;
  const frame = (origin: OriginRef | null): DiagnosticFrame | null =>
    origin === null || resolver === null ? null : resolver.resolve(origin);

  const primary = frame(error.origin);
  const notes = [...(error.notes ?? [])];
  // Aliases of one file map distinct origins to one source location.
  const seen = new Set<string>();
  const locationOf = (other: DiagnosticFrame): string =>
    `${other.label}:${other.range.startLine}:${other.range.startCol}`;
  if (primary !== null) seen.add(locationOf(primary));
  for (const origin of error.related) {
    const other = frame(origin);
    if (other === null) continue;
    const location = locationOf(other);
    if (seen.has(location)) continue;
    seen.add(location);
    notes.push(`related declaration at ${location}`);
  }
  return UNLOCATED_DIAGNOSTICS.error(error.message, {
    frames: primary === null ? [] : [primary],
    notes,
    cause: error,
  });
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
  const resolver = flow.originResolver;
  const frame = (origin: OriginRef | null): DiagnosticFrame | null =>
    origin === null || resolver === null ? null : resolver.resolve(origin);

  const valueFrame = frame(found.origin);
  const field = fieldLabel(found.field);
  let subject: string;
  if (found.callsite === null) {
    subject = `flow ${field}`;
  } else {
    // A CALL, MAP or LOOP callsite names its child flow.
    const name = reprText((found.callsite as Node & { flowId: string }).flowId);
    const called = `called by node ${reprText(found.owner.join("/"))}`;
    const callsiteFrame = frame(found.callsite.origin);
    if (valueFrame === null || callsiteFrame === null) subject = `${field} of the flow ${name} (${called})`;
    else if (valueFrame.label !== callsiteFrame.label) {
      subject = `${field} of the flow imported through uses: ${name} (${called})`;
    } else subject = `defs entry ${name} ${field} (${called})`;
  }
  const agent = found.gate ?? `agent ${reprText(found.agent.join("/"))} in a child flow`;
  throw UNLOCATED_DIAGNOSTICS.error(
    `${subject}: ${agent} would inherit this value, which is read only when the flow ` +
      `runs; ${inheritedValueFix(found.field)}`,
    { frames: valueFrame === null ? [] : [valueFrame] },
  );
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
 */
async function loadFlowFile(
  document: SourceDocument,
  childResolver: ChildResolver | null,
  searchPaths: readonly string[] | null,
  ctx: LoadCtx,
  baseDir: string,
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
    effective = [...searchPaths.map(pathText), ...paths.map((path) => joinPath(own, path))];
  }

  const external = childResolver ?? (effective.length > 0 ? makeFileResolver(effective, ctx) : null);

  const scope = new CallableScope(f.defs, registry, external, f.uses, {
    defsCursor: root.has("defs") ? root.child("defs") : null,
    usesCursor: root.has("uses") ? root.child("uses") : null,
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
  const rootMemories =
    root.has("nodes") && f.longTermMemory !== null
      ? parseMemoryDeclarations(f.longTermMemory, {
          diagnostics,
          sourceRange: root.rangeOf("long_term_memory"),
          site: new DeclSite([], ["root"], null),
        })
      : [];
  // A collision is reported before a colliding alias's broken target.
  const collide = Object.keys(f.uses)
    .filter((alias) => Object.hasOwn(f.nodes, alias))
    .sort(compareCodeUnits);
  if (collide.length > 0) throw diagnostics.error(`uses: alias ${writeAsText(collide)} collides with a node id`);
  // Imports load before anything assembles, so assembly reads every alias from the memo.
  for (const alias of Object.keys(f.uses)) await scope.resolveAlias(alias);

  const staged = callNestingScope(() => {
    scope.eagerLoadDefs();
    return assemble({
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
      memories: rootMemories,
      memoriesCursor: root.has("nodes") && root.has("long_term_memory") ? root.child("long_term_memory") : null,
      baseDir,
    });
  });
  const code: CodeCheckOptions = {
    diagnostics,
    resolveOrigin: (origin) => scope.catalogBuilder.resolveRegistered(origin),
    // Shared so the root skips the defs its calls bake in, already checked.
    seen: new Set(),
  };
  await scope.checkDefsCode(code);
  await checkFlowCode(staged.compiled, code);
  // Defs, inline bodies and the root are assembled now, so every slot has read its tags; a
  // `uses:` child checks its own document in its own load.
  rejectUnreadTags(document, scope.catalogBuilder.consumedTags, diagnostics);
  return replaceFlow(scope.finalize(staged), { importDepth: scope.importDepth });
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
 * belongs to the selected external resolver. A failed callable construction drops its new private
 * declarations, nested cache entries and source-catalog sites together, so a later retry can't
 * see a stale partial attempt.
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
    } = {},
  ) {
    const { defsCursor = null, usesCursor = null, catalogBuilder = null, diagnostics = null, baseDir = null } = options;
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
   * @param flowId The callable name: a compiler-generated callable, a `defs:` entry or a `uses:`
   *   alias.
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
    if (Object.hasOwn(this.#defsSection, flowId)) {
      const defCursor = sourceChild(this.#defsCursor, flowId);
      if (this.#inProgress.has(flowId)) {
        throw this.#diagnostics.error(
          `recursive defs reference: callable ${reprText(flowId)} is defined in terms ` +
            "of itself (the baked call graph must be finite)",
          { range: defCursor?.range ?? null },
        );
      }
      this.#inProgress.add(flowId);
      let loaded: ComposedFlow;
      try {
        loaded = this.#buildTransactionally(() =>
          loadDef(flowId, this.#defsSection[flowId]!, this.#registry, this, defCursor),
        );
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

  /**
   * Check the CODE nodes of every def, in written order, each in its own frame.
   *
   * A def no node calls isn't reachable from the root graph, so each def is checked here. A def
   * calling another checks the callee inside its own frame, as building it does, so a callee's
   * failure carries both prefixes and both frames, outer first. A def is checked once.
   *
   * @param options The CODE check options; their `seen` set records each checked flow.
   * @throws LoadError (as a rejection), prefixed `defs entry '<name>': ` and located at the def,
   *   at the first CODE node whose check fails.
   */
  async checkDefsCode(options: CodeCheckOptions): Promise<void> {
    const seen = options.seen ?? new Set<object>();
    // A callsite's child flow is the built def's own `compiled`, so identity names the def.
    const defNames = new Map<object, string>();
    for (const [name, flow] of this.#built) {
      if (Object.hasOwn(this.#defsSection, name)) defNames.set(flow.compiled, name);
    }
    const visit = async (flow: CheckedFlow): Promise<void> => {
      if (seen.has(flow)) return;
      seen.add(flow);
      for (const node of flow.nodes.values()) {
        if (node instanceof CodeNode) await checkCodeNode(node, options);
        for (const child of node.originChildren()) {
          const name = defNames.get(child as object);
          await (name === undefined ? visit(child as CheckedFlow) : checkDef(name));
        }
      }
    };
    const checkDef = async (name: string): Promise<void> => {
      try {
        await visit(this.#built.get(name)!.compiled);
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
    throw diagnostics.error(`${where}: body must be a mapping, got ${pythonTypeName(body)}`, {
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
  if (Object.hasOwn(body, "memory_policy")) {
    throw diagnostics.error(`${where}: field \`memory_policy:\` was renamed to \`long_term_memory:\``, {
      range: defCursor?.keyRangeOf("memory_policy") ?? null,
    });
  }
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
  if (defEnv !== null) checkEnvValues(defEnv, `${where}: `, defCursor?.child("env") ?? null, diagnostics);
  checkDefAsserts(name, field("asserts"), defCursor, diagnostics);
  const nodesCursor = defAssemblyCursor(defCursor, body["nodes"]!);
  try {
    const memories =
      field("long_term_memory") !== null
        ? parseMemoryDeclarations(body["long_term_memory"]!, {
            diagnostics,
            sourceRange: defCursor?.rangeOf("long_term_memory") ?? null,
            site: new DeclSite([], ["def", name], null),
          })
        : [];
    const inputs = field("inputs");
    const asserts = field("asserts");
    return assemble({
      inputsSection: (isEmptyValue(inputs) ? {} : inputs) as ValueRecord,
      inputsCursor: sourceChild(defCursor, "input") ?? sourceChild(defCursor, "inputs"),
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
      outputsCursor: sourceChild(defCursor, "output") ?? sourceChild(defCursor, "outputs"),
      flowId: name,
      memories,
      memoriesCursor: sourceChild(defCursor, "long_term_memory"),
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
 * @throws LoadError, located and prefixed `defs entry '<name>': `, when `input:` isn't a param
 *   map, a parameter is named `inf`, or the node fails to assemble.
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
  const inputs = Object.hasOwn(body, "inputs") ? body["inputs"]! : null;
  const inputCursor = sourceChild(defCursor, "input") ?? sourceChild(defCursor, "inputs");
  if (!isRecord(inputs) && !isEmptyValue(inputs)) {
    throw diagnostics.error(
      `${where}: single-node \`input:\` must be a param map (\`name: !TYPE\`, e.g. \`topic: !str\`)`,
      { range: inputCursor?.range ?? defCursor?.range ?? null },
    );
  }
  const paramTypes: ValueRecord = isRecord(inputs) ? inputs : {};
  // The node reads each parameter by bare name, so `inf` would read infinity instead.
  if (Object.hasOwn(paramTypes, "inf")) {
    rejectInfinityName(`${where}: input 'inf'`, "inf", inputCursor?.keyRangeOf("inf") ?? null, diagnostics);
  }
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
