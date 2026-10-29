/**
 * Helpers for the Compose tests: read authored `input:` or `variables:` declarations straight from
 * YAML text, load conformance flows, hand-build child flows, catch load errors, and run case rows
 * that parse a flow's nodes or assemble its root as the loader does.
 */

import { existsSync, readFileSync, symlinkSync } from "node:fs";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

import { expect } from "bun:test";

import { CompiledFlow, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { assemble, type CallableScope as AssemblyScope, ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { readSourceParameters } from "../../src/compose/helpers/types.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
import { CallableScope, loadFlow, readAuthoredLlmConfig, rootAssembleOptions } from "../../src/compose/loader.ts";
import { flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { reload } from "../../src/defaults.ts";
import { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Binding, type OutputContract, type Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { type Case, prepare, readCase } from "../conformance/runner.ts";
import { withKinds } from "./stand_ins.ts";

const CASES = fileURLToPath(new URL("../conformance/cases/", import.meta.url));

/**
 * The repository's `tests/conformance/` directory: the `baseDir` of a flow loaded through
 * `loadFlow` whose CODE nodes name `./fixtures/compose_codefns.ts:*`.
 */
export const CONFORMANCE_DIR = fileURLToPath(new URL("../conformance", import.meta.url));

/** Load `text` with `CONFORMANCE_DIR` as its default `baseDir`, so the fixture CODE references resolve. */
export const loadConformanceFlow = (text: string, options: Parameters<typeof loadFlow>[1] = {}) =>
  loadFlow(text, { baseDir: CONFORMANCE_DIR, ...options });

/**
 * Link `<dir>/fixtures` to the conformance fixtures, once: a flow file read from `dir` resolves its
 * CODE references against `dir`.
 */
export function linkConformanceFixtures(dir: string): void {
  const link = join(dir, "fixtures");
  if (!existsSync(link)) symlinkSync(join(CONFORMANCE_DIR, "fixtures"), link);
}

/**
 * Parse `text` and read its `field` section as located parameters.
 *
 * @param text A YAML document holding a `field:` mapping.
 * @param registry Typedefs the declarations resolve against.
 * @param options.field The section to read (default `input`).
 * @param options.requireDefault Read the section as agent variables, each needing a default
 *   (default `false`).
 * @returns One parameter per declaration, in source order.
 */
export function parametersFromYaml(
  text: string,
  registry: TypeRegistry,
  options: { field?: string; requireDefault?: boolean } = {},
): Parameter[] {
  const { field = "input", requireDefault = false } = options;
  const cursor = SourceDocument.parse(text).root.child(field);
  return readSourceParameters(cursor, registry, {
    sitePath: ["test"],
    catalogBuilder: new SourceCatalogBuilder(),
    fieldName: field,
    declarationLabel: requireDefault ? "variable" : "input",
    requireDefault,
  });
}

/**
 * Build a child flow by hand, standing in for a `defs:` entry the loader would compose: START over
 * `inputs`, one producer per entry of `producers`, and an END returning `output`.
 *
 * @param options.inputs The child's parameters, e.g. from `parametersFromYaml` over the def's
 *   `input:` text (default none).
 * @param options.producers Node id to producer: an output contract (or `null`) becomes a CODE node
 *   `return null` carrying that contract, as the def's typed node would; a `Node` is used as is
 *   (default none).
 * @param options.output A source or `Binding` for a value END (its output named `result`, as
 *   the loader names it), a record of them for a record END, or `null` for a value END with no
 *   output.
 */
export function handChild(options: {
  inputs?: readonly Parameter[];
  producers?: Readonly<Record<string, OutputContract | Node | null>>;
  output: string | Binding | Readonly<Record<string, string | Binding>> | null;
}): ComposedFlow {
  const { inputs = [], producers = {}, output } = options;
  const nodes = new Map<string, Node>([[START_ID, new StartNode(START_ID, { params: inputs })]]);
  for (const [id, producer] of Object.entries(producers)) {
    if (producer instanceof Node) {
      nodes.set(id, producer);
      continue;
    }
    const node = new CodeNode(id, { code: "return null" });
    node.outputContract = producer;
    nodes.set(id, node);
  }
  let outputs: FlowOutput[];
  if (output === null) {
    nodes.set(END_ID, EndNode.value(END_ID, { outputName: null }));
    outputs = [];
  } else if (typeof output === "string" || output instanceof Binding) {
    nodes.set(END_ID, EndNode.value(END_ID, { outputName: "result" }));
    outputs = [new FlowOutput("result", output)];
  } else {
    const entries = Object.entries(output);
    nodes.set(END_ID, EndNode.record(END_ID, { outputNames: entries.map(([name]) => name) }));
    outputs = entries.map(([name, binding]) => new FlowOutput(name, binding));
  }
  return new ComposedFlow(new CompiledFlow(nodes, [], { outputs }));
}

/**
 * Parse flow `text`'s `nodes:` as the loader reads them (`parseSourceDocument`, `parseFile`,
 * `flowNodesCursor`, `parseNodes`), with `classes` (default none) registered as the node kinds.
 *
 * @param label The file name diagnostics name (default `<flow>`).
 * @returns The node descriptors by id.
 */
export function nodesOf(
  text: string,
  classes: readonly DescriptorClass[] = [],
  label = "<flow>",
): Map<string, NodeDescriptor> {
  const document = parseSourceDocument(text, label);
  const cursor = flowNodesCursor(document, parseFile(document).nodes);
  return withKinds(classes, () => parseNodes(cursor));
}

/** Parse flow `text`'s top level as the loader does (`parseSourceDocument`, `parseFile`). */
export const fileOf = (text: string, label: string) => parseFile(parseSourceDocument(text, label));

/**
 * Read conformance case `id` (`group/name`) as the TS runner runs it: the prepared case, its flow
 * text and the label it parses under, `flow.yaml` (another file of the case with `file`, labelled
 * by its base name).
 *
 * The case's `env:` is applied with `reload`, since parsing and typedef reading use the
 * configured limits; this changes the process's defaults, and the caller restores them with
 * `forget()`, e.g. in `afterEach`.
 *
 * @throws Error when the case has no such flow or file text.
 */
export function caseText(
  id: string,
  options: { file?: string } = {},
): { case_: Case; text: string; label: string } {
  const { file } = options;
  const case_ = prepare(readCase(readFileSync(`${CASES}${id}.yaml`, "utf8")));
  const env = (case_["env"] ?? {}) as Record<string, string>;
  reload((name) => (Object.hasOwn(env, name) ? env[name] : undefined));
  const files = case_["files"];
  const text =
    file === undefined
      ? case_["flow"]
      : typeof files === "object" && files !== null && !Array.isArray(files)
        ? files[file]
        : undefined;
  if (typeof text !== "string") {
    throw new Error(`case ${id} has no ${file === undefined ? "flow" : `file ${file}`}`);
  }
  return { case_, text, label: file === undefined ? "flow.yaml" : basename(file) };
}

/**
 * Read conformance case `id` as `caseText` does, its flow (or `file`) parsed as a document.
 *
 * @throws Error when the case has no such flow or file text.
 */
export function caseDocument(
  id: string,
  options: { file?: string } = {},
): { case_: Case; document: SourceDocument } {
  const { case_, text, label } = caseText(id, options);
  return { case_, document: SourceDocument.parse(text, label) };
}

// ---------- the harvested-case harness ----------

/** Return the `LoadError` `run` throws. */
export function loadError(run: () => unknown): LoadError {
  try {
    run();
  } catch (error) {
    if (error instanceof LoadError) return error;
    throw error;
  }
  throw new Error("expected a LoadError");
}

/** Return the `LoadError` `promise` rejects with. */
export async function loadErrorAsync(promise: Promise<unknown>): Promise<LoadError> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof LoadError) return error;
    throw error;
  }
  throw new Error("expected a LoadError");
}

/** Return the cursor at `path` from the document root. */
export function cursorAt(document: SourceDocument, path: readonly (string | number)[]): SourceCursor {
  return path.reduce((cursor: SourceCursor, step) => cursor.child(step), document.root);
}

/** Return the registry the loader reads for `document`: its `typedefs:`, else an empty one. */
export function caseRegistry(document: SourceDocument): TypeRegistry {
  const root = document.root;
  return root.has("typedefs") ? readTypedefs(root.mapping("typedefs")) : new Map();
}

/** The call the loader makes on a case's document; `builder` holds the file's consumed-tag ledger. */
export type CaseCall = (
  document: SourceDocument,
  registry: TypeRegistry,
  diagnostics: LoadDiagnostics,
  builder: SourceCatalogBuilder,
) => unknown;

/** One harvested conformance case and the loader call it decides. */
export interface CaseRow {
  /** The case, `group/name`. */
  id: string;
  call: CaseCall;
  /** A file of the case other than its flow. */
  file?: string;
  /** The loader's prefix on the expected message. */
  prefix?: string;
  /** Compare every frame and the notes too, not only the last frame. */
  full?: boolean;
}

/** Read the `input:` declarations at `path` as the loader reads owner `sitePath`'s (the root's: `[]`). */
export const inputsAt =
  (path: readonly string[], sitePath: readonly string[] = []): CaseCall =>
  (document, registry, diagnostics, builder) =>
    readSourceParameters(cursorAt(document, path), registry, {
      sitePath,
      catalogBuilder: builder,
      diagnostics,
    });

/** Read the agent variables at `path` as the loader reads them, under `sitePath`. */
export const variablesAt =
  (path: readonly string[], sitePath: readonly string[]): CaseCall =>
  (document, registry, diagnostics, builder) =>
    readSourceParameters(cursorAt(document, path), registry, {
      sitePath,
      catalogBuilder: builder,
      diagnostics,
      fieldName: "variables",
      declarationLabel: "variable",
      requireDefault: true,
    });

/** Parse the flow's `nodes:` as `assemble` does (`parseFile`, `flowNodesCursor`, `parseNodes`). */
export const parsedNodes: CaseCall = (document) =>
  parseNodes(flowNodesCursor(document, parseFile(document).nodes));

/** Child flows by callable name, standing in for the `defs:` entries the loader would compose. */
export type ChildFlows = Readonly<Record<string, ComposedFlow>>;

/**
 * Return `scope` with a thin lookup of hand-built `children` in front of its `resolve`: a name in
 * `children` resolves to that flow; any other goes to the real scope.
 */
function resolveFrom(scope: CallableScope, children: ChildFlows = {}): AssemblyScope {
  return {
    catalogBuilder: scope.catalogBuilder,
    registerPrivate: (flowId, owner, label, build) => scope.registerPrivate(flowId, owner, label, build),
    resolve: (flowId) => (Object.hasOwn(children, flowId) ? children[flowId]! : scope.resolve(flowId)),
  };
}

/**
 * Assemble the root flow of `document` with the loader's own `assemble` options
 * (`rootAssembleOptions`) and `llm_config:` reader, through the loader's `CallableScope` over the
 * file's `defs:` (no `uses:`), whose catalog is `builder`, with the hand-built `children` (default
 * none) looked up in front of it. Defs resolve lazily, as calls reach them. The loader's `uses:`
 * aliases, root memories and CODE checks are left out.
 *
 * @param diagnostics The case's factory for located load errors.
 * @param inputParameters The flow's inputs as an earlier pass resolved them, read in place of its
 *   `input:` section; `null` (the default) reads the section.
 */
function assembleRoot(
  document: SourceDocument,
  registry: TypeRegistry,
  builder: SourceCatalogBuilder,
  diagnostics: LoadDiagnostics,
  children: ChildFlows = {},
  inputParameters: readonly Parameter[] | null = null,
): ComposedFlow {
  const file = parseFile(document);
  const root = document.root;
  const realScope = new CallableScope(file.defs, registry, null, {}, {
    defsCursor: root.has("defs") ? root.child("defs") : null,
    catalogBuilder: builder,
    diagnostics,
  });
  const flowLlmConfig = readAuthoredLlmConfig(file.llmConfig, "flow", {
    registry,
    sitePath: ["root", "llm_config"],
    catalogBuilder: builder,
    diagnostics,
    cursor: root,
  });
  const options = rootAssembleOptions(document, file, {
    registry,
    scope: resolveFrom(realScope, children),
    flowLlmConfig,
  });
  if (inputParameters === null) return assemble(options);
  return assemble({ ...options, inputsSection: null, inputsCursor: null, inputParameters });
}

/** Assemble the case's root flow as the loader calls `assemble` on it. */
export const assembled: CaseCall = (document, registry, diagnostics, builder) =>
  assembleRoot(document, registry, builder, diagnostics);

/**
 * Assemble the case's root flow as `assembled` does, with a scope that resolves a callable named
 * in `children` to that hand-built flow, for a case whose CALL, MAP or LOOP calls a `defs:` entry.
 */
export const assembledWith =
  (children: ChildFlows): CaseCall =>
  (document, registry, diagnostics, builder) =>
    assembleRoot(document, registry, builder, diagnostics, children);

/**
 * Assemble flow `text`'s root as `assembled` does, returning the flow and the catalog builder
 * its origins resolve through (`builder.freeze()` or `builder.resolveRegistered(origin)`).
 *
 * @param label The file name diagnostics name (default `flow.yaml`).
 * @param options.children Hand-built child flows the scope resolves by name, as `assembledWith`'s.
 * @param options.inputParameters The flow's inputs, read in place of its `input:` section.
 */
export function assembleText(
  text: string,
  label = "flow.yaml",
  options: { children?: ChildFlows; inputParameters?: readonly Parameter[] } = {},
): { flow: ComposedFlow; builder: SourceCatalogBuilder } {
  const document = parseSourceDocument(text, label);
  const builder = new SourceCatalogBuilder();
  return {
    flow: assembleRoot(
      document,
      caseRegistry(document),
      builder,
      LoadDiagnostics.fromDocument(document),
      options.children,
      options.inputParameters,
    ),
    builder,
  };
}

/**
 * Make `row.call` on its case and compare the result with the case's `expect`: an `ok` case
 * must return; otherwise the call must throw a `LoadError` whose message (after the row's
 * prefix; skipped when the expected diagnostic has none), last frame label and last frame range
 * are the expected diagnostic's. With `row.full`, every frame (label, range and note, in order)
 * and the notes must be the expected diagnostic's, an omitted `notes` or frame `note` reading as
 * empty, as the conformance spec writes them.
 */
export function runCase(row: CaseRow): void {
  const { case_, document } = caseDocument(row.id, { file: row.file });
  const registry = caseRegistry(document);
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const builder = new SourceCatalogBuilder();
  expectCaseResult(case_, () => row.call(document, registry, diagnostics, builder), row);
}

/** One harvested conformance case whose error comes from parsing its text, and that call. */
export interface TextCaseRow {
  /** The case, `group/name`. */
  id: string;
  /** Parse the case's `text`, labelled `label`, as the loader does. */
  call: (text: string, label: string) => unknown;
  /** A file of the case other than its flow. */
  file?: string;
  /** The loader's prefix on the expected message. */
  prefix?: string;
  /** The TS message, where the case leaves it out but the text is yian's. */
  message?: string;
}

/** Make `row.call` on its case's text and compare the result as `runCase` does. */
export function runTextCase(row: TextCaseRow): void {
  const { case_, text, label } = caseText(row.id, { file: row.file });
  expectCaseResult(case_, () => row.call(text, label), row);
}

/**
 * Compare `run`'s result with `case_`'s `expect`, as `runCase` describes; `message`, when given,
 * is the message the error must have where the expected diagnostic has none.
 */
function expectCaseResult(
  case_: Case,
  run: () => unknown,
  options: { prefix?: string; message?: string; full?: boolean },
): void {
  const expected = (case_["expect"] as Case)["load"];
  if (expected === "ok") {
    run();
    return;
  }
  const error = loadError(run);
  const diagnostic = ((expected as Case)["diagnostics"] as Case[])[0]!;
  const frames = error.diagnostic.frames.map(({ label, range, note }) => ({
    label,
    range: [range.startLine, range.startCol, range.endLine, range.endCol],
    note,
  }));
  const expectedFrames = ((diagnostic["frames"] as Case[] | undefined) ?? []).map((frame) => ({
    label: frame["label"] as string,
    range: (frame["range"] as bigint[]).map(Number),
    note: (frame["note"] as string | undefined) ?? null,
  }));
  if (options.full) {
    expect(frames).toEqual(expectedFrames);
    expect(error.diagnostic.notes).toEqual((diagnostic["notes"] as string[] | undefined) ?? []);
  } else {
    const { label, range } = frames.at(-1)!;
    const last = expectedFrames.at(-1)!;
    expect([label, range]).toEqual([last.label, last.range]);
  }
  const message = diagnostic["message"] as string | undefined;
  if (message === undefined) {
    if (options.message !== undefined) expect(error.message).toBe(options.message);
    return;
  }
  expect(options.message).toBeUndefined();
  const prefix = options.prefix ?? "";
  expect(message.startsWith(prefix)).toBe(true);
  expect(error.message).toBe(message.slice(prefix.length));
}
