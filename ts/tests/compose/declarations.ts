/**
 * Read authored `input:` or `variables:` declarations straight from YAML text: a port of
 * `tests/compose/_declarations.py`.
 *
 * TS-only: `nodesOf`, which parses a flow's `nodes:` with stand-in kinds registered, and
 * `fileOf`, which parses its top level; `caseText` and `caseDocument`, which read a conformance
 * case's flow as text or as a parsed document; and the harvested-case harness built on them
 * (`loadError`, `cursorAt`, `caseRegistry`, `CaseRow`, `runCase`, `TextCaseRow`, `runTextCase`,
 * `inputsAt`, `variablesAt`, `parsedNodes`, `assembled`, `assembledWith`, `assembleText`), which
 * makes the loader's call on a case and compares the result with the case's `expect`; and
 * `handChild`, a child flow built by hand (START, producers with given output contracts, END)
 * where a case's `defs:` entry would be loaded, for the rows that read a child's signature or
 * build a CALL or MAP over one (`assembledWith` and `assembleText`'s `children`). The harness
 * scope registers and resolves inline body children itself; the loader's callable scope, with
 * its collision and recursion checks, is T3-22's.
 */

import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";

import { expect } from "bun:test";

import { CompiledFlow, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { assemble, type CallableScope, ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { boundaryFieldRanges } from "../../src/compose/helpers/source.ts";
import { readSourceFlowInputs, readSourceParameters } from "../../src/compose/helpers/types.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
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
import { type Case, prepare, readCase } from "../conformance.ts";
import { withKinds } from "./stand_ins.ts";

const CASES = fileURLToPath(new URL("../../../conformance/cases/", import.meta.url));

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
}

/** Read the `input:` declarations at `path` as the loader reads owner `sitePath`'s (the root's: `[]`). */
export const inputsAt =
  (path: readonly string[], sitePath: readonly string[] = []): CaseCall =>
  (document, registry, diagnostics, builder) =>
    readSourceFlowInputs(cursorAt(document, path), registry, {
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

/** Return a scope `resolve` that hands back `children[flowId]`, else throws (no loader yet). */
function resolveFrom(children: ChildFlows = {}): CallableScope["resolve"] {
  return (flowId) => {
    if (Object.hasOwn(children, flowId)) return children[flowId]!;
    throw new Error("child flows load in T3-22");
  };
}

/**
 * Return a scope whose catalog is `builder` and which registers private (inline body) children:
 * `resolve` builds a registered child and memoizes only a successful build; any other callable
 * goes to `resolve`. A repeated registration keeps the first builder.
 */
function scopeOf(builder: SourceCatalogBuilder, resolve: CallableScope["resolve"]): CallableScope {
  const builders = new Map<string, () => ComposedFlow>();
  const built = new Map<string, ComposedFlow>();
  return {
    catalogBuilder: builder,
    registerPrivate: (flowId, _owner, _label, build) => {
      if (!builders.has(flowId)) builders.set(flowId, build);
    },
    resolve: (flowId, version) => {
      const flow = built.get(flowId);
      if (flow !== undefined) return flow;
      const build = builders.get(flowId);
      if (build === undefined) return resolve(flowId, version);
      const child = build();
      built.set(flowId, child);
      return child;
    },
  };
}

/**
 * Assemble the root flow of `document` as the loader calls `assemble` on it, with a scope whose
 * catalog is `builder`, which registers and resolves inline body children, and which resolves
 * any other callable through `resolve` (default: none resolves); the loader's `uses:` aliases,
 * flow `llm_config:` and root memories are left out.
 */
function assembleRoot(
  document: SourceDocument,
  registry: TypeRegistry,
  builder: SourceCatalogBuilder,
  resolve: CallableScope["resolve"] = resolveFrom(),
): ComposedFlow {
  const file = parseFile(document);
  const root = document.root;
  const scope = scopeOf(builder, resolve);
  // A compact flow (no `nodes:`) lends its top-level `output:` to its one synthesized node, so the
  // loader locates the flow's `asserts:` and `output:` only in a full flow.
  const childOf = (key: string, flowOnly = false) =>
    (!flowOnly || root.has("nodes")) && root.has(key) ? root.child(key) : null;
  return assemble({
    inputsSection: file.inputs,
    inputsCursor: childOf("input"),
    nodesCursor: flowNodesCursor(document, file.nodes),
    outputsSection: file.outputs,
    assertsSection: file.asserts,
    registry,
    scope,
    owner: ["root"],
    version: file.version,
    name: file.name,
    description: file.description,
    flowEnv: file.env,
    boundaryFieldRanges: boundaryFieldRanges(root),
    assertsCursor: childOf("asserts", true),
    outputsCursor: childOf("output", true),
    flowId: file.id,
  });
}

/** Assemble the case's root flow as the loader calls `assemble` on it. */
export const assembled: CaseCall = (document, registry, _diagnostics, builder) =>
  assembleRoot(document, registry, builder);

/**
 * Assemble the case's root flow as `assembled` does, with a scope that resolves a callable named
 * in `children` to that hand-built flow, for a case whose CALL, MAP or LOOP calls a `defs:` entry.
 */
export const assembledWith =
  (children: ChildFlows): CaseCall =>
  (document, registry, _diagnostics, builder) =>
    assembleRoot(document, registry, builder, resolveFrom(children));

/**
 * Assemble flow `text`'s root as `assembled` does, returning the flow and the catalog builder
 * its origins resolve through (`builder.freeze()` or `builder.resolveRegistered(origin)`).
 *
 * @param label The file name diagnostics name (default `flow.yaml`).
 * @param options.children Hand-built child flows the scope resolves by name, as `assembledWith`'s.
 */
export function assembleText(
  text: string,
  label = "flow.yaml",
  options: { children?: ChildFlows } = {},
): { flow: ComposedFlow; builder: SourceCatalogBuilder } {
  const document = parseSourceDocument(text, label);
  const builder = new SourceCatalogBuilder();
  return {
    flow: assembleRoot(document, caseRegistry(document), builder, resolveFrom(options.children)),
    builder,
  };
}

/**
 * Make `row.call` on its case and compare the result with the case's `expect`: an `ok` case
 * must return; otherwise the call must throw a `LoadError` whose message (after the row's
 * prefix; skipped when the expected diagnostic has none), last frame label and last frame range
 * are the expected diagnostic's.
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
  options: { prefix?: string; message?: string },
): void {
  const expected = (case_["expect"] as Case)["load"];
  if (expected === "ok") {
    run();
    return;
  }
  const error = loadError(run);
  const frame = error.diagnostic.frames.at(-1)!;
  const range = frame.range;
  const diagnostic = ((expected as Case)["diagnostics"] as Case[])[0]!;
  const expectedFrame = (diagnostic["frames"] as Case[]).at(-1)!;
  expect([frame.label, [range.startLine, range.startCol, range.endLine, range.endCol]]).toEqual([
    expectedFrame["label"] as string,
    (expectedFrame["range"] as bigint[]).map(Number),
  ]);
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
