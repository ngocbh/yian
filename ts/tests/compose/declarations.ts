/**
 * Read authored `input:` or `variables:` declarations straight from YAML text: a port of
 * `tests/compose/_declarations.py`.
 *
 * TS-only: `nodesOf`, which parses a flow's `nodes:` with stand-in kinds registered, and
 * `fileOf`, which parses its top level; `caseText` and `caseDocument`, which read a conformance
 * case's flow as text or as a parsed document; and the harvested-case harness built on them
 * (`loadError`, `cursorAt`, `caseRegistry`, `CaseRow`, `runCase`, `TextCaseRow`, `runTextCase`,
 * `inputsAt`, `variablesAt`), which makes the loader's call on a case and compares the result
 * with the case's `expect`.
 */

import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";

import { expect } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { readSourceFlowInputs, readSourceParameters } from "../../src/compose/helpers/types.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
import { flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { reload } from "../../src/defaults.ts";
import type { Parameter } from "../../src/nodes/contracts.ts";
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
 * Parse flow `text`'s `nodes:` as the loader reads them (`parseSourceDocument`, `parseFile`,
 * `flowNodesCursor`, `parseNodes`), with `classes` registered as the node kinds.
 *
 * @param label The file name diagnostics name (default `<flow>`).
 * @returns The node descriptors by id.
 */
export function nodesOf(
  text: string,
  classes: readonly DescriptorClass[],
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
