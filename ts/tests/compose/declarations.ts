/**
 * Read authored `input:` or `variables:` declarations straight from YAML text: a port of
 * `tests/compose/_declarations.py`.
 *
 * TS-only: `caseDocument`, which reads a conformance case's flow as a parsed document, and the
 * harvested-case harness built on it (`loadError`, `cursorAt`, `caseRegistry`, `CaseRow`,
 * `runCase`, `inputsAt`, `variablesAt`), which makes the loader's call on a case and compares
 * the result with the case's `expect`.
 */

import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";

import { expect } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { readSourceFlowInputs, readSourceParameters } from "../../src/compose/helpers/types.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { reload } from "../../src/defaults.ts";
import type { Parameter } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { type Case, prepare, readCase } from "../conformance.ts";

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
 * Read conformance case `id` (`group/name`) as the TS runner runs it: the prepared case and its
 * flow parsed as `flow.yaml` (another file of the case with `file`).
 *
 * The case's `env:` is applied with `reload` before the flow parses, since parsing and typedef
 * reading use the configured limits; this changes the process's defaults, and the caller
 * restores them with `forget()`, e.g. in `afterEach`.
 *
 * @throws Error when the case has no such flow or file text.
 */
export function caseDocument(
  id: string,
  options: { file?: string } = {},
): { case_: Case; document: SourceDocument } {
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
  const document = SourceDocument.parse(text, file === undefined ? "flow.yaml" : basename(file));
  return { case_, document };
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
export function cursorAt(document: SourceDocument, path: readonly string[]): SourceCursor {
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
 * prefix), last frame label and last frame range are the expected diagnostic's.
 */
export function runCase(row: CaseRow): void {
  const { case_, document } = caseDocument(row.id, { file: row.file });
  const registry = caseRegistry(document);
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const builder = new SourceCatalogBuilder();
  const expected = (case_["expect"] as Case)["load"];
  if (expected === "ok") {
    row.call(document, registry, diagnostics, builder);
    return;
  }
  const error = loadError(() => row.call(document, registry, diagnostics, builder));
  const frame = error.diagnostic.frames.at(-1)!;
  const range = frame.range;
  const diagnostic = ((expected as Case)["diagnostics"] as Case[])[0]!;
  const expectedFrame = (diagnostic["frames"] as Case[]).at(-1)!;
  const message = diagnostic["message"] as string;
  const prefix = row.prefix ?? "";
  expect(message.startsWith(prefix)).toBe(true);
  expect([
    error.message,
    frame.label,
    [range.startLine, range.startCol, range.endLine, range.endCol],
  ]).toEqual([
    message.slice(prefix.length),
    expectedFrame["label"] as string,
    (expectedFrame["range"] as bigint[]).map(Number),
  ]);
}
