/**
 * Agent variables declared with YAML tags (`num_sentences: !int(1..12) 4`): a port of the
 * declaration tests of `tests/compose/test_tagged_variables.py`, reading the declarations with
 * `parametersFromYaml` (`readSourceParameters`) where Python loads a flow. Ints are `bigint`.
 *
 * Ported: `test_tagged_variables_read_their_types_and_defaults`, and the declarations of
 * `test_a_tagged_variable_needs_a_default_that_fits`, read once rather than at each placement.
 *
 * TS-only: the harvested-case table, which makes the loader's call on the `variables:`
 * declarations of the conformance case this reader decides, comparing the message and the last
 * frame.
 *
 * Later (T3-15, T3-22): the agent runs at each placement
 * (`test_tagged_variables_render_their_defaults`) and the `load_flow` placements.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { readSourceParameters } from "../../src/compose/helpers/types.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import type { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import type { Case } from "../conformance.ts";
import { caseDocument, parametersFromYaml } from "./declarations.ts";

// Pinned before `REG` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

/** Return the `LoadError` `run` throws. */
function loadError(run: () => unknown): LoadError {
  try {
    run();
  } catch (error) {
    if (error instanceof LoadError) return error;
    throw error;
  }
  throw new Error("expected a LoadError");
}

const REG = readTypedefs({ Tone: "Literal(concise, detailed)" });

const VARIABLES = `\
num_sentences: !int(1..12) 4
tone: !Tone concise
instruction: !str "Summarize clearly"
`;

/** Return the shared variables with `declaration` replacing (or added to) its name. */
function withDeclaration(declaration: string): string {
  const name = declaration.split(":", 1)[0]!;
  const kept = VARIABLES.split(/(?<=\n)/).filter((line) => !line.startsWith(`${name}:`));
  return kept.join("") + declaration;
}

/** Read `variables` (unindented declarations) as agent variables. */
function variablesFrom(variables: string) {
  const text = "variables:\n" + variables.replace(/^(?=.)/gm, "  ");
  return parametersFromYaml(text, REG, { field: "variables", requireDefault: true });
}

describe("tagged variables", () => {
  test.each([
    ["x: !int\n", "variable 'x': a variable needs a default: `x: !int <default>`"],
    ["num_sentences: !int(1..12) 13\n", "'num_sentences': `13` is not a valid int(1..12)"],
    ["tone: !Tone loud\n", "'tone': `loud` is not a valid Tone"],
  ])("a tagged variable needs a default that fits: %p", (declaration, fragment) => {
    expect(loadError(() => variablesFrom(withDeclaration(declaration))).message).toContain(fragment);
  });

  test("tagged variables read their types and defaults", () => {
    const variables = variablesFrom(VARIABLES);
    expect(Object.fromEntries(variables.map((variable) => [variable.name, variable.default]))).toEqual({
      num_sentences: 4n,
      tone: "concise",
      instruction: "Summarize clearly",
    });
    expect(variables.every((variable) => variable.hasDefault && !variable.required)).toBe(true);
    expect(variables[0]!.contract!.type.bounds).toEqual([1n, 12n]);
    expect(variables[1]!.contract!.type.tags).toEqual(new Set(["concise", "detailed"]));
  });
});

// ---------- harvested conformance cases ----------

/** The call the loader makes on a case's document. */
type Call = (document: SourceDocument, registry: TypeRegistry, diagnostics: LoadDiagnostics) => unknown;

interface Row {
  /** The case, `group/name`. */
  id: string;
  call: Call;
  /** The loader's prefix on the expected message. */
  prefix?: string;
}

/** Return the cursor at `path` from the document root. */
function at(document: SourceDocument, path: string[]): SourceCursor {
  return path.reduce((cursor: SourceCursor, step) => cursor.child(step), document.root);
}

/** Read the agent variables at `path` as the loader reads them, under `sitePath`. */
const variables =
  (path: string[], sitePath: string[]): Call =>
  (document, registry, diagnostics) =>
    readSourceParameters(at(document, path), registry, {
      sitePath,
      catalogBuilder: new SourceCatalogBuilder(),
      diagnostics,
      fieldName: "variables",
      declarationLabel: "variable",
      requireDefault: true,
    });

/** Make the row's call on its case and compare the result with the case's `expect`. */
function runRow(row: Row): void {
  const { case_, document } = caseDocument(row.id);
  const root = document.root;
  const registry = root.has("typedefs") ? readTypedefs(root.mapping("typedefs")) : new Map();
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const expected = (case_["expect"] as Case)["load"];
  if (expected === "ok") {
    row.call(document, registry, diagnostics);
    return;
  }
  const error = loadError(() => row.call(document, registry, diagnostics));
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

test.each<Row>([
  {
    id: "load-types/compose_tagged_variables__a_variable_needs_a_default",
    // A compact flow's one node is named by the flow's `id:`.
    call: variables(["variables"], ["node", "tagged_variables"]),
  },
])("readSourceParameters: $id", runRow);
