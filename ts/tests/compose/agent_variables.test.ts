/**
 * Agent-local variable declarations, read without building the agent: a port of the
 * declaration tests of `tests/compose/test_agent_variables.py`, reading the declarations with
 * `parametersFromYaml` (`readSourceParameters`). Ints are `bigint`.
 *
 * Ported: `test_agent_variables_reject_missing_empty_or_invalid_defaults`, and the reading half
 * of `test_agent_build_resolves_typed_variables_without_input_bindings` (the defaults; a float
 * slot keeps `compatible`'s int `4n`).
 *
 * TS-only: `readVariables` on its own, which `AgentDescriptor` calls for a node with no
 * authored source; and the harvested-case table, which makes the loader's call on the
 * `variables:` declarations of the conformance cases this reader decides, comparing the message
 * and the last frame.
 *
 * Later (T3-15): the `AgentDescriptor` build tests
 * (`test_agent_variables_reject_malformed_mappings_and_names`, the variable declaration reader,
 * the build half of `test_agent_build_resolves_typed_variables_without_input_bindings`) and the
 * `load_flow` tests (the prompt naming inputs and variables, the variable-input collision, and
 * the ranges of an invalid default and name).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { readSourceParameters, readVariables } from "../../src/compose/helpers/types.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import type { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import type { Case } from "../conformance.ts";
import { caseDocument, parametersFromYaml } from "./declarations.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const EMPTY: TypeRegistry = new Map();

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

/** Read `declarations` (indented `name: !T default` lines) as agent variables. */
function yamlVariables(declarations: string, registry: TypeRegistry = EMPTY): Parameter[] {
  return parametersFromYaml("variables:\n" + declarations, registry, {
    field: "variables",
    requireDefault: true,
  });
}

describe("tagged agent variables", () => {
  test("typed variables read their defaults", () => {
    const registry = readTypedefs({
      Settings: { limit: "int(1..12)", tags: "list(str)" },
      Tone: "Literal(concise, detailed, technical)",
    });
    const variables = yamlVariables(
      "  num_sentences: !int(1..12) 4\n" +
        "  temperature: !float(0..1) 0.4\n" +
        "  tone: !Tone concise\n" +
        '  instruction: !str "Summarize clearly"\n' +
        "  settings: !Settings {limit: 4, tags: [a, b]}\n" +
        "  items: !list(int(1..12)) [1, 4]\n" +
        "  optional: !Optional(str) null\n" +
        '  empty: !str ""\n' +
        "  compatible: !float(0..4) 4\n" +
        "  literal: !str ${input.topic}\n",
      registry,
    );
    expect(variables.every((variable) => variable.constructor === Parameter)).toBe(true);
    expect(variables.every((variable) => variable.contract?.constructor === ValueContract)).toBe(true);
    expect(variables.every((variable) => variable.hasDefault && !variable.required)).toBe(true);
    expect(Object.fromEntries(variables.map((variable) => [variable.name, variable.default]))).toEqual({
      num_sentences: 4n,
      temperature: 0.4,
      tone: "concise",
      instruction: "Summarize clearly",
      settings: { limit: 4n, tags: ["a", "b"] },
      items: [1n, 4n],
      optional: null,
      empty: "",
      compatible: 4n,
      literal: "${input.topic}",
    });
  });

  test.each([
    "!int",
    "!Optional(str)",
    "!int ''",
    "!int 4.5",
    '!int "4"',
    "!int(1..12) 13",
    "!float(0..1) 1.1",
    "!AB c",
    "!str null",
    "!int true",
    "!list(int) [1, wrong]",
    "{field: str}",
    "4",
  ])("a variable declared %p is rejected", (declaration) => {
    const registry = readTypedefs({ AB: "Literal(a, b)" });
    loadError(() => yamlVariables(`  value: ${declaration}\n`, registry));
  });
});

describe("readVariables", () => {
  // Each row is wrapped, since `test.each` spreads an array row into the arguments.
  test.each([[null], [[]], ["x"], [4n]])("%p is not a mapping", (variables) => {
    expect(loadError(() => readVariables(variables, EMPTY)).message).toBe("variables must be a mapping");
  });

  test("an empty name is rejected", () => {
    expect(loadError(() => readVariables({ "": "int" }, EMPTY)).message).toBe(
      "variable names must be non-empty strings",
    );
  });

  test("any declaration needs a default, which a source-free one can't carry", () => {
    expect(loadError(() => readVariables({ n: "int" }, EMPTY)).message).toBe(
      "variable 'n': a variable needs a default; " +
        "source-free declarations take a type only; write defaults in YAML with a tag",
    );
  });

  test("an empty mapping declares nothing", () => {
    expect(readVariables({}, EMPTY)).toEqual([]);
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

const writerVariables = variables(["nodes", "writer", "variables"], ["node", "writer"]);

test.each<Row>([
  { id: "load-agent/compose_agent_variables__empty_variable_name", call: writerVariables },
  {
    id: "load-core/compose_input_contract_origins__agent_variable_out_of_range_compact",
    // A compact flow's one node is named by the flow's `id:`.
    call: variables(["variables"], ["node", "projected_variables"]),
  },
  {
    id: "load-core/engine_value_nesting_boundaries__agent_variable_default_over_the_limit",
    call: writerVariables,
  },
  {
    id: "load-core/compose_input_contract_origins__agent_variable_out_of_range_in_a_def",
    // A flat definition's one node is named by the definition.
    call: variables(["defs", "child", "variables"], ["def", "child", "node", "child"]),
    prefix: "defs entry 'child': ",
  },
])("readSourceParameters: $id", runRow);
