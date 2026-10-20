/**
 * Agent-local variable declarations and the AGENT build that reads them: a port of
 * `tests/compose/test_agent_variables.py`, reading declarations with `parametersFromYaml`
 * (`readSourceParameters`) and loading flows with `assembleText` where Python calls `load_flow`.
 * Ints are `bigint`.
 *
 * Ported: `test_agent_variables_reject_missing_empty_or_invalid_defaults`,
 * `test_agent_build_resolves_typed_variables_without_input_bindings` (the defaults read, then
 * built; a float slot keeps `compatible`'s int `4n`), `test_agent_variables_reject_malformed_mappings_and_names`,
 * `test_variable_reader_can_decline_a_source_free_declaration`,
 * `test_variable_reader_rejects_a_noncanonical_return_value`,
 * `test_variable_input_collision_selects_the_variable_declaration_key`,
 * `test_invalid_variable_default_selects_the_declaration_value`,
 * `test_invalid_variable_name_selects_its_declaration_key`, and
 * `test_loaded_agent_prompt_can_name_inputs_and_variables_without_default_data_edges`.
 *
 * Changed: the malformed mappings drop Python's `{1: "int"}` row: a record's keys are always
 * text, so it has no TS counterpart (`{"1": "int"}` would fail for its missing default instead),
 * and `test_invalid_variable_name_selects_its_declaration_key` covers a non-text YAML key; the
 * noncanonical returns are a record, text, a set, and lists holding a record
 * or a `Parameter` subclass, where Python's are a tuple, a dict and a list holding an `object()`.
 *
 * TS-only: `readVariables` on its own, which `AgentDescriptor` calls for a node with no
 * authored source; and the harvested-case table, which makes the loader's call on the
 * conformance cases of these declarations, comparing the message and the last frame, or
 * checking that the flow loads.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { readVariables } from "../../src/compose/helpers/types.ts";
import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { BuildContext, type VariableDeclarationReader } from "../../src/compose/nodes/base.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadError,
  parametersFromYaml,
  runCase,
  variablesAt,
} from "./declarations.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const EMPTY: TypeRegistry = new Map();

/** Read `declarations` (indented `name: !T default` lines) as agent variables. */
function yamlVariables(declarations: string, registry: TypeRegistry = EMPTY): Parameter[] {
  return parametersFromYaml("variables:\n" + declarations, registry, {
    field: "variables",
    requireDefault: true,
  });
}

/** Return a build context whose variable reader returns `parameters`. */
function reading(parameters: Parameter[], registry: TypeRegistry = EMPTY): BuildContext {
  return new BuildContext({ registry, variableDeclarationReader: () => parameters });
}

describe("tagged agent variables", () => {
  test("the agent build resolves typed variables without input bindings", () => {
    const registry = readTypedefs({
      Settings: { limit: "int(1..12)", tags: "list(str)" },
      Tone: "Literal(concise, detailed, technical)",
    });
    const read = yamlVariables(
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
    const declarations = Object.fromEntries(read.map((variable) => [variable.name, null]));
    const result = new AgentDescriptor({ id: "writer", variables: declarations }).build(reading(read, registry));
    const node = result.node as AgentNode;
    const variables = node.variables;

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
    expect(node.params).toEqual([]);
    expect(result.wiring.size).toBe(0);
    expect(node.llmConfig).toEqual({});
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

describe("the AGENT build", () => {
  // Each row is wrapped, since `test.each` spreads an array row into the arguments.
  test.each([[null], [[]], ["x"], [4n], [{ "": "int" }]])(
    "variables %p are rejected",
    (variables) => {
      loadError(() => new AgentDescriptor({ id: "writer", variables }).build(new BuildContext({ registry: EMPTY })));
    },
  );

  test("a variable reader can decline a source-free declaration", () => {
    const calls: unknown[] = [];
    const reader = (nodeId: string, declarations: unknown) => {
      calls.push([nodeId, declarations]);
      return null;
    };
    const declarations = { count: "int(1..12)" };

    // Declining hands the declarations to the source-free reader, which takes a type only, so
    // the variable has no default.
    expect(
      loadError(() =>
        new AgentDescriptor({ id: "writer", variables: declarations }).build(
          new BuildContext({ registry: EMPTY, variableDeclarationReader: reader }),
        ),
      ).message,
    ).toContain("variable 'count'");
    expect(calls).toEqual([["writer", declarations]]);
  });

  class LooseParameter extends Parameter {}

  test.each<[unknown]>([[{}], ["x"], [new Set()], [[{}]], [[new LooseParameter("x")]]])(
    "a variable reader returning %p is rejected",
    (returned) => {
      const reader = (() => returned) as unknown as VariableDeclarationReader;
      const ctx = new BuildContext({ registry: EMPTY, variableDeclarationReader: reader });

      expect(() => new AgentDescriptor({ id: "writer" }).build(ctx)).toThrow(
        new TypeError("variable declaration reader must return a list of exact Parameters or null"),
      );
    },
  );
});

describe("loaded flows", () => {
  test("a loaded agent prompt can name inputs and variables", () => {
    const text = `\
id: local-variables
name: local-variables
typedefs:
  Tone: Literal(concise, detailed)
nodes:
  writer:
    kind: agent
    input:
      topic: supplied
    variables:
      tone: !Tone concise
      instruction: !str \${missing.output}
    prompt: "\${topic}: \${tone}, \${instruction}"
    output: str
output: \${writer.output}
`;
    const compiled = assembleText(text).flow.compiled;
    const writer = compiled.nodes.get("writer") as AgentNode;

    expect(writer.params!.map((parameter) => parameter.name)).toEqual(["topic"]);
    expect(writer.variables.map((variable) => variable.name)).toEqual(["tone", "instruction"]);
    expect(writer.variables[1]!.default).toBe("${missing.output}");
    expect([...compiled.wiring.get("writer")!.keys()]).toEqual(["topic"]);
    expect(compiled.edges.every((edge) => edge.from !== "missing")).toBe(true);
    expect(loadError(() => assembleText(text.replace("${tone},", "${unknown},"))).message).toContain(
      "not a declared input or variable",
    );
  });

  test("a variable-input collision selects the variable declaration key", () => {
    const text = `\
id: variable-collision
name: variable-collision
nodes:
  writer:
    kind: agent
    input:
      topic: supplied
    variables:
      topic: !str fallback
    prompt: \${topic}
    output: str
output: \${writer.output}
`;
    const document = SourceDocument.parse(text);
    const error = loadError(() => assembleText(text));

    expect(error.message).toMatch(/variable.*topic.*input/);
    const cursor = document.root.child("nodes").child("writer").child("variables").child("topic");
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.keyRange!);
    expect(
      loadError(() =>
        new AgentDescriptor({ id: "writer", inputs: { topic: "supplied" }, variables: { topic: null } }).build(
          reading(yamlVariables("  topic: !str fallback\n")),
        ),
      ).message,
    ).toMatch(/variable.*topic.*input/);
  });

  test.each(['!int "4"', "!int(1..12) 13", "!str", "!Optional(str)"])(
    "an invalid variable default %p selects the declaration value",
    (declaration) => {
      const text = `\
id: invalid-variable-default
name: invalid-variable-default
nodes:
  writer:
    kind: agent
    variables:
      value: ${declaration}
    output: str
output: \${writer.output}
`;
      const document = SourceDocument.parse(text);
      const error = loadError(() => assembleText(text));
      const cursor = document.root.child("nodes").child("writer").child("variables").child("value");

      expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.range!);
    },
  );

  test.each([
    ["''", "names must be non-empty strings"],
    ["1", "names in a flow must be text"],
    ["true", "names in a flow must be text"],
  ])("an invalid variable name %p selects its declaration key", (key, message) => {
    const text = `\
id: invalid-variable-name
name: invalid-variable-name
nodes:
  writer:
    kind: agent
    variables:
      ${key}: !int 4
    output: str
output: \${writer.output}
`;
    const variables = SourceDocument.parse(text).root.child("nodes").child("writer").child("variables");
    const error = loadError(() => assembleText(text));

    expect(error.message).toContain(message);
    const first = Object.keys(variables.value as Record<string, unknown>)[0]!;
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(variables.child(first).keyRange!);
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-agent/compose_agent_variables__empty_variable_name", call: assembled },
  { id: "load-agent/compose_agent_variables__prompt_reads_inputs_and_variables_loads", call: assembled },
  { id: "load-agent/compose_agent_variables__variable_input_collision", call: assembled },
  { id: "load-core/compose_input_contract_origins__agent_variable_out_of_range_compact", call: assembled },
  { id: "load-core/engine_value_nesting_boundaries__agent_variable_default_over_the_limit", call: assembled },
])("assemble: $id", runCase);

test.each<CaseRow>([
  {
    id: "load-core/compose_input_contract_origins__agent_variable_out_of_range_in_a_def",
    // A flat definition's one node is named by the definition.
    call: variablesAt(["defs", "child", "variables"], ["def", "child", "node", "child"]),
    prefix: "defs entry 'child': ",
  },
])("readSourceParameters: $id", runCase);
