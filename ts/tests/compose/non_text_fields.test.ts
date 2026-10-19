/**
 * A text or expression field YAML reads as non-text is a load error at that value: a port of
 * `tests/compose/test_non_text_fields.py`. Values are as TS YAML gives them: ints are `bigint`.
 * The flow rows read through `parseFile`, the node rows through `parseFile`, `flowNodesCursor`
 * and `parseNodes` with the registered kinds (plus stand-ins for the kinds not registered yet, for
 * the kind list).
 *
 * Changed: a flow with no `nodes:` says "missing required top-level key 'nodes'", where Python's
 * text is pydantic's. The null and text `node_name:` rows and the null optional flow field rows
 * check the parsed value, where Python loads the flow. `test_text_code_and_tool_id_still_load`
 * assembles the root flow with `assembleText`, where Python loads it.
 *
 * The `llm_config` rows (`test_non_text_llm_config_field_is_a_load_error`,
 * `test_nested_llm_config_text_field`, `test_other_llm_config_errors_are_unchanged`) load the node
 * site with `assembleText`, as the loader assembles the root flow.
 *
 * The harvested `load-core/compose_non_text_fields__*` cases are pinned at the end of each block,
 * by the call that raises their error; the inline-body ones are Later (T3-17).
 *
 * Dropped: the date rows (TS reads an unquoted date as text) and
 * `test_not_text_detail_reads_the_first_string_error` (`not_text_detail` reads a pydantic error).
 *
 * Later (T3-16): the `call:`, `on:` and loop/case rows (`test_non_text_call_is_a_load_error`,
 * `test_null_call_is_empty`, `test_text_call_still_loads`, `test_non_text_case_on_is_a_load_error`,
 * `test_null_case_on_is_absent`, `test_text_case_on_still_loads`, and the while, until,
 * searched-when, on-when, then and else rows of `test_already_located_fields_are_unchanged`).
 * Later (T3-17): the inline-body rows of `test_code_in_def_compact_merge_and_inline_body` and
 * `test_non_text_kind_in_a_def_and_inline_body`, and the `adaptive_questions.llm_config` site of
 * the `llm_config` rows. Later (T3-22): their defs, compact and merge rows, the flow and def sites
 * of the `llm_config` rows, and the def `asserts:` rows
 * (`test_non_text_def_assert_is_a_load_error`, `test_non_list_def_asserts_is_a_load_error`,
 * `test_null_def_asserts_loads`).
 */

import { afterEach, describe, expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { notTextError } from "../../src/compose/nodes/base.ts";
import { parseFile } from "../../src/compose/parse/file.ts";
import { forget } from "../../src/defaults.ts";
import {
  assembled,
  assembleText,
  fileOf,
  loadError,
  nodesOf,
  parsedNodes,
  runCase,
  runTextCase,
} from "./declarations.ts";
import { standIn, UNREGISTERED_KIND_NAMES } from "./stand_ins.ts";

afterEach(() => forget());

test("notTextError names the field", () => {
  expect(notTextError("prompt", 5n)).toBe(
    "`prompt:` must be text, but YAML read it as an integer; put it in quotes",
  );
  expect(notTextError("questions[0].question", [1n])).toBe(
    "`questions[0].question:` must be text, but YAML read it as a list; put it in quotes",
  );
  expect(notTextError("questions[0].header", null)).toBe(
    "`questions[0].header:` must be text, but it is empty; write the text",
  );
});

test("notTextError takes a fix", () => {
  expect(notTextError("kind", [1n], "write one of a, b")).toBe(
    "`kind:` must be text, but YAML read it as a list; write one of a, b",
  );
  expect(notTextError("kind", null, "write one of a, b")).toBe(
    "`kind:` must be text, but it is empty; write one of a, b",
  );
});

// ---------- the parse rows ----------

/** Non-text values with what YAML read each as; the date row drops, since TS reads it as text. */
const NON_TEXT = [
  ["5", "an integer"],
  ["0", "an integer"],
  ["true", "a boolean"],
  ["false", "a boolean"],
  ["1.5", "a float"],
  ["[a]", "a list"],
  ["[]", "a list"],
  ["{x: 1}", "a mapping"],
  ["{}", "a mapping"],
] as const;

/** `NON_TEXT` with each phrase as the message words it, plus the empty `null`. */
const NON_TEXT_OR_NULL = [
  ...NON_TEXT.map(([value, phrase]) => [value, `YAML read it as ${phrase}`] as const),
  ["null", "it is empty"] as const,
];

/** The 1-based line of `text` holding `marker` (the first match). */
function lineOf(text: string, marker: string): number {
  const index = text.split("\n").findIndex((line) => line.includes(marker));
  if (index < 0) throw new Error(`${marker} not in text`);
  return index + 1;
}

/** Return the start line of the last frame of `error`'s diagnostic. */
function startLine(error: LoadError): number {
  return error.diagnostic.frames.at(-1)!.range.startLine;
}

/** Return the `LoadError` `parseFile` throws for `text`. */
function fileError(text: string): LoadError {
  return loadError(() => parseFile(text));
}

/** A flow whose `nodes:` entry `a` has `kind: <v>`. */
function kindFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: ${v}
    prompt: hi
output: \${a.output}
`;
}

/** A compact (single-node) flow with `kind: <v>`. */
function compactKind(v: string): string {
  return `id: f\nname: f\noutput: int\nkind: ${v}\ncode: return 1\n`;
}

/** A compact (single-node) flow with `id: <v>`, or no `id:` when `v` is null. */
function compactId(v: string | null): string {
  const head = v === null ? "" : `id: ${v}\n`;
  return `${head}name: f\noutput: int\nkind: code\ncode: return 1\n`;
}

/** A flow whose code node `a` has `node_name: <v>`. */
function nodeNameFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: code
    node_name: ${v}
    output: str
    code: "return 'x'"
output: \${a.output}
`;
}

/** A flow whose CODE node `a` has `code: <v>`. */
function codeFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: code
    output: str
    code: ${v}
output: \${a.output}
`;
}

/** A flow whose TOOL node `a` has `tool_id: <v>`. */
function toolFlow(v: string): string {
  return `id: f
name: f
nodes:
  a:
    kind: tool
    tool_id: ${v}
    input: {symbol: x}
output: \${a.output}
`;
}

const FLOW = `id: f
name: f
nodes:
  a:
    kind: code
    output: str
    code: "return 'x'"
output: \${a.output}
`;

/** A flow whose top-level `<field>:` is `<v>`. */
function flowField(field: string, v: string): string {
  const text = `id: f
name: f
description: d
version: '1'
nodes:
  a:
    kind: code
    output: str
    code: "return 'x'"
output: \${a.output}
`;
  return text
    .split("\n")
    .map((line) => (line.startsWith(`${field}: `) ? `${field}: ${v}` : line))
    .join("\n");
}

describe("parseFile", () => {
  test.each(NON_TEXT_OR_NULL)("non-text compact kind is a load error: %s", (value, phrase) => {
    const text = compactKind(value);

    const error = fileError(text);

    expect(error.message).toContain(
      `compact (single-node) flow: \`kind:\` must be text, but ${phrase}; ` +
        "write one of agent, code, human_input, tool",
    );
    expect(startLine(error)).toBe(lineOf(text, "kind: "));
  });

  test.each(NON_TEXT)("non-text compact id is a load error: %s", (value, phrase) => {
    const text = compactId(value);

    const error = fileError(text);

    expect(error.message).toContain(
      `compact (single-node) flow: \`id:\` must be text, but YAML read it as ${phrase}; ` +
        "put it in quotes",
    );
    expect(startLine(error)).toBe(lineOf(text, "id: "));
  });

  test.each([
    ["null", "null"],
    ["empty", "''"],
    ["absent", null],
  ] as const)("null, empty and absent compact id are required: %s", (_, value) => {
    expect(fileError(compactId(value)).message).toContain(
      "compact (single-node) flow: a top-level `id:` is required",
    );
  });

  test.each(
    ["id", "name", "description", "version"].flatMap((field) =>
      (
        [
          ["5", "an integer"],
          ["true", "a boolean"],
          ["1.5", "a float"],
          ["[a]", "a list"],
        ] as const
      ).map(([value, phrase]) => [field, value, phrase] as const),
    ),
  )("non-text flow field is a load error: %s %s", (field, value, phrase) => {
    const text = flowField(field, value);

    const error = fileError(text);

    expect(
      error.message.startsWith(
        `\`${field}:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
      ),
    ).toBe(true);
    expect(startLine(error)).toBe(lineOf(text, `${field}: `));
  });

  test.each(["id", "name"])("null flow field: %s", (field) => {
    expect(fileError(flowField(field, "null")).message).toContain(
      `\`${field}:\` must be text, but it is empty; write the text`,
    );
  });

  test.each(["description", "version"])("null optional flow field parses: %s", (field) => {
    expect(parseFile(flowField(field, "null"))[field as "description" | "version"]).toBeNull();
  });

  test.each(NON_TEXT_OR_NULL)("non-text uses ref is a load error: %s", (value, phrase) => {
    const text = `${FLOW}uses:\n  lib: ${value}\n`;

    const error = fileError(text);

    expect(error.message).toContain(`\`uses.lib:\` must be text, but ${phrase}`);
    expect(startLine(error)).toBe(lineOf(text, "lib: "));
  });

  test.each(NON_TEXT)("non-text flow assert is a load error: %s", (value, phrase) => {
    const text = `${FLOW}asserts:\n  - "true"\n  - ${value}\n`;

    const error = fileError(text);

    expect(error.message).toContain(`\`asserts[1]:\` must be text, but YAML read it as ${phrase}`);
    expect(startLine(error)).toBe(text.split("\n").length - 1);
  });

  test("a missing nodes section names it", () => {
    // Changed: Python's text is pydantic's "validation error for ComposeFile".
    expect(fileError("id: f\nname: f\noutput: x\n").message).toBe(
      "missing required top-level key 'nodes'",
    );
  });

  test.each([
    "load-core/compose_non_text_fields__missing_nodes_keeps_the_schema_error",
    "load-core/compose_non_text_fields__non_text_compact_id",
    "load-core/compose_non_text_fields__non_text_compact_kind",
    "load-core/compose_non_text_fields__non_text_flow_assert",
    "load-core/compose_non_text_fields__non_text_flow_version",
    "load-core/compose_non_text_fields__null_compact_id",
    "load-core/compose_non_text_fields__null_compact_kind",
    "load-core/compose_non_text_fields__null_flow_id",
  ])("case %s", (id) => runTextCase({ id, call: fileOf }));
});

describe("parseNodes", () => {
  const UNREGISTERED_KINDS = UNREGISTERED_KIND_NAMES.map((kind) => standIn(kind));

  test.each(NON_TEXT)("non-text node name is a load error: %s", (value, phrase) => {
    const text = nodeNameFlow(value);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=code): \`node_name:\` must be text, but YAML read it as ${phrase}; ` +
        "put it in quotes",
    );
    expect(startLine(error)).toBe(lineOf(text, "node_name: "));
  });

  test.each([
    ["null", null],
    ["'5'", "5"],
  ])("null and text node name still parse: %s", (value, expected) => {
    expect(nodesOf(nodeNameFlow(value)).get("a")!.nodeName).toBe(expected);
  });

  test.each(NON_TEXT)("non-text kind is a load error: %s", (value, phrase) => {
    const text = kindFlow(value);

    const error = loadError(() => nodesOf(text, UNREGISTERED_KINDS));

    expect(error.message).toContain(
      `node 'a': \`kind:\` must be text, but YAML read it as ${phrase}; ` +
        "write one of agent, call, case, code, human_input, loop, map, tool, wait",
    );
    expect(startLine(error)).toBe(lineOf(text, "kind: "));
  });

  test.each([
    ["null", "add `kind:` for a node"],
    ["llm", "unknown kind"],
    ["''", "unknown kind"],
  ])("null and text kinds are unchanged: %s", (value, fragment) => {
    expect(loadError(() => nodesOf(kindFlow(value), UNREGISTERED_KINDS)).message).toContain(fragment);
  });

  test("a node's asserts must be a list of strings, located", () => {
    const text = FLOW.replace("    output: str\n", '    output: str\n    asserts: ["true", 5]\n');

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain("`asserts:` must be a list of strings");
    expect(startLine(error)).toBe(lineOf(text, "asserts: "));
  });

  test.each(NON_TEXT)("non-text code is a load error: %s", (value, phrase) => {
    const text = codeFlow(value);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=code): \`code:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
    );
    expect(startLine(error)).toBe(lineOf(text, "code: "));
  });

  test.each(NON_TEXT)("non-text tool id is a load error: %s", (value, phrase) => {
    const text = toolFlow(value);

    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=tool): \`tool_id:\` must be text, but YAML read it as ${phrase}; put it in quotes`,
    );
    expect(startLine(error)).toBe(lineOf(text, "tool_id: "));
  });

  test.each([
    [codeFlow("null"), "code", "code"],
    [toolFlow("null"), "tool", "tool_id"],
  ])("null code and tool id are empty: %#", (text, kind, key) => {
    const error = loadError(() => nodesOf(text));

    expect(error.message).toContain(
      `node 'a' (kind=${kind}): \`${key}:\` must be text, but it is empty; write the text`,
    );
    expect(startLine(error)).toBe(lineOf(text, `${key}: `));
  });

  test.each([codeFlow(`"return 'x'"`), toolFlow("'5'")])("text code and tool id still load: %#", (text) => {
    expect(() => assembleText(text)).not.toThrow();
  });

  test("tagged non-text tool id", () => {
    // `!str 5` still reads as an integer here; the tag's own error comes once it is quoted.
    expect(loadError(() => nodesOf(toolFlow("!str 5"))).message).toContain(
      "`tool_id:` must be text, but YAML read it as an integer; put it in quotes",
    );
  });

  test.each([
    ["load-core/compose_non_text_fields__non_text_kind", UNREGISTERED_KINDS],
    ["load-core/compose_non_text_fields__null_kind", []],
  ] as const)("case %s", (id, kinds) =>
    runTextCase({ id, call: (text, label) => nodesOf(text, kinds, label) }),
  );

  test.each([
    "load-core/compose_non_text_fields__node_asserts_holding_a_number",
    "load-core/compose_non_text_fields__non_text_code",
    "load-core/compose_non_text_fields__non_text_node_name",
    "load-core/compose_non_text_fields__non_text_tool_id",
    "load-core/compose_non_text_fields__null_tool_id",
  ])("case %s", (id) => runCase({ id, call: parsedNodes }));
});

describe("llm_config", () => {
  /** A flow whose agent node `a` has `llm_config: <v>`. */
  const llmOnNode = (v: string) => `id: f
name: f
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: ${v}
output: \${a.output}
`;
  const PREFIX = "node 'a': llm_config: ";

  test("a non-text llm_config field is a load error", () => {
    const text = llmOnNode("{model: 5}");

    const error = loadError(() => assembleText(text));

    expect(error.message).toContain(
      PREFIX + "`model:` must be text, but YAML read it as an integer; put it in quotes",
    );
    expect(error.message).not.toContain("validation error for LLMConfig");
    expect(startLine(error)).toBe(lineOf(text, "llm_config: "));
  });

  test("a nested llm_config text field", () => {
    const error = loadError(() => assembleText(llmOnNode("{provider: openai, openai: {user: 5}}")));

    expect(error.message).toContain(PREFIX + "`openai.user:` must be text");
  });

  test.each([
    ["{provider: 5}", "1 validation error"],
    ["{provider: azure, azure: {stop: 5}}", "2 validation errors"],
    ["{provider: azure, azure: {stop: [5]}}", "2 validation errors"],
  ])("other llm_config errors are unchanged: %s", (value, count) => {
    expect(loadError(() => assembleText(llmOnNode(value))).message).toContain(PREFIX + count + " for LLMConfig");
  });

  test.each(["load-core/compose_non_text_fields__non_text_node_llm_config_field"])("case %s", (id) =>
    runCase({ id, call: assembled }),
  );
});
