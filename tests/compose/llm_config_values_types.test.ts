/** The sources of `llm_config` values must be typed, and their types must fit the field. */

import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "../helpers/declarations.ts";

const TYPEDEFS = `typedefs:
  WithKey:
    model: str
    api_key: str
  WithInherit:
    model: str
    inherit: bool
  WithBogus:
    model: str
    bogus: int
  WithUrl:
    model: str
    base_url: str
  WithHeaders:
    model: str
    headers: object
  AzureUrl:
    provider: Literal(azure)
    model: str
    base_url: str
  Fine:
    provider: Optional(str)
    model: str
  BadBlock:
    bogus: int
  GoodBlock:
    max_tokens: int
  BadProvider: Literal(nope, other)
  SomeProvider: Literal(openai, nope)
`;

const INPUTS = `input:
  s: !str
  i: !int
  b: !int
  bad_p: !BadProvider
  some_p: !SomeProvider
  key: !WithKey
  inh: !WithInherit
  bogus: !WithBogus
  url: !WithUrl
  headers: !WithHeaders
  az: !AzureUrl
  fine: !Fine
  obj: !object
  bad_block: !BadBlock
  good_block: !GoodBlock
`;

/** Return a flow whose agent `chat` has the node lines `llmLines` (indented 4). */
function flow(llmLines: string, options: { extraNodes?: string; flowLayer?: string } = {}): string {
  const { extraNodes = "", flowLayer = "" } = options;
  return `id: f
name: f
${TYPEDEFS}${INPUTS}${flowLayer}nodes:
${extraNodes}  chat:
    kind: agent
    prompt: Go.
${llmLines}output: \${chat.output}
`;
}

const loads = (text: string): void => {
  assembleText(text, "f.yaml");
};

const errorOf = (text: string): LoadError => loadError(() => assembleText(text, "f.yaml"));

const message = (text: string): string => errorOf(text).diagnostic.message;

/** Return the start line of the innermost frame of `text`'s load error. */
const line = (text: string): number => errorOf(text).diagnostic.frames.at(-1)!.range.startLine;

/** Return the 1-based line number of the first line containing `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((row) => row.includes(needle)) + 1;
}

const UNTYPED = `  pick:
    kind: code
    code: |
      return "gpt-x"
`;

test("an untyped producer must be tagged", () => {
  const text = flow("    llm_config: {model: '${pick.output}'}\n", { extraNodes: UNTYPED });
  const got = message(text);
  expect(got.startsWith("node 'chat' llm_config.model: ")).toBe(true);
  expect(got).toContain("tag it");
  expect(line(text)).toBe(lineOf(text, "${pick.output}"));
  loads(flow("    llm_config: {model: !str '${pick.output}'}\n", { extraNodes: UNTYPED }));
});

test("an int into model is disjoint", () => {
  const got = message(flow("    llm_config: {model: '${input.i}'}\n"));
  expect(got.startsWith("node 'chat' llm_config.model: ")).toBe(true);
  expect(got).toContain("must be str");
  expect(got).toContain("int");
});

test("a provider with no known member fails naming them", () => {
  const got = message(flow("    llm_config: {provider: '${input.bad_p}'}\n"));
  expect(got.startsWith("node 'chat' llm_config.provider: ")).toBe(true);
  expect(got).toContain("nope");
  expect(got).toContain("other");
});

test("a provider with a known member or any text loads", () => {
  loads(flow("    llm_config: {provider: '${input.some_p}'}\n"));
  loads(flow("    llm_config: {provider: '${input.s}'}\n"));
});

test("a block field takes its annotation type", () => {
  const template = (source: string) =>
    `    llm_config: {anthropic: {thinking: {type: enabled, budget_tokens: '${source}'}}}\n`;
  const got = message(flow(template("${input.s}")));
  expect(got.startsWith("node 'chat' llm_config.anthropic.thinking.budget_tokens: ")).toBe(true);
  expect(got).toContain("must be int");
  // The bound (at least 1024) is checked when the value resolves.
  loads(flow(template("${input.b}")));
});

test.each([
  ["${input.key}", "api_key"],
  ["${input.url}", "base_url"],
  ["${input.headers}", "headers"],
  ["${input.inh}", "inherit"],
  ["${input.bogus}", "bogus"],
])("a whole record type must declare only llm_config fields: %s", (source, field) => {
  const got = message(flow(`    llm_config: ${source}\n`));
  expect(got.startsWith("node 'chat' llm_config: ")).toBe(true);
  expect(got).toContain(`\`${field}\``);
});

test("a whole record type declaring base_url beside provider azure names the value rule", () => {
  expect(message(flow("    llm_config: ${input.az}\n"))).toBe(
    "node 'chat' llm_config: its type declares `base_url`; llm_config.base_url can't come from a value: it " +
      "decides where the key is sent, which stays with the host or `yian auth`; write the Azure resource " +
      "endpoint as literal text",
  );
});

test("a whole record of allowed fields or object loads", () => {
  loads(flow("    llm_config: ${input.fine}\n"));
  loads(flow("    llm_config: ${input.obj}\n"));
});

test("a whole record that is not a record fails", () => {
  expect(message(flow("    llm_config: ${input.s}\n")).startsWith("node 'chat' llm_config: ")).toBe(true);
});

test("a whole block type must declare only block fields", () => {
  const got = message(flow("    llm_config: {anthropic: '${input.bad_block}'}\n"));
  expect(got.startsWith("node 'chat' llm_config.anthropic: ")).toBe(true);
  expect(got).toContain("`bogus`");
  loads(flow("    llm_config: {anthropic: '${input.good_block}'}\n"));
});

test("a constructor fails", () => {
  const got = message(flow("    llm_config: {model: '${Fine(input.fine).model}'}\n"));
  expect(got.startsWith("node 'chat' llm_config.model: ")).toBe(true);
  expect(got).toContain("Fine(...)");
});

test.each([
  ["    llm_config: {model: '${isinstance(input.s, str)}'}\n", "", "node 'chat'"],
  ["    llm_config: {model: m0}\n", "llm_config: {model: '${isinstance(input.s, str)}'}\n", "flow"],
])("isinstance fails: %#", (llmLines, flowLayer, subject) => {
  const got = message(flow(llmLines, { flowLayer }));
  expect(got.startsWith(`${subject} llm_config.model: `)).toBe(true);
  expect(got).toContain("isinstance is not supported");
});

test.each([
  ["node", "    llm_config: {model: !str '${input.i}'}\n", "", "node 'chat' llm_config.model"],
  ["flow-carried", "", "llm_config: {model: !str '${input.i}'}\n", "flow llm_config.model"],
  [
    "flow-shielded",
    "    llm_config: {model: m0}\n",
    "llm_config: {model: !str '${input.i}'}\n",
    "flow llm_config.model",
  ],
  ["whole-record", "    llm_config: !Fine ${input.s}\n", "", "node 'chat' llm_config"],
])("a tag the source cannot fit fails at the field: %s", (_id, llmLines, flowLayer, subject) => {
  const text = flow(llmLines, { flowLayer });
  const got = message(text);
  expect(got.startsWith(`${subject}: declared `)).toBe(true);
  expect(got).not.toContain("__llm_config");
  expect(line(text)).toBe(lineOf(text, "${input."));
});

test("a flow layer value no agent carries is still checked", () => {
  const text = flow("    llm_config: {model: m0}\n", { flowLayer: "llm_config: {model: '${input.i}'}\n" });
  const got = message(text);
  expect(got.startsWith("flow llm_config.model: ")).toBe(true);
  expect(got).toContain("must be str");
  expect(line(text)).toBe(lineOf(text, "${input.i}"));
  const unknown = flow("    llm_config: {model: m0}\n", { flowLayer: "llm_config: {model: '${input.nope}'}\n" });
  expect(message(unknown)).toContain("unknown flow input 'nope'");
});

test.each([
  ["    llm_config: !WithInherit ${input.obj}\n", ""],
  ["    llm_config: {model: m0}\n", "llm_config: ${input.inh}\n"],
  ["    llm_config: {model: m0}\n", "llm_config: !WithInherit ${input.obj}\n"],
])("a whole record type declaring inherit is refused: %#", (llmLines, flowLayer) => {
  const got = message(flow(llmLines, { flowLayer }));
  expect(got).toContain("llm_config");
  expect(got).toContain("`inherit`");
});

test.each<CaseRow>(
  [
    "a_block_field_takes_its_annotation_type",
    "a_constructor_fails",
    "a_tag_the_source_cannot_fit",
    "a_whole_record_that_is_not_a_record",
    "an_int_into_model_is_disjoint",
    "an_uncarried_flow_value_is_checked",
    "an_uncarried_flow_value_unknown_input",
    "an_untyped_producer_must_be_tagged",
    "isinstance_fails",
    "provider_with_no_known_member",
    "whole_block_type_with_a_bogus_field",
    "whole_record_type_declaring_inherit",
    "whole_record_type_with_a_bogus_field",
  ].map((name) => ({ id: `load-agent/compose_llm_config_values_types__${name}`, call: assembled })),
)("assemble: $id", runCase);
