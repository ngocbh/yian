/**
 * Where a memory value may appear: whole as an agent input, rendered in its prompt text, or as a
 * child's memory input of the same type. Anything else is a load error naming those places.
 */
import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../../src/memory/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { memoryType, type Type, Type as TypeClass, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError } from "../helpers/declarations.ts";

const FILE = memoryType("FileMemory", null);
const REGISTRY = readTypedefs({ Case: { title: "str" }, Other: { name: "str" } });
const BOOL = TypeClass.scalar(ValueKind.BOOLEAN);
const STR = TypeClass.scalar(ValueKind.STRING);
const param = (name: string, typ: Type, required = true) =>
  new Parameter(name, { contract: new ValueContract(typ), required });
/** A child taking a memory `notes` of type `typ`, returning a str. */
const child = (typ: Type, required = true) =>
  handChild({ inputs: [param("notes", typ, required)], producers: { p: new ValueContract(STR) }, output: "${p.output}" });
/** A loop body taking memory `m` and a bool `done`, returning `{done}`. */
const BODY = handChild({ inputs: [param("m", FILE), param("done", BOOL)], output: { done: "${input.done}" } });
/** A loop body returning its memory field too. */
const RETURNS_M = handChild({ inputs: [param("m", FILE), param("done", BOOL)], output: { done: "${input.done}", m: "${input.m}" } });
/** A loop body taking only `done`. */
const DONE_ONLY = handChild({ inputs: [param("done", BOOL)], output: { done: "${input.done}" } });
const CHILDREN = {
  children: {
    file: child(FILE),
    optional_file: child(FILE.replace({ nullable: true }), false),
    vector: child(memoryType("VectorDBMemory", null)),
    other_cases: child(typeFor("VectorDBMemory(Other)", REGISTRY)),
    body: BODY,
    returns_m: RETURNS_M,
    done_only: DONE_ONLY,
  },
};

const HEADER = `id: f
name: f
typedefs:
  Case: {title: str}
  Other: {name: str}
input:
  lessons: !FileMemory
  maybe: !Optional(FileMemory)
  cases: !VectorDBMemory(Case)
nodes:
`;
const load = (nodes: string, output = "null") => assembleText(`${HEADER}${nodes}output: ${output}\n`, "flow.yaml", CHILDREN);
const refused = (nodes: string, output = "null") => loadError(() => load(nodes, output)).message;

const USE =
  "is a memory; use it whole as an agent input, in prompt text, as a child's memory input of the same type, " +
  "or name it in `tools:`/`tool_id:`";
const agent = (input: string, prompt = "hi") =>
  `  a:\n    kind: agent\n    output: str\n    prompt: ${JSON.stringify(prompt)}\n    input:\n${input}`;
const code = (input: string) => `  c:\n    kind: code\n    code: return 1\n    input:\n${input}`;

test("an agent takes a memory whole and renders it in its prompt and system prompt", () => {
  load(agent("      lessons: ${input.lessons}\n", "<memory name=\"lessons\">\n${lessons}\n</memory> Answer."));
  load(`${agent("      lessons: ${input.lessons}\n")}    system_prompt: "Notes: \${lessons}"\n`);
  load(agent("      maybe: ${input.maybe}\n", "${maybe}"));
  load(`  a:\n    kind: agent\n    output: str\n    input:\n      lessons: \${input.lessons}\n    prompt:\n      - user: "\${lessons}"\n      - user: hi\n`);
});

test("a memory may stay unread", () => {
  load(code("      t: 1\n"));
});

test("a named call, a loop and a map pass a memory to a child's memory input of the same type", () => {
  load(`  c:\n    kind: call\n    call: file\n    input:\n      notes: \${input.lessons}\n`);
  load(`  c:\n    kind: call\n    call: optional_file\n    input:\n      notes: \${input.maybe}\n`);
  load(`  l:\n    kind: loop\n    call: body\n    input:\n      m: \${input.lessons}\n      done: false\n    while: not done\n`);
  load(`  m:\n    kind: map\n    call: file\n    over: [1, 2]\n    input:\n      notes: \${input.lessons}\n`);
});

test.each([
  ["a field read", code("      t: ${input.lessons.text}\n"), "(a memory has no fields)"],
  ["an expression", agent("      t: ${input.lessons :- 1}\n"), "(not inside text or an expression)"],
  ["text in a binding", agent('      t: "notes: ${input.lessons}"\n'), "(not inside text or an expression)"],
  ["a case condition", `  k:\n    kind: case\n    cases:\n      - when: \${input.lessons == null}\n        then: done\n  done: !int 1\n`, "(not inside text or an expression)"],
  ["a CODE input", code("      t: ${input.lessons}\n"), "(a code node can't take a memory)"],
  ["a TOOL input", `  t:\n    kind: tool\n    tool_id: f\n    input:\n      t: \${input.lessons}\n`, "(a tool node can't take a memory)"],
  ["a record member", agent("      t:\n        a: ${input.lessons}\n"), "(not inside a list, record or dict)"],
  ["a list member", agent("      t:\n        - ${input.lessons}\n"), "(not inside a list, record or dict)"],
  ["an inline call", code("      t: call(file, notes=${input.lessons})\n"), "(pass it through a named `kind: call` node)"],
  ["a map's over", `  m:\n    kind: map\n    call: file\n    over: \${input.lessons}\n    input:\n      notes: \${input.lessons}\n`, "(a map's `over:` can't read a memory)"],
  ["a loop field the body doesn't take as a memory", `  l:\n    kind: loop\n    call: done_only\n    input:\n      m: \${input.lessons}\n      done: false\n    while: not done\n`, "(the child's `m` isn't a memory input)"],
])("refuses %s", (_label, nodes, tail) => {
  const message = refused(nodes);
  expect(message).toContain(`\${input.lessons} ${USE}`);
  expect(message).toContain(tail);
});

test("a flow output can't be a memory", () => {
  expect(refused(code("      t: 1\n"), "${input.lessons}")).toContain(`flow output from: \${input.lessons} ${USE} (a flow output can't be a memory)`);
});

test("the refusal names the site", () => {
  expect(refused(code("      t: ${input.lessons}\n"))).toContain(`node 'c' input 't' from: \${input.lessons} ${USE}`);
});

test.each([
  ["a field read", "${lessons.text}", "(a memory has no fields)"],
  ["an expression", "${lessons :- 1}", "(not inside an expression)"],
  ["a call", "${upper(lessons)}", "(not inside an expression)"],
])("a prompt refuses %s of a memory input", (_label, prompt, tail) => {
  const message = refused(agent("      lessons: ${input.lessons}\n", prompt));
  expect(message).toContain(`node 'a' prompt: \${lessons} ${USE} ${tail}`);
});

test("a prompt can't splice a memory", () => {
  const nodes = `  a:\n    kind: agent\n    output: str\n    input:\n      lessons: \${input.lessons}\n    prompt:\n      - \${lessons}\n`;
  expect(refused(nodes)).toContain(`\${lessons} ${USE} (not as a prompt splice)`);
});

test("a loop carries a memory but never returns it, and its predicate can't read one", () => {
  expect(
    refused(`  l:\n    kind: loop\n    call: returns_m\n    input:\n      m: \${input.lessons}\n      done: false\n    while: not done\n`),
  ).toContain("body output field `m` names a memory input; a loop carries a memory, never returns it");
  expect(
    refused(`  l:\n    kind: loop\n    call: body\n    input:\n      m: \${input.lessons}\n      done: false\n    while: m == null\n`),
  ).toContain("`while:` reads `m`, a memory; a loop predicate can't read a memory");
});

test("a child's memory input must be wired from its parent", () => {
  expect(refused(`  c:\n    kind: call\n    call: file\n    input: {}\n`)).toContain(
    "call node 'c': the runner binds only the top-level flow's memories; pass `notes` from the caller",
  );
  expect(refused(`  c:\n    kind: call\n    call: optional_file\n    input: {}\n`)).toContain(
    "call node 'c': the runner binds only the top-level flow's memories; pass `notes` from the caller",
  );
});

test.each([
  ["another type", "vector", "${input.lessons}", "child expects `VectorDBMemory`, got `FileMemory`"],
  ["an Optional value", "file", "${input.maybe}", "child expects `FileMemory`, got `Optional(FileMemory)`"],
  ["a different item type", "other_cases", "${input.cases}", "child expects `VectorDBMemory(Other)`, got `VectorDBMemory(Case)`"],
  ["a value that isn't a memory", "file", "notes.md", "child expects `FileMemory`, a memory; pass a memory input of this flow, as `${input.<name>}`"],
])("a child's memory input refuses %s", (_label, call, source, message) => {
  expect(refused(`  c:\n    kind: call\n    call: ${call}\n    input:\n      notes: ${JSON.stringify(source)}\n`)).toContain(message);
});

test.each([
  ["a flow assert", "asserts:\n  - ${input.lessons != null}\n", "", "${input.lessons} is a memory; an assert can't read it"],
  [
    "a call node assert on its memory local",
    "",
    "  c:\n    kind: call\n    call: file\n    input:\n      notes: ${input.lessons}\n    asserts:\n      - ${notes != null}\n",
    "${notes} is a memory; an assert can't read it",
  ],
  [
    "an llm_config value",
    "",
    "  a:\n    kind: agent\n    output: str\n    prompt: hi\n    llm_config:\n      model: ${input.lessons}\n",
    `node 'a' llm_config: \${input.lessons} ${USE}`,
  ],
])("refuses %s", (_label, top, nodes, message) => {
  const text = HEADER.replace("nodes:\n", `${top}nodes:\n`) + `${nodes || code("      t: 1\n")}output: null\n`;
  expect(loadError(() => assembleText(text, "flow.yaml", CHILDREN)).message).toContain(message);
});

/** A host's own memory type. */
class TradeJournal extends MemoryBase {
  static override readonly contract: MemoryContract = { type: "TradeJournal", tools: [] };
  override open(): Promise<MemorySession> {
    throw new Error("not opened at load");
  }
  override async reset(): Promise<void> {}
}

test("a host memory type has no constructor", async () => {
  const text = `id: f\nname: f\nnodes:\n  c:\n    kind: code\n    code: return 1\n    input:\n      t: \${TradeJournal("x")}\noutput: null\n`;
  await expect(loadFlow(text, { memoryTypes: [TradeJournal] })).rejects.toThrow(
    "`TradeJournal` is a memory type; it has no constructor",
  );
});
