/**
 * Import scoping: a `uses:` alias prefixes every reachable memory key. A port of
 * `tests/engine/test_memory_import_scope.py`.
 *
 * Ported: every row. Changed: CODE references are TS's form
 * (`./fixtures/compose_codefns.ts:echo_x`), resolved against the conformance directory: a flow
 * loaded from text names it as `baseDir`, and each temporary directory links `fixtures` to the
 * conformance fixtures, since a file read through the search path resolves its CODE against its
 * own directory. Python's `copy.copy` of a child flow is a new object with the same prototype and
 * fields; its `SimpleNamespace` stand-in for the loaded flow is a plain object carrying
 * `originResolver`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { resolveMemoryLayout, scopeMemoryDeclarations } from "../../src/compile/memory_layout.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow, memorySetupLoadError } from "../../src/compose/loader.ts";
import { DiagnosticFrame, OriginRef, type OriginResolver, SourceRange } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl, MemorySetupError } from "../../src/memory/contracts.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";

import { CONFORMANCE_DIR, linkConformanceFixtures, loadErrorAsync } from "../compose/declarations.ts";

const CODE = "./fixtures/compose_codefns.ts:echo_x";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-memory-import-"));
  linkConformanceFixtures(tmp);
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Return a library flow `lib` declaring `memory`, whose node `a` is a CALL of `call` or a CODE leaf. */
function lib(memory = "{notes: episodic}", options: { call?: string; uses?: Record<string, string> } = {}): string {
  const { call, uses } = options;
  const lines = ["id: lib", "name: lib", `long_term_memory: ${memory}`, "input:", "  x: !str", "nodes:"];
  if (call !== undefined) lines.push(`  a: {kind: call, call: ${call}, input: {x: "\${input.x}"}}`);
  else lines.push(`  a: {kind: code, code: ${CODE}, input: {x: "\${input.x}"}, output: str}`);
  lines.push('output: {y: "${a.output}"}');
  if (uses !== undefined) {
    lines.push("uses:");
    for (const [alias, target] of Object.entries(uses)) lines.push(`  ${alias}: ${target}`);
  }
  return lines.join("\n") + "\n";
}

/** Return a root flow calling `calls` (node id to callable) through the `uses:` aliases. */
function root(calls: Record<string, string>, uses: Record<string, string>, memory: string | null = null): string {
  const lines = ["id: root", "name: root"];
  if (memory !== null) lines.push(`long_term_memory: ${memory}`);
  lines.push("input:", "  x: !str", "nodes:");
  for (const [node, target] of Object.entries(calls)) {
    lines.push(`  ${node}: {kind: call, call: ${target}, input: {x: "\${input.x}"}}`);
  }
  lines.push('output: {y: "${input.x}"}');
  lines.push("uses:");
  for (const [alias, target] of Object.entries(uses)) lines.push(`  ${alias}: ${target}`);
  return lines.join("\n") + "\n";
}

/** Write `text` to `<tmp>/<name>.yaml`. */
function write(name: string, text: string): void {
  writeFileSync(join(tmp, `${name}.yaml`), text);
}

/** Return the memory keys of `flow`'s layout, in table order. */
function keys(flow: ComposedFlow): string[] {
  return resolveMemoryLayout(flow.compiled).memories.map((memory) => memory.key);
}

/** The baked child of `flow`'s CALL node `nodeId`. */
const childOf = (flow: CompiledFlow, nodeId: string) => (flow.nodes.get(nodeId) as CallNode).child as CompiledFlow;

test("scope rewrite prefixes every reachable declaration", async () => {
  const flow = await loadFlow(
    "id: outer\nname: outer\n" +
      "long_term_memory: {notes: episodic}\n" +
      "llm_config: {provider: openai, model: fake-model}\n" +
      "defs:\n" +
      "  child:\n" +
      "    long_term_memory: {facts: reflective}\n" +
      "    nodes:\n" +
      "      ask:\n" +
      "        kind: agent\n" +
      "        long_term_memory: {chat: episodic}\n" +
      "        prompt: hi\n" +
      "        output: str\n" +
      "    output: ${ask.output}\n" +
      "nodes:\n" +
      "  one: {kind: call, call: child}\n" +
      "  two: {kind: call, call: child}\n" +
      "output: ${one.output}\n",
  );
  const compiled = flow.compiled;
  scopeMemoryDeclarations(compiled, "lib");
  expect(compiled.memories.map((decl) => decl.site)).toEqual([new DeclSite(["lib"], ["root"], null)]);
  const child = childOf(compiled, "one");
  expect(child.memories.map((decl) => decl.key)).toEqual(["lib.facts"]);
  expect((child.nodes.get("ask") as AgentNode).memories.map((decl) => decl.key)).toEqual(["lib.chat"]);
  // Whether or not the two call sites share one child object, each declaration gains the alias
  // exactly once.
  const other = childOf(compiled, "two");
  expect(other.memories.map((decl) => decl.key)).toEqual(["lib.facts"]);
  expect((other.nodes.get("ask") as AgentNode).memories.map((decl) => decl.key)).toEqual(["lib.chat"]);
  expect(keys(flow).sort()).toEqual(["lib.chat", "lib.facts", "lib.notes"]);
});

test("scope rewrite visits a shared node once", async () => {
  const flow = await loadFlow(
    "id: outer\nname: outer\n" +
      "llm_config: {provider: openai, model: fake-model}\n" +
      "defs:\n" +
      "  child:\n" +
      "    nodes:\n" +
      "      ask:\n" +
      "        kind: agent\n" +
      "        long_term_memory: {chat: episodic}\n" +
      "        prompt: hi\n" +
      "        output: str\n" +
      "    output: ${ask.output}\n" +
      "nodes:\n" +
      "  one: {kind: call, call: child}\n" +
      "  two: {kind: call, call: child}\n" +
      "output: ${one.output}\n",
  );
  const compiled = flow.compiled;
  // Two distinct child flow objects that hold the same AGENT node object.
  const first = childOf(compiled, "one");
  const second = Object.assign(Object.create(Object.getPrototypeOf(first)) as CompiledFlow, first);
  second.nodes = new Map(first.nodes);
  (compiled.nodes.get("two") as CallNode).child = second;
  scopeMemoryDeclarations(compiled, "lib");
  expect((first.nodes.get("ask") as AgentNode).memories.map((decl) => decl.key)).toEqual(["lib.chat"]);
});

test("two libraries with the same name stay independent", async () => {
  write("liba", lib());
  write("libb", lib());
  const text = root({ a: "liba", b: "libb" }, { liba: "liba", libb: "libb" }, "{notes: episodic}");
  const flow = await loadFlow(text, { searchPaths: [tmp] });
  expect(keys(flow)).toEqual(["liba.notes", "libb.notes", "notes"]);
});

test("one file under two aliases gives two sets", async () => {
  write(
    "lib",
    lib("null", { call: "inner" }).replace(
      "nodes:\n",
      "defs:\n" +
        "  inner:\n" +
        "    input:\n" +
        "      x: !str\n" +
        "    nodes:\n" +
        "      ask:\n" +
        "        kind: agent\n" +
        "        llm_config: {provider: openai, model: fake-model}\n" +
        "        long_term_memory: {chat: episodic}\n" +
        "        prompt: hi\n" +
        "        output: str\n" +
        "    output: ${ask.output}\n" +
        "nodes:\n",
    ),
  );
  const flow = await loadFlow(root({ a: "one", b: "two" }, { one: "lib", two: "lib" }), { searchPaths: [tmp] });
  const layout = resolveMemoryLayout(flow.compiled);
  expect(layout.memories.map((memory) => memory.key)).toEqual(["one.chat", "two.chat"]);
  expect(layout.memories.map((memory) => memory.owners)).toEqual([
    [new DeclSite(["one"], ["def", "inner"], "ask")],
    [new DeclSite(["two"], ["def", "inner"], "ask")],
  ]);
});

test("diamond import gives independent sets", async () => {
  write("d", lib());
  write("b", lib("null", { call: "d", uses: { d: "d" } }));
  write("c", lib("null", { call: "d", uses: { d: "d" } }));
  // A uses: alias may not equal a node id, so the CALL nodes are nb and nc.
  const flow = await loadFlow(root({ nb: "b", nc: "c" }, { b: "b", c: "c" }), { searchPaths: [tmp] });
  expect(keys(flow)).toEqual(["b.d.notes", "c.d.notes"]);
});

test("scoping leaves the cached child untouched", async () => {
  const library = await loadFlow(lib(), { baseDir: CONFORMANCE_DIR });
  const flow = await loadFlow(root({ a: "x", b: "y" }, { x: "lib", y: "lib" }), { childResolver: () => library });
  expect(keys(flow)).toEqual(["x.notes", "y.notes"]);
  // The resolver handed the same object to both aliases; neither rewrite reached it.
  expect(library.compiled.memories.map((decl) => decl.key)).toEqual(["notes"]);
});

test("library without memory loads under a dotted alias", async () => {
  write("lib", lib("null"));
  const flow = await loadFlow(root({ a: "my.lib" }, { "my.lib": "lib" }), { searchPaths: [tmp] });
  expect(keys(flow)).toEqual([]);
});

test("malformed imported graph is a load error", async () => {
  const library = await loadFlow(lib(), { baseDir: CONFORMANCE_DIR });
  // A node filed under a key other than its id; the memory walk rejects it.
  library.compiled.nodes = new Map([["b", library.compiled.nodes.get("a")!]]);
  const error = await loadErrorAsync(loadFlow(root({ n: "lib" }, { lib: "lib" }), { childResolver: () => library }));
  expect(error.message).toMatch(/does not match node id 'a'/);
});

/** Return a loaded one-node flow's compiled graph carrying `decls` as its flow memories. */
async function flowWith(...decls: MemoryDecl[]): Promise<CompiledFlow> {
  const flow = (
    await loadFlow(
      "id: f\nname: f\nnodes:\n" +
        "  echo: {kind: code, code: ./fixtures/compose_codefns.ts:echo, input: {topic: hi}, output: str}\n" +
        "output: ${echo.output}\n",
      { baseDir: CONFORMANCE_DIR },
    )
  ).compiled;
  flow.memories = decls;
  return flow;
}

/** Return a root flow declaration `name` scoped under `scope`. */
function decl(name: string, scope: string[] = []): MemoryDecl {
  return new MemoryDecl(name, "episodic", {}, new DeclSite(scope, ["root"], null));
}

/** Expect `fn` to throw a `MemorySetupError` whose message matches `pattern`. */
function expectSetupError(fn: () => unknown, pattern: RegExp): void {
  let caught: unknown = null;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(MemorySetupError);
  expect((caught as Error).message).toMatch(pattern);
}

test("scope segment must match the name pattern", async () => {
  const flow = await flowWith(decl("notes", ["1lib"]));
  expectSetupError(() => resolveMemoryLayout(flow), /alias '1lib'.*\[A-Za-z_\]/);
});

test("keys equal under casefold are rejected", async () => {
  const flow = await flowWith(decl("notes"), decl("Notes"));
  expectSetupError(() => resolveMemoryLayout(flow), /'Notes' and 'notes' differ only by case/);
});

test("aliases equal under casefold are rejected", async () => {
  const flow = await flowWith(decl("notes", ["lib"]), decl("notes", ["LIB"]));
  expectSetupError(() => resolveMemoryLayout(flow), /'LIB\.notes' and 'lib\.notes'/);
});

test("dotted alias is rejected at load", async () => {
  write("lib", lib());
  const error = await loadErrorAsync(loadFlow(root({ a: "my.lib" }, { "my.lib": "lib" }), { searchPaths: [tmp] }));
  expect(error.message).toContain("'my.lib'");
});

test("dotted alias is named before a rendered key clash", async () => {
  // "a.b" + notes and "a" -> "b" + notes both render as a.b.notes.
  write("lib", lib());
  write("lib2", lib("{notes: reflective}"));
  write("mid", lib("null", { call: "b", uses: { b: "lib2" } }));
  const error = await loadErrorAsync(
    loadFlow(root({ n1: "a.b", n2: "a" }, { "a.b": "lib", a: "mid" }), { searchPaths: [tmp] }),
  );
  expect(error.message).toContain("alias 'a.b'");
  expect(error.message).not.toContain("different type or config");
});

test("imported flow memory is owned by a scoped root site", async () => {
  write("lib", lib());
  const flow = await loadFlow(root({ a: "lib" }, { lib: "lib" }, "{notes: episodic}"), { searchPaths: [tmp] });
  const owners = Object.fromEntries(
    resolveMemoryLayout(flow.compiled).memories.map((memory) => [memory.key, memory.owners]),
  );
  expect(owners).toEqual({
    "lib.notes": [new DeclSite(["lib"], ["root"], null)],
    notes: [new DeclSite([], ["root"], null)],
  });
});

test("a clash inside an import names the scoped sites", async () => {
  write(
    "lib",
    "id: lib\nname: lib\n" +
      "long_term_memory: {notes: episodic}\n" +
      "llm_config: {provider: openai, model: fake-model}\n" +
      "input:\n" +
      "  x: !str\n" +
      "nodes:\n" +
      "  ask:\n" +
      "    kind: agent\n" +
      "    long_term_memory: {notes: reflective}\n" +
      "    prompt: hi\n" +
      "    output: str\n" +
      "output: ${ask.output}\n",
  );
  const error = await loadErrorAsync(loadFlow(root({ a: "lib" }, { lib: "lib" }), { searchPaths: [tmp] }));
  const { diagnostic } = error;
  expect(diagnostic.frames.at(-1)!.label).toBe("lib.yaml");
  expect(diagnostic.message).toEndWith("at lib: root and lib: root AGENT 'ask'");
  expect(diagnostic.notes[0]).toBe("also declared at lib: root AGENT 'ask'");
});

test("a related note at the primary location is skipped", async () => {
  write("lib", lib());
  // Both aliases map one declaration in lib.yaml, so both origins share a location.
  const error = await loadErrorAsync(
    loadFlow(root({ a: "Lib", b: "lib" }, { Lib: "lib", lib: "lib" }), { searchPaths: [tmp] }),
  );
  expect(error.message).toMatch(/differ only by case/);
  expect(error.diagnostic.frames.at(-1)!.label).toBe("lib.yaml");
  expect(error.diagnostic.notes).toEqual([]);
});

test("a related note repeating an earlier location is skipped", () => {
  const lines: Record<string, number> = { p: 1, a: 2, a2: 2, b: 3 };
  const resolver: OriginResolver = {
    resolve: (origin) => {
      const line = lines[origin.key]!;
      return new DiagnosticFrame({ label: "f.yaml", text: "x\ny\nz\n", range: new SourceRange(line, 1, line, 2) });
    },
  };
  const error = new MemorySetupError("clash", {
    origin: new OriginRef("p"),
    related: [new OriginRef("a"), new OriginRef("a2"), new OriginRef("b")],
  });
  expect(memorySetupLoadError({ originResolver: resolver }, error).diagnostic.notes).toEqual([
    "related declaration at f.yaml:2:1",
    "related declaration at f.yaml:3:1",
  ]);
});
