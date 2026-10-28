/** The `long_term_memory:` surface, the declaration parser and its loader wiring. */

import { afterEach, describe, expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { parseMemoryDeclarations } from "../../src/compose/helpers/memory.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { DeclSite, type MemoryDecl } from "../../src/memory/contracts.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { Value } from "../../src/typesys/values.ts";
import { type CaseRow, CONFORMANCE_DIR, loadErrorAsync, parsedNodes, runCase } from "../helpers/declarations.ts";

afterEach(() => forget());

const SITE = new DeclSite([], ["root"], null);

/** Parse `value` with an unlocated context, no range and `SITE`. */
function parse(value: Value): readonly MemoryDecl[] {
  return parseMemoryDeclarations(value, {
    diagnostics: new LoadDiagnostics(),
    sourceRange: null,
    site: SITE,
  });
}

/** Return each declaration's name, type, config and site. */
function shape(decls: readonly MemoryDecl[]): unknown[] {
  return decls.map((decl) => [decl.name, decl.type, decl.config, decl.site]);
}

/** Return the `LoadError` parsing `value` throws. */
function loadError(value: Value): LoadError {
  try {
    parse(value);
  } catch (error) {
    expect(error).toBeInstanceOf(LoadError);
    return error as LoadError;
  }
  throw new Error("expected a LoadError");
}

describe("parseMemoryDeclarations", () => {
  test("sugar is one memory named after its type", () => {
    const decls = parse("episodic");
    expect(shape(decls)).toEqual([["episodic", "episodic", {}, SITE]]);
    expect(decls[0]!.site).toBe(SITE);
    expect(Object.isFrozen(decls)).toBe(true);
  });

  test("mapping and long form keep authored order", () => {
    const value = {
      lessons: "reflective",
      notes: { type: "episodic", config: { max_episodes: 50n } },
      plain: { type: "vector" },
    };
    const decls = parse(value);
    expect(shape(decls)).toEqual([
      ["lessons", "reflective", {}, SITE],
      ["notes", "episodic", { max_episodes: 50n }, SITE],
      ["plain", "vector", {}, SITE],
    ]);
    expect(decls.every((decl) => decl.origin === null)).toBe(true);
  });

  test.each<[Value, string]>([
    [{}, "declares no memory"],
    [3n, "type name or a mapping"],
    [["episodic"], "type name or a mapping"],
    [{ name: "reflective", config: { max_items: 3n } }, "reserved"],
    [{ name: "reflective" }, "reserved"],
    [{ type: "episodic" }, "reserved"],
    [{ notes: { type: "episodic", extra: 1n } }, "unknown key"],
    [{ notes: { config: {} } }, "requires 'type'"],
    [{ notes: { type: 3n } }, "type must be a string"],
    [{ notes: { type: "episodic", config: [] } }, "config must be a mapping"],
    [{ notes: 3n }, "type name or a mapping"],
    [{ "9notes": "episodic" }, "memory name"],
    [{ notes: "my.type" }, "memory type"],
    ["", "memory name"],
    ["name", "reserved"],
    ["type", "reserved"],
    ["config", "reserved"],
    [{ notes: { type: "episodic", config: { path: "x.sqlite" } } }, "is not allowed"],
    [{ notes: { type: "episodic", config: { max_items: 3n } } }, "unknown key"],
  ])("malformed row %# is a load error", (value, message) => {
    const { diagnostic } = loadError(value);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toContain(message);
  });

  test.each<[Value, string]>([
    [{}, "long_term_memory: {} declares no memory; remove the key or name one"],
    [{ name: "reflective" }, "long_term_memory name 'name' is reserved (reserved: config, name, type)"],
    [{ notes: { type: "episodic", config: [] } }, "long_term_memory 'notes' config must be a mapping"],
    ["", "long_term_memory '': memory name '' must match [A-Za-z_][A-Za-z0-9_-]*"],
    [
      { notes: { type: "episodic", extra: 1n } },
      'long_term_memory \'notes\' has unknown key(s) ["extra"] (allowed: type, config)',
    ],
    [3n, "long_term_memory must be a type name or a mapping of memory names to types"],
    [
      { notes: 3n },
      "long_term_memory 'notes' must be a type name or a mapping with 'type' and optional 'config'",
    ],
    [{ notes: { config: {} } }, "long_term_memory 'notes' requires 'type'"],
    [{ notes: { type: 3n } }, "long_term_memory 'notes' type must be a string"],
  ])("pinned message row %#", (value, message) => {
    expect(loadError(value).diagnostic.message).toBe(message);
  });

  test("unknown keys are listed in code-unit order", () => {
    const value = { notes: { type: "episodic", b: 1n, Z: 1n, a: 1n } };
    expect(loadError(value).diagnostic.message).toBe(
      'long_term_memory \'notes\' has unknown key(s) ["Z","a","b"] (allowed: type, config)',
    );
  });

  test("a wrapped declaration error keeps the MemoryDecl error as its cause", () => {
    const error = loadError({ notes: "my.type" });
    // `LoadError` is a `RangeError` too, so rule it out.
    expect(error.cause).toBeInstanceOf(RangeError);
    expect(error.cause).not.toBeInstanceOf(LoadError);
    expect(error.diagnostic.message).toBe(`long_term_memory 'notes': ${(error.cause as Error).message}`);
  });

  test("a located error has the value's range as its one frame", () => {
    const document = SourceDocument.parse("long_term_memory: {}\n");
    const sourceRange = document.root.rangeOf("long_term_memory");
    expect(sourceRange).not.toBeNull();
    let caught: unknown = null;
    try {
      parseMemoryDeclarations(document.root.get("long_term_memory"), {
        diagnostics: LoadDiagnostics.fromDocument(document),
        sourceRange,
        site: SITE,
      });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(LoadError);
    const { frames } = (caught as LoadError).diagnostic;
    expect(frames).toHaveLength(1);
    expect(frames[0]!.range).toEqual(sourceRange!);
  });
});

test.each<CaseRow>([
  { id: "load-agent/engine_memory_surface__empty_agent_declaration", call: parsedNodes },
])("parseNodes: $id", runCase);

/** Return a one-CODE-node flow with `memoryBlock` written before its `nodes:`. */
function flow(memoryBlock: string): string {
  return (
    "id: surface\n" +
    "name: surface\n" +
    memoryBlock +
    "nodes:\n" +
    "  echo:\n" +
    "    kind: code\n" +
    "    code: ./fixtures/compose_codefns.ts:echo\n" +
    "    input: {topic: hello}\n" +
    "    output: str\n" +
    "output: ${echo.output}\n"
  );
}

describe("loadFlow", () => {
  test("root flow declarations are parsed and stamped", async () => {
    const loaded = await loadFlow(
      flow(
        "long_term_memory:\n" +
          "  lessons: reflective\n" +
          "  notes:\n" +
          "    type: episodic\n" +
          "    config: {max_episodes: 50}\n",
      ),
      { baseDir: CONFORMANCE_DIR },
    );
    expect(shape(loaded.compiled.memories)).toEqual([
      ["lessons", "reflective", {}, SITE],
      ["notes", "episodic", { max_episodes: 50n }, SITE],
    ]);
    const resolver = loaded.originResolver!;
    expect(loaded.compiled.memories.map((decl) => resolver.resolve(decl.origin!)!.range.startLine)).toEqual([4, 5]);
  });

  test("null flow declaration is the same as absent", async () => {
    const loaded = await loadFlow(flow("long_term_memory: null\n"), { baseDir: CONFORMANCE_DIR });
    expect(loaded.compiled.memories).toEqual([]);
  });

  test.each([
    ["long_term_memory: {}\n", "declares no memory"],
    ["long_term_memory: {name: reflective, config: {}}\n", "reserved"],
  ])("flow shape error %# points at the value", async (block, message) => {
    const error = await loadErrorAsync(loadFlow(flow(block), { baseDir: CONFORMANCE_DIR }));
    expect(error.message).toContain(message);
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(3);
  });

  test("def flow declarations carry the def owner", async () => {
    const loaded = await loadFlow(
      "id: outer\n" +
        "name: outer\n" +
        "defs:\n" +
        "  child:\n" +
        "    long_term_memory: {notes: episodic}\n" +
        "    nodes:\n" +
        "      echo:\n" +
        "        kind: code\n" +
        "        code: ./fixtures/compose_codefns.ts:echo\n" +
        "        input: {topic: child}\n" +
        "        output: str\n" +
        "    output: ${echo.output}\n" +
        "nodes:\n" +
        "  invoke: {kind: call, call: child}\n" +
        "output: ${invoke.output}\n",
      { baseDir: CONFORMANCE_DIR },
    );
    const child = (loaded.compiled.nodes.get("invoke") as CallNode).child as CompiledFlow;
    expect(shape(child.memories)).toEqual([["notes", "episodic", {}, new DeclSite([], ["def", "child"], null)]]);
  });

  test("agent declarations are parsed and stamped with the node", async () => {
    const loaded = await loadFlow(
      "id: agents\n" +
        "name: agents\n" +
        "llm_config: {provider: openai, model: fake-model}\n" +
        "nodes:\n" +
        "  ask:\n" +
        "    kind: agent\n" +
        "    long_term_memory:\n" +
        "      chat: notes\n" +
        "      facts: {type: episodic, config: {recall_limit: 2}}\n" +
        "    prompt: hi\n" +
        "    output: str\n" +
        "output: ${ask.output}\n",
    );
    const site = new DeclSite([], ["root"], "ask");
    const memories = (loaded.compiled.nodes.get("ask") as AgentNode).memories;
    expect(shape(memories)).toEqual([
      ["chat", "notes", {}, site],
      ["facts", "episodic", { recall_limit: 2n }, site],
    ]);
    const resolver = loaded.originResolver!;
    expect(memories.map((decl) => resolver.resolve(decl.origin!)!.range.startLine)).toEqual([8, 9]);
  });

  test("agent null declaration is absent", async () => {
    const text =
      "id: agents\nname: agents\n" +
      "llm_config: {provider: openai, model: fake-model}\n" +
      "nodes:\n  ask:\n    kind: agent\n    long_term_memory: null\n" +
      "    prompt: hi\n    output: str\noutput: ${ask.output}\n";
    const loaded = await loadFlow(text);
    expect((loaded.compiled.nodes.get("ask") as AgentNode).memories).toEqual([]);
  });

  test("resolver errors become located load errors", async () => {
    const twoSites =
      "id: clash\n" +
      "name: clash\n" +
      "long_term_memory: {notes: episodic}\n" +
      "llm_config: {provider: openai, model: fake-model}\n" +
      "nodes:\n" +
      "  ask:\n" +
      "    kind: agent\n" +
      "    long_term_memory: {notes: reflective}\n" +
      "    prompt: hi\n" +
      "    output: str\n" +
      "output: ${ask.output}\n";
    const { diagnostic } = await loadErrorAsync(loadFlow(twoSites, { sourceLabel: "clash.yaml" }));
    expect(diagnostic.message).toContain("memory 'notes' is declared with different type or config");
    expect(diagnostic.frames.at(-1)!.range.startLine).toBe(3);
    const notes = [...diagnostic.notes];
    expect(notes[0]).toBe("also declared at root AGENT 'ask'");
    expect(notes).toHaveLength(2);
    expect(notes[1]!).toStartWith("related declaration at clash.yaml:8:");
  });
});
