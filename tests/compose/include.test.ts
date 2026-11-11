/**
 * `!include PATH` reads a string value from a UTF-8 text file, the path relative to the including
 * file's folder: the text, its final newline kept and its line endings read as LF, is the value,
 * as if written there quoted. It is refused on a key, a list, a mapping and a merge source, and in
 * a type position; a file that is missing, not a regular file, not UTF-8, over
 * `YIAN_MAX_INCLUDE_BYTES` or outside the code root fails at the tag. The flow's source records
 * each included file's text unchanged, and a load from the source reads it from there, never from
 * disk.
 *
 * `!include(T) PATH` reads the same file as a value of type `T`, wherever a slot takes `!T value`:
 * `!include(str)` the text exactly, never a template, and any other `T` the text as one plain YAML
 * value (no tags, anchors, aliases, merge keys, repeated keys or second document), checked against
 * `T` as an inline literal is, its strings never templates. Each error is at the tag and names the
 * file.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, realpathSync, renameSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import fsp from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow, loadFlowSource } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../../src/memory/base.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { runFlow } from "../../src/run.ts";
import type { FlowSource } from "../../src/runtime/log.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";
import { NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

let tmp: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-include-")));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Write each of `files` at its path under `tmp`. */
function write(files: Readonly<Record<string, string | Uint8Array>>): void {
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(dirname(join(tmp, path)), { recursive: true });
    writeFileSync(join(tmp, path), text);
  }
}

/** Load `text` as `flow.yaml` in `dir` (default `tmp`), as the CLI does. */
function load(text: string, options: { dir?: string; codeRoot?: string; env?: Env } = {}): Promise<ComposedFlow> {
  const { dir = tmp, codeRoot = null, env = NO_ENV } = options;
  return loadFlow(text, { searchPaths: [dir], sourceLabel: "flow.yaml", codeRoot, env });
}

/** A flow with no nodes whose `output:` is `output`. */
const outputFlow = (output: string) => `id: f\nname: f\nnodes: {}\noutput: ${output}\n`;

/** Run `flow` with no inputs and return its output. */
async function outputOf(flow: ComposedFlow): Promise<unknown> {
  const result = await runFlow(flow, {}, { env: NO_ENV, numWorkers: 0 });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

/** A flow whose agent `answer` reads `question`, with `prompts` (YAML lines) under the node. */
const agentFlow = (prompts: string) =>
  "id: f\nname: f\nllm_config: {provider: openai, model: gpt-test}\ninput:\n  question: !str\n" +
  "nodes:\n  answer:\n    kind: agent\n    input:\n      question: ${input.question}\n" +
  `${prompts}    output: str\noutput: \${answer.output}\n`;

/** The message, and the last frame's file and first line, of the `LoadError` `promise` rejects with. */
async function failure(promise: Promise<unknown>): Promise<{ message: string; label: string; line: number }> {
  const error = await loadErrorAsync(promise);
  const frame = error.diagnostic.frames.at(-1)!;
  return { message: error.message, label: frame.label, line: frame.range!.startLine };
}

// ---------- the included text ----------

test("a prompt and a system prompt read from files reach the model request", async () => {
  write({ "prompts/system.md": "You answer briefly.\n", "prompts/ask.md": "Question: ${question}\n" });
  const flow = await load(
    agentFlow("    system_prompt: !include prompts/system.md\n    prompt: !include prompts/ask.md\n"),
  );
  const model = new ScriptedModel([textReply("Four.")], { provider: "openai", model: "gpt-test" });

  const result = await runFlow(flow, { question: "2+2?" }, { env: NO_ENV, numWorkers: 0, models: serving(model) });

  expect(result.output).toBe("Four.");
  expect(model.requests[0]!.messages.map((message) => [message.role, message.text()])).toEqual([
    ["system", "You answer briefly.\n"],
    ["user", "Question: 2+2?\n"],
  ]);
});

test("an unknown name in an included template fails at load, at the !include", async () => {
  write({ "ask.md": "Tell ${nobody}.\n" });

  const { message, label, line } = await failure(load(agentFlow("    prompt: !include ask.md\n")));

  expect(message).toContain("nobody");
  expect([label, line]).toEqual(["flow.yaml", 11]);
});

test("included text keeps its final newline and reads CRLF and CR as LF", async () => {
  write({ "a.txt": "one\r\ntwo\rthree\n\n" });
  expect(await outputOf(await load(outputFlow("!include a.txt")))).toBe("one\ntwo\nthree\n\n");
});

test.each(["true", "3", "null", "~", "2024-01-01", "[a, b]", "call(x)"])("included %p is text", async (text) => {
  write({ "a.txt": text });
  expect(await outputOf(await load(outputFlow("{v: !include a.txt}")))).toEqual({ v: text });
});

test("an alias of an included value reads the same text", async () => {
  write({ "a.txt": "shared" });
  expect(await outputOf(await load(outputFlow("{a: &t !include a.txt, b: *t}")))).toEqual({ a: "shared", b: "shared" });
});

test("a uses: child includes from its own folder", async () => {
  write({
    "lib/child.yaml": outputFlow("!include note.txt").replace("id: f\nname: f", "id: child\nname: child"),
    "lib/note.txt": "from the child",
    "note.txt": "from the root",
  });
  const root = "id: f\nname: f\nuses:\n  child: lib/child\nnodes:\n  c:\n    kind: call\n    call: child\noutput: ${c.output}\n";
  expect(await outputOf(await load(root))).toBe("from the child");
});

test("a path is resolved against loadFlow's baseDir when the text has no file", async () => {
  write({ "base/a.txt": "from base" });
  const flow = await loadFlow(outputFlow("!include a.txt"), { baseDir: join(tmp, "base"), env: NO_ENV });
  expect(await outputOf(flow)).toBe("from base");
});

test("an error about an included value quotes the !include, never the file's text", async () => {
  write({ "big.txt": "a long prompt\n" });
  const { message } = await failure(load("id: f\nname: f\nnodes:\n  v: !include big.txt\noutput: ${v.output}\n"));
  expect(message).toBe(
    "node 'v': an entry in `nodes:` is a node or a value; add `kind:` for a node, or a tag (`!object`, " +
      "`!Buy`, ...) for a value",
  );
});

// ---------- where !include may go ----------

test.each([
  ["a mapping key", "output:\n  !include a.txt: x\n", 5, "a mapping key"],
  ["a list", "output: !include [a]\n", 4, "not a list"],
  ["a mapping", "output: !include {a: b}\n", 4, "not a mapping"],
  ["a merge source", "output:\n  <<: !include a.txt\n", 5, "`<<:` merges a mapping"],
  ["a merge source in a list", "output:\n  <<: [!include a.txt]\n", 5, "`<<:` merges a mapping"],
])("!include on %s fails at load", async (_, yaml, line, words) => {
  write({ "a.txt": "x" });
  const { message, label, line: at } = await failure(load(`id: f\nname: f\nnodes: {}\n${yaml}`));
  expect(message).toContain("`!include`");
  expect(message).toContain(words);
  expect([label, at]).toEqual(["flow.yaml", line]);
});

test.each([
  ["a flow input", "input:\n  q: !include a.txt\nnodes: {}\noutput: done\n", 4, "flow input 'q'"],
  [
    "a def input",
    "defs:\n  d:\n    input:\n      q: !include a.txt\n    nodes: {}\n    output: done\nnodes: {}\noutput: done\n",
    6,
    "flow input 'q'",
  ],
  [
    "a single-node def input",
    "defs:\n  d:\n    kind: code\n    code: return 1\n    input:\n      q: !include a.txt\n    output: int\n" +
      "nodes: {}\noutput: done\n",
    8,
    "flow input 'q'",
  ],
  ["a node output", "nodes:\n  a:\n    kind: code\n    code: return 1\n    output: !include a.txt\noutput: ${a.output}\n", 7, "`output:`"],
  ["a compact flow's output", "kind: code\ncode: return 1\noutput: !include a.txt\n", 5, "`output:`"],
  ["a typedef", "typedefs:\n  T: !include a.txt\nnodes: {}\noutput: done\n", 4, "typedef 'T'"],
  ["a typedef field", "typedefs:\n  T:\n    f: !include a.txt\nnodes: {}\noutput: done\n", 5, "typedef 'T'"],
])("!include in %s, a type position, fails at load", async (_, yaml, line, subject) => {
  write({ "a.txt": "str" });
  const { message, label, line: at } = await failure(load(`id: f\nname: f\n${yaml}`));
  expect(message).toContain(subject);
  expect(message).toContain("`!include`");
  expect([label, at]).toEqual(["flow.yaml", line]);
});

// ---------- files that can't be included ----------

test("a missing file fails at the tag, naming the path and the folder", async () => {
  const { message, line } = await failure(load(outputFlow("!include prompts/none.md")));
  expect(message).toBe(
    `\`!include\` file 'prompts/none.md' does not exist in '${tmp}'; its path is relative to the including file's folder`,
  );
  expect(line).toBe(4);
});

test.each(["!include", '!include ""'])("an empty path %p fails at the tag", async (tag) => {
  const { message, line } = await failure(load(outputFlow(tag)));
  expect(message).toStartWith("`!include` needs a file path");
  expect(line).toBe(4);
});

test.each([false, true])("a path holding a NUL fails at the tag, under a code root: %p", async (rooted) => {
  const { message, line } = await failure(load(outputFlow('!include "in\\0.txt"'), rooted ? { codeRoot: tmp } : {}));
  expect(message).toBe("`!include` file 'in\\x00.txt' holds a NUL character; write the path without it");
  expect(line).toBe(4);
});

test("a folder is refused", async () => {
  mkdirSync(join(tmp, "prompts"));
  const { message } = await failure(load(outputFlow("!include prompts")));
  expect(message).toBe("`!include` file 'prompts' is not a regular file; include a text file");
});

test("a FIFO is refused without being opened", async () => {
  const fifo = join(tmp, "pipe");
  if (spawnSync("mkfifo", [fifo]).status !== 0) return;
  const { message } = await failure(load(outputFlow("!include pipe")));
  expect(message).toBe("`!include` file 'pipe' is not a regular file; include a text file");
}, 5000);

test("a file that can't be opened says why and what to do", async () => {
  if (process.getuid?.() === 0) return;
  write({ "a.txt": "secret" });
  chmodSync(join(tmp, "a.txt"), 0o000);
  const { message } = await failure(load(outputFlow("!include a.txt")));
  expect(message).toBe("`!include` file 'a.txt' can't be read (EACCES); include a text file you can read");
});

test("a file that isn't UTF-8 is refused", async () => {
  write({ "a.txt": new Uint8Array([0x68, 0xff, 0x69]) });
  const { message } = await failure(load(outputFlow("!include a.txt")));
  expect(message).toBe("`!include` file 'a.txt' is not UTF-8 text; save it as UTF-8");
});

test("a file over YIAN_MAX_INCLUDE_BYTES is refused; one at the cap loads", async () => {
  write({ "a.txt": "1234", "b.txt": "12345" });
  const env: Env = (name) => (name === "YIAN_MAX_INCLUDE_BYTES" ? "4" : NO_ENV(name));
  expect(await outputOf(await load(outputFlow("!include a.txt"), { env }))).toBe("1234");
  const { message } = await failure(load(outputFlow("!include b.txt"), { env }));
  expect(message).toBe(
    "`!include` file 'b.txt' is larger than 4 bytes (YIAN_MAX_INCLUDE_BYTES); shorten it or raise the cap",
  );
});

test("under a code root, a file outside it is refused whether or not it exists, through a symlink too", async () => {
  const root = join(tmp, "root");
  write({ "secret.txt": "outside", "root/inside.txt": "inside" });
  symlinkSync(join(tmp, "secret.txt"), join(root, "link.txt"));
  for (const path of ["../secret.txt", "../missing.txt", "link.txt"]) {
    const { message } = await failure(load(outputFlow(`!include ${path}`), { dir: root, codeRoot: root }));
    expect(message).toBe(`\`!include\` file '${path}' is outside the code root; include a file inside it`);
  }
  expect(await outputOf(await load(outputFlow("!include inside.txt"), { dir: root, codeRoot: root }))).toBe("inside");
});

/**
 * Run `fn` with `fsp.realpath` calling `swap` once, right after it first resolves `target`: a
 * change on disk between the loader's check of a path and its read.
 */
async function swappingAfterCheck<T>(target: string, swap: () => void, fn: () => Promise<T>): Promise<T> {
  const realpath = fsp.realpath;
  let swapped = false;
  const spy = spyOn(fsp, "realpath").mockImplementation((async (path: string) => {
    const real = await realpath(path);
    if (!swapped && path === target) {
      swapped = true;
      swap();
    }
    return real;
  }) as typeof fsp.realpath);
  try {
    return await fn();
  } finally {
    spy.mockRestore();
  }
}

test("a folder swapped for a symlink out of the code root after the check is refused", async () => {
  const root = join(tmp, "root");
  write({ "root/dir/a.txt": "inside", "outside/a.txt": "outside" });
  const swap = () => {
    renameSync(join(root, "dir"), join(root, "old"));
    symlinkSync(join(tmp, "outside"), join(root, "dir"));
  };
  const loading = () => load(outputFlow("!include dir/a.txt"), { dir: root, codeRoot: root });

  const { message } = await failure(swappingAfterCheck(join(root, "dir", "a.txt"), swap, loading));

  expect(message).toBe("`!include` file 'dir/a.txt' is outside the code root; include a file inside it");
});

test("a file swapped for a FIFO after the check is refused without waiting for a writer", async () => {
  if (spawnSync("mkfifo", [join(tmp, "probe")]).status !== 0) return;
  write({ "a.txt": "text" });
  const swap = () => {
    rmSync(join(tmp, "a.txt"));
    spawnSync("mkfifo", [join(tmp, "a.txt")]);
  };

  const { message } = await failure(swappingAfterCheck(join(tmp, "a.txt"), swap, () => load(outputFlow("!include a.txt"))));

  expect(message).toBe("`!include` file 'a.txt' is not a regular file; include a text file");
}, 5000);

test("under a code root, a missing file names no folder", async () => {
  const { message } = await failure(load(outputFlow("!include none.md"), { codeRoot: tmp }));
  expect(message).toBe("`!include` file 'none.md' does not exist; its path is relative to the including file's folder");
});

test("a typedef can't be named include", async () => {
  const { message } = await failure(load("id: f\nname: f\ntypedefs:\n  include: str\nnodes: {}\noutput: done\n"));
  expect(message).toContain("typedef name 'include' must be PascalCase");
});

test("a host memory type can't be named include", async () => {
  const contract: MemoryContract = { type: "include", tools: [] };
  class Include extends MemoryBase {
    static override readonly contract = contract;
    override open(): Promise<MemorySession> {
      throw new Error("not opened at load");
    }
    override async reset(): Promise<void> {}
  }
  await expect(loadFlow(outputFlow("done"), { memoryTypes: [Include], env: NO_ENV })).rejects.toThrow(
    "memory type name 'include' must be PascalCase",
  );
});

// ---------- recording ----------

test("the flow's source records each included file's text unchanged, under the include: ref", async () => {
  write({ "prompts/a.md": "a\r\n", "lib/child.yaml": outputFlow("!include ../prompts/a.md") });
  const root = "id: f\nname: f\nuses:\n  child: lib/child\nnodes:\n  c:\n    kind: call\n    call: child\noutput: {a: !include prompts/a.md, c: '${c.output}'}\n";

  const flow = await load(root);

  expect(structuredClone(flow.source)).toEqual({
    root: "flow.yaml",
    dir: tmp,
    codeRoot: null,
    files: {
      "flow.yaml": { text: root, refs: { "include:prompts/a.md": "prompts/a.md", "lib/child": "lib/child.yaml" } },
      "lib/child.yaml": { text: outputFlow("!include ../prompts/a.md"), refs: { "include:../prompts/a.md": "prompts/a.md" } },
      "prompts/a.md": { text: "a\r\n", refs: {} },
    },
  });
});

test("a load from the recorded source reads the included text it recorded, not the file on disk", async () => {
  write({ "a.txt": "before" });
  const loaded = await load(outputFlow("!include a.txt"));
  write({ "a.txt": "after" });

  const again = await loadFlowSource(loaded.source!, { env: NO_ENV });

  expect(await outputOf(again)).toBe("before");
  expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
});

test("the fingerprint differs when an included file's text differs", async () => {
  write({ "a.txt": "before" });
  const before = flowFingerprint((await load(outputFlow("!include a.txt"))).compiled);
  write({ "a.txt": "after" });
  expect(flowFingerprint((await load(outputFlow("!include a.txt"))).compiled)).not.toBe(before);
});

/** A recorded source of `flow.yaml` including `a.txt` (recorded under `key`), with `codeRoot`. */
const recorded = (key: string, codeRoot: string | null): FlowSource => ({
  root: "flow.yaml",
  dir: tmp,
  codeRoot,
  files: {
    "flow.yaml": { text: outputFlow("!include a.txt"), refs: { "include:a.txt": key } },
    [key]: { text: "12345", refs: {} },
  },
});

test("a load from a recorded source holds an included file to its code root and the cap", async () => {
  expect(await outputOf(await loadFlowSource(recorded("a.txt", tmp), { env: NO_ENV }))).toBe("12345");
  const outside = await failure(loadFlowSource(recorded("../a.txt", tmp), { env: NO_ENV }));
  expect(outside.message).toBe("`!include` file 'a.txt' is outside the code root; include a file inside it");
  expect([outside.label, outside.line]).toEqual(["flow.yaml", 4]);
  const env: Env = (name) => (name === "YIAN_MAX_INCLUDE_BYTES" ? "4" : NO_ENV(name));
  const large = await failure(loadFlowSource(recorded("a.txt", null), { env }));
  expect(large.message).toContain("is larger than 4 bytes");
});

test.each([
  ["has no include: ref", {}],
  ["has an include: ref to a file the source lacks", { "include:a.txt": "gone.txt" }],
])("a recorded file that %s fails at the tag", async (_, refs) => {
  const source = recorded("a.txt", null);
  const bare = { ...source, files: { ...source.files, "flow.yaml": { text: outputFlow("!include a.txt"), refs } } };
  const { message, line } = await failure(loadFlowSource(bare, { env: NO_ENV }));
  expect(message).toBe(
    "`!include` file 'a.txt' is not among the recorded files; record it under the including file's " +
      "`include:a.txt` ref",
  );
  expect(line).toBe(4);
});

test("a host child's included file outside the code root is refused at load, as a load from its source would be", async () => {
  const root = join(tmp, "root");
  write({ "note.txt": "outside", "root/lib/child.yaml": outputFlow("!include ../../note.txt") });
  const childResolver = () =>
    loadFlow(outputFlow("!include ../../note.txt"), { baseDir: join(root, "lib"), sourceLabel: "child.yaml", env: NO_ENV });
  const text = "id: f\nname: f\nuses:\n  child: lib/child\nnodes:\n  c:\n    kind: call\n    call: child\noutput: ${c.output}\n";

  const { message } = await loadErrorAsync(loadFlow(text, { childResolver, baseDir: root, codeRoot: root, env: NO_ENV }));

  expect(message).toBe("child.yaml: `!include` file '../../note.txt' is outside the code root; include a file inside it");
});

// ---------- !include(T): the value ----------

/** Text with what a template would change: `$$`, `${...}`, CRLF and a lone CR. */
const RAW = "a $$ b ${HOME} $${x}\r\nc\rd\n";

/** A flow with typedefs `Row` and `Rows`, no nodes, and `output: output`. */
const typedOutputFlow = (output: string) =>
  "id: f\nname: f\ntypedefs:\n  Row: {sku: str, qty: int}\n  Rows: {rows: list(Row)}\n" +
  `nodes: {}\noutput: ${output}\n`;

test.each([
  ["a flow output", (tag: string) => outputFlow(tag)],
  ["a value entry", (tag: string) => `id: f\nname: f\nnodes:\n  v: ${tag}\noutput: \${v.output}\n`],
  [
    "a node input",
    (tag: string) =>
      `id: f\nname: f\nnodes:\n  c:\n    kind: code\n    input:\n      x: ${tag}\n    output: str\n` +
      "    code: return inputs.x\noutput: ${c.output}\n",
  ],
  ["a flow input's default", (tag: string) => `id: f\nname: f\ninput:\n  x: ${tag}\nnodes: {}\noutput: \${input.x}\n`],
  ["a record's field", (tag: string) => `id: f\nname: f\nnodes:\n  v: !object {t: ${tag}}\noutput: \${v.output.t}\n`],
])("!include(str) in %s reads the text exactly, never as a template", async (_, flow) => {
  write({ "a.md": RAW });
  expect(await outputOf(await load(flow("!include(str) a.md")))).toBe(RAW);
});

test.each([
  ["int", "123456789012345678901234567890\n", 123456789012345678901234567890n],
  ["list(str)", "- one\n- ${HOME}\n- $$x\n- 'call(x)'\n- call(y)\n", ["one", "${HOME}", "$$x", "call(x)", "call(y)"]],
  ["list(int)", "[1, 2, 3]", [1n, 2n, 3n]],
  ["Optional(int)", "", null],
  ["Optional(int)", "null\n", null],
  ["Rows", "rows:\n  - sku: a\n    qty: 2\n", { rows: [{ sku: "a", qty: 2n }] }],
  ["object", '{"a": [1, "x", 2.5], "b": null, "c": true}', { a: [1n, "x", 2.5], b: null, c: true }],
])("!include(%s) reads the file as one YAML value", async (type, text, value) => {
  write({ "data.yaml": text });
  expect(await outputOf(await load(typedOutputFlow(`!include(${type}) data.yaml`)))).toEqual(value);
  const entry = `id: f\nname: f\ntypedefs:\n  Row: {sku: str, qty: int}\n  Rows: {rows: list(Row)}\nnodes:\n  v: !include(${type}) data.yaml\noutput: \${v.output}\n`;
  expect(await outputOf(await load(entry))).toEqual(value);
});

test.each([
  ["a typedef naming str", "Text"],
  ["Optional(str)", "Optional(str)"],
  ["an optional typedef", "Note"],
  ["a Literal", "Literal(yes)"],
])("!include(T) of %s, whose values are text, reads the file's text exactly", async (_, type) => {
  write({ "a.md": type.startsWith("Literal") ? "yes" : "# Notes\r\n- $${x}: ${y}\n" });
  const flow =
    "id: f\nname: f\ntypedefs:\n  Text: str\n  Note: Optional(Text)\n" +
    `nodes:\n  v: !include(${type}) a.md\noutput: \${v.output}\n`;
  expect(await outputOf(await load(flow))).toBe(type.startsWith("Literal") ? "yes" : "# Notes\r\n- $${x}: ${y}\n");
});

test("text a text T reads is checked against T", async () => {
  write({ "a.md": "yes\n" });
  const { message } = await failure(load("id: f\nname: f\nnodes:\n  v: !include(Literal(yes)) a.md\noutput: ${v.output}\n"));
  expect(message).toBe(
    "'v': `!include(Literal(yes))` file 'a.md' is not a valid Literal(yes): \"yes\\n\" is not one of Literal(yes)",
  );
});

test("a typed value's strings stay as written inside a value holding ${...}", async () => {
  write({ "ex.yaml": "- ${HOME}\n- $$x\n" });
  const flow = await load(
    "id: f\nname: f\ninput:\n  n: !int 3\nnodes:\n  v: !object\n    n: ${input.n}\n    ex: !include(list(str)) ex.yaml\n" +
      "output: ${v.output}\n",
  );
  expect(await outputOf(flow)).toEqual({ n: 3n, ex: ["${HOME}", "$$x"] });
});

test("one file reads as a template with !include and as written with !include(str)", async () => {
  write({ "a.txt": "x $$ y\r\n" });
  expect(await outputOf(await load(outputFlow("{t: !include a.txt, v: !include(str) a.txt}")))).toEqual({
    t: "x $ y\n",
    v: "x $$ y\r\n",
  });
});

test("an alias of a typed !include reads the same value", async () => {
  write({ "n.yaml": "[1, 2]" });
  expect(await outputOf(await load(outputFlow("{a: &x !include(list(int)) n.yaml, b: *x}")))).toEqual({
    a: [1n, 2n],
    b: [1n, 2n],
  });
});

// ---------- !include(T): files that aren't one plain YAML value ----------

test.each([
  ["a syntax error", "rows: [a,\n", 2, 1, "Flow sequence in block collection must be sufficiently indented"],
  ["a second document", "- a\n---\n- b\n", 2, 1, "expected a single document in the stream, but found another document"],
  ["a local tag", "- a\n- !str b\n", 2, 3, "a tag (`!str`) can't be used here; write the value out in plain YAML"],
  ["a core tag", "- !!str 1\n", 1, 3, "a tag (`!!str`) can't be used here; write the value out in plain YAML"],
  ["an !include", "- !include b.md\n", 1, 3, "a tag (`!include`) can't be used here; write the value out in plain YAML"],
  ["an anchor and alias", "- &a x\n- *a\n", 1, 3, "an anchor (`&a`) can't be used here; write the value out in plain YAML"],
  ["a merge key", "a: 1\n<<: {b: 2}\n", 2, 1, "a `<<:` merge key can't be used here; write the value out in plain YAML"],
  ["a repeated key", "a: 1\nb: 2\na: 3\n", 3, 1, "duplicate key 'a' (first at line 1); a YAML mapping's keys must be unique"],
  ["a key YAML reads as non-text", "1: a\n", 1, 1, "the key `1` was read by YAML as an int"],
])("a file holding %s fails at the tag, naming the file, line and column", async (_, text, line, column, words) => {
  write({ "data.yaml": text });
  const { message, label, line: at } = await failure(load(outputFlow("!include(object) data.yaml")));
  expect(message).toStartWith(`\`!include(object)\` file 'data.yaml', line ${line}, column ${column}: `);
  expect(message).toContain(words);
  expect([label, at]).toEqual(["flow.yaml", 4]);
});

// ---------- !include(T): values that don't fit T ----------

test.each([
  [
    "a list item",
    "list(str)",
    "- a\n- b\n- 5\n",
    "'v': `!include(list(str))` file 'data.yaml' is not a valid list(str): 5 does not match declared type str " +
      "(at `v[2]`, line 3 of the file)",
  ],
  [
    "a record field",
    "Rows",
    "rows:\n  - sku: a\n    qty: x\n",
    "'v': `!include(Rows)` file 'data.yaml' is not a valid Rows: \"x\" does not match declared type int " +
      "(at `v.rows[0].qty`, line 3 of the file)",
  ],
  [
    "a scalar",
    "int",
    "abc\n",
    "'v': `!include(int)` file 'data.yaml' is not a valid int: \"abc\" does not match declared type int (line 1 of the file)",
  ],
])("a value that doesn't fit T at %s fails at the tag, saying where", async (_, type, text, expected) => {
  write({ "data.yaml": text });
  const flow =
    "id: f\nname: f\ntypedefs:\n  Row: {sku: str, qty: int}\n  Rows: {rows: list(Row)}\n" +
    `nodes:\n  v: !include(${type}) data.yaml\noutput: \${v.output}\n`;
  const { message, label, line } = await failure(load(flow));
  expect(message).toBe(expected);
  expect([label, line]).toEqual(["flow.yaml", 7]);
});

test("a T that isn't a type fails at the tag, naming the file and a close match", async () => {
  write({ "a.md": "text" });
  const { message, line } = await failure(load(`id: f\nname: f\nnodes:\n  v: !include(strr) a.md\noutput: \${v.output}\n`));
  expect(message).toBe(
    "'v': `!include(strr)` file 'a.md': `strr` is not a type (unknown type 'strr'). Did you mean `str`?",
  );
  expect(line).toBe(4);
});

// ---------- !include(T): where it may go ----------

test.each([
  ["a mapping key", "output:\n  !include(str) a.txt: x\n", 5, "can't be on a mapping key"],
  ["a list", "output: !include(str) [a]\n", 4, "not a list"],
  ["a mapping", "output: !include(int) {a: b}\n", 4, "not a mapping"],
  ["a merge source", "output:\n  <<: !include(object) a.txt\n", 5, "`<<:` merge source"],
  ["a typedef", "typedefs:\n  T: !include(str) a.txt\nnodes: {}\noutput: done\n", 4, "typedef 'T': a type definition is written as a type"],
  [
    "a node's output type",
    "nodes:\n  a:\n    kind: code\n    code: return 1\n    output: !include(str) a.txt\noutput: ${a.output}\n",
    7,
    "`output:` declares a type",
  ],
  ["a field that takes no tag", "nodes: {}\noutput: done\nversion: !include(list(str)) a.txt\n", 5, "`version:` is not one"],
])("!include(T) on %s fails at load", async (_, yaml, line, words) => {
  write({ "a.txt": "str" });
  const head = yaml.startsWith("nodes:") || yaml.startsWith("typedefs:") ? "id: f\nname: f\n" : "id: f\nname: f\nnodes: {}\n";
  const { message, label, line: at } = await failure(load(head + yaml));
  expect(message).toContain("`!include(");
  expect(message).toContain(words);
  expect([label, at]).toEqual(["flow.yaml", line]);
});

test("!include(str) with no path fails at the tag", async () => {
  const { message, line } = await failure(load("id: f\nname: f\ninput:\n  x: !include(str)\nnodes: {}\noutput: ${input.x}\n"));
  expect(message).toBe("`!include(str)` needs a file path, e.g. `prompt: !include(str) prompts/answer.md`");
  expect(line).toBe(4);
});

// ---------- !include(T): files that can't be included ----------

test("a typed !include of a missing file names the tag with its type", async () => {
  const { message } = await failure(load(outputFlow("!include(list(str)) none.yaml")));
  expect(message).toBe(
    `\`!include(list(str))\` file 'none.yaml' does not exist in '${tmp}'; its path is relative to the including file's folder`,
  );
});

test("a typed !include is held to the cap and the code root", async () => {
  const root = join(tmp, "root");
  write({ "root/a.yaml": "[1, 2]", "secret.yaml": "[3]" });
  const env: Env = (name) => (name === "YIAN_MAX_INCLUDE_BYTES" ? "4" : NO_ENV(name));
  const large = await failure(load(outputFlow("!include(list(int)) a.yaml"), { dir: root, env }));
  expect(large.message).toBe(
    "`!include(list(int))` file 'a.yaml' is larger than 4 bytes (YIAN_MAX_INCLUDE_BYTES); shorten it or raise the cap",
  );
  const outside = await failure(load(outputFlow("!include(list(int)) ../secret.yaml"), { dir: root, codeRoot: root }));
  expect(outside.message).toBe(
    "`!include(list(int))` file '../secret.yaml' is outside the code root; include a file inside it",
  );
  expect(await outputOf(await load(outputFlow("!include(list(int)) a.yaml"), { dir: root, codeRoot: root }))).toEqual([
    1n,
    2n,
  ]);
});

// ---------- !include(T): recording ----------

test("a load from the recorded source reads the typed value it recorded, with the same fingerprint", async () => {
  write({ "a.md": "before $$\r\n", "n.yaml": "[1]" });
  const loaded = await load(outputFlow("{t: !include(str) a.md, n: !include(list(int)) n.yaml}"));
  write({ "a.md": "after", "n.yaml": "[2]" });

  const again = await loadFlowSource(loaded.source!, { env: NO_ENV });

  expect(await outputOf(again)).toEqual({ t: "before $$\r\n", n: [1n] });
  expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
});

test("the fingerprint follows a typed !include's file", async () => {
  write({ "n.yaml": "[1]" });
  const before = flowFingerprint((await load(outputFlow("!include(list(int)) n.yaml"))).compiled);
  write({ "n.yaml": "[2]" });
  expect(flowFingerprint((await load(outputFlow("!include(list(int)) n.yaml"))).compiled)).not.toBe(before);
});

test("a source recorded with LF line endings, as one was before typed includes, loads the same", async () => {
  const source: FlowSource = {
    root: "flow.yaml",
    dir: tmp,
    codeRoot: null,
    files: {
      "flow.yaml": { text: outputFlow("!include a.txt"), refs: { "include:a.txt": "a.txt" } },
      "a.txt": { text: "one\ntwo\n", refs: {} },
    },
  };
  write({ "a.txt": "one\r\ntwo\r\n" });
  const recorded = await loadFlowSource(source, { env: NO_ENV });
  const fresh = await load(outputFlow("!include a.txt"));

  expect(await outputOf(recorded)).toBe("one\ntwo\n");
  expect(flowFingerprint(recorded.compiled)).toBe(flowFingerprint(fresh.compiled));
});

// ---------- !include(T) in an agent ----------

/** Run `flow`'s agent on the scripted reply `Four.` and return each message's role and text it sent. */
async function sentMessages(flow: ComposedFlow): Promise<[string, string][]> {
  const model = new ScriptedModel([textReply("Four.")], { provider: "openai", model: "gpt-test" });
  const result = await runFlow(flow, { question: "2+2?" }, { env: NO_ENV, numWorkers: 0, models: serving(model) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return model.requests[0]!.messages.map((message) => [message.role, message.text()]);
}

test("a prompt and a system prompt !include(str) reads are sent exactly as written", async () => {
  write({ "system.md": "Costs are in $$.\r\n", "ask.md": "Shell advice: use ${HOME}, not ${question}.\n" });
  const flow = await load(agentFlow("    system_prompt: !include(str) system.md\n    prompt: !include(str) ask.md\n"));
  expect(await sentMessages(flow)).toEqual([
    ["system", "Costs are in $$.\r\n"],
    ["user", "Shell advice: use ${HOME}, not ${question}.\n"],
  ]);
});

test("a list prompt's message !include(str) reads is sent as written, beside a template message", async () => {
  write({ "a.md": "LaTeX: $$x^2$$" });
  const prompts = "    prompt:\n      - assistant: !include(str) a.md\n      - user: \"Q: ${question}\"\n";
  expect(await sentMessages(await load(agentFlow(prompts)))).toEqual([
    ["system", expect.any(String)],
    ["assistant", "LaTeX: $$x^2$$"],
    ["user", "Q: 2+2?"],
  ]);
});

test("variables !include(T) reads are written into a prompt without being read again", async () => {
  write({ "system.md": "Use $${x} and ${HOME}.", "examples.yaml": "- 'a: ${b}'\n- c $$ d\n" });
  const variables =
    "    variables:\n      system_md: !include(str) system.md\n      examples: !include(list(str)) examples.yaml\n" +
    '    system_prompt: "${system_md}\\n\\n## Examples\\n${join(examples, \'\\n\')}"\n    prompt: ${question}\n';
  expect(await sentMessages(await load(agentFlow(variables)))).toEqual([
    ["system", "Use $${x} and ${HOME}.\n\n## Examples\na: ${b}\nc $$ d"],
    ["user", "2+2?"],
  ]);
});

test("a system prompt built from !include(str) files with rstrip sends their text exactly", async () => {
  write({ "prompts/system.md": "Costs are in $$; keep ${HOME}.\n\n", "prompts/skill.md": "Use ${x} and $$y.\n" });
  const prompts =
    "    variables:\n" +
    "      system_md: !include(str) prompts/system.md\n" +
    "      skill_md: !include(str) prompts/skill.md\n" +
    '    system_prompt: "${rstrip(system_md)}\\n\\n## Skill\\n\\n${skill_md}"\n' +
    "    prompt: ${question}\n";
  expect(await sentMessages(await load(agentFlow(prompts)))).toEqual([
    ["system", "Costs are in $$; keep ${HOME}.\n\n## Skill\n\nUse ${x} and $$y.\n"],
    ["user", "2+2?"],
  ]);
});

test("the text builtins and str take a value !include(str) reads as written", async () => {
  write({ "a.md": "  $${x} $$ \n" });
  const flow = await load(
    "id: f\nname: f\nnodes:\n  raw: !include(str) a.md\n" +
      "  out: !list(str) \"${[rstrip(raw.output), strip(raw.output), replace(raw.output, '$$', 'D'), str(raw.output)]}\"\n" +
      "output: ${out.output}\n",
  );
  expect(await outputOf(flow)).toEqual(["  $${x} $$", "$${x} $$", "  D{x} D \n", "  $${x} $$ \n"]);
});

test("a compact flow's agent sends its !include(str) prompt as written", async () => {
  write({ "ask.md": "Say $$ for ${question}." });
  const flow = await load(
    "id: answer\nname: answer\nkind: agent\nllm_config: {provider: openai, model: gpt-test}\ninput:\n  question: !str\n" +
      "prompt: !include(str) ask.md\noutput: str\n",
  );
  expect((await sentMessages(flow)).at(-1)).toEqual(["user", "Say $$ for ${question}."]);
});

test.each([
  ["prompt:", "    prompt: !include(list(str)) a.yaml\n", 11],
  ["system_prompt:", "    system_prompt: !include(list(str)) a.yaml\n    prompt: hi\n", 11],
  ["user:", "    prompt:\n      - user: !include(object) a.yaml\n", 12],
])("prompt text a typed !include reads, as %s, fails at the tag: the field takes text", async (field, prompts, line) => {
  write({ "a.yaml": "[a]" });
  const { message, label, line: at } = await failure(load(agentFlow(prompts)));
  expect(message).toMatch(/^`!include\([^`]*\)` file 'a\.yaml': /);
  expect(message).toEndWith(`\`${field}\` takes text; read the file with \`!include(str)\``);
  expect([label, at]).toEqual(["flow.yaml", line]);
});

test("a lone list prompt item !include(str) reads fails without quoting the file", async () => {
  write({ "a.md": "the whole file" });
  const { message } = await failure(load(agentFlow("    prompt:\n      - !include(str) a.md\n")));
  expect(message).toContain("put the `!include(str)` under `user:`");
  expect(message).not.toContain("the whole file");
});

test("a steer render !include(str) reads fails at the tag: a render is a template", async () => {
  write({ "render.md": "Steer: ${event}" });
  const flow =
    "id: f\nname: f\nllm_config: {provider: openai, model: gpt-test}\ninput:\n  s: !Inbox(str)\nnodes:\n  a:\n" +
    "    kind: agent\n    steer_inbox:\n      from: ${input.s}\n      render: !include(str) render.md\n    prompt: hi\n" +
    "    output: str\noutput: ${a.output}\n";
  const { message, line } = await failure(load(flow));
  expect(message).toBe(
    "`!include(str)` file 'render.md': `render:` holds a template over each steer event, written in the flow; a file can't give it",
  );
  expect(line).toBe(11);
});

test("an llm_config value !include(str) reads is its text as written", async () => {
  write({ "model.txt": "gpt-$$x" });
  const flow = await load(
    "id: f\nname: f\nnodes:\n  a:\n    kind: agent\n    llm_config: {provider: openai, model: !include(str) model.txt}\n" +
      "    prompt: hi\n    output: str\noutput: ${a.output}\n",
  );
  expect((flow.compiled.nodes.get("a") as AgentNode).llmConfig).toEqual({ provider: "openai", model: "gpt-$$x" });
});

// ---------- !include(T): the YAML node limit ----------

/** An env whose YAML node limit is 50. */
const FIFTY_NODES: Env = (name) => (name === "YIAN_MAX_YAML_NODES" ? "50" : NO_ENV(name));

test("each !include(T) counts toward the node limit as the value it reads, at every site", async () => {
  write({ "s.yaml": JSON.stringify(Array.from({ length: 20 }, (_, i) => i)) });
  const entries = (count: number) =>
    Array.from({ length: count }, (_, i) => `  v${i}: !include(list(int)) s.yaml\n`).join("");
  const flow = (count: number) => `id: f\nname: f\nnodes:\n${entries(count)}output: \${v0.output}\n`;

  expect(await outputOf(await load(flow(1), { env: FIFTY_NODES }))).toHaveLength(20);
  const { message, label } = await failure(load(flow(3), { env: FIFTY_NODES }));
  expect(message).toContain("holds more than 50 YAML nodes");
  expect(message).toContain("each `!include(T)` as the value it reads");
  expect(label).toBe("flow.yaml");
});

test("an alias of a typed !include counts as a copy of its value, as the value written inline would", async () => {
  const items = JSON.stringify(Array.from({ length: 20 }, (_, i) => i));
  write({ "s.yaml": items });
  const included = outputFlow("{a: &x !include(list(int)) s.yaml, b: [*x, *x]}");
  const inline = outputFlow(`{a: &x !list(int) ${items}, b: [*x, *x]}`);

  expect((await failure(load(included, { env: FIFTY_NODES }))).message).toContain("holds more than 50 YAML nodes");
  expect((await failure(load(inline, { env: FIFTY_NODES }))).message).toContain("holds more than 50 YAML nodes");
  expect(await outputOf(await load(outputFlow("{a: &x !include(list(int)) s.yaml}"), { env: FIFTY_NODES }))).toEqual({
    a: Array.from({ length: 20 }, (_, i) => BigInt(i)),
  });
});

test("a prompt read with a text T other than str is sent as written", async () => {
  write({ "ask.md": "Say $$ for ${question}." });
  const prompts = "    prompt: !include(Optional(str)) ask.md\n";
  expect((await sentMessages(await load(agentFlow(prompts)))).at(-1)).toEqual(["user", "Say $$ for ${question}."]);
});

// ---------- !include(T): text fields and fields written in the flow ----------

test("text fields take the exact text a text !include(T) reads, as they take !include's", async () => {
  write({ "code.ts": 'return inputs.x + "$$" + `${inputs.x}`;\r\n', "d.md": "About ${x}\n", "n.txt": "Step $$1" });
  const flow = await load(
    "id: f\nname: f\ndescription: !include(str) d.md\ntypedefs:\n  Text: str\nnodes:\n  c:\n    kind: code\n" +
      "    node_name: !include(Text) n.txt\n    input:\n      x: hi\n    output: str\n    code: !include(str) code.ts\n" +
      "output: ${c.output}\n",
  );
  expect(await outputOf(flow)).toBe("hi$$hi");
  const code = flow.compiled.nodes.get("c") as CodeNode;
  expect([code.code, code.title]).toEqual(['return inputs.x + "$$" + `${inputs.x}`;\r\n', "Step $$1"]);
});

test.each([
  ["names no type", "Txt", "x", "'description': `!include(Txt)` file 'd.md': `Txt` is not a type"],
  ["doesn't fit T", "Literal(a)", "b", "'description': `!include(Literal(a))` file 'd.md' is not a valid Literal(a)"],
])("text a text !include(T) reads for a text field fails at the tag when T %s", async (_, type, text, start) => {
  write({ "d.md": text });
  const { message, line } = await failure(load(`id: f\nname: f\ndescription: !include(${type}) d.md\nnodes: {}\noutput: done\n`));
  expect(message).toStartWith(start);
  expect(line).toBe(3);
});

/** A map node `m` over the flow's def `d`, its `over:` written as `over`. */
const mapFlow = (over: string) =>
  "id: f\nname: f\ndefs:\n  d:\n    input:\n      item: !str\n    nodes: {}\n    output: ${input.item}\n" +
  `nodes:\n  m:\n    kind: map\n    call: d\n    over: ${over}\noutput: \${m.output}\n`;

test.each([
  ["a case's when:", "input:\n  x: !int\nnodes:\n  g:\n    kind: case\n    cases:\n      - when: !include(str) c.txt\n        then: a\n    else: a\n  a: !str done\noutput: ${a.output}\n", 9, "`when:` holds a condition"],
  ["a case's on:", "input:\n  x: !str\nnodes:\n  g:\n    kind: case\n    on: !include(str) c.txt\n    cases:\n      - when: a\n        then: a\n    else: a\n  a: !str done\noutput: ${a.output}\n", 8, "`on:` holds the value a case matches"],
  ["a case's cases:", "nodes:\n  g:\n    kind: case\n    cases: !include(list(object)) c.yaml\n    else: a\n  a: !str done\noutput: ${a.output}\n", 6, "`cases:` holds `when:`, a condition"],
  ["a node's asserts:", "nodes:\n  c:\n    kind: code\n    code: return 1\n    output: int\n    asserts:\n      - !include(str) c.txt\noutput: ${c.output}\n", 9, "`asserts:` holds assertions"],
  ["a flow's asserts:", "nodes: {}\noutput: done\nasserts:\n  - !include(str) c.txt\n", 6, "`asserts:` holds assertions"],
])("an !include(T) in %s, written in the flow, fails first, at the tag", async (_, yaml, line, words) => {
  write({ "c.txt": "${HOME} > 1", "c.yaml": "[{when: '${HOME}', then: a}]" });
  const { message, label, line: at } = await failure(load(`id: f\nname: f\n${yaml}`));
  expect(message).toMatch(/^`!include\([^`]*\)` file 'c\.(txt|yaml)': /);
  expect(message).toContain(`${words}, written in the flow; a file can't give it`);
  expect([label, at]).toEqual(["flow.yaml", line]);
});

test.each(["!include(list(str)) h.yaml", "!include(str) h.txt"])(
  "a map's over: %s fails first, at the tag, whatever the file holds",
  async (over) => {
    write({ "h.yaml": "- ${HOME}\n", "h.txt": "${HOME}" });
    const { message, line } = await failure(load(mapFlow(over)));
    expect(message).toEndWith("`over:` holds a list or a reference, written in the flow; a file can't give it");
    expect(line).toBe(13);
  },
);

test("a case branch an !include(T) reads is a value branch", async () => {
  write({ "a.md": "the whole file ${HOME}" });
  const flow =
    "id: f\nname: f\ninput:\n  x: !int\nnodes:\n  g:\n    kind: case\n    cases:\n      - when: input.x > 1\n" +
    "        then: !include(str) a.md\n    else: b\n  b: !str no\noutput: ${b.output}\n";
  const { message, line } = await failure(load(flow));
  expect(message).toBe(
    "case node 'g' then: a branch names a node or calls a flow; put the value in an entry in `nodes:` " +
      "and name that entry",
  );
  expect(line).toBe(10);
});

test.each([
  ["steer_inbox: !include(str) s.txt", "`steer_inbox:` holds an inbox reference"],
  ["steer_inbox:\n      from: ${input.s}\n      render: !include(str) s.txt", "`render:` holds a template over each steer event"],
  ["steer_inbox: !include(object) s.yaml", "`steer_inbox:` holds `render:`, a template over each steer event"],
])("an agent's %p fails first, at the tag", async (steer, words) => {
  write({ "s.txt": "${input.s}", "s.yaml": "{from: '${input.s}'}" });
  const flow =
    "id: f\nname: f\nllm_config: {provider: openai, model: gpt-test}\ninput:\n  s: !Inbox(str)\nnodes:\n  a:\n" +
    `    kind: agent\n    ${steer}\n    prompt: hi\n    output: str\noutput: \${a.output}\n`;
  expect((await failure(load(flow))).message).toContain(`${words}, written in the flow; a file can't give it`);
});

test("a file both a uses: child and read with !include(str) is recorded as decoded, so a replay matches", async () => {
  const child = outputFlow("done").replace("id: f\nname: f", "id: child\nname: child").replaceAll("\n", "\r\n");
  write({ "lib/child.yaml": child });
  const root =
    "id: f\nname: f\nuses:\n  child: lib/child\nnodes:\n  c:\n    kind: call\n    call: child\n" +
    "output: {text: !include(str) lib/child.yaml, c: '${c.output}'}\n";

  const loaded = await load(root);
  const again = await loadFlowSource(loaded.source!, { env: NO_ENV });

  expect(loaded.source!.files["lib/child.yaml"]!.text).toBe(child);
  expect(await outputOf(loaded)).toEqual({ text: child, c: "done" });
  expect(await outputOf(again)).toEqual({ text: child, c: "done" });
  expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
});
