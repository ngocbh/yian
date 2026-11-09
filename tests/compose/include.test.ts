/**
 * `!include PATH` reads a string value from a UTF-8 text file, the path relative to the including
 * file's folder: the text, its final newline kept and its line endings read as LF, is the value,
 * as if written there quoted. It is refused on a key, a list, a mapping and a merge source, and in
 * a type position; a file that is missing, not a regular file, not UTF-8, over
 * `YIAN_MAX_INCLUDE_BYTES` or outside the code root fails at the tag. The flow's source records
 * each included file, and a load from the source reads it from there, never from disk.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, renameSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import fsp from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow, loadFlowSource } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../../src/memory/base.ts";
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
  expect(message).toBe(`\`!include\` file 'prompts/none.md' does not exist in '${tmp}'`);
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
  expect(message).toBe("`!include` file 'none.md' does not exist");
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

test("the flow's source records each included file under the include: ref", async () => {
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
      "prompts/a.md": { text: "a\n", refs: {} },
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

test("a recorded file whose include: ref is missing fails at the tag", async () => {
  const source = recorded("a.txt", null);
  const bare = { ...source, files: { ...source.files, "flow.yaml": { text: outputFlow("!include a.txt"), refs: {} } } };
  const { message, line } = await failure(loadFlowSource(bare, { env: NO_ENV }));
  expect(message).toBe("`!include` file 'a.txt' is not among the recorded files");
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
