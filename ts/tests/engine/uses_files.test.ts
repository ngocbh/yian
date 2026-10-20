/**
 * External callables through the default local file resolver (search path, inert paths, cycle,
 * cache): a port of `tests/engine/test_uses_files.py`.
 *
 * Ported: every row but the seed run. The read-once rows count `fsp.readFile` calls with
 * `spyOn`. Changed: Python also counts source-document parses; a Bun test can't spy on a call
 * inside the loader's own module, so that half is dropped (the loader parses each text it reads
 * once). CODE references are TS's form (`./fixtures/compose_codefns.ts:echo`); nothing checks
 * them at load. The malformed child's message is TS's YAML error, so only its frames are
 * compared.
 *
 * TS-only: a child file that isn't UTF-8 fails at its alias as `could not resolve`.
 *
 * Later (T4-8): `test_seed_24_runs_via_search_path`, which runs the loaded flow.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import fsp from "node:fs/promises";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { CompiledFlow } from "../../src/compile/model.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { iterFlowOrigins } from "../../src/compile/origins.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { LoadCtx, loadFlow, makeFileResolver } from "../../src/compose/loader.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";

import { loadErrorAsync } from "../compose/declarations.ts";

const CODE = "./fixtures/compose_codefns.ts:echo";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-uses-files-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Return a one-node flow `flowId`: a CALL of `call`, else a CODE leaf, with its sections. */
function flowText(
  flowId: string,
  options: { call?: string; uses?: Record<string, string>; system?: string[]; version?: string } = {},
): string {
  const { call, uses, system, version } = options;
  const lines = [`id: ${flowId}`, `name: ${flowId}`];
  // Quoted: a bare 2.0/yes/null reads as non-text.
  if (version !== undefined) lines.push(`version: "${version}"`);
  lines.push("input:", "  x: !str", "nodes:");
  if (call !== undefined) lines.push(`  a: {kind: call, call: ${call}, input: {x: "\${input.x}"}}`);
  else lines.push(`  a: {kind: code, code: ${CODE}, input: {x: "\${input.x}"}}`);
  lines.push('output: {y: "${a.output}"}');
  if (uses !== undefined) {
    lines.push("uses:");
    for (const [alias, ref] of Object.entries(uses)) lines.push(`  ${alias}: ${ref}`);
  }
  if (system !== undefined) lines.push("system:", `  paths: [${system.join(", ")}]`);
  return lines.join("\n") + "\n";
}

/** Write `flowText(flowId, options)` to `<dir>/<filename>.yaml` and return the path. */
function write(dir: string, filename: string, flowId: string, options: Parameters<typeof flowText>[1] = {}): string {
  const path = join(dir, `${filename}.yaml`);
  writeFileSync(path, flowText(flowId, options));
  return path;
}

/** Return the distinct opaque origin keys reachable from one compiled flow. */
function originKeys(flow: CompiledFlow): Set<string> {
  return new Set([...iterFlowOrigins(flow)].map((origin) => origin.key));
}

/** The baked child of `flow`'s CALL node `nodeId`. */
const childOf = (flow: CompiledFlow, nodeId: string) => (flow.nodes.get(nodeId) as CallNode).child as CompiledFlow;

/** Run `fn` counting `fsp.readFile` calls by path, restoring `readFile` after. */
async function countReads(fn: () => Promise<ComposedFlow>): Promise<[ComposedFlow, Map<string, number>]> {
  const spy = spyOn(fsp, "readFile");
  try {
    const flow = await fn();
    const reads = new Map<string, number>();
    for (const [path] of spy.mock.calls) reads.set(String(path), (reads.get(String(path)) ?? 0) + 1);
    return [flow, reads];
  } finally {
    spy.mockRestore();
  }
}

const isDisjoint = (a: Set<string>, b: Set<string>) => [...a].every((key) => !b.has(key));

// ---------- system.paths inert without searchPaths ----------

test("system.paths is inert without searchPaths", async () => {
  // Loaded with no searchPaths, a `system: paths:` stays inert: no file is searched.
  const text = flowText("f", { call: "mom", uses: { mom: "child" }, system: ["../shared"] });
  expect((await loadErrorAsync(loadFlow(text))).message).toContain("needs a search path or child_resolver");
});

test("no searchPaths and no resolver is loud", async () => {
  const error = await loadErrorAsync(loadFlow(flowText("f", { call: "mom", uses: { mom: "child" } })));
  expect(error.message).toContain("needs a search path or child_resolver");
});

// ---------- makeFileResolver: relative resolve, version guard, precedence, skip, miss ----------

test("a sibling resolves relative to the search path", async () => {
  write(tmp, "child", "child");
  const parent = write(tmp, "parent", "parent", { call: "ch", uses: { ch: "child" } });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();
});

test("the version guard matches", async () => {
  write(tmp, "child", "child", { version: "v1" });
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child@v1" } });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();
});

test("the version guard matches a dotted tag", async () => {
  write(tmp, "child", "child", { version: "2.0" });
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child@2.0" } });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();
});

test("a version mismatch is loud and located", async () => {
  write(tmp, "child", "child", { version: "v2" });
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child@v1" } });
  const text = readFileSync(parent, "utf8");
  const expected = SourceDocument.parse(text).root.child("uses").child("ch").range;
  const error = await loadErrorAsync(loadFlow(text, { searchPaths: [tmp] }));
  expect(error.message).toBe(
    'uses: ref pinned child@v1 but the resolved file declares version: "v2" ' +
      '(file-per-version: name a file that declares "v1")',
  );
  expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([expected!]);
});

test("a ref without a version skips the guard", async () => {
  write(tmp, "child", "child", { version: "v9" });
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child" } });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();
});

test("a pinned version on an unversioned file is loud", async () => {
  write(tmp, "child", "child");
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child@v1" } });
  const error = await loadErrorAsync(loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] }));
  expect(error.message).toContain("version: null");
});

test("conflicting version pins on a diamond are loud", async () => {
  // The guard runs per pin over the shared cache, so the cached child still refuses @v2.
  write(tmp, "child", "child", { version: "v1" });
  write(tmp, "m1", "m1", { call: "c", uses: { c: "child@v1" } });
  write(tmp, "m2", "m2", { call: "c", uses: { c: "child@v2" } });
  const text =
    "id: p\nname: p\ninput:\n  x: !str\nnodes:\n" +
    '  a: {kind: call, call: a1, input: {x: "${input.x}"}}\n' +
    '  b: {kind: call, call: a2, input: {x: "${a.output}"}}\n' +
    'output: {y: "${b.output}"}\nuses:\n  a1: m1\n  a2: m2\n';
  const error = await loadErrorAsync(loadFlow(text, { searchPaths: [tmp] }));
  expect(error.message).toContain("version");
  expect(error.diagnostic.frames.map((frame) => [frame.label, frame.note])).toEqual([
    ["<flow>", "uses alias 'a2'"],
    ["m2.yaml", "uses alias 'c'"],
  ]);
});

test("a nested path resolves", async () => {
  const sub = join(tmp, "library");
  mkdirSync(sub);
  write(sub, "relevance", "relevance");
  const parent = write(tmp, "p", "p", { call: "m", uses: { m: "library/relevance" } });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();
});

test("a missing ref is loud", async () => {
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "nope" } });
  const error = await loadErrorAsync(loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] }));
  expect(error.message).toBe(`flow 'nope' not found on search path ["${tmp}"]`);
  // A local ref has no hub: scheme, so the miss says nothing about a marketplace.
  expect(error.message).not.toContain("marketplace");
});

test("search path entries are spelled as Python's Path spells them", async () => {
  const own = join(tmp, "f");
  mkdirSync(own);
  const system = ["../shared", "./lib", "x//y/", `${tmp}//abs/./`];
  const parent = write(own, "p", "p", { call: "ch", uses: { ch: "nope" }, system });
  const error = await loadErrorAsync(loadFlow(readFileSync(parent, "utf8"), { searchPaths: [`${own}/`] }));
  expect(error.message).toBe(
    `flow 'nope' not found on search path ["${own}","${own}/../shared","${own}/lib","${own}/x/y","${tmp}/abs"]`,
  );
});

test("a CRLF child reads with universal newlines", async () => {
  // TS-only: Python's `Path.read_text()` turns `\r\n` and a lone `\r` into `\n`.
  writeFileSync(join(tmp, "child.yaml"), flowText("child").replaceAll("\n", "\r\n"));
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child" } });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();

  writeFileSync(join(tmp, "bad.yaml"), "id: bad\r\nname: bad\rnodes: [\r\n");
  const badParent = write(tmp, "q", "q", { call: "b", uses: { b: "bad" } });
  const error = await loadErrorAsync(loadFlow(readFileSync(badParent, "utf8"), { searchPaths: [tmp] }));
  expect(error.diagnostic.frames.at(-1)!.text).toBe("id: bad\nname: bad\nnodes: [\n");
});

test("a malformed external child keeps the parent and child sources, read once", async () => {
  const child = join(tmp, "child.yaml");
  const childText = "id: [";
  writeFileSync(child, childText);
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child" } });
  const parentText = readFileSync(parent, "utf8");
  const expectedRange = SourceDocument.parse(parentText, "p.yaml").root.child("uses").child("ch").range;

  const spy = spyOn(fsp, "readFile");
  let error: LoadError;
  try {
    error = await loadErrorAsync(loadFlow(parentText, { searchPaths: [tmp], sourceLabel: "p.yaml" }));
    expect(spy.mock.calls.map(([path]) => String(path))).toEqual([realpathSync(child)]);
  } finally {
    spy.mockRestore();
  }

  expect(error.message).toStartWith("invalid YAML");
  const frames = error.diagnostic.frames;
  expect(frames.map((frame) => frame.label)).toEqual(["p.yaml", "child.yaml"]);
  expect(frames.map((frame) => frame.text)).toEqual([parentText, childText]);
  expect(frames[0]!.range).toEqual(expectedRange!);
  expect(frames[1]!.range.startLine).toBe(1);
});

test("the own directory comes first on the search path", async () => {
  const shared = join(tmp, "shared");
  mkdirSync(shared);
  write(tmp, "child", "child");
  write(shared, "child", "child");
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child" }, system: ["shared"] });
  const [, reads] = await countReads(() => loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] }));
  expect([...reads.keys()]).toEqual([realpathSync(join(tmp, "child.yaml"))]);
});

test("a search directory that doesn't exist is skipped", async () => {
  write(tmp, "child", "child");
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child" }, system: ["does_not_exist"] });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();
});

test("a child's own uses: re-root at its directory", async () => {
  // parent -> lib/mid -> lib/leaf: mid's uses resolve in mid's directory.
  const sub = join(tmp, "lib");
  mkdirSync(sub);
  write(sub, "leaf", "leaf");
  write(sub, "mid", "mid", { call: "l", uses: { l: "leaf" } });
  const parent = write(tmp, "p", "p", { call: "m", uses: { m: "lib/mid" } });
  expect(await loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] })).toBeDefined();
});

// ---------- the cross-file cycle guard and diamond reuse ----------

test("the load context has one typed flow cache", () => {
  const context = new LoadCtx();
  expect(context.loading).toEqual(new Set());
  expect(context.flows).toEqual(new Map());
  expect("cache" in context).toBe(false);
  expect("documents" in context).toBe(false);
});

test("a cross-file cycle is refused", async () => {
  // fa calls alias `b` (file fb); fb calls alias `back` (file fa).
  write(tmp, "fa", "fa", { call: "b", uses: { b: "fb" } });
  write(tmp, "fb", "fb", { call: "back", uses: { back: "fa" } });
  const error = await loadErrorAsync(loadFlow(readFileSync(join(tmp, "fa.yaml"), "utf8"), { searchPaths: [tmp] }));
  // The root text isn't a loading file, so the cycle closes when fa.yaml imports fb.yaml again.
  expect(error.message).toBe(`flow-reference cycle across files: ${realpathSync(join(tmp, "fb.yaml"))}`);
});

test("two aliases of one file read it once", async () => {
  const child = write(tmp, "child", "child");
  const text =
    "id: parent\nname: parent\ninput:\n  x: !str\nnodes:\n" +
    '  a: {kind: call, call: first, input: {x: "${input.x}"}}\n' +
    '  b: {kind: call, call: second, input: {x: "${a.output}"}}\n' +
    'output: {y: "${b.output}"}\n' +
    "uses:\n  first: child\n  second: child\n";
  const [loaded, reads] = await countReads(() => loadFlow(text, { searchPaths: [tmp] }));
  const first = childOf(loaded.compiled, "a");
  const second = childOf(loaded.compiled, "b");

  expect(reads).toEqual(new Map([[realpathSync(child), 1]]));
  expect(first).not.toBe(second);
  expect(isDisjoint(originKeys(first), originKeys(second))).toBe(true);
  const resolver = loaded.originResolver!;
  for (const origin of [...iterFlowOrigins(first), ...iterFlowOrigins(second)]) {
    expect(resolver.resolve(origin)!.label).toBe("child.yaml");
  }
});

test("repeated calls through one alias reuse one rebased child", async () => {
  const child = write(tmp, "child", "child");
  const text =
    "id: parent\nname: parent\ninput:\n  x: !str\nnodes:\n" +
    '  a: {kind: call, call: child_view, input: {x: "${input.x}"}}\n' +
    '  b: {kind: call, call: child_view, input: {x: "${a.output}"}}\n' +
    'output: {y: "${b.output}"}\n' +
    "uses:\n  child_view: child\n";
  const [loaded, reads] = await countReads(() => loadFlow(text, { searchPaths: [tmp] }));
  const first = childOf(loaded.compiled, "a");
  const second = childOf(loaded.compiled, "b");

  expect(reads).toEqual(new Map([[realpathSync(child), 1]]));
  expect(first).toBe(second);
  for (const origin of iterFlowOrigins(first)) {
    expect(loaded.originResolver!.resolve(origin)!.label).toBe("child.yaml");
  }
});

test("rebasing an alias doesn't change the canonical path cache", async () => {
  write(tmp, "child", "child");
  const parent = flowText("parent", { call: "child_view", uses: { child_view: "child" } });
  const resolver = makeFileResolver([tmp], new LoadCtx());
  const canonical = await resolver("child", null);
  const canonicalOrigins = originKeys(canonical.compiled);
  const canonicalFingerprint = flowFingerprint(canonical.compiled);
  const canonicalResolver = canonical.originResolver!;
  const framesOf = (flow: ComposedFlow) =>
    new Map([...iterFlowOrigins(flow.compiled)].map((origin) => [origin.key, flow.originResolver!.resolve(origin)]));
  const canonicalFrames = framesOf(canonical);

  const loaded = await loadFlow(parent, { childResolver: resolver });
  const embedded = childOf(loaded.compiled, "a");

  expect(await resolver("child", null)).toBe(canonical);
  expect(canonical.originResolver).toBe(canonicalResolver);
  expect(embedded).not.toBe(canonical.compiled);
  expect(originKeys(canonical.compiled)).toEqual(canonicalOrigins);
  expect(framesOf(canonical)).toEqual(canonicalFrames);
  expect(isDisjoint(originKeys(embedded), canonicalOrigins)).toBe(true);
  expect(flowFingerprint(embedded)).toBe(canonicalFingerprint);
});

test("diamond reuse reads each path once", async () => {
  // p -> (mid1, mid2) -> leaf: the leaf is loaded once, from the cache the second time.
  const leaf = write(tmp, "leaf", "leaf");
  const mid1 = write(tmp, "mid1", "mid1", { call: "l", uses: { l: "leaf" } });
  const mid2 = write(tmp, "mid2", "mid2", { call: "l", uses: { l: "leaf" } });
  const text =
    "id: p\nname: p\ninput:\n  x: !str\nnodes:\n" +
    '  a: {kind: call, call: m1, input: {x: "${input.x}"}}\n' +
    '  b: {kind: call, call: m2, input: {x: "${a.output}"}}\n' +
    'output: {y: "${b.output}"}\nuses:\n  m1: mid1\n  m2: mid2\n';
  const [loaded, reads] = await countReads(() => loadFlow(text, { searchPaths: [tmp] }));
  const leftLeaf = childOf(childOf(loaded.compiled, "a"), "a");
  const rightLeaf = childOf(childOf(loaded.compiled, "b"), "a");

  expect(reads).toEqual(
    new Map([
      [realpathSync(mid1), 1],
      [realpathSync(leaf), 1],
      [realpathSync(mid2), 1],
    ]),
  );
  expect(leftLeaf).not.toBe(rightLeaf);
  expect(isDisjoint(originKeys(leftLeaf), originKeys(rightLeaf))).toBe(true);
  for (const origin of [...iterFlowOrigins(leftLeaf), ...iterFlowOrigins(rightLeaf)]) {
    expect(loaded.originResolver!.resolve(origin)!.label).toBe("leaf.yaml");
  }
});

// ---------- TS-only ----------

test("a child file that isn't UTF-8 fails at its alias", async () => {
  writeFileSync(join(tmp, "child.yaml"), new Uint8Array([0x69, 0x64, 0x3a, 0x20, 0xff, 0xfe, 0x0a]));
  const parent = write(tmp, "p", "p", { call: "ch", uses: { ch: "child" } });
  const error = await loadErrorAsync(loadFlow(readFileSync(parent, "utf8"), { searchPaths: [tmp] }));
  expect(error.message).toStartWith("callable 'ch' (uses: 'child'): could not resolve: ");
  expect(error.diagnostic.frames.map((frame) => frame.note)).toEqual(["uses alias 'ch'"]);
});
