/**
 * TS-only: a loaded flow keeps the text of every flow file the loader read, and `run_started`
 * records it, with the skills, the tool specs, the yian version and the parent run; resume
 * compiles the recorded flow, never the files on disk.
 *
 * TS-only too: `loadFlowSource` with `codeRoot` checks every recorded file's CODE references, a
 * host child's included, against it.
 */

import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow, loadFlowSource } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { VERSION, withWarnings } from "../../src/platform.ts";
import { resumeFlow, type RunFlowOptions, runFlow } from "../../src/run.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { mcpServer } from "../../src/tools/mcp.ts";
import { handChild, loadErrorAsync } from "./../compose/declarations.ts";
import { expectClosed, hostEnv, killMarkers, stdioOptions, strayProcesses } from "./mcp_helpers.ts";
import { crashAfter, NO_ENV, rejected, textReply } from "./run_state_fixtures.ts";

// Not resolved: on macOS the temp folder sits under the /var -> /private/var symlink.
const tmp = mkdtempSync(join(tmpdir(), "yian-run-started-sources-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let folders = 0;
/** A fresh folder under the test's temp folder. */
const freshDir = (name: string) => {
  const dir = join(tmp, `${name}-${++folders}`);
  mkdirSync(dir, { recursive: true });
  return dir;
};

/** Write each `files` text at its path under `dir`. */
function writeTree(dir: string, files: Readonly<Record<string, string>>): void {
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), text);
  }
}

/** A flow `id` taking `q` that calls `alias` (the `uses:` ref `ref`) with it. */
const caller = (id: string, alias: string, ref: string, extra = "") =>
  `id: ${id}\nname: ${id}\n${extra}input:\n  q: !str\nuses:\n  ${alias}: ${ref}\n` +
  `nodes:\n  c:\n    kind: call\n    call: ${alias}\n    input:\n      q: \${input.q}\noutput: \${c.output}\n`;

/** A flow `id` taking `q` that runs the tool `lookup` on it. */
const leaf = (id: string) =>
  `id: ${id}\nname: ${id}\ninput:\n  q: !str\n` +
  `nodes:\n  fetch:\n    kind: tool\n    tool_id: lookup\n    input:\n      q: \${input.q}\noutput: \${fetch.output}\n`;

/** A flow `id` with no input whose output is its id. */
const leafOut = (id: string) => `id: ${id}\nname: ${id}\nnodes: {}\noutput: ${id}\n`;

const ROOT = caller("root", "child", "lib/child");
/** A child flow whose CODE node runs `./fn.ts:up`, a file beside the child. */
const CODE_CHILD =
  "id: child\nname: child\ninput:\n  q: !str\nnodes:\n  up:\n    kind: code\n    code: ./fn.ts:up\n" +
  "    input:\n      q: ${input.q}\n    output: str\noutput: ${up.output}\n";
const UP = "export function up(inputs: Record<string, any>): string {\n  return `up:${inputs.q}`;\n}\n";
const CHILD = leaf("child");
const NESTED_CHILD = caller("child", "grand", "grand");
const GRAND = leaf("grand");
const PATHS_ROOT = caller("root", "util", "util", "system:\n  paths: [../shared]\n");
const UTIL = leaf("util");

/** A recorded file: its text and the file each `uses:` path written in it resolved to. */
const file = (text: string, refs: Record<string, string> = {}) => ({ text, refs });

/** The load shapes on disk: the files to write under a folder, the root's folder, the recorded files. */
const SHAPES: Record<
  string,
  {
    files: Record<string, string>;
    root: string;
    keys: Record<string, { text: string; refs: Record<string, string> }>;
  }
> = {
  "one level": {
    files: { "flow.yaml": ROOT, "lib/child.yaml": CHILD },
    root: ".",
    keys: { "flow.yaml": file(ROOT, { "lib/child": "lib/child.yaml" }), "lib/child.yaml": file(CHILD) },
  },
  "two levels": {
    files: { "flow.yaml": ROOT, "lib/child.yaml": NESTED_CHILD, "lib/grand.yaml": GRAND },
    root: ".",
    keys: {
      "flow.yaml": file(ROOT, { "lib/child": "lib/child.yaml" }),
      "lib/child.yaml": file(NESTED_CHILD, { grand: "lib/grand.yaml" }),
      "lib/grand.yaml": file(GRAND),
    },
  },
  "system.paths": {
    files: { "proj/flow.yaml": PATHS_ROOT, "shared/util.yaml": UTIL },
    root: "proj",
    keys: { "flow.yaml": file(PATHS_ROOT, { util: "../shared/util.yaml" }), "../shared/util.yaml": file(UTIL) },
  },
  // The child file is a symlink: it is recorded under its real path.
  "a symlinked child file": {
    files: { "flow.yaml": ROOT, "shared/real-child.yaml": CHILD },
    root: ".",
    keys: {
      "flow.yaml": file(ROOT, { "lib/child": "shared/real-child.yaml" }),
      "shared/real-child.yaml": file(CHILD),
    },
  },
};

/** Write `shape`'s files under `dir`, linking `lib/child.yaml` when the shape has no such file. */
function writeShape(dir: string, shape: (typeof SHAPES)[string]): void {
  writeTree(dir, shape.files);
  if (Object.hasOwn(shape.files, "shared/real-child.yaml")) {
    mkdirSync(join(dir, "lib"), { recursive: true });
    symlinkSync(join(dir, "shared", "real-child.yaml"), join(dir, "lib", "child.yaml"));
  }
}

/** Load `<dir>/flow.yaml` as the CLI does: its folder searched first, named by its file name. */
async function loadRoot(dir: string): Promise<ComposedFlow> {
  const path = join(dir, "flow.yaml");
  return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dir], sourceLabel: basename(path) });
}

/** A flow's source as plain data. */
function sourceOf(flow: ComposedFlow) {
  return flow.source === null ? null : structuredClone(flow.source);
}

for (const [name, shape] of Object.entries(SHAPES)) {
  test(`a loaded flow records each file it read and where each reference went: ${name}`, async () => {
    const dir = freshDir("load");
    writeShape(dir, shape);
    const root = join(dir, shape.root);

    expect(sourceOf(await loadRoot(root))).toEqual({ root: "flow.yaml", dir: realpathSync(root), files: shape.keys });
  });
}

for (const [name, shape] of Object.entries(SHAPES)) {
  test(`the recorded files load the same flow once the folder is gone: ${name}`, async () => {
    const dir = freshDir("reload");
    writeShape(dir, shape);
    const loaded = await loadRoot(join(dir, shape.root));
    rmSync(dir, { recursive: true });

    const again = await loadFlowSource(loaded.source!);

    expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
    expect(sourceOf(again)).toEqual(sourceOf(loaded));
  });
}

test("one file reached by two paths is recorded once", async () => {
  const dir = freshDir("twice");
  writeTree(join(dir, "real"), { "b.yaml": leafOut("b") });
  symlinkSync(join(dir, "real"), join(dir, "link"));
  const root =
    "id: root\nname: root\nsystem:\n  paths: [../real]\nuses:\n  b: b\n  c: ../real/b\nnodes: {}\noutput: done\n";
  const link = join(dir, "link");

  const loaded = await loadFlow(root, { searchPaths: [link], baseDir: link, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({
    root: "flow.yaml",
    dir: realpathSync(join(dir, "real")),
    files: { "flow.yaml": file(root, { b: "b.yaml", "../real/b": "b.yaml" }), "b.yaml": file(leafOut("b")) },
  });
});

test("a root given as text that uses the file on disk under its own name records both", async () => {
  const dir = freshDir("self");
  writeTree(dir, { "flow.yaml": leafOut("old") });
  const root =
    "id: root\nname: root\nuses:\n  old: flow\nnodes:\n  c:\n    kind: call\n    call: old\noutput: ${c.output}\n";

  const loaded = await loadFlow(root, { searchPaths: [dir], baseDir: dir, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({
    root: "flow.yaml",
    dir: realpathSync(dir),
    files: { "flow.yaml": file(root, { flow: "flow.yaml~1" }), "flow.yaml~1": file(leafOut("old")) },
  });
  const again = await loadFlowSource(loaded.source!);
  expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
});

test("a chain 50 files deep, and imports shared along several paths, record each file once", async () => {
  const dir = freshDir("chain");
  /** A flow `name` importing each of `refs`, calling none. */
  const importing = (name: string, refs: string[]) =>
    `id: ${name}\nname: ${name}\n` +
    (refs.length > 0 ? `uses:\n${refs.map((ref) => `  ${ref}: ${ref}\n`).join("")}` : "") +
    "nodes: {}\noutput: done\n";
  const chain: Record<string, string> = {};
  for (let i = 0; i < 50; i += 1) chain[`u${i}.yaml`] = importing(`u${i}`, i < 49 ? [`u${i + 1}`] : []);
  writeTree(join(dir, "chain"), chain);
  const lattice = { "a.yaml": importing("a", ["c"]), "b.yaml": importing("b", ["c"]), "c.yaml": importing("c", ["d"]) };
  writeTree(join(dir, "lattice"), { ...lattice, "d.yaml": importing("d", []) });

  const deep = await loadFlow(chain["u0.yaml"]!, { searchPaths: [join(dir, "chain")], sourceLabel: "u0.yaml" });
  const latticeDir = join(dir, "lattice");
  const shared = await loadFlow(importing("root", ["a", "b"]), { searchPaths: [latticeDir], sourceLabel: "root.yaml" });

  expect(Object.keys(deep.source!.files)).toHaveLength(50);
  expect(Object.keys(shared.source!.files).sort()).toEqual(["a.yaml", "b.yaml", "c.yaml", "d.yaml", "root.yaml"]);
});

test("a uses: path keeps its pinned version in refs", async () => {
  const dir = freshDir("pinned");
  const pinned =
    "id: root\nname: root\nuses:\n  a: lib/child@1\n  b: lib/child@2\n" +
    "nodes:\n  x:\n    kind: call\n    call: a\n  y:\n    kind: call\n    call: b\n";
  const versioned = (version: string) =>
    `id: child\nname: child\nversion: "${version}"\nnodes: {}\noutput: ${version}\n`;
  // A host resolver serves each version from its own folder.
  const childResolver = async (ref: string, version: string | null) => {
    const own = join(dir, `v${version}`);
    mkdirSync(own, { recursive: true });
    return loadFlow(versioned(version!), { baseDir: own, sourceLabel: `${ref.split("/").at(-1)}.yaml` });
  };

  const loaded = await loadFlow(pinned, { childResolver, baseDir: dir, sourceLabel: "flow.yaml" });

  expect(loaded.source!.files["flow.yaml"]!.refs).toEqual({
    "lib/child@1": "v1/child.yaml",
    "lib/child@2": "v2/child.yaml",
  });
  const again = await loadFlowSource(loaded.source!);
  expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
});

test("a folder reached through a symlink keys its files from its real folder", async () => {
  const dir = freshDir("symlink");
  writeTree(join(dir, "real"), SHAPES["one level"]!.files);
  symlinkSync(join(dir, "real"), join(dir, "link"));

  const link = join(dir, "link");
  const loaded = await loadFlow(ROOT, { searchPaths: [link], baseDir: link, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({
    root: "flow.yaml",
    dir: realpathSync(join(dir, "real")),
    files: SHAPES["one level"]!.keys,
  });
});

/** A host resolver that loads `<dir>/<ref>.yaml` searching its own folder, as a file resolver would. */
const hostResolver = (dir: string) => async (ref: string) => {
  const path = join(dir, `${ref}.yaml`);
  return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dirname(path)], sourceLabel: basename(path) });
};

test("a host resolver's loaded children join the root's files, keyed from the root's folder", async () => {
  const dir = freshDir("host");
  writeTree(dir, SHAPES["two levels"]!.files);

  const loaded = await loadFlow(ROOT, { childResolver: hostResolver(dir), baseDir: dir, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({ root: "flow.yaml", dir: realpathSync(dir), files: SHAPES["two levels"]!.keys });
});

test("a host child read from outside the root's folder is keyed from the root's folder", async () => {
  const dir = freshDir("outside");
  writeTree(dir, { "other/lib/child.yaml": CHILD });
  // As `yian check --root` loads a child: labelled from another root, its CODE folder its own.
  const childResolver = async () => {
    const path = join(dir, "other", "lib", "child.yaml");
    return loadFlow(readFileSync(path, "utf8"), { sourceLabel: "lib/child.yaml", baseDir: dirname(path) });
  };
  const rootDir = join(dir, "root");
  mkdirSync(rootDir);

  const loaded = await loadFlow(ROOT, { childResolver, baseDir: rootDir, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({
    root: "flow.yaml",
    dir: realpathSync(rootDir),
    files: {
      "flow.yaml": file(ROOT, { "lib/child": "../other/lib/child.yaml" }),
      "../other/lib/child.yaml": file(CHILD),
    },
  });
});

test("host children given as texts in one folder get keys of their own", async () => {
  const dir = freshDir("texts");
  const two =
    "id: root\nname: root\nuses:\n  a: a\n  b: b\n" +
    "nodes:\n  x:\n    kind: call\n    call: a\n  y:\n    kind: call\n    call: b\n";
  const texts: Record<string, string> = { a: leafOut("a"), b: leafOut("b") };
  const childResolver = (ref: string) => loadFlow(texts[ref]!, { baseDir: dir });

  const loaded = await loadFlow(two, { childResolver, baseDir: dir, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({
    root: "flow.yaml",
    dir: realpathSync(dir),
    files: {
      "flow.yaml": file(two, { a: "<flow>", b: "<flow>~1" }),
      "<flow>": file(texts["a"]!),
      "<flow>~1": file(texts["b"]!),
    },
  });
  const again = await loadFlowSource(loaded.source!);
  expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
});

test("a flow built in code, or loaded over one, has no source", async () => {
  const hand = handChild({ output: null });
  expect(hand.source).toBeNull();
  expect(new ComposedFlow(hand.compiled).source).toBeNull();

  const dir = freshDir("hand");
  const text = "id: root\nname: root\nuses:\n  child: lib/child\nnodes:\n  c:\n    kind: call\n    call: child\n";
  const over = await loadFlow(text, {
    childResolver: () => handChild({ output: null }),
    baseDir: dir,
  });
  expect(over.source).toBeNull();
});

/** A flow `id` taking `q` whose CODE node runs `ref`. */
const codeFlow = (id: string, ref: string) =>
  `id: ${id}\nname: ${id}\ninput:\n  q: !str\nnodes:\n  up:\n    kind: code\n    code: ${ref}\n` +
  "    input:\n      q: ${input.q}\n    output: str\noutput: ${up.output}\n";

test("loadFlowSource with codeRoot refuses a recorded file whose CODE reference resolves outside it", async () => {
  const dir = freshDir("code-root");
  const other = freshDir("code-out");
  writeTree(other, { "h.ts": UP });
  symlinkSync(join(other, "h.ts"), join(dir, "link.ts"));
  const loaded = await loadFlow(codeFlow("root", "./link.ts:up"), {
    baseDir: dir,
    searchPaths: [dir],
    sourceLabel: "flow.yaml",
  });

  const error = await loadErrorAsync(loadFlowSource(loaded.source!, { codeRoot: dir }));
  expect(error.message).toEndWith(`outside the code root '${realpathSync(dir)}'`);
  expect((await loadFlowSource(loaded.source!, { codeRoot: tmp })).compiled.flowId).toBe("root");
});

test("loadFlowSource with codeRoot checks a recorded child file outside flow.dir", async () => {
  const dir = freshDir("code-root");
  const other = freshDir("code-out");
  writeTree(other, { "child.yaml": codeFlow("child", "./h.ts:up"), "h.ts": UP });
  const ref = `../${basename(other)}/child`;
  const loaded = await loadFlow(caller("root", "child", ref), {
    baseDir: dir,
    searchPaths: [dir],
    sourceLabel: "flow.yaml",
  });
  const source = loaded.source!;
  expect(Object.keys(source.files).sort()).toEqual([`${ref}.yaml`, "flow.yaml"]);

  const error = await loadErrorAsync(loadFlowSource(source, { codeRoot: dir }));
  expect(error.message).toEndWith(`outside the code root '${realpathSync(dir)}'`);
  expect((await loadFlowSource(source, { codeRoot: tmp })).compiled.flowId).toBe("root");
});

test("a host child loads with codeRoot on disk, but its recorded file is checked by loadFlowSource", async () => {
  const dir = freshDir("code-root");
  const other = freshDir("code-out");
  writeTree(other, { "child.yaml": codeFlow("child", "./h.ts:up"), "h.ts": UP });
  const childResolver = async () =>
    loadFlow(readFileSync(join(other, "child.yaml"), "utf8"), { baseDir: other, sourceLabel: "child.yaml" });

  const loaded = await loadFlow(caller("root", "child", "child"), {
    childResolver,
    baseDir: dir,
    sourceLabel: "flow.yaml",
    codeRoot: dir,
  });

  const error = await loadErrorAsync(loadFlowSource(loaded.source!, { codeRoot: dir }));
  expect(error.message).toEndWith(`outside the code root '${realpathSync(dir)}'`);
});

// ---------- the run start ----------

const logs = mkdtempSync(join(tmpdir(), "yian-run-started-logs-"));
afterAll(() => rmSync(logs, { recursive: true, force: true }));
let logCount = 0;
/** A fresh log path, outside the flow folders. */
const freshLog = () => join(logs, `run-${++logCount}.jsonl`);

/** The tool `name` the flows call: it answers `<name>:<q>`, or runs `action`. */
const lookup = (name = "lookup", description = "Look a word up.", action?: () => void): FunctionTool =>
  tool({
    name,
    description,
    parameters: { type: "object", properties: { q: { type: "string" } }, required: ["q"] },
    execute: (args) => {
      action?.();
      return `${name}:${args.q}`;
    },
  });

/** The skill `s`, with `instructions`. */
const skill = (instructions = "Be brief.") =>
  new Map([["s", new Skill({ name: "s", description: "Answer briefly.", instructions })]]);

const AGENT_FLOW = `
id: agent_flow
name: agent_flow
llm_config: {provider: openai, model: gpt-test}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [t]
    skills: [s]
output: \${agent.output}
`;

/** The models of an AGENT run: each answers "done" at once. */
const models = {
  env: NO_ENV,
  createModel: () => new ScriptedModel([textReply("done")], { provider: "openai", model: "gpt-test" }),
};

/** Run `flow` on `inputs` into a fresh log with the tool `lookup`; return the log. */
async function runInto(
  flow: Parameters<typeof runFlow>[0],
  options: RunFlowOptions = {},
  inputs: Record<string, unknown> = { q: "x" },
): Promise<string> {
  const log = freshLog();
  const result = await runFlow(flow, inputs, { log, env: hostEnv(logs), tools: { lookup: lookup() }, ...options });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return log;
}

/** The `run_started` record of the log at `log`. */
const started = (log: string) => readRunLog(log)[0]!.record as RunStartedRecord;

test("run_started records the flow files, the yian version and no parent", async () => {
  const dir = freshDir("start");
  writeTree(dir, SHAPES["two levels"]!.files);

  const record = started(await runInto(await loadRoot(dir)));

  expect(record.flow).toEqual({ root: "flow.yaml", dir: realpathSync(dir), files: SHAPES["two levels"]!.keys });
  expect(record.yianVersion).toBe(VERSION);
  expect(record.parent).toBeNull();
  // A TOOL node's tool is not one an AGENT grants.
  expect(record.tools).toEqual({});
  expect(record.skills).toEqual({});
});

test("run_started records the parent the host names", async () => {
  const dir = freshDir("parent");
  writeTree(dir, SHAPES["one level"]!.files);
  const flow = await loadRoot(dir);

  expect(started(await runInto(flow, { parent: { runId: "p", node: "n" } })).parent).toEqual({ runId: "p", node: "n" });
  expect(started(await runInto(flow, { parent: { runId: "p" } })).parent).toEqual({ runId: "p", node: null });
});

test("run_started records the tools and skills the AGENTs grant", async () => {
  const flow = await loadFlow(AGENT_FLOW);
  const t = lookup("t", "Look it up.");

  const record = started(await runInto(flow, { tools: { t }, skills: skill(), models }, {}));

  expect(record.tools).toEqual({ t: { description: "Look it up.", parameters: t.parametersJsonSchema } });
  expect(record.skills).toEqual({ s: { description: "Answer briefly.", instructions: "Be brief." } });
});

test("run_started records no flow for a flow built in code", async () => {
  const record = started(await runInto(new ComposedFlow((await loadFlow(leaf("solo"))).compiled)));

  expect(record.flow).toBeNull();
});

// ---------- resume ----------

const markers: string[] = [];
afterAll(() => {
  killMarkers(markers.splice(0));
  expect(strayProcesses()).toEqual([]);
});

/** Cut a run of `flow` after its first `k` records, into a fresh log; return the log. */
async function cutRun(
  flow: ComposedFlow,
  k: number,
  options: RunFlowOptions = {},
  inputs: Record<string, unknown> = { q: "x" },
) {
  const log = freshLog();
  await crashAfter(k, () =>
    runFlow(flow, inputs, { log, env: hostEnv(logs), tools: { lookup: lookup() }, ...options }),
  );
  return log;
}

/** The load shapes, each as a function writing its files and loading the root. */
const LOAD_SHAPES: Record<string, (dir: string) => Promise<ComposedFlow>> = {
  ...Object.fromEntries(
    Object.entries(SHAPES).map(([name, shape]) => [
      name,
      async (dir: string) => {
        writeShape(dir, shape);
        return loadRoot(join(dir, shape.root));
      },
    ]),
  ),
  "a symlinked folder": async (dir: string) => {
    writeTree(join(dir, "real"), SHAPES["one level"]!.files);
    symlinkSync(join(dir, "real"), join(dir, "link"));
    return loadRoot(join(dir, "link"));
  },
  "host children": async (dir: string) => {
    writeTree(dir, SHAPES["two levels"]!.files);
    return loadFlow(ROOT, { childResolver: hostResolver(dir), baseDir: dir, sourceLabel: "flow.yaml" });
  },
  "a host child from outside the root's folder": async (dir: string) => {
    writeTree(dir, { "other/lib/child.yaml": NESTED_CHILD, "other/lib/grand.yaml": GRAND, "root/flow.yaml": ROOT });
    const childResolver = async () => {
      const path = join(dir, "other", "lib", "child.yaml");
      const own = dirname(path);
      const text = readFileSync(path, "utf8");
      return loadFlow(text, { childResolver: hostResolver(own), sourceLabel: "lib/child.yaml", baseDir: own });
    };
    return loadFlow(ROOT, { childResolver, baseDir: join(dir, "root"), sourceLabel: "flow.yaml" });
  },
};

for (const [name, load] of Object.entries(LOAD_SHAPES)) {
  test(`resume runs the recorded flow once its folder is gone: ${name}`, async () => {
    const dir = freshDir("resume");
    const flow = await load(dir);
    // Cut after the root's CALL started, so the resume runs the child.
    const log = await cutRun(flow, 4);
    rmSync(dir, { recursive: true });

    const resumed = await resumeFlow({ log, env: hostEnv(logs), tools: { lookup: lookup() } });

    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(resumed.output).toBe("lookup:x");
  });
}

test("a symlinked child's CODE node resolves beside the link's target on resume too", async () => {
  // CODE files are read from disk, so the folder stays in place; the file sits beside the target.
  const dir = freshDir("resume-code");
  writeTree(dir, { "flow.yaml": ROOT, "store/c.yaml": CODE_CHILD, "store/fn.ts": UP });
  mkdirSync(join(dir, "lib"));
  symlinkSync(join(dir, "store", "c.yaml"), join(dir, "lib", "child.yaml"));
  const log = await cutRun(await loadRoot(dir), 4);

  const resumed = await resumeFlow({ log, env: hostEnv(logs) });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("up:x");
});

/** Cut a run of `AGENT_FLOW` after its AGENT ran, with the tool `t`; return the log. */
async function cutAgentRun(t: FunctionTool = lookup("t", "Look it up.")): Promise<string> {
  return cutRun(await loadFlow(AGENT_FLOW), 4, { tools: { t }, skills: skill(), models }, {});
}

test.each<[string, Partial<{ t: FunctionTool; skills: Map<string, Skill> }>, string]>([
  ["a changed tool description", { t: lookup("t", "Look it up, now.") }, "tools.t description"],
  ["a changed skill text", { skills: skill("Be long.") }, "skills.s instructions"],
])("an unfinished resume refuses %s before it appends anything", async (_name, change, field) => {
  const log = await cutAgentRun();
  const before = readFileSync(log);
  const t = change.t ?? lookup("t", "Look it up.");

  const error = await rejected(
    resumeFlow({ log, env: hostEnv(logs), tools: { t }, skills: change.skills ?? skill(), models }),
  );

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toStartWith(`resume refused: ${field}`);
  expect(readFileSync(log).equals(before)).toBe(true);
});

test(
  "an unfinished resume refuses an MCP tool whose description changed, once the server opened",
  async () => {
    const echo = tool({
      name: "echo",
      description: "Echo the text, as it used to.",
      parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
      execute: (args) => String(args.text),
    });
    const flow = await loadFlow(AGENT_FLOW.replace("tools: [t]", "tools: [echo]").replace("    skills: [s]\n", ""));
    const log = await cutRun(flow, 4, { tools: { echo }, models }, {});
    const before = readFileSync(log);
    const marker = join(logs, `marker-${markers.length + 1}`);
    markers.push(marker);
    const server = mcpServer(stdioOptions("--marker", marker));

    const error = await rejected(resumeFlow({ log, env: hostEnv(logs), tools: { fake: server }, models }));

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toStartWith("resume refused: tools.echo description");
    // The server listed its tools before the check, and the resume closed it.
    expect(readFileSync(marker, "utf8")).not.toBe("");
    await expectClosed(server, marker);
    expect(readFileSync(log).equals(before)).toBe(true);
  },
  30_000,
);

test("a finished log resumes without its tools or skills", async () => {
  const log = await runInto(await loadFlow(AGENT_FLOW), { tools: { t: lookup("t") }, skills: skill(), models }, {});

  const resumed = await resumeFlow({ log, env: hostEnv(logs) });

  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("done");
});

test("a log of another yian version resumes with a warning naming both versions", async () => {
  const dir = freshDir("version");
  writeTree(dir, SHAPES["one level"]!.files);
  const log = await cutRun(await loadRoot(dir), 4);
  const text = readFileSync(log, "utf8");
  writeFileSync(log, text.replace(`"yian_version":"${VERSION}"`, '"yian_version":"0.0.0-other"'));
  const warnings: string[] = [];

  const resumed = await withWarnings(
    (message) => warnings.push(message),
    () => resumeFlow({ log, env: hostEnv(logs), tools: { lookup: lookup() } }),
  );

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(warnings).toHaveLength(1);
  expect(warnings[0]).toContain("0.0.0-other");
  expect(warnings[0]).toContain(`yian ${VERSION}`);

  // Reading the finished log back resumes nothing, so it warns of nothing.
  await withWarnings(
    (message) => warnings.push(message),
    () => resumeFlow({ log, env: hostEnv(logs) }),
  );
  expect(warnings).toHaveLength(1);
});

test("a run of a flow built in code resumes only with that flow passed in", async () => {
  const flow = new ComposedFlow((await loadFlow(leaf("solo"))).compiled);
  const log = await cutRun(flow, 3);
  const before = readFileSync(log);

  const missing = await rejected(resumeFlow({ log, env: hostEnv(logs), tools: { lookup: lookup() } }));
  expect(missing).toBeInstanceOf(RangeError);
  expect((missing as Error).message).toContain("built in code");
  expect(readFileSync(log).equals(before)).toBe(true);

  const resumed = await resumeFlow({ log, flow, env: hostEnv(logs), tools: { lookup: lookup() } });
  expect(resumed.output).toBe("lookup:x");
});

test("a run with recorded flow files refuses a flow passed in", async () => {
  const dir = freshDir("refuse");
  writeTree(dir, SHAPES["one level"]!.files);
  const flow = await loadRoot(dir);
  const log = await cutRun(flow, 4);
  const before = readFileSync(log);

  const error = await rejected(resumeFlow({ log, flow, env: hostEnv(logs), tools: { lookup: lookup() } }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("leave out flow");
  expect(readFileSync(log).equals(before)).toBe(true);
});
