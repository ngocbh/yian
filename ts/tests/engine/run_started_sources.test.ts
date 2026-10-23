/**
 * TS-only: a loaded flow keeps the text of every flow file the loader read, and `run_started`
 * records it, with the skills, the tool specs, the yian version and the parent run; resume
 * compiles the recorded flow, never the files on disk.
 */

import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow, recordedFiles } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { VERSION, withWarnings } from "../../src/platform.ts";
import { resumeFlow, type RunFlowOptions, runFlow } from "../../src/run.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { mcpServer } from "../../src/tools/mcp.ts";
import { handChild } from "./../compose/declarations.ts";
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

const ROOT = caller("root", "child", "lib/child");
const CHILD = leaf("child");
const NESTED_CHILD = caller("child", "grand", "grand");
const GRAND = leaf("grand");
const PATHS_ROOT = caller("root", "util", "util", "system:\n  paths: [../shared]\n");
const UTIL = leaf("util");

/** The four load shapes: the files to write under a folder, and the folder the root is in. */
const SHAPES: Record<string, { files: Record<string, string>; root: string; keys: Record<string, string> }> = {
  "one level": {
    files: { "flow.yaml": ROOT, "lib/child.yaml": CHILD },
    root: ".",
    keys: { "flow.yaml": ROOT, "lib/child.yaml": CHILD },
  },
  "two levels": {
    files: { "flow.yaml": ROOT, "lib/child.yaml": NESTED_CHILD, "lib/grand.yaml": GRAND },
    root: ".",
    keys: { "flow.yaml": ROOT, "lib/child.yaml": NESTED_CHILD, "lib/grand.yaml": GRAND },
  },
  "system.paths": {
    files: { "proj/flow.yaml": PATHS_ROOT, "shared/util.yaml": UTIL },
    root: "proj",
    keys: { "flow.yaml": PATHS_ROOT, "../shared/util.yaml": UTIL },
  },
};

/** Load `<dir>/flow.yaml` as the CLI does: its folder searched first, named by its file name. */
async function loadRoot(dir: string): Promise<ComposedFlow> {
  const path = join(dir, "flow.yaml");
  return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dir], sourceLabel: basename(path) });
}

/** A flow's source as plain data. */
function sourceOf(flow: ComposedFlow) {
  const source = flow.source;
  if (source === null) return null;
  const { path, dir, searchPaths, files } = source;
  return { path, dir, searchPaths, files: Object.fromEntries(files) };
}

for (const [name, shape] of Object.entries(SHAPES)) {
  test(`a loaded flow keeps the text of each file it read: ${name}`, async () => {
    const dir = freshDir("load");
    writeTree(dir, shape.files);
    const root = join(dir, shape.root);

    expect(sourceOf(await loadRoot(root))).toEqual({
      path: "flow.yaml",
      dir: realpathSync(root),
      searchPaths: ["."],
      files: shape.keys,
    });
  });
}

for (const [name, shape] of Object.entries(SHAPES)) {
  test(`the recorded files load the same flow once the folder is gone: ${name}`, async () => {
    const dir = freshDir("reload");
    writeTree(dir, shape.files);
    const loaded = await loadRoot(join(dir, shape.root));
    const { path, dir: real, searchPaths, files } = loaded.source!;
    rmSync(dir, { recursive: true });

    const again = await loadFlow(files.get(path)!, {
      sourceLabel: path,
      baseDir: real,
      searchPaths: searchPaths.map((item) => join(real, item)),
      files: recordedFiles(real, files),
    });

    expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
    expect(sourceOf(again)).toEqual(sourceOf(loaded));
  });
}

test("a folder reached through a symlink keys its files from the real folder", async () => {
  const dir = freshDir("symlink");
  writeTree(join(dir, "real"), SHAPES["one level"]!.files);
  symlinkSync(join(dir, "real"), join(dir, "link"));

  const link = join(dir, "link");
  const loaded = await loadFlow(ROOT, { searchPaths: [link], baseDir: link, sourceLabel: "flow.yaml" });

  expect(sourceOf(loaded)).toEqual({
    path: "flow.yaml",
    dir: realpathSync(join(dir, "real")),
    searchPaths: ["."],
    files: { "flow.yaml": ROOT, "lib/child.yaml": CHILD },
  });
});

test("a host resolver's loaded children join the root's files, keyed from the root's folder", async () => {
  const dir = freshDir("host");
  writeTree(dir, SHAPES["two levels"]!.files);
  const childResolver = async (ref: string) => {
    const path = join(dir, `${ref}.yaml`);
    return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dirname(path)], sourceLabel: basename(path) });
  };

  const loaded = await loadFlow(ROOT, { childResolver, baseDir: dir, sourceLabel: "flow.yaml" });

  // Resume finds each child where the host found it: in the root's folder.
  expect(sourceOf(loaded)).toEqual({
    path: "flow.yaml",
    dir: realpathSync(dir),
    searchPaths: ["."],
    files: { "flow.yaml": ROOT, "lib/child.yaml": NESTED_CHILD, "lib/grand.yaml": GRAND },
  });
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

test("a host child resume couldn't find where the host found it leaves the root without a source", async () => {
  const dir = freshDir("elsewhere");
  // The host reads the child from another folder, so the root's folder doesn't hold it.
  writeTree(dir, { "other/child.yaml": CHILD });
  const childResolver = async () => {
    const path = join(dir, "other", "child.yaml");
    return loadFlow(readFileSync(path, "utf8"), { searchPaths: [dirname(path)], sourceLabel: basename(path) });
  };

  expect((await loadFlow(ROOT, { childResolver, baseDir: dir, sourceLabel: "flow.yaml" })).source).toBeNull();
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

  expect(record.flow).toEqual({
    path: "flow.yaml",
    dir: realpathSync(dir),
    searchPaths: ["."],
    files: SHAPES["two levels"]!.keys,
  });
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

/** The four load shapes, each as a function writing its files and loading the root. */
const LOAD_SHAPES: Record<string, (dir: string) => Promise<ComposedFlow>> = {
  ...Object.fromEntries(
    Object.entries(SHAPES).map(([name, shape]) => [
      name,
      async (dir: string) => {
        writeTree(dir, shape.files);
        return loadRoot(join(dir, shape.root));
      },
    ]),
  ),
  "a symlinked folder": async (dir: string) => {
    writeTree(join(dir, "real"), SHAPES["one level"]!.files);
    symlinkSync(join(dir, "real"), join(dir, "link"));
    return loadRoot(join(dir, "link"));
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
