/**
 * The start-up steps `yian run` and `yian chat` share: reading and loading the flow file, the
 * `--provider`/`--model` layer, the key preflight and a fresh run's log path
 * (`defaultRunLogPath`).
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { PassThrough } from "node:stream";

import { cliLayer, defaultRunLogPath, freshLog, loadFlowFile, preflightKeys } from "../../src/cli/startup.ts";
import { type CliIo, CommandEnv } from "../../src/cli/utils.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Diagnostic } from "../../src/diagnostics.ts";
import { AuthError } from "../../src/llm_clients/auth/index.ts";
import { folderSyncs } from "../helpers/folder_syncs.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

const GREET = `id: greet
name: greet
nodes:
  hello:
    kind: code
    output: !str
    code: |
      return "hi";
output: \${hello.output}
`;

const BROKEN = `id: f
name: f
nodes:
  b:
    kind: code
    input:
      x: \${frame_typo.output}
    output: !str
    code: |
      return "hi";
output: \${b.output}
`;

let tmp: string;

const mode = (path: string) => statSync(path).mode & 0o777;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-startup-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Write `content` to `tmp/flow.yaml` and return its path. */
function flowFile(content: string | Uint8Array): string {
  const path = join(tmp, "flow.yaml");
  writeFileSync(path, content);
  return path;
}

/** A non-TTY io whose stderr is collected. */
function collectingIo() {
  const chunks: string[] = [];
  const io: CliIo = {
    stdout: { write: async () => {} },
    stderr: { write: async (chunk) => void chunks.push(chunk) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: NO_ENV,
  };
  return { io, stderr: () => chunks.join("") };
}

describe("loadFlowFile", () => {
  test("returns the loaded flow of a valid file", async () => {
    const { io, stderr } = collectingIo();
    const loaded = await loadFlowFile(flowFile(GREET), io, { engineTrace: false });
    expect(loaded).not.toBeInstanceOf(Diagnostic);
    if (loaded instanceof Diagnostic) return;
    expect(loaded.name).toBe("greet");
    expect(stderr()).toBe("");
  });

  test("a load error returns the located diagnostic, printed with no stack", async () => {
    const { io, stderr } = collectingIo();
    const loaded = await loadFlowFile(flowFile(BROKEN), io, { engineTrace: false });
    expect(loaded).toBeInstanceOf(Diagnostic);
    expect((loaded as Diagnostic).code).toBe("compose.load");
    expect(stderr()).toContain("flow.yaml:");
    expect(stderr()).toContain("frame_typo");
    expect(stderr()).not.toContain("    at ");
  });

  test("a load error with engineTrace also prints the stack", async () => {
    const { io, stderr } = collectingIo();
    const loaded = await loadFlowFile(flowFile(BROKEN), io, { engineTrace: true });
    expect(loaded).toBeInstanceOf(Diagnostic);
    expect(stderr()).toContain("flow.yaml:");
    expect(stderr()).toContain("LoadError");
    expect(stderr()).toContain("    at ");
  });

  test("a file that isn't UTF-8 returns a diagnostic naming it", async () => {
    const { io, stderr } = collectingIo();
    const path = flowFile(new Uint8Array([0x69, 0x64, 0x3a, 0x20, 0xff, 0x0a]));
    const loaded = await loadFlowFile(path, io, { engineTrace: false });
    expect(loaded).toBeInstanceOf(Diagnostic);
    expect((loaded as Diagnostic).code).toBe("compose.load");
    expect(stderr()).toBe(`can't read ${path}: not UTF-8 text\n`);
  });
});

describe("cliLayer", () => {
  test("holds only the flags that are set", () => {
    expect(cliLayer({ provider: "x" })).toEqual({ provider: "x" });
    expect(cliLayer({ provider: "x", model: "m" })).toEqual({ provider: "x", model: "m" });
    expect(cliLayer({ provider: undefined, model: "" })).toEqual({});
  });
});

describe("preflightKeys", () => {
  const AGENT = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n";
  const agentFlow = (): Promise<ComposedFlow> => loadFlow(AGENT, { env: NO_ENV });
  const layer = { provider: "openai", model: "gpt-5.5" };
  const noKeys = () => new CommandEnv(NO_ENV);

  test("passes when every key is set", async () => {
    const { io, stderr } = collectingIo();
    const env = new CommandEnv((name) => (name === "OPENAI_API_KEY" ? "sk-test" : NO_ENV(name)));
    expect(await preflightKeys(await agentFlow(), layer, {}, io, { env, terminal: null })).toBeNull();
    expect(stderr()).toBe("");
  });

  test("a missing key returns a run.setup diagnostic and prints its message", async () => {
    const { io, stderr } = collectingIo();
    const failed = await preflightKeys(await agentFlow(), layer, {}, io, { env: noKeys(), terminal: null });
    expect(failed).toBeInstanceOf(Diagnostic);
    expect(failed?.code).toBe("run.setup");
    expect(stderr()).toStartWith("provider 'openai' needs an API key but $OPENAI_API_KEY is not set");
  });

  test("a failed login returns a diagnostic and prints its message", async () => {
    const { io, stderr } = collectingIo();
    const terminal = {
      askText: async () => {
        throw new AuthError("login refused");
      },
    };
    const failed = await preflightKeys(await agentFlow(), layer, {}, io, { env: noKeys(), terminal });
    expect(failed?.message).toBe("login refused");
    expect(stderr()).toBe("login refused\n");
  });

  test("any other error is thrown", async () => {
    const { io } = collectingIo();
    const terminal = {
      askText: async () => {
        throw new TypeError("a bug");
      },
    };
    const checking = preflightKeys(await agentFlow(), layer, {}, io, { env: noKeys(), terminal });
    await expect(checking).rejects.toThrow("a bug");
  });

  test("a stop that closed the key prompt is thrown, not reported", async () => {
    const { io, stderr } = collectingIo();
    const stop = new AbortController();
    stop.abort(new RangeError("stopped"));
    const terminal = { askText: async () => null };
    const checking = preflightKeys(await agentFlow(), layer, {}, io, {
      env: noKeys(),
      terminal,
      signal: stop.signal,
    });
    await expect(checking).rejects.toThrow("stopped");
    expect(stderr()).toBe("");
  });
});

describe("defaultRunLogPath", () => {
  test("creates the private directories and the run's folder, and not the file", () => {
    const path = defaultRunLogPath("run-1", (name) => (name === "HOME" ? tmp : undefined));
    expect(path).toBe(join(tmp, ".yian", "runs", "run-1", "log.jsonl"));
    expect(mode(join(tmp, ".yian"))).toBe(0o700);
    expect(mode(join(tmp, ".yian", "runs"))).toBe(0o700);
    expect(mode(join(tmp, ".yian", "runs", "run-1"))).toBe(0o700);
    expect(existsSync(path)).toBe(false);
  });

  test("syncs the folder above each folder it makes, and the runs folder once per run", () => {
    const env = (name: string) => (name === "HOME" ? tmp : undefined);
    const home = realpathSync(tmp);
    expect(folderSyncs(() => defaultRunLogPath("run-1", env))).toEqual([
      home,
      join(home, ".yian"),
      join(home, ".yian", "runs"),
    ]);
    expect(folderSyncs(() => defaultRunLogPath("run-2", env))).toEqual([join(home, ".yian", "runs")]);
  });

  test("forces existing directories to 0700", () => {
    mkdirSync(join(tmp, ".yian", "runs"), { recursive: true });
    chmodSync(join(tmp, ".yian"), 0o755);
    chmodSync(join(tmp, ".yian", "runs"), 0o755);
    defaultRunLogPath("run-2", (name) => (name === "HOME" ? tmp : undefined));
    expect(mode(join(tmp, ".yian"))).toBe(0o700);
    expect(mode(join(tmp, ".yian", "runs"))).toBe(0o700);
  });

  test("a state path that is a file is an error", () => {
    writeFileSync(join(tmp, ".yian"), "");
    expect(() => defaultRunLogPath("run-3", (name) => (name === "HOME" ? tmp : undefined))).toThrow(/EEXIST/);
  });

  test("a run folder path that is a file is an error", () => {
    mkdirSync(join(tmp, ".yian", "runs"), { recursive: true });
    writeFileSync(join(tmp, ".yian", "runs", "run-4"), "");
    expect(() => defaultRunLogPath("run-4", (name) => (name === "HOME" ? tmp : undefined))).toThrow(/EEXIST/);
  });
});

describe("freshLog", () => {
  test("without a log, a new run folder under ~/.yian/runs", () => {
    const fresh = freshLog(undefined, (name) => (name === "HOME" ? tmp : undefined));
    expect(fresh.path).toBe(join(tmp, ".yian", "runs", fresh.runId, "log.jsonl"));
    expect(fresh.runDir).toBe(dirname(fresh.path));
    expect(existsSync(fresh.runDir!)).toBe(true);
  });

  test("a given log is used as is, with no run folder", () => {
    const fresh = freshLog("p", (name) => (name === "HOME" ? tmp : undefined));
    expect({ path: fresh.path, runDir: fresh.runDir }).toEqual({ path: "p", runDir: null });
    expect(fresh.runId).not.toBe("");
    expect(existsSync(join(tmp, ".yian"))).toBe(false);
  });
});
