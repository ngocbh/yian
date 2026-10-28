// The run-log options: `--resume` against the fresh-only options, an existing `--log`, the
// default log path named by the run id, and the resume rows (missing, finished, corrupt, locked or
// interrupted logs) through `yian run`.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { defaultRunLogPath, reportRunLog, validateRunLogOptions } from "../../src/cli/run.ts";
import { type CliIo, HostStop, UsageError, type Writer } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { readRunIndex, runFlow } from "../../src/run.ts";
import { JsonlRunLog, readRunLog, RunResumedRecord, RunStartedRecord } from "../../src/runtime/log.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-run-log-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

const mode = (path: string) => statSync(path).mode & 0o777;

describe("validateRunLogOptions", () => {
  test("resume names each fresh-only option given, in order", () => {
    const check = (options: Parameters<typeof validateRunLogOptions>[0]) => () => validateRunLogOptions(options);
    const resume = join(tmp, "prior.jsonl");
    for (const [conflict, extra] of [
      ["--input", { inputValues: ["topic=new"] }],
      ["--inputs", { inputs: "inputs.json" }],
      ["--provider", { provider: "openai" }],
      ["--model", { model: "gpt-5.5" }],
      ["--log", { log: "new.jsonl" }],
    ] as const) {
      expect(check({ inputValues: [], resume, ...extra })).toThrow(
        new UsageError(`Invalid value: --resume cannot be combined with ${conflict}`),
      );
    }
    const all = { inputValues: ["a=1"], inputs: "i.json", provider: "", model: "m", log: "l.jsonl", resume };
    expect(check(all)).toThrow(
      new UsageError("Invalid value: --resume cannot be combined with --input, --inputs, --provider, --model, --log"),
    );
    expect(check({ inputValues: [], resume })).not.toThrow();
  });

  test("an existing log is rejected, a dangling symlink too", () => {
    const existing = join(tmp, "existing.jsonl");
    writeFileSync(existing, "keep");
    expect(() => validateRunLogOptions({ flow: "flow.yaml", inputValues: [], log: existing })).toThrow(
      new UsageError(`Invalid value for --log: --log path already exists: ${existing}`),
    );
    // A broken symlink is still an occupied directory entry, never a fresh path.
    const dangling = join(tmp, "dangling.jsonl");
    symlinkSync(join(tmp, "missing-target.jsonl"), dangling);
    expect(() => validateRunLogOptions({ flow: "flow.yaml", inputValues: [], log: dangling })).toThrow(
      new UsageError(`Invalid value for --log: --log path already exists: ${dangling}`),
    );
    expect(() =>
      validateRunLogOptions({ flow: "flow.yaml", inputValues: ["a=1"], log: join(tmp, "new.jsonl") }),
    ).not.toThrow();
  });

  test("a resume takes no flow file, and a fresh run needs one", () => {
    const resume = join(tmp, "prior.jsonl");
    expect(() => validateRunLogOptions({ flow: "flow.yaml", inputValues: [], resume })).toThrow(
      new UsageError("Invalid value: --resume runs the flow recorded in the log; leave out <flow>"),
    );
    expect(() => validateRunLogOptions({ inputValues: [] })).toThrow(new UsageError("Missing argument 'FLOW'."));
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

test("reportRunLog prints the path", async () => {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  await reportRunLog("/tmp/a.jsonl", out, () => undefined);
  expect(chunks.join("")).toBe("Run log: /tmp/a.jsonl\n");
});

// ---------- fresh runs through `yian run` ----------

const FLOW = `id: cli_log
name: cli_log
input:
  topic: !str
nodes:
  echo:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: return inputs.topic
output: \${echo.output}
`;

/**
 * Run `yian run` on `FLOW` with `args` and `HOME` at `tmp/home`, `io`'s fields replacing the
 * defaults; return the code and output. A resume (`--resume` in `args`) names no flow file.
 */
async function runCli(args: string[], io: Partial<CliIo> = {}) {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW);
  const out: string[] = [];
  const err: string[] = [];
  const code = await main(["run", ...(args.includes("--resume") ? [] : [flow]), ...args], {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "HOME" ? join(tmp, "home") : undefined),
    ...io,
  });
  return { code, stdout: out.join(""), stderr: err.join("") };
}

describe("fresh runs", () => {
  test("a fresh run uses the private default path and reports it", async () => {
    const stateDir = join(tmp, "home", ".yian");
    const runsDir = join(stateDir, "runs");
    mkdirSync(runsDir, { recursive: true, mode: 0o755 });
    const { code, stdout, stderr } = await runCli(["--input", "topic=clouds", "--quiet"]);
    expect(code).toBe(0);
    // The run's folder, and the run index the run updated when it ended.
    const runs = readdirSync(runsDir).filter((name) => name !== "index.sqlite");
    expect(runs).toHaveLength(1);
    expect(statSync(join(runsDir, "index.sqlite")).isFile()).toBe(true);
    // The writer's lock file stays beside the log, in the run's folder.
    expect(readdirSync(join(runsDir, runs[0]!)).sort()).toEqual(["log.jsonl", "log.jsonl.lock"]);
    const path = join(runsDir, runs[0]!, "log.jsonl");
    const started = readRunLog(path)[0]!;
    expect(started.record).toBeInstanceOf(RunStartedRecord);
    expect(started.runId).toBe(runs[0]!);
    expect(mode(stateDir)).toBe(0o700);
    expect(mode(runsDir)).toBe(0o700);
    expect(mode(join(runsDir, runs[0]!))).toBe(0o700);
    expect(mode(path)).toBe(0o600);
    expect(stderr).toContain(`Run log: ${path}`);
    expect(stdout).toBe("clouds\n");
  });

  test("a fresh run uses the exact explicit log path", async () => {
    const path = join(tmp, "chosen.jsonl");
    const { code, stderr } = await runCli(["--input", "topic=rivers", "--log", path]);
    expect(code).toBe(0);
    expect(statSync(path).isFile()).toBe(true);
    expect(mode(path)).toBe(0o600);
    expect(stderr).toContain(`Run log: ${path}`);
    expect(existsSync(join(tmp, "home", ".yian"))).toBe(false);
  });

  test("a default directory failure is a clean CLI error", async () => {
    mkdirSync(join(tmp, "home"));
    writeFileSync(join(tmp, "home", ".yian"), "");
    const { code, stdout, stderr } = await runCli(["--input", "topic=blocked", "--quiet", "--json"]);
    expect(code).toBe(1);
    expect(stderr).toStartWith("run failed: ");
    expect(stderr).toContain(join(tmp, "home", ".yian"));
    expect(stderr).not.toContain("    at ");
    expect(stderr).not.toContain("Run log:");
    expect(JSON.parse(stdout).error.code).toBe("run.setup");
  });
});

// ---------- resumed runs through `yian run` ----------

/** Run `FLOW` to its end on a fresh log at `tmp/<name>`; return the log's path. */
async function finishedLog(name: string, topic: string, runId: string): Promise<string> {
  const log = join(tmp, name);
  const flow = await loadFlow(FLOW, { searchPaths: [tmp] });
  const result = await runFlow(flow, { topic }, { log, runId, env: NO_ENV });
  expect(result.status).toBe("succeeded");
  return log;
}

/** Strip whitespace, so a path the output wraps still matches. */
const unwrapped = (text: string) => text.replace(/\s+/g, "");

describe("resumed runs", () => {
  test.each([
    ["--input", ["--input", "topic=new"]],
    ["--inputs", ["--inputs", "inputs.json"]],
    ["--provider", ["--provider", "openai"]],
    ["--model", ["--model", "gpt-5.5"]],
    ["--log", ["--log", "new.jsonl"]],
  ])("resume rejects fresh option %s before input or provider side effects", async (conflict, extra) => {
    const resume = join(tmp, "prior.jsonl");
    writeFileSync(resume, "not reached\n");
    writeFileSync(join(tmp, "inputs.json"), '{"topic": "new"}');
    const args = extra.map((arg) => (arg === "inputs.json" ? join(tmp, arg) : arg));
    const { code, stderr } = await runCli(["--resume", resume, ...args]);
    expect(code).toBe(2);
    expect(stderr).toContain("--resume cannot be combined");
    expect(stderr).toContain(conflict);
  });

  test("a missing resume log is rejected before command side effects", async () => {
    const { code, stderr } = await runCli(["--resume", join(tmp, "missing.jsonl")]);
    expect(code).toBe(2);
    expect(stderr).toContain("does not exist");
  });

  test("a terminal resume is read-only and skips fresh side effects", async () => {
    const log = await finishedLog("terminal.jsonl", "mountains", "terminal-run");
    const before = readFileSync(log);
    const { code, stdout, stderr } = await runCli(["--resume", log, "--quiet"]);
    expect(code, stderr).toBe(0);
    expect(readFileSync(log)).toEqual(before);
    expect(stderr).toContain(`Run log: ${log}`);
    expect(stdout).toContain("mountains");
    expect(existsSync(join(tmp, "home", ".yian"))).toBe(false);
  });

  test.each([
    ["plain", [], false],
    ["engine trace", ["--engine-trace"], true],
  ] as const)("a corrupt resume is a clean CLI error (%s)", async (_id, traceArgs, hasTraceback) => {
    const log = join(tmp, "corrupt.jsonl");
    writeFileSync(log, "not-json\n");
    const { code, stderr } = await runCli(["--resume", log, "--quiet", ...traceArgs]);
    expect(code).toBe(1);
    const message = `resume failed: run log ${log} line 1 (seq 0): invalid run-record JSON`;
    expect(unwrapped(stderr)).toContain(unwrapped(message));
    expect(stderr.includes("─ traceback ")).toBe(hasTraceback);
  });

  test("a locked resume names the log and the likely cause", async () => {
    const log = await finishedLog("held.jsonl", "lakes", "held-run");
    const held = JsonlRunLog.resume(log);
    let result;
    try {
      result = await runCli(["--resume", log, "--quiet"]);
    } finally {
      held.close();
    }
    expect(result.code).toBe(1);
    expect(unwrapped(result.stderr)).toContain(
      unwrapped(`resume failed: run log ${log} is locked by another writer (is the run still active?)`),
    );
    expect(result.stderr).not.toContain("    at ");
  });

  test.each([
    [[], 2],
    [["-w", "0"], 0],
    [["-w", "3"], 3],
  ] as const)("an unfinished resume with %p keeps or overrides only future workers", async (workerArgs, expected) => {
    const log = join(tmp, `unfinished-${expected}.jsonl`);
    const flow = await loadFlow(FLOW, { searchPaths: [tmp] });
    const interrupt = (event: RunEvent) => {
      if (event instanceof NodeStarted && event.nodeId === "echo") throw new Error("host interruption");
    };
    const runId = `unfinished-${expected}`;
    const options = { log, runId, numWorkers: 2, onEvent: interrupt, env: NO_ENV };
    await expect(runFlow(flow, { topic: "recovered" }, options)).rejects.toThrow("host interruption");

    const { code, stdout, stderr } = await runCli(["--resume", log, "--quiet", ...workerArgs]);

    expect(code, stderr).toBe(0);
    const resumed = readRunLog(log)
      .map((envelope) => envelope.record)
      .filter((record) => record instanceof RunResumedRecord);
    expect(resumed).toHaveLength(1);
    expect(resumed[0]!.numWorkers).toBe(expected);
    expect(stdout).toContain("recovered");
  });

  test("an interrupt leaves a reported, unlocked, recoverable log", async () => {
    mkdirSync(join(tmp, "home"));
    const stop = new AbortController();
    const err: string[] = [];
    // A TTY stderr, so `echo` starting draws a spinner line.
    const stderr: Writer = {
      isTTY: true,
      write: async (text) => {
        err.push(text);
        if (text.includes(" echo")) stop.abort(new HostStop("SIGINT"));
      },
    };
    const interrupted = await runCli(["--input", "topic=continued"], { stderr, stop: stop.signal });

    expect(interrupted.code, err.join("")).toBe(130);
    expect(err.join("")).toContain("cancelled by user");
    const runsDir = join(tmp, "home", ".yian", "runs");
    const runs = readdirSync(runsDir).filter((name) => name !== "index.sqlite");
    expect(runs).toHaveLength(1);
    const log = join(runsDir, runs[0]!, "log.jsonl");
    expect(err.join("")).toContain(`Run log: ${log}`);
    // Each run, the stopped one and its resume, updates the run index when it ends.
    expect(readRunIndex(runsDir).map((row) => [row.runId, row.status])).toEqual([[runs[0]!, "stopped"]]);

    const resumed = await runCli(["--resume", log, "--quiet"]);

    expect(resumed.code, resumed.stderr).toBe(0);
    expect(resumed.stdout).toContain("continued");
    expect(readRunLog(log).some((envelope) => envelope.record instanceof RunResumedRecord)).toBe(true);
    expect(readRunIndex(runsDir).map((row) => [row.runId, row.status])).toEqual([[runs[0]!, "succeeded"]]);
  });
});
