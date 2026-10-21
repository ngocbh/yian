// Ported from tests/cli/test_run_log_options.py: the run-log option checks (`--resume` against
// the fresh-only options, an existing `--log`) and the private default log path, called directly;
// and the fresh-run rows through `yian run`: the default path named by the recorded run id, an
// exact `--log`, and a default directory that can't be made. The resume rows come with `--resume`.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { defaultRunLogPath, reportRunLog, validateRunLogOptions } from "../../src/cli/run.ts";
import { type CliIo, UsageError, type Writer } from "../../src/cli/utils.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";

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
    expect(() => validateRunLogOptions({ inputValues: [], log: existing })).toThrow(
      new UsageError(`Invalid value for --log: --log path already exists: ${existing}`),
    );
    // A broken symlink is still an occupied directory entry, never a fresh path.
    const dangling = join(tmp, "dangling.jsonl");
    symlinkSync(join(tmp, "missing-target.jsonl"), dangling);
    expect(() => validateRunLogOptions({ inputValues: [], log: dangling })).toThrow(
      new UsageError(`Invalid value for --log: --log path already exists: ${dangling}`),
    );
    expect(() => validateRunLogOptions({ inputValues: ["a=1"], log: join(tmp, "new.jsonl") })).not.toThrow();
  });
});

describe("defaultRunLogPath", () => {
  test("creates both private directories and not the file", () => {
    const path = defaultRunLogPath("run-1", (name) => (name === "HOME" ? tmp : undefined));
    expect(path).toBe(join(tmp, ".yian", "runs", "run-1.jsonl"));
    expect(mode(join(tmp, ".yian"))).toBe(0o700);
    expect(mode(join(tmp, ".yian", "runs"))).toBe(0o700);
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

/** Run `yian run` on `FLOW` with `args` and `HOME` at `tmp/home`; return the code and output. */
async function runCli(args: string[]) {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW);
  const out: string[] = [];
  const err: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "HOME" ? join(tmp, "home") : undefined),
  };
  const code = await main(["run", flow, ...args], io);
  return { code, stdout: out.join(""), stderr: err.join("") };
}

describe("fresh runs", () => {
  test("a fresh run uses the private default path and reports it", async () => {
    const stateDir = join(tmp, "home", ".yian");
    const runsDir = join(stateDir, "runs");
    mkdirSync(runsDir, { recursive: true, mode: 0o755 });
    const { code, stdout, stderr } = await runCli(["--input", "topic=clouds", "--quiet"]);
    expect(code).toBe(0);
    const logs = readdirSync(runsDir).filter((name) => name.endsWith(".jsonl"));
    expect(logs).toHaveLength(1);
    const path = join(runsDir, logs[0]!);
    const started = readRunLog(path)[0]!;
    expect(started.record).toBeInstanceOf(RunStartedRecord);
    expect(started.runId).toBe(basename(path, ".jsonl"));
    expect(mode(stateDir)).toBe(0o700);
    expect(mode(runsDir)).toBe(0o700);
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
