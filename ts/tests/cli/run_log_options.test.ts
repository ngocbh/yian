// Ported from tests/cli/test_run_log_options.py: the run-log option checks (`--resume` against
// the fresh-only options, an existing `--log`) and the private default log path, called directly.
// The CLI-run tests come with `yian run`.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { defaultRunLogPath, reportRunLog, validateRunLogOptions } from "../../src/cli/run.ts";
import { UsageError, type Writer } from "../../src/cli/utils.ts";

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
