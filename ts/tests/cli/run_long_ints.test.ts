// Ported from tests/cli/test_run_long_ints.py, the four `yian run` rows, inverted: Python refuses an
// int past its 4300-digit limit at the inputs, while TS yian's ints are `bigint`s with no digit
// limit, so a long int from `--input` or `--inputs` runs and prints whole. A lone long int in the
// `--inputs` file is still not an object. The structured-prompt row comes with the prompts.

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";

const D = "1".repeat(5000);

const LIST_FLOW = `id: long_list
name: long_list
input:
  xs: !list(int)
nodes:
  echo:
    kind: code
    input:
      xs: \${input.xs}
    output: list(int)
    code: return inputs.xs
output: \${echo.output}
`;

const INT_FLOW = `id: long_int
name: long_int
input:
  n: !int
nodes:
  echo:
    kind: code
    input:
      n: \${input.n}
    output: int
    code: return inputs.n
output: \${echo.output}
`;

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-long-ints-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Write `text` at `tmp/name` and return its path. */
function write(name: string, text: string): string {
  const path = join(tmp, name);
  writeFileSync(path, text);
  return path;
}

/** Run `yian run` with `args`; return the exit code and output. */
async function runCli(args: string[]) {
  const out: string[] = [];
  const err: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    env: (name) => (name === "HOME" ? tmp : undefined),
  };
  const code = await main(["run", ...args], io);
  return { code, stdout: out.join(""), stderr: err.join("") };
}

test("input flag accepts a long int", async () => {
  const flow = write("flow.yaml", INT_FLOW);
  const log = join(tmp, "run.jsonl");
  const { code, stdout } = await runCli([flow, "-q", "-i", `n=${D}`, "--log", log]);
  expect(code).toBe(0);
  expect(stdout).toBe(`${D}\n`);
  expect(existsSync(log)).toBe(true);
});

test("inputs file accepts a long int in a list", async () => {
  const flow = write("flow.yaml", LIST_FLOW);
  const inputs = write("in.json", `{"xs": [1, ${D}]}`);
  const { code, stdout } = await runCli([flow, "--inputs", inputs, "--log", join(tmp, "run.jsonl"), "--json"]);
  expect(code).toBe(0);
  expect(stdout).toBe(`{"format":"yian.cli.v1","output":[1,${D}],"status":"succeeded"}\n`);
});

test("inputs file holding a lone long int is not an object", async () => {
  const flow = write("flow.yaml", INT_FLOW);
  const inputs = write("in.json", D);
  const { code, stderr } = await runCli([flow, "--inputs", inputs, "--log", join(tmp, "run.jsonl")]);
  expect(code).toBe(2);
  expect(stderr).toContain("must contain a JSON object, got number");
  expect(stderr).not.toMatch(/[0-9]{600}/);
});

test("input flag accepts an int at Python's limit", async () => {
  const flow = write("flow.yaml", INT_FLOW);
  const nines = "9".repeat(4300);
  const { code, stdout } = await runCli([flow, "-q", "-i", `n=${nines}`, "--log", join(tmp, "run.jsonl")]);
  expect(code).toBe(0);
  expect(stdout).toBe(`${nines}\n`);
});
