// Stops at a run's prompt, as scenarios on `yian run` in process: the end of input is
// `run.cancelled`, the first stop wins, and a `HostStop` a tool throws fails only its node.

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { HostStop } from "../../src/cli/utils.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { registerTool, TOOL_REGISTRY } from "../../src/tools/registry.ts";
import { NO_ENV } from "../engine/run_state_fixtures.ts";
import { FakeInput, FakeOutput } from "./terminal_both_runtimes.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-controller-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

const ASK = '  ask:\n    kind: tool\n    tool_id: ask\n    input:\n      prompt: "Proceed?"\n    output: str\n';

/** A flow whose nodes are `nodes` (YAML lines) and whose output is `output`. */
function flowFile(nodes: string, output: string): string {
  mkdirSync(tmp, { recursive: true });
  const path = join(tmp, "flow.yaml");
  writeFileSync(path, `id: f\nname: f\nnodes:\n${nodes}output: ${output}\n`);
  return path;
}

/** A writer collecting what it is given. */
function sink() {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  return { out, text: () => chunks.join("") };
}

let runs = 0;

/** Run `yian run flow -q --json` on a fresh log and `stdin`, with `overrides` over the io. */
async function runJson(flow: string, stdin: CliIo["stdin"], overrides: Partial<CliIo> = {}) {
  const out = sink();
  const err = sink();
  const io: CliIo = {
    stdout: out.out,
    stderr: err.out,
    stdin,
    stderrStream: new FakeOutput(),
    env: NO_ENV,
    ...overrides,
  };
  const log = join(tmp, `run-${++runs}.jsonl`);
  const code = await main(["run", flow, "-q", "--json", "--log", log], io);
  return { code, stdout: out.text(), stderr: err.text() };
}

/** Resolve once `check` holds, polling each turn of the event loop. */
async function until(check: () => boolean): Promise<void> {
  while (!check()) await new Promise((resolve) => setTimeout(resolve, 1));
}

/** Register a tool `name` running `execute` for the test, removed afterwards. */
async function withTool<T>(
  name: string,
  execute: (args: Record<string, unknown>, ctx: ToolContext) => unknown,
  body: () => Promise<T>,
): Promise<T> {
  const parametersJsonSchema = { type: "object", properties: {} };
  const tool = new FunctionTool({ name, description: "Probe.", parametersJsonSchema, execute });
  registerTool(name, tool);
  try {
    return await body();
  } finally {
    TOOL_REGISTRY.delete(name);
  }
}

const CANCELLED = {
  format: "yian.cli.v1",
  error: { code: "run.cancelled", message: "run cancelled" },
};

test("the end of input at a run's prompt is run.cancelled, exit 1", async () => {
  const stdin = new PassThrough();
  stdin.end();
  const { code, stdout, stderr } = await runJson(flowFile(ASK, "${ask.output}"), stdin);
  expect(code).toBe(1);
  expect(stderr.endsWith("run cancelled\n")).toBe(true);
  expect(JSON.parse(stdout)).toEqual(CANCELLED);
});

test("the first stop wins: the end of input, then the host's, and the reverse", async () => {
  // The end of input first, then a host stop while a sibling still holds the run open.
  const ended = new FakeInput();
  const drawn = new FakeOutput();
  const late = new AbortController();
  let returned = false;
  let lateDuringRun = false;
  const hold = async (_args: unknown, ctx: ToolContext) => {
    await until(() => drawn.text.includes("Proceed?"));
    ended.end();
    await until(() => ctx.signal.aborted);
    late.abort(new HostStop("SIGINT"));
    lateDuringRun = !returned;
    return "held";
  };
  const nodes = `${ASK}  hold:\n    kind: tool\n    tool_id: cli_hold_run\n`;
  const held = flowFile(nodes, "\n  ask: ${ask.output}\n  hold: ${hold.output}");
  const eofFirst = await withTool("cli_hold_run", hold, () =>
    runJson(held, ended, { stop: late.signal, stderrStream: drawn }),
  );
  returned = true;
  expect(lateDuringRun).toBe(true);
  expect(eofFirst.code).toBe(1);
  expect(JSON.parse(eofFirst.stdout)).toEqual(CANCELLED);

  // The host's stop at the drawn prompt first, then the end of input.
  const flow = flowFile(ASK, "${ask.output}");
  const input = new FakeInput();
  const early = new AbortController();
  const output = new FakeOutput();
  const running = runJson(flow, input, { stop: early.signal, stderrStream: output });
  await until(() => output.text.includes("Proceed?"));
  early.abort(new HostStop("SIGINT"));
  input.end();
  const hostFirst = await running;
  expect(hostFirst.code).toBe(130);
  expect(hostFirst.stdout).toBe("");
  expect(hostFirst.stderr.endsWith("cancelled by user\n")).toBe(true);
  expect(input.rawCalls).toEqual([true, false]);
});

test("a failed result wins over the stop it gives a sibling's prompt", async () => {
  const input = new FakeInput();
  const output = new FakeOutput();
  const nodes = `${ASK}  fail:\n    kind: tool\n    tool_id: cli_fail_at_prompt\n`;
  const flow = flowFile(nodes, "\n  ask: ${ask.output}\n  fail: ${fail.output}");
  const execute = async () => {
    await until(() => output.text.includes("Proceed?"));
    throw new Error("boom");
  };
  const { code, stdout, stderr } = await withTool("cli_fail_at_prompt", execute, () =>
    runJson(flow, input, { stderrStream: output }),
  );
  expect(code).toBe(1);
  const document = JSON.parse(stdout);
  expect(document.status).toBe("failed");
  expect(document.failure.node).toBe("fail");
  expect(stderr).not.toContain("cancelled");
  expect(input.rawCalls).toEqual([true, false]);
});

test("an observer failure wins over the prompt it cancels", async () => {
  const input = new FakeInput();
  const output = new FakeOutput();
  const nodes = `${ASK}  done:\n    kind: tool\n    tool_id: cli_done_at_prompt\n`;
  const flow = flowFile(nodes, "\n  ask: ${ask.output}\n  done: ${done.output}");
  // Progress on a TTY whose line for `done`, printed while the prompt is up, throws.
  const chunks: string[] = [];
  const stderr: Writer = {
    isTTY: true,
    write: (text) => {
      if (text.includes("✓ done")) throw new TypeError("stderr broke");
      chunks.push(text);
      return Promise.resolve();
    },
  };
  const execute = async () => {
    await until(() => output.text.includes("Proceed?"));
    return "done";
  };
  const out = sink();
  const env = (name: string) => (name === "NO_COLOR" ? "1" : NO_ENV(name));
  const io: CliIo = { stdout: out.out, stderr, stdin: input, stderrStream: output, env };
  const code = await withTool("cli_done_at_prompt", execute, () =>
    main(["run", flow, "--log", join(tmp, "run.jsonl")], io),
  );
  expect(code).toBe(1);
  const text = chunks.join("");
  expect(text).toContain("run failed: stderr broke");
  expect(text).not.toContain("cancelled");
  expect(input.rawCalls).toEqual([true, false]);
});

test("a HostStop a tool throws fails its node, not reported as Ctrl-C", async () => {
  const nodes = "  stop:\n    kind: tool\n    tool_id: cli_throw_host_stop\n";
  const flow = flowFile(nodes, "${stop.output}");
  const execute = () => {
    throw new HostStop("SIGINT");
  };
  const { code, stdout, stderr } = await withTool("cli_throw_host_stop", execute, () =>
    runJson(flow, new PassThrough()),
  );
  expect(code).toBe(1);
  expect(JSON.parse(stdout).status).toBe("failed");
  expect(stderr).not.toContain("cancelled by user");
});
