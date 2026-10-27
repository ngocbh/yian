// The progress reporter: per-node progress, `✓` on success, `✗` and the error on failure, the
// output when verbose. Most rows run on a non-TTY writer, the path the CLI takes off a terminal: no
// spinner, only each node's final line. Also: the spinner a TTY writer gets, `pause`/`resume`, and
// `warn`'s line above the spinners, held while paused. The CLI-run tests drive a CODE-only flow
// through `yian run`: `--verbose` prints the output, `--quiet` streams no events, and the worker
// count is read from the run log's `RunStarted` record.

import { afterEach, beforeEach, expect, jest, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { ProgressReporter } from "../../src/cli/run.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { NodeExpanded, NodeFailed, NodeRouted, NodeStarted, NodeSucceeded } from "../../src/events.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";

// `NO_COLOR` keeps a TTY writer's text plain, so the tests see only the cursor control.
const noColor = (name: string) => (name === "NO_COLOR" ? "1" : undefined);

/** A reporter writing to an in-memory writer, a TTY one when `isTTY`, and what it was given. */
function sinkReporter(options: { verbose?: boolean; isTTY?: boolean } = {}) {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text), isTTY: options.isTTY ?? false };
  const reporter = new ProgressReporter(out, { verbose: options.verbose ?? false, env: noColor });
  return { reporter, chunks, text: () => chunks.join("") };
}

const started = (nodeId: string) => new NodeStarted({ nodeId });
const succeeded = (nodeId: string, output: unknown) => new NodeSucceeded({ nodeId, output });

afterEach(() => {
  jest.useRealTimers();
});

test("success renders check", () => {
  const { reporter, text } = sinkReporter();
  reporter.handle(started("frame"));
  reporter.handle(succeeded("frame", "ok"));
  expect(text()).toBe("✓ frame\n");
});

test("failure renders cross and error", () => {
  const { reporter, text } = sinkReporter();
  reporter.handle(started("verdict"));
  const failure = new Failure({ code: "node.execute", message: "boom", nodeId: "verdict", causeType: "ValueError" });
  reporter.handle(new NodeFailed({ failure }));
  expect(text()).toBe("✗ verdict\n    boom\n");
});

test("routed renders check with handle", () => {
  // A router (CASE) emits NodeRouted, not NodeSucceeded: the reporter still prints a completion
  // line naming the chosen handle.
  const { reporter, text } = sinkReporter();
  reporter.handle(started("gate"));
  reporter.handle(new NodeRouted({ nodeId: "gate", handle: "yes" }));
  expect(text()).toBe("✓ gate → yes\n");
});

test("verbose prints output", () => {
  const { reporter, text } = sinkReporter({ verbose: true });
  reporter.handle(started("frame"));
  reporter.handle(succeeded("frame", "line one\nline two"));
  reporter.handle(started("count"));
  reporter.handle(succeeded("count", { n: 3n, tags: ["a"] }));
  expect(text()).toBe('✓ frame\n    line one\n    line two\n✓ count\n    {"n":3,"tags":["a"]}\n');
});

test("non-verbose omits output", () => {
  const { reporter, text } = sinkReporter({ verbose: false });
  reporter.handle(started("frame"));
  reporter.handle(succeeded("frame", "secret-payload"));
  expect(text()).not.toContain("secret-payload");
});

test("boundary nodes are ignored", () => {
  const { reporter, text } = sinkReporter();
  for (const nodeId of [START_ID, END_ID]) {
    reporter.handle(started(nodeId));
    reporter.handle(succeeded(nodeId, "x"));
  }
  expect(text()).toBe("");
});

test("namespaced boundary nodes are ignored", () => {
  // A loop or map iteration's boundary nodes are namespaced (`polish#0/__start__`); they are
  // skipped too, while a real body node in the same namespace still renders.
  const { reporter, text } = sinkReporter();
  for (const nodeId of ["polish#0/__start__", "polish#0/__end__"]) {
    reporter.handle(started(nodeId));
    reporter.handle(succeeded(nodeId, "x"));
  }
  reporter.handle(started("polish#0/critic"));
  reporter.handle(succeeded("polish#0/critic", "x"));
  expect(text()).toBe("✓ polish#0/critic\n");
});

test("expanded clears spinner without check", () => {
  // A spawner emits NodeExpanded, not NodeSucceeded: no ✓ or ✗ line, and its spinner is cleared,
  // else the driver clones would spin forever.
  const { reporter, text } = sinkReporter();
  reporter.handle(started("polish~1"));
  reporter.handle(new NodeExpanded({ nodeId: "polish~1", grow: { seed: null, prune: new Set() } }));
  expect(text()).toBe("");

  jest.useFakeTimers();
  const tty = sinkReporter({ isTTY: true });
  tty.reporter.handle(started("polish~1"));
  tty.reporter.handle(new NodeExpanded({ nodeId: "polish~1", grow: { seed: null, prune: new Set() } }));
  expect(tty.chunks).toEqual(["\r\x1b[K⠋ polish~1", "\r\x1b[K"]);
  jest.advanceTimersByTime(400);
  expect(tty.chunks).toHaveLength(2);
});

test("on a TTY a running node draws a spinner the finish line replaces", () => {
  jest.useFakeTimers();
  const { reporter, chunks } = sinkReporter({ isTTY: true });
  reporter.handle(started("frame"));
  expect(chunks).toEqual(["\r\x1b[K⠋ frame"]);
  jest.advanceTimersByTime(80);
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ frame");
  reporter.handle(started("other"));
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ frame\n⠙ other"); // one spinner line per running node
  reporter.handle(succeeded("frame", "ok"));
  // The finish line erases both spinner lines, then the remaining one is drawn again.
  expect(chunks.at(-2)).toBe("\r\x1b[K\x1b[A\x1b[K✓ frame\n");
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ other");
  reporter.handle(succeeded("other", "ok"));
  expect(chunks.at(-1)).toBe("\r\x1b[K✓ other\n");
  const count = chunks.length;
  jest.advanceTimersByTime(400);
  expect(chunks).toHaveLength(count); // nothing running: the redraw stopped
});

test("pause and resume stop and restart the redraw", () => {
  jest.useFakeTimers();
  const { reporter, chunks } = sinkReporter({ isTTY: true });
  reporter.handle(started("ask"));
  reporter.pause();
  expect(chunks).toEqual(["\r\x1b[K⠋ ask", "\r\x1b[K"]);
  jest.advanceTimersByTime(400);
  expect(chunks).toHaveLength(2);
  // A node finishing while paused prints its line and draws no spinner.
  reporter.handle(started("other"));
  reporter.handle(succeeded("other", "ok"));
  expect(chunks.slice(2)).toEqual(["✓ other\n"]);
  reporter.resume();
  expect(chunks.at(-1)).toBe("\r\x1b[K⠋ ask");
  jest.advanceTimersByTime(80);
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ ask");
  reporter.stop();
  expect(chunks.at(-1)).toBe("\r\x1b[K");
  const count = chunks.length;
  jest.advanceTimersByTime(400);
  expect(chunks).toHaveLength(count);
});

test("a warning prints on its own line above the spinners", () => {
  jest.useFakeTimers();
  const tty = sinkReporter({ isTTY: true });
  tty.reporter.handle(started("frame"));
  tty.reporter.warn("w");
  expect(tty.chunks).toEqual(["\r\x1b[K⠋ frame", "\r\x1b[Kwarning: w\n", "\r\x1b[K⠋ frame"]);
  tty.reporter.stop();

  const plain = sinkReporter();
  plain.reporter.handle(started("frame"));
  plain.reporter.warn("w");
  expect(plain.text()).toBe("warning: w\n");
});

test("a warning while paused is held until resume or stop", () => {
  jest.useFakeTimers();
  const { reporter, chunks } = sinkReporter({ isTTY: true });
  reporter.handle(started("frame"));
  reporter.pause();
  reporter.warn("w");
  expect(chunks).toEqual(["\r\x1b[K⠋ frame", "\r\x1b[K"]);
  reporter.resume();
  expect(chunks.slice(2)).toEqual(["warning: w\n", "\r\x1b[K⠋ frame"]);
  reporter.pause();
  reporter.warn("x");
  const count = chunks.length;
  reporter.stop();
  expect(chunks.slice(count)).toEqual(["warning: x\n"]);
});

// ---------- the flags of `yian run` ----------

const FLOW = "id: f\nname: f\nnodes:\n  a: {kind: code, output: str, code: return 'hello'}\noutput: ${a.output}\n";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-progress-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Run `yian run` on `FLOW` with `args`, its log at `tmp/<log>`; return the exit code and output. */
async function runCli(args: string[], log = "run.jsonl") {
  const flow = join(tmp, "f.yaml");
  writeFileSync(flow, FLOW);
  const out: string[] = [];
  const err: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "HOME" ? tmp : undefined),
  };
  const code = await main(["run", flow, "--log", join(tmp, log), ...args], io);
  return { code, stdout: out.join(""), stderr: err.join("") };
}

/** Return the worker count the run logged at `tmp/<log>` recorded in its `RunStarted`. */
function loggedWorkers(log: string): number {
  const started = readRunLog(join(tmp, log))[0]!.record;
  expect(started).toBeInstanceOf(RunStartedRecord);
  return (started as RunStartedRecord).numWorkers;
}

test("cli accepts verbose flag", async () => {
  const { code, stdout, stderr } = await runCli(["--verbose"]);
  expect(code).toBe(0);
  expect(stdout).toBe("hello\n");
  expect(stderr).toContain("✓ a\n    hello\n");
});

test("cli quiet suppresses events", async () => {
  const { code, stdout, stderr } = await runCli(["--quiet"]);
  expect(code).toBe(0);
  expect(stdout).toBe("hello\n");
  expect(stderr).toBe(`Run log: ${join(tmp, "run.jsonl")}\n`);
});

test("cli num workers threads through", async () => {
  // Fresh runs use eight workers unless explicitly overridden, including zero.
  for (const [args, workers] of [
    [[], 8],
    [["--num-workers", "4"], 4],
    [["-w", "2"], 2],
    [["-w", "0"], 0],
  ] as const) {
    const log = `run-${workers}.jsonl`;
    expect((await runCli([...args, "-q"], log)).code).toBe(0);
    expect(loggedWorkers(log)).toBe(workers);
  }
});
