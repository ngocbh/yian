// The progress reporter: per-node progress, `✓` on success, `✗` and the error on failure, each
// with the node's elapsed time on an injected clock, the output when verbose. Most rows run on a
// non-TTY writer, the path the CLI takes off a terminal: no spinner, only each node's final line.
// Also: the spinner a TTY writer gets, its elapsed time counting up, `pause`/`resume`, and `warn`'s
// line above the spinners, held while paused. The CLI-run tests drive a CODE-only flow through
// `yian run`: `--verbose` prints the output, `--quiet` streams no events, and the worker count is
// read from the run log's `RunStarted` record.

import { afterEach, beforeEach, describe, expect, jest, test } from "bun:test";
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

/**
 * A reporter writing to an in-memory writer, a TTY one when `isTTY`, on a clock the test moves
 * with `tick(ms)`; and what it was given.
 */
function sinkReporter(options: { verbose?: boolean; isTTY?: boolean; columns?: number; rows?: number } = {}) {
  const chunks: string[] = [];
  const out: Writer = {
    write: async (text) => void chunks.push(text),
    isTTY: options.isTTY ?? false,
    columns: options.columns,
    rows: options.rows,
  };
  let clock = 0;
  const reporter = new ProgressReporter(out, { verbose: options.verbose ?? false, env: noColor, now: () => clock });
  const tick = (ms: number) => void (clock += ms);
  return { reporter, chunks, tick, text: () => chunks.join("") };
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
  expect(text()).toBe("✓ frame [0s]\n");
});

test("failure renders cross and error", () => {
  const { reporter, text } = sinkReporter();
  reporter.handle(started("verdict"));
  const failure = new Failure({ code: "node.execute", message: "boom", nodeId: "verdict", causeType: "ValueError" });
  reporter.handle(new NodeFailed({ failure }));
  expect(text()).toBe("✗ verdict [0s]\n    boom\n");
});

describe("model and tool text with escape sequences", () => {
  const raw = "out\x1b]52;c;aGk=\x07put\x1b[2J";
  const run = (isTTY: boolean) => {
    const { reporter, text } = sinkReporter({ verbose: true, isTTY });
    reporter.handle(started("a"));
    reporter.handle(succeeded("a", raw));
    reporter.handle(started("b"));
    const failure = new Failure({ code: "node.execute", message: raw, nodeId: "b", causeType: "ValueError" });
    reporter.handle(new NodeFailed({ failure }));
    reporter.warn(raw);
    reporter.stop();
    return text();
  };

  test("is stripped on a TTY: output, failure message and warning", () => {
    const text = run(true);
    expect(text.match(/output/g)).toHaveLength(3);
    expect(text).not.toContain("\x1b]");
    expect(text).not.toContain("\x07");
    expect(text).not.toContain("\x1b[2J");
  });

  test("is written exactly off a TTY", () => {
    expect(run(false)).toBe(`✓ a [0s]\n    ${raw}\n✗ b [0s]\n    ${raw}\nwarning: ${raw}\n`);
  });
});

test("routed renders check with handle", () => {
  // A router (CASE) emits NodeRouted, not NodeSucceeded: the reporter still prints a completion
  // line naming the chosen handle.
  const { reporter, text } = sinkReporter();
  reporter.handle(started("gate"));
  reporter.handle(new NodeRouted({ nodeId: "gate", handle: "yes" }));
  expect(text()).toBe("✓ gate → yes [0s]\n");
});

test("verbose prints output", () => {
  const { reporter, text } = sinkReporter({ verbose: true });
  reporter.handle(started("frame"));
  reporter.handle(succeeded("frame", "line one\nline two"));
  reporter.handle(started("count"));
  reporter.handle(succeeded("count", { n: 3n, tags: ["a"] }));
  expect(text()).toBe('✓ frame [0s]\n    line one\n    line two\n✓ count [0s]\n    {"n":3,"tags":["a"]}\n');
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
  expect(text()).toBe("✓ polish#0/critic [0s]\n");
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
  expect(tty.chunks).toEqual(["\r\x1b[K⠋ polish~1 [0s]", "\r\x1b[K"]);
  jest.advanceTimersByTime(400);
  expect(tty.chunks).toHaveLength(2);
});

test("on a TTY a running node draws a spinner the finish line replaces", () => {
  jest.useFakeTimers();
  const { reporter, chunks, tick } = sinkReporter({ isTTY: true });
  reporter.handle(started("frame"));
  expect(chunks).toEqual(["\r\x1b[K⠋ frame [0s]"]);
  jest.advanceTimersByTime(80);
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ frame [0s]");
  tick(4_000);
  reporter.handle(started("other"));
  // One spinner line per running node, each with its own elapsed time.
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ frame [4s]\n⠙ other [0s]");
  tick(1_500);
  reporter.handle(succeeded("frame", "ok"));
  // The finish line erases both spinner lines, then the remaining one is drawn again.
  expect(chunks.at(-2)).toBe("\r\x1b[K\x1b[A\x1b[K✓ frame [5s]\n");
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ other [1s]");
  reporter.handle(succeeded("other", "ok"));
  expect(chunks.at(-1)).toBe("\r\x1b[K✓ other [1s]\n");
  const count = chunks.length;
  jest.advanceTimersByTime(400);
  expect(chunks).toHaveLength(count); // nothing running: the redraw stopped
});

test("on a TTY the elapsed time counts up between events", () => {
  // A long model call sends no events: the redraw alone moves the time on.
  jest.useFakeTimers();
  const { reporter, chunks, tick } = sinkReporter({ isTTY: true });
  reporter.handle(started("think"));
  tick(12_400);
  jest.advanceTimersByTime(80);
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ think [12s]");
  tick(60_000);
  jest.advanceTimersByTime(80);
  expect(chunks.at(-1)).toBe("\r\x1b[K⠹ think [1m12s]");
  reporter.stop();
});

test("a spinner line too wide for the terminal drops its elapsed time whole, then cuts the id", () => {
  jest.useFakeTimers();
  // The text after `⠋ ` gets the columns but three, keeping one spare, so the line can't wrap.
  const fits = sinkReporter({ isTTY: true, columns: 15 });
  fits.reporter.handle(started("abcdef"));
  fits.tick(12_000);
  jest.advanceTimersByTime(80);
  expect(fits.chunks.at(-1)).toBe("\r\x1b[K⠙ abcdef [12s]");
  fits.reporter.stop();

  const tagCut = sinkReporter({ isTTY: true, columns: 12 });
  tagCut.reporter.handle(started("abcdef"));
  tagCut.tick(12_000);
  jest.advanceTimersByTime(80);
  expect(tagCut.chunks.at(-1)).toBe("\r\x1b[K⠙ abcdef");
  tagCut.reporter.stop();

  const idCut = sinkReporter({ isTTY: true, columns: 12 });
  idCut.reporter.handle(started("abcdefghij"));
  expect(idCut.chunks).toEqual(["\r\x1b[K⠋ abcdefghi"]);
  idCut.reporter.stop();
});

test("each node's duration runs from its own start, frozen at its end", () => {
  const { reporter, text, tick } = sinkReporter();
  reporter.handle(started("a"));
  tick(2_000);
  reporter.handle(started("b"));
  tick(3_000);
  reporter.handle(succeeded("a", "x"));
  tick(65_000);
  const failure = new Failure({ code: "node.execute", message: "boom", nodeId: "b", causeType: "ValueError" });
  reporter.handle(new NodeFailed({ failure }));
  tick(3_600_000);
  reporter.handle(started("gate"));
  tick(3_725_000);
  reporter.handle(new NodeRouted({ nodeId: "gate", handle: "yes" }));
  expect(text()).toBe("✓ a [5s]\n✗ b [1m08s]\n    boom\n✓ gate → yes [1h02m]\n");
});

test("a node that finishes without a start seen prints no duration", () => {
  const { reporter, text } = sinkReporter();
  reporter.handle(succeeded("frame", "ok"));
  expect(text()).toBe("✓ frame\n");
});

test("pause and resume stop and restart the redraw", () => {
  jest.useFakeTimers();
  const { reporter, chunks } = sinkReporter({ isTTY: true });
  reporter.handle(started("ask"));
  reporter.pause();
  expect(chunks).toEqual(["\r\x1b[K⠋ ask [0s]", "\r\x1b[K"]);
  jest.advanceTimersByTime(400);
  expect(chunks).toHaveLength(2);
  // A node finishing while paused prints its line and draws no spinner.
  reporter.handle(started("other"));
  reporter.handle(succeeded("other", "ok"));
  expect(chunks.slice(2)).toEqual(["✓ other [0s]\n"]);
  reporter.resume();
  expect(chunks.at(-1)).toBe("\r\x1b[K⠋ ask [0s]");
  jest.advanceTimersByTime(80);
  expect(chunks.at(-1)).toBe("\r\x1b[K⠙ ask [0s]");
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
  expect(tty.chunks).toEqual(["\r\x1b[K⠋ frame [0s]", "\r\x1b[Kwarning: w\n", "\r\x1b[K⠋ frame [0s]"]);
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
  expect(chunks).toEqual(["\r\x1b[K⠋ frame [0s]", "\r\x1b[K"]);
  reporter.resume();
  expect(chunks.slice(2)).toEqual(["warning: w\n", "\r\x1b[K⠋ frame [0s]"]);
  reporter.pause();
  reporter.warn("x");
  const count = chunks.length;
  reporter.stop();
  expect(chunks.slice(count)).toEqual(["warning: x\n"]);
});

test("a console line prints on its own line above the spinners, which come back below it", () => {
  jest.useFakeTimers();
  const { reporter, chunks } = sinkReporter({ isTTY: true });
  reporter.handle(started("c"));
  reporter.around(() => chunks.push("hello from code\n"));
  expect(chunks).toEqual(["\r\x1b[K⠋ c [0s]", "\r\x1b[K", "hello from code\n", "\r\x1b[K⠋ c [0s]"]);
  reporter.stop();
});

test("a console line and a warning while paused are held, in order, until resume", () => {
  jest.useFakeTimers();
  const { reporter, chunks } = sinkReporter({ isTTY: true });
  reporter.handle(started("c"));
  reporter.pause();
  reporter.around(() => chunks.push("one\n"));
  reporter.warn("w");
  reporter.around(() => chunks.push("two\n"));
  expect(chunks).toEqual(["\r\x1b[K⠋ c [0s]", "\r\x1b[K"]);
  reporter.resume();
  expect(chunks.slice(2)).toEqual(["one\n", "warning: w\n", "two\n", "\r\x1b[K⠋ c [0s]"]);
  reporter.stop();
});

test("off a TTY a console line is printed as it comes", () => {
  const { reporter, chunks } = sinkReporter();
  reporter.handle(started("c"));
  reporter.around(() => chunks.push("hello\n"));
  expect(chunks).toEqual(["hello\n"]);
});

test("a spinner block taller than the screen keeps to one row less, its last row the rest", () => {
  jest.useFakeTimers();
  const { reporter, chunks } = sinkReporter({ isTTY: true, rows: 5, columns: 80 });
  for (let i = 0; i < 20; i++) reporter.handle(started(`n${i}`));
  expect(chunks.at(-1)).toBe("\r\x1b[K\x1b[A\x1b[K\x1b[A\x1b[K\x1b[A\x1b[K⠋ n0 [0s]\n⠋ n1 [0s]\n⠋ n2 [0s]\n⠋ … and 17 more running");
  jest.advanceTimersByTime(80);
  // The erase moves up exactly the rows drawn, never past the top of the block.
  expect(chunks.at(-1)).toStartWith(`\r\x1b[K${"\x1b[A\x1b[K".repeat(3)}⠙ n0 [0s]`);
  reporter.stop();
});

// ---------- the flags of `yian run` ----------

const FLOW = "id: f\nname: f\nnodes:\n  a: {kind: code, output: !str, code: return 'hello'}\noutput: ${a.output}\n";

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
  expect(stderr).toMatch(/✓ a \[\d+s\]\n    hello\n/);
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

test("on a TTY a CODE body's console line prints above the spinners, and console comes back after", async () => {
  const flow = join(tmp, "conlog.yaml");
  const wait = "await new Promise((r) => setTimeout(r, 200))";
  const code = `|\n      ${wait};\n      console.log('hello from code');\n      return 'ok';`;
  writeFileSync(flow, `id: c\nname: c\nnodes:\n  c:\n    kind: code\n    output: !str\n    code: ${code}\noutput: \${c.output}\n`);
  const seen: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void seen.push(`out ${text}`) },
    stderr: { write: async (text) => void seen.push(`err ${text}`), isTTY: true, columns: 80 },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "HOME" ? tmp : name === "NO_COLOR" ? "1" : undefined),
  };
  const log = console.log;
  const logged = (...args: unknown[]) => void seen.push(`log ${args.join(" ")}`);
  console.log = logged;
  try {
    expect(await main(["run", flow, "--log", join(tmp, "run.jsonl")], io)).toBe(0);
    expect(console.log).toBe(logged);
  } finally {
    console.log = log;
  }
  const at = seen.indexOf("log hello from code");
  // The spinner line was erased just before, and is drawn again just after.
  expect(seen[at - 1]).toBe("err \r\x1b[K");
  expect(seen[at + 1]).toMatch(/^err \r\x1b\[K. c \[0s\]$/);
  expect(seen.at(-1)).toBe("out ok\n");
});
