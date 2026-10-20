// Ported from tests/engine/test_cli_progress.py (the reporter tests): per-node progress, `✓` on
// success, `✗` and the error on failure, the output when verbose. They run on a non-TTY writer,
// the path the CLI takes off a terminal: no spinner, only each node's final line. Added here: the
// spinner a TTY writer gets, and `pause`/`resume`. The CLI-run tests come with `yian run`.

import { afterEach, expect, jest, test } from "bun:test";

import { ProgressReporter } from "../../src/cli/run.ts";
import type { Writer } from "../../src/cli/utils.ts";
import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { NodeExpanded, NodeFailed, NodeRouted, NodeStarted, NodeSucceeded } from "../../src/events.ts";

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
