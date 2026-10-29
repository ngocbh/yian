/**
 * The chat's child runs: each driven in process by its own driver over a held clock, from flows
 * loaded confined to a temporary root.
 */
import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ChildRuns, type ChildRunsOptions, type ChildStatus } from "../../src/cli/chat_runs.ts";
import { loadConfinedText } from "../../src/cli/confined_load.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { advance, inspectRun } from "../../src/run.ts";
import { readRunLog, type RunStartedRecord } from "../../src/runtime/log.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";

const TIMEOUT = 10_000;
const CHAT = "chat-1";

let root: string;
let runs: ChildRuns | null;

beforeEach(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-runs-")));
  runs = null;
});

afterEach(async () => {
  await runs?.dispose();
  rmSync(root, { recursive: true, force: true });
});

/** A `ChildRuns` over the temporary root, its logs under `<root>/runs`, on a held clock. */
function childRuns(options: Partial<ChildRunsOptions> = {}): ChildRuns {
  runs = new ChildRuns({
    root,
    logDir: join(root, "runs"),
    env: NO_ENV,
    llmConfig: {},
    scheduler: heldScheduler().scheduler,
    chatRunId: CHAT,
    skillsFor: () => null,
    ...options,
  });
  return runs;
}

/** Load `text` as the flow `<root>/<name>`. */
function flow(text: string, name = "child.yaml"): Promise<ComposedFlow> {
  return loadConfinedText(root, join(root, name), text, NO_ENV);
}

/** Resolve once `check` holds, polling each turn of the event loop. */
async function until(check: () => boolean): Promise<void> {
  while (!check()) await new Promise((resolve) => setTimeout(resolve, 1));
}

/** Child `id`'s first status from now on that isn't `running`. */
async function settled(children: ChildRuns, id: string): Promise<ChildStatus> {
  let status = children.status(id);
  while (status.status === "running") status = await children.next(id, status.seq);
  return status;
}

const GREET = `id: greet
name: greet
input:
  who: !str
nodes:
  hi:
    kind: code
    input:
      who: \${input.who}
    output: str
    code: |
      return "hi " + inputs["who"];
output: \${hi.output}
`;

test(
  "a started child runs in the background and its next status is its output",
  async () => {
    const children = childRuns();
    const started = await children.start(await flow(GREET), { who: "ada" });
    expect(started).toEqual({ id: "r1", log: join(root, "runs", `${CHAT}-r1`, "log.jsonl") });
    expect(children.status("r1")).toMatchObject({ run: "r1", status: "running", log: started.log });
    const status = await children.next("r1");
    expect(status).toMatchObject({ run: "r1", status: "succeeded", output: "hi ada", log: started.log });
    const inspected = inspectRun(started.log);
    expect(inspected.runId).toBe(`${CHAT}-r1`);
    expect(inspected.status).toBe("succeeded");
    const start = readRunLog(started.log)[0]!.record as RunStartedRecord;
    expect(start.parent).toMatchObject({ runId: CHAT });
  },
  TIMEOUT,
);

test(
  "each child gets the next id and a run id under the chat's",
  async () => {
    const children = childRuns();
    const greet = await flow(GREET);
    const first = await children.start(greet, { who: "a" });
    const second = await children.start(greet, { who: "b" });
    expect([first.id, second.id]).toEqual(["r1", "r2"]);
    expect((await settled(children, "r2")).output).toBe("hi b");
    expect((await settled(children, "r1")).output).toBe("hi a");
    expect(inspectRun(second.log).runId).toBe(`${CHAT}-r2`);
  },
  TIMEOUT,
);

test("next on an id that names no child throws", () => {
  expect(() => childRuns().status("r9")).toThrow(new RangeError("no run 'r9' in this chat"));
});

/** Sleep a minute, then succeed with "rested". */
const NAP = `id: nap
name: nap
nodes:
  nap:
    kind: tool
    tool_id: sleep
    input:
      for: "1m"
  done:
    kind: code
    input:
      slept: \${nap.output}
    output: str
    code: |
      return "rested";
output: \${done.output}
`;

/** NAP after a CODE node that keeps its worker busy for 300 ms, so the engine loops meanwhile. */
const BUSY_NAP = NAP.replace(
  "nodes:\n",
  `nodes:
  busy:
    kind: code
    output: str
    code: |
      const until = Date.now() + 300;
      while (Date.now() < until) {}
      return "1m";
`,
).replace('for: "1m"', "for: ${busy.output}");

test(
  "a sleeping child wakes at its wakeAt with nothing else moving it",
  async () => {
    const { scheduler, open } = heldScheduler();
    const children = childRuns({ scheduler });
    await children.start(await flow(NAP), {});
    const waiting = await children.next("r1");
    expect(waiting).toMatchObject({ status: "waiting", wake_at: "2026-01-01T00:01:00.000Z" });
    expect(waiting.waits).toEqual([expect.objectContaining({ node: "nap", tool: "sleep" })]);
    open();
    expect(await children.next("r1")).toMatchObject({ status: "succeeded", output: "rested" });
  },
  TIMEOUT,
);

test(
  "a nudge while the engine loops moves the child once more after the pass, never alongside it",
  async () => {
    const { scheduler, open } = heldScheduler();
    let advances = 0;
    const children = childRuns({
      scheduler,
      advance: (options) => {
        advances += 1;
        return advance(options);
      },
    });
    await children.start(await flow(BUSY_NAP), {});
    children.nudge("r1");
    children.nudge("r1");
    expect(await settled(children, "r1")).toMatchObject({ status: "waiting" });
    // The pass the nudges asked for runs once the start's pass has ended.
    await until(() => advances === 1);
    open();
    expect(await children.next("r1")).toMatchObject({ status: "succeeded", output: "rested" });
    expect(advances).toBe(2);
  },
  TIMEOUT,
);
