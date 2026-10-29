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
import { inspectRun } from "../../src/run.ts";
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

/** The status after `seen` (default: the current one) that isn't a pass still running. */
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
