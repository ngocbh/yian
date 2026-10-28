/**
 * The wait scenarios (`scenarioLogs`): each log replays to the live state on every prefix; the
 * records-only fold agrees with the reducer's; `inspectRun` without a flow agrees with a replay on
 * every cut; and each scenario's own rules hold at the records that matter. `foldRunLog` yields a
 * send's value at its commit and a take's at its commit once its success rebuilds it.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { foldRunLog, inspectRun, type RunInspection } from "../../src/run.ts";
import { START_ID } from "../../src/compile/model.ts";
import { iterRunRecords, type NodeStepRecord, PoolCommitRecord, readRunLog, recordTimeMs } from "../../src/runtime/log.ts";
import {
  DeliveryRefused,
  prepareInboxClosed,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareNodeWoken,
  prepareRunWaiting,
  prepareSend,
  prepareTakeCommit,
  type RunState,
} from "../../src/runtime/state.ts";
import { applyWaitsRecord, decideWait, dueTimeouts, unreadEvents, wakeableNodes, WaitsState } from "../../src/runtime/waits.ts";
import { NO_ENV, reducerSnapshot } from "./run_state_fixtures.ts";
import {
  askPending,
  call,
  DEADLINE,
  replayed,
  type ScenarioLog,
  scenarioLogs,
  sleepEntry,
  takeEntry,
  traceParts,
} from "./waits_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-wait-scenarios-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const SCENARIOS = scenarioLogs();
const plain = (name: string) => SCENARIOS.find((scenario) => scenario.name === name && !scenario.learnable)!;

/** The state replayed through `scenario`'s records up to and including the one marked `mark`. */
const stateAt = (scenario: ScenarioLog, mark: string): RunState => replayed(scenario.flow, scenario.path, scenario.marks[mark]! + 1);

let cuts = 0;
/** A copy of `scenario`'s log holding its first `count` records. */
function cut(scenario: ScenarioLog, count: number): string {
  const lines = readFileSync(scenario.path, "utf8").split("\n").filter((line) => line !== "");
  const path = join(tmp, `cut-${++cuts}.jsonl`);
  writeFileSync(path, lines.slice(0, count).map((line) => `${line}\n`).join(""));
  return path;
}

/** The parts of an inspection the records and a replay must agree on. */
function progress(inspection: RunInspection) {
  const { status, inFlight, crashedAttempts, failure, waits, ready, wakeAt } = inspection;
  return { status, inFlight, crashedAttempts: [...crashedAttempts], failure, waits, ready, wakeAt };
}

/** The records-only inspection of `scenario`'s first `count` records. */
const inspectCut = (scenario: ScenarioLog, count: number) => inspectRun(cut(scenario, count));

describe.each(SCENARIOS.map((scenario) => [`${scenario.name}${scenario.learnable ? " (learnable)" : ""}`, scenario] as const))(
  "scenario %s",
  (_name, scenario) => {
    test("every prefix replays, and the whole log replays to the live state", () => {
      const count = readRunLog(scenario.path).length;
      for (let k = 1; k < count; k++) replayed(scenario.flow, scenario.path, k);
      const state = replayed(scenario.flow, scenario.path);
      expect(reducerSnapshot(state)).toEqual(reducerSnapshot(scenario.state));
      if (scenario.learnable) expect(traceParts(state.trace!)).toEqual(traceParts(scenario.state.trace!));
    });

    test("the records-only fold is the reducer's", () => {
      const w = new WaitsState();
      for (const envelope of readRunLog(scenario.path)) applyWaitsRecord(w, envelope);
      expect(w.toJSON()).toEqual(scenario.state.waits.toJSON());
    });

    test.skipIf(scenario.learnable)("inspectRun reads every cut past the run start from the records as a replay does", () => {
      const envelopes = readRunLog(scenario.path);
      // A replay refuses a log that ends inside its run start; its last record is the START inputs' commit.
      const started = envelopes.findIndex(({ record }) => record instanceof PoolCommitRecord && record.key === START_ID) + 1;
      for (let k = started; k <= envelopes.length; k++) {
        const path = cut(scenario, k);
        expect([k, progress(inspectRun(path))]).toEqual([k, progress(inspectRun(path, { flow: scenario.composed, env: NO_ENV }))]);
      }
    });
  },
);

describe("the scenarios' own rules", () => {
  test("a: a waiting take is ready once a send arrives; the run ends with nothing left in the inbox", () => {
    const a = plain("a");
    expect([inspectCut(a, a.marks.runWaiting! + 1).status, inspectCut(a, a.marks.runWaiting! + 1).ready]).toEqual(["waiting", false]);
    expect([inspectCut(a, a.marks.send! + 1).status, inspectCut(a, a.marks.send! + 1).ready]).toEqual(["waiting", true]);
    expect(a.state.inboxEvents.size).toBe(0);
    expect(unreadEvents(a.state.waits)).toEqual(new Map());
    expect(readFileSync(plain("a-inline").path, "utf8")).not.toContain('"node_waiting"');
  });

  test("b: after a crash past the wake, the take commits its assigned event, not one sent while down", () => {
    const b = plain("b");
    const state = stateAt(b, "restarted");
    const request = { tool: "take", inbox: "msgs", mode: "one", timeoutSeconds: null } as const;
    expect(decideWait(state.waits, call("heard"), request, 0)).toEqual({
      kind: "take",
      commit: { inbox: "msgs", events: [b.marks.send!], closed: false },
    });
    for (const events of [[b.marks.downtime!], []]) {
      expect(() => prepareTakeCommit(state, "heard", { inbox: "msgs", events, closed: false })).toThrow(RangeError);
    }
    expect(() => prepareNodeWaiting(state, "heard", { callIds: [call("heard")], waits: [takeEntry("heard")] })).toThrow(
      /already registered/,
    );
    expect(state.crashedAttempts.get("heard")).toBe(1);
    expect(unreadEvents(b.state.waits)).toEqual(new Map([["msgs", [b.marks.downtime!]]]));
  });

  test("b-timeout: a take its timeout woke commits nothing after a crash, though an event is unread", () => {
    const scenario = plain("b-timeout");
    const state = stateAt(scenario, "restarted");
    expect(() => prepareTakeCommit(state, "heard", { inbox: "msgs", events: [scenario.marks.downtime!], closed: false })).toThrow(
      RangeError,
    );
    expect(unreadEvents(scenario.state.waits)).toEqual(new Map([["msgs", [scenario.marks.downtime!]]]));
  });

  test("b-sleep: a sleep its deadline woke is done after a crash, with no new wait", () => {
    const state = stateAt(plain("b-sleep"), "restarted");
    expect(decideWait(state.waits, call("nap"), { tool: "sleep", until: DEADLINE, forSeconds: null }, 0)).toEqual({ kind: "slept" });
    expect(() => prepareNodeWaiting(state, "nap", { callIds: [call("nap")], waits: [sleepEntry("nap")] })).toThrow(/already registered/);
  });

  test("c: two host calls wait; answered b then a, both wake in waiting order", () => {
    const c = plain("c");
    expect(inspectCut(c, c.marks.runWaiting! + 1).waits.map((wait) => wait.node)).toEqual(["a", "b"]);
    expect(wakeableNodes(stateAt(c, "answered").waits)).toEqual(["a", "b"]);
    const crash = plain("c-crash");
    expect(() => prepareNodeStep(stateAt(crash, "restarted"), "a", askPending("a"))).toThrow(/already registered/);
  });

  test("d: a sleep waits while another branch runs; it is due at its deadline", () => {
    const d = plain("d");
    expect(() => prepareRunWaiting(stateAt(d, "napping"))).toThrow(/in flight or ready/);
    const state = stateAt(d, "runWaiting");
    const at = recordTimeMs(DEADLINE);
    expect(dueTimeouts(state.waits, at - 1)).toEqual(new Map());
    expect(dueTimeouts(state.waits, at)).toEqual(new Map([["nap", [call("nap")]]]));
    expect(() => prepareNodeWoken(state, "nap")).toThrow(/still waits/);
  });

  test("e: a close while a take waits makes the run ready; later sends are refused and a second close is a no-op", () => {
    const e = plain("e");
    expect(inspectCut(e, e.marks.closed! + 1).ready).toBe(true);
    const state = stateAt(e, "closed");
    expect(state.waits.calls.get(call("heard"))!.closedInbox).toBe(true);
    expect(() => prepareSend(state, "msgs", "late")).toThrow(DeliveryRefused);
    expect(prepareInboxClosed(state, "msgs")).toBeNull();
    expect(e.state.pool.store.get("heard")!.toObject()).toEqual({ events: [], closed: true });
  });
});

test("f: a steer step consumes what it read; a send after it stays unread through a crash", () => {
  const f = plain("f");
  expect(unreadEvents(f.state.waits)).toEqual(new Map([["msgs", [f.marks.sc!]]]));
  expect(unreadEvents(stateAt(f, "steer").waits)).toEqual(new Map());
  expect(f.state.inboxEvents.has(f.marks.sa!)).toBe(false);
});

test("foldRunLog yields the steer step with inbox and from; each from seq's record is the send with its value", async () => {
  const f = plain("f");
  const { records } = await foldRunLog(f.path, { flow: f.composed, env: NO_ENV });
  const step = records[f.marks.steer!]!.entry.envelope.record as NodeStepRecord;
  expect(step.step).toEqual({ kind: "steer", inbox: "msgs", from: [BigInt(f.marks.sa!), BigInt(f.marks.sb!)] });
  expect(records[f.marks.sa!]!.value!.toObject()).toBe("a");
  expect(records[f.marks.sb!]!.value!.toObject()).toBe("b");
  // A reader that skips step kinds it doesn't know still decodes every record.
  expect([...iterRunRecords(f.path)]).toHaveLength(readRunLog(f.path).length);
});

test("foldRunLog yields a send's value at its commit, and a take's at its commit once its success rebuilds it", async () => {
  const a = plain("a");
  const { records } = await foldRunLog(a.path, { flow: a.composed, env: NO_ENV });
  expect(records[a.marks.send!]!.value!.toObject()).toBe("hi");
  expect(records[a.marks.take!]!.value!.toObject()).toEqual({ events: ["hi"], closed: false });
});
