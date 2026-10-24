/**
 * Small run logs whose nodes wait, as record lists, for tests that fold or index them at every
 * cut. Records only: each list follows the run start (`run_started`, then the START inputs'
 * commit, seqs 0 and 1), so its first record is seq 2. No reducer checks them.
 */
import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import {
  InboxClosedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RunCancelledRecord,
  RunFailedRecord,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
  RunStoppedRecord,
  RunWaitingRecord,
} from "../../src/runtime/log.ts";

export const DEADLINE = "2026-10-24T09:00:00.000Z";

/** The run start: `run_started`, then the START inputs' commit. */
export function runStart(): RunRecord[] {
  return [
    new RunStartedRecord({ flowFingerprint: "a".repeat(64), startWiringOverride: {}, effectiveLlmByStaticPath: {}, numWorkers: 0 }),
    new PoolCommitRecord(START_ID, START_ID, { kind: "object", value: {} }),
  ];
}

/** A `node_waiting` on one take of `inbox`. */
export function takeWait(node: string, mode: "one" | "all" = "one", inbox = "msgs", deadline: string | null = null): NodeWaitingRecord {
  return new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "take", inbox, mode, deadline }] });
}

/** A `node_waiting` on one sleep until `deadline`. */
export function sleepWait(node: string, deadline = DEADLINE): NodeWaitingRecord {
  return new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "sleep", inbox: null, mode: null, deadline }] });
}

/** A send of `value` to `inbox`. */
export function send(value: string, inbox = "msgs"): PoolCommitRecord {
  return new PoolCommitRecord(START_ID, `${START_ID}.${inbox}`, { kind: "str", value });
}

const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "END is unreachable", nodeId: "gate" });
const take = (node: string, events: number[], closed = false) =>
  new PoolCommitRecord(node, node, null, { take: { inbox: "msgs", events, closed } });
const value = (node: string) => new PoolCommitRecord(node, node, { kind: "str", value: node });

/** Each scenario's records after the run start; seqs count from 2. */
export const WAIT_LOGS: Readonly<Record<string, readonly RunRecord[]>> = {
  "a take waits for a send": [
    new NodeStartedRecord("heard"), // 2
    takeWait("heard"), // 3
    new RunWaitingRecord(["heard"]), // 4
    send("hi"), // 5
    new NodeWokenRecord("heard"), // 6
    take("heard", [5]), // 7
    new NodeSucceededRecord("heard", 7), // 8
    new NodeStartedRecord(END_ID), // 9
    value(END_ID), // 10
    new NodeSucceededRecord(END_ID, 10), // 11
  ],
  "a sleep times out": [
    new NodeStartedRecord("nap"),
    sleepWait("nap"),
    new RunWaitingRecord(["nap"]),
    new NodeWokenRecord("nap", ["r/nap"]),
    value("nap"), // 6
    new NodeSucceededRecord("nap", 6),
  ],
  "a host call is answered": [
    new NodeStartedRecord("ask"),
    new NodeStepRecord("ask", 1, { kind: "pending", call_id: "r/ask/1", tool: "ask", arguments: { q: "ok?" }, deadline: DEADLINE }),
    new NodeWaitingRecord("ask", ["r/ask/1"]),
    new RunWaitingRecord(["ask"]),
    new NodeStepRecord("ask", 2, { kind: "tool_result", call_id: "r/ask/1", output: { kind: "str", value: "yes" } }),
    new NodeWokenRecord("ask"),
    value("ask"), // 8
    new NodeSucceededRecord("ask", 8),
  ],
  "two takes, a close, then a cancel": [
    new NodeStartedRecord("a"),
    new NodeStartedRecord("b"),
    takeWait("a", "all"),
    takeWait("b", "one"),
    send("x"),
    new InboxClosedRecord("msgs"),
    new RunWaitingRecord(["a", "b"]),
    new RunCancelledRecord(),
  ],
  "END unreachable while a take waits, then run_failed": [
    new NodeStartedRecord("heard"),
    takeWait("heard"),
    new NodeStartedRecord("gate"),
    new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }),
    new RunWaitingRecord(["heard"]),
    new RunFailedRecord(unreachable),
  ],
  "END succeeds while a take waits; the woken take crashes": [
    new NodeStartedRecord("heard"), // 2
    takeWait("heard", "one", "msgs", DEADLINE), // 3
    new NodeStartedRecord(END_ID), // 4
    value(END_ID), // 5
    new NodeSucceededRecord(END_ID, 5), // 6
    new RunWaitingRecord(["heard"]), // 7
    send("hi"), // 8
    new NodeWokenRecord("heard"), // 9
  ],
  "a stop and a resume, then a wait": [
    new NodeStartedRecord("work"),
    new NodeStartedRecord("heard"),
    new RunStoppedRecord("signal", "SIGINT", ["work", "heard"]),
    new RunResumedRecord(["work", "heard"], 0),
    new NodeStartedRecord("work"),
    new NodeStartedRecord("heard"),
    takeWait("heard"),
    value("work"), // 9
    new NodeSucceededRecord("work", 9),
    new RunWaitingRecord(["heard"]),
  ],
  "an inline take reads queued sends": [
    send("x"), // 2
    send("y"), // 3
    new NodeStartedRecord("heard"), // 4
    take("heard", [2]), // 5
    new NodeSucceededRecord("heard", 5), // 6
    new NodeStartedRecord("next"), // 7
    takeWait("next", "one", "other"), // 8
    new RunWaitingRecord(["next"]), // 9
  ],
};
