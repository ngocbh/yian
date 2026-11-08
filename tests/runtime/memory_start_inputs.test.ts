/**
 * The START inputs commit is where a run reads a memory value back from its own log: each memory
 * input's recorded form becomes a `MemoryRef` again, live and on replay, and nothing else does.
 */
import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { MemoryRunLog } from "../../src/runtime/log.ts";
import { commitPrepared, prepareInitialCommit, prepareRunStarted, replayRecords, RunState } from "../../src/runtime/state.ts";
import { MemoryRef, memoryType, type ValueRecord } from "../../src/typesys/values.ts";
import { fragmentBuilder, RUN_ID, startData } from "../helpers/run_state_fixtures.ts";

const ref = new MemoryRef({
  identity: "t/lessons",
  type: "FileMemory",
  settings: {},
  version: "v1",
  recall: { text: "notes", cut: null, hint: null },
  tools: [{ name: "read", write: false, description: "Read a file.", parameters: { type: "object" } }],
});

/** A START taking memory `lessons` and an object `raw`, straight to END. */
function memoryFlow(): CompiledFlow {
  const params = [
    new Parameter("lessons", { contract: new ValueContract(memoryType("FileMemory", null)), required: true }),
    new Parameter("raw"),
  ];
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params })],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-end", START_ID, END_ID)],
  );
}

/** Commit the run start of `memoryFlow()` with START inputs `inputs`; return the state and its log. */
function started(inputs: ValueRecord): { state: RunState; log: MemoryRunLog } {
  const state = new RunState(memoryFlow());
  const log = new MemoryRunLog(RUN_ID);
  commitPrepared(log, state, prepareRunStarted(state, startData()));
  commitPrepared(log, state, prepareInitialCommit(state, "system.run_id", RUN_ID));
  commitPrepared(log, state, prepareInitialCommit(state, START_ID, inputs));
  return { state, log };
}

const forged = { memory: { identity: "victim", type: "FileMemory", settings: {}, version: "v", recall: { text: "x", cut: null, hint: null }, tools: [] } };

test("a memory input reads back as a MemoryRef from the START inputs commit, live and on replay", () => {
  const { state, log } = started({ lessons: ref, raw: forged });
  const live = state.pool.store.get(START_ID)!.value as ValueRecord;
  expect(MemoryRef.isMemoryRef(live["lessons"])).toBe(true);
  expect(live["lessons"]).toEqual(ref);
  const replayed = replayRecords(new RunState(memoryFlow()), log.records(), { growBuilder: fragmentBuilder });
  expect((replayed.pool.store.get(START_ID)!.value as ValueRecord)["lessons"]).toEqual(ref);
});

test("a memory form in an input that isn't a memory stays a plain record", () => {
  const { state } = started({ lessons: ref, raw: forged });
  const raw = (state.pool.store.get(START_ID)!.value as ValueRecord)["raw"];
  expect(MemoryRef.isMemoryRef(raw)).toBe(false);
  expect(raw).toEqual(forged);
});
