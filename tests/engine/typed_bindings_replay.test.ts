/**
 * Typed bindings across a durable run log: replay, resume, fingerprint, and recovery: a port of
 * `tests/engine/test_typed_bindings_replay.py`.
 *
 * Replay folds recorded node results without binding completed nodes again; a node that runs after
 * resume binds (and type-checks) its inputs then. A declared binding type is part of the flow
 * fingerprint, and END's recorded value is re-validated against the declared output types when the
 * log is recovered.
 *
 * Ported: every row. The CODE functions Python patches onto its codefns module to count calls are
 * inline JS CODE here, and the rows read which CODE nodes ran from the result's `NodeStarted`
 * events (a finished log publishes none on resume), where Python counts the calls. A host stop is
 * the `ask` handler aborting the run's signal and throwing its reason, where Python raises
 * `_HostStop`; the run rejects with that reason. The gate is a TOOL `ask` where Python's is a
 * HUMAN_INPUT node (TS yian has no HUMAN_INPUT kind). A refused resume is a `RangeError` (a
 * `TypeCheckError` for the recovered END value), where Python raises `ValueError`. Ints read as
 * `bigint`s (decision 4a). Resume compiles the flow the run recorded, so a changed binding type
 * is an edit of the recorded text (Python passes the changed flow).
 *
 * The `resume/typed_bindings_replay__*` cases crash the log after `ask` starts and give the
 * answer to the retried gate, and their consumer returns its input undoubled, so they don't cover
 * `test_typed_consumer_after_resume_rejects_a_mistyped_answer` or
 * `test_typed_consumer_after_resume_accepts_a_matching_answer`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunResult, advance, runFlow } from "../../src/run.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { editRecordedFlow, rejected, startedIds } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-typed-bindings-replay-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

const TYPED_FLOW = `
id: typed-replay
name: typed_replay
defs:
  bump:
    input:
      rounds: !int
    nodes:
      step:
        kind: code
        code: return inputs["rounds"] + 1
        input:
          rounds: \${input.rounds}
        output: int
    output:
      rounds: \${step.output}
nodes:
  p:
    kind: code
    code: return 4
    output: int
  use:
    kind: code
    code: return inputs["qty"] * 10
    input:
      qty: !int \${p.output}
    output: int
  loop:
    kind: loop
    call: bump
    times: 2
    input:
      rounds: !int 0
output:
  total: !int \${use.output}
  rounds: \${loop.output.rounds}
`;

/** The CODE nodes `result` started, by name (`bump` for each loop occurrence's `step`), sorted. */
const codeRuns = (result: RunResult) =>
  startedIds(result)
    .flatMap((id) => (id === "p" || id === "use" ? [id] : id.endsWith("/step") ? ["bump"] : []))
    .sort();

test("resuming a completed typed run replays without running code", async () => {
  const path = freshLog();
  const first = await runFlow(await loadFlow(TYPED_FLOW, { env }), {}, { env, log: path });
  expect(first.failure).toBeNull();
  expect(first.status).toBe("succeeded");
  expect(first.output).toEqual({ total: 40n, rounds: 2n });
  expect(codeRuns(first)).toEqual(["bump", "bump", "p", "use"]);

  const resumed = await advance({ log: path, env });

  expect(resumed.status).toBe(first.status);
  expect(resumed.output).toEqual(first.output);
  expect(codeRuns(resumed)).toEqual([]);
});

const GATE_FLOW = `
id: typed-gate-replay
name: typed_gate_replay
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      prompt: "How many?"
  use:
    kind: code
    input:
      n: !int \${ask.output}
    code: |
      return inputs["n"] * 2
output: \${use.output}
`;

/** Interrupt the gate before its answer, then resume the log with `answer`. */
async function interruptThenResume(answer: unknown): Promise<RunResult> {
  const path = freshLog();
  const interruption = new Error("interrupt before human answer");
  const host = new AbortController();
  const interrupt = () => {
    host.abort(interruption);
    throw interruption;
  };

  const caught = await rejected(
    runFlow(await loadFlow(GATE_FLOW, { env }), {}, { env, log: path, signal: host.signal, tools: { ask: interrupt } }),
  );
  expect(caught).toBe(interruption);

  return advance({ log: path, env, tools: { ask: () => answer } });
}

test("typed consumer after resume rejects a mistyped answer", async () => {
  const result = await interruptThenResume("three");

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("use");
});

test("typed consumer after resume accepts a matching answer", async () => {
  const result = await interruptThenResume(3);

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(6n);
});

test("resume rejects a changed declared binding type", async () => {
  const path = freshLog();
  expect((await runFlow(await loadFlow(TYPED_FLOW, { env }), {}, { env, log: path })).status).toBe("succeeded");
  const changed = TYPED_FLOW.replace("qty: !int ${p.output}", "qty: !float ${p.output}");
  expect(changed).not.toBe(TYPED_FLOW);
  // Resume compiles the recorded flow, so the change is to the recorded text.
  editRecordedFlow(path, () => changed);

  const error = await rejected(advance({ log: path, env }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("flow fingerprint mismatch");
});

test("resume rejects a recorded end value outside the declared output", async () => {
  const flow = `
id: typed-end
name: typed_end
nodes:
  n:
    kind: code
    code: |
      return 5
output:
  total: !int \${n.output}
`;
  const path = freshLog();
  const first = await runFlow(await loadFlow(flow, { env }), {}, { env, log: path });
  expect(first.failure).toBeNull();
  expect(first.status).toBe("succeeded");
  expect(first.output).toEqual({ total: 5n });

  const lines = readFileSync(path, "utf8").split("\n").filter((line) => line !== "");
  let edited = 0;
  for (const [i, line] of lines.entries()) {
    const record = JSON.parse(line);
    const payload = record.payload;
    if (record.kind === "pool_commit" && payload.node_id === "__end__") {
      expect(payload).toEqual({ node_id: "__end__", key: "__end__", kind: "object", value: { total: 5 } });
      payload.value.total = "five";
      lines[i] = JSON.stringify(record);
      edited++;
    }
  }
  expect(edited).toBe(1);
  writeFileSync(path, lines.join("\n") + "\n");

  const error = await rejected(advance({ log: path, env }));

  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toContain("does not match declared type int");
});
