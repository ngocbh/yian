/**
 * The other process of `host_api.test.ts`'s lock tests, which holds a run while the test calls it.
 *
 * Run as `bun --no-install tests/helpers/yian-host-api-child-hold.ts <log>` (or with Node). It runs
 * `processChatFlow()` at `<log>`. Its `gate` tool writes `locked` on stdout, reads one line from
 * stdin and returns, so the run holds the log's lock until the test writes that line. The run
 * then ends waiting on its take and the process exits 0. It exits 2 after 20 seconds in any case.
 *
 * It also exports `processChatFlow`, which the test and `yian-host-api-child-send.ts` build the
 * same flow with. It imports no test fixture, since those import `bun:test`.
 */
import { fileURLToPath } from "node:url";

import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";

/**
 * START, with an `Inbox(str)` input `m`, feeds a TOOL `take: all` from `m` (`heard`) and a host
 * tool `gate`; END returns `heard`'s output.
 */
export function processChatFlow(): ComposedFlow {
  const str = Type.scalar(ValueKind.STRING);
  const start = new StartNode(START_ID, { params: [new Parameter("m", { contract: new ValueContract(inboxType(str)) })] });
  const heard = new ToolNode("heard", { toolId: "take" });
  heard.params = [new Parameter("from"), new Parameter("take")];
  const gate = new ToolNode("gate", { toolId: "gate" });
  const outputs = [new FlowOutput("heard", "${heard.output}")];
  return new ComposedFlow(
    new CompiledFlow(
      new Map<string, Node>([
        [START_ID, start],
        [heard.id, heard],
        [gate.id, gate],
        [END_ID, EndNode.value(END_ID, { outputName: "heard" })],
      ]),
      [
        new Edge("start-heard", START_ID, heard.id),
        new Edge("start-gate", START_ID, gate.id),
        new Edge("heard-end", heard.id, END_ID, { inputGroup: "heard" }),
      ],
      {
        outputs,
        wiring: new Map([
          [heard.id, new Map<string, unknown>([["from", "${input.m}"], ["take", "all"]])],
          [END_ID, new Map<string, unknown>([["heard", outputs[0]!.binding]])],
        ]),
      },
    ),
  );
}

/** The environment both processes run with: no `YIAN_*` values. */
export const processEnv = (): undefined => undefined;

/** Write `locked`, then wait for one stdin line, then stop reading stdin. */
function holdUntilLine(): Promise<string> {
  process.stdout.write("locked\n");
  return new Promise((resolve) => {
    process.stdin.once("data", () => {
      process.stdin.pause();
      process.stdin.destroy();
      resolve("opened");
    });
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  // A bound, so a stuck run can't outlive its test.
  setTimeout(() => process.exit(2), 20_000).unref();
  const gate = new FunctionTool({
    name: "gate",
    description: "holds the run until the test writes a line",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: holdUntilLine,
  });
  const result = await runFlow(processChatFlow(), {}, { log: process.argv[2]!, env: processEnv, tools: { gate } });
  process.exit(result.status === "waiting" ? 0 : 1);
}
