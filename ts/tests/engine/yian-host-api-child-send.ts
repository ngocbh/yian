/**
 * The other process of `host_api.test.ts`'s registry tests, which appends to a run this test's
 * process left waiting.
 *
 * Run as `bun --no-install tests/engine/yian-host-api-child-send.ts <log> <op>` (or with Node).
 * `send` sends `from child` to inbox `m` of a `processChatFlow()` run; `cancel` cancels a run whose
 * flow files were recorded. It writes the result as one JSON line and exits 0, or writes the error
 * and exits 1. It exits 2 after 20 seconds in any case.
 */
import { cancel, send } from "../../src/run.ts";
import { logJson } from "../../src/typesys/json.ts";
import { processChatFlow, processEnv } from "./yian-host-api-child-hold.ts";

// A bound, so a stuck call can't outlive its test.
setTimeout(() => process.exit(2), 20_000).unref();
const [log, op] = process.argv.slice(2);
try {
  const result =
    op === "send"
      ? await send(log!, "m", "from child", { env: processEnv, flow: processChatFlow() })
      : await cancel(log!, { env: processEnv });
  process.stdout.write(`${logJson(result)}\n`);
  process.exit(0);
} catch (error) {
  process.stdout.write(`${String(error)}\n`);
  process.exit(1);
}
