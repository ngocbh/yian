/**
 * The prober of `host_api.test.ts`'s lock tests, which probes a run's writer lock in a loop while
 * the test calls the run.
 *
 * Run as `bun --no-install tests/helpers/yian-host-api-child-probe.ts <log>` (or with Node). It
 * writes `probing` on stdout after its first probe, then probes `isRunLogLocked(<log>)` until its
 * stdin ends, and exits 0. It exits 2 after 20 seconds in any case.
 */
import { isRunLogLocked } from "../../src/runtime/log.ts";

// A bound, so a stuck test can't leave the prober running.
setTimeout(() => process.exit(2), 20_000).unref();
const log = process.argv[2]!;
let open = true;
process.stdin.on("end", () => {
  open = false;
});
process.stdin.resume();
isRunLogLocked(log);
process.stdout.write("probing\n");
while (open) {
  for (let i = 0; i < 200; i++) isRunLogLocked(log);
  await new Promise((resolve) => setImmediate(resolve));
}
