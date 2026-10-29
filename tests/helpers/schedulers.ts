/**
 * Schedulers for tests that drive a run's clock by hand.
 */
import { type Scheduler, virtualScheduler } from "../../src/platform.ts";

/**
 * The virtual scheduler with every sleep held until `open` is called, so a prompt or a driver the
 * run's `wakeAt` would wake stays put meanwhile; once open, sleeps run on the virtual clock.
 */
export function heldScheduler(): { scheduler: Scheduler; open: () => void } {
  const base = virtualScheduler(Date.parse("2026-01-01T00:00:00Z"));
  const { promise: gate, resolve: open } = Promise.withResolvers<void>();
  const scheduler: Scheduler = {
    now: () => base.now(),
    async sleep(ms, signal) {
      await new Promise<void>((resolve, reject) => {
        if (signal?.aborted) return reject(signal.reason);
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        void gate.then(resolve);
      });
      return base.sleep(ms, signal);
    },
  };
  return { scheduler, open };
}
