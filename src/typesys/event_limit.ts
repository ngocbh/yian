/**
 * The built-in value of `YIAN_MAX_EVENT_BYTES`, the cap on one event's run-log JSON. `defaults.ts`
 * reads it for the entry's default, so the number has one source, and layers that may not import
 * `defaults.ts` (`nodes/contracts.ts`) read the built-in value here.
 *
 * Imports nothing.
 */

/** Largest event sent to a run's inbox, host call's pending step or completed answer, by default. */
export const MAX_EVENT_BYTES = 1024 * 1024;
