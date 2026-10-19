/**
 * Ports the terminal checks of `tests/cli/test_terminal_broker.py` and the prompt checks of
 * `tests/engine/test_cli_prompt.py` that drive a real widget: the scenarios of
 * `terminal_both_runtimes.ts`, run here on Bun.
 */

import { describe, test } from "bun:test";

import { SCENARIOS } from "./terminal_both_runtimes.ts";

describe("the terminal scenarios", () => {
  for (const [name, run] of Object.entries(SCENARIOS)) test(name, run, 30_000);
});
