/**
 * Mirrors `tests/engine/test_no_builtin_defaults.py`: the `resolve_model_config` test; the others
 * go with the cascade, `compile/llm_cascade.ts`, and `runFlow`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { resolveModelConfig } from "../../src/llm_clients/connection.ts";

const FIXES = ["llm_config.provider/model", "--provider/--model", "yian auth <provider>"];

// Each test's environment: a private store in a fresh directory and no default provider or
// model, as Python's `no_environment_defaults` leaves it.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-no-defaults-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

test("python api resolve model config without model", () => {
  let error: unknown;
  try {
    resolveModelConfig({}, { env });
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(RangeError);
  const message = (error as RangeError).message;
  expect(message).toStartWith("No model is set");
  for (const fix of FIXES) expect(message).toContain(fix);
});
