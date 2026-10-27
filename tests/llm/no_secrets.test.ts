/** No JSON file under `tests/llm` (cassettes, goldens) holds anything shaped like a secret. */

import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { findSecrets } from "../../src/llm/testing.ts";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const JSON_FILES = readdirSync(ROOT, { recursive: true, encoding: "utf8" })
  .filter((name) => name.endsWith(".json"))
  .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

test("there are JSON files to check", () => {
  // Otherwise the secret check would pass vacuously.
  expect(JSON_FILES.length).toBeGreaterThan(0);
});

for (const name of JSON_FILES) {
  test(`${name} has no secrets`, () => {
    expect(findSecrets(readFileSync(join(ROOT, name), "utf8"))).toEqual([]);
  });
}
