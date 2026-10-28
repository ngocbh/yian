/**
 * `tests/` mirrors `src/`: its top level holds one folder per `src/` folder, the folders for
 * other kinds of test (`e2e`, `repo`, `docs`, `examples`, `conformance`) and their support
 * (`helpers`, `fixtures`), the tests of `src/*.ts` files, and the two runner files. A folder or
 * helper anywhere else fails here, so the layout doesn't drift back.
 */

import { expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const TESTS = fileURLToPath(new URL("..", import.meta.url));
const SRC = fileURLToPath(new URL("../../src/", import.meta.url));

const EXTRA_FOLDERS = ["conformance", "docs", "e2e", "examples", "fixtures", "helpers", "repo"];
const ROOT_FILES = ["conftest.ts", "node_runner.ts"];

const entries = readdirSync(TESTS, { withFileTypes: true });

test("the top-level folders of tests/ are the src/ folders and the fixed extras", () => {
  const srcFolders = readdirSync(SRC, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name);
  const allowed = new Set([...srcFolders, ...EXTRA_FOLDERS]);
  const stray = entries.filter((e) => e.isDirectory() && !allowed.has(e.name)).map((e) => e.name);
  expect(stray).toEqual([]);
});

test("the top level of tests/ holds only test files and the two runner files", () => {
  const stray = entries
    .filter((e) => !e.isDirectory() && !e.name.endsWith(".test.ts") && !ROOT_FILES.includes(e.name))
    .map((e) => e.name);
  expect(stray).toEqual([]);
});
