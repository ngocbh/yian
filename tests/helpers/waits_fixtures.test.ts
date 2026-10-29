/** `started`: a scenario log past the run start, in a dir the preload removes. */

import { expect, test } from "bun:test";
import { dirname, sep } from "node:path";

import { started, tool, waitFlow } from "./waits_fixtures.ts";

test("a started log lives in the test's own dir, which the preload removes", () => {
  const { path, log } = started(waitFlow(tool("heard", "take")));
  log.close();
  // The preload's per-test dir holds the test's HOME.
  expect(path.startsWith(dirname(process.env.HOME!) + sep)).toBe(true);
});
