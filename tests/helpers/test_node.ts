/**
 * The Node binary the Node rows run on, shared by every test file that has them. It imports no
 * `bun:test`, so it also runs built on Node.
 */

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";

/** The oldest Node the `engines` field allows, as `[major, minor]`. */
const NODE_FLOOR = [22, 18] as const;

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set value that
 * can't run, a Node older than the `engines` floor, or a missing `built` file throws, failing the
 * importing test file, so a typo can't pass as a run with the Node rows skipped.
 */
export function testNode(...built: string[]): string | undefined {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  const [floorMajor, floorMinor] = NODE_FLOOR;
  if (major < floorMajor || (major === floorMajor && minor < floorMinor)) {
    throw new Error(
      `YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need ${floorMajor}.${floorMinor} or later`,
    );
  }
  for (const file of built) {
    if (!existsSync(file)) throw new Error(`${file} is missing; run bun run build first`);
  }
  return node;
}
