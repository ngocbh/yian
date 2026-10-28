#!/usr/bin/env node
/**
 * The `yian` bin: checks that the runtime meets the floor in `package.json`'s `engines`, then runs
 * the CLI.
 *
 * The CLI is imported only after the check, because on an older runtime it can fail to load or,
 * lacking `import.meta.main`, exit 0 without running. So this file uses nothing newer than the
 * check needs.
 */

/** The oldest Node and Bun yian runs on, as in `package.json`'s `engines`. */
const FLOOR = { node: "22.18", bun: "1.4" };

/** Whether the dotted version `version` is at least `floor`, compared part by part as numbers. */
function atLeast(version: string, floor: string): boolean {
  const have = version.split(".").map(Number);
  const need = floor.split(".").map(Number);
  for (let i = 0; i < need.length; i++) {
    const part = have[i] ?? 0;
    if (part !== need[i]) return part > need[i]!;
  }
  return true;
}

// Bun reports a Node version too, so a set `bun` decides.
const bun = process.versions.bun;
const [runtime, version] = bun !== undefined ? (["Bun", bun] as const) : (["Node", process.versions.node] as const);
if (!atLeast(version, runtime === "Bun" ? FLOOR.bun : FLOOR.node)) {
  process.stderr.write(
    `Error: yian needs Node ${FLOOR.node} or later, or Bun ${FLOOR.bun} or later; this is ${runtime} ${version}.\n`,
  );
  process.exit(1);
}

const { runEntry } = await import("./index.ts");
await runEntry();
