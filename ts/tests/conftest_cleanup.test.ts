/**
 * The preload's temp dirs: a run puts every test's dir inside one `yian-test-*` dir under
 * `$TMPDIR`, a test file's own `afterAll` still sees its test's home, and the run removes that
 * dir when it ends. A fixture file runs in a child of the same runtime with the preload and its
 * own `TMPDIR`.
 */

import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, sep } from "node:path";
import process from "node:process";

import { runtime } from "../src/platform.ts";

const FIXTURE = `import { afterAll, test } from "bun:test";
import fs from "node:fs";

let home = "";
afterAll(() => console.log(\`AFTERALL_HOME_EXISTS=\${fs.existsSync(home)}\`));
test("first", () => {
  home = process.env.HOME ?? "";
  console.log(\`HOME=\${home}\`);
});
test("second", () => {
  home = process.env.HOME ?? "";
  console.log(\`HOME=\${home}\`);
});
`;

test("a run keeps its tests' dirs in one yian-test dir and removes it at the end", () => {
  const work = fs.mkdtempSync(join(tmpdir(), "yian-conftest-"));
  try {
    const tmp = join(work, "tmp");
    fs.mkdirSync(tmp);
    fs.writeFileSync(join(work, "fixture.test.ts"), FIXTURE);
    const conftest = join(import.meta.dirname, "conftest.ts");
    // Run outside `ts/`, so Bun's `bunfig.toml` doesn't load the preload a second time.
    const args =
      runtime === "bun"
        ? ["test", "--preload", conftest, "./fixture.test.ts"]
        : ["--import", join(import.meta.dirname, "node_runner.ts"), "--import", conftest, "--test", "fixture.test.ts"];
    // Without `NODE_TEST_CONTEXT` removed, a nested `node --test` reports to this runner and runs
    // no tests.
    const env: Record<string, string | undefined> = { ...process.env, TMPDIR: tmp };
    delete env.NODE_TEST_CONTEXT;
    // Its own timeout, since Node's test timeout can't interrupt a sync spawn.
    const result = spawnSync(process.execPath, args, { cwd: work, env, encoding: "utf8", timeout: 50_000 });
    const output = `${result.stdout}\n${result.stderr}`;
    expect(result.status, `status ${result.status}, signal ${result.signal}, error ${result.error}\n${output}`).toBe(0);

    // Unanchored: Node's TAP reporter prints a test's output as `# HOME=...`.
    const homes = [...output.matchAll(/HOME=(\S+)/g)].map((match) => relative(tmp, match[1]!).split(sep));
    expect(homes).toHaveLength(2);
    for (const parts of homes) {
      expect(parts).toHaveLength(3);
      expect(parts[0]).toStartWith("yian-test-");
      expect(parts[2]).toBe("home");
    }
    expect(homes[1]![0]).toBe(homes[0]![0]);
    expect(output).toContain("AFTERALL_HOME_EXISTS=true");
    expect(fs.readdirSync(tmp).filter((name) => name.startsWith("yian-test-"))).toEqual([]);
  } finally {
    fs.rmSync(work, { recursive: true, force: true });
  }
}, 60_000);
