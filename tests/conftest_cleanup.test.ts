/**
 * The preload's temp dirs and environment: a run puts every test's dir inside one `yian-test-*`
 * dir under `$TMPDIR`, a test file's own `afterAll` still sees its test's home, and the run
 * removes that dir when it ends; a test never sees the developer's provider credentials. A fixture
 * file runs in a child of the same runtime with the preload and its own `TMPDIR`.
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

/**
 * Run the test file `fixture` in `work` under the preload, in a child of this runtime with
 * `TMPDIR` at `tmp` and `extra` added to this process's environment; return what it printed.
 */
function runFixture(work: string, tmp: string, fixture: string, extra: Record<string, string> = {}): string {
  fs.writeFileSync(join(work, "fixture.test.ts"), fixture);
  const conftest = join(import.meta.dirname, "conftest.ts");
  // Run outside the package root, so Bun's `bunfig.toml` doesn't load the preload a second time.
  const args =
    runtime === "bun"
      ? ["test", "--preload", conftest, "./fixture.test.ts"]
      : ["--import", join(import.meta.dirname, "node_runner.ts"), "--import", conftest, "--test", "fixture.test.ts"];
  // Without `NODE_TEST_CONTEXT` removed, a nested `node --test` reports to this runner and runs
  // no tests.
  const env: Record<string, string | undefined> = { ...process.env, TMPDIR: tmp, ...extra };
  delete env.NODE_TEST_CONTEXT;
  // Its own timeout, since Node's test timeout can't interrupt a sync spawn.
  const result = spawnSync(process.execPath, args, { cwd: work, env, encoding: "utf8", timeout: 50_000 });
  const output = `${result.stdout}\n${result.stderr}`;
  expect(result.status, `status ${result.status}, signal ${result.signal}, error ${result.error}\n${output}`).toBe(0);
  return output;
}

test("a run keeps its tests' dirs in one yian-test dir and removes it at the end", () => {
  const work = fs.mkdtempSync(join(tmpdir(), "yian-conftest-"));
  try {
    const tmp = join(work, "tmp");
    fs.mkdirSync(tmp);
    const output = runFixture(work, tmp, FIXTURE);

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

// The variables the conformance runner strips too, and Azure's AD token.
const CREDENTIALS = {
  OPENAI_API_KEY: "sk-dev-real",
  OPENAI_BASE_URL: "https://dev.example/v1",
  AZURE_OPENAI_ENDPOINT: "https://dev.openai.azure.com/",
  AZURE_OPENAI_API_VERSION: "2025-01-01",
  AZURE_OPENAI_AD_TOKEN: "dev-ad-token",
};

test("a test sees none of the developer's provider credentials, and keeps the YIAN_TEST_ gates", () => {
  const work = fs.mkdtempSync(join(tmpdir(), "yian-conftest-"));
  try {
    const tmp = join(work, "tmp");
    fs.mkdirSync(tmp);
    const names = JSON.stringify([...Object.keys(CREDENTIALS), "YIAN_TEST_LIVE"]);
    const fixture = `import { test } from "bun:test";
test("env", () => {
  for (const name of ${names}) console.log(\`ENV \${name}=\${process.env[name] ?? "unset"}\`);
});
`;
    const output = runFixture(work, tmp, fixture, { ...CREDENTIALS, YIAN_TEST_LIVE: "1" });
    const seen = Object.fromEntries([...output.matchAll(/ENV (\w+)=(\S+)/g)].map((match) => [match[1], match[2]]));
    const unset = Object.fromEntries(Object.keys(CREDENTIALS).map((name) => [name, "unset"]));
    expect(seen).toEqual({ ...unset, YIAN_TEST_LIVE: "1" });
  } finally {
    fs.rmSync(work, { recursive: true, force: true });
  }
}, 60_000);
