/**
 * `install.sh` and `uninstall.sh` at the repository root: install puts a `yian` wrapper for this
 * checkout in a bin folder and that folder on PATH through the shell's rc file; uninstall takes
 * both back out. Each test runs the scripts with a temp `HOME` and this test's Bun as `YIAN_BUN`,
 * so nothing outside the temp dir changes. On an already installed checkout the scripts'
 * `bun install --frozen-lockfile` changes nothing.
 */

import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { VERSION } from "../../src/platform.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** A temp home with an empty bin folder inside it. */
function setup(): { home: string; bin: string } {
  const home = mkdtempSync(join(tmpdir(), "yian-install-"));
  const bin = join(home, ".local", "bin");
  return { home, bin };
}

/** Runs `script` from the repo root with an explicit env: the temp home, bash, a plain PATH. */
function run(script: string, home: string, extra: Record<string, string> = {}) {
  return spawnSync("sh", [join(ROOT, script)], {
    env: {
      HOME: home,
      SHELL: "/bin/bash",
      PATH: "/usr/bin:/bin",
      YIAN_BUN: process.execPath,
      ...extra,
    },
    encoding: "utf8",
    timeout: 60_000,
  });
}

/** How many times the installer's PATH line is in `rc`. */
function pathLines(rc: string): number {
  if (!existsSync(rc)) return 0;
  return readFileSync(rc, "utf8")
    .split("\n")
    .filter((line) => line.includes("yian install.sh")).length;
}

describe.if(process.versions.bun !== undefined)("install.sh", () => {
  test("writes a yian wrapper that runs this checkout and puts its folder on PATH once", () => {
    const { home, bin } = setup();
    for (let i = 0; i < 2; i++) {
      const result = run("install.sh", home);
      expect(result.stderr).not.toContain("Error");
      expect(result.status).toBe(0);
    }
    const version = spawnSync(join(bin, "yian"), ["--version"], { env: { PATH: "/usr/bin:/bin" }, encoding: "utf8" });
    expect(version.stdout.trim()).toBe(VERSION);
    expect(pathLines(join(home, ".bashrc"))).toBe(1);
  });

  test("leaves the rc file alone when the bin folder is already on PATH", () => {
    const { home, bin } = setup();
    const result = run("install.sh", home, { PATH: `${bin}:/usr/bin:/bin` });
    expect(result.status).toBe(0);
    expect(existsSync(join(bin, "yian"))).toBe(true);
    expect(pathLines(join(home, ".bashrc"))).toBe(0);
  });

  test("takes the bin folder from YIAN_BIN_DIR", () => {
    const { home } = setup();
    const bin = join(home, "tools");
    const result = run("install.sh", home, { YIAN_BIN_DIR: bin });
    expect(result.status).toBe(0);
    expect(existsSync(join(bin, "yian"))).toBe(true);
  });

  test("refuses to overwrite a yian it didn't write", () => {
    const { home, bin } = setup();
    mkdirSync(bin, { recursive: true });
    writeFileSync(join(bin, "yian"), "#!/bin/sh\necho mine\n");
    const result = run("install.sh", home);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(join(bin, "yian"));
    expect(readFileSync(join(bin, "yian"), "utf8")).toContain("echo mine");
  });

  test("refuses a Bun older than 1.4 and names its version", () => {
    const { home, bin } = setup();
    const oldBun = join(home, "old-bun");
    writeFileSync(oldBun, "#!/bin/sh\necho 1.3.9\n");
    chmodSync(oldBun, 0o755);
    const result = run("install.sh", home, { YIAN_BUN: oldBun });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("1.3.9");
    expect(existsSync(join(bin, "yian"))).toBe(false);
  });
});

describe.if(process.versions.bun !== undefined)("uninstall.sh", () => {
  test("removes the wrapper and the PATH line and keeps the rest of the rc file", () => {
    const { home, bin } = setup();
    writeFileSync(join(home, ".bashrc"), "alias ll='ls -l'\n");
    expect(run("install.sh", home).status).toBe(0);
    const result = run("uninstall.sh", home);
    expect(result.status).toBe(0);
    expect(existsSync(join(bin, "yian"))).toBe(false);
    expect(readFileSync(join(home, ".bashrc"), "utf8")).toBe("alias ll='ls -l'\n");
  });

  test("leaves a yian it didn't write in place", () => {
    const { home, bin } = setup();
    mkdirSync(bin, { recursive: true });
    writeFileSync(join(bin, "yian"), "#!/bin/sh\necho mine\n");
    const result = run("uninstall.sh", home);
    expect(result.status).toBe(0);
    expect(readFileSync(join(bin, "yian"), "utf8")).toContain("echo mine");
  });
});
