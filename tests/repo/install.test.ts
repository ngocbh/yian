/**
 * `install.sh` and `uninstall.sh` at the repository root. Run from a checkout, install puts a
 * `yian` wrapper for that checkout in a bin folder, and that folder on PATH through the login
 * shell's rc files. Piped (`curl ... | bash`), it first clones the repository, or updates the clone
 * it made. Uninstall takes the wrapper and the PATH lines back out.
 *
 * Each test has a temp `HOME` and an explicit env, so nothing outside its temp dir changes. The
 * checkout tests run this test's Bun on this checkout, where the scripts'
 * `bun install --frozen-lockfile` changes nothing. The piped tests clone a fixture repository from
 * a local path and run a stub `bun`, and a stub `curl` stands in for Bun's installer, so nothing
 * touches the network.
 */

import { afterAll, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { VERSION } from "../../src/platform.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const TEMP: string[] = [];
afterAll(() => {
  for (const dir of TEMP) rmSync(dir, { recursive: true, force: true });
});

/** A temp folder holding a home, and the bin folder inside that home (not created). */
function setup(): { base: string; home: string; bin: string } {
  const base = mkdtempSync(join(tmpdir(), "yian-install-"));
  TEMP.push(base);
  const home = join(base, "home");
  mkdirSync(home);
  return { base, home, bin: join(home, ".local", "bin") };
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

/** Every rc file install.sh may write, relative to the home. */
const RC_FILES = [".bashrc", ".profile", ".bash_profile", ".bash_login", ".zshrc", ".zprofile", ".config/fish/config.fish"];

/** How many times the installer's PATH line is in `rc`. */
function pathLines(rc: string): number {
  if (!existsSync(rc)) return 0;
  return readFileSync(rc, "utf8")
    .split("\n")
    .filter((line) => line.includes("yian install.sh")).length;
}

/** A temp home with a stub `bun` and `curl`, and a fixture repository to clone. */
interface Fixture {
  base: string;
  home: string;
  bin: string;
  /** The folder of stub commands, first on PATH. */
  stubs: string;
  /** A copy of the stub `bun`, for a stub Bun installer to put in place. */
  stubBun: string;
  /** The fixture repository, cloned from its path. */
  remote: string;
  /** Where install.sh clones by default. */
  clone: string;
  /** The stubs' calls, a line each: the stub bun's folder and arguments, or `curl` and its arguments. */
  log: string;
}

const GIT_ENV = {
  PATH: "/usr/bin:/bin",
  GIT_CONFIG_NOSYSTEM: "1",
  GIT_CONFIG_GLOBAL: "/dev/null",
  GIT_AUTHOR_NAME: "Fixture",
  GIT_AUTHOR_EMAIL: "fixture@example.com",
  GIT_COMMITTER_NAME: "Fixture",
  GIT_COMMITTER_EMAIL: "fixture@example.com",
};

/** Runs git in `cwd` with no user or system config, and returns its trimmed stdout. */
function git(cwd: string, ...args: string[]): string {
  const result = spawnSync("git", args, { cwd, env: { ...GIT_ENV, HOME: cwd }, encoding: "utf8", timeout: 30_000 });
  if (result.status !== 0) throw new Error(`git ${args.join(" ")} failed: ${result.stderr}`);
  return result.stdout.trim();
}

/** Writes an executable script. */
function writeScript(path: string, text: string) {
  writeFileSync(path, text);
  chmodSync(path, 0o755);
}

/** Commits a new file named `file`, holding `message`, to `repo`'s current branch. */
function commit(repo: string, file: string, message: string) {
  writeFileSync(join(repo, file), `${message}\n`);
  git(repo, "add", file);
  git(repo, "commit", "--quiet", "-m", message);
}

function fixture(): Fixture {
  const { base, home, bin } = setup();
  const log = join(base, "calls.log");
  const stubs = join(base, "stubs");
  mkdirSync(stubs);
  // Bun 1.4.2, whose install writes a node_modules folder and which runs a CLI by printing what it was asked to run.
  const stubBun = join(base, "stub-bun");
  writeScript(
    stubBun,
    `#!/bin/sh
printf '%s %s\\n' "$(pwd -P)" "$*" >>'${log}'
case $1 in
  --version) echo 1.4.2 ;;
  install) mkdir -p node_modules/dep && echo >node_modules/dep/index.js ;;
  *) shift 3; echo "ran $*" ;;
esac
`,
  );
  copyFileSync(stubBun, join(stubs, "bun"));
  chmodSync(join(stubs, "bun"), 0o755);
  writeScript(join(stubs, "curl"), `#!/bin/sh\necho "curl $*" >>'${log}'\nexit 22\n`);
  const remote = join(base, "remote");
  mkdirSync(join(remote, "src", "cli"), { recursive: true });
  writeFileSync(join(remote, "src", "cli", "bin.ts"), "// the fixture's CLI\n");
  writeFileSync(join(remote, "package.json"), '{ "name": "fixture" }\n');
  writeFileSync(join(remote, ".gitignore"), "node_modules/\n.env\n");
  git(remote, "init", "--quiet", "--initial-branch=main");
  git(remote, "add", ".");
  git(remote, "commit", "--quiet", "-m", "first");
  return { base, home, bin, stubs, stubBun, remote, clone: join(home, ".local", "share", "yian"), log };
}

/**
 * Runs `script` the way `curl ... | bash -s -- args` does, read from stdin so that `$0` is the
 * shell, with the stubs first on PATH. It starts in the home unless `cwd` says otherwise.
 */
function pipe(
  script: string,
  fx: Fixture,
  args: string[] = [],
  extra: Record<string, string> = {},
  cwd: string = fx.home,
) {
  return spawnSync("sh", ["-s", "--", ...args], {
    input: readFileSync(join(ROOT, script)),
    cwd,
    env: { HOME: fx.home, SHELL: "/bin/bash", PATH: `${fx.stubs}:/usr/bin:/bin`, ...extra },
    encoding: "utf8",
    timeout: 60_000,
  });
}

/** Installs the fixture piped, and fails the test unless that succeeds. */
function install(fx: Fixture, args: string[] = [], extra: Record<string, string> = {}) {
  const result = pipe("install.sh", fx, ["--repo", fx.remote, ...args], extra);
  expect(result.stderr).not.toContain("✗");
  expect(result.status).toBe(0);
  return result;
}

/** The stub calls logged so far. */
function calls(fx: Fixture): string {
  return existsSync(fx.log) ? readFileSync(fx.log, "utf8") : "";
}

describe.if(process.versions.bun !== undefined)("install.sh", () => {
  test("writes a yian wrapper that runs this checkout and puts its folder on PATH once", () => {
    const { home, bin } = setup();
    for (let i = 0; i < 2; i++) {
      const result = run("install.sh", home);
      expect(result.stderr).not.toContain("✗");
      expect(result.status).toBe(0);
    }
    const version = spawnSync(join(bin, "yian"), ["--version"], { env: { PATH: "/usr/bin:/bin" }, encoding: "utf8" });
    expect(version.stdout.trim()).toBe(VERSION);
    expect(pathLines(join(home, ".bashrc"))).toBe(1);
  });

  test("leaves the rc files alone when the bin folder is already on PATH", () => {
    const { home, bin } = setup();
    const result = run("install.sh", home, { PATH: `${bin}:/usr/bin:/bin` });
    expect(result.status).toBe(0);
    expect(existsSync(join(bin, "yian"))).toBe(true);
    for (const rc of RC_FILES) expect(existsSync(join(home, rc))).toBe(false);
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

describe.if(process.versions.bun !== undefined)("install.sh, piped", () => {
  test("clones the repository into ~/.local/share/yian and installs that clone", () => {
    const fx = fixture();
    // Started inside this checkout: a piped run still clones, whatever folder it starts in.
    const result = pipe("install.sh", fx, ["--repo", fx.remote], {}, ROOT);
    expect(result.stderr).not.toContain("✗");
    expect(result.status).toBe(0);
    expect(git(fx.clone, "rev-parse", "HEAD")).toBe(git(fx.remote, "rev-parse", "HEAD"));
    expect(existsSync(join(fx.clone, ".git", "yian-install"))).toBe(true);
    expect(calls(fx)).toContain(`${realpathSync(fx.clone)} install --frozen-lockfile`);
    expect(calls(fx)).not.toContain("curl");
    const cli = join(fx.clone, "src", "cli", "bin.ts");
    const yian = (env: Record<string, string>) =>
      spawnSync(join(fx.bin, "yian"), ["--version"], { env: { PATH: "/usr/bin:/bin", ...env }, encoding: "utf8" });
    expect(yian({}).stdout.trim()).toBe(`ran ${cli} --version`);
    // A YIAN_BUN the caller sets wins over the Bun the wrapper was written with.
    expect(yian({ YIAN_BUN: "/bin/echo" }).stdout.trim()).toBe(
      `--no-install --no-env-file --config=/dev/null ${cli} --version`,
    );
    expect(result.stdout).toContain("yian auth anthropic");
  });

  test("run again, fast-forwards the clone it made", () => {
    const fx = fixture();
    install(fx);
    commit(fx.remote, "NEWS", "second");
    const result = install(fx);
    expect(git(fx.clone, "rev-parse", "HEAD")).toBe(git(fx.remote, "rev-parse", "HEAD"));
    expect(result.stdout).toContain(`Updated ${fx.clone}`);
  });

  test("takes the folder from --dir, the branch from --branch and the repository from YIAN_REPO_URL", () => {
    const fx = fixture();
    git(fx.remote, "switch", "--quiet", "--create", "next");
    commit(fx.remote, "NEXT", "next");
    git(fx.remote, "switch", "--quiet", "main");
    const dir = join(fx.base, "elsewhere", "yian");
    const result = pipe("install.sh", fx, ["--dir", dir, "--branch", "next"], { YIAN_REPO_URL: fx.remote });
    expect(result.stderr).not.toContain("✗");
    expect(result.status).toBe(0);
    expect(git(dir, "rev-parse", "--abbrev-ref", "HEAD")).toBe("next");
    expect(existsSync(join(dir, "NEXT"))).toBe(true);
    expect(existsSync(fx.clone)).toBe(false);
    expect(readFileSync(join(fx.bin, "yian"), "utf8")).toContain(join(dir, "src", "cli", "bin.ts"));
  });

  test("refuses to update a clone with uncommitted changes and leaves it as it was", () => {
    const fx = fixture();
    install(fx);
    const head = git(fx.clone, "rev-parse", "HEAD");
    writeFileSync(join(fx.clone, "package.json"), "{}\n");
    commit(fx.remote, "NEWS", "second");
    const result = pipe("install.sh", fx, ["--repo", fx.remote]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(`${fx.clone} has uncommitted changes`);
    expect(git(fx.clone, "rev-parse", "HEAD")).toBe(head);
    expect(readFileSync(join(fx.clone, "package.json"), "utf8")).toBe("{}\n");
  });

  test("refuses a folder that holds a clone of another repository", () => {
    const fx = fixture();
    install(fx);
    const other = join(fx.base, "other");
    const result = pipe("install.sh", fx, ["--repo", other]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(`${fx.clone} is a clone of ${fx.remote}, not ${other}`);
  });

  test("refuses someone else's yian before it clones or installs anything", () => {
    const fx = fixture();
    mkdirSync(fx.bin, { recursive: true });
    writeFileSync(join(fx.bin, "yian"), "#!/bin/sh\necho mine\n");
    const result = pipe("install.sh", fx, ["--repo", fx.remote]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(join(fx.bin, "yian"));
    expect(readFileSync(join(fx.bin, "yian"), "utf8")).toContain("echo mine");
    expect(existsSync(fx.clone)).toBe(false);
    expect(calls(fx)).toBe("");
  });

  test("installs Bun with Bun's installer when it finds none", () => {
    const fx = fixture();
    rmSync(join(fx.stubs, "bun"));
    // Bun's installer, as the stub curl serves it, puts the stub Bun in ~/.bun/bin.
    writeScript(
      join(fx.stubs, "curl"),
      `#!/bin/sh\necho "curl $*" >>'${fx.log}'\necho 'mkdir -p "$HOME/.bun/bin" && cp ${fx.stubBun} "$HOME/.bun/bin/bun"'\n`,
    );
    install(fx);
    expect(calls(fx)).toContain("curl -fsSL https://bun.sh/install");
    expect(readFileSync(join(fx.bin, "yian"), "utf8")).toContain(join(fx.home, ".bun", "bin", "bun"));
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
