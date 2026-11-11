/**
 * `install.sh` and `uninstall.sh` at the repository root. Run from a checkout, install puts a
 * `yian` wrapper for that checkout in a bin folder, and that folder on PATH through the login
 * shell's rc files. Piped (`curl ... | bash`), it first clones the repository, or updates the clone
 * it made. Uninstall takes the wrapper, the PATH lines and a clone with no local work back out.
 *
 * Each test has a temp `HOME` and an explicit env, so nothing outside its temp dir changes. The
 * checkout tests run this test's Bun on this checkout, where the scripts'
 * `bun install --frozen-lockfile` changes nothing, behind a proxy that refuses every connection.
 * The piped tests clone a fixture repository from a local path and run a stub `bun`, and a stub
 * `curl` stands in for Bun's installer, so nothing touches the network. With `YIAN_TEST_INSTALL=1`,
 * one more test installs a clone of this checkout with real Bun, offline, from the Bun cache that
 * `BUN_INSTALL_CACHE_DIR` names.
 */

import { afterAll, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
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

/** A proxy that refuses every connection, so a Bun that tries to fetch a package fails instead. */
const NO_NETWORK = {
  HTTP_PROXY: "http://127.0.0.1:9",
  HTTPS_PROXY: "http://127.0.0.1:9",
  http_proxy: "http://127.0.0.1:9",
  https_proxy: "http://127.0.0.1:9",
};

/** Runs `script` from the repo root with an explicit env: the temp home, bash, a plain PATH. */
function run(script: string, home: string, extra: Record<string, string> = {}) {
  return spawnSync("sh", [join(ROOT, script)], {
    env: {
      HOME: home,
      SHELL: "/bin/bash",
      PATH: "/usr/bin:/bin",
      YIAN_BUN: process.execPath,
      ...NO_NETWORK,
      ...extra,
    },
    encoding: "utf8",
    timeout: 60_000,
  });
}

/** Every rc file install.sh may write, relative to the home. */
const RC_FILES = [".bashrc", ".profile", ".bash_profile", ".bash_login", ".zshrc", ".zprofile", ".config/fish/config.fish"];

/** How many of the installer's marked PATH lines `rc` holds. */
function pathLines(rc: string): number {
  if (!existsSync(rc)) return 0;
  return readFileSync(rc, "latin1")
    .split("\n")
    .filter((line) => line.endsWith("  # added by yian install.sh")).length;
}

/** The line install.sh adds to an rc file for `shell` to put `bin` on PATH. */
function markedLine(shell: string, bin: string): string {
  const line =
    shell === "fish" ? `fish_add_path --path '${bin}'` : `case ":$PATH:" in *:'${bin}':*) ;; *) export PATH='${bin}':"$PATH" ;; esac`;
  return `${line}  # added by yian install.sh`;
}

/**
 * Writes the wrapper and the PATH line that install.sh wrote before it matched its marks as whole
 * lines, in their exact formats, for a checkout at `checkout`.
 */
function writeEarlierInstall(home: string, bin: string, checkout: string) {
  mkdirSync(bin, { recursive: true });
  writeScript(
    join(bin, "yian"),
    `#!/bin/sh
# Written by yian install.sh: runs the yian checkout at ${checkout} with Bun.
# The flags keep Bun from reading the current folder's .env and bunfig.toml and from installing packages.
exec '/usr/bin/true' --no-install --no-env-file --config=/dev/null '${checkout}/src/cli/bin.ts' "$@"
`,
  );
  writeFileSync(join(home, ".bashrc"), `alias ll='ls -l'\nexport PATH='${bin}':"$PATH"  # added by yian install.sh\n`);
  writeFileSync(join(home, ".zshrc"), `export PATH='${bin}':"$PATH"  # added by yian install.sh\n`);
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
  script: string | Buffer,
  fx: Fixture,
  args: string[] = [],
  extra: Record<string, string> = {},
  cwd: string = fx.home,
) {
  return spawnSync("sh", ["-s", "--", ...args], {
    input: typeof script === "string" ? readFileSync(join(ROOT, script)) : script,
    cwd,
    env: { HOME: fx.home, SHELL: "/bin/bash", PATH: `${fx.stubs}:/usr/bin:/bin`, ...NO_NETWORK, ...extra },
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
    const version = spawnSync(join(bin, "yian"), ["--version"], {
      env: { HOME: home, PATH: "/usr/bin:/bin" },
      encoding: "utf8",
    });
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

  test("refuses to overwrite a yian it didn't write, even one that names yian install.sh", () => {
    const { home, bin } = setup();
    mkdirSync(bin, { recursive: true });
    writeFileSync(join(bin, "yian"), "#!/bin/sh\n# Written by yian install.sh, then changed\necho mine\n");
    const result = run("install.sh", home);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(join(bin, "yian"));
    expect(readFileSync(join(bin, "yian"), "utf8")).toContain("echo mine");
  });

  test("run as sh install.sh in a checkout, installs that checkout", () => {
    const fx = fixture();
    const checkout = join(fx.base, "co");
    mkdirSync(join(checkout, "src", "cli"), { recursive: true });
    writeFileSync(join(checkout, "src", "cli", "bin.ts"), "");
    copyFileSync(join(ROOT, "install.sh"), join(checkout, "install.sh"));
    const result = spawnSync("sh", ["install.sh", "--repo", fx.remote, "--skip-setup"], {
      cwd: checkout,
      env: { HOME: fx.home, SHELL: "/bin/bash", PATH: `${fx.stubs}:/usr/bin:/bin`, ...NO_NETWORK },
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(result.stderr).not.toContain("✗");
    expect(result.stdout).toContain(`Installing yian from the checkout at ${realpathSync(checkout)}`);
    expect(existsSync(fx.clone)).toBe(false);
  });

  test("replaces the wrapper and the PATH line an earlier install.sh wrote", () => {
    const fx = fixture();
    writeEarlierInstall(fx.home, fx.bin, fx.clone);
    const result = install(fx);
    expect(readFileSync(join(fx.bin, "yian"), "utf8").split("\n")[1]).toBe("# Written by yian install.sh");
    expect(readFileSync(join(fx.home, ".bashrc"), "utf8")).toBe(`alias ll='ls -l'\n${markedLine("bash", fx.bin)}\n`);
    expect(pathLines(join(fx.home, ".profile"))).toBe(1);
    expect(result.stdout).toContain(`Replaced the old PATH line in ${join(fx.home, ".bashrc")}`);
  });

  test("swaps an earlier install.sh's PATH line in its own file when the folder is on PATH already, as that line put it", () => {
    const fx = fixture();
    writeEarlierInstall(fx.home, fx.bin, fx.clone);
    install(fx, [], { PATH: `${fx.bin}:${fx.stubs}:/usr/bin:/bin` });
    expect(readFileSync(join(fx.home, ".bashrc"), "utf8")).toBe(`alias ll='ls -l'\n${markedLine("bash", fx.bin)}\n`);
    expect(readFileSync(join(fx.home, ".zshrc"), "utf8")).toBe(`${markedLine("zsh", fx.bin)}\n`);
    expect(existsSync(join(fx.home, ".profile"))).toBe(false);
  });

  test("finds the checkout it is in when CDPATH names another folder of the same name", () => {
    const fx = fixture();
    const checkout = join(fx.base, "co");
    mkdirSync(join(checkout, "src", "cli"), { recursive: true });
    writeFileSync(join(checkout, "src", "cli", "bin.ts"), "");
    copyFileSync(join(ROOT, "install.sh"), join(checkout, "install.sh"));
    mkdirSync(join(fx.base, "elsewhere", "co"), { recursive: true });
    const result = spawnSync("sh", ["co/install.sh", "--repo", fx.remote], {
      cwd: fx.base,
      env: { HOME: fx.home, SHELL: "/bin/bash", PATH: `${fx.stubs}:/usr/bin:/bin`, ...NO_NETWORK, CDPATH: join(fx.base, "elsewhere") },
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(result.stderr).not.toContain("✗");
    expect(result.stdout).toContain(`Installing yian from the checkout at ${realpathSync(checkout)}`);
    expect(existsSync(fx.clone)).toBe(false);
  });

  test("refuses a checkout whose path holds a newline", () => {
    const fx = fixture();
    const checkout = join(fx.base, "co\nx");
    mkdirSync(join(checkout, "src", "cli"), { recursive: true });
    writeFileSync(join(checkout, "src", "cli", "bin.ts"), "");
    copyFileSync(join(ROOT, "install.sh"), join(checkout, "install.sh"));
    const result = spawnSync("sh", [join(checkout, "install.sh")], {
      env: { HOME: fx.home, SHELL: "/bin/bash", PATH: `${fx.stubs}:/usr/bin:/bin`, ...NO_NETWORK },
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("The checkout's path must not hold a newline");
    expect(calls(fx)).toBe("");
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
  test("clones the repository into ~/.local/share/yian and installs that clone, even from a folder that looks like a checkout", () => {
    const fx = fixture();
    // Piped, $0 is the shell's name, `sh`: a folder holding a file of that name, an install.sh and
    // the CLI is still not taken for the checkout to install.
    const decoy = join(fx.base, "decoy");
    mkdirSync(join(decoy, "src", "cli"), { recursive: true });
    for (const file of ["sh", "install.sh", "src/cli/bin.ts"]) writeFileSync(join(decoy, file), "");
    const result = pipe("install.sh", fx, ["--repo", fx.remote], {}, decoy);
    expect(result.stderr).not.toContain("✗");
    expect(result.status).toBe(0);
    expect(git(fx.clone, "rev-parse", "HEAD")).toBe(git(fx.remote, "rev-parse", "HEAD"));
    expect(existsSync(join(fx.clone, ".git", "yian-install"))).toBe(true);
    expect(calls(fx)).toContain(`${realpathSync(fx.clone)} install --frozen-lockfile`);
    expect(calls(fx)).not.toContain(`${realpathSync(decoy)} install`);
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

  test("refuses a clone on a detached HEAD", () => {
    const fx = fixture();
    install(fx);
    git(fx.clone, "switch", "--quiet", "--detach");
    const result = pipe("install.sh", fx, ["--repo", fx.remote]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(`${fx.clone} is not on a branch`);
  });

  test("refuses a folder of files that isn't a git clone, and a file, and changes neither", () => {
    const fx = fixture();
    const dir = join(fx.base, "notes");
    mkdirSync(dir);
    writeFileSync(join(dir, "todo.md"), "mine\n");
    const file = join(fx.base, "file");
    writeFileSync(file, "mine\n");
    for (const [path, error] of [
      [dir, `${dir} holds files but isn't a git clone`],
      [file, `${file} is a file, not a folder`],
    ] as const) {
      const result = pipe("install.sh", fx, ["--repo", fx.remote, "--dir", path]);
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain(error);
    }
    expect(readdirSync(dir)).toEqual(["todo.md"]);
    expect(readFileSync(file, "utf8")).toBe("mine\n");
    expect(calls(fx)).toBe("");
  });

  test("refuses a folder that holds a clone of another repository", () => {
    const fx = fixture();
    install(fx);
    const other = join(fx.base, "other");
    const result = pipe("install.sh", fx, ["--repo", other]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(`${fx.clone} is a clone of ${fx.remote}, not ${other}`);
  });

  test("given a --branch other than the clone's, warns and updates the branch the clone is on", () => {
    const fx = fixture();
    install(fx);
    git(fx.remote, "branch", "next");
    commit(fx.remote, "NEWS", "second");
    const result = install(fx, ["--branch", "next"]);
    expect(result.stdout).toContain(`⚠ ${fx.clone} is on branch main, not next`);
    expect(git(fx.clone, "rev-parse", "--abbrev-ref", "HEAD")).toBe("main");
    expect(git(fx.clone, "rev-parse", "HEAD")).toBe(git(fx.remote, "rev-parse", "main"));
  });

  test("runs nothing when the download stops anywhere in its last line", () => {
    const fx = fixture();
    const text = readFileSync(join(ROOT, "install.sh"));
    const last = text.lastIndexOf("\n", text.length - 2) + 1;
    for (let cut = last; cut < text.length - 1; cut++) {
      pipe(text.subarray(0, cut), fx, ["--repo", fx.remote, "--no-modify-path", "--dir", join(fx.base, "elsewhere")]);
    }
    expect(calls(fx)).toBe("");
    expect(readdirSync(fx.home)).toEqual([]);
    expect(existsSync(join(fx.base, "elsewhere"))).toBe(false);
  });

  test("fails when the installed yian doesn't run", () => {
    const fx = fixture();
    writeScript(join(fx.stubs, "bun"), `#!/bin/sh\ncase $1 in --version) echo 1.4.2 ;; install) ;; *) exit 3 ;; esac\n`);
    const result = pipe("install.sh", fx, ["--repo", fx.remote]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(`✗ ${join(fx.bin, "yian")} --version failed`);
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

  const RELATIVE: Record<string, [string[], Record<string, string>]> = {
    "--dir": [["--dir", "yian"], {}],
    YIAN_INSTALL_DIR: [[], { YIAN_INSTALL_DIR: "yian" }],
    YIAN_BIN_DIR: [[], { YIAN_BIN_DIR: "bin" }],
  };
  for (const [name, [args, env]] of Object.entries(RELATIVE)) {
    test(`refuses a relative ${name} before it installs anything`, () => {
      const fx = fixture();
      const result = pipe("install.sh", fx, ["--repo", fx.remote, ...args], env);
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("must be an absolute path");
      expect(calls(fx)).toBe("");
      expect(readdirSync(fx.home)).toEqual([]);
    });
  }

  test("refuses a folder whose path holds a newline", () => {
    const fx = fixture();
    for (const [args, env] of [
      [["--dir", join(fx.base, "a\nb")], {}],
      [[], { YIAN_BIN_DIR: join(fx.base, "a\nb") }],
    ] as const) {
      const result = pipe("install.sh", fx, ["--repo", fx.remote, ...args], env);
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("must not hold a newline");
    }
    expect(calls(fx)).toBe("");
  });

  test("refuses an empty HOME", () => {
    const fx = fixture();
    const result = pipe("install.sh", fx, ["--repo", fx.remote], { HOME: "" });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("HOME must be an absolute path");
    expect(calls(fx)).toBe("");
  });

  test("never takes a bun from a relative PATH entry, the current folder or a folder inside it", () => {
    const fx = fixture();
    const project = join(fx.base, "project");
    // Where direnv's PATH_add puts a project's own commands.
    const tools = join(project, "node_modules", ".bin");
    mkdirSync(tools, { recursive: true });
    for (const dir of [project, tools]) {
      copyFileSync(fx.stubBun, join(dir, "bun"));
      chmodSync(join(dir, "bun"), 0o755);
    }
    const path = `.:${project}:${tools}:${fx.stubs}:/usr/bin:/bin`;
    const result = pipe("install.sh", fx, ["--repo", fx.remote], { PATH: path }, project);
    expect(result.stderr).not.toContain("✗");
    expect(result.status).toBe(0);
    const wrapper = readFileSync(join(fx.bin, "yian"), "utf8");
    expect(wrapper).toContain(`'${join(fx.stubs, "bun")}'`);
    expect(wrapper).not.toContain(project);
  });

  test("run from the home, takes a bun from a folder inside the home", () => {
    const fx = fixture();
    const tools = join(fx.home, "tools");
    mkdirSync(tools);
    copyFileSync(fx.stubBun, join(tools, "bun"));
    chmodSync(join(tools, "bun"), 0o755);
    install(fx, [], { PATH: `${tools}:/usr/bin:/bin` });
    expect(readFileSync(join(fx.bin, "yian"), "utf8")).toContain(`'${join(tools, "bun")}'`);
  });

  // The installer finds a bun in /usr/bin or /bin before it would install one.
  test.skipIf(existsSync("/usr/bin/bun") || existsSync("/bin/bun"))("installs Bun with Bun's installer when it finds none", () => {
    const fx = fixture();
    rmSync(join(fx.stubs, "bun"));
    // Bun's installer, as the stub curl serves it, puts the stub Bun in ~/.bun/bin.
    writeScript(
      join(fx.stubs, "curl"),
      `#!/bin/sh\necho "curl $*" >>'${fx.log}'\necho 'mkdir -p "$HOME/.bun/bin" && cp ${fx.stubBun} "$HOME/.bun/bin/bun"'\n`,
    );
    const result = install(fx, ["--no-modify-path"]);
    expect(result.stdout).toContain("⚠ Bun's installer adds Bun to PATH in your shell's rc files itself");
    expect(calls(fx)).toContain("curl -fsSL https://bun.sh/install");
    expect(readFileSync(join(fx.bin, "yian"), "utf8")).toContain(join(fx.home, ".bun", "bin", "bun"));
  });
});

describe.if(process.versions.bun !== undefined)("install.sh, PATH", () => {
  const SHELL_FILES: Record<string, string[]> = {
    bash: [".bashrc", ".profile"],
    zsh: [".zshrc", ".zprofile"],
    fish: [".config/fish/config.fish"],
    ksh: [".profile"],
  };
  for (const [shell, files] of Object.entries(SHELL_FILES)) {
    test(`for ${shell}, adds one guarded PATH line to ${files.join(" and ")}, and a second run adds none`, () => {
      const fx = fixture();
      for (let i = 0; i < 2; i++) install(fx, [], { SHELL: `/bin/${shell}` });
      for (const rc of RC_FILES) {
        const want = files.includes(rc) ? `${markedLine(shell, fx.bin)}\n` : null;
        expect([rc, existsSync(join(fx.home, rc)) ? readFileSync(join(fx.home, rc), "utf8") : null]).toEqual([rc, want]);
      }
    });
  }

  test("for bash, also writes a .bash_profile there is, and the lines put the folder on PATH once however many a shell reads", () => {
    const fx = fixture();
    // Its last line has no newline, and stays whole.
    writeFileSync(join(fx.home, ".bash_profile"), "export EDITOR=vi");
    const result = install(fx);
    expect(readFileSync(join(fx.home, ".bash_profile"), "utf8").split("\n")[0]).toBe("export EDITOR=vi");
    expect(pathLines(join(fx.home, ".bash_profile"))).toBe(1);
    expect(existsSync(join(fx.home, ".bash_login"))).toBe(false);
    const sourced = spawnSync(
      "sh",
      ["-c", '. "$HOME/.bash_profile"; . "$HOME/.profile"; . "$HOME/.bashrc"; printf %s "$PATH"'],
      { env: { HOME: fx.home, PATH: "/usr/bin:/bin" }, encoding: "utf8" },
    );
    expect(sourced.stdout.split(":").filter((dir) => dir === fx.bin)).toHaveLength(1);
    expect(result.stdout).toContain(`source '${join(fx.home, ".bashrc")}'`);
  });

  test("leaves alone an rc file that already puts the folder on PATH", () => {
    const fx = fixture();
    const own = 'export PATH="$HOME/.local/bin:$PATH"\n';
    writeFileSync(join(fx.home, ".zshrc"), own);
    install(fx, [], { SHELL: "/bin/zsh" });
    expect(readFileSync(join(fx.home, ".zshrc"), "utf8")).toBe(own);
    expect(pathLines(join(fx.home, ".zprofile"))).toBe(1);
  });

  test("writes the rc files it can and warns about one it can't write", () => {
    const fx = fixture();
    writeFileSync(join(fx.home, ".bashrc"), "alias ll='ls -l'\n");
    chmodSync(join(fx.home, ".bashrc"), 0o444);
    const result = install(fx);
    expect(readFileSync(join(fx.home, ".bashrc"), "utf8")).toBe("alias ll='ls -l'\n");
    expect(pathLines(join(fx.home, ".profile"))).toBe(1);
    expect(result.stdout).toContain(`⚠ Could not write ${join(fx.home, ".bashrc")}`);
  });

  test("with --no-modify-path, writes no rc file and says the folder isn't on PATH", () => {
    const fx = fixture();
    const result = install(fx, ["--no-modify-path"]);
    for (const rc of RC_FILES) expect(existsSync(join(fx.home, rc))).toBe(false);
    expect(result.stdout).toContain(`${fx.bin} is not on PATH`);
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

  test("leaves a yian it didn't write in place, even one that names yian install.sh", () => {
    const { home, bin } = setup();
    mkdirSync(bin, { recursive: true });
    writeFileSync(join(bin, "yian"), "#!/bin/sh\n# Written by yian install.sh, then changed\necho mine\n");
    const result = run("uninstall.sh", home);
    expect(result.status).toBe(0);
    expect(readFileSync(join(bin, "yian"), "utf8")).toContain("echo mine");
  });

  test("keeps every other byte of an rc file, a NUL and a byte that isn't UTF-8 included", () => {
    const fx = fixture();
    const own = Buffer.from("export A=1\n# caf\xe9\nprintf 'a\x00b'\nexport B=2\n", "latin1");
    writeFileSync(join(fx.home, ".bashrc"), own);
    const utf8 = { LANG: "en_US.UTF-8" };
    install(fx, [], utf8);
    expect(pathLines(join(fx.home, ".bashrc"))).toBe(1);
    expect(pipe("uninstall.sh", fx, [], utf8).status).toBe(0);
    expect(readFileSync(join(fx.home, ".bashrc"))).toEqual(own);
  });

  test("leaves a line of the user's that names yian install.sh alone, in install and uninstall", () => {
    const fx = fixture();
    const own = "alias yian-update='sh ~/yian/install.sh'  # re-runs yian install.sh\n";
    writeFileSync(join(fx.home, ".bashrc"), own);
    install(fx);
    expect(pathLines(join(fx.home, ".bashrc"))).toBe(1);
    expect(pipe("uninstall.sh", fx).status).toBe(0);
    expect(readFileSync(join(fx.home, ".bashrc"), "utf8")).toBe(own);
  });

  test("leaves an rc file as it was, and fails, when its new text can't be written in full", () => {
    const fx = fixture();
    writeFileSync(join(fx.home, ".bashrc"), "# a line of the user's\n".repeat(1000));
    install(fx);
    const before = readFileSync(join(fx.home, ".bashrc"));
    const tmp = join(fx.base, "tmp");
    mkdirSync(tmp);
    // A file-size limit, with SIGXFSZ ignored, makes the write fail partway, as a full disk does.
    const result = spawnSync("sh", ["-c", 'trap "" XFSZ; ulimit -f 8; exec sh -s'], {
      input: readFileSync(join(ROOT, "uninstall.sh")),
      cwd: fx.home,
      env: { HOME: fx.home, SHELL: "/bin/bash", PATH: `${fx.stubs}:/usr/bin:/bin`, ...NO_NETWORK, TMPDIR: tmp },
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(`✗ Could not write the new text of ${join(fx.home, ".bashrc")}`);
    expect(result.stdout).not.toContain(`Removed the PATH line from ${join(fx.home, ".bashrc")}`);
    expect(readFileSync(join(fx.home, ".bashrc"))).toEqual(before);
    expect(pathLines(join(fx.home, ".profile"))).toBe(0);
    expect(readdirSync(tmp)).toEqual([]);
  });

  test("goes on past an rc file it can't write, and leaves no temp file", () => {
    const fx = fixture();
    install(fx);
    chmodSync(join(fx.home, ".bashrc"), 0o444);
    const tmp = join(fx.base, "tmp");
    mkdirSync(tmp);
    const result = pipe("uninstall.sh", fx, [], { TMPDIR: tmp });
    expect(result.status).toBe(0);
    expect(result.stdout).toContain(`⚠ Left the PATH line in ${join(fx.home, ".bashrc")}`);
    expect(pathLines(join(fx.home, ".bashrc"))).toBe(1);
    expect(pathLines(join(fx.home, ".profile"))).toBe(0);
    expect(existsSync(fx.clone)).toBe(false);
    expect(readdirSync(tmp)).toEqual([]);
  });

  test("keeps an rc file's last line whole when it had no newline", () => {
    const fx = fixture();
    writeFileSync(join(fx.home, ".bashrc"), "alias ll='ls -l'");
    install(fx);
    expect(pipe("uninstall.sh", fx).status).toBe(0);
    expect(readFileSync(join(fx.home, ".bashrc"), "utf8")).toBe("alias ll='ls -l'\n");
  });

  test("piped, removes the clone install.sh made, the wrapper and every PATH line, and leaves Bun and yian's data", () => {
    const fx = fixture();
    writeFileSync(join(fx.home, ".bashrc"), "alias ll='ls -l'\n");
    install(fx);
    install(fx, [], { SHELL: "/bin/zsh" });
    install(fx, [], { SHELL: "/bin/fish" });
    const kept = [".bun/bin/bun", ".config/yian/auth.json", ".yian/runs/r1/log.jsonl"];
    for (const file of kept) {
      mkdirSync(dirname(join(fx.home, file)), { recursive: true });
      writeFileSync(join(fx.home, file), "mine\n");
    }
    const result = pipe("uninstall.sh", fx);
    expect(result.stderr).not.toContain("✗");
    expect(result.status).toBe(0);
    expect(existsSync(fx.clone)).toBe(false);
    expect(existsSync(join(fx.bin, "yian"))).toBe(false);
    for (const rc of RC_FILES) expect([rc, pathLines(join(fx.home, rc))]).toEqual([rc, 0]);
    expect(readFileSync(join(fx.home, ".bashrc"), "utf8")).toBe("alias ll='ls -l'\n");
    expect(result.stdout).toContain(`Removed ${fx.clone}`);
    for (const file of kept) expect([file, readFileSync(join(fx.home, file), "utf8")]).toEqual([file, "mine\n"]);
    expect(result.stdout).toContain("Left Bun");
  });

  test("refuses to remove the home folder, even a clean clone install.sh marked", () => {
    const fx = fixture();
    const home = join(fx.base, "home2");
    git(fx.base, "clone", "--quiet", fx.remote, home);
    writeFileSync(join(home, ".git", "yian-install"), "folder=created\n");
    const result = pipe("uninstall.sh", fx, ["--dir", home], { HOME: home });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(`Refusing to remove ${home}`);
    expect(existsSync(join(home, "package.json"))).toBe(true);
  });

  test("keeps a folder that was there before install.sh cloned into it, empty", () => {
    const fx = fixture();
    const dir = join(fx.base, "mine");
    mkdirSync(dir);
    install(fx, ["--dir", dir]);
    const result = pipe("uninstall.sh", fx, ["--dir", dir]);
    expect(result.status).toBe(0);
    expect(readdirSync(dir)).toEqual([]);
    expect(result.stdout).toContain(`Removed the clone in ${dir}`);
  });

  test("takes the clone's folder from --dir", () => {
    const fx = fixture();
    const dir = join(fx.base, "elsewhere");
    install(fx, ["--dir", dir]);
    expect(pipe("uninstall.sh", fx, ["--dir", dir]).status).toBe(0);
    expect(existsSync(dir)).toBe(false);
  });

  /** The user's git config in the fixture's home, which uninstall.sh reads but must not be steered by. */
  const gitconfig = (fx: Fixture, text: string) => writeFileSync(join(fx.home, ".gitconfig"), text);
  /** Each kind of local work, how to make it, and what the kept-clone message names. */
  const LOCAL_WORK: Record<string, [(fx: Fixture) => void, (fx: Fixture) => string]> = {
    "an uncommitted change": [(fx) => writeFileSync(join(fx.clone, "package.json"), "{}\n"), () => "package.json"],
    "an untracked file": [(fx) => writeFileSync(join(fx.clone, "notes.md"), "mine\n"), () => "notes.md"],
    "an ignored file": [(fx) => writeFileSync(join(fx.clone, ".env"), "KEY=mine\n"), () => ".env"],
    "an untracked file that status.showUntrackedFiles=no hides": [
      (fx) => {
        gitconfig(fx, "[status]\n\tshowUntrackedFiles = no\n");
        writeFileSync(join(fx.clone, "notes.md"), "mine\n");
      },
      () => "notes.md",
    ],
    "an ignored file that status.showUntrackedFiles=no hides": [
      (fx) => {
        gitconfig(fx, "[status]\n\tshowUntrackedFiles = no\n");
        writeFileSync(join(fx.clone, ".env"), "KEY=mine\n");
      },
      () => ".env",
    ],
    "a file that a global excludes file ignores": [
      (fx) => {
        writeFileSync(join(fx.home, ".gitignore_global"), "notes/\n");
        gitconfig(fx, `[core]\n\texcludesFile = ${join(fx.home, ".gitignore_global")}\n`);
        mkdirSync(join(fx.clone, "notes"));
        writeFileSync(join(fx.clone, "notes", "todo.md"), "mine\n");
      },
      () => "notes/",
    ],
    "a stash": [
      (fx) => {
        writeFileSync(join(fx.clone, "package.json"), "{}\n");
        git(fx.clone, "stash", "--quiet");
      },
      () => "stash@{0}",
    ],
    "a commit that isn't pushed": [(fx) => commit(fx.clone, "notes.md", "mine"), () => "commits that are not pushed: main"],
    "a commit on a detached HEAD": [
      (fx) => {
        git(fx.clone, "switch", "--quiet", "--detach");
        commit(fx.clone, "notes.md", "mine");
      },
      () => "commits that are not pushed: HEAD",
    ],
    "a commit that only a tag holds": [
      (fx) => {
        commit(fx.clone, "notes.md", "mine");
        git(fx.clone, "tag", "mine");
        git(fx.clone, "reset", "--quiet", "--hard", "origin/main");
      },
      () => "commits that are not pushed: mine",
    ],
    "a linked worktree": [
      (fx) => git(fx.clone, "worktree", "add", "--quiet", join(fx.base, "feature"), "-b", "feature"),
      (fx) => realpathSync(join(fx.base, "feature")),
    ],
  };
  for (const [work, [make, named]] of Object.entries(LOCAL_WORK)) {
    test(`keeps a clone with ${work}, and names it`, () => {
      const fx = fixture();
      install(fx);
      make(fx);
      const result = pipe("uninstall.sh", fx);
      expect(result.status).toBe(0);
      expect(existsSync(join(fx.clone, ".git"))).toBe(true);
      expect(result.stdout).toContain(`Kept ${fx.clone}: it has`);
      expect(result.stdout).toContain(named(fx));
      expect(existsSync(join(fx.bin, "yian"))).toBe(false);
    });
  }

  test("names five of the files that keep a clone, and counts the rest", () => {
    const fx = fixture();
    install(fx);
    for (const name of ["a", "b", "c", "d", "e", "f", "g"]) writeFileSync(join(fx.clone, `${name}.md`), "mine\n");
    const result = pipe("uninstall.sh", fx);
    expect(result.stdout).toContain(`Kept ${fx.clone}: it has files that are not committed: a.md, b.md, c.md, d.md, e.md, and 2 more.`);
  });

  test("removes a clone whose only untracked files are .DS_Store", () => {
    const fx = fixture();
    install(fx);
    writeFileSync(join(fx.clone, ".DS_Store"), "");
    writeFileSync(join(fx.clone, "src", ".DS_Store"), "");
    const result = pipe("uninstall.sh", fx);
    expect(result.status).toBe(0);
    expect(existsSync(fx.clone)).toBe(false);
  });

  test("never deletes an rc file, even one left empty", () => {
    const fx = fixture();
    writeFileSync(join(fx.home, ".zshrc"), "");
    install(fx);
    install(fx, [], { SHELL: "/bin/zsh" });
    expect(pipe("uninstall.sh", fx).status).toBe(0);
    for (const rc of [".bashrc", ".profile", ".zshrc", ".zprofile"]) {
      expect([rc, readFileSync(join(fx.home, rc), "utf8")]).toEqual([rc, ""]);
    }
  });

  test("keeps a clone install.sh didn't make", () => {
    const fx = fixture();
    git(fx.base, "clone", "--quiet", fx.remote, fx.clone);
    const result = pipe("uninstall.sh", fx);
    expect(result.status).toBe(0);
    expect(existsSync(join(fx.clone, ".git"))).toBe(true);
    expect(result.stdout).toContain(`Kept ${fx.clone}: install.sh didn't clone it`);
  });
});

describe.if(process.versions.bun !== undefined && process.env["YIAN_TEST_INSTALL"] === "1")("install.sh with real Bun", () => {
  test(
    "installs a clone of this checkout offline, runs it, finds it clean on a re-run, and uninstall removes it",
    () => {
      const cache = process.env["BUN_INSTALL_CACHE_DIR"];
      if (!cache) throw new Error("Set BUN_INSTALL_CACHE_DIR to a Bun cache that holds this checkout's packages.");
      const { base, home, bin } = setup();
      // A repository whose main is this checkout's HEAD, whatever branch the checkout is on.
      const src = join(base, "src");
      // Shallow, since a full clone of the history copies far more than the test needs.
      git(base, "clone", "--quiet", "--depth=1", `file://${ROOT}`, src);
      git(src, "switch", "--quiet", "-C", "main");
      const env = {
        HOME: home,
        SHELL: "/bin/bash",
        PATH: `${dirname(process.execPath)}:/usr/bin:/bin`,
        BUN_INSTALL_CACHE_DIR: cache,
        ...NO_NETWORK,
      };
      const script = (name: string, args: string[] = []) =>
        spawnSync("sh", ["-s", "--", ...args], {
          input: readFileSync(join(ROOT, name)),
          cwd: home,
          env,
          encoding: "utf8",
          timeout: 240_000,
        });
      for (let i = 0; i < 2; i++) {
        const result = script("install.sh", ["--repo", src]);
        expect(result.stderr).not.toContain("✗");
        expect(result.status).toBe(0);
      }
      const version = spawnSync(join(bin, "yian"), ["--version"], { env: { HOME: home, PATH: "/usr/bin:/bin" }, encoding: "utf8" });
      expect(version.status).toBe(0);
      expect(version.stdout.trim()).toBe(VERSION);
      const removed = script("uninstall.sh");
      expect(removed.status).toBe(0);
      expect(existsSync(join(home, ".local", "share", "yian"))).toBe(false);
    },
    600_000,
  );
});
