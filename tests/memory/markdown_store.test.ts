/**
 * Versioned compare-and-set markdown stores. The golden replay writes exactly the files in
 * `tests/conformance/fixtures/markdown_store.json` for the same operations, byte for byte, since
 * they are the format of documents and write logs on users' disks; with `YIAN_TEST_NODE` set to a
 * Node binary, it and the read of the fixture copy also run under Node.
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { MemoryOperationError } from "../../src/memory/contracts.ts";
import {
  FileMarkdownStore,
  InMemoryMarkdownStore,
  type MarkdownStore,
  MarkdownVersionMismatch,
  markdownVersion,
} from "../../src/memory/storage/markdown.ts";
import { FileLockError } from "../../src/platform.ts";
import { type ExtendResult, readAndExtend, replay } from "./markdown_both_runtimes.ts";

/** The package root. */
const ROOT = path.resolve(import.meta.dirname, "../..");
const BUILT = path.join(ROOT, "dist/src/memory/storage/markdown.js");
const BUILT_SCENARIOS = path.join(ROOT, "dist/tests/memory/markdown_both_runtimes.js");
const FIXTURE: Record<string, string> = JSON.parse(
  fs.readFileSync(path.join(ROOT, "tests/conformance/fixtures/markdown_store.json"), "utf8"),
);

/** Return what the promise or call `fn` rejects or throws with; fail when it succeeds. */
async function failure(fn: () => unknown): Promise<unknown> {
  try {
    await fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a failure");
}

/** Assert `error` has a Node system error's shape (`isSystemError` in `sqlite.ts`). */
function expectSystemError(error: unknown): asserts error is NodeJS.ErrnoException {
  expect(error).toBeInstanceOf(Error);
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  expect(typeof code).toBe("string");
  expect(typeof syscall === "string" || typeof errno === "number").toBe(true);
}

/** Return the paths of every file under `root`, relative to it, sorted. */
function walk(root: string, prefix = ""): string[] {
  const files: string[] = [];
  for (const entry of fs.readdirSync(path.join(root, prefix), { withFileTypes: true })) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...walk(root, relative));
    else files.push(relative);
  }
  return files.sort();
}

/** Return the kept-copy path of `version` for the document `ada.md` in `root`. */
function keptPath(root: string, version: string): string {
  return path.join(root, ".ada.md.versions", `${version.slice("sha256:".length)}.md`);
}

/** Write the fixture's files into `root`. */
function copyFixture(root: string): void {
  for (const [relative, text] of Object.entries(FIXTURE)) {
    fs.mkdirSync(path.dirname(path.join(root, relative)), { recursive: true });
    fs.writeFileSync(path.join(root, relative), text);
  }
}

/** The kept versions in the fixture, sorted. */
const FIXTURE_VERSIONS = Object.keys(FIXTURE)
  .filter((relative) => relative.startsWith(".ada.md.versions/"))
  .map((relative) => `sha256:${relative.slice(".ada.md.versions/".length, -".md".length)}`)
  .sort();

/** Assert `root` holds exactly the fixture's files, each byte-equal. */
function expectFixtureFiles(root: string): void {
  expect(walk(root)).toEqual(Object.keys(FIXTURE).sort());
  for (const [relative, text] of Object.entries(FIXTURE)) {
    expect(fs.readFileSync(path.join(root, relative))).toEqual(Buffer.from(text, "utf8"));
  }
}

/** Assert `result` is what `readAndExtend` returns on the fixture copy in `root`, and check the files it changed. */
function expectExtendedFixture(root: string, result: ExtendResult): void {
  const current = markdownVersion(FIXTURE["ada.md"]!);
  expect(result.current).toEqual([FIXTURE["ada.md"]!, current]);
  expect(result.kept).toEqual(
    FIXTURE_VERSIONS.map((version) => [FIXTURE[`.ada.md.versions/${version.slice(7)}.md`]!, version]),
  );
  expect(result.written).toBe(markdownVersion("# Ada\nFrom TS.\n"));
  expect(fs.readFileSync(path.join(root, "ada.md"), "utf8")).toBe("# Ada\nFrom TS.\n");
  expect(fs.readFileSync(keptPath(root, result.written), "utf8")).toBe("# Ada\nFrom TS.\n");
  expect(fs.readFileSync(path.join(root, ".ada.md.writes.jsonl"), "utf8")).toBe(
    `${FIXTURE[".ada.md.writes.jsonl"]}{"base_version": "${current}", "provenance": {"run_id": "ts", "score": 0.5}, "version": "${result.written}"}\n`,
  );
}

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-markdown-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a missing file reads as an empty document", async () => {
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  expect(await store.read()).toEqual(["", markdownVersion("")]);
  expect(fs.existsSync(path.join(dir, "ada.md"))).toBe(false);
});

test("an unknown or malformed version is a RangeError", async () => {
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  for (const version of [`sha256:${"0".repeat(64)}`, "sha256:../../etc", "md5:abc"]) {
    const error = await failure(() => store.read(version));
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(`unknown document version '${version}'`);
  }
});

test("a missing parent directory fails on first use", async () => {
  const store = new FileMarkdownStore(path.join(dir, "missing", "ada.md"));
  expect(fs.existsSync(path.join(dir, "missing"))).toBe(false);
  for (const operation of [() => store.read(), () => store.write("x", { baseVersion: markdownVersion("") })]) {
    const error = await failure(operation);
    expectSystemError(error);
    expect([error.code, error.syscall]).toEqual(["ENOENT", "stat"]);
    expect(error.message).toBe(`document directory does not exist: ${fs.realpathSync(dir)}/missing`);
  }
  expect(fs.existsSync(path.join(dir, "missing"))).toBe(false);
});

test.each([
  ["empty", ""],
  [":memory:", ":memory:"],
  ["a file: URI", "file:ada.md"],
  ["a FILE: URI", "FILE:ada.md"],
  ["a NUL", "ada\0.md"],
  ["a Buffer", Buffer.from("ada.md")],
  ["a lone surrogate", "\udcff.md"],
])("a non-file path (%s) is rejected at construction", (_label, candidate) => {
  const previous = process.cwd();
  process.chdir(dir);
  try {
    let caught: unknown;
    try {
      new FileMarkdownStore(candidate);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(MemoryOperationError);
    expect((caught as MemoryOperationError).code).toBe("storage.path");
  } finally {
    process.chdir(previous);
  }
  expect(fs.readdirSync(dir)).toEqual([]);
});

test("a directory path fails on first use", async () => {
  fs.mkdirSync(path.join(dir, "ada.md"));
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  expectSystemError(await failure(() => store.read()));
});

test("createParent makes the directory on first use", async () => {
  const file = path.join(dir, "a", "b", "ada.md");
  const store = new FileMarkdownStore(file, { createParent: true });
  expect(fs.existsSync(path.join(dir, "a"))).toBe(false);
  expect(await store.read()).toEqual(["", markdownVersion("")]);
  expect(fs.statSync(path.dirname(file)).isDirectory()).toBe(true);
  expect(() => new FileMarkdownStore(file, { createParent: 1 as unknown as boolean })).toThrow(
    new TypeError("create_parent must be a bool"),
  );
});

test("a relative path is bound to the construction directory", async () => {
  fs.mkdirSync(path.join(dir, "a"));
  fs.mkdirSync(path.join(dir, "b"));
  const previous = process.cwd();
  try {
    process.chdir(path.join(dir, "a"));
    const store = new FileMarkdownStore("ada.md");
    expect(fs.readdirSync(path.join(dir, "a"))).toEqual([]);
    process.chdir(path.join(dir, "b"));
    await store.write("hello", { baseVersion: markdownVersion("") });
  } finally {
    process.chdir(previous);
  }
  expect(fs.readFileSync(path.join(dir, "a", "ada.md"), "utf8")).toBe("hello");
  expect(fs.existsSync(path.join(dir, "b", "ada.md"))).toBe(false);
});

test("a symlinked path writes through to its target", async () => {
  fs.mkdirSync(path.join(dir, "repo"));
  fs.mkdirSync(path.join(dir, "work"));
  const target = path.join(dir, "repo", "ada.md");
  fs.writeFileSync(target, "# Ada\n");
  const link = path.join(dir, "work", "ada.md");
  fs.symlinkSync(target, link);
  const store = new FileMarkdownStore(link);
  const first = await store.write("# Ada\nLikes tea.\n", { baseVersion: (await store.read())[1] });
  expect(fs.lstatSync(link).isSymbolicLink()).toBe(true);
  expect(fs.readFileSync(target, "utf8")).toBe("# Ada\nLikes tea.\n");
  expect(fs.readdirSync(path.join(dir, "work"))).toEqual(["ada.md"]);
  expect(fs.statSync(path.join(dir, "repo", ".ada.md.writes.jsonl")).isFile()).toBe(true);
  fs.writeFileSync(target, "# Ada\nEdited by hand.\n");
  const error = await failure(() => new FileMarkdownStore(target).write("lost update\n", { baseVersion: first }));
  expect(error).toBeInstanceOf(MarkdownVersionMismatch);
  expect((await store.read())[0]).toBe("# Ada\nEdited by hand.\n");
});

test("a symlinked parent follows the link before parent steps", async () => {
  fs.mkdirSync(path.join(dir, "real", "sub"), { recursive: true });
  fs.symlinkSync(path.join(dir, "real", "sub"), path.join(dir, "link"));
  const store = new FileMarkdownStore(`${dir}/link/../ada.md`);
  await store.write("hello", { baseVersion: markdownVersion("") });
  expect(fs.readFileSync(path.join(dir, "real", "ada.md"), "utf8")).toBe("hello");
  expect(fs.statSync(path.join(dir, "real", ".ada.md.lock")).isFile()).toBe(true);
  expect(fs.existsSync(path.join(dir, "ada.md"))).toBe(false);
});

test("a dangling symlink creates its target", async () => {
  const link = path.join(dir, "ada.md");
  fs.symlinkSync(path.join(dir, "notes.md"), link);
  const store = new FileMarkdownStore(link);
  expect(await store.read()).toEqual(["", markdownVersion("")]);
  await store.write("hello", { baseVersion: markdownVersion("") });
  expect(fs.lstatSync(link).isSymbolicLink()).toBe(true);
  expect(fs.readFileSync(path.join(dir, "notes.md"), "utf8")).toBe("hello");
});

test("a write is compare-and-set and keeps every version", async () => {
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  const [, empty] = await store.read();
  const first = await store.write("# Ada\n", { baseVersion: empty });
  expect(first).toBe(markdownVersion("# Ada\n"));
  expect(fs.readFileSync(path.join(dir, "ada.md"), "utf8")).toBe("# Ada\n");
  const second = await store.write("# Ada\nPrefers email.\n", { baseVersion: first });
  const error = await failure(() => store.write("lost update\n", { baseVersion: first }));
  expect(error).toBeInstanceOf(MarkdownVersionMismatch);
  const mismatch = error as MarkdownVersionMismatch;
  expect([mismatch.expected, mismatch.current]).toEqual([first, second]);
  expect(mismatch.message).toBe(`base version '${first}' is not current ('${second}')`);
  expect(await store.read(first)).toEqual(["# Ada\n", first]);
  expect(await store.read(empty)).toEqual(["", empty]);
  expect(await store.read()).toEqual(["# Ada\nPrefers email.\n", second]);
});

test("a direct file edit is a new version and breaks stale writes", async () => {
  const file = path.join(dir, "ada.md");
  const store = new FileMarkdownStore(file);
  const base = await store.write("# Ada\n", { baseVersion: (await store.read())[1] });
  fs.writeFileSync(file, "# Ada\nEdited by hand.\n");
  expect(await failure(() => store.write("# Ada\nFrom a model.\n", { baseVersion: base }))).toBeInstanceOf(
    MarkdownVersionMismatch,
  );
  const [text, version] = await store.read();
  expect(text).toBe("# Ada\nEdited by hand.\n");
  expect(version).toBe(markdownVersion(text));
  fs.writeFileSync(file, "# Ada\nEdited again.\n");
  expect(await store.read(version)).toEqual(["# Ada\nEdited by hand.\n", version]);
});

test.each(["file", "memory"])("carriage returns round-trip exactly (%s)", async (kind) => {
  const store: MarkdownStore =
    kind === "file" ? new FileMarkdownStore(path.join(dir, "ada.md")) : new InMemoryMarkdownStore();
  let base = (await store.read())[1];
  for (const text of ["# Ada\r\nPrefers email.\r\n", "# Ada\rOld Mac line.\n"]) {
    const version = await store.write(text, { baseVersion: base });
    expect(version).toBe(markdownVersion(text));
    expect(await store.read()).toEqual([text, version]);
    base = version;
  }
  const first = markdownVersion("# Ada\r\nPrefers email.\r\n");
  expect(await store.read(first)).toEqual(["# Ada\r\nPrefers email.\r\n", first]);
});

test("a hand-edited CRLF file reads back byte for byte", async () => {
  const file = path.join(dir, "ada.md");
  fs.writeFileSync(file, Buffer.from("# Ada\r\nEdited on Windows.\r\n"));
  const store = new FileMarkdownStore(file);
  const [text, version] = await store.read();
  expect(text).toBe("# Ada\r\nEdited on Windows.\r\n");
  expect(version).toBe(markdownVersion(text));
  fs.writeFileSync(file, Buffer.from("# Ada\n"));
  expect(await store.read(version)).toEqual([text, version]);
});

test("every write appends its provenance and nothing is replayed", async () => {
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  const base = (await store.read())[1];
  const provenance = { commit_id: "c1", run_id: "r1" };
  const first = await store.write("# Ada\n", { baseVersion: base, provenance });
  expect(await failure(() => store.write("# Ada\n", { baseVersion: base, provenance }))).toBeInstanceOf(
    MarkdownVersionMismatch,
  );
  await store.write("# Ada\nMore.\n", { baseVersion: first, provenance: { commit_id: "c2" } });
  const log = fs
    .readFileSync(path.join(dir, ".ada.md.writes.jsonl"), "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  expect(log.map((entry) => entry.provenance.commit_id)).toEqual(["c1", "c2"]);
  expect([log[0].base_version, log[0].version]).toEqual([base, first]);
});

test.each([
  ["a Date", { at: new Date(2026, 9, 13) }, TypeError],
  ["NaN", { score: Number.NaN }, RangeError],
  ["Infinity", { score: Infinity }, RangeError],
  ["a Set", { tags: new Set(["a"]) }, TypeError],
  ["a symbol key", { [Symbol("key")]: "x" }, TypeError],
  ["an array", [], TypeError],
])("non-JSON provenance (%s) is rejected before anything is written", async (_label, provenance, kind) => {
  const file = path.join(dir, "ada.md");
  const store = new FileMarkdownStore(file);
  const base = await store.write("# Ada\n", { baseVersion: (await store.read())[1], provenance: { commit_id: "c1" } });
  const log = path.join(dir, ".ada.md.writes.jsonl");
  const before = fs.readFileSync(log, "utf8");
  const error = await failure(() =>
    store.write("# Ada\nEdited.\n", { baseVersion: base, provenance: provenance as Record<string, never> }),
  );
  expect(error).toBeInstanceOf(kind);
  expect((error as Error).message).toContain("provenance");
  expect(fs.readFileSync(file, "utf8")).toBe("# Ada\n");
  expect(fs.readFileSync(log, "utf8")).toBe(before);
  expect(fs.existsSync(keptPath(dir, markdownVersion("# Ada\nEdited.\n")))).toBe(false);
});

test("concurrent writers from one base have exactly one winner", async () => {
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  const base = (await store.read())[1];
  const results = await Promise.all(
    Array.from({ length: 8 }, async (_, index) => {
      try {
        return await store.write(`writer ${index}\n`, { baseVersion: base });
      } catch (error) {
        if (error instanceof MarkdownVersionMismatch) return null;
        throw error;
      }
    }),
  );
  const winners = results.filter((version) => version !== null);
  expect(winners.length).toBe(1);
  expect((await store.read())[1]).toBe(winners[0]!);
});

test("the in-memory store is a compare-and-set markdown store", () => {
  const store = new InMemoryMarkdownStore();
  const [text, empty] = store.read();
  expect([text, empty]).toEqual(["", markdownVersion("")]);
  const first = store.write("# Ada\n", { baseVersion: empty, provenance: { run: "r1" } });
  expect(store.read()).toEqual(["# Ada\n", first]);
  let caught: unknown;
  try {
    store.write("lost\n", { baseVersion: empty });
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(MarkdownVersionMismatch);
  expect([(caught as MarkdownVersionMismatch).expected, (caught as MarkdownVersionMismatch).current]).toEqual([
    empty,
    first,
  ]);
  expect(store.read(empty)).toEqual(["", empty]);
  expect(() => store.read(`sha256:${"0".repeat(64)}`)).toThrow(RangeError);
});

test("the in-memory store accepts initial text and checks types", () => {
  const store = new InMemoryMarkdownStore("seed\n");
  const current = markdownVersion("seed\n");
  expect(store.read()).toEqual(["seed\n", current]);
  const message = "text and base_version must be strings";
  expect(() => store.write(Buffer.from("x") as unknown as string, { baseVersion: current })).toThrow(
    new TypeError(message),
  );
  expect(() => store.write("x", { baseVersion: null as unknown as string })).toThrow(new TypeError(message));
  expect(() => new InMemoryMarkdownStore(null as unknown as string)).toThrow(new TypeError("text must be a string"));
  expect(store.read()).toEqual(["seed\n", current]);
});

test("the same operations write exactly the fixture's files", async () => {
  await replay(new FileMarkdownStore(path.join(dir, "ada.md")));
  expectFixtureFiles(dir);
});

test("a copy of the fixture's files reads every version and takes a compare-and-set write", async () => {
  copyFixture(dir);
  expectExtendedFixture(dir, await readAndExtend(new FileMarkdownStore(path.join(dir, "ada.md")), FIXTURE_VERSIONS));
});

test("a lone surrogate in written text is a RangeError and writes nothing", async () => {
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  const error = await failure(() => store.write("ab\ud800", { baseVersion: markdownVersion("") }));
  const message = "text holds a lone surrogate, which has no UTF-8 encoding";
  expect(error).toEqual(new RangeError(message));
  expect(() => markdownVersion("😀\udc00")).toThrow(new RangeError(message));
  expect(() => new InMemoryMarkdownStore("\ud800")).toThrow(RangeError);
  expect(fs.readdirSync(dir)).toEqual([]);
});

test("invalid UTF-8 on disk is a RangeError", async () => {
  const file = path.join(dir, "ada.md");
  fs.writeFileSync(file, Buffer.from([0x23, 0xff, 0x0a]));
  const store = new FileMarkdownStore(file);
  expect(await failure(() => store.read())).toEqual(new RangeError("document is not valid UTF-8"));
  expect(await failure(() => store.write("x", { baseVersion: markdownVersion("") }))).toEqual(
    new RangeError("document is not valid UTF-8"),
  );
});

test("a BOM survives a write and a hand edit", async () => {
  const file = path.join(dir, "ada.md");
  const store = new FileMarkdownStore(file);
  const version = await store.write("\ufeff# Ada\n", { baseVersion: markdownVersion("") });
  expect(fs.readFileSync(file)).toEqual(Buffer.from([0xef, 0xbb, 0xbf, ...Buffer.from("# Ada\n")]));
  expect(await store.read()).toEqual(["\ufeff# Ada\n", version]);
  fs.writeFileSync(file, Buffer.from([0xef, 0xbb, 0xbf, 0x78]));
  expect(await store.read()).toEqual(["\ufeffx", markdownVersion("\ufeffx")]);
});

test("the write log sorts keys, escapes non-ASCII and keeps a float's fraction or exponent", async () => {
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  const base = markdownVersion("");
  const version = await store.write("x", {
    baseVersion: base,
    provenance: { neg: -0, small: 1e-7, big: 1e16, "😀": 1n, "\uffff": 2n, Z: 0.1 },
  });
  expect(fs.readFileSync(path.join(dir, ".ada.md.writes.jsonl"), "utf8")).toBe(
    `{"base_version": "${base}", "provenance": {"Z": 0.1, "big": 1e+16, "neg": -0.0, "small": 1e-07, "\\uffff": 2, "\\ud83d\\ude00": 1}, "version": "${version}"}\n`,
  );
});

test("a written document is private", async () => {
  const file = path.join(dir, "ada.md");
  fs.writeFileSync(file, "");
  fs.chmodSync(file, 0o644);
  const store = new FileMarkdownStore(file);
  const version = await store.write("# Ada\n", { baseVersion: markdownVersion("") });
  expect(fs.statSync(file).mode & 0o777).toBe(0o600);
  expect(fs.statSync(keptPath(dir, version)).mode & 0o777).toBe(0o600);
});

test.each(["kept copy", "document"])(
  "a failed rename of the %s rejects the write, leaves no temporary file and releases the lock",
  async (failing) => {
    const file = path.join(dir, "ada.md");
    const store = new FileMarkdownStore(file);
    const base = await store.write("# Ada\n", { baseVersion: markdownVersion("") });
    const renameError = Object.assign(new Error("rename failed"), { code: "EIO", syscall: "rename" });
    const rename = fs.renameSync;
    const spy = spyOn(fs, "renameSync").mockImplementation((from, to) => {
      if (failing === "kept copy" || to === store.path) throw renameError;
      rename(from, to);
    });
    try {
      expect(await failure(() => store.write("# Ada\nMore.\n", { baseVersion: base }))).toBe(renameError);
      expect(spy).toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
    // `.ada.md.<hex>` beside the document, `.<hex>.md.<hex>` among the kept copies.
    expect(walk(dir).filter((relative) => /(^|\/)\.[^/]+\.md\.[0-9a-f]{16}$/.test(relative))).toEqual([]);
    expect(await store.read()).toEqual(["# Ada\n", base]);
  },
);

test("a lock path that isn't a file fails with a system error and writes nothing", async () => {
  fs.mkdirSync(path.join(dir, ".ada.md.lock"));
  const store = new FileMarkdownStore(path.join(dir, "ada.md"));
  for (const operation of [() => store.read(), () => store.write("x", { baseVersion: markdownVersion("") })]) {
    const error = await failure(operation);
    expectSystemError(error);
    expect([error.code, error.syscall]).toEqual(["EIO", "lock"]);
    expect(error.cause).toBeInstanceOf(FileLockError);
  }
  expect(fs.readdirSync(dir)).toEqual([".ada.md.lock"]);
});

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/platform.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  for (const built of [BUILT, BUILT_SCENARIOS]) {
    if (!fs.existsSync(built)) throw new Error(`${built} is missing; run bun run build first`);
  }
  return node;
})();

describe.skipIf(!NODE)("the built markdown store on Node", () => {
  /**
   * Run `body` under Node with `FileMarkdownStore`, `replay` and `readAndExtend` imported from the
   * build and `file` bound to `<dir>/ada.md`; return its stdout.
   */
  function runNode(body: string): string {
    const script = `import { FileMarkdownStore } from ${JSON.stringify(pathToFileURL(BUILT).href)};
import { readAndExtend, replay } from ${JSON.stringify(pathToFileURL(BUILT_SCENARIOS).href)};
const file = ${JSON.stringify(path.join(dir, "ada.md"))};
${body}`;
    const result = spawnSync(NODE!, ["--input-type=module", "-e", script], { encoding: "utf8", timeout: 20_000 });
    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    return result.stdout;
  }

  test(
    "the same operations write exactly the fixture's files",
    () => {
      runNode("await replay(new FileMarkdownStore(file));");
      expectFixtureFiles(dir);
    },
    30_000,
  );

  test(
    "a copy of the fixture's files reads every version and takes a compare-and-set write",
    () => {
      copyFixture(dir);
      const stdout = runNode(`const result = await readAndExtend(
  new FileMarkdownStore(file),
  ${JSON.stringify(FIXTURE_VERSIONS)},
);
console.log(JSON.stringify(result));`);
      expectExtendedFixture(dir, JSON.parse(stdout));
    },
    30_000,
  );
});
