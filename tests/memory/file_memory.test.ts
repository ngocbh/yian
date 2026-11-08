/** FileMemory: its identity, members, version, recall, read tools and writes. */

import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  type Applied,
  type Author,
  BUILTIN_MEMORY_TOOLS,
  checkContract,
  MemoryBusyError,
  memoryErrorKind,
  type MemorySession,
} from "../../src/memory/base.ts";
import { FileMemory, type FileMemoryOptions, memoryVersion, readMembers } from "../../src/memory/file.ts";
import { lockFile, virtualScheduler } from "../../src/platform.ts";
import { isPermanentToolError, isReportedToolError } from "../../src/tools/errors.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

let dir: string;
let root: string;
const signal = new AbortController().signal;
const o = { create: true, itemType: null, signal };

beforeEach(() => {
  dir = fs.mkdtempSync(join(tmpdir(), "yian-fm-"));
  root = join(dir, "m");
});

afterEach(() => {
  mock.restore();
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Return the sha256 hex digest of `data`. */
function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Write `text` to `rel` under `base`, making its folders. */
function put(base: string, rel: string, text: string | Uint8Array): void {
  fs.mkdirSync(join(base, rel, ".."), { recursive: true });
  fs.writeFileSync(join(base, rel), text);
}

test("a new memory takes its id, or at-<16 hex> of the real path, and stores it", async () => {
  const m = new FileMemory({ root: join(dir, "a") });
  const s = await m.open(o);
  expect(s.created).toBe(true);
  expect(s.info.identity).toMatch(/^at-[0-9a-f]{16}$/);
  expect(s.info.identity).toBe(`at-${sha256(fs.realpathSync(join(dir, "a"))).slice(0, 16)}`);
  expect(fs.readFileSync(join(dir, "a/.yian/identity"), "utf8")).toBe(s.info.identity);
  expect(fs.statSync(join(dir, "a")).mode & 0o777).toBe(0o700);
  expect(fs.statSync(join(dir, "a/.yian")).mode & 0o777).toBe(0o700);
  expect(fs.statSync(join(dir, "a/.yian/identity")).mode & 0o777).toBe(0o600);
  await s.close();

  const named = await new FileMemory({ root: join(dir, "b"), id: "people/ada" }).open(o);
  expect(named.info.identity).toBe("people/ada");
  await named.close();
});

test("a symlinked root is resolved once, at construction, and its identity hashes the real path", async () => {
  fs.mkdirSync(join(dir, "real"));
  fs.symlinkSync(join(dir, "real"), join(dir, "link"));
  const s = await new FileMemory({ root: join(dir, "link") }).open(o);
  expect(s.info.identity).toBe(`at-${sha256(fs.realpathSync(join(dir, "real"))).slice(0, 16)}`);
  expect(fs.existsSync(join(dir, "real/.yian/identity"))).toBe(true);
  await s.close();
});

test("a stored identity wins; a different constructor id is refused", async () => {
  await (await new FileMemory({ root, id: "people/ada" }).open(o)).close();
  await expect(new FileMemory({ root, id: "people/bob" }).open(o)).rejects.toThrow(
    "this memory is `people/ada`, not `people/bob`",
  );
  const again = await new FileMemory({ root }).open(o);
  expect(again.created).toBe(false);
  expect(again.info.identity).toBe("people/ada");
  await again.close();
  const resumed = await new FileMemory({ root, id: "people/ada" }).open({ ...o, create: false });
  expect(resumed.created).toBe(false);
  await resumed.close();
});

test("open({create: false}) refuses a missing memory and creates nothing", async () => {
  await expect(new FileMemory({ root }).open({ ...o, create: false })).rejects.toThrow(
    `\`${root}\` isn't a memory yet: there is no folder`,
  );
  expect(fs.existsSync(root)).toBe(false);
});

test("open({create: true}) adopts an existing folder with files and no .yian/, keeping them", async () => {
  put(root, "profile.md", "Ada.");
  fs.chmodSync(root, 0o755);
  const s = await new FileMemory({ root, id: "people/ada", index: "profile.md" }).open(o);
  expect(s.created).toBe(true);
  expect(fs.readFileSync(join(root, "profile.md"), "utf8")).toBe("Ada.");
  expect(fs.readFileSync(join(root, ".yian/identity"), "utf8")).toBe("people/ada");
  // An adopted folder keeps its own mode; only what the memory makes is private.
  expect(fs.statSync(root).mode & 0o777).toBe(0o755);
  await s.close();
});

test("open({create: false}) refuses an existing folder with no .yian/: it isn't a memory yet", async () => {
  put(root, "profile.md", "Ada.");
  await expect(new FileMemory({ root }).open({ ...o, create: false })).rejects.toThrow(
    `\`${root}\` isn't a memory yet`,
  );
  expect(fs.existsSync(join(root, ".yian"))).toBe(false);
});

test("a root that is a file, or a .yian that isn't a folder, is refused", async () => {
  put(dir, "file", "x");
  await expect(new FileMemory({ root: join(dir, "file") }).open(o)).rejects.toThrow("isn't a folder");
  put(root, ".yian", "x");
  await expect(new FileMemory({ root }).open(o)).rejects.toThrow("`.yian` isn't a folder");
});

test("an item type is refused: a FileMemory holds files, not typed items", async () => {
  await expect(new FileMemory({ root }).open({ ...o, itemType: { type: "object" } })).rejects.toThrow(
    "a FileMemory holds files, not typed items",
  );
});

test("the session's info holds the type, identity, settings with no location, and the contract's tools", async () => {
  const s = await new FileMemory({ root, id: "t/x", maxFiles: 7, index: "INDEX.md" }).open(o);
  expect(s.info.type).toBe("FileMemory");
  expect(s.info.tools.map((tool) => tool.name)).toEqual([...BUILTIN_MEMORY_TOOLS.FileMemory]);
  expect(s.info.tools).toEqual(FileMemory.contract.tools);
  expect(s.info.settings).toEqual({
    index: "INDEX.md",
    max_file_chars: 100_000n,
    max_total_chars: 1_000_000n,
    max_files: 7n,
    max_depth: 4n,
    max_recall_chars: 8_000n,
    max_read_chars: 20_000n,
    max_results: 100n,
    max_undo: 100n,
    max_change_ids: 10_000n,
    max_history_chars: 1_000_000n,
  });
  expect(JSON.stringify(s.info.settings, (_, v) => (typeof v === "bigint" ? `${v}` : v))).not.toContain(dir);
  expect(() => checkContract(new FileMemory({ root }))).not.toThrow();
  for (const tool of FileMemory.contract.tools) {
    expect(tool.description).not.toMatch(/\d/);
  }
  await s.close();
});

test("members: hidden, badly named, binary and oversized files are skipped; a symlink is refused", () => {
  put(root, "MEMORY.md", "index");
  put(root, "notes/a.md", "a");
  put(root, ".hidden.md", "h");
  put(root, ".git/config", "g");
  put(root, "-dash.md", "d");
  put(root, "sp ace.md", "s");
  put(root, "bin.dat", new Uint8Array([0xff, 0xfe, 0x00]));
  put(root, "big.md", "x".repeat(11));
  put(dir, "outside.md", "o");
  fs.symlinkSync(join(dir, "outside.md"), join(root, "link.md"));
  fs.symlinkSync(join(root, "notes"), join(root, "linked"));
  put(root, "a/b/c/d/deep.md", "ok");
  put(root, "a/b/c/d/e/deeper.md", "too deep");
  put(root, ".yian/identity", "t/x");

  const members = readMembers(root, { maxDepth: 4, maxFileChars: 10 });
  expect(members.files.map((file) => file.path)).toEqual(["MEMORY.md", "a/b/c/d/deep.md", "notes/a.md"]);
  // `.yian/` is the memory's own and never counts; everything else that isn't a member does.
  expect(members.skipped).toBe(9);
  expect(members.files[0]).toEqual({ path: "MEMORY.md", text: "index", hash: sha256("index"), chars: 5 });
});

test("a member's size counts characters, not bytes", () => {
  put(root, "a.md", "é😀");
  expect(readMembers(root, { maxDepth: 4, maxFileChars: 2 }).files.map((file) => file.chars)).toEqual([2]);
});

test("the version is sha256 over sorted (path, file hash) pairs of members only", () => {
  put(root, "b.md", "B");
  put(root, "a/z.md", "Z");
  put(root, "a.md", "A");
  put(root, ".git/config", "ignored");
  put(root, "bin.dat", new Uint8Array([0xff]));
  const members = readMembers(root, { maxDepth: 4, maxFileChars: 100 });
  const pairs = [
    ["a.md", sha256("A")],
    ["a/z.md", sha256("Z")],
    ["b.md", sha256("B")],
  ];
  expect(memoryVersion(members.files)).toBe(sha256(JSON.stringify(pairs)));
  fs.writeFileSync(join(root, ".git/config"), "changed");
  expect(memoryVersion(readMembers(root, { maxDepth: 4, maxFileChars: 100 }).files)).toBe(
    sha256(JSON.stringify(pairs)),
  );
  expect(memoryVersion([])).toBe(sha256("[]"));
});

test.each([
  ["../x.md", "a `..` part"],
  ["a/../../x.md", "a `..` part"],
  ["./x.md", "a `.` part"],
  ["a//x.md", "an empty part"],
  [".yian/identity", "a hidden name"],
  [".git/config", "a hidden name"],
  ["-x.md", "a name other than letters, digits, `.`, `_` and `-`, starting with a letter or digit"],
  ["/etc/passwd", "an absolute path"],
  ["a/b/c/d/e/f.md", "more than 4 folders deep"],
  ["", "an empty path"],
])("an index option %s is refused at construction, naming why", (index, why) => {
  expect(() => new FileMemory({ root, index })).toThrow(/index/);
  expect(() => new FileMemory({ root, index })).toThrow(why);
});

test("the depth cap counts the folders above a file", () => {
  expect(() => new FileMemory({ root, index: "a/b/c/d/x.md" })).not.toThrow();
  expect(() => new FileMemory({ root, index: "a/b/x.md", maxDepth: 1 })).toThrow("more than 1 folder deep");
});

test("a depth cap of 0 makes a flat memory: files at the top only", async () => {
  put(root, "a.md", "A");
  put(root, "sub/b.md", "B");
  expect(() => new FileMemory({ root, index: "x/MEMORY.md", maxDepth: 0 })).toThrow("more than 0 folders deep");
  const s = await new FileMemory({ root, id: "t/flat", maxDepth: 0 }).open(o);
  expect((await s.call("list", {}, { signal })).files).toEqual([{ path: "a.md", hash: sha256("A"), chars: 1n }]);
  const write = { id: "w", change: { op: "append", path: "sub/c.md", text: "C" }, author, signal };
  expect(await s.apply(write)).toEqual(refusal("`sub/c.md` isn't part of the memory: more than 0 folders deep"));
  await s.close();
});

test("bad settings are refused at construction", () => {
  expect(() => new FileMemory({ root: "" })).toThrow("root");
  expect(() => new FileMemory({ root, id: "" })).toThrow("id");
  expect(() => new FileMemory({ root, maxFiles: 0 })).toThrow("maxFiles");
  expect(() => new FileMemory({ root, maxDepth: 1.5 })).toThrow("maxDepth");
  expect(() => new FileMemory({ root, lockTimeoutMs: -1 })).toThrow("lockTimeoutMs");
  expect(() => new FileMemory({ root, maxUndo: 0 })).not.toThrow();
});

/** Open a FileMemory on `root` with `options`, creating it. */
async function openAt(options: Partial<FileMemoryOptions> = {}): Promise<MemorySession> {
  return new FileMemory({ root, id: "t/x", ...options }).open(o);
}

/** Return what `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

test("a closed session's methods return rejected promises; none throws at the call", async () => {
  const s = await openAt();
  await s.close();
  const host: Author = { kind: "host", name: "h" };
  const calls: (() => Promise<unknown>)[] = [
    () => s.snapshot(),
    () => s.call("list", {}, { signal }),
    () => s.apply({ id: "x", change: { op: "append", path: "a.md", text: "x" }, author: host, signal }),
    () => s.undo({ id: "u", changeId: "x", author: host, signal }),
    () => s.history({ limit: 1 }),
    () => s.history({ limit: -1 }),
  ];
  for (const call of calls) {
    let promise: Promise<unknown> | undefined;
    expect(() => {
      promise = call();
    }).not.toThrow();
    await expect(promise!).rejects.toThrow();
  }
  await expect(s.snapshot()).rejects.toThrow("this FileMemory session is closed");
});

describe("recall and snapshot", () => {
  test("the recall is the index when it exists, with no hint when nothing else is there", async () => {
    put(root, "profile.md", "Ada.");
    const s = await new FileMemory({ root, id: "people/ada", index: "profile.md" }).open(o);
    expect((await s.snapshot()).recall).toEqual({ text: "Ada.", cut: null, hint: null });
    await s.close();
  });

  test("an index over maxRecallChars is cut, and other files bring a hint to read them", async () => {
    put(root, "MEMORY.md", "abcdefghij");
    const s = await openAt({ maxRecallChars: 4 });
    expect((await s.snapshot()).recall).toEqual({
      text: "abcd",
      cut: "6 more characters of MEMORY.md not shown",
      hint: { tool: "read", text: "Read the rest with {tool}." },
    });
    fs.writeFileSync(join(root, "MEMORY.md"), "ab");
    put(root, "notes/a.md", "a");
    expect((await s.snapshot()).recall).toEqual({
      text: "ab",
      cut: null,
      hint: { tool: "read", text: "Read the rest with {tool}." },
    });
    await s.close();
  });

  test("without an index the recall lists each member's path and first line", async () => {
    put(root, "b.md", "Bee\nmore");
    put(root, "a/x.md", "Ex\r\nmore");
    put(root, "empty.md", "");
    const s = await openAt();
    expect((await s.snapshot()).recall).toEqual({
      text: "a/x.md: Ex\nb.md: Bee\nempty.md",
      cut: null,
      hint: { tool: "read", text: "Read them with {tool}." },
    });
    await s.close();
  });

  test("over maxRecallChars the listing is cut, saying how many files it doesn't show", async () => {
    for (const name of ["a", "b", "c", "d", "e"]) put(root, `${name}.md`, `${name} line`);
    const s = await openAt({ maxRecallChars: 20 });
    // b.md's line doesn't fit, but its path does; nothing more fits after it.
    expect((await s.snapshot()).recall).toEqual({
      text: "a.md: a line\nb.md",
      cut: "3 more files not shown",
      hint: { tool: "read", text: "Read them with {tool}." },
    });
    await s.close();
  });

  test("a first line longer than maxRecallChars shows its file's path alone, and later files still show", async () => {
    put(root, "a.md", `${"x".repeat(100)}\nmore`);
    put(root, "b.md", "short b");
    put(root, "c.md", "short c");
    const s = await openAt({ maxRecallChars: 50 });
    expect((await s.snapshot()).recall).toEqual({
      text: "a.md\nb.md: short b\nc.md: short c",
      cut: null,
      hint: { tool: "read", text: "Read them with {tool}." },
    });
    await s.close();
  });

  test("the listing counts the entries that aren't part of the memory as skipped", async () => {
    put(root, "a.md", "A");
    put(root, ".git/config", "g");
    put(root, "bin.dat", new Uint8Array([0xff]));
    const s = await openAt();
    expect((await s.snapshot()).recall).toEqual({
      text: "a.md: A",
      cut: "2 entries skipped: not part of the memory",
      hint: { tool: "read", text: "Read them with {tool}." },
    });
    await s.close();
  });

  test("an empty memory recalls nothing", async () => {
    const s = await openAt();
    expect(await s.snapshot()).toEqual({ version: memoryVersion([]), recall: { text: "", cut: null, hint: null } });
    await s.close();
  });

  test("snapshot() returns the version and the recall from one read", async () => {
    put(root, "MEMORY.md", "index");
    put(root, "a.md", "A");
    const s = await openAt();
    const snapshot = await s.snapshot();
    expect(snapshot.version).toBe(memoryVersion(readMembers(root, { maxDepth: 4, maxFileChars: 100_000 }).files));
    expect(snapshot.recall.text).toBe("index");
    await s.close();
  });

  test("a writer holding the lock makes snapshot() wait until it lets go", async () => {
    const s = await openAt({ lockTimeoutMs: 10_000 });
    const lock = lockFile(join(root, ".yian/lock"), { private: true });
    let settled = false;
    const snapshot = s.snapshot().finally(() => {
      settled = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(settled).toBe(false);
    lock.release();
    expect((await snapshot).recall.text).toBe("");
    await s.close();
  });

  test("with lockTimeoutMs: 0 a held lock makes snapshot() fail busy", async () => {
    const s = await openAt({ lockTimeoutMs: 0, scheduler: virtualScheduler() });
    const lock = lockFile(join(root, ".yian/lock"), { private: true });
    try {
      const error = await rejection(s.snapshot());
      expect(error).toBeInstanceOf(MemoryBusyError);
      expect(memoryErrorKind(error)).toBe("busy");
    } finally {
      lock.release();
    }
    await s.close();
  });
});

describe("read tools", () => {
  test("list lists paths, hashes and sizes in characters", async () => {
    put(root, "b.md", "Bé");
    put(root, "a/x.md", "X");
    put(root, ".git/config", "g");
    const s = await openAt();
    expect(await s.call("list", {}, { signal })).toEqual({
      files: [
        { path: "a/x.md", hash: sha256("X"), chars: 1n },
        { path: "b.md", hash: sha256("Bé"), chars: 2n },
      ],
    });
    expect(await s.call("list", { dir: "a" }, { signal })).toEqual({
      files: [{ path: "a/x.md", hash: sha256("X"), chars: 1n }],
    });
    await s.close();
  });

  test("list shows at most maxResults files, and says how many more there are", async () => {
    for (const name of ["a", "b", "c", "d"]) put(root, `${name}.md`, name);
    const s = await openAt({ maxResults: 2 });
    const listed = await s.call("list", {}, { signal });
    expect((listed.files as JsonObject[]).map((file) => file.path)).toEqual(["a.md", "b.md"]);
    expect(listed.cut).toBe("2 more; list a folder with dir");
    await s.close();
  });

  test("list keeps its paths within maxReadChars, and always shows one", async () => {
    for (const name of ["a", "b", "c"]) put(root, `${name}.md`, name);
    // A list counts the characters of the paths it shows: 4 for each of these.
    const s = await openAt({ maxReadChars: 9 });
    const listed = await s.call("list", {}, { signal });
    expect((listed.files as JsonObject[]).map((file) => file.path)).toEqual(["a.md", "b.md"]);
    expect(listed.cut).toBe("1 more; list a folder with dir");
    const narrow = await openAt({ maxReadChars: 2 });
    const one = await narrow.call("list", {}, { signal });
    expect((one.files as JsonObject[]).map((file) => file.path)).toEqual(["a.md"]);
    expect(one.cut).toBe("2 more; list a folder with dir");
    await s.close();
    await narrow.close();
  });

  test("list of a folder that isn't there, or isn't part of the memory, is refused", async () => {
    put(root, "a.md", "A");
    const s = await openAt();
    await expect(s.call("list", { dir: "nope" }, { signal })).rejects.toThrow("`nope` doesn't exist");
    await expect(s.call("list", { dir: "a.md" }, { signal })).rejects.toThrow(
      "`a.md` isn't a folder of the memory: a file",
    );
    await expect(s.call("list", { dir: ".git" }, { signal })).rejects.toThrow(
      "`.git` isn't a folder of the memory: a hidden name",
    );
    await s.close();
  });

  test("read returns lines and the file's hash, computed from the bytes it read", async () => {
    put(root, "a.md", "one\ntwo\nthree\n");
    const s = await openAt();
    expect(await s.call("read", { path: "a.md" }, { signal })).toEqual({
      path: "a.md",
      hash: sha256("one\ntwo\nthree\n"),
      text: "one\ntwo\nthree",
    });
    expect(await s.call("read", { path: "a.md", offset: 2n, limit: 1n }, { signal })).toEqual({
      path: "a.md",
      hash: sha256("one\ntwo\nthree\n"),
      text: "two",
      cut: "showing lines 2-2 of 3; read on with offset 3",
    });
    expect(await s.call("read", { path: "a.md", offset: 9 }, { signal })).toEqual({
      path: "a.md",
      hash: sha256("one\ntwo\nthree\n"),
      text: "",
      cut: "the file has 3 lines",
    });
    await s.close();
  });

  test("read shows at most maxReadChars, with a cut note", async () => {
    put(root, "a.md", "aaaa\nbbbb\ncccc");
    const s = await openAt({ maxReadChars: 10 });
    expect(await s.call("read", { path: "a.md" }, { signal })).toEqual({
      path: "a.md",
      hash: sha256("aaaa\nbbbb\ncccc"),
      text: "aaaa\nbbbb",
      cut: "showing lines 1-2 of 3; read on with offset 3",
    });
    put(root, "long.md", "x".repeat(25));
    expect(await s.call("read", { path: "long.md" }, { signal })).toEqual({
      path: "long.md",
      hash: sha256("x".repeat(25)),
      text: "x".repeat(10),
      cut: "line 1 of 1 is cut after 10 characters",
    });
    await s.close();
  });

  test("search returns matching lines, ignoring case, each with its path, line and the file's hash", async () => {
    put(root, "a.md", "Alpha\nbeta\nALPHABET");
    put(root, "b/c.md", "no\nalpha");
    put(root, ".git/config", "alpha");
    const s = await openAt();
    expect(await s.call("search", { query: "alpha" }, { signal })).toEqual({
      matches: [
        { path: "a.md", hash: sha256("Alpha\nbeta\nALPHABET"), line: 1n, text: "Alpha" },
        { path: "a.md", hash: sha256("Alpha\nbeta\nALPHABET"), line: 3n, text: "ALPHABET" },
        { path: "b/c.md", hash: sha256("no\nalpha"), line: 2n, text: "alpha" },
      ],
    });
    await s.close();
  });

  test("search shows at most maxResults lines and maxReadChars, saying how many more match", async () => {
    put(root, "long.md", "y".repeat(30));
    put(root, "a.md", "x1\nx2\nx3\nx4");
    const s = await openAt({ maxResults: 2 });
    const found = await s.call("search", { query: "x" }, { signal });
    expect((found.matches as JsonObject[]).map((match) => match.text)).toEqual(["x1", "x2"]);
    expect(found.cut).toBe("2 more matching lines not shown");
    // A search counts each match's path and line: 8 characters for a.md's first.
    const narrow = await openAt({ maxReadChars: 12 });
    put(root, "a.md", "xxxx\nxxxx");
    const cut = await narrow.call("search", { query: "x" }, { signal });
    expect((cut.matches as JsonObject[]).map((match) => match.text)).toEqual(["xxxx"]);
    expect(cut.cut).toBe("1 more matching line not shown");
    // A first match longer than maxReadChars is shown cut, so the result is never empty.
    expect(await narrow.call("search", { query: "y" }, { signal })).toEqual({
      matches: [{ path: "long.md", hash: sha256("y".repeat(30)), line: 1n, text: "yyyyy" }],
      cut: "line 1 of long.md is cut after 5 characters",
    });
    await s.close();
    await narrow.close();
  });

  test("a result's size counts the text it shows, so many short matches stay within maxReadChars", async () => {
    for (let i = 0; i < 30; i++) put(root, `f${String(i).padStart(2, "0")}.md`, "hit");
    const s = await openAt({ maxReadChars: 50 });
    const found = await s.call("search", { query: "hit" }, { signal });
    const matches = found.matches as JsonObject[];
    const shownChars = matches.reduce((sum, match) => sum + String(match.path).length + String(match.text).length, 0);
    expect(shownChars).toBeLessThanOrEqual(50);
    expect(matches).toHaveLength(5);
    expect(found.cut).toBe("25 more matching lines not shown");
    await s.close();
  });

  test.each([
    [".git/config", "`.git/config` isn't part of the memory: a hidden name"],
    ["../x.md", "`../x.md` isn't part of the memory: a `..` part"],
    ["/etc/passwd", "`/etc/passwd` isn't part of the memory: an absolute path"],
    ["linked/a.md", "`linked/a.md` isn't part of the memory: `linked` is a symlink"],
    ["link.md", "`link.md` isn't part of the memory: a symlink"],
    ["hard.md", "`hard.md` isn't part of the memory: a hard link"],
    ["bin.dat", "`bin.dat` isn't part of the memory: not UTF-8 text"],
    ["big.md", "`big.md` isn't part of the memory: over 10 characters"],
    ["a", "`a` isn't part of the memory: a folder"],
    ["a/b/c/d/e/f.md", "`a/b/c/d/e/f.md` isn't part of the memory: more than 4 folders deep"],
    ["missing.md", "`missing.md` doesn't exist"],
    [`${"n".repeat(300)}.md`, `\`${"n".repeat(300)}.md\` isn't part of the memory: a name longer than 255 bytes`],
  ])("a read of %s is refused with its reason", async (path, reason) => {
    put(root, ".git/config", "g");
    put(root, "a/b.md", "b");
    put(root, "bin.dat", new Uint8Array([0xff]));
    put(root, "big.md", "x".repeat(11));
    put(dir, "outside/a.md", "secret");
    fs.symlinkSync(join(dir, "outside"), join(root, "linked"));
    fs.symlinkSync(join(dir, "outside/a.md"), join(root, "link.md"));
    fs.linkSync(join(dir, "outside/a.md"), join(root, "hard.md"));
    const s = await openAt({ maxFileChars: 10 });
    const error = await rejection(s.call("read", { path }, { signal }));
    expect((error as Error).message).toBe(reason);
    expect(isReportedToolError(error)).toBe(true);
    expect(isPermanentToolError(error)).toBe(true);
    await expect(s.call("search", { query: "secret" }, { signal })).resolves.toEqual({ matches: [] });
    await s.close();
  });

  test("a file hard-linked elsewhere isn't part of the memory: list, search and recall skip it", async () => {
    put(dir, "secret.txt", "SECRET");
    put(root, "a.md", "A");
    fs.linkSync(join(dir, "secret.txt"), join(root, "notes.md"));
    const s = await openAt();
    expect(await s.call("list", {}, { signal })).toEqual({ files: [{ path: "a.md", hash: sha256("A"), chars: 1n }] });
    expect(await s.call("search", { query: "secret" }, { signal })).toEqual({ matches: [] });
    const { version, recall } = await s.snapshot();
    expect(recall.cut).toBe("1 entry skipped: not part of the memory");
    expect(version).toBe(memoryVersion([{ path: "a.md", text: "A", hash: sha256("A"), chars: 1 }]));
    await s.close();
  });

  test("a list of a folder with a name past the file system's limit is refused, not failed", async () => {
    const s = await openAt();
    const dirName = "n".repeat(300);
    await expect(s.call("list", { dir: dirName }, { signal })).rejects.toThrow(
      `\`${dirName}\` isn't a folder of the memory: a name longer than 255 bytes`,
    );
    await s.close();
  });

  test("a write tool or an unknown tool isn't a read tool", async () => {
    const s = await openAt();
    await expect(s.call("append", { path: "a.md", text: "x" }, { signal })).rejects.toThrow(
      "append is a write tool: apply it as a change",
    );
    await expect(s.call("upsert", {}, { signal })).rejects.toThrow(
      "FileMemory has no tool upsert; it has list, read, search, append, create, edit, delete, rewrite",
    );
    await s.close();
  });
});

const author: Author = { kind: "tool", name: "save", run: "r", node: "save" };

/** Return the refusal `reason`, which wrote nothing. */
function refusal(reason: string): Applied {
  return { ok: false, reason, current: null };
}

/** Apply `change` as change `id` on `s`. */
function apply(s: MemorySession, change: JsonObject, id = "c1"): Promise<Applied> {
  return s.apply({ id, change, author, signal });
}

/**
 * Return everything under `base`, symlinks not followed: each file's text, `dir` for a folder and
 * the target of a symlink, keyed by relative path. The lock file, whose bytes SQLite owns, counts
 * only by name.
 */
function tree(base: string): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (rel: string): void => {
    for (const name of fs.readdirSync(join(base, rel)).sort()) {
      const path = rel === "" ? name : `${rel}/${name}`;
      const stats = fs.lstatSync(join(base, path));
      if (stats.isSymbolicLink()) out[path] = `-> ${fs.readlinkSync(join(base, path))}`;
      else if (stats.isDirectory()) {
        out[path] = "dir";
        walk(path);
      } else out[path] = path.endsWith(".yian/lock") ? "lock" : fs.readFileSync(join(base, path), "utf8");
    }
  };
  walk("");
  return out;
}

/** A path of four folders of 255-byte names: each name fits, the whole is 1028 bytes. */
const LONG_PATH = ["a", "b", "c", "d"].map((letter) => letter.repeat(255)).join("/") + "/x.md";

/** The paths every write refuses, with the reason it gives. */
const REFUSED_PATHS: [string, string][] = [
  ["../x.md", "`../x.md` isn't part of the memory: a `..` part"],
  ["a/../../x.md", "`a/../../x.md` isn't part of the memory: a `..` part"],
  [".yian/journal.jsonl", "`.yian/journal.jsonl` isn't part of the memory: a hidden name"],
  [".yian/identity", "`.yian/identity` isn't part of the memory: a hidden name"],
  [".git/config", "`.git/config` isn't part of the memory: a hidden name"],
  [
    "-x.md",
    "`-x.md` isn't part of the memory: a name other than letters, digits, `.`, `_` and `-`, starting with a letter or digit",
  ],
  ["a/b/c/d/e/x.md", "`a/b/c/d/e/x.md` isn't part of the memory: more than 4 folders deep"],
  ["/tmp/x.md", "`/tmp/x.md` isn't part of the memory: an absolute path"],
  ["linked/x.md", "`linked/x.md` isn't part of the memory: `linked` is a symlink"],
  ["link.md", "`link.md` isn't part of the memory: a symlink"],
  ["hard.md", "`hard.md` isn't part of the memory: a hard link"],
  ["bin.dat", "`bin.dat` isn't part of the memory: not UTF-8 text"],
  [
    `notes/${"n".repeat(256)}.md`,
    `\`notes/${"n".repeat(256)}.md\` isn't part of the memory: a name longer than 255 bytes`,
  ],
  [LONG_PATH, `\`${LONG_PATH}\` isn't part of the memory: a path longer than 1024 bytes`],
];

/**
 * Lay out a memory with a symlinked folder, a symlinked file, a hidden folder and a binary file,
 * pointing outside it, then check that `change` on each refused path gives its reason and writes
 * nothing anywhere: not outside the root, not the journal, not the members.
 */
async function expectConfined(change: (path: string) => JsonObject): Promise<void> {
  put(root, "a.md", "A");
  put(root, ".git/config", "g");
  put(root, "bin.dat", new Uint8Array([0xff]));
  put(dir, "outside/x.md", "outside");
  fs.symlinkSync(join(dir, "outside"), join(root, "linked"));
  fs.symlinkSync(join(dir, "outside/x.md"), join(root, "link.md"));
  fs.linkSync(join(dir, "outside/x.md"), join(root, "hard.md"));
  const s = await openAt();
  // A first change makes the journal, so a refusal can be seen to leave it alone.
  expect((await apply(s, { op: "append", path: "a.md", text: "B" }, "first")).ok).toBe(true);
  const before = tree(dir);
  for (const [path, reason] of REFUSED_PATHS) {
    expect(await apply(s, change(path), `id-${path}`)).toEqual({ ok: false, reason, current: null });
  }
  expect(tree(dir)).toEqual(before);
  await s.close();
}

describe("append", () => {
  test("append creates a missing file and its folders, private, and never conflicts", async () => {
    const s = await openAt();
    const applied = await apply(s, { op: "append", path: "a/b/c.md", text: "one" });
    expect(applied).toMatchObject({ ok: true, result: { path: "a/b/c.md", hash: sha256("one") } });
    expect(fs.readFileSync(join(root, "a/b/c.md"), "utf8")).toBe("one");
    expect(fs.statSync(join(root, "a")).mode & 0o777).toBe(0o700);
    expect(fs.statSync(join(root, "a/b")).mode & 0o777).toBe(0o700);
    expect(fs.statSync(join(root, "a/b/c.md")).mode & 0o777).toBe(0o600);
    expect(fs.readdirSync(join(root, ".yian/tmp"))).toEqual([]);
    await s.close();
  });

  test("append adds its text on a new line", async () => {
    put(root, "a.md", "one");
    put(root, "b.md", "one\n");
    put(root, "c.md", "");
    const s = await openAt();
    for (const path of ["a.md", "b.md", "c.md"]) {
      expect((await apply(s, { op: "append", path, text: "two" }, path)).ok).toBe(true);
    }
    expect(["a.md", "b.md", "c.md"].map((path) => fs.readFileSync(join(root, path), "utf8"))).toEqual([
      "one\ntwo",
      "one\ntwo",
      "two",
    ]);
    await s.close();
  });

  test("its result's version is the memory's version after the change", async () => {
    put(root, "keep.md", "K");
    const s = await openAt();
    const applied = await apply(s, { op: "append", path: "a.md", text: "x" });
    const version = (await s.snapshot()).version;
    expect(applied).toEqual({ ok: true, version, result: { path: "a.md", hash: sha256("x"), version } });
    await s.close();
  });

  test("an append over a file, total or file-count cap is refused with the reason, writing nothing", async () => {
    put(root, "a.md", "12345");
    put(root, "b.md", "1234");
    // 9 characters in 2 files; an append adds its line break too.
    const s = await openAt({ maxFileChars: 6, maxTotalChars: 10, maxFiles: 2 });
    const before = tree(dir);
    expect(await apply(s, { op: "append", path: "a.md", text: "6" }, "1")).toEqual({
      ok: false,
      reason: "`a.md` would hold 7 characters; a file may hold 6",
      current: null,
    });
    expect(await apply(s, { op: "append", path: "b.md", text: "x" }, "2")).toEqual({
      ok: false,
      reason: "the memory would hold 11 characters; it may hold 10",
      current: null,
    });
    expect(await apply(s, { op: "append", path: "c.md", text: "x" }, "3")).toEqual({
      ok: false,
      reason: "the memory would hold 3 files; it may hold 2",
      current: null,
    });
    expect(tree(dir)).toEqual(before);
    await s.close();
  });

  test("confinement: append on a path outside the members is refused with the reason, writing nothing", async () => {
    await expectConfined((path) => ({ op: "append", path, text: "x" }));
  });

  test("a name too long for the file system is refused before it is journaled; the memory keeps working", async () => {
    const s = await openAt();
    const path = `notes/${"b".repeat(300)}.md`;
    expect(await apply(s, { op: "append", path, text: "x" }, "long")).toEqual(
      refusal(`\`${path}\` isn't part of the memory: a name longer than 255 bytes`),
    );
    expect(fs.existsSync(join(root, ".yian/journal.jsonl"))).toBe(false);
    expect((await apply(s, { op: "append", path: "a.md", text: "x" }, "next")).ok).toBe(true);
    await s.close();
  });

  test("a change of the wrong shape is a fault, not a refusal", async () => {
    const s = await openAt();
    await expect(apply(s, { op: "upsert", path: "a.md" })).rejects.toThrow(
      "FileMemory has no change upsert; its changes are append, create, edit, delete, rewrite",
    );
    await expect(apply(s, { op: "append", path: "a.md" })).rejects.toThrow(
      "a FileMemory append needs text as a string",
    );
    await expect(apply(s, { op: "append", path: "a.md", text: "x", mode: "fast" })).rejects.toThrow(
      "a FileMemory append takes path and text, not mode",
    );
    const unnamed = s.apply({ id: "", change: { op: "append", path: "a.md", text: "x" }, author, signal });
    await expect(unnamed).rejects.toThrow("a change id must be a non-empty string");
    await s.close();
  });
});

describe("create and edit", () => {
  test("create applies and returns {path, hash, version}, making the file's folders", async () => {
    const s = await openAt();
    const applied = await apply(s, { op: "create", path: "people/ada.md", text: "Ada." });
    const version = (await s.snapshot()).version;
    expect(applied).toEqual({ ok: true, version, result: { path: "people/ada.md", hash: sha256("Ada."), version } });
    expect(fs.readFileSync(join(root, "people/ada.md"), "utf8")).toBe("Ada.");
    expect(fs.statSync(join(root, "people")).mode & 0o777).toBe(0o700);
    await s.close();
  });

  test("create on an existing file conflicts with its current hash and content, capped, writing nothing", async () => {
    put(root, "a.md", "0123456789abc");
    const s = await openAt({ maxReadChars: 10 });
    const before = tree(dir);
    expect(await apply(s, { op: "create", path: "a.md", text: "new" })).toEqual({
      ok: false,
      reason: "`a.md` already exists",
      current: {
        path: "a.md",
        hash: sha256("0123456789abc"),
        text: "0123456789",
        cut: "3 more characters not shown; read the file for the rest",
      },
    });
    expect(tree(dir)).toEqual(before);
    await s.close();
  });

  test("edit replaces the exact old text once", async () => {
    put(root, "a.md", "Prefer tabs.\nUse bun.\n");
    const s = await openAt();
    const applied = await apply(s, { op: "edit", path: "a.md", old: "tabs", new: "spaces" });
    const version = (await s.snapshot()).version;
    const text = "Prefer spaces.\nUse bun.\n";
    expect(applied).toEqual({ ok: true, version, result: { path: "a.md", hash: sha256(text), version } });
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe(text);
    await s.close();
  });

  test.each([
    ["absent", "Prefer tabs.", "spaces", "the old text isn't in `a.md`; read the file and copy it exactly"],
    ["present twice", "Use", "Try", "the old text appears more than once in `a.md`; give more of it"],
    ["overlapping itself", "aa", "b", "the old text appears more than once in `a.md`; give more of it"],
    ["empty", "", "x", "the old text is empty; give the exact text to replace"],
  ])("edit with old %s conflicts with the current hash and content, writing nothing", async (_, old, text, reason) => {
    const content = "Prefer Tabs.\nUse bun.\nUse git.\naaa";
    put(root, "a.md", content);
    const s = await openAt();
    const before = tree(dir);
    expect(await apply(s, { op: "edit", path: "a.md", old, new: text })).toEqual({
      ok: false,
      reason,
      current: { path: "a.md", hash: sha256(content), text: content },
    });
    expect(tree(dir)).toEqual(before);
    await s.close();
  });

  test("edit of a file that doesn't exist conflicts, writing nothing", async () => {
    const s = await openAt();
    expect(await apply(s, { op: "edit", path: "a.md", old: "x", new: "y" })).toEqual({
      ok: false,
      reason: "`a.md` doesn't exist",
      current: null,
    });
    expect(fs.existsSync(join(root, "a.md"))).toBe(false);
    await s.close();
  });

  test("create and edit keep to the caps", async () => {
    put(root, "a.md", "1234");
    const s = await openAt({ maxFileChars: 5, maxTotalChars: 8, maxFiles: 2 });
    const before = tree(dir);
    expect(await apply(s, { op: "create", path: "b.md", text: "123456" }, "1")).toEqual(
      refusal("`b.md` would hold 6 characters; a file may hold 5"),
    );
    expect(await apply(s, { op: "create", path: "b.md", text: "12345" }, "2")).toEqual(
      refusal("the memory would hold 9 characters; it may hold 8"),
    );
    expect(await apply(s, { op: "edit", path: "a.md", old: "4", new: "456" }, "3")).toEqual(
      refusal("`a.md` would hold 6 characters; a file may hold 5"),
    );
    expect(tree(dir)).toEqual(before);
    expect((await apply(s, { op: "create", path: "b.md", text: "1" }, "4")).ok).toBe(true);
    expect(await apply(s, { op: "create", path: "c.md", text: "1" }, "5")).toEqual(
      refusal("the memory would hold 3 files; it may hold 2"),
    );
    await s.close();
  });

  test("confinement: create on a path outside the members is refused with the reason, writing nothing", async () => {
    await expectConfined((path) => ({ op: "create", path, text: "x" }));
  });

  test("confinement: edit on a path outside the members is refused with the reason, writing nothing", async () => {
    await expectConfined((path) => ({ op: "edit", path, old: "x", new: "y" }));
  });
});

describe("delete and rewrite", () => {
  test("delete applies when the file's hash is if_hash, and returns {path, version}", async () => {
    put(root, "a.md", "A");
    put(root, "b.md", "B");
    const s = await openAt();
    const applied = await apply(s, { op: "delete", path: "a.md", if_hash: sha256("A") });
    const version = (await s.snapshot()).version;
    expect(applied).toEqual({ ok: true, version, result: { path: "a.md", version } });
    expect(fs.existsSync(join(root, "a.md"))).toBe(false);
    expect(fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8")).toContain(
      `"before":"${sha256("A")}","after":"absent","inverse":{"text":"A"}`,
    );
    await s.close();
  });

  test("rewrite applies when the file's hash is if_hash, and returns {path, hash, version}", async () => {
    put(root, "a.md", "old");
    const s = await openAt();
    const applied = await apply(s, { op: "rewrite", path: "a.md", text: "new", if_hash: sha256("old") });
    const version = (await s.snapshot()).version;
    expect(applied).toEqual({ ok: true, version, result: { path: "a.md", hash: sha256("new"), version } });
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("new");
    await s.close();
  });

  test.each([
    ["delete", { op: "delete", path: "a.md", if_hash: sha256("stale") }],
    ["rewrite", { op: "rewrite", path: "a.md", text: "new", if_hash: sha256("stale") }],
  ])("%s with another hash conflicts with the current state, writing nothing", async (_, change) => {
    put(root, "a.md", "now");
    const s = await openAt();
    const before = tree(dir);
    expect(await apply(s, change)).toEqual({
      ok: false,
      reason: "`a.md` changed since it was read: its hash isn't if_hash",
      current: { path: "a.md", hash: sha256("now"), text: "now" },
    });
    expect(tree(dir)).toEqual(before);
    await s.close();
  });

  test.each([
    ["delete", { op: "delete", path: "a.md", if_hash: sha256("x") }],
    ["rewrite", { op: "rewrite", path: "a.md", text: "new", if_hash: sha256("x") }],
  ])("%s of a file that doesn't exist conflicts, writing nothing", async (_, change) => {
    const s = await openAt();
    expect(await apply(s, change)).toEqual({ ok: false, reason: "`a.md` doesn't exist", current: null });
    expect(fs.existsSync(join(root, "a.md"))).toBe(false);
    await s.close();
  });

  test("rewrite keeps to the caps", async () => {
    put(root, "a.md", "1234");
    put(root, "b.md", "1234");
    const s = await openAt({ maxFileChars: 5, maxTotalChars: 9 });
    const before = tree(dir);
    const hash = sha256("1234");
    expect(await apply(s, { op: "rewrite", path: "a.md", text: "123456", if_hash: hash }, "1")).toEqual(
      refusal("`a.md` would hold 6 characters; a file may hold 5"),
    );
    expect(tree(dir)).toEqual(before);
    expect((await apply(s, { op: "rewrite", path: "a.md", text: "12345", if_hash: hash }, "2")).ok).toBe(true);
    expect(await apply(s, { op: "rewrite", path: "b.md", text: "12345", if_hash: hash }, "3")).toEqual(
      refusal("the memory would hold 10 characters; it may hold 9"),
    );
    await s.close();
  });

  test("a memory already over its total cap by hand still takes a change that doesn't grow it", async () => {
    put(root, "a.md", "1234567890");
    const s = await openAt({ maxTotalChars: 5 });
    expect(await apply(s, { op: "append", path: "a.md", text: "x" }, "1")).toEqual(
      refusal("the memory would hold 12 characters; it may hold 5"),
    );
    expect((await apply(s, { op: "rewrite", path: "a.md", text: "12", if_hash: sha256("1234567890") }, "2")).ok).toBe(
      true,
    );
    await s.close();
  });

  test("confinement: delete on a path outside the members is refused with the reason, writing nothing", async () => {
    await expectConfined((path) => ({ op: "delete", path, if_hash: sha256("x") }));
  });

  test("confinement: rewrite on a path outside the members is refused with the reason, writing nothing", async () => {
    await expectConfined((path) => ({ op: "rewrite", path, text: "x", if_hash: sha256("x") }));
  });
});

/** Make `fs.writeSync` throw, as a crash would stop it, instead of writing bytes that hold `marker`. */
function crashOnWrite(marker: string): void {
  const { writeSync } = fs;
  type Offset = number | null | undefined;
  spyOn(fs, "writeSync").mockImplementation(((fd: number, data: Uint8Array, offset: Offset, length: Offset) => {
    const start = offset ?? 0;
    const bytes = Buffer.from(data.buffer, data.byteOffset + start, length ?? data.length - start);
    if (bytes.includes(marker)) throw new Error("crash before the commit line");
    return writeSync(fd, bytes);
  }) as typeof fs.writeSync);
}

/** Return the journal's lines, parsed. */
function journalLines(): JsonObject[] {
  const text = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8");
  return text
    .trimEnd()
    .split("\n")
    .map((line) => JSON.parse(line) as JsonObject);
}

/** Undo change `changeId` on `s` as change `id`. */
function undo(s: MemorySession, changeId: string, id = `undo-${changeId}`): Promise<Applied> {
  return s.undo({ id, changeId, author: { kind: "user", name: "ada" }, signal });
}

describe("history and undo", () => {
  test("history lists changes newest first, with author, run and node; aborted ones are left out", async () => {
    const s = await openAt({ scheduler: virtualScheduler(Date.UTC(2026, 10, 7)) });
    await apply(s, { op: "append", path: "a.md", text: "one" }, "r/answer/1/0");
    await s.apply({
      id: "cli-1",
      change: { op: "create", path: "b.md", text: "B" },
      author: { kind: "user", name: "ada" },
      signal,
    });
    // An entry a crash cut off before its file was replaced is aborted, and history leaves it out.
    spyOn(fs, "renameSync").mockImplementation(() => {
      throw Object.assign(new Error("no space"), { code: "ENOSPC" });
    });
    await expect(apply(s, { op: "append", path: "a.md", text: "lost" }, "cut")).rejects.toThrow("no space");
    mock.restore();
    await apply(s, { op: "append", path: "a.md", text: "two" }, "r/answer/1/1");
    const at = "2026-11-07T00:00:00.000Z";
    expect(await s.history({ limit: 10 })).toEqual([
      { id: "r/answer/1/1", op: "append", target: { path: "a.md" }, author, at },
      { id: "cli-1", op: "create", target: { path: "b.md" }, author: { kind: "user", name: "ada" }, at },
      { id: "r/answer/1/0", op: "append", target: { path: "a.md" }, author, at },
    ]);
    expect((await s.history({ limit: 1 })).map((change) => change.id)).toEqual(["r/answer/1/1"]);
    expect(await s.history({ limit: 0 })).toEqual([]);
    await s.close();
  });

  test("undo applies a change's inverse when its file is still what that change wrote", async () => {
    put(root, "a.md", "Prefer tabs.");
    const s = await openAt();
    await apply(s, { op: "edit", path: "a.md", old: "tabs", new: "spaces" }, "e");
    await apply(s, { op: "append", path: "b.md", text: "one" }, "b1");
    await apply(s, { op: "append", path: "b.md", text: "two" }, "b2");
    await apply(s, { op: "create", path: "c.md", text: "C" }, "c");
    expect(await undo(s, "c")).toMatchObject({ ok: true, result: { path: "c.md" } });
    expect(fs.existsSync(join(root, "c.md"))).toBe(false);
    const undone = await undo(s, "b2");
    expect(undone).toEqual({
      ok: true,
      version: (await s.snapshot()).version,
      result: { path: "b.md", hash: sha256("one"), version: (await s.snapshot()).version },
    });
    expect(fs.readFileSync(join(root, "b.md"), "utf8")).toBe("one");
    expect((await undo(s, "e")).ok).toBe(true);
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("Prefer tabs.");
    // An undo is a change like any other: it is in history, and undoing it redoes the change.
    expect((await s.history({ limit: 1 }))[0]).toMatchObject({
      id: "undo-e",
      op: "undo",
      target: { path: "a.md", undoes: "e" },
      author: { kind: "user", name: "ada" },
    });
    expect((await undo(s, "undo-e")).ok).toBe(true);
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("Prefer spaces.");
    await s.close();
  });

  test("undo of a deleted file brings it back; an undo's id answers again with its stored result", async () => {
    put(root, "a.md", "keep me");
    const s = await openAt();
    await apply(s, { op: "delete", path: "a.md", if_hash: sha256("keep me") }, "d");
    const undone = await undo(s, "d", "u");
    expect(undone.ok).toBe(true);
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("keep me");
    expect(await undo(s, "d", "u")).toEqual(undone);
    await expect(s.undo({ id: "u", changeId: "other", author, signal })).rejects.toThrow(
      "change id `u` was used for another change",
    );
    await s.close();
  });

  test("undo conflicts when the file changed after that change, writing nothing", async () => {
    const s = await openAt();
    await apply(s, { op: "append", path: "a.md", text: "one" }, "1");
    fs.writeFileSync(join(root, "a.md"), "one\nby hand");
    const before = tree(dir);
    expect(await undo(s, "1")).toEqual({
      ok: false,
      reason: "`a.md` changed after change `1`; undoing it would lose that",
      current: { path: "a.md", hash: sha256("one\nby hand"), text: "one\nby hand" },
    });
    expect(tree(dir)).toEqual(before);
    await s.close();
  });

  test("undo of a change the history doesn't hold, or whose outcome is unknown, is refused", async () => {
    const s = await openAt();
    expect(await undo(s, "nope")).toEqual(refusal("change `nope` isn't in the memory's history"));
    await apply(s, { op: "append", path: "a.md", text: "one" }, "1");
    // x's entry has no outcome, and its file is at neither of its hashes: its outcome is unknown.
    const lines = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8").trimEnd().split("\n");
    const entry = { ...(JSON.parse(lines[1]!) as JsonObject), id: "x", before: sha256("one"), after: sha256("two") };
    fs.appendFileSync(join(root, ".yian/journal.jsonl"), `${JSON.stringify(entry)}\n`);
    fs.writeFileSync(join(root, "a.md"), "by hand");
    expect(await undo(s, "x")).toEqual(
      refusal("change `x` may not have landed, so it can't be undone; check the file"),
    );
    await s.close();
  });

  test("inverses are kept for the last maxUndo changes, within maxHistoryChars", async () => {
    put(root, "big.md", "0123456789");
    const s = await openAt({ maxUndo: 2, maxHistoryChars: 5 });
    await apply(s, { op: "append", path: "a.md", text: "1" }, "1");
    await apply(s, { op: "append", path: "b.md", text: "2" }, "2");
    await apply(s, { op: "append", path: "c.md", text: "3" }, "3");
    const gone = refusal("change `1` can't be undone: the memory no longer keeps what it replaced");
    expect(await undo(s, "1")).toEqual(gone);
    // A rewrite's inverse is the old text, which is over maxHistoryChars; an append's costs nothing.
    await apply(s, { op: "rewrite", path: "big.md", text: "x", if_hash: sha256("0123456789") }, "4");
    expect((await undo(s, "4")).ok).toBe(false);
    expect((await undo(s, "3")).ok).toBe(true);
    await s.close();
  });

  test.each([
    ["maxUndo is 0", { maxUndo: 0 }],
    ["its old text is over maxHistoryChars", { maxHistoryChars: 5 }],
  ])("a change keeps no old text in the journal when %s, since no undo could use it", async (_, caps) => {
    put(root, "a.md", "old secret text");
    const s = await openAt(caps);
    await apply(s, { op: "rewrite", path: "a.md", text: "new", if_hash: sha256("old secret text") }, "r");
    const journal = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8");
    expect(journal).not.toContain("secret");
    expect(JSON.parse(journal.split("\n")[1]!).inverse).toBeNull();
    expect(await undo(s, "r")).toEqual(
      refusal("change `r` can't be undone: the memory no longer keeps what it replaced"),
    );
    await s.close();
  });

  test("an undo first resolves another session's cut-off change, then conflicts on the moved hash", async () => {
    const s = await openAt();
    await apply(s, { op: "append", path: "a.md", text: "one" }, "1");
    const other = await openAt();
    crashOnWrite('"ok":true');
    await expect(apply(other, { op: "append", path: "a.md", text: "two" }, "x")).rejects.toThrow("crash");
    mock.restore();
    const result = await undo(s, "1");
    expect(result).toMatchObject({ ok: false, reason: "`a.md` changed after change `1`; undoing it would lose that" });
    const journal = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8").trimEnd().split("\n");
    expect(JSON.parse(journal.at(-1)!)).toMatchObject({ id: "x", ok: true });
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one\ntwo");
    await s.close();
    await other.close();
  });

  test("an undo of a hand-edited append that left no file is refused, not failed", async () => {
    const s = await openAt();
    await apply(s, { op: "append", path: "a.md", text: "one" }, "1");
    const entry = { ...journalLines()[1]!, id: "h", path: "gone.md", after: "absent", inverse: { truncate: 3 } };
    const commit = { id: "h", ok: true, version: "v", result: { path: "gone.md" } };
    fs.appendFileSync(join(root, ".yian/journal.jsonl"), `${JSON.stringify(entry)}\n${JSON.stringify(commit)}\n`);
    const before = tree(dir);
    expect(await undo(s, "h")).toEqual(refusal("change `h` can't be undone: its journal entry is damaged"));
    expect(tree(dir)).toEqual(before);
    await s.close();
  });

  test.each([
    ["../x.md", "`../x.md` isn't part of the memory: a `..` part"],
    [".yian/identity", "`.yian/identity` isn't part of the memory: a hidden name"],
  ])("confinement: an undo whose hand-edited entry names %s is refused, writing nothing", async (path, reason) => {
    put(dir, "x.md", "outside");
    const s = await openAt();
    await apply(s, { op: "append", path: "a.md", text: "one" }, "1");
    const target = path === "../x.md" ? join(dir, "x.md") : join(root, ".yian/identity");
    const journalPath = join(root, ".yian/journal.jsonl");
    const lines = fs.readFileSync(journalPath, "utf8").trimEnd().split("\n");
    const entry = JSON.parse(lines[1]!) as JsonObject;
    const after = sha256(fs.readFileSync(target));
    const forged = { ...entry, id: "evil", path, before: "absent", after, inverse: { text: "overwritten" } };
    const commit = { id: "evil", ok: true, version: "v", result: { path } };
    fs.appendFileSync(journalPath, `${JSON.stringify(forged)}\n${JSON.stringify(commit)}\n`);
    const before = tree(dir);
    expect(await undo(s, "evil")).toEqual(refusal(reason));
    expect(tree(dir)).toEqual(before);
    await s.close();
  });
});

describe("errors name no location", () => {
  const asUser = process.platform !== "win32" && process.getuid!() !== 0;

  test.if(asUser)("an OS error from a write names paths in the memory, never its location", async () => {
    put(root, "ro/a.md", "A");
    const s = await openAt();
    fs.chmodSync(join(root, "ro"), 0o500);
    let error: unknown;
    try {
      error = await rejection(apply(s, { op: "append", path: "ro/a.md", text: "x" }, "x"));
    } finally {
      fs.chmodSync(join(root, "ro"), 0o700);
    }
    expect((error as NodeJS.ErrnoException).code).toBe("EACCES");
    expect((error as Error).message).toContain("ro/a.md");
    expect(`${(error as Error).message} ${(error as Error).stack}`).not.toContain(dir);
    expect(Object.values(error as object).join(" ")).not.toContain(dir);
    // The change was journaled before the rename failed; the next write resolves it and goes on.
    expect((await apply(s, { op: "append", path: "ro/a.md", text: "x" }, "x")).ok).toBe(true);
    await s.close();
  });

  test.if(asUser)("an OS error from a read names no location", async () => {
    put(root, "a.md", "A");
    fs.chmodSync(join(root, "a.md"), 0o000);
    const s = await openAt();
    const error = await rejection(s.call("read", { path: "a.md" }, { signal }));
    expect((error as NodeJS.ErrnoException).code).toBe("EACCES");
    expect(`${(error as Error).message} ${(error as Error).stack}`).not.toContain(dir);
    await s.close();
  });

  test("a busy memory's error names no location, not even in its cause", async () => {
    const s = await openAt({ lockTimeoutMs: 0 });
    const lock = lockFile(join(root, ".yian/lock"), { private: true });
    try {
      const error = await rejection(s.snapshot());
      expect(error).toBeInstanceOf(MemoryBusyError);
      expect(String((error as Error).cause ?? "")).not.toContain(dir);
      expect((error as Error).message).not.toContain(dir);
    } finally {
      lock.release();
    }
    await s.close();
  });
});

/** Erase `target` on `s` as change `id`, by a person at the CLI. */
function erase(s: MemorySession, target: JsonObject, id = "erase-1") {
  return s.erase({ id, target, author: { kind: "user", name: "ada" }, signal });
}

/** Return every file under `base` whose bytes hold `text`, `.yian/` included. */
function holding(base: string, text: string): string[] {
  const found: string[] = [];
  const walk = (rel: string): void => {
    for (const name of fs.readdirSync(join(base, rel))) {
      const path = rel === "" ? name : `${rel}/${name}`;
      const stats = fs.lstatSync(join(base, path));
      if (stats.isDirectory()) walk(path);
      else if (stats.isFile() && fs.readFileSync(join(base, path)).includes(text)) found.push(path);
    }
  };
  walk("");
  return found;
}

describe("erase and reset", () => {
  test("erase({path, text}) removes the text from the file and every inverse, and names the undos lost", async () => {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "Ada lives at SECRET-42." }, "c1");
    await apply(s, { op: "append", path: "a.md", text: "She likes tea." }, "c2");
    await apply(s, { op: "edit", path: "a.md", old: "tea", new: "coffee" }, "c3");
    await apply(s, { op: "create", path: "b.md", text: "Bob knows SECRET-42 too." }, "b1");
    await apply(s, { op: "edit", path: "b.md", old: "Bob knows SECRET-42 too.", new: "Bob knows." }, "b2");
    await apply(s, { op: "create", path: "c.md", text: "Carol." }, "d1");
    expect(holding(root, "SECRET-42").sort()).toEqual([".yian/journal.jsonl", "a.md"]);
    const erased = await erase(s, { path: "a.md", text: "SECRET-42" });
    expect(erased).toEqual({ version: (await s.snapshot()).version, disabledUndos: ["c1", "c2", "c3", "b2"] });
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("Ada lives at .\nShe likes coffee.");
    expect(holding(root, "SECRET-42")).toEqual([]);
    // Only the undos that could have brought the text back are gone.
    expect((await undo(s, "c3")).ok).toBe(false);
    expect((await undo(s, "d1")).ok).toBe(true);
    await s.close();
  });

  test("erase removes its text until none is left, though removing it once forms it again", async () => {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "keep secsecretret and sesecretcret keep" }, "a1");
    await apply(s, { op: "create", path: "b.md", text: "b: secsecretret" }, "b1");
    await apply(s, { op: "rewrite", path: "b.md", text: "b", if_hash: sha256("b: secsecretret") }, "b2");
    await erase(s, { path: "a.md", text: "secret" });
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("keep  and  keep");
    // b2's inverse held the text inside a longer word: the whole inverse goes.
    expect(holding(root, "secret")).toEqual([]);
    expect(holding(root, "sec")).toEqual([]);
    await s.close();
  });

  test("erase({path}) deletes the file and every inverse for it", async () => {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "private" }, "c1");
    await apply(s, { op: "rewrite", path: "a.md", text: "more private", if_hash: sha256("private") }, "c2");
    const erased = await erase(s, { path: "a.md" });
    expect(erased.disabledUndos).toEqual(["c1", "c2"]);
    expect(fs.existsSync(join(root, "a.md"))).toBe(false);
    expect(holding(root, "private")).toEqual([]);
    expect(await undo(s, "c2")).toEqual(
      refusal("change `c2` can't be undone: the memory no longer keeps what it replaced"),
    );
    await s.close();
  });

  test("an erase is in history, with its author and time, and can't be undone", async () => {
    const s = await openAt({ scheduler: virtualScheduler(Date.UTC(2026, 10, 7)) });
    await apply(s, { op: "create", path: "a.md", text: "x" }, "c1");
    await erase(s, { path: "a.md", text: "x" }, "e1");
    expect((await s.history({ limit: 1 }))[0]).toEqual({
      id: "e1",
      op: "erase",
      target: { path: "a.md" },
      author: { kind: "user", name: "ada" },
      at: "2026-11-07T00:00:00.000Z",
    });
    expect(await undo(s, "e1")).toEqual(
      refusal("change `e1` can't be undone: the memory no longer keeps what it replaced"),
    );
    await s.close();
  });

  test("an erase's id answers again with its stored result; another erase under it is refused", async () => {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "x y" }, "c1");
    const erased = await erase(s, { path: "a.md", text: "x" });
    const journal = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8");
    expect(await erase(s, { path: "a.md", text: "x" })).toEqual(erased);
    expect(fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8")).toBe(journal);
    await expect(erase(s, { path: "a.md", text: "y" })).rejects.toThrow(
      "change id `erase-1` was used for another change",
    );
    await s.close();
  });

  test("an erase first resolves another session's cut-off change on its path", async () => {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "one" }, "c1");
    const other = await openAt();
    crashOnWrite('"ok":true');
    await expect(apply(other, { op: "append", path: "a.md", text: "SECRET" }, "x")).rejects.toThrow("crash");
    mock.restore();
    const erased = await erase(s, { path: "a.md", text: "SECRET" });
    expect(erased.disabledUndos).toEqual(["c1", "x"]);
    const xLines = journalLines().filter((line) => line.id === "x");
    expect(xLines.map((line) => line.ok ?? line.inverse)).toEqual([null, true]);
    expect(holding(root, "SECRET")).toEqual([]);
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one\n");
    await s.close();
    await other.close();
  });

  test("an erase a crash cut off is aborted before its file is written, and committed after", async () => {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "keep SECRET" }, "c1");
    const { renameSync } = fs;
    spyOn(fs, "renameSync").mockImplementation((from: fs.PathLike, to: fs.PathLike) => {
      if (String(to).endsWith("a.md")) throw Object.assign(new Error("no space"), { code: "ENOSPC" });
      renameSync(from, to);
    });
    await expect(erase(s, { path: "a.md", text: "SECRET" }, "e1")).rejects.toThrow("no space");
    mock.restore();
    // The journal was rewritten first, so the text is gone from it though the file still holds it.
    expect(holding(root, "SECRET")).toEqual(["a.md"]);
    const t = await openAt();
    expect(journalLines().at(-1)).toEqual({ id: "e1", aborted: true });
    expect(await erase(t, { path: "a.md", text: "SECRET" }, "e1")).toMatchObject({ disabledUndos: [] });
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("keep ");

    await apply(t, { op: "append", path: "a.md", text: "AGAIN" }, "c2");
    // e2's own commit line, not the rewritten journal's lines.
    crashOnWrite('{"id":"e2","ok":true');
    await expect(erase(t, { path: "a.md", text: "AGAIN" }, "e2")).rejects.toThrow("crash");
    mock.restore();
    const u = await openAt();
    expect(journalLines().at(-1)).toMatchObject({ id: "e2", ok: true, result: { disabled_undos: ["c2"] } });
    expect(await erase(u, { path: "a.md", text: "AGAIN" }, "e2")).toMatchObject({ disabledUndos: ["c2"] });
    expect(holding(root, "AGAIN")).toEqual([]);
    for (const session of [s, t, u]) await session.close();
  });

  test("confinement: an erase target outside the members is refused with the reason, writing nothing", async () => {
    put(root, "a.md", "A");
    put(root, ".git/config", "g");
    put(root, "bin.dat", new Uint8Array([0xff]));
    put(dir, "outside/x.md", "outside");
    fs.symlinkSync(join(dir, "outside"), join(root, "linked"));
    fs.symlinkSync(join(dir, "outside/x.md"), join(root, "link.md"));
    fs.linkSync(join(dir, "outside/x.md"), join(root, "hard.md"));
    const s = await openAt();
    expect((await apply(s, { op: "append", path: "a.md", text: "B" }, "first")).ok).toBe(true);
    // A file a hand edit put on disk has its history scrubbed: only the journal may change.
    const outsideJournal = () => {
      const { ["m/.yian/journal.jsonl"]: _, ...rest } = tree(dir);
      return rest;
    };
    const before = outsideJournal();
    for (const [path, reason] of REFUSED_PATHS) {
      await expect(erase(s, { path }, `erase-${path}`)).rejects.toThrow(reason);
      await expect(erase(s, { path, text: "outside" }, `erase-text-${path}`)).rejects.toThrow(reason);
    }
    expect(outsideJournal()).toEqual(before);
    await s.close();
  });

  test.each([
    ["too big by hand", (path: string) => fs.writeFileSync(path, "x".repeat(60)), "over 50 characters"],
    ["hard-linked by hand", (path: string) => fs.linkSync(path, join(dir, "elsewhere.md")), "a hard link"],
  ])(
    "an erase on a file %s scrubs the history anyway, and says to remove the file by hand",
    async (_, handEdit, why) => {
      const s = await openAt({ maxFileChars: 50 });
      await apply(s, { op: "create", path: "a.md", text: "PIN 1234" }, "c1");
      await apply(s, { op: "rewrite", path: "a.md", text: "clean", if_hash: sha256("PIN 1234") }, "c2");
      handEdit(join(root, "a.md"));
      expect(holding(join(root, ".yian"), "1234")).toEqual(["journal.jsonl"]);
      await expect(erase(s, { path: "a.md", text: "1234" })).rejects.toThrow(
        `\`a.md\` isn't part of the memory: ${why}, so erase left the file alone; the memory's history no ` +
          "longer holds the text, but the file may: remove it by hand",
      );
      expect(holding(join(root, ".yian"), "1234")).toEqual([]);
      await s.close();
    },
  );

  test("an erase target of the wrong shape is a fault", async () => {
    const s = await openAt();
    await expect(erase(s, { path: "a.md", text: "" })).rejects.toThrow("an erase's text must be a non-empty string");
    await expect(erase(s, { path: "a.md", other: "x" })).rejects.toThrow("an erase takes path and text, not other");
    await expect(erase(s, {})).rejects.toThrow("an erase needs path as a string");
    await s.close();
  });

  test("reset empties the members, keeps the identity, and starts a new journal generation", async () => {
    put(root, ".git/config", "kept");
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "A" }, "c1");
    await apply(s, { op: "create", path: "notes/b.md", text: "B" }, "c2");
    const { generation } = journalLines()[0]!;
    const memory = new FileMemory({ root, id: "t/x" });
    await memory.reset({ signal });
    expect(readMembers(root, { maxDepth: 4, maxFileChars: 100 }).files).toEqual([]);
    expect(fs.existsSync(join(root, "notes"))).toBe(false);
    expect(fs.readFileSync(join(root, ".git/config"), "utf8")).toBe("kept");
    expect(fs.readFileSync(join(root, ".yian/identity"), "utf8")).toBe("t/x");
    expect(journalLines()).toHaveLength(1);
    expect(journalLines()[0]!.generation).not.toBe(generation);
    // A session opened before the reset reads the new generation: its ids are gone, so c1 applies again.
    expect((await apply(s, { op: "create", path: "a.md", text: "A" }, "c1")).ok).toBe(true);
    expect(await s.history({ limit: 10 })).toHaveLength(1);
    await s.close();
  });

  test("reset refuses a folder that isn't a memory", async () => {
    put(root, "a.md", "A");
    await expect(new FileMemory({ root }).reset({ signal })).rejects.toThrow(`\`${root}\` isn't a memory yet`);
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("A");
  });
});

describe("the memory's own entries in .yian/", () => {
  /** Make a memory with one change, then a folder outside it holding a file to protect. */
  async function withVictim(): Promise<string> {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "x" }, "c1");
    await s.close();
    const victim = join(dir, "victim");
    put(victim, "important.txt", "keep me");
    put(victim, "sub/deep.txt", "keep me too");
    return victim;
  }

  test.each([
    ["open", (memory: FileMemory) => memory.open({ ...o, create: false })],
    ["reset", (memory: FileMemory) => memory.reset({ signal })],
  ])("%s refuses a .yian/tmp that is a symlink, and deletes nothing through it", async (_, use) => {
    const victim = await withVictim();
    fs.rmSync(join(root, ".yian/tmp"), { recursive: true });
    fs.symlinkSync(victim, join(root, ".yian/tmp"));
    await expect(use(new FileMemory({ root, id: "t/x" }))).rejects.toThrow(
      "`.yian/tmp` isn't a folder of this memory (a symlink); remove it",
    );
    expect(tree(victim)).toEqual({ "important.txt": "keep me", sub: "dir", "sub/deep.txt": "keep me too" });
    expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("x");
  });

  test("a folder that arrives with .yian/tmp pointing above it is refused, and its neighbours are kept", async () => {
    const parent = join(dir, "projects");
    const memoryRoot = join(parent, "mem");
    put(memoryRoot, ".yian/identity", "team/notes");
    put(memoryRoot, "notes.md", "hi");
    fs.symlinkSync("../..", join(memoryRoot, ".yian/tmp"));
    put(parent, "other-project/work.txt", "precious");
    await expect(new FileMemory({ root: memoryRoot }).open({ ...o, create: false })).rejects.toThrow(
      "`.yian/tmp` isn't a folder of this memory (a symlink); remove it",
    );
    expect(fs.readFileSync(join(parent, "other-project/work.txt"), "utf8")).toBe("precious");
  });

  test("a session's next call refuses a .yian/tmp swapped for a symlink after it opened", async () => {
    const s = await openAt();
    const victim = join(dir, "victim");
    put(victim, "important.txt", "keep me");
    fs.rmSync(join(root, ".yian/tmp"), { recursive: true });
    fs.symlinkSync(victim, join(root, ".yian/tmp"));
    for (const call of [() => s.history({ limit: 5 }), () => apply(s, { op: "append", path: "a.md", text: "x" })]) {
      await expect(call()).rejects.toThrow("`.yian/tmp` isn't a folder of this memory (a symlink); remove it");
    }
    expect(fs.readFileSync(join(victim, "important.txt"), "utf8")).toBe("keep me");
    await s.close();
  });

  test.each([
    [".yian", "symlink", "`.yian` isn't a folder of this memory (a symlink); remove it"],
    [".yian/journal.jsonl", "symlink", "`.yian/journal.jsonl` isn't a file of this memory (a symlink); replace it"],
    [".yian/journal.jsonl", "hard link", "`.yian/journal.jsonl` isn't a file of this memory (a hard link); replace it"],
    [".yian/identity", "symlink", "`.yian/identity` isn't a file of this memory (a symlink); replace it with a copy"],
    [".yian/lock", "symlink", "`.yian/lock` isn't a file of this memory (a symlink); remove it"],
    [".yian/tmp", "file", "`.yian/tmp` isn't a folder of this memory (not a folder); remove it"],
  ])("open refuses %s as a %s", async (name, kind, message) => {
    const s = await openAt();
    await apply(s, { op: "create", path: "a.md", text: "x" }, "c1");
    await s.close();
    const path = join(root, name);
    const copy = join(dir, "copy");
    if (kind === "hard link") {
      fs.linkSync(path, copy);
    } else {
      fs.cpSync(path, copy, { recursive: true });
      fs.rmSync(path, { recursive: true });
      if (kind === "symlink") fs.symlinkSync(copy, path);
      else fs.writeFileSync(path, "");
    }
    await expect(new FileMemory({ root, id: "t/x" }).open({ ...o, create: false })).rejects.toThrow(message);
  });

  test("temp cleanup unlinks only plain files directly in .yian/tmp, and follows nothing", async () => {
    const s = await openAt();
    const victim = join(dir, "victim");
    put(victim, "important.txt", "keep me");
    fs.symlinkSync(victim, join(root, ".yian/tmp/link"));
    put(root, ".yian/tmp/folder/inner.txt", "left alone");
    put(root, ".yian/tmp/0123456789abcdef", "left by a crash");
    expect((await apply(s, { op: "create", path: "a.md", text: "x" }, "c1")).ok).toBe(true);
    expect(fs.readdirSync(join(root, ".yian/tmp")).sort()).toEqual(["folder", "link"]);
    expect(fs.readFileSync(join(victim, "important.txt"), "utf8")).toBe("keep me");
    expect(fs.readFileSync(join(root, ".yian/tmp/folder/inner.txt"), "utf8")).toBe("left alone");
    await s.close();
  });
});
