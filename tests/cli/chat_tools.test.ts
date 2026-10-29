// The chat assistant's file tools, confined to the folder the chat started in.

import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { confine, ToolRefused } from "../../src/cli/chat_tools.ts";
import { isReportedToolError } from "../../src/tools/errors.ts";

let tmp: string;
let root: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-tools-")));
  root = join(tmp, "root");
  mkdirSync(join(root, "sub"), { recursive: true });
  writeFileSync(join(root, "sub", "a.yaml"), "id: a\n");
  writeFileSync(join(tmp, "outside.yaml"), "id: out\n");
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** What `action` threw. */
function thrown(action: () => unknown): unknown {
  try {
    action();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

describe("confine", () => {
  test("accepts a path inside the root", () => {
    expect(confine(root, "sub/a.yaml")).toBe(join(root, "sub", "a.yaml"));
    expect(confine(root, join(root, "sub"))).toBe(join(root, "sub"));
    expect(confine(root, ".")).toBe(root);
  });

  test("refuses a path that leaves the root", () => {
    const error = thrown(() => confine(root, "../outside.yaml"));
    expect(error).toBeInstanceOf(ToolRefused);
    expect(isReportedToolError(error)).toBe(true);
    expect((error as Error).message).toContain("../outside.yaml");
  });

  test("refuses an absolute path outside the root", () => {
    const path = join(tmp, "outside.yaml");
    const error = thrown(() => confine(root, path));
    expect(error).toBeInstanceOf(ToolRefused);
    expect((error as Error).message).toContain(path);
  });

  test("refuses a symlink pointing out of the root", () => {
    symlinkSync(join(tmp, "outside.yaml"), join(root, "link.yaml"));
    symlinkSync(tmp, join(root, "up"));
    expect(thrown(() => confine(root, "link.yaml"))).toBeInstanceOf(ToolRefused);
    expect(thrown(() => confine(root, "up/new.yaml"))).toBeInstanceOf(ToolRefused);
  });

  test("refuses a dangling symlink", () => {
    symlinkSync(join(tmp, "missing.yaml"), join(root, "dangling.yaml"));
    expect(thrown(() => confine(root, "dangling.yaml"))).toBeInstanceOf(ToolRefused);
  });

  test("accepts a new file whose nearest existing folder is inside", () => {
    expect(confine(root, "sub/new/b.yaml")).toBe(join(root, "sub", "new", "b.yaml"));
  });
});
