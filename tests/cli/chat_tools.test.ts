// The chat assistant's file tools, confined to the folder the chat started in.

import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { type ChatToolHost, chatFileTools, checkFocus, confine, ToolRefused } from "../../src/cli/chat_tools.ts";
import type { ApproveRequest } from "../../src/cli/terminal.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { isReportedToolError } from "../../src/tools/errors.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

const FLOW = "id: f\nname: f\ninput:\n  n: !int\nnodes: {}\noutput: ${input.n}\n";

/** A flow calling the flow `ref` through a `uses:` alias. */
function parentFlow(ref: string): string {
  return (
    "id: main\nname: main\ninput:\n  n: !int\nnodes:\n  work:\n    kind: call\n    call: child\n" +
    `    input:\n      n: \${input.n}\noutput: \${work.output}\nuses:\n  child: ${ref}\n`
  );
}

/** A flow taking `n` whose one node `a` runs the CODE reference `ref`. */
function codeFlowText(ref: string): string {
  return `id: c\nname: c\ninput:\n  n: !int\nnodes:\n  a: {kind: code, code: ${ref}, input: {n: "\${input.n}"}}\noutput: \${a.output}\n`;
}

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

/** A host over `root` whose approvals `approve` answers; no docs unless given. */
function host(fields: Partial<ChatToolHost> = {}): ChatToolHost {
  return {
    root,
    approve: async () => {
      throw new Error("not expected to ask");
    },
    env: NO_ENV,
    docs: new Map(),
    ...fields,
  };
}

/** Call the tool `name` of `chatFileTools(chatHost)` with `args`, returning its result. */
async function call(name: string, args: Record<string, unknown>, chatHost: ChatToolHost = host()): Promise<unknown> {
  const found = chatFileTools(chatHost).find((each) => each.name === name);
  if (found === undefined) throw new Error(`no tool ${name}`);
  const ctx = new ToolContext({
    callId: "run-1/chat",
    attempt: 1,
    toolName: name,
    runId: "run-1",
    node: "chat",
    context: undefined,
    signal: new AbortController().signal,
  });
  return found.execute(args, ctx);
}

/** What `promise` rejected with. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

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

  test("a symlink loop is an error the model reads", () => {
    symlinkSync(join(root, "loop.yaml"), join(root, "loop.yaml"));
    expect(isReportedToolError(thrown(() => confine(root, "loop.yaml")))).toBe(true);
  });

  test("refuses a path with a control character, naming it", () => {
    for (const path of ["a\x00.yaml", "a\x1b[2J.yaml", "a\x7f.yaml", "a\x9b.yaml", "a\n.yaml"]) {
      const error = thrown(() => confine(root, path));
      expect(error).toBeInstanceOf(ToolRefused);
      expect((error as Error).message).not.toMatch(/[\x00-\x1f\x7f-\x9f]/);
    }
    expect((thrown(() => confine(root, "a\x00.yaml")) as Error).message).toContain("'a\\x00.yaml'");
  });

  test("refuses a path through a symlink to a missing folder", () => {
    symlinkSync(join(tmp, "newdir"), join(root, "dl"));
    const error = thrown(() => confine(root, "dl/x.yaml"));
    expect(error).toBeInstanceOf(ToolRefused);
    expect((error as Error).message).toContain("goes through a symlink to a missing folder");
    expect(thrown(() => confine(root, "dl/deeper/x.yaml"))).toBeInstanceOf(ToolRefused);
  });

  test("accepts a new file whose nearest existing folder is inside", () => {
    expect(confine(root, "sub/new/b.yaml")).toBe(join(root, "sub", "new", "b.yaml"));
  });
});

describe("list_flows", () => {
  test("lists the YAML files under the root, sorted, skipping node_modules and dot-folders", async () => {
    mkdirSync(join(root, "node_modules", "x"), { recursive: true });
    mkdirSync(join(root, ".git"));
    mkdirSync(join(root, "B"));
    writeFileSync(join(root, "node_modules", "x", "f.yaml"), "");
    writeFileSync(join(root, ".git", "f.yaml"), "");
    writeFileSync(join(root, "B", "c.yml"), "");
    writeFileSync(join(root, "b.yaml"), "");
    writeFileSync(join(root, "notes.txt"), "");
    writeFileSync(join(root, ".env"), "");
    expect(await call("list_flows", {})).toBe(["B/c.yml", "b.yaml", "sub/a.yaml"].join("\n"));
  });

  test("lists one folder, paths still relative to the root", async () => {
    writeFileSync(join(root, "b.yaml"), "");
    expect(await call("list_flows", { dir: "sub" })).toBe("sub/a.yaml");
  });

  test("says when there are none", async () => {
    mkdirSync(join(root, "empty"));
    expect(await call("list_flows", { dir: "empty" })).toBe("no flows");
  });

  test("stops at the limit with a note", async () => {
    mkdirSync(join(root, "many"));
    for (let i = 0; i < 501; i++) writeFileSync(join(root, "many", `${String(i).padStart(3, "0")}.yaml`), "");
    const lines = String(await call("list_flows", { dir: "many" })).split("\n");
    expect(lines).toHaveLength(501);
    expect(lines[0]).toBe("many/000.yaml");
    expect(lines[499]).toBe("many/499.yaml");
    expect(lines[500]).toBe("(stopped at 500 flows; pass dir to list one folder)");
  });

  test("refuses a folder outside the root", async () => {
    expect(await rejection(call("list_flows", { dir: ".." }))).toBeInstanceOf(ToolRefused);
  });
});

describe("read_flow", () => {
  test("returns a flow's text", async () => {
    expect(await call("read_flow", { path: "sub/a.yaml" })).toBe("id: a\n");
  });

  test("reads only YAML files", async () => {
    writeFileSync(join(root, ".env"), "KEY=secret\n");
    symlinkSync(join(root, ".env"), join(root, "env.yaml"));
    for (const path of [".env", "env.yaml"]) {
      const error = await rejection(call("read_flow", { path }));
      expect(error).toBeInstanceOf(ToolRefused);
      expect(error.message).not.toContain("secret");
    }
  });

  test("refuses a file over the size cap", async () => {
    writeFileSync(join(root, "big.yaml"), "#".repeat(256 * 1024 + 1));
    const error = await rejection(call("read_flow", { path: "big.yaml" }));
    expect(error).toBeInstanceOf(ToolRefused);
    expect(error.message).toContain("262144");
  });

  test("a missing file is an error the model reads", async () => {
    const error = await rejection(call("read_flow", { path: "missing.yaml" }));
    expect(isReportedToolError(error)).toBe(true);
    expect(error.message).toContain("missing.yaml");
  });

  test("refuses a path outside the root", async () => {
    expect(await rejection(call("read_flow", { path: "../outside.yaml" }))).toBeInstanceOf(ToolRefused);
  });
});

describe("check_flow", () => {
  test("a flow that loads is ok, with its name and inputs", async () => {
    writeFileSync(join(root, "main.yaml"), parentFlow("sub/child"));
    writeFileSync(join(root, "sub", "child.yaml"), FLOW);
    expect(await call("check_flow", { path: "main.yaml" })).toBe("ok\nname: main\ninputs:\n  n: int");
  });

  test("a flow that doesn't load shows the diagnostic as yian check does, without colour", async () => {
    writeFileSync(join(root, "bad.yaml"), "id: [\n");
    const text = String(await call("check_flow", { path: "bad.yaml" }));
    expect(text).toStartWith("invalid:\n");
    expect(text).toContain("bad.yaml");
    expect(text).not.toContain("\x1b");
  });

  test("a uses: child outside the root is refused", async () => {
    writeFileSync(join(tmp, "child.yaml"), FLOW);
    writeFileSync(join(root, "main.yaml"), parentFlow("../child"));
    const text = String(await call("check_flow", { path: "main.yaml" }));
    expect(text).toStartWith("invalid:\n");
    expect(text).toContain(`${join("..", "child.yaml")} is outside the root`);
  });

  test("a CODE reference outside the root is refused", async () => {
    writeFileSync(join(tmp, "helpers.ts"), "export function fn(inputs) { return 1 }\n");
    writeFileSync(join(root, "code.yaml"), codeFlowText("../helpers.ts:fn"));
    const text = String(await call("check_flow", { path: "code.yaml" }));
    expect(text).toContain("`code:` reference '../helpers.ts:fn' is outside the code root");
  });

  test("checks only YAML files inside the root", async () => {
    writeFileSync(join(root, ".env"), "KEY: [secret\n");
    expect(await rejection(call("check_flow", { path: ".env" }))).toBeInstanceOf(ToolRefused);
    expect(await rejection(call("check_flow", { path: "../outside.yaml" }))).toBeInstanceOf(ToolRefused);
  });
});

describe("checkFocus", () => {
  test("checks a flow outside the chat's root against its own folder", async () => {
    mkdirSync(join(tmp, "elsewhere", "lib"), { recursive: true });
    const path = join(tmp, "elsewhere", "main.yaml");
    writeFileSync(path, parentFlow("lib/child"));
    writeFileSync(join(tmp, "elsewhere", "lib", "child.yaml"), FLOW);
    expect(await checkFocus(path, NO_ENV)).toEqual({
      text: parentFlow("lib/child"),
      check: "ok\nname: main\ninputs:\n  n: int",
    });
  });

  test("cuts a long flow's text, with a note, but checks all of it", async () => {
    const path = join(root, "long.yaml");
    writeFileSync(path, FLOW + `# ${"x".repeat(40 * 1024)}\n`);
    const focus = await checkFocus(path, NO_ENV);
    expect(focus.text).toStartWith(FLOW);
    expect(focus.text).toEndWith("\n(cut at 32768 bytes)");
    expect(Buffer.byteLength(focus.text)).toBeLessThan(33 * 1024);
    expect(focus.check).toBe("ok\nname: f\ninputs:\n  n: int");
  });

  test("a flow that doesn't load gives its diagnostic", async () => {
    const path = join(root, "bad.yaml");
    writeFileSync(path, "id: [\n");
    const focus = await checkFocus(path, NO_ENV);
    expect(focus.text).toBe("id: [\n");
    expect(focus.check).toStartWith("invalid:\n");
  });
});

/**
 * A host whose approvals `answer` gives, after running `during` (to change the files meanwhile);
 * `requests` collects each approval asked for.
 */
function approving(answer: string | null, during: () => void = () => {}) {
  const requests: ApproveRequest[] = [];
  const chatHost = host({
    approve: async (request) => {
      requests.push(request);
      during();
      return answer;
    },
  });
  return { chatHost, requests };
}

describe("write_flow", () => {
  test("content that doesn't load is invalid and never asks", async () => {
    const { chatHost, requests } = approving("Approve");
    const result = String(await call("write_flow", { path: "a.yaml", content: "id: [\n" }, chatHost));
    expect(result).toStartWith("invalid:\n");
    expect(requests).toEqual([]);
    expect(existsSync(join(root, "a.yaml"))).toBe(false);
  });

  test("an approved new file is written, its folders created", async () => {
    const { chatHost, requests } = approving("Approve");
    expect(await call("write_flow", { path: "a.yaml", content: FLOW }, chatHost)).toBe("wrote a.yaml");
    expect(readFileSync(join(root, "a.yaml"), "utf8")).toBe(FLOW);
    expect(requests).toHaveLength(1);
    expect(requests[0]!.title).toBe("Write a.yaml");
    expect(requests[0]!.choices).toEqual(["Approve", "Reject"]);
    expect(requests[0]!.diff).toStartWith("--- /dev/null\n+++ a.yaml\n@@ -0,0 +1,6 @@\n+id: f\n");
    const deep = await call("write_flow", { path: "new/deep/b.yaml", content: FLOW }, chatHost);
    expect(deep).toBe("wrote new/deep/b.yaml");
    expect(readFileSync(join(root, "new", "deep", "b.yaml"), "utf8")).toBe(FLOW);
  });

  test("a rejected write is not written", async () => {
    const { chatHost } = approving("Reject");
    expect(await call("write_flow", { path: "a.yaml", content: FLOW }, chatHost)).toBe("not written: rejected");
    expect(existsSync(join(root, "a.yaml"))).toBe(false);
  });

  test("an overwrite shows the diff and replaces the file, leaving no temporary file", async () => {
    writeFileSync(join(root, "sub", "f.yaml"), FLOW.replace("name: f", "name: old"));
    const { chatHost, requests } = approving("Approve");
    const content = FLOW.replace("name: f", "name: new");
    expect(await call("write_flow", { path: "sub/f.yaml", content }, chatHost)).toBe("wrote sub/f.yaml");
    expect(requests[0]!.diff).toContain("\n-name: old\n+name: new\n");
    expect(readFileSync(join(root, "sub", "f.yaml"), "utf8")).toBe(content);
    expect(readdirSync(join(root, "sub")).toSorted()).toEqual(["a.yaml", "f.yaml"]);
  });

  test("a file changed during approval is not written", async () => {
    const path = join(root, "sub", "f.yaml");
    writeFileSync(path, FLOW);
    const { chatHost } = approving("Approve", () => writeFileSync(path, FLOW + "# edited\n"));
    const content = FLOW.replace("name: f", "name: new");
    const result = await call("write_flow", { path: "sub/f.yaml", content }, chatHost);
    expect(result).toBe("not written: changed during approval");
    expect(readFileSync(path, "utf8")).toBe(FLOW + "# edited\n");
  });

  test("a file created during approval is not overwritten", async () => {
    const path = join(root, "a.yaml");
    const { chatHost } = approving("Approve", () => writeFileSync(path, "theirs\n"));
    const result = await call("write_flow", { path: "a.yaml", content: FLOW }, chatHost);
    expect(result).toBe("not written: changed during approval");
    expect(readFileSync(path, "utf8")).toBe("theirs\n");
  });

  test("the same bytes under a new file are a change", async () => {
    const path = join(root, "sub", "f.yaml");
    writeFileSync(path, FLOW);
    // Written beside it and renamed over it, as an editor saves, so the old inode isn't reused.
    const { chatHost } = approving("Approve", () => {
      writeFileSync(`${path}.new`, FLOW);
      renameSync(`${path}.new`, path);
    });
    const result = await call("write_flow", { path: "sub/f.yaml", content: FLOW + "# new\n" }, chatHost);
    expect(result).toBe("not written: changed during approval");
    expect(readFileSync(path, "utf8")).toBe(FLOW);
  });

  test("a folder swapped for a symlink out during approval is refused", async () => {
    writeFileSync(join(root, "sub", "f.yaml"), FLOW);
    mkdirSync(join(tmp, "out"));
    writeFileSync(join(tmp, "out", "f.yaml"), FLOW);
    const swap = (): void => {
      renameSync(join(root, "sub"), join(root, "moved"));
      symlinkSync(join(tmp, "out"), join(root, "sub"));
    };
    for (const path of ["sub/f.yaml", "sub/new.yaml"]) {
      if (existsSync(join(root, "moved"))) {
        rmSync(join(root, "sub"));
        renameSync(join(root, "moved"), join(root, "sub"));
      }
      const { chatHost } = approving("Approve", swap);
      expect(await rejection(call("write_flow", { path, content: FLOW + "# new\n" }, chatHost))).toBeInstanceOf(
        ToolRefused,
      );
    }
    expect(readdirSync(join(tmp, "out"))).toEqual(["f.yaml"]);
    expect(readFileSync(join(tmp, "out", "f.yaml"), "utf8")).toBe(FLOW);
  });

  test("a file swapped for a symlink out during approval is refused", async () => {
    const path = join(root, "sub", "f.yaml");
    writeFileSync(path, FLOW);
    writeFileSync(join(tmp, "out.yaml"), FLOW);
    const { chatHost } = approving("Approve", () => {
      rmSync(path);
      symlinkSync(join(tmp, "out.yaml"), path);
    });
    const content = FLOW + "# new\n";
    expect(await rejection(call("write_flow", { path: "sub/f.yaml", content }, chatHost))).toBeInstanceOf(ToolRefused);
    expect(readFileSync(join(tmp, "out.yaml"), "utf8")).toBe(FLOW);
  });

  // Root reads and writes a read-only folder anyway.
  test.skipIf(process.getuid?.() === 0)("a folder that can't be written is reported, not thrown", async () => {
    const folder = join(root, "ro");
    mkdirSync(folder);
    writeFileSync(join(folder, "a.yaml"), FLOW);
    chmodSync(folder, 0o555);
    try {
      const { chatHost } = approving("Approve");
      const overwrite = await call("write_flow", { path: "ro/a.yaml", content: FLOW + "# x\n" }, chatHost);
      expect(overwrite).toBe("not written: Permission denied");
      expect(await call("write_flow", { path: "ro/b.yaml", content: FLOW }, chatHost)).toBe(
        "not written: Permission denied",
      );
      expect(await call("write_flow", { path: "ro/deep/c.yaml", content: FLOW }, chatHost)).toBe(
        "not written: Permission denied",
      );
      expect(readdirSync(folder)).toEqual(["a.yaml"]);
    } finally {
      chmodSync(folder, 0o755);
    }
  });

  test("a folder at the temporary file's name is reported and left alone", async () => {
    writeFileSync(join(root, "t.yaml"), FLOW);
    const temporary = join(root, `.t.yaml.yian-tmp-${process.pid}`);
    mkdirSync(temporary);
    const { chatHost } = approving("Approve");
    const result = String(await call("write_flow", { path: "t.yaml", content: FLOW + "# y\n" }, chatHost));
    expect(result).toStartWith("not written: ");
    expect(readFileSync(join(root, "t.yaml"), "utf8")).toBe(FLOW);
    expect(existsSync(temporary)).toBe(true);
  });

  test("a leftover temporary file is replaced and removed", async () => {
    writeFileSync(join(root, "t.yaml"), FLOW);
    writeFileSync(join(root, `.t.yaml.yian-tmp-${process.pid}`), "stale\n");
    const { chatHost } = approving("Approve");
    expect(await call("write_flow", { path: "t.yaml", content: FLOW + "# y\n" }, chatHost)).toBe("wrote t.yaml");
    expect(readFileSync(join(root, "t.yaml"), "utf8")).toBe(FLOW + "# y\n");
    expect(readdirSync(root).toSorted()).toEqual(["sub", "t.yaml"]);
  });

  test("without a terminal to approve, nothing is written", async () => {
    const { chatHost } = approving(null);
    const result = await call("write_flow", { path: "a.yaml", content: FLOW }, chatHost);
    expect(result).toBe("not written: needs a terminal to approve");
    expect(existsSync(join(root, "a.yaml"))).toBe(false);
  });

  test("refuses a path outside the root or not a YAML file, without asking", async () => {
    const { chatHost, requests } = approving("Approve");
    expect(await rejection(call("write_flow", { path: "../x.yaml", content: FLOW }, chatHost))).toBeInstanceOf(
      ToolRefused,
    );
    expect(await rejection(call("write_flow", { path: ".env", content: FLOW }, chatHost))).toBeInstanceOf(ToolRefused);
    expect(requests).toEqual([]);
    expect(existsSync(join(tmp, "x.yaml"))).toBe(false);
  });
});

describe("read_docs", () => {
  const docs = new Map([
    ["writing-flows", "# Writing flows\n"],
    ["agents", "# Agents\n"],
  ]);

  test("without a page, lists the page names, sorted", async () => {
    expect(await call("read_docs", {}, host({ docs }))).toBe("agents\nwriting-flows");
  });

  test("returns a page's text", async () => {
    expect(await call("read_docs", { page: "agents" }, host({ docs }))).toBe("# Agents\n");
  });

  test("an unknown page is an error naming the known ones", async () => {
    const error = await rejection(call("read_docs", { page: "nope" }, host({ docs })));
    expect(isReportedToolError(error)).toBe(true);
    expect(error.message).toBe("no docs page 'nope'; the pages are: agents, writing-flows");
  });
});
