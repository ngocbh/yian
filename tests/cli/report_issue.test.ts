/**
 * The host side of `yian chat`'s issue reports: the `file_issue` tool's approval card and each way
 * it files (gh faked by an injected runner, the GitHub API by an injected send), the prefilled
 * link, the saved copy, and the facts the chat adds to a report.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  type FileIssueHost,
  fileIssueTool,
  hostFacts,
  issueLink,
  LINK_CHARS,
  loadReportFlow,
  type Runner,
} from "../../src/cli/report_issue.ts";
import type { ApproveRequest } from "../../src/cli/terminal.ts";
import type { Send } from "../../src/platform.ts";
import { ToolContext } from "../../src/tools/context.ts";

let home: string;

beforeEach(() => {
  home = realpathSync(mkdtempSync(join(tmpdir(), "yian-report-issue-")));
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** 2026-10-29T14:03:05.250Z, the time the reports are saved at. */
const NOW = Date.UTC(2026, 9, 29, 14, 3, 5, 250);

const REPORTS = () => join(home, ".yian", "reports");

/** A runner where gh is installed and logged in, recording each call; `create` answers the create. */
function ghRunner(create: { code: number; stdout?: string; stderr?: string } = { code: 0 }) {
  const calls: { command: string; args: readonly string[]; input?: string | undefined }[] = [];
  const run: Runner = async (command, args, options = {}) => {
    calls.push({ command, args, input: options.input });
    if (args[0] === "auth") return { code: 0, stdout: "", stderr: "Logged in to github.com" };
    return { code: create.code, stdout: create.stdout ?? "", stderr: create.stderr ?? "" };
  };
  return { calls, run };
}

/** A runner where gh isn't installed: each start fails as a missing command does. */
const noGh: Runner = async () => {
  throw Object.assign(new Error("spawn gh ENOENT"), { code: "ENOENT" });
};

/** A host that answers the card with `choice` (null: no terminal), keeping each request. */
function host(choice: string | null, fields: Partial<FileIssueHost> = {}) {
  const asked: ApproveRequest[] = [];
  const fileHost: FileIssueHost = {
    approve: async (request) => {
      asked.push(request);
      return choice;
    },
    env: (name) => (name === "HOME" ? home : undefined),
    now: () => NOW,
    run: noGh,
    send: async () => {
      throw new Error("not expected to send");
    },
    ...fields,
  };
  return { asked, fileHost };
}

/** Call `file_issue` on `fileHost` with `title` and `body`. */
async function fileIssue(fileHost: FileIssueHost, title: string, body: string): Promise<string> {
  const ctx = new ToolContext({
    callId: "chat-1-r1/file",
    attempt: 1,
    toolName: "file_issue",
    runId: "chat-1-r1",
    node: "file",
    context: undefined,
    signal: new AbortController().signal,
  });
  return (await fileIssueTool(fileHost).execute({ title, body }, ctx)) as string;
}

/** The reports saved so far, by file name. */
const saved = () => (existsSync(REPORTS()) ? readdirSync(REPORTS()) : []);

const TITLE = "yian check crashes on an empty flow";
const BODY = "What I ran: yian check empty.yaml\nWhat happened: a stack trace & no diagnostic #1";

test("the card shows the whole title and body and offers only the ways that work", async () => {
  const { asked, fileHost } = host("Cancel");
  expect(await fileIssue(fileHost, TITLE, BODY)).toBe("not filed: cancelled; nothing was saved");
  expect(asked).toHaveLength(1);
  const card = asked[0]!;
  expect(card.title).toBe("Report this issue to github.com/ngocbh/yian?");
  expect(card.body).toContain(`Title: ${TITLE}\n\n${BODY}`);
  expect(card.diff).toBeNull();
  // No gh and no token: the link, the save and the cancel.
  expect(card.choices).toEqual(["Open a prefilled link in the browser", "Save it only", "Cancel"]);
  expect(saved()).toEqual([]);
  const { asked: both, fileHost: ready } = host("Cancel", {
    run: ghRunner().run,
    env: (name) => ({ HOME: home, GITHUB_TOKEN: "ghp_secret" })[name],
  });
  await fileIssue(ready, TITLE, BODY);
  expect(both[0]!.choices).toEqual([
    "File it with gh",
    "File it with GITHUB_TOKEN",
    "Open a prefilled link in the browser",
    "Save it only",
    "Cancel",
  ]);
  expect(JSON.stringify(both[0])).not.toContain("ghp_secret");
});

test("gh files the issue with the body on its stdin, and the report is saved too", async () => {
  const gh = ghRunner({ code: 0, stdout: "https://github.com/ngocbh/yian/issues/7\n" });
  const { fileHost } = host("File it with gh", { run: gh.run });
  const result = await fileIssue(fileHost, TITLE, BODY);
  expect(result).toBe(
    "filed with gh: https://github.com/ngocbh/yian/issues/7\n" +
      "saved to ~/.yian/reports/2026-10-29T14-03-05Z.md",
  );
  expect(gh.calls.at(-1)).toEqual({
    command: "gh",
    args: ["issue", "create", "-R", "ngocbh/yian", "--title", TITLE, "--body-file", "-"],
    input: BODY,
  });
  const file = join(REPORTS(), "2026-10-29T14-03-05Z.md");
  expect(readFileSync(file, "utf8")).toBe(`# ${TITLE}\n\n${BODY}\n`);
  expect(statSync(file).mode & 0o777).toBe(0o600);
});

test("a gh that fails says why, and the report stays saved", async () => {
  const gh = ghRunner({ code: 1, stderr: "GraphQL: Could not resolve to a Repository\n" });
  const { fileHost } = host("File it with gh", { run: gh.run });
  expect(await fileIssue(fileHost, TITLE, BODY)).toBe(
    "not filed: gh failed: GraphQL: Could not resolve to a Repository\n" +
      "saved to ~/.yian/reports/2026-10-29T14-03-05Z.md",
  );
  expect(saved()).toEqual(["2026-10-29T14-03-05Z.md"]);
});

test("GITHUB_TOKEN files the issue with one API call, and never shows the token", async () => {
  const sent: { url: string; init: RequestInit }[] = [];
  const send: Send = async (url, init) => {
    sent.push({ url: String(url), init });
    return new Response(JSON.stringify({ html_url: "https://github.com/ngocbh/yian/issues/8" }), {
      status: 201,
    });
  };
  const { fileHost } = host("File it with GITHUB_TOKEN", {
    send,
    env: (name) => ({ HOME: home, GITHUB_TOKEN: "ghp_secret" })[name],
  });
  const result = await fileIssue(fileHost, TITLE, BODY);
  expect(result).toBe(
    "filed with GITHUB_TOKEN: https://github.com/ngocbh/yian/issues/8\n" +
      "saved to ~/.yian/reports/2026-10-29T14-03-05Z.md",
  );
  expect(sent).toHaveLength(1);
  expect(sent[0]!.url).toBe("https://api.github.com/repos/ngocbh/yian/issues");
  expect(sent[0]!.init.method).toBe("POST");
  const headers = new Headers(sent[0]!.init.headers);
  expect(headers.get("authorization")).toBe("Bearer ghp_secret");
  expect(JSON.parse(String(sent[0]!.init.body))).toEqual({ title: TITLE, body: BODY });
  expect(result).not.toContain("ghp_secret");
});

test("a refused GITHUB_TOKEN says what GitHub answered, without the token", async () => {
  const send: Send = async () => new Response(JSON.stringify({ message: "Bad credentials" }), { status: 401 });
  const { fileHost } = host("File it with GITHUB_TOKEN", {
    send,
    env: (name) => ({ HOME: home, GITHUB_TOKEN: "ghp_secret" })[name],
  });
  const result = await fileIssue(fileHost, TITLE, BODY);
  expect(result).toStartWith("not filed: GitHub answered 401: Bad credentials\nsaved to ");
  expect(result).not.toContain("ghp_secret");
});

test("the browser link is prefilled and saved beside it", async () => {
  const { fileHost } = host("Open a prefilled link in the browser");
  const result = await fileIssue(fileHost, TITLE, BODY);
  const link = issueLink(TITLE, BODY, "~/.yian/reports/2026-10-29T14-03-05Z.md").url;
  expect(result).toBe(
    `not filed yet: open this link to file it: ${link}\n` + "saved to ~/.yian/reports/2026-10-29T14-03-05Z.md",
  );
  expect(saved()).toEqual(["2026-10-29T14-03-05Z.md"]);
});

test("save only saves the report, and a second report the same second gets its own file", async () => {
  const { fileHost } = host("Save it only");
  expect(await fileIssue(fileHost, TITLE, BODY)).toBe(
    "not filed: saved only\nsaved to ~/.yian/reports/2026-10-29T14-03-05Z.md",
  );
  expect(await fileIssue(fileHost, TITLE, BODY)).toBe(
    "not filed: saved only\nsaved to ~/.yian/reports/2026-10-29T14-03-05Z-2.md",
  );
  expect(statSync(REPORTS()).mode & 0o777).toBe(0o700);
});

test("without a terminal to approve on, nothing is filed or saved", async () => {
  const { fileHost } = host(null);
  expect(await fileIssue(fileHost, TITLE, BODY)).toBe(
    "not filed: not approved (no terminal, or the prompt ended); nothing was saved",
  );
  expect(saved()).toEqual([]);
});

test("the link encodes the title and body so they read back unchanged", () => {
  const title = "a & b = c? #1 100%";
  const body = "line one\nline two: ü 😀 + /path?x=1&y=2";
  const { url, cut } = issueLink(title, body, "~/r.md");
  expect(cut).toBe(false);
  expect(url).toStartWith("https://github.com/ngocbh/yian/issues/new?title=");
  const params = new URL(url).searchParams;
  expect(params.get("title")).toBe(title);
  expect(params.get("body")).toBe(body);
});

test("an over-long body is cut to fit the link, which says so and where the rest is", () => {
  const body = "😀".repeat(LINK_CHARS);
  const { url, cut } = issueLink("long", body, "~/.yian/reports/x.md");
  expect(cut).toBe(true);
  expect(url.length).toBeLessThanOrEqual(LINK_CHARS);
  const shown = new URL(url).searchParams.get("body")!;
  expect(shown).toEndWith("\n\n(cut to fit a link; the whole report is in ~/.yian/reports/x.md)");
  // The cut falls between characters.
  expect(shown.split("\n")[0]).toMatch(/^(😀)+$/);
});

test("the host's facts name the version, runtime, platform and the chat's log under ~", () => {
  const facts = hostFacts(join(home, ".yian", "runs", "c1", "log.jsonl"), home);
  expect(facts).toMatch(/^yian version: \d+\.\d+\.\d+\nruntime: (Bun|Node) \S+\nplatform: \S+ \S+\n/);
  expect(facts).toEndWith("chat run log: ~/.yian/runs/c1/log.jsonl");
});

test("the bundled report flow loads, with problem, context and home inputs", async () => {
  const flow = await loadReportFlow((name) => (name === "HOME" ? home : undefined));
  expect(flow.compiled.flowId).toBe("report_issue");
  expect(flow.input.map((param) => [param.name, param.required && !param.hasDefault])).toEqual([
    ["problem", true],
    ["context", true],
    ["home", false],
  ]);
});
