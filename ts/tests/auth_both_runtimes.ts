/**
 * Authentication scenarios written once for both runtimes: the credential store across processes,
 * the browser callback listener, `http.ts`'s `request`, and the device flow over HTTP. They are
 * plain async functions using `node:assert/strict`, so the same checks run as Bun tests
 * (`provider_auth.test.ts`) and, built, on Node, where this module is the main one and prints one
 * JSON line per scenario. Run with `child` as its first argument, it is one of the processes a
 * scenario spawns.
 */

import assert from "node:assert/strict";
import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net, { type AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { ProviderError } from "../src/llm/errors.ts";
import { AuthError, HttpResponse, OAuthCredentials } from "../src/llm_clients/auth/common.ts";
import { request } from "../src/llm_clients/auth/http.ts";
import { type AuthRequest, OpenAIAuth } from "../src/llm_clients/auth/openai.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { realScheduler, USER_AGENT } from "../src/platform.ts";

/** A spawned child, with everything it printed so far and its exit. */
interface Child {
  process: ChildProcess;
  stdout: () => string;
  stderr: () => string;
  exited: Promise<number | null>;
}

/** Spawn this module as a child resolving `credentialId` from the store at `path`. */
function spawnChild(path: string, credentialId: string, counterFile: string): Child {
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), "child", path, credentialId, counterFile], {
    stdio: "pipe",
  });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8").on("data", (chunk: string) => {
    stdout += chunk;
  });
  child.stderr.setEncoding("utf8").on("data", (chunk: string) => {
    stderr += chunk;
  });
  // A child that dies after `ready` makes the `go` write fail with EPIPE; its exit code and
  // stderr already report why.
  child.stdin.on("error", () => {});
  const exited = new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code) => resolve(code));
  });
  return { process: child, stdout: () => stdout, stderr: () => stderr, exited };
}

/** Resolve once `child` has printed `ready`; reject if it exits first. */
function ready(child: Child): Promise<void> {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (child.stdout().split("\n").includes("ready")) resolve();
    };
    child.process.stdout!.on("data", check);
    check();
    child.exited.then(
      () => reject(new Error(`a child exited before it was ready; stderr:\n${child.stderr()}`)),
      reject,
    );
  });
}

/** Three processes refresh one expiring login at once: exactly one refresh runs. */
async function refreshIsSerializedAcrossProcesses(): Promise<void> {
  const children: Child[] = [];
  let tmp: string | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    tmp = fs.mkdtempSync(join(tmpdir(), "yian-auth-"));
    const path = join(tmp, "auth", "auth.json");
    const counterFile = join(tmp, "refreshes");
    fs.writeFileSync(counterFile, "");
    const expiring = new OAuthCredentials({ accessToken: "old", refreshToken: "refresh", expiresAt: 900, accountId: "acct" });
    const selection = await new AuthStore(path).save("openai", expiring, "model");
    for (let i = 0; i < 3; i++) children.push(spawnChild(path, selection.credentialId, counterFile));
    const limit = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("the children didn't finish within 20 s")), 20_000);
    });
    await Promise.race([Promise.all(children.map(ready)), limit]);
    for (const child of children) child.process.stdin!.end("go\n");
    const codes = await Promise.race([Promise.all(children.map((child) => child.exited)), limit]);
    for (const [i, child] of children.entries()) {
      assert.equal(codes[i], 0, `a child failed; stderr:\n${child.stderr()}`);
      assert.deepEqual(child.stdout().trim().split("\n"), ["ready", "new"]);
    }
    assert.equal(fs.readFileSync(counterFile, "utf8"), "refresh\n");
    assert.deepEqual(new AuthStore(path).selection(), selection);
  } finally {
    clearTimeout(timer);
    for (const child of children) child.process.kill("SIGKILL");
    // Reaped first, so no child still writes into the directory being removed.
    await Promise.allSettled(children.map((child) => child.exited));
    if (tmp !== undefined) fs.rmSync(tmp, { recursive: true, force: true });
  }
}

/** The child side: print `ready`, wait for `go`, resolve the login, print its access token. */
async function child(path: string, credentialId: string, counterFile: string): Promise<void> {
  console.log("ready");
  let input = "";
  for await (const chunk of process.stdin) {
    input += String(chunk);
    if (input.includes("go\n")) break;
  }
  const store = new AuthStore(path, { scheduler: { now: () => 1_000_000, sleep: realScheduler.sleep } });
  const credentials = await store.credentials("openai", credentialId, {
    refresh: async () => {
      fs.appendFileSync(counterFile, "refresh\n");
      await realScheduler.sleep(200);
      return new OAuthCredentials({ accessToken: "new", refreshToken: "rotated", expiresAt: 5000, accountId: "acct" });
    },
  });
  console.log((credentials as OAuthCredentials).accessToken);
}

/** Send `data` to the login callback listener and return everything it answers before closing. */
function callbackExchange(data: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const socket = net.connect({ host: "127.0.0.1", port: 1455 }, () => socket.write(data));
    const chunks: Buffer[] = [];
    socket.on("data", (chunk: Buffer) => chunks.push(chunk));
    socket.on("error", reject);
    socket.on("close", () => resolve(Buffer.concat(chunks).toString("latin1")));
  });
}

/** The listener's exact answer for `status` and `body`. */
function callbackAnswer(status: string, body: string): string {
  return (
    `HTTP/1.1 ${status}\r\nContent-Type: text/plain; charset=utf-8\r\n` +
    `Content-Length: ${Buffer.byteLength(body)}\r\nConnection: close\r\n\r\n${body}`
  );
}

/** Bind 127.0.0.1:1455 and release it, proving no listener still holds it. */
async function expectCallbackPortFree(): Promise<void> {
  const server = net.createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen({ host: "127.0.0.1", port: 1455 }, resolve);
  });
  await new Promise((resolve) => server.close(resolve));
}

/** An unsigned JWT whose payload names `account` in OpenAI's auth claim. */
function jwt(account: string): string {
  const claims = { "https://api.openai.com/auth": { chatgpt_account_id: account } };
  return `header.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.signature`;
}

/** A request a local server received. */
interface Received {
  method: string;
  path: string;
  headers: http.IncomingHttpHeaders;
  body: string;
}

/** A local HTTP server on 127.0.0.1. */
interface LocalServer {
  origin: string;
  received: Received[];
  close(): Promise<void>;
}

/** Start a server on 127.0.0.1 that records each request, then lets `handle` answer it. */
async function localServer(handle: (res: http.ServerResponse, received: Received) => void): Promise<LocalServer> {
  const received: Received[] = [];
  const server = http.createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk: string) => (body += chunk));
    req.on("end", () => {
      const each = { method: req.method!, path: req.url!, headers: req.headers, body };
      received.push(each);
      handle(res, each);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    origin: `http://127.0.0.1:${port}`,
    received,
    async close() {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

/** A browser login whose callback arrives over TCP returns credentials and frees the port. */
async function browserLoginThroughTheListener(): Promise<void> {
  const forms: Readonly<Record<string, string>>[] = [];
  const exchange: AuthRequest = async (_method, _url, options) => {
    forms.push(options.form!);
    const tokens = { access_token: jwt("account-test"), refresh_token: "refresh-secret", expires_in: 3600n };
    return new HttpResponse(200, tokens);
  };
  const replies: Promise<string>[] = [];
  const auth = new OpenAIAuth({ request: exchange, scheduler: realScheduler });
  const credentials = await auth.browserLogin(
    (url) => {
      const state = new URL(url).searchParams.get("state");
      const callback = `GET /auth/callback?code=code&state=${state} HTTP/1.1\r\nHost: localhost:1455\r\n\r\n`;
      const reply = callbackExchange(callback);
      // Awaited below only when the login succeeds; a failed one must not leave it unhandled.
      reply.catch(() => {});
      replies.push(reply);
    },
    { timeoutMs: 5000 },
  );
  assert.equal(await replies[0], callbackAnswer("200 OK", "Authorization received. Return to Yian."));
  assert.equal(credentials.accountId, "account-test");
  assert.equal(credentials.refreshToken, "refresh-secret");
  assert.equal(forms.length, 1);
  assert.equal(forms[0]!.code, "code");
  assert.equal(forms[0]!.redirect_uri, "http://localhost:1455/auth/callback");
  await expectCallbackPortFree();
}

/** The listener answers an unknown path, the login then times out, and a throwing `openUrl` propagates. */
async function listenerAnswersBadRequestsAndTimesOut(): Promise<void> {
  const unhandled: unknown[] = [];
  const guard = (reason: unknown) => void unhandled.push(reason);
  process.on("unhandledRejection", guard);
  try {
    const nothing: AuthRequest = async () => assert.fail("no request is expected");
    const auth = new OpenAIAuth({ request: nothing, scheduler: realScheduler });
    let reply: string | undefined;
    await assert.rejects(
      auth.browserLogin(
        async () => {
          reply = await callbackExchange("GET /elsewhere HTTP/1.1\r\nHost: localhost:1455\r\n\r\n");
        },
        { timeoutMs: 300 },
      ),
      (error: unknown) => error instanceof AuthError && error.message === "OpenAI browser login timed out",
    );
    assert.equal(reply, callbackAnswer("404 Not Found", "Unknown callback path."));
    await expectCallbackPortFree();

    const cancelled = new Error("cancelled by user");
    await assert.rejects(
      auth.browserLogin(() => {
        throw cancelled;
      }),
      (error: unknown) => error === cancelled,
    );
    // Long enough for a stray rejection to be reported.
    await realScheduler.sleep(50);
    await expectCallbackPortFree();
  } finally {
    process.off("unhandledRejection", guard);
  }
  assert.deepEqual(unhandled, []);
}

/** `request` decodes JSON, returns a 302 unfollowed, and refuses a body over its limit. */
async function requestDecodesRefusesRedirectsAndCapsBodies(): Promise<void> {
  const server = await localServer((res, received) => {
    if (received.path === "/json") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end('{"expires_in": 3600, "ratio": 1.5}');
    } else if (received.path === "/redirect") {
      res.writeHead(302, { location: "/json" });
      res.end('{"followed": false}');
    } else {
      res.writeHead(200);
      res.end(JSON.stringify({ pad: "x".repeat(2048) }));
    }
  });
  try {
    const decoded = await request("GET", server.origin + "/json", { headers: {}, timeoutMs: 2000 });
    assert.equal(Object.getPrototypeOf(decoded), HttpResponse.prototype);
    assert.equal(decoded.status, 200);
    assert.deepEqual(decoded.data, { expires_in: 3600n, ratio: 1.5 });

    const redirected = await request("GET", server.origin + "/redirect", { headers: {}, timeoutMs: 2000 });
    assert.deepEqual([redirected.status, redirected.data], [302, null]);

    await assert.rejects(
      request("GET", server.origin + "/big", { headers: {}, timeoutMs: 2000, bodyLimit: 1024 }),
      (error: unknown) =>
        error instanceof ProviderError &&
        error.kind === "protocol" &&
        error.message === "HTTP response is larger than 1024 bytes",
    );
    assert.deepEqual(
      server.received.map((each) => each.path),
      ["/json", "/redirect", "/big"],
    );
  } finally {
    await server.close();
  }
}

/** The device flow runs end to end through `request` against a local fake of OpenAI's endpoints. */
async function deviceFlowEndToEndOverHttp(): Promise<void> {
  let polls = 0;
  const server = await localServer((res, received) => {
    let status = 200;
    let body: unknown;
    if (received.path === "/api/accounts/deviceauth/usercode") {
      body = { device_auth_id: "device-id", user_code: "ABC-DEF", interval: 1 };
    } else if (received.path === "/api/accounts/deviceauth/token") {
      polls++;
      if (polls === 1) [status, body] = [403, { error: "pending" }];
      else body = { authorization_code: "auth-code", code_verifier: "server-verifier" };
    } else if (received.path === "/oauth/token") {
      body = { access_token: jwt("account-test"), refresh_token: "refresh-secret", expires_in: 3600 };
    } else {
      [status, body] = [404, {}];
    }
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  });
  try {
    const origins: string[] = [];
    // Every OpenAI URL goes to the local server instead.
    const local: AuthRequest = (method, url, options) => {
      const target = new URL(url);
      origins.push(target.origin);
      return request(method, server.origin + target.pathname + target.search, options);
    };
    const displayed: [string, string][] = [];
    const auth = new OpenAIAuth({ request: local, scheduler: realScheduler });
    const start = Date.now();
    const credentials = await auth.deviceLogin((url, code) => void displayed.push([url, code]), { timeoutMs: 10_000 });
    const elapsed = Date.now() - start;
    assert.deepEqual(displayed, [["https://auth.openai.com/codex/device", "ABC-DEF"]]);
    assert.ok(elapsed >= 900, `the poll interval was not waited: ${elapsed} ms`);
    assert.equal(credentials.accountId, "account-test");
    assert.equal(credentials.refreshToken, "refresh-secret");
    assert.ok(credentials.expiresAt >= start / 1000 + 3600 && credentials.expiresAt <= Date.now() / 1000 + 3600);
    assert.deepEqual(new Set(origins), new Set(["https://auth.openai.com"]));
    const sent = server.received;
    assert.deepEqual(
      sent.map((each) => [each.method, each.path]),
      [
        ["POST", "/api/accounts/deviceauth/usercode"],
        ["POST", "/api/accounts/deviceauth/token"],
        ["POST", "/api/accounts/deviceauth/token"],
        ["POST", "/oauth/token"],
      ],
    );
    for (const each of sent) assert.equal(each.headers["user-agent"], USER_AGENT);
    assert.equal(sent[0]!.headers["content-type"], "application/json");
    assert.deepEqual(JSON.parse(sent[0]!.body), { client_id: "app_EMoamEEZ73f0CkXaXp7hrann" });
    assert.deepEqual(JSON.parse(sent[1]!.body), { device_auth_id: "device-id", user_code: "ABC-DEF" });
    assert.equal(sent[3]!.headers["content-type"], "application/x-www-form-urlencoded");
    assert.deepEqual(Object.fromEntries(new URLSearchParams(sent[3]!.body)), {
      grant_type: "authorization_code",
      code: "auth-code",
      code_verifier: "server-verifier",
      client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
      redirect_uri: "https://auth.openai.com/deviceauth/callback",
    });
  } finally {
    await server.close();
  }
}

/** The scenarios by name, in the order `main` runs them. */
export const SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  test_refresh_is_serialized_across_processes: refreshIsSerializedAcrossProcesses,
  "browser login through the listener": browserLoginThroughTheListener,
  "listener answers bad requests and times out": listenerAnswersBadRequestsAndTimesOut,
  "request decodes, refuses redirects and caps bodies": requestDecodesRefusesRedirectsAndCapsBodies,
  "the device flow end to end over HTTP": deviceFlowEndToEndOverHttp,
};

/** Run every scenario, printing `{"name", "ok", "error"?}` per line; return how many failed. */
async function main(): Promise<number> {
  let failures = 0;
  for (const [name, run] of Object.entries(SCENARIOS)) {
    try {
      await run();
      console.log(JSON.stringify({ name, ok: true }));
    } catch (error) {
      failures++;
      const text = error instanceof Error ? (error.stack ?? error.message) : String(error);
      console.log(JSON.stringify({ name, ok: false, error: text }));
    }
  }
  return failures;
}

// `import.meta.main` doesn't exist on Node 22.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === "child") {
    const [path, credentialId, counterFile] = process.argv.slice(3) as [string, string, string];
    await child(path, credentialId, counterFile);
  } else {
    // No process.exit: a leaked timer or socket then shows as a hang.
    process.exitCode = (await main()) ? 1 : 0;
  }
}
