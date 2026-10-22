/**
 * A test-only module that runs the `bun:test` files on Node (>= 22.18): `node:test` runs the
 * tests and Jest's `expect`, whose semantics Bun's follows, checks them.
 *
 * Load it before the preload: `node --import ./tests/node_runner.ts --import ./tests/conftest.ts
 * --test ...`. It registers a resolve hook mapping `bun:test` to this module and exports what the
 * suite imports from `bun:test`. Anything else is missing, so a test importing it fails to load.
 *
 * Differences from Bun a test may meet: Node runs a `describe` body when it is defined, not after
 * the module has loaded, so a body can't read a `const` declared below it; a skipped `describe`
 * counts as one skipped suite, not as its tests; `jest.useFakeTimers()` fakes `setInterval` only.
 */

import { registerHooks } from "node:module";
import * as nodeTest from "node:test";
import { inspect } from "node:util";

import { expect as jestExpect, type MatcherFunction } from "expect";
import { ModuleMocker } from "jest-mock";

// Synchronous, so it already applies to the static imports of the next `--import` and of each
// test file.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "bun:test") return { url: import.meta.url, shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

type Body = (...args: never[]) => unknown;
type Timeout = number | { timeout?: number };
type Define = (name: string, body: () => unknown, settings: nodeTest.TestOptions) => void;

/** `node:test` options for Bun's timeout argument: milliseconds or `{ timeout }`. */
function options(timeout?: Timeout): nodeTest.TestOptions {
  const ms = typeof timeout === "number" ? timeout : timeout?.timeout;
  return ms === undefined ? {} : { timeout: ms };
}

/**
 * Bun's `.each` title: `%s`, `%d`, `%i`, `%f`, `%p` (a string quoted), `%o` and `%j` (JSON), `%#` (the
 * row's index), `%%` and, for an object row, `$key`. Other values in `%p` and `$key` are shown with
 * `inspect`, not Bun's pretty-format.
 */
function eachTitle(name: string, row: unknown, index: number): string {
  const show = (value: unknown) => (typeof value === "string" ? value : inspect(value));
  const args: unknown[] = Array.isArray(row) ? row : [row];
  let next = 0;
  let title = name.replace(/%[sdifjop#%]/g, (directive) => {
    switch (directive) {
      case "%%":
        return "%";
      case "%#":
        return String(index);
      case "%p": {
        const value = args[next++];
        return typeof value === "string" ? JSON.stringify(value) : inspect(value);
      }
      case "%o":
      case "%j":
        // As on Bun: `undefined` gives an empty string and a bigint throws.
        return JSON.stringify(args[next++]) ?? "";
      default:
        return show(args[next++]);
    }
  });
  if (row !== null && typeof row === "object" && !Array.isArray(row)) {
    title = title.replace(/\$([A-Za-z_][\w.]*)/g, (_, path: string) =>
      show(path.split(".").reduce<unknown>((value, key) => (value as Record<string, unknown> | undefined)?.[key], row)),
    );
  }
  return title;
}

/** Bun's chainable `test`/`describe` (`.each`, `.skip`, `.skipIf`, `.if`, `.todo`) over `define`. */
function chainable(define: Define) {
  const skip = (name: string, _body?: Body) => define(name, () => {}, { skip: true });
  const run = (name: string, body: Body, timeout?: Timeout) => define(name, body as () => unknown, options(timeout));
  // A new function: `skipIf` and `if` return `run`, which has no `.each` or `.skip`.
  const api = Object.assign((name: string, body: Body, timeout?: Timeout) => run(name, body, timeout), {
    each:
      (rows: readonly unknown[]) =>
      (name: string, body: Body, timeout?: Timeout) => {
        rows.forEach((row, index) => {
          const args = (Array.isArray(row) ? row : [row]) as never[];
          define(eachTitle(name, row, index), () => body(...args), options(timeout));
        });
      },
    skip,
    todo: (name: string, _body?: Body) => define(name, () => {}, { todo: true }),
    skipIf: (condition: unknown) => (condition ? skip : run),
    if: (condition: unknown) => (condition ? run : skip),
  });
  return api;
}

// Test bodies and hooks are called with no arguments: Node passes a test context to a function
// that declares a parameter, and with two it also passes a `done` callback and waits for it.
export const test = chainable((name, body, settings) => {
  nodeTest.test(name, settings, async () => {
    await body();
  });
});
export const it = test;
export const describe = chainable(
  (name, body, settings) => void nodeTest.describe(name, settings, body as nodeTest.SuiteFn),
);

export const beforeEach = (hook: () => unknown) => nodeTest.beforeEach(async () => void (await hook()));
export const afterEach = (hook: () => unknown) => nodeTest.afterEach(async () => void (await hook()));
export const beforeAll = (hook: () => unknown) => nodeTest.before(async () => void (await hook()));
export const afterAll = (hook: () => unknown) => nodeTest.after(async () => void (await hook()));

/** One of Bun's matchers that Jest lacks: passes when `check` holds for the received value. */
function bunMatcher(what: string, check: (received: unknown, expected: string) => boolean): MatcherFunction<[string?]> {
  return function (received, expected = "") {
    const pass = check(received, expected);
    const shown = expected === "" ? "" : ` ${this.utils.printExpected(expected)}`;
    const message = () => `expected ${this.utils.printReceived(received)} ${pass ? "not " : ""}${what}${shown}`;
    return { pass, message };
  };
}

jestExpect.extend({
  toStartWith: bunMatcher("to start with", (value, prefix) => typeof value === "string" && value.startsWith(prefix)),
  toEndWith: bunMatcher("to end with", (value, suffix) => typeof value === "string" && value.endsWith(suffix)),
  toBeNumber: bunMatcher("to be a number", (value) => typeof value === "number"),
  toBeString: bunMatcher("to be a string", (value) => typeof value === "string"),
});

/** `matchers` whose failures, thrown or rejected, start with `message` (Bun shows only `message`). */
function withMessage<T extends object>(matchers: T, message: string): T {
  const prefixed = (error: unknown): never => {
    if (error instanceof Error) {
      // Read first: V8 builds `stack` from `message` when it is first read.
      const { message: original, stack } = error;
      error.message = `${message}\n\n${original}`;
      error.stack = stack?.replace(original, () => error.message);
    }
    throw error;
  };
  return new Proxy(matchers, {
    get(target, key) {
      const value: unknown = Reflect.get(target, key);
      // `not`, `resolves` and `rejects` are matcher objects too.
      if (typeof value === "object" && value !== null) return withMessage(value, message);
      if (typeof value !== "function") return value;
      return (...args: unknown[]) => {
        try {
          const result: unknown = value.apply(target, args);
          return result instanceof Promise ? result.catch(prefixed) : result;
        } catch (error) {
          return prefixed(error);
        }
      };
    },
  });
}

// Bun's `expect` takes an optional failure message as a second argument, which Jest's rejects, so
// the wrapper adds it.
export const expect = Object.assign(
  (actual: unknown, message?: string) =>
    message === undefined ? jestExpect(actual) : withMessage(jestExpect(actual), message),
  jestExpect,
);

const mocker = new ModuleMocker(globalThis);
export const spyOn = mocker.spyOn.bind(mocker);
export const mock = Object.assign(mocker.fn.bind(mocker), { restore: () => mocker.restoreAllMocks() });

// Only `setInterval` is faked: that is what the suite's fake-timer tests drive, and Node throws
// when timers are enabled twice, hence the reset first.
export const jest = {
  useFakeTimers() {
    nodeTest.mock.timers.reset();
    nodeTest.mock.timers.enable({ apis: ["setInterval"] });
  },
  advanceTimersByTime(ms: number) {
    nodeTest.mock.timers.tick(ms);
  },
  useRealTimers() {
    nodeTest.mock.timers.reset();
  },
};
