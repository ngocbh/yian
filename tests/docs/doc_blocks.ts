/**
 * Read the fenced code blocks of the docs pages, and check `yian` command lines against the CLI's
 * command table. `docs/STYLE.md` says how writers mark blocks; `examples.test.ts` runs the checks.
 */

import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";

import { COMMANDS, GROUPS } from "../../src/cli/index.ts";
import { type CommandSpec, type ParsedArgs, parseCommand } from "../../src/cli/utils.ts";

/** How a YAML block is checked; `loads` when the block names none. */
export type Check = "loads" | "runnable" | "fails" | "fragment";

const CHECKS: readonly Check[] = ["loads", "runnable", "fails", "fragment"];

/** One fenced code block of a page. */
export interface Block {
  /** The page, relative to the docs content root. */
  file: string;
  /** The 1-based line of the opening fence. */
  line: number;
  /** The fence's language, such as `yaml` or `console`. */
  lang: string;
  /** The fence's `key=value` options, such as `title` and `check`. */
  meta: Record<string, string>;
  /** The block's text, without the fence's indent. */
  text: string;
  /** The JSON object of a `<!-- yian-test {...} -->` comment just before the block, else null. */
  note: Record<string, unknown> | null;
}

/** One `$ ...` line of a console block and the output printed under it. */
export interface CommandLine {
  line: number;
  argv: string[];
  output: string;
}

const OPEN = /^( *)(`{3,}|~{3,})\s*([^\s`~]*)\s*(.*)$/;
const NOTE = /^\s*<!--\s*yian-test\b(.*?)(-->)?\s*$/;
const META = /([A-Za-z][\w-]*)=(?:"([^"]*)"|'([^']*)'|(\S+))/g;

/** Return the `key=value` options of a fence's info string after the language. */
export function readMeta(info: string): Record<string, string> {
  const meta: Record<string, string> = {};
  for (const m of info.matchAll(META)) meta[m[1]!] = m[2] ?? m[3] ?? m[4]!;
  return meta;
}

/** Return the JSON object of a one-line `<!-- yian-test {...} -->` note; throw for any other shape. */
function readNote(body: string, close: string | undefined, where: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = close === undefined ? undefined : JSON.parse(body);
  } catch {
    value = undefined;
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${where}: a yian-test note is one line holding a JSON object`);
  }
  return value as Record<string, unknown>;
}

/** Return the fenced code blocks of `text`, a page whose path is `file`. */
export function readBlocks(text: string, file: string): Block[] {
  const lines = text.split("\n");
  const blocks: Block[] = [];
  let note: Record<string, unknown> | null = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const noted = NOTE.exec(line);
    if (noted) {
      note = readNote(noted[1]!, noted[2], `${file}:${i + 1}`);
      continue;
    }
    const open = OPEN.exec(line);
    if (!open) {
      // A note belongs to the block right after it, blank lines aside.
      if (line.trim() !== "") note = null;
      continue;
    }
    const [, indent, fence, lang, info] = open as unknown as [string, string, string, string, string];
    const body: string[] = [];
    let end = i + 1;
    for (; end < lines.length; end++) {
      const close = lines[end]!.trim();
      if (close.startsWith(fence) && close === fence[0]!.repeat(close.length)) break;
      body.push(lines[end]!.startsWith(indent) ? lines[end]!.slice(indent.length) : lines[end]!.trimStart());
    }
    if (end === lines.length) throw new Error(`${file}:${i + 1}: unclosed code fence`);
    blocks.push({ file, line: i + 1, lang, meta: readMeta(info), text: `${body.join("\n")}\n`, note });
    note = null;
    i = end;
  }
  return blocks;
}

/** Return every block of the `.md` and `.mdx` pages under `root`, in path order. */
export function readDocs(root: string): Block[] {
  const files = readdirSync(root, { recursive: true, encoding: "utf8" })
    .filter((path) => /\.mdx?$/.test(path))
    .sort();
  return files.flatMap((path) => readBlocks(readFileSync(join(root, path), "utf8"), relative(root, join(root, path))));
}

/** Return a YAML block's check; throw for one the style guide doesn't name. */
export function checkOf(block: Block): Check {
  const check = block.meta["check"] ?? "loads";
  if (!(CHECKS as readonly string[]).includes(check)) {
    throw new Error(`${block.file}:${block.line}: check=${check} is not one of ${CHECKS.join(", ")}`);
  }
  return check as Check;
}

/** Split a shell line into words, as a POSIX shell does for quotes and backslashes. */
export function shellWords(line: string): string[] {
  const words: string[] = [];
  let word = "";
  let inWord = false;
  let quote: string | null = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i]!;
    if (quote !== null) {
      if (c === quote) quote = null;
      else if (c === "\\" && quote === '"' && i + 1 < line.length) word += line[++i];
      else word += c;
    } else if (c === "'" || c === '"') {
      quote = c;
      inWord = true;
    } else if (c === "\\" && i + 1 < line.length) {
      word += line[++i];
      inWord = true;
    } else if (/\s/.test(c)) {
      if (inWord) words.push(word);
      word = "";
      inWord = false;
    } else {
      word += c;
      inWord = true;
    }
  }
  if (quote !== null) throw new Error(`unclosed quote in: ${line}`);
  if (inWord) words.push(word);
  return words;
}

/** Return the `$ ` lines of a console block, a trailing `\` joining the next line, with their output. */
export function readCommands(block: Block): CommandLine[] {
  const lines = block.text.replace(/\n$/, "").split("\n");
  const commands: CommandLine[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i]!.startsWith("$ ")) continue;
    const line = block.line + 1 + i;
    let text = lines[i]!.slice(2);
    while (text.endsWith("\\") && i + 1 < lines.length) text = `${text.slice(0, -1)} ${lines[++i]!.trim()}`;
    const output: string[] = [];
    while (i + 1 < lines.length && !lines[i + 1]!.startsWith("$ ")) output.push(lines[++i]!);
    commands.push({ line, argv: shellWords(text), output: output.join("\n") });
  }
  return commands;
}

/** `spec` without its path checks: a page's paths name files the reader has, not the repo. */
function withoutPathChecks(spec: CommandSpec): CommandSpec {
  return {
    ...spec,
    positionals: spec.positionals.map(({ path: _path, ...rest }) => rest),
    options: spec.options.map(({ path: _path, ...rest }) => rest),
  };
}

/**
 * Parse a `yian ...` command line, split, with the CLI's own command table and parser; return the
 * command's key and its parsed arguments, or null for `--help`, `--version`, `--licenses` or a
 * group's help.
 *
 * @throws Error for an unknown command, and the CLI's `UsageError` for a line it would refuse.
 */
export function parseYian(argv: readonly string[]): { key: string; parsed: ParsedArgs } | null {
  const [yian, first, ...rest] = argv;
  if (yian !== "yian") throw new Error(`not a yian line: ${argv.join(" ")}`);
  if (first === undefined) throw new Error("yian with no command");
  if (["--help", "--version", "--licenses"].includes(first) && rest.length === 0) return null;
  let key = first;
  let args = rest;
  if (Object.hasOwn(GROUPS, first)) {
    const [sub, ...subRest] = rest;
    if (sub === undefined || sub === "--help") return null;
    key = `${first} ${sub}`;
    args = subRest;
  }
  if (!Object.hasOwn(COMMANDS, key)) throw new Error(`no command 'yian ${key}'`);
  const parsed = parseCommand(withoutPathChecks(COMMANDS[key]!.spec), args);
  return parsed.help ? null : { key, parsed };
}

/** Return why the CLI would refuse a `yian ...` line, or null when it takes it. */
export function cliProblem(argv: readonly string[]): string | null {
  try {
    parseYian(argv);
    return null;
  } catch (error) {
    return (error as Error).message;
  }
}
