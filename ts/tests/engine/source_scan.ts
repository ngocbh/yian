/**
 * A small lexer for the source-scanning tests (Python used `ast`).
 *
 * It understands comments, strings, templates and regex literals well enough to
 * blank them, not full TypeScript.
 */

/** Keywords after which a `/` starts a regex literal rather than a division. */
const REGEX_KEYWORDS = new Set([
  "return",
  "typeof",
  "case",
  "do",
  "else",
  "in",
  "of",
  "new",
  "delete",
  "void",
  "throw",
  "instanceof",
  "yield",
  "await",
]);

const IDENTIFIER = /[\p{ID_Continue}$]/u;

type State = "code" | "line" | "block" | "string" | "template" | "regex" | "class";

/**
 * Return `text` with comments and the contents of strings, templates and regex
 * literals replaced by spaces.
 *
 * Quote, backtick and regex delimiters, `${`/`}` around template code, and every
 * newline stay in place, so offsets and line numbers match the original.
 */
export function blank(text: string): string {
  const out = text.split("");
  const hide = (i: number): void => {
    if (i < text.length && text[i] !== "\n" && text[i] !== "\r") out[i] = " ";
  };
  let state: State = "code";
  let quote = "";
  // One entry per open `${`: the depth of `{` nested inside it, so templates nest.
  const substitutions: number[] = [];
  // Whether a `/` here would start a regex literal, judged by the previous token.
  let regexOk = true;

  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    const next = text[i + 1];
    switch (state) {
      case "code":
        if (/\s/.test(c)) break;
        if (c === "/" && next === "/") {
          state = "line";
          hide(i);
        } else if (c === "/" && next === "*") {
          state = "block";
          hide(i);
          hide(++i);
        } else if (c === "/") {
          if (regexOk) state = "regex";
          regexOk = true;
        } else if (c === '"' || c === "'") {
          state = "string";
          quote = c;
        } else if (c === "`") {
          state = "template";
        } else if (IDENTIFIER.test(c)) {
          let j = i;
          while (j + 1 < text.length && IDENTIFIER.test(text[j + 1]!)) j++;
          // After `.` the word is a property name (`obj.return / 2`), not a keyword.
          regexOk = text[i - 1] !== "." && REGEX_KEYWORDS.has(text.slice(i, j + 1));
          i = j;
        } else if (c === ")" || c === "]") {
          regexOk = false;
        } else if (c === "!" && !regexOk) {
          // A postfix non-null assertion keeps the operand: `a! / 2` divides.
        } else if ((c === "+" || c === "-") && next === c && !regexOk) {
          i++; // postfix `++`/`--`: `i++ / n` divides
        } else if (c === "}" && substitutions.at(-1) === 0) {
          substitutions.pop();
          state = "template";
        } else {
          if (substitutions.length > 0 && (c === "{" || c === "}")) {
            substitutions.push(substitutions.pop()! + (c === "{" ? 1 : -1));
          }
          regexOk = true;
        }
        break;
      case "line":
        if (c === "\n") state = "code";
        else hide(i);
        break;
      case "block":
        hide(i);
        if (c === "*" && next === "/") {
          hide(++i);
          state = "code";
        }
        break;
      case "string":
      case "regex":
      case "class":
        // Neither a quoted string nor a regex literal can span lines.
        if (c === "\n") {
          state = "code";
          regexOk = true;
        } else if (c === "\\") {
          hide(i);
          if (next !== "\n") hide(++i);
        } else if (state === "string" && c === quote) {
          state = "code";
          regexOk = false;
        } else if (state === "regex" && c === "/") {
          state = "code";
          regexOk = false;
        } else if (state === "regex" && c === "[") {
          state = "class";
          hide(i);
        } else if (state === "class" && c === "]") {
          state = "regex";
          hide(i);
        } else {
          hide(i);
        }
        break;
      case "template":
        if (c === "\\") {
          hide(i);
          hide(++i);
        } else if (c === "`") {
          state = "code";
          regexOk = false;
        } else if (c === "$" && next === "{") {
          substitutions.push(0);
          state = "code";
          regexOk = true;
          i++;
        } else {
          hide(i);
        }
        break;
    }
  }
  return out.join("");
}

// `import`/`export ... from "x"`, side-effect `import "x"`, and `import("x")`; the
// keyword guard keeps a semicolon-less `export { a }` from swallowing the next import.
const IMPORT =
  /(?<![\w$.])(?:(?:import|export)(?![\w$])(?:(?!(?<![\w$])(?:import|export)(?![\w$]))[\w$\s{},*])*?(?<![\w$])from\s*|import\s*(?:\(\s*)?)(["'])([^"'\n]*)\1/g;

/** One import or re-export found by `importStatements`. */
export interface ImportStatement {
  start: number;
  end: number;
  line: number;
  specifier: string;
}

/**
 * Return each import or re-export in `text`: its span from the keyword to the
 * specifier's closing quote, the 1-based line of the specifier, and the specifier.
 *
 * Only quoted specifiers count; a backtick one (`` import(`./x.ts`) ``) isn't recognized.
 */
export function importStatements(text: string): ImportStatement[] {
  const found: ImportStatement[] = [];
  for (const match of blank(text).matchAll(IMPORT)) {
    const end = match.index + match[0].length;
    const open = end - 2 - match[2]!.length;
    found.push({
      start: match.index,
      end,
      line: text.slice(0, open).split("\n").length,
      specifier: text.slice(open + 1, end - 1),
    });
  }
  return found;
}
