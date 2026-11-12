/**
 * What a condition proves at each of its reads: checks, guards, and literals.
 *
 * A **check** is a test a condition makes on one value: `isinstance(<ref>, T)`, a tag
 * comparison `<ref>.<field> == "<text>"` (`!=` is the same check, negated), or a null check
 * `<ref> != null` (`==` is the same check, negated). A **guard** is a small boolean formula
 * over checks in negation normal form: `Holds(check, value)` says the check evaluated to
 * `value`; `AllOf` and `AnyOf` combine guards; `NOTHING` proves nothing, and `NEVER` never
 * holds.
 *
 * `and` and `or` short-circuit, so in `A and B` the operand `B` only runs once `A` is true,
 * and in `A or B` once `A` is false; in `A if C else B`, `A` runs once `C` is true and `B` once
 * it is false; `not` flips what a check proves. `proves` says what holds
 * when a condition evaluates to a value, `guardedRefs` says what holds whenever each reference
 * is read, and `literalsInContext` says what holds whenever each check is evaluated. This
 * module is syntax only: what a guard means for a value's type is `narrowing_facts.ts`'s job.
 *
 * A new check form adds a frozen class with a `base` path, adds it to `Check`, and handles it
 * in every function that dispatches on the check's kind; each of those throws `unknownCheck`
 * for a kind it doesn't know, so a form left out fails loudly:
 *
 * - here: its equality in `sameCheck`, its recognition in `checkIn` and its source in
 *   `checkText`;
 * - `narrowing_facts.ts`'s `checkFact`: the facts it proves, true and false;
 * - `narrowing_facts.ts`'s `unresolvedType`: whether it names a type that does not resolve;
 * - `narrowing_facts.ts`'s `declaredImpossible`: when its base's type alone decides it, and
 *   `neverValue`: the value such a reported check never has;
 * - `validate/asserts.ts`'s `narrows`: whether it can narrow an assert's root.
 *
 * `proves(<and chain>, false)` repeats each prefix of the chain, so a guard grows
 * quadratically with the length of one chain. An operand that makes no check proves `NOTHING`:
 * it drops out of a true `and` chain (a false `or` chain), and it makes a false `and` chain (a
 * true `or` chain) prove `NOTHING`, since it may be the operand that decided the chain. So guards
 * of authored conditions stay small.
 *
 * A comprehension variable read is keyed by its binding occurrence (`compVarKey`, such as
 * `m@12`), so its path is `m@12.k` and two comprehensions that both bind `m` don't mix.
 *
 * Knows about: `expr` parse trees and `directReferenceParts`.
 */

import { isinstanceSubject, isinstanceTypeNames } from "../../expr/index.ts";
import { ISINSTANCE, referencePath, splitReference, valueArgs } from "../../expr/expressions.ts";
import { callSuffixOf, compVarKey, type ExprNode, Token, Tree } from "../../expr/grammar.ts";
import { directReferenceParts } from "../../compile/output_contracts.ts";
import { YianTypeError } from "../../typesys/errors.ts";

/** An `isinstance(<base>, <typeName>)` check on a plain reference; frozen. */
export class IsinstanceCheck {
  /** The dotted path of the tested reference, such as `"d.output"`. */
  readonly base: string;
  /** The type name the check names. */
  readonly typeName: string;

  constructor(base: string, typeName: string) {
    this.base = base;
    this.typeName = typeName;
    Object.freeze(this);
  }
}

/**
 * A comparison of a field of a plain reference with a text literal; frozen.
 *
 * `<base>.<field> == "<value>"` and `"<value>" == <base>.<field>` are this check; `!=` is the
 * same check, negated.
 */
export class TagCheck {
  /** The dotted path of the value whose field is compared, such as `"v"`. */
  readonly base: string;
  /** The compared field's name. */
  readonly field: string;
  /** The text, decoded as the evaluator decodes a string literal (the quotes dropped, nothing unescaped). */
  readonly value: string;

  constructor(base: string, field: string, value: string) {
    this.base = base;
    this.field = field;
    this.value = value;
    Object.freeze(this);
  }
}

/**
 * A comparison of a plain reference with `null`: the check that its value is not null; frozen.
 *
 * `<base> != null` and `null != <base>` are this check, true when the value is not null; `==` is
 * the same check, negated. `none` spells `null` too.
 */
export class NotNullCheck {
  /** The dotted path of the compared reference, such as `"input.a.b"`. */
  readonly base: string;

  constructor(base: string) {
    this.base = base;
    Object.freeze(this);
  }
}

/** One check a condition can make on a value. */
export type Check = IsinstanceCheck | TagCheck | NotNullCheck;

/**
 * Return the error a dispatch on a check's kind throws for a kind it doesn't handle.
 *
 * @param check The check.
 * @returns The error, naming the check's class.
 */
export function unknownCheck(check: object): TypeError {
  return new YianTypeError(`unhandled check kind ${check.constructor.name}`);
}

/** Return whether two checks are the same check (the same kind and fields). */
export function sameCheck(a: Check, b: Check): boolean {
  if (a instanceof IsinstanceCheck) {
    return b instanceof IsinstanceCheck && a.base === b.base && a.typeName === b.typeName;
  }
  if (a instanceof TagCheck) {
    return b instanceof TagCheck && a.base === b.base && a.field === b.field && a.value === b.value;
  }
  if (a instanceof NotNullCheck) return b instanceof NotNullCheck && a.base === b.base;
  throw unknownCheck(a);
}

/** The guard that `check` evaluated to `value`; frozen. */
export class Holds {
  /** The check. */
  readonly check: Check;
  /** What it evaluated to. */
  readonly value: boolean;

  constructor(check: Check, value: boolean) {
    this.check = check;
    this.value = value;
    Object.freeze(this);
  }
}

/** The guard that every part holds; frozen. Build one with `allOf`. */
export class AllOf {
  /** The parts; `[]` is `NOTHING`, which proves nothing. */
  readonly parts: readonly Guard[];

  constructor(parts: readonly Guard[]) {
    this.parts = Object.freeze([...parts]);
    Object.freeze(this);
  }
}

/** The guard that at least one part holds; frozen. Build one with `anyOf`. */
export class AnyOf {
  /** The parts, at least two when built by `anyOf`. */
  readonly parts: readonly Guard[];

  constructor(parts: readonly Guard[]) {
    this.parts = Object.freeze([...parts]);
    Object.freeze(this);
  }
}

/** A boolean formula over checks, in negation normal form. */
export type Guard = Holds | AllOf | AnyOf;

/** The guard that proves nothing. */
export const NOTHING: Guard = new AllOf([]);

/** The guard that never holds, such as what a check that is never true proves when true. */
export const NEVER: Guard = new AnyOf([]);

/** Return whether two guards are the same formula (the same kinds, checks and parts, in order). */
export function sameGuard(a: Guard, b: Guard): boolean {
  // Guards share sub-guards, so comparing a shared one by identity skips its whole subtree.
  if (a === b) return true;
  if (a instanceof Holds) return b instanceof Holds && a.value === b.value && sameCheck(a.check, b.check);
  if (a instanceof AllOf) return b instanceof AllOf && sameParts(a.parts, b.parts);
  return b instanceof AnyOf && sameParts(a.parts, b.parts);
}

/** Return whether two part lists are the same guards, in order. */
function sameParts(a: readonly Guard[], b: readonly Guard[]): boolean {
  return a.length === b.length && a.every((part, i) => sameGuard(part, b[i]!));
}

/**
 * Return the guard that every part holds.
 *
 * Nested `AllOf`s are flattened and `NOTHING` parts dropped; one remaining part is returned as
 * is. A `NEVER` part makes the whole guard `NEVER`.
 *
 * @param parts The parts.
 * @returns `NEVER` when a part is `NEVER`, `NOTHING` for no parts, the lone part for one, else
 *   an `AllOf`.
 *
 * @example
 * `allOf(a, NOTHING, allOf(b, c))` is `new AllOf([a, b, c])`.
 */
export function allOf(...parts: Guard[]): Guard {
  const flat: Guard[] = [];
  for (const part of parts) {
    if (sameGuard(part, NEVER)) return NEVER;
    if (part instanceof AllOf) flat.push(...part.parts);
    else flat.push(part);
  }
  return flat.length === 1 ? flat[0]! : new AllOf(flat);
}

/**
 * Return the guard that at least one part holds.
 *
 * Nested `AnyOf`s are flattened (so `NEVER` parts drop out); a part that proves nothing makes
 * the whole guard prove nothing. One part is returned as is.
 *
 * @param parts The parts.
 * @returns `NOTHING` when a part is `NOTHING`, `NEVER` when no part remains, the lone part for
 *   one, else an `AnyOf`.
 *
 * @example
 * `anyOf(a, anyOf(b, c))` is `new AnyOf([a, b, c])`; `anyOf(a, NOTHING)` is `NOTHING`.
 */
export function anyOf(...parts: Guard[]): Guard {
  const flat: Guard[] = [];
  for (const part of parts) {
    if (sameGuard(part, NOTHING)) return NOTHING;
    if (part instanceof AnyOf) flat.push(...part.parts);
    else flat.push(part);
  }
  return flat.length === 1 ? flat[0]! : new AnyOf(flat);
}

/** The data of a node `masked` wraps a check in; no rule here reads it as a check. */
const MASKED = "masked_check";

/**
 * The data of a node `falsified` wraps a check in; its first child is the value the check never
 * has.
 */
const NEVER_HAS = "never_has";

/** The operands of an `and_expr` or `or_expr`, without the operator tokens. */
function operandsOf(node: Tree): ExprNode[] {
  return node.children.filter((child) => !(child instanceof Token && (child.type === "AND" || child.type === "OR")));
}

/** Whether `node` is a string literal token. */
function isText(node: ExprNode): node is Token {
  return node instanceof Token && node.type === "STRING";
}

/** Whether `node` is the `null` (or `none`) literal token. */
function isNull(node: ExprNode): boolean {
  return node instanceof Token && node.type === "NULL";
}

/** Return a comprehension variable read's path parts, `[key, ...steps]`, else `null`. */
function variableParts(node: ExprNode): string[] | null {
  if (!(node instanceof Tree) || node.data !== "comp_var") return null;
  const key = compVarKey(node);
  return key === null ? null : [key, ...splitReference(node).steps.map((step) => step.name)];
}

/** Return the path parts of a check's subject: a plain reference or a comprehension variable read. */
function subjectParts(node: ExprNode): string[] | null {
  return variableParts(node) ?? directReferenceParts(node);
}

/**
 * Return the check a sub-expression is, and the value it has when the sub-expression is true.
 *
 * A comprehension variable read is a plain reference here, by its key.
 *
 * @param tree A parsed sub-expression of a condition.
 * @returns `[IsinstanceCheck, true]` for `isinstance(<plain ref>, T)`; `[TagCheck, true]` for
 *   `<plain ref>.<field> == "<text>"` either way round, and `[TagCheck, false]` for `!=`;
 *   `[NotNullCheck, true]` for `<plain ref> != null` either way round, and
 *   `[NotNullCheck, false]` for `==`; `null` for anything else (a number or bool operand, a
 *   call, a tag comparison on a reference without a field, `not`, ...).
 *
 * @example
 * `checkIn(parseCondition("v.action != 'buy'"))` is `[new TagCheck("v", "action", "buy"),
 * false]`, and `checkIn(parseCondition("v != null"))` is `[new NotNullCheck("v"), true]`.
 */
export function checkIn(tree: ExprNode): [Check, boolean] | null {
  const subject = isinstanceSubject(tree);
  if (subject !== null) {
    const parts = subjectParts(subject);
    if (parts === null) return null;
    return [new IsinstanceCheck(parts.join("."), isinstanceTypeNames(tree)[0]!), true];
  }
  if (!(tree instanceof Tree) || tree.data !== "compare") return null;
  let [left, op, right] = tree.children as [ExprNode, ExprNode, ExprNode];
  const operator = String(op);
  if (operator !== "==" && operator !== "!=") return null;
  if (isNull(left)) [left, right] = [right, left];
  if (isNull(right)) {
    const parts = subjectParts(left);
    if (parts === null) return null;
    return [new NotNullCheck(parts.join(".")), operator === "!="];
  }
  if (isText(left)) [left, right] = [right, left];
  if (!isText(right)) return null;
  const parts = subjectParts(left);
  if (parts === null || parts.length < 2) return null;
  // The evaluator decodes a string literal by dropping its quotes, with no unescaping.
  const value = right.value.slice(1, -1);
  return [new TagCheck(parts.slice(0, -1).join("."), parts.at(-1)!, value), operator === "=="];
}

/**
 * Return what holds when a condition evaluates to `value`.
 *
 * A check gives `Holds`; `not` proves what its operand proves for the other value; an `and`
 * chain is true when every operand is, and false when some operand is false after the ones
 * before it were true (an `or` chain the other way round); a conditional has the value of the
 * branch its test picked; a false `and` chain with an operand
 * that proves nothing when false proves `NOTHING`, since that operand may have decided it.
 * Anything else (a comparison that is no check, `:-`, a call, a literal) proves `NOTHING`.
 *
 * @param tree A parsed condition or sub-expression.
 * @param value The value it evaluates to.
 * @returns What holds then.
 *
 * @example
 * `proves(parseCondition("isinstance(v, Buy) or v.action == 'sell'"), false)` is
 * `new AllOf([new Holds(new IsinstanceCheck("v", "Buy"), false),
 * new Holds(new TagCheck("v", "action", "sell"), false)])`.
 */
export function proves(tree: ExprNode, value: boolean): Guard {
  return provesWith(tree, value, new Map());
}

/** What sub-trees prove within one top-level call, by node identity: `[when false, when true]`. */
type ProvesMemo = Map<ExprNode, [Guard | undefined, Guard | undefined]>;

/**
 * `proves`, reusing what `memo` already holds and adding to it.
 *
 * A chain proves its operands for both values, and a nested chain does the same for its own, so
 * without the memo the work doubles at each level of alternating `and`/`or` nesting.
 */
function provesWith(tree: ExprNode, value: boolean, memo: ProvesMemo): Guard {
  let known = memo.get(tree);
  if (known === undefined) {
    known = [undefined, undefined];
    memo.set(tree, known);
  }
  const index = Number(value);
  const guard = known[index] ?? provesUncached(tree, value, memo);
  known[index] = guard;
  return guard;
}

/** `proves` for a node not in `memo` yet. */
function provesUncached(tree: ExprNode, value: boolean, memo: ProvesMemo): Guard {
  const found = checkIn(tree);
  if (found !== null) {
    const [check, positive] = found;
    return new Holds(check, value === positive);
  }
  if (!(tree instanceof Tree)) return NOTHING;
  if (tree.data === NEVER_HAS) return value === (String(tree.children[0]) === "true") ? NEVER : NOTHING;
  if (tree.data === "negate") return provesWith(tree.children.at(-1)!, !value, memo);
  if (tree.data === "if_expr") {
    // The branch the test picked has the value.
    const [then, test, otherwise] = tree.children as [ExprNode, ExprNode, ExprNode];
    return anyOf(
      allOf(provesWith(test, true, memo), provesWith(then, value, memo)),
      allOf(provesWith(test, false, memo), provesWith(otherwise, value, memo)),
    );
  }
  if (tree.data !== "and_expr" && tree.data !== "or_expr") return NOTHING;
  const operands = operandsOf(tree);
  // An `and` is decided by its first false operand, an `or` by its first true one.
  const deciding = tree.data === "or_expr";
  if (value !== deciding) return allOf(...operands.map((operand) => provesWith(operand, value, memo)));
  const decided = operands.map((operand) => provesWith(operand, deciding, memo));
  // An operand that proves nothing may be the one that decided the chain.
  if (decided.some((guard) => sameGuard(guard, NOTHING))) return NOTHING;
  const branches: Guard[] = [];
  const passed: Guard[] = [];
  for (const [i, operand] of operands.entries()) {
    branches.push(allOf(...passed, decided[i]!));
    passed.push(provesWith(operand, !deciding, memo));
  }
  return anyOf(...branches);
}

/** Each operand of an `and_expr`/`or_expr` with what holds whenever it is evaluated. */
function operandContexts(node: Tree, context: Guard, memo: ProvesMemo): [ExprNode, Guard][] {
  // The value an operand must have for the next one to run.
  const continues = node.data === "and_expr";
  const contexts: [ExprNode, Guard][] = [];
  const passed: Guard[] = [context];
  for (const operand of operandsOf(node)) {
    contexts.push([operand, allOf(...passed)]);
    passed.push(provesWith(operand, continues, memo));
  }
  return contexts;
}

/**
 * The parts of an `if_expr` in source order, each with what holds whenever it is evaluated: the
 * test under `context`, and each branch under `context` and what the test proves for it.
 */
function branchContexts(node: Tree, context: Guard, memo: ProvesMemo): [ExprNode, Guard][] {
  const [then, test, otherwise] = node.children as [ExprNode, ExprNode, ExprNode];
  return [
    [then, allOf(context, provesWith(test, true, memo))],
    [test, context],
    [otherwise, allOf(context, provesWith(test, false, memo))],
  ];
}

/**
 * One read occurrence of a condition: its refcall (or `comp_var`), its flat path, and the guard it
 * is read under.
 */
export interface GuardedOccurrence {
  readonly node: Tree;
  readonly path: string;
  readonly guard: Guard;
}

/**
 * Return each reference occurrence in a condition with its refcall node and the guard that holds
 * when it is read.
 *
 * The `i`-th operand of an `and` chain is evaluated only once the operands before it were true,
 * and of an `or` chain once they were false; a conditional's `then` once its test was true and
 * its `else` once it was false; every other node (`not`, `:-`, `|`, a call, a comparison)
 * evaluates its parts under its own guard. So a check after a read never guards it.
 * The occurrences are those of `exprRefs` and `conditionReads`, in the same order (plain
 * references, builtin-call argument values, and an `isinstance` call's subject), plus each
 * comprehension variable read, with path `<key>.<steps>`. A comprehension's clauses run left to
 * right before its elem, and each `if` adds what it proves true for the clauses after it and the
 * elem, as an `and` chain would; reads stay in source order (the elem first).
 *
 * @param tree A condition parsed with `parseCondition`, in any spelling.
 * @returns One occurrence per read, in source order.
 */
export function guardedReads(tree: ExprNode): GuardedOccurrence[] {
  const reads: GuardedOccurrence[] = [];
  const memo: ProvesMemo = new Map();

  const walk = (node: ExprNode, context: Guard): void => {
    if (node instanceof Token) return;
    if (node.data === "refcall") {
      const callSuffix = callSuffixOf(node);
      if (callSuffix === undefined) {
        reads.push({ node, path: referencePath(node), guard: context });
        return;
      }
      for (const arg of valueArgs(String(node.children[0]), callSuffix)) {
        // A keyword argument's value is its last child.
        walk((arg as Tree).children.at(-1)!, context);
      }
      return;
    }
    if (node.data === "comp_var") {
      const parts = variableParts(node);
      if (parts !== null) reads.push({ node, path: parts.join("."), guard: context });
      return;
    }
    if (node.data === "and_expr" || node.data === "or_expr") {
      for (const [operand, operandContext] of operandContexts(node, context, memo)) walk(operand, operandContext);
      return;
    }
    if (node.data === "if_expr") {
      // In source order: the branch the test's truth picks is read under what the test proves.
      for (const [part, partContext] of branchContexts(node, context, memo)) walk(part, partContext);
      return;
    }
    if (node.data === "list_comp") {
      const [elem, ...clauses] = node.children as Tree[];
      const contexts: Guard[] = [];
      let passed = context;
      for (const clause of clauses) {
        contexts.push(passed);
        if (clause.data === "comp_if") passed = allOf(passed, provesWith(clause.children[0]!, true, memo));
      }
      walk(elem!, passed);
      // A `comp_for`'s iterable is its second child; a `comp_if`'s condition its only one.
      clauses.forEach((clause, i) => walk(clause.children[clause.data === "comp_for" ? 1 : 0]!, contexts[i]!));
      return;
    }
    for (const child of node.children) walk(child, context);
  };

  walk(tree, NOTHING);
  return reads;
}

/**
 * Return each reference occurrence in a condition with the guard that holds when it is read: the
 * pairs of `guardedReads`, in the same order.
 *
 * @param tree A condition parsed with `parseCondition`, in any spelling.
 * @returns One `[reference path, guard]` pair per occurrence.
 *
 * @example
 * For `not isinstance(r, Done) or r.answer == 'ok'`, the pairs are `["r", NOTHING]` and
 * `["r.answer", new Holds(new IsinstanceCheck("r", "Done"), true)]`.
 */
export function guardedRefs(tree: ExprNode): [string, Guard][] {
  return guardedReads(tree).map((read) => [read.path, read.guard]);
}

/** A check, or `not` applied to a check, in a boolean position of a condition; frozen. */
export class CheckLiteral {
  /** What holds whenever the literal is evaluated. */
  readonly context: Guard;
  /** The check. */
  readonly check: Check;
  /** The value the check has when the literal is true. */
  readonly value: boolean;
  /** Whether the literal is `not` applied to the check (after `not not` cancels). */
  readonly negated: boolean;
  /** The check's sub-tree in the parsed condition, for `masked`; not part of the literal's value. */
  readonly node: ExprNode | null;

  constructor(context: Guard, check: Check, value: boolean, negated: boolean, node: ExprNode | null = null) {
    this.context = context;
    this.check = check;
    this.value = value;
    this.negated = negated;
    this.node = node;
    Object.freeze(this);
  }
}

/**
 * Return each literal in a condition's boolean positions with what holds when it is evaluated.
 *
 * Boolean positions are the condition itself, the operands of `and`, `or` and `not`, the first
 * operand of `:-` and `:?`, the branches of a conditional in one, and a comprehension's `if`
 * clause and a conditional's test, wherever the comprehension or conditional is. A
 * check anywhere else (a comparison operand, a call argument, a list item, an operand of `|`) is
 * no literal. Contexts follow the rules of `guardedRefs`.
 *
 * @param tree A condition parsed with `parseCondition`, in any spelling.
 * @returns The literals in source order.
 *
 * @example
 * For `isinstance(v, Buy) and v.action == 'hold'`, the second literal is
 * `new CheckLiteral(new Holds(new IsinstanceCheck("v", "Buy"), true),
 * new TagCheck("v", "action", "hold"), true, false)`.
 */
export function literalsInContext(tree: ExprNode): CheckLiteral[] {
  const literals: CheckLiteral[] = [];
  const memo: ProvesMemo = new Map();

  // A boolean position.
  const walk = (node: ExprNode, context: Guard, negated: boolean): void => {
    const found = checkIn(node);
    if (found !== null) {
      const [check, positive] = found;
      literals.push(new CheckLiteral(context, check, positive !== negated, negated, node));
      return;
    }
    if (!(node instanceof Tree)) return;
    if (node.data === "negate") {
      walk(node.children.at(-1)!, context, !negated);
    } else if (node.data === "and_expr" || node.data === "or_expr") {
      for (const [operand, operandContext] of operandContexts(node, context, memo)) walk(operand, operandContext, false);
    } else if (node.data === "default_expr" || node.data === "required_expr") {
      walk(node.children[0]!, context, false);
      scan(node.children[1]!, context);
    } else if (node.data === "if_expr") {
      // A branch's truth is the conditional's.
      for (const [part, partContext] of branchContexts(node, context, memo)) walk(part, partContext, false);
    } else {
      scan(node, context);
    }
  };

  // Any other position, searched only for comprehensions, whose `if` clauses are boolean positions.
  const scan = (node: ExprNode, context: Guard): void => {
    if (!(node instanceof Tree)) return;
    if (node.data === "list_comp") {
      const [elem, ...clauses] = node.children as Tree[];
      let passed = context;
      for (const clause of clauses) {
        if (clause.data === "comp_for") {
          scan(clause.children[1]!, passed);
        } else {
          walk(clause.children[0]!, passed, false);
          passed = allOf(passed, provesWith(clause.children[0]!, true, memo));
        }
      }
      scan(elem!, passed);
    } else if (node.data === "and_expr" || node.data === "or_expr") {
      for (const [operand, operandContext] of operandContexts(node, context, memo)) scan(operand, operandContext);
    } else if (node.data === "if_expr") {
      // The test is tested for truth wherever the conditional is.
      for (const [part, partContext] of branchContexts(node, context, memo)) {
        if (part === node.children[1]) walk(part, partContext, false);
        else scan(part, partContext);
      }
    } else {
      for (const child of node.children) scan(child, context);
    }
  };

  walk(tree, NOTHING, false);
  return literals;
}

/**
 * Return a copy of a condition in which the given sub-trees prove nothing.
 *
 * Each sub-tree in `nodes` (matched by identity, such as a `CheckLiteral.node`) is wrapped in a
 * node no rule reads as a check, so `proves` gives `NOTHING` for it while `guardedRefs` still
 * finds the same references in the same order.
 *
 * @param tree A condition parsed with `parseCondition`.
 * @param nodes Sub-trees of `tree` to mask.
 * @returns The copy; `tree` itself is left unchanged.
 *
 * @example
 * In `masked(tree, [literal.node])` for the literal `isinstance(v, Buy)` of
 * `isinstance(v, Buy) and v.qty > 0`, the read `v.qty` is guarded by `NOTHING`.
 */
export function masked(tree: ExprNode, nodes: Iterable<ExprNode>): ExprNode {
  const targets = new Set(nodes);

  const copy = (node: ExprNode): ExprNode => {
    if (targets.has(node)) return new Tree(MASKED, [node]);
    if (node instanceof Token) return node;
    return new Tree(node.data, node.children.map(copy));
  };

  return copy(tree);
}

/**
 * Return a copy of a condition in which the given checks never have the given values.
 *
 * Each `[sub-tree, value]` pair (the sub-tree matched by identity, such as a
 * `CheckLiteral.node`) wraps that check so `proves` gives `NEVER` for `value` and `NOTHING` for
 * the other value, while `guardedRefs` still finds the same references in the same order. A
 * reference whose guard is then `NEVER` is read only when one of those checks has its
 * impossible value.
 *
 * @param tree A condition parsed with `parseCondition`.
 * @param never Check sub-trees of `tree` and the value each never has.
 * @returns The copy; `tree` itself is left unchanged.
 *
 * @example
 * In `falsified(tree, [[literal.node, true]])` for the literal `isinstance(v, Text)` of
 * `isinstance(v, Text) and v.qty > 0`, the read `v.qty` is guarded by `NEVER`.
 */
export function falsified(tree: ExprNode, never: Iterable<readonly [ExprNode, boolean]>): ExprNode {
  const values = new Map(never);

  const copy = (node: ExprNode): ExprNode => {
    const value = values.get(node);
    if (value !== undefined) return new Tree(NEVER_HAS, [new Token("NEVER_VALUE", String(value)), node]);
    if (node instanceof Token) return node;
    return new Tree(node.data, node.children.map(copy));
  };

  return copy(tree);
}

/** `text` as a string literal, in double quotes unless it holds one. */
function quoted(text: string): string {
  return text.includes('"') ? `'${text}'` : `"${text}"`;
}

/**
 * Return the canonical source of a check that evaluates to `value` when true.
 *
 * @param check The check.
 * @param value For a `TagCheck`, `false` spells the comparison with `!=`; for a `NotNullCheck`,
 *   with `==`. An `IsinstanceCheck` is always spelled positively.
 * @returns Such as `isinstance(d.output, Buy)`, `v.action != "buy"` or `v == null`.
 */
export function checkText(check: Check, value = true): string {
  if (check instanceof IsinstanceCheck) return `${ISINSTANCE}(${check.base}, ${check.typeName})`;
  if (check instanceof TagCheck) {
    const op = value ? "==" : "!=";
    return `${check.base}.${check.field} ${op} ${quoted(check.value)}`;
  }
  if (check instanceof NotNullCheck) {
    const op = value ? "!=" : "==";
    return `${check.base} ${op} null`;
  }
  throw unknownCheck(check);
}

/**
 * Return the canonical source of a literal.
 *
 * @param literal A literal from `literalsInContext`.
 * @returns The check's canonical source, after `not ` when the literal is negated:
 *   `not isinstance(v, Buy)`, `v.action != "buy"`, `not v.action == "buy"`.
 */
export function literalText(literal: CheckLiteral): string {
  const positive = literal.value !== literal.negated;
  const text = checkText(literal.check, positive);
  return literal.negated ? `not ${text}` : text;
}

/**
 * Return every check in a guard.
 *
 * @param guard The guard.
 * @returns The checks in first-seen order, without duplicates.
 */
export function guardChecks(guard: Guard): readonly Check[] {
  const found: Check[] = [];
  // `proves` shares sub-guards, so visiting each once keeps the walk linear in the guard's graph.
  const visited = new Set<Guard>();

  const walk = (node: Guard): void => {
    if (visited.has(node)) return;
    visited.add(node);
    if (node instanceof Holds) {
      if (!found.some((check) => sameCheck(check, node.check))) found.push(node.check);
      return;
    }
    for (const part of node.parts) walk(part);
  };

  walk(guard);
  return Object.freeze(found);
}

/**
 * Return the parts of a top-level `AllOf`, else the guard itself.
 *
 * @param guard The guard.
 * @returns `guard.parts` for an `AllOf` (`[]` for `NOTHING`), else `[guard]`.
 */
export function conjuncts(guard: Guard): readonly Guard[] {
  return guard instanceof AllOf ? guard.parts : Object.freeze([guard]);
}
