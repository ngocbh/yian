/**
 * Strict type inference for generated inline-composite input declarations.
 *
 * Inline composite bodies become private child flows, so every outer binding must produce a
 * concrete child-parameter type. This module infers only types that the Compose source proves:
 * an expression's type comes from the shared output-contract inference, and a literal's from its
 * structure. It never falls back to the runtime value builder's lenient `Any` behavior.
 * Top-level strings and the leaf strings of a value template are binding templates; a span-free
 * literal container's strings are ordinary strings.
 *
 * Knows about: `expr` template syntax, node-owned semantic binding/output-contract carriers,
 * `typesys` structural types, the shared output-contract inference in `compile`, and Compose
 * source ranges/errors.
 * Never imports: the loader, the runtime, or node kinds.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import type { Tree } from "../../expr/grammar.ts";
import { scanTemplate, Span, templateRefs } from "../../expr/template.ts";
import {
  Binding,
  bindingSource,
  isExactly,
  type OutputContract,
  ValueContract,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { messagesKey, NEW_MESSAGES } from "../../typesys/pool.ts";
import { typeName } from "../../typesys/types.ts";
import {
  canonicalTypeLabel,
  isListKind,
  isRecord,
  listKindForElement,
  Type,
  typeEquals,
  ValueKind,
  writeAsJson,
  writeAsText,
} from "../../typesys/values.ts";
import { LoadDiagnostics, type LoadError } from "../helpers/errors.ts";
import {
  directReferenceParts,
  hasOptionalStep,
  outputContractPathError,
  projectOutputContract,
  resolveOutputBindingContract,
  type TypeFn,
} from "../../compile/output_contracts.ts";
import { SYSTEM_AMBIENTS } from "../../compile/validation.ts";

/**
 * Return the advice for an inline binding whose type can't be proven.
 *
 * @param bindingLabel The binding's node/input path, such as `"each.input.text"`; its last
 *   segment names the slot.
 * @returns Advice to tag the value with its type or use a typed named body.
 */
export function explicitTypeHint(bindingLabel: string): string {
  const name = bindingLabel.split(".").at(-1)!;
  return `declare its type with a tag (\`${name}: !<type> ...\`) or use an explicitly typed named \`defs:\` body`;
}

/**
 * Resolves a producer key to its output type or canonical output contract, or `null` when the
 * producer has no static type: a node id for its output, or `messagesKey(id)` for an AGENT's new
 * messages. It receives the requesting binding label and range so lazy-build failures keep their
 * source site.
 */
export type ProducerType = (
  producerKey: string,
  bindingLabel: string,
  bindingRange: SourceRange | null,
) => Type | OutputContract | null;

/** Lazily supplies a MAP element type for a direct `${item...}` read. */
export type ItemType = (bindingLabel: string, bindingRange: SourceRange | null) => Type;

// The reference heads that never name a producer, even before `.output`.
const NON_PRODUCER_HEADS: ReadonlySet<string> = new Set(["input", "item", "system", "inputs", "outputs"]);

/** The shared context of one binding's inference walk. */
interface Walk {
  readonly parentInputTypes: ReadonlyMap<string, Type>;
  readonly producerType: ProducerType;
  readonly itemType: ItemType | null;
  readonly typeFn: TypeFn | null;
  readonly bindingLabel: string;
  readonly bindingRange: SourceRange | null;
  readonly diagnostics: LoadDiagnostics;
}

/**
 * Infer one generated child parameter's strict structural type.
 *
 * `bindingLabel` names the outer binding for diagnostics, such as `"each.input.text"`. A string
 * passed directly as `value` and each leaf string of a `ValueTemplate` are scanned as binding
 * templates; a typed binding nested in a template gives its declared type. A plain record/list
 * value is a literal, so its strings infer as `str` without scanning. A lone direct reference
 * obtains its root type from the parent signature, a system ambient, or an injected producer/MAP-
 * item provider, then projects its fields with the shared output-contract walk. Any other
 * whole-value expression (an operator, a builtin, a `T(...)` constructor that `typeFn` resolves)
 * has the exact type `resolveOutputBindingContract` proves over the parent inputs and the
 * producers it reads; a `${item}` read inside such an expression stays unproven.
 *
 * @example
 * `inferInlineBindingType(3n, { parentInputTypes: new Map(), producerType, bindingLabel:
 * "call.input.limit" })` is `Type.scalar(ValueKind.INTEGER)`.
 *
 * @param value The authored outer binding value.
 * @param options.parentInputTypes Parent flow input declarations available through
 *   `${input.<name>}`.
 * @param options.producerType Resolves a node id, or `messagesKey(id)` for an AGENT's new
 *   messages, to its output type or canonical output contract, or `null` when it has none.
 * @param options.itemType Lazily supplies a MAP element type for a direct `${item...}` read;
 *   `null` (the default) rejects item references outside an outer MAP input.
 * @param options.typeFn Resolves a constructor name to its type; `null` (the default) leaves
 *   constructors unproven.
 * @param options.bindingLabel The author-facing node/input path included in inference errors.
 * @param options.bindingRange The binding value's source range, or `null` (the default) for
 *   spanless input.
 * @param options.diagnostics Factory for located load errors (default: source-free).
 * @returns The exact scalar or recursively structural type proven by `value`.
 * @throws LoadError when the value has no statically proven type under the inline-body rules.
 */
export function inferInlineBindingType(
  value: unknown,
  options: {
    parentInputTypes: ReadonlyMap<string, Type>;
    producerType: ProducerType;
    itemType?: ItemType | null;
    typeFn?: TypeFn | null;
    bindingLabel: string;
    bindingRange?: SourceRange | null;
    diagnostics?: LoadDiagnostics | null;
  },
): Type {
  const walk: Walk = {
    parentInputTypes: options.parentInputTypes,
    producerType: options.producerType,
    itemType: options.itemType ?? null,
    typeFn: options.typeFn ?? null,
    bindingLabel: options.bindingLabel,
    bindingRange: options.bindingRange ?? null,
    diagnostics: options.diagnostics ?? new LoadDiagnostics(),
  };
  const source = bindingSource(value);
  let inferred: Type;
  if (typeof source === "string") {
    inferred = inferBindingString(source, walk);
  } else if (source instanceof ValueTemplate) {
    inferred = inferTemplate(source.value, walk.bindingLabel, walk);
  } else {
    inferred = inferLiteral(source, walk.bindingLabel, walk);
  }
  return requireProvenType(inferred, walk);
}

/**
 * Require every structural refinement in `typ` to be statically proven.
 *
 * A refined record is closed: each declared field must either be required or nullable, and every
 * child type is checked recursively. A union requires each member to be proven. An unrefined
 * object remains valid as a whole value. Every list storage kind, including `list(Optional(Any))`
 * with no element, requires a recursively proven structural element.
 *
 * @param typ The type to prove without copying or mutating it.
 * @param options.bindingLabel The author-facing node/input path included in inference errors.
 * @param options.bindingRange The binding value's source range, or `null` (the default) for
 *   spanless input.
 * @param options.diagnostics Factory for located load errors (default: source-free).
 * @returns The exact same object passed as `typ` after successful proof.
 * @throws LoadError when a record or list refinement is incomplete or internally invalid.
 */
export function requireProvenType(
  typ: Type,
  options: { bindingLabel: string; bindingRange?: SourceRange | null; diagnostics?: LoadDiagnostics | null },
): Type {
  const bindingLabel = options.bindingLabel;
  const bindingRange = options.bindingRange ?? null;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const prove = (child: Type) => requireProvenType(child, { bindingLabel, bindingRange, diagnostics });
  const fail = (reason: string) => inferenceError(bindingLabel, bindingRange, reason, diagnostics);

  if (typ.fields !== undefined) {
    const fields = typ.fields;
    const required = typ.required ?? new Set<string>();
    const missing = [...required].filter((name) => !fields.has(name));
    if (missing.length > 0) {
      throw fail(`record requires unknown fields ${writeAsText(missing.sort(compareCodeUnits))}`);
    }
    for (const [name, fieldType] of fields) {
      if (!required.has(name) && !fieldType.nullable) {
        throw fail(`record field ${reprText(name)} is neither required nor nullable`);
      }
      prove(fieldType);
    }
  }

  for (const member of typ.members ?? []) prove(member);

  if (isListKind(typ.kind)) {
    if (typ.element === undefined) {
      throw fail(`list type ${reprText(canonicalTypeLabel(typ))} has no proven element type`);
    }
    prove(typ.element);
  }

  return typ;
}

/**
 * Return a recursively copied, deterministically ordered inline type.
 *
 * Record fields use code-unit order. Their exact valid required membership is retained, including
 * required nullable fields; a refined record with no `required` normalizes to an empty set. Other
 * type metadata is copied without changing its structural meaning.
 *
 * @param typ The already-proven type to canonicalize.
 * @returns A deep canonical copy; `typ` is never mutated.
 */
export function canonicalizeInlineType(typ: Type): Type {
  let fields: Map<string, Type> | undefined;
  let required: Set<string> | undefined;
  if (typ.fields === undefined) {
    required = typ.required === undefined ? undefined : new Set([...typ.required].sort(compareCodeUnits));
  } else {
    const source = typ.fields;
    fields = new Map(
      [...source.keys()].sort(compareCodeUnits).map((name) => [name, canonicalizeInlineType(source.get(name)!)]),
    );
    const sourceRequired = typ.required ?? new Set<string>();
    required = new Set([...fields.keys()].filter((name) => sourceRequired.has(name)));
  }

  const tags = typ.tags === undefined ? undefined : new Set([...typ.tags].sort(compareCodeUnits));
  const element = typ.element === undefined ? undefined : canonicalizeInlineType(typ.element);
  // Union members are already in canonical order; only their contents are copied.
  const members = typ.members?.map(canonicalizeInlineType);
  return new Type({
    kind: typ.kind,
    fields,
    required,
    tags,
    element,
    nullable: typ.nullable,
    bounds: typ.bounds,
    members,
  });
}

/**
 * Infer a top-level string using binding-template semantics: text, embedded or several spans are
 * `str`, a direct reference is resolved by `inferDirectReference`, and any other whole-value
 * expression has the exact type `resolveOutputBindingContract` proves.
 */
function inferBindingString(source: string, walk: Walk): Type {
  let segments;
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw inferenceError(
      walk.bindingLabel,
      walk.bindingRange,
      `invalid binding template: ${error.message}`,
      walk.diagnostics,
      error,
    );
  }

  const only = segments[0];
  if (segments.length !== 1 || !(only instanceof Span)) return Type.scalar(ValueKind.STRING);

  const expression = only.tree;
  // A `?.` step reads null when its receiver is null or lacks the key, as `inferExprType` types it.
  const directParts = directReferenceParts(expression);
  if (directParts !== null) {
    const typ = inferDirectReference(directParts, walk);
    return hasOptionalStep(expression as Tree) ? typ.replace({ nullable: true }) : typ;
  }

  const inputs = new Map([...walk.parentInputTypes].map(([name, typ]) => [name, new ValueContract(typ)]));
  const contract = resolveOutputBindingContract(source, readProducers(templateRefs(segments), walk), inputs, {
    typeFn: walk.typeFn,
  });
  if (!isExactly(contract, ValueContract)) {
    throw inferenceError(
      walk.bindingLabel,
      walk.bindingRange,
      "a whole-value expression has no statically proven type",
      walk.diagnostics,
    );
  }
  return contract.type;
}

/**
 * Return the output contracts of the producers that `refs` read, by producer key; a producer with
 * no static type is left out, so an expression over it stays unproven.
 */
function readProducers(refs: readonly string[], walk: Walk): Map<string, OutputContract> {
  const producers = new Map<string, OutputContract>();
  for (const path of refs) {
    const [head, second] = path.split(".") as [string, string | undefined];
    if (NON_PRODUCER_HEADS.has(head) || (second !== "output" && second !== NEW_MESSAGES)) continue;
    const key = second === NEW_MESSAGES ? messagesKey(head) : head;
    if (producers.has(key)) continue;
    const produced = walk.producerType(key, walk.bindingLabel, walk.bindingRange);
    if (produced !== null) producers.set(key, produced instanceof Type ? new ValueContract(produced) : produced);
  }
  return producers;
}

/** Resolve one direct reference root, then project its fields with the shared output-contract walk. */
function inferDirectReference(parts: readonly string[], walk: Walk): Type {
  const path = parts.join(".");
  const head = parts[0]!;
  const fail = (reason: string) => inferenceError(walk.bindingLabel, walk.bindingRange, reason, walk.diagnostics);
  let root: OutputContract;
  let fields: readonly string[];

  if (head === "input") {
    if (parts.length < 2) throw fail(`reference \${${path}} must be input.<name>[.field]`);
    const inputName = parts[1]!;
    const declared = walk.parentInputTypes.get(inputName);
    if (declared === undefined) {
      throw fail(`reference \${${path}} points to unknown parent input ${reprText(inputName)}`);
    }
    root = new ValueContract(requireProvenType(declared, walk));
    fields = parts.slice(2);
  } else if (head === "item") {
    if (walk.itemType === null) throw fail(`reference \${${path}} is only typed inside an outer MAP input`);
    root = new ValueContract(requireProvenType(walk.itemType(walk.bindingLabel, walk.bindingRange), walk));
    fields = parts.slice(1);
  } else if (head === "inputs" || head === "outputs") {
    throw fail(`reference \${${path}} uses retired plural head ${reprText(head)}`);
  } else if (head === "system") {
    const kind = parts.length >= 2 ? SYSTEM_AMBIENTS.get(parts[1]!) : undefined;
    if (kind === undefined) throw fail(`reference \${${path}} has no statically proven system type`);
    root = new ValueContract(Type.scalar(kind));
    fields = parts.slice(2);
  } else {
    if (parts.length < 2 || (parts[1] !== "output" && parts[1] !== NEW_MESSAGES)) {
      throw fail(`reference \${${path}} must select ${head}.output[.field] or an AGENT's ${head}.${NEW_MESSAGES}`);
    }
    const key = parts[1] === NEW_MESSAGES ? messagesKey(head) : head;
    const produced = walk.producerType(key, walk.bindingLabel, walk.bindingRange);
    if (produced === null) throw fail(`producer ${reprText(head)} has no available static type`);
    root = produced instanceof Type ? new ValueContract(requireProvenType(produced, walk)) : produced;
    fields = parts.slice(2);
  }

  const pathError = outputContractPathError(root, fields, path);
  if (pathError !== null) throw fail(pathError);
  const projected = projectOutputContract(root, fields);
  if (!isExactly(projected, ValueContract)) throw fail(`reference \${${path}} has no statically proven exact type`);
  return projected.type;
}

/** Infer one node of a value template: strings are templates, typed bindings declared. */
function inferTemplate(node: unknown, path: string, walk: Walk): Type {
  if (node instanceof Binding) {
    if (node.contract !== null) return node.contract.type;
    node = node.source;
  }
  if (typeof node === "string") return inferBindingString(node, walk);

  if (isRecord(node)) {
    const fields = new Map(
      Object.keys(node)
        .sort(compareCodeUnits)
        .map((key) => [key, inferTemplate(node[key], `${path}[${writeAsJson(key)}]`, walk)]),
    );
    return new Type({ kind: ValueKind.OBJECT, fields, required: new Set(fields.keys()) });
  }
  if (Array.isArray(node)) {
    if (node.length === 0) {
      throw inferenceError(
        walk.bindingLabel,
        walk.bindingRange,
        `empty literal list at ${path} has no provable element type`,
        walk.diagnostics,
      );
    }
    const element = commonElementType(
      node.map((item, index) => inferTemplate(item, `${path}[${index}]`, walk)),
      path,
      walk,
      "list",
    );
    return new Type({ kind: listKindForElement(element), element });
  }
  return inferLiteral(node, path, walk);
}

/** Infer a recursive literal; nested strings never enter the template parser. */
function inferLiteral(value: unknown, path: string, walk: Walk): Type {
  if (value === null) return Type.scalar(ValueKind.NONE);
  if (typeof value === "boolean") return Type.scalar(ValueKind.BOOLEAN);
  if (typeof value === "bigint") return Type.scalar(ValueKind.INTEGER);
  if (typeof value === "number") return Type.scalar(ValueKind.NUMBER);
  if (typeof value === "string") return Type.scalar(ValueKind.STRING);

  if (isRecord(value)) return inferMapping(value, path, walk);
  if (Array.isArray(value)) return inferList(value, path, walk);

  throw inferenceError(
    walk.bindingLabel,
    walk.bindingRange,
    `literal at ${path} has unsupported type ${reprText(typeName(value))}`,
    walk.diagnostics,
  );
}

/**
 * Infer a sorted closed record from one literal mapping. A record key is always text, so no
 * key needs checking.
 */
function inferMapping(value: Readonly<Record<string, unknown>>, path: string, walk: Walk): Type {
  const fields = new Map(
    Object.keys(value)
      .sort(compareCodeUnits)
      .map((key) => [key, inferLiteral(value[key], `${path}[${writeAsJson(key)}]`, walk)]),
  );
  return new Type({ kind: ValueKind.OBJECT, fields, required: new Set(fields.keys()) });
}

/** Infer one homogeneous structural list, widening only int/float scalars. */
function inferList(value: readonly unknown[], path: string, walk: Walk): Type {
  if (value.length === 0) {
    throw inferenceError(
      walk.bindingLabel,
      walk.bindingRange,
      `empty literal list at ${path} has no provable element type`,
      walk.diagnostics,
    );
  }
  const element = commonElementType(
    value.map((item, index) => inferLiteral(item, `${path}[${index}]`, walk)),
    path,
    walk,
  );
  return new Type({ kind: listKindForElement(element), element });
}

/** Return the common type up to `Optional`, or the scalar numeric widening, for a list. */
function commonElementType(elementTypes: readonly Type[], path: string, walk: Walk, noun = "literal list"): Type {
  // A generated parameter must have one concrete type, so unlike a template's output contract
  // (left unproven) a list with no common type fails the load.
  const bare = elementTypes.map((element) => element.replace({ nullable: false }));
  if (bare.every((element) => typeEquals(element, bare[0]!))) {
    return bare[0]!.replace({ nullable: elementTypes.some((element) => element.nullable) });
  }

  const integer = Type.scalar(ValueKind.INTEGER);
  const number = Type.scalar(ValueKind.NUMBER);
  if (elementTypes.every((element) => typeEquals(element, integer) || typeEquals(element, number))) return number;

  throw inferenceError(
    walk.bindingLabel,
    walk.bindingRange,
    `${noun} at ${path} has heterogeneous element types`,
    walk.diagnostics,
  );
}

/** Create the one located error shape used by strict inline inference. */
function inferenceError(
  bindingLabel: string,
  bindingRange: SourceRange | null,
  reason: string,
  diagnostics: LoadDiagnostics,
  cause?: unknown,
): LoadError {
  return diagnostics.error(
    `cannot infer type for inline binding ${reprText(bindingLabel)}: ${reason}; ${explicitTypeHint(bindingLabel)}`,
    { range: bindingRange, cause },
  );
}
