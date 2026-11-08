/**
 * The type system's public surface: typed values, the type language and the variable pool.
 *
 * Every typed value, a date or datetime included, is a `TypedValue`. `StorageKind`, `Value`,
 * `ValueRecord` and `PoolWire` name the shapes `TypedValue` and `VariablePool` use in their
 * public signatures. `isRecord` and `isList` are the plain-object and plain-array tests
 * `buildValueAs` uses, so callers read a record or list exactly as the type system does, and
 * `jsTypeName` names a rejected value's type the same way; `typeName` names it as messages
 * spell types (`int`, `str`, `NoneType`).
 */

export type { PoolWire } from "./pool.ts";
export { VariablePool } from "./pool.ts";
export type { RecordDef, TypeRegistry, VariantDef } from "./types.ts";
export {
  BUILTIN_TYPE_NAMES,
  parseType,
  typeName,
  resolveType,
  typeFor,
} from "./types.ts";
export type {
  FieldAbsent,
  FieldPartial,
  FieldRead,
  StorageKind,
  Value,
  ValueRecord,
} from "./values.ts";
export {
  buildValue,
  buildValueAs,
  canonicalTypeLabel,
  canonicalUnionOrder,
  FileRef,
  InboxRef,
  inboxName,
  inboxType,
  isInboxType,
  isList,
  isMemoryType,
  isRecord,
  jsTypeName,
  listKindForElement,
  MemoryRef,
  memoryType,
  normalizeFiles,
  restrictUnion,
  Type,
  TypeCheckError,
  TypedValue,
  typesOverlap,
  unionField,
  unionMemberForTag,
  unionTagFields,
  ValueKind,
} from "./values.ts";
