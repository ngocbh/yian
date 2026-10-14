"""Bundled YAML, docs, skills, and load messages never spell the old type forms.

The old forms are load errors now: a bracket type (`Optional[str]`), a binding or
declaration written `name: T = value`, and an untagged flow input or agent variable.
This test walks every bundled YAML file and every ```yaml block in the docs, the
READMEs, the project skills, and `src/yian` docstrings, and fails on the first old
form. A block that shows an old form on purpose starts with `# old form`. It also
scans the message strings in `src/yian/compose` and `src/yian/typesys/types.py` for
hints that suggest an old form; docstrings, where `Optional[str]` is a Python
annotation, are not scanned.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Iterator
from pathlib import Path

from yian.compose.source import SourceCursor, SourceDocument, SourceSyntaxError
from yian.typesys import TypeCheckError
from yian.typesys.types import RefExpr, parse_type
from yian.typesys.types import _parse_legacy_type

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_GLOBS = (
    "tests/seeds/**/*.yaml",
    "examples/**/*.yaml",
    ".claude/skills/**/*.yaml",
    "src/yian/**/*.yaml",
)
# Design plans under docs/plans are repository history, not reader documentation.
DOC_GLOBS = ("docs/*.md", "README.md", "src/yian/README.md", ".claude/skills/**/*.md")
MESSAGE_GLOBS = ("src/yian/compose/**/*.py", "src/yian/typesys/types.py")
OLD_FORM_MARKER = "# old form"

_FENCE = re.compile(r"^([ \t]*)```ya?ml[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
_SPAN = re.compile(r"\$\{[^}]*\}")
_BRACKET_TYPE = re.compile(r"\b(Optional|List|list|Literal|Union|Dict|dict|int|float)\[[\w\"'-]")
# A `name: T = value` line, for a block YAML can't parse.
_OLD_LINE = re.compile(r"^\s*(?:- )?[\w.-]+:\s+([A-Za-z_][^#=\"']*?) =(?: |$)")
_BUILTIN_NAMES = frozenset(
    {"str", "int", "float", "bool", "date", "datetime", "object", "file", "None", "none",
     "Any", "any", "list", "dict", "List", "Dict", "Optional", "Literal", "Union"}
)

# Message hints that suggest an old form; a formatted value reads as `X`.
_MESSAGE_PATTERNS = (
    (re.compile(r"(?:^|[`\s'\"])[\w.X]+: [\w\[\](),.|X]+ = "), "a `name: T = value` hint"),
    (re.compile(r"Optional\["), "a bracket type"),
    (re.compile(r"!\w+\["), "a bracket inside a tag"),
)


def _files(globs: tuple[str, ...]) -> list[Path]:
    """Return the files matching `globs`, sorted and without duplicates."""
    return sorted({path for pattern in globs for path in REPO_ROOT.glob(pattern)})


def _yaml_blocks(text: str) -> Iterator[tuple[int, str]]:
    """Yield `(line, block)` for each ```yaml block of a Markdown or docstring text."""
    for match in _FENCE.finditer(text):
        block = textwrap.dedent(match.group(2))
        if block.lstrip().startswith(OLD_FORM_MARKER):
            continue
        yield text.count("\n", 0, match.start(2)) + 1, block


def _docstrings(path: Path) -> Iterator[tuple[int, str]]:
    """Yield `(line, text)` for each docstring of a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                yield node.body[0].lineno, doc


def _looks_like_type(text: str) -> bool:
    """Whether `text` reads as a type: builtin names, or capitalized typedef names."""
    for parse in (parse_type, _parse_legacy_type):
        try:
            expr = parse(text)
        except TypeCheckError:
            continue
        names = re.findall(r"[A-Za-z_]\w*", text)
        if isinstance(expr, RefExpr) and not expr.name[:1].isupper():
            return False
        return all(name in _BUILTIN_NAMES or name[:1].isupper() for name in names[:1])
    return False


def _old_text(value: str) -> bool:
    """Whether plain text is the old `T = value` form."""
    if " = " in value:
        left = value.split(" = ", 1)[0]
    elif value.endswith(" ="):
        left = value[:-2]
    else:
        return False
    return not left.lstrip().startswith("call(") and _looks_like_type(left.strip())


def _walk(cursor: SourceCursor, path: str) -> Iterator[tuple[str, SourceCursor]]:
    """Yield `(path, cursor)` for `cursor` and every node below it."""
    yield path, cursor
    value = cursor.value
    if type(value) is dict:
        for key, child in zip(value, cursor.children()):
            yield from _walk(child, f"{path}.{key}" if path else str(key))
    elif type(value) is list:
        for index, child in enumerate(cursor.children()):
            yield from _walk(child, f"{path}[{index}]")


def _declaration_sections(root: SourceCursor) -> Iterator[tuple[str, SourceCursor, bool]]:
    """Yield `(where, mapping, needs_default)` for each input or variable declaration map.

    A flow's and each `defs:` entry's `input:` declare parameters; an agent node's
    `variables:` declare variables, wherever the node sits.
    """
    def flow_inputs(flow: SourceCursor, where: str):
        if type(flow.value) is dict and flow.has("input") and (
            flow.has("nodes") or flow.has("kind")
        ):
            yield f"{where}input", flow.child("input"), False

    if type(root.value) is not dict:
        return
    if root.has("id") or root.has("name") or root.has("typedefs") or root.has("defs"):
        yield from flow_inputs(root, "")
    if root.has("defs") and type(root.child("defs").value) is dict:
        defs = root.child("defs")
        for key, child in zip(defs.value, defs.children()):
            yield from flow_inputs(child, f"defs.{key}.")
    for where, cursor in _walk(root, ""):
        if (
            type(cursor.value) is dict
            and cursor.has("variables")
            and cursor.has("kind")
            and cursor.value.get("kind") == "agent"
        ):
            yield f"{where}.variables".lstrip("."), cursor.child("variables"), True


def _old_forms(text: str, label: str) -> list[str]:
    """Return `label:line: problem` for each old type form in one YAML text."""
    found: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        code = _SPAN.sub("", line.split(" #", 1)[0])
        if code.lstrip().startswith("#"):
            continue
        if _BRACKET_TYPE.search(code):
            found.append(f"{label}:{number}: bracket type: {line.strip()}")
    try:
        root = SourceDocument.parse(text, label).root
    except SourceSyntaxError:
        for number, line in enumerate(text.splitlines(), 1):
            match = _OLD_LINE.match(line)
            if match and _looks_like_type(match.group(1).strip()):
                found.append(f"{label}:{number}: `name: T = value`: {line.strip()}")
        return found

    def at(cursor: SourceCursor) -> int | str:
        return cursor.range.start_line if cursor.range is not None else "?"

    for where, cursor in _walk(root, ""):
        value = cursor.value
        if (
            isinstance(value, str)
            and cursor.tag is None
            and cursor.scalar_style == "plain"
            and _old_text(value)
        ):
            found.append(f"{label}:{at(cursor)}: `name: T = value` at {where}: {value!r}")
    for where, section, _needs_default in _declaration_sections(root):
        if type(section.value) is not dict:
            continue
        for name, entry in zip(section.value, section.children()):
            # `input: { ... }` elides the declarations.
            if entry.tag is None and name != "...":
                found.append(f"{label}:{at(entry)}: untagged declaration {where}.{name}")
    return found


def _yaml_sources() -> Iterator[tuple[str, str]]:
    """Yield `(label, text)` for every bundled YAML file and scanned ```yaml block."""
    for path in _files(BUNDLED_GLOBS):
        yield str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8")
    for path in _files(DOC_GLOBS):
        for line, block in _yaml_blocks(path.read_text(encoding="utf-8")):
            yield f"{path.relative_to(REPO_ROOT)}:{line}", block
    for path in _files(("src/yian/**/*.py",)):
        for start, doc in _docstrings(path):
            for line, block in _yaml_blocks(doc):
                yield f"{path.relative_to(REPO_ROOT)}:{start + line - 1}", block


def _message_text(node: ast.AST) -> str:
    """Spell a string constant or f-string, each formatted value as `X`."""
    if isinstance(node, ast.Constant):
        return node.value
    return "".join(
        part.value if isinstance(part, ast.Constant) else "X"
        for part in node.values  # type: ignore[attr-defined]
    )


def _message_hints(path: Path) -> list[str]:
    """Return `path:line: problem` for each message string with an old-form hint."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    parts_of_fstrings = set()
    annotations = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts_of_fstrings.update(id(part) for part in node.values)
        # A quoted annotation (`"Optional[X]"`) is Python, not a message.
        for annotation in (
            getattr(node, "annotation", None),
            getattr(node, "returns", None),
        ):
            if annotation is not None:
                annotations.update(id(part) for part in ast.walk(annotation))
    found: list[str] = []
    for node in ast.walk(tree):
        is_text = isinstance(node, ast.Constant) and isinstance(node.value, str)
        if not (is_text or isinstance(node, ast.JoinedStr)):
            continue
        if id(node) in docstrings or id(node) in parts_of_fstrings or id(node) in annotations:
            continue
        text = _message_text(node)
        for pattern, problem in _MESSAGE_PATTERNS:
            if pattern.search(text):
                found.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {problem}: {text!r}")
    return found


# ---------- the scanners themselves ----------


def test_scanner_flags_each_old_form():
    text = (
        "id: f\n"
        "typedefs:\n"
        "  Buy:\n"
        "    action: Literal[buy]\n"
        "input:\n"
        "  qty: !int\n"
        "  old: int\n"
        "nodes:\n"
        "  ask:\n"
        "    kind: agent\n"
        "    variables:\n"
        "      n: !int 4\n"
        "      m: int = 4\n"
        "    input:\n"
        "      q: Buy = ${x}\n"
        "      text: a = b\n"
        "      quoted: \"int = 5\"\n"
        "      code: |\n"
        "        n: int = 3\n"
        "    output: str\n"
        "output: ${ask.output}\n"
    )
    assert _old_forms(text, "f.yaml") == [
        "f.yaml:4: bracket type: action: Literal[buy]",
        "f.yaml:13: `name: T = value` at nodes.ask.variables.m: 'int = 4'",
        "f.yaml:15: `name: T = value` at nodes.ask.input.q: 'Buy = ${x}'",
        "f.yaml:7: untagged declaration input.old",
        "f.yaml:13: untagged declaration nodes.ask.variables.m",
    ]


def test_scanner_reads_unparsable_blocks_line_by_line():
    assert _old_forms("x:\n  qty: int = ${a}\n  - oops\n", "b") == [
        "b:2: `name: T = value`: qty: int = ${a}"
    ]


def test_old_form_marker_exempts_a_block():
    text = "```yaml\n# old form\nqty: int = ${x}\n```\n\n```yaml\nqty: !int ${x}\n```\n"
    assert list(_yaml_blocks(text)) == [(7, "qty: !int ${x}\n")]


def test_message_scanner_flags_old_hints_but_not_docstrings(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(
        '"""Return `Optional[str]`."""\n'
        'def f(label, typ: "Optional[str]") -> "Optional[str]":\n'
        '    """`x: int = 3` in a docstring."""\n'
        '    a = f"write `{label}: {typ} = {label}`"\n'
        '    b = "add `output: Optional[...]`"\n'
        '    c = f"`!{typ}[int]`"\n'
        '    d = f"write `{label}: !{typ} {label}`"\n'
        "    return a, b, c, d\n"
    )
    problems = ("a `name: T = value` hint", "a bracket type", "a bracket inside a tag")
    found = [
        next(problem for problem in problems if f": {problem}: " in line)
        for line in _message_hints_at(module, tmp_path)
    ]
    assert found == list(problems)


def _message_hints_at(path: Path, root: Path) -> list[str]:
    """Run `_message_hints` on a file outside the repository."""
    global REPO_ROOT
    saved, REPO_ROOT = REPO_ROOT, root
    try:
        return sorted(_message_hints(path), key=lambda line: int(line.split(":")[1]))
    finally:
        REPO_ROOT = saved


# ---------- the scans ----------


def test_bundled_yaml_and_docs_spell_no_old_type_form():
    sources = list(_yaml_sources())
    assert sources, f"no YAML found under {REPO_ROOT}"
    found = [problem for label, text in sources for problem in _old_forms(text, label)]
    assert not found, (
        "old type forms are load errors; write tags and parenthesized types "
        f"(or start a deliberate example with `{OLD_FORM_MARKER}`):\n" + "\n".join(found)
    )


def test_load_messages_suggest_no_old_type_form():
    found = [problem for path in _files(MESSAGE_GLOBS) for problem in _message_hints(path)]
    assert not found, "messages must suggest tags and parentheses:\n" + "\n".join(found)
