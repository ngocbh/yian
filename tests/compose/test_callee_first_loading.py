"""The loader builds `defs:` and loads `uses:` files callee-first, without recursing per level.

A def builds the defs it calls nested, up to a few def builds deep; past that, the defs
being built are set aside, the callee is built first, and each def set aside is built
again once the def it called is done. A file likewise loads the files it imports nested,
up to a few file loads deep, and past that each file set aside is loaded again once the
file it imported is loaded. Results are the same as building each callee nested inside
its caller: the same flow, the same call nesting limit failures, and the same error,
frames, and cause chain when a def or file in the chain fails.
"""

from collections import Counter
from pathlib import Path

import pytest

from tests.compose.test_call_nesting_limit import (
    _LEAF,
    _chain_texts,
    _defs_chain,
    _importing,
    _uses_files,
    _write_files,
)
from tests.deep_chains import lifted_limit, tight_stack
from yian.compile.manifest import flow_fingerprint
from yian.compose import LoadError, load_flow
from yian.compose import loader
from yian.compose.nodes.call import CallDescriptor

_HEAD = "id: f\nname: f\ninput:\n  x: !str\n"
_ROOT = (
    "nodes:\n"
    "  top:\n"
    "    kind: call\n"
    "    call: {target}\n"
    "    input:\n"
    "      x: ${{input.x}}\n"
    "output: ${{top.output}}\n"
)
# Calls a callable that is not declared, so the node fails when it is built.
_BAD = (
    "      bad:\n"
    "        kind: call\n"
    "        call: missing\n"
    "        input:\n"
    "          x: ${input.x}\n"
)
_ECHO = (
    "      step:\n"
    "        kind: code\n"
    "        code: tests.engine._compose_codefns:echo\n"
    "        input:\n"
    "          topic: ${input.x}\n"
    "        output: str\n"
)


def _step_calling(target: str) -> str:
    """Return a `step` node that calls `target` with `x`."""
    return (
        "      step:\n"
        "        kind: call\n"
        f"        call: {target}\n"
        "        input:\n"
        "          x: ${input.x}\n"
    )


def _def(name: str, nodes: str) -> str:
    """Return a `defs:` entry `name` over `nodes` whose output is `step`'s."""
    return (
        f"  {name}:\n    input:\n      x: !str\n    nodes:\n{nodes}    output: ${{step.output}}\n"
    )


def _flow(*defs: str, root: str = "a") -> str:
    """Return a flow over `defs` whose root `top` calls `root`."""
    return _HEAD + "defs:\n" + "".join(defs) + _ROOT.format(target=root)


def _chain_with_bad_def(n: int, bad: int) -> str:
    """Return `_defs_chain(n)` with a bad node added to def `c{bad}` after its call."""
    entries = []
    for i in range(n):
        if i < n - 1:
            nodes = _step_calling(f"c{i + 1}")
            entries.append(_def(f"c{i}", nodes + _BAD if i == bad else nodes))
        else:
            entries.append(f"  c{i}:\n{_LEAF}")
    return _flow(*entries, root="c0")


_BROKEN = _def("broken", _ECHO + _BAD)
_LEAF_DEF = _def("leaf", _ECHO)

_INLINE_CALLING_BROKEN = (
    "      step:\n"
    "        kind: call\n"
    "        input:\n"
    "          x: ${input.x}\n"
    "        nodes:\n"
    "          inner:\n"
    "            kind: call\n"
    "            call: broken\n"
    "            input:\n"
    "              x: ${input.x}\n"
    "        output: ${inner.output}\n"
)
_DIRECTIVE_CALLING_BROKEN = (
    "      step:\n"
    "        kind: code\n"
    "        code: tests.engine._compose_codefns:echo\n"
    "        input:\n"
    "          topic: call(broken, x=${input.x})\n"
    "        output: str\n"
)

_ERROR_CASES = {
    "bad def deep in a chain": _chain_with_bad_def(40, 30),
    "bad node before the call to a broken def": _flow(
        _def("a", _BAD + _step_calling("broken")), _BROKEN
    ),
    "bad node after the call to a broken def": _flow(
        _def("a", _step_calling("broken") + _BAD), _BROKEN
    ),
    "two-def cycle": _flow(_def("a", _step_calling("b")), _def("b", _step_calling("a"))),
    "self-call": _flow(_def("a", _step_calling("a"))),
    "broken def called from an inline body": _flow(_def("a", _INLINE_CALLING_BROKEN), _BROKEN),
    "broken def called through a call directive": _flow(
        _def("a", _DIRECTIVE_CALLING_BROKEN), _BROKEN
    ),
}

# Recorded from building each callee nested inside its caller.
_MAIN_ERRORS: dict[str, dict] = {
    "bad def deep in a chain": {
        "message": "".join(f"defs entry 'c{i}': " for i in range(31))
        + (
            "callable 'missing' is neither an in-file defs: callable nor a uses: alias — "
            "external flows are reachable only through a uses: alias (declare `uses: "
            "{missing: <ref>}`)"
        ),
        "notes": [],
        "frames": [
            ("<flow>", (7, 5, 16, 3), "definition 'c0'"),
            ("<flow>", (17, 5, 26, 3), "definition 'c1'"),
            ("<flow>", (27, 5, 36, 3), "definition 'c2'"),
            ("<flow>", (37, 5, 46, 3), "definition 'c3'"),
            ("<flow>", (47, 5, 56, 3), "definition 'c4'"),
            ("<flow>", (57, 5, 66, 3), "definition 'c5'"),
            ("<flow>", (67, 5, 76, 3), "definition 'c6'"),
            ("<flow>", (77, 5, 86, 3), "definition 'c7'"),
            ("<flow>", (87, 5, 96, 3), "definition 'c8'"),
            ("<flow>", (97, 5, 106, 3), "definition 'c9'"),
            ("<flow>", (107, 5, 116, 3), "definition 'c10'"),
            ("<flow>", (117, 5, 126, 3), "definition 'c11'"),
            ("<flow>", (127, 5, 136, 3), "definition 'c12'"),
            ("<flow>", (137, 5, 146, 3), "definition 'c13'"),
            ("<flow>", (147, 5, 156, 3), "definition 'c14'"),
            ("<flow>", (157, 5, 166, 3), "definition 'c15'"),
            ("<flow>", (167, 5, 176, 3), "definition 'c16'"),
            ("<flow>", (177, 5, 186, 3), "definition 'c17'"),
            ("<flow>", (187, 5, 196, 3), "definition 'c18'"),
            ("<flow>", (197, 5, 206, 3), "definition 'c19'"),
            ("<flow>", (207, 5, 216, 3), "definition 'c20'"),
            ("<flow>", (217, 5, 226, 3), "definition 'c21'"),
            ("<flow>", (227, 5, 236, 3), "definition 'c22'"),
            ("<flow>", (237, 5, 246, 3), "definition 'c23'"),
            ("<flow>", (247, 5, 256, 3), "definition 'c24'"),
            ("<flow>", (257, 5, 266, 3), "definition 'c25'"),
            ("<flow>", (267, 5, 276, 3), "definition 'c26'"),
            ("<flow>", (277, 5, 286, 3), "definition 'c27'"),
            ("<flow>", (287, 5, 296, 3), "definition 'c28'"),
            ("<flow>", (297, 5, 306, 3), "definition 'c29'"),
            ("<flow>", (307, 5, 321, 3), "definition 'c30'"),
            ("<flow>", (315, 7, 315, 10), None),
        ],
        "chain": [("LoadError", n, True) for n in range(32, 0, -1)] + [("LoadError", 0, False)],
    },
    "bad node before the call to a broken def": {
        "message": (
            "defs entry 'a': callable 'missing' is neither an in-file defs: callable nor a "
            "uses: alias — external flows are reachable only through a uses: alias "
            "(declare `uses: {missing: <ref>}`)"
        ),
        "notes": [],
        "frames": [
            ("<flow>", (7, 5, 21, 3), "definition 'a'"),
            ("<flow>", (10, 7, 10, 10), None),
        ],
        "chain": [
            ("LoadError", 2, True),
            ("LoadError", 1, True),
            ("LoadError", 0, False),
        ],
    },
    "bad node after the call to a broken def": {
        "message": (
            "defs entry 'a': defs entry 'broken': callable 'missing' is neither an in-file "
            "defs: callable nor a uses: alias — external flows are reachable only through "
            "a uses: alias (declare `uses: {missing: <ref>}`)"
        ),
        "notes": [],
        "frames": [
            ("<flow>", (7, 5, 21, 3), "definition 'a'"),
            ("<flow>", (22, 5, 37, 1), "definition 'broken'"),
            ("<flow>", (31, 7, 31, 10), None),
        ],
        "chain": [
            ("LoadError", 3, True),
            ("LoadError", 2, True),
            ("LoadError", 1, True),
            ("LoadError", 0, False),
        ],
    },
    "two-def cycle": {
        "message": (
            "defs entry 'a': defs entry 'b': recursive defs reference: callable 'a' is "
            "defined in terms of itself (the baked call graph must be finite)"
        ),
        "notes": [],
        "frames": [
            ("<flow>", (7, 5, 16, 3), "definition 'a'"),
            ("<flow>", (17, 5, 26, 1), "definition 'b'"),
            ("<flow>", (7, 5, 16, 3), None),
        ],
        "chain": [
            ("LoadError", 3, True),
            ("LoadError", 2, True),
            ("LoadError", 1, False),
        ],
    },
    "self-call": {
        "message": (
            "defs entry 'a': recursive defs reference: callable 'a' is defined in terms of "
            "itself (the baked call graph must be finite)"
        ),
        "notes": [],
        "frames": [
            ("<flow>", (7, 5, 16, 1), "definition 'a'"),
            ("<flow>", (7, 5, 16, 1), None),
        ],
        "chain": [
            ("LoadError", 2, True),
            ("LoadError", 1, False),
        ],
    },
    "broken def called from an inline body": {
        "message": (
            "defs entry 'a': defs entry 'broken': callable 'missing' is neither an in-file "
            "defs: callable nor a uses: alias — external flows are reachable only through "
            "a uses: alias (declare `uses: {missing: <ref>}`)"
        ),
        "notes": [],
        "frames": [
            ("<flow>", (7, 5, 22, 3), "definition 'a'"),
            ("<flow>", (23, 5, 38, 1), "definition 'broken'"),
            ("<flow>", (32, 7, 32, 10), None),
        ],
        "chain": [
            ("LoadError", 3, True),
            ("LoadError", 2, True),
            ("LoadError", 1, True),
            ("LoadError", 0, False),
        ],
    },
    "broken def called through a call directive": {
        "message": (
            "defs entry 'a': defs entry 'broken': callable 'missing' is neither an in-file "
            "defs: callable nor a uses: alias — external flows are reachable only through "
            "a uses: alias (declare `uses: {missing: <ref>}`)"
        ),
        "notes": [],
        "frames": [
            ("<flow>", (7, 5, 17, 3), "definition 'a'"),
            ("<flow>", (18, 5, 33, 1), "definition 'broken'"),
            ("<flow>", (27, 7, 27, 10), None),
        ],
        "chain": [
            ("LoadError", 3, True),
            ("LoadError", 2, True),
            ("LoadError", 1, True),
            ("LoadError", 0, False),
        ],
    },
}


def _frames(error: LoadError) -> list[tuple]:
    """Return each frame of `error` as `(label, (start/end line and col), note)`."""
    return [
        (
            frame.label,
            (
                None
                if frame.range is None
                else (
                    frame.range.start_line,
                    frame.range.start_col,
                    frame.range.end_line,
                    frame.range.end_col,
                )
            ),
            frame.note,
        )
        for frame in error.diagnostic.frames
    ]


def _chain(error: BaseException) -> list[tuple]:
    """Return `(type, frame count, cause is context)` for `error` and each exception it chains to.

    The walk follows `__cause__`, else `__context__`.
    """
    found: list[tuple] = []
    seen: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in seen:
        seen.append(current)
        diagnostic = getattr(current, "diagnostic", None)
        found.append(
            (
                type(current).__name__,
                None if diagnostic is None else len(diagnostic.frames),
                current.__cause__ is not None and current.__cause__ is current.__context__,
            )
        )
        current = current.__cause__ or current.__context__
    return found


def _describe(error: LoadError) -> dict:
    """Return the message, notes, frames, and cause chain of `error`."""
    return {
        "message": str(error),
        "notes": list(error.diagnostic.notes),
        "frames": _frames(error),
        "chain": _chain(error),
    }


def _load_error(text: str) -> LoadError:
    """Load `text`, which must fail, and return its `LoadError`."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


@pytest.mark.parametrize("reverse", [False, True], ids=["caller-first", "callee-first"])
def test_a_long_def_chain_loads_without_recursing_per_def(reverse: bool) -> None:
    text = _defs_chain(500, reverse=reverse)
    with lifted_limit(), tight_stack(300):
        flow = load_flow(text)
    assert set(flow.compiled.nodes) >= {"top"}


_MAIN_FINGERPRINT_40 = "ce767b2ced9783ec378208400474e3ea21892f0784effc2a830a3c4ecf4abc69"
_PAST_LIMIT = (
    " past the call nesting limit of 50 (each call, map, or loop body is a level); flatten "
    "the chain"
)


def test_a_def_chain_loads_to_the_same_flow() -> None:
    flow = load_flow(_defs_chain(40))
    assert flow_fingerprint(flow.compiled) == _MAIN_FINGERPRINT_40


def test_a_chain_written_caller_first_still_fails_at_the_first_call_past_the_limit() -> None:
    error = _load_error(_defs_chain(97))
    assert str(error) == "".join(f"defs entry 'c{i}': " for i in range(51)) + (
        "call node 'step': calling callable 'c51' nests calls more than 50 levels deep,"
        + _PAST_LIMIT
    )
    frames = _frames(error)
    assert len(frames) == 52
    assert frames[-1] == ("<flow>", (512, 15, 512, 18), None)


def test_a_chain_written_callee_first_still_fails_from_built_callables() -> None:
    error = _load_error(_defs_chain(97, reverse=True))
    assert str(error) == (
        "defs entry 'c45': call node 'step': calling callable 'c46' nests calls 51 levels deep,"
        + _PAST_LIMIT
    )
    assert _frames(error) == [
        ("<flow>", (518, 5, 527, 3), "definition 'c45'"),
        ("<flow>", (523, 15, 523, 18), None),
    ]


@pytest.mark.parametrize("case", list(_ERROR_CASES))
def test_a_failing_def_reports_the_same_error(case: str) -> None:
    assert _describe(_load_error(_ERROR_CASES[case])) == _MAIN_ERRORS[case]


def _raise_type_error_calling(target: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make building any call node that calls `target` raise a plain `TypeError`."""
    build = CallDescriptor.build

    def failing_build(self, ctx):
        if self.call == target:
            raise TypeError("cannot build this call")
        return build(self, ctx)

    monkeypatch.setattr(CallDescriptor, "build", failing_build)


_PLAIN_ERROR_TEXT = _flow(
    _def("a", _step_calling("b")), _def("b", _step_calling("leaf")), _LEAF_DEF
)
_MAIN_PLAIN_ERROR: dict = {
    "message": (
        "defs entry 'a': call node 'step': could not resolve callable 'b': cannot "
        "build this call"
    ),
    "notes": [],
    "frames": [
        ("<flow>", (7, 5, 16, 3), "definition 'a'"),
        ("<flow>", (12, 15, 12, 16), None),
    ],
    "chain": [
        ("LoadError", 2, True),
        ("LoadError", 1, True),
        ("TypeError", None, False),
    ],
}


def test_a_callee_that_raises_a_plain_exception_could_not_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raise_type_error_calling("leaf", monkeypatch)
    error = _load_error(_PLAIN_ERROR_TEXT)
    assert _describe(error) == _MAIN_PLAIN_ERROR
    assert type(error.__cause__.__cause__) is TypeError


def _count_def_builds(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count each build of a def by name, from now until the test ends."""
    builds: Counter[str] = Counter()
    load_def = loader._load_def

    def counting_load_def(name, *args, **kwargs):
        builds[name] += 1
        return load_def(name, *args, **kwargs)

    monkeypatch.setattr(loader, "_load_def", counting_load_def)
    return builds


@pytest.mark.parametrize("reverse", [False, True], ids=["caller-first", "callee-first"])
def test_each_def_in_a_long_chain_is_built_at_most_twice(
    reverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = _count_def_builds(monkeypatch)
    with lifted_limit():
        load_flow(_defs_chain(500, reverse=reverse))

    assert set(builds) == {f"c{i}" for i in range(500)}
    assert max(builds.values()) == (1 if reverse else 2)


def _calls(targets: list[str], first: int = 0) -> str:
    """Return one call node per target, `s{first}` calling `targets[0]` and so on."""
    return "".join(
        f"      s{k}:\n"
        "        kind: call\n"
        f"        call: {target}\n"
        "        input:\n"
        "          x: ${input.x}\n"
        for k, target in enumerate(targets, start=first)
    )


def test_a_def_calling_many_defs_written_after_it_is_built_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaves = [f"l{i}" for i in range(400)]
    orchestrator = _def("a", _calls(leaves) + _step_calling("l0"))
    builds = _count_def_builds(monkeypatch)
    load_flow(_flow(orchestrator, *(_def(leaf, _ECHO) for leaf in leaves)))

    assert dict(builds) == {"a": 1, **{leaf: 1 for leaf in leaves}}


def _chain_with_leaves(n: int, broken: str) -> str:
    """Return a chain `c0` .. `c{n-1}` whose defs each also call five leaves written after.

    Def `ci` calls `ci_l0` .. `ci_l2`, then `c(i+1)`, then `ci_l3` and `ci_l4`; the last
    def echoes instead of calling on. Leaf `broken` is a broken def.
    """
    chain = []
    leaves = []
    for i in range(n):
        own = [f"c{i}_l{j}" for j in range(5)]
        step = _step_calling(f"c{i + 1}") if i < n - 1 else _ECHO
        chain.append(_def(f"c{i}", _calls(own[:3]) + step + _calls(own[3:], first=3)))
        leaves.extend(_def(leaf, _ECHO + _BAD if leaf == broken else _ECHO) for leaf in own)
    return _flow(*chain, *leaves, root="c0")


# Recorded from building each callee nested inside its caller.
_MAIN_CHAIN_WITH_LEAVES_ERROR: dict = {
    "message": "".join(f"defs entry 'c{i}': " for i in range(25))
    + (
        "defs entry 'c24_l3': callable 'missing' is neither an in-file defs: callable nor a "
        "uses: alias — external flows are reachable only through a uses: alias (declare "
        "`uses: {missing: <ref>}`)"
    ),
    "notes": [],
    "frames": [
        ("<flow>", (7 + 35 * i, 5, 41 + 35 * i, 3), f"definition 'c{i}'") for i in range(25)
    ]
    + [
        ("<flow>", (2411, 5, 2426, 3), "definition 'c24_l3'"),
        ("<flow>", (2420, 7, 2420, 10), None),
    ],
    "chain": [("LoadError", n, True) for n in range(27, 0, -1)] + [("LoadError", 0, False)],
}


def test_a_broken_leaf_deep_in_a_chain_whose_defs_call_leaves_reports_the_same_error() -> None:
    error = _load_error(_chain_with_leaves(30, broken="c24_l3"))
    assert _describe(error) == _MAIN_CHAIN_WITH_LEAVES_ERROR


_SHORT_THEN_LONG_CHAIN = _flow(
    _def("a", _calls(["x0"]) + _step_calling("b0")),
    *(_def(f"x{i}", _step_calling(f"x{i + 1}") if i < 4 else _ECHO) for i in range(5)),
    *(_def(f"b{i}", _step_calling(f"b{i + 1}" if i < 45 else "x0")) for i in range(46)),
)
# Recorded from building each callee nested inside its caller.
_MAIN_SHORT_THEN_LONG_CHAIN_ERROR: dict = {
    "message": "defs entry 'a': "
    + "".join(f"defs entry 'b{i}': " for i in range(46))
    + "call node 'step': calling callable 'x0' nests calls 51 levels deep,"
    + _PAST_LIMIT,
    "notes": [],
    "frames": [("<flow>", (7, 5, 21, 3), "definition 'a'")]
    + [("<flow>", (73 + 10 * i, 5, 82 + 10 * i, 3), f"definition 'b{i}'") for i in range(45)]
    + [
        ("<flow>", (523, 5, 532, 1), "definition 'b45'"),
        ("<flow>", (528, 15, 528, 17), None),
    ],
    "chain": [("LoadError", n, True) for n in range(48, 1, -1)] + [("LoadError", 1, False)],
}


def test_a_def_built_before_a_deep_call_is_still_built_when_called_past_the_limit() -> None:
    # `a` builds `x0` nested, then its chain from `b0` nests too deep and is set aside,
    # which rolls `x0` back; `b45` must find `x0` built, as a nested build does.
    error = _load_error(_SHORT_THEN_LONG_CHAIN)
    assert _describe(error) == _MAIN_SHORT_THEN_LONG_CHAIN_ERROR


def test_a_long_uses_chain_loads_without_recursing_per_file(tmp_path: Path) -> None:
    text = _uses_files(tmp_path, 100)
    with lifted_limit(), tight_stack(300):
        flow = load_flow(text, search_paths=[tmp_path])
    assert flow.import_depth == 100


_MAIN_USES_FINGERPRINT_30 = "b37d012d60f751a987fb9757aea411ef244a4f1cae9a699fa78d64104b6860cc"


def test_a_uses_chain_loads_to_the_same_flow(tmp_path: Path) -> None:
    flow = load_flow(_uses_files(tmp_path, 30), search_paths=[tmp_path])
    assert flow_fingerprint(flow.compiled) == _MAIN_USES_FINGERPRINT_30


_USES_ROOT = (
    "id: root\nname: root\ninput:\n  x: !str\nuses:\n  first: {first}\nnodes:\n"
    "  top:\n"
    "    kind: call\n"
    "    call: first\n"
    "    input:\n"
    "      x: ${{input.x}}\n"
    "output: ${{top.output}}\n"
)


def _count_file_loads(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count each load of a flow document by label, from now until the test ends."""
    loads: Counter[str] = Counter()
    load = loader._load_flow

    def counting_load(document, *args, **kwargs):
        loads[document.label] += 1
        return load(document, *args, **kwargs)

    monkeypatch.setattr(loader, "_load_flow", counting_load)
    return loads


def test_each_file_is_parsed_once_though_its_load_is_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parses: Counter[str] = Counter()
    parse = loader.parse_source_document

    def counting_parse(text, label, *args, **kwargs):
        parses[label] += 1
        return parse(text, label, *args, **kwargs)

    monkeypatch.setattr(loader, "parse_source_document", counting_parse)
    loads = _count_file_loads(monkeypatch)
    load_flow(_uses_files(tmp_path, 30), search_paths=[tmp_path])

    assert dict(parses) == {"<flow>": 1, **{f"u{i}.yaml": 1 for i in range(30)}}
    # `u27` is the last file set aside, and it loads `u28` and `u29` nested; each file
    # set aside before it is loaded again once its import is loaded.
    assert dict(loads) == {"<flow>": 1, **{f"u{i}.yaml": 2 if i < 27 else 1 for i in range(30)}}


def test_a_file_importing_many_files_is_loaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aliases = {f"a{i}": f"leaf{i}" for i in range(150)}
    texts = {"hub": _importing("hub", aliases)}
    texts.update((leaf, _importing(leaf, {})) for leaf in aliases.values())
    _write_files(tmp_path, texts)
    loads = _count_file_loads(monkeypatch)
    rebases: Counter[str] = Counter()
    rebase = loader._rebase_external_child

    def counting_rebase(alias, child):
        rebases[alias] += 1
        return rebase(alias, child)

    monkeypatch.setattr(loader, "_rebase_external_child", counting_rebase)
    load_flow(_USES_ROOT.format(first="hub"), search_paths=[tmp_path])

    assert dict(loads) == {
        "<flow>": 1,
        "hub.yaml": 1,
        **{f"{leaf}.yaml": 1 for leaf in aliases.values()},
    }
    assert dict(rebases) == {"first": 1, **{alias: 1 for alias in aliases}}


def test_a_file_loaded_before_a_deep_import_is_not_loaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `a` loads the chain from `x0` nested, then its chain from `b0` nests too deep and
    # is set aside; `b19` and `a` loaded again find `x0` loaded, as nested loads do.
    texts = {"a": _importing("a", {"short": "x0", "long": "b0"})}
    texts.update(_chain_texts(5, prefix="x", calls=False))
    texts.update(_chain_texts(20, prefix="b", calls=False, last_imports="x0"))
    _write_files(tmp_path, texts)
    loads = _count_file_loads(monkeypatch)
    flow = load_flow(_USES_ROOT.format(first="a"), search_paths=[tmp_path])

    assert flow.import_depth == 26
    assert {label: loads[label] for label in loads if label.startswith("x")} == {
        f"x{i}.yaml": 1 for i in range(5)
    }
    assert loads["a.yaml"] == 2


def _broken_file_chain() -> dict[str, str]:
    """Return `_chain_texts(25)` with `u20` calling an undeclared callable after its import."""
    texts = _chain_texts(25)
    texts["u20"] = texts["u20"].replace("call: nxt", "call: missing")
    return texts


def _mismatched_version_chain() -> dict[str, str]:
    """Return `_chain_texts(15)` with `u9` pinning a version `u10` does not declare."""
    texts = _chain_texts(15)
    texts["u9"] = texts["u9"].replace("nxt: u10", "nxt: u10@2")
    return texts


_USES_ERROR_CASES = {
    "broken file 20 deep": (_broken_file_chain, "u0"),
    "three-file cycle": (lambda: _chain_texts(3, prefix="k", last_imports="k0"), "k0"),
    "pinned version mismatch 10 deep": (_mismatched_version_chain, "u0"),
}


def _alias_frames(labels: list[str], last_end_col: int) -> list[tuple]:
    """Return the root's `first` alias frame, then each of `labels`' `nxt` alias frames.

    The `nxt` value ends at column 10 for one-digit targets and 11 for two-digit ones;
    the last file's ends at `last_end_col`.
    """
    frames = [("<flow>", (6, 10, 6, 12), "uses alias 'first'")]
    for index, label in enumerate(labels):
        target = int(label[1:-5]) + 1
        end_col = last_end_col if index == len(labels) - 1 else (10 if target < 10 else 11)
        frames.append((label, (6, 8, 6, end_col), "uses alias 'nxt'"))
    return frames


def _located_chain(frames: int) -> list[tuple]:
    """Return the cause chain of an error re-located once per frame over an unlocated one."""
    return [("LoadError", n, True) for n in range(frames, 0, -1)] + [("LoadError", 0, False)]


_UNDECLARED_MISSING = (
    "callable 'missing' is neither an in-file defs: callable nor a uses: alias — external "
    "flows are reachable only through a uses: alias (declare `uses: {missing: <ref>}`)"
)

# Recorded from loading each import nested inside its importer; `<dir>` is the directory.
_MAIN_USES_ERRORS: dict[str, dict] = {
    "broken file 20 deep": {
        "message": _UNDECLARED_MISSING,
        "notes": [],
        "frames": _alias_frames([f"u{i}.yaml" for i in range(20)], 11)
        + [("u20.yaml", (8, 3, 8, 7), None)],
        "chain": _located_chain(22),
    },
    "three-file cycle": {
        "message": "flow-reference cycle across files: <dir>/k0.yaml",
        "notes": [],
        "frames": _alias_frames(["k0.yaml", "k1.yaml", "k2.yaml"], 10),
        "chain": _located_chain(4),
    },
    "pinned version mismatch 10 deep": {
        "message": (
            "uses: ref pinned u10@2 but the resolved file declares version: None "
            "(file-per-version: name a file that declares '2')"
        ),
        "notes": [],
        "frames": _alias_frames([f"u{i}.yaml" for i in range(10)], 13),
        "chain": _located_chain(11),
    },
}


@pytest.mark.parametrize("case", list(_USES_ERROR_CASES))
def test_a_failing_uses_file_reports_the_same_error(case: str, tmp_path: Path) -> None:
    texts, first = _USES_ERROR_CASES[case]
    directory = tmp_path.resolve()
    _write_files(directory, texts())
    with pytest.raises(LoadError) as caught:
        load_flow(_USES_ROOT.format(first=first), search_paths=[directory])
    described = _describe(caught.value)
    described["message"] = described["message"].replace(str(directory), "<dir>")
    assert described == _MAIN_USES_ERRORS[case]
