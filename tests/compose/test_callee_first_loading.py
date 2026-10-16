"""The loader builds `defs:` callee-first, so a def chain loads without recursing per def.

A def that calls a def not built yet is set aside, the callee is built first, and the
caller is built again. Results are the same as building each callee nested inside its
caller: the same flow, the same call nesting limit failures, and the same error, frames,
and cause chain when a def in the chain fails.
"""

from collections import Counter

import pytest

from tests.compose.test_call_nesting_limit import _LEAF, _defs_chain
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


@pytest.mark.parametrize("reverse", [False, True], ids=["caller-first", "callee-first"])
def test_each_def_is_built_again_once_per_callee_not_built_yet(
    reverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: Counter[str] = Counter()
    load_def = loader._load_def

    def counting_load_def(name, *args, **kwargs):
        builds[name] += 1
        return load_def(name, *args, **kwargs)

    monkeypatch.setattr(loader, "_load_def", counting_load_def)
    with lifted_limit():
        load_flow(_defs_chain(500, reverse=reverse))

    expected = {f"c{i}": 1 if reverse or i == 499 else 2 for i in range(500)}
    assert dict(builds) == expected
