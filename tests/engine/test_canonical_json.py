"""Canonical JSON keeps one spelling for any nesting depth.

The C encoder spells shallow values; a value nested past its recursion limit falls back to an
iterative encoder whose output is byte-identical.
"""

import json
import math
from pathlib import Path
import random

import pytest

from yian.compile import manifest
from yian.compile.manifest import _canonical_json, _iterative_canonical_json, flow_manifest
from yian.compose import load_flow

_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED = sorted((_ROOT / "examples").glob("*.yaml")) + sorted(
    (_ROOT / "tests" / "seeds").glob("*.yaml")
)
_KEYS = ["", "a", "b", "B", "é", "日本", "\n", '"', "\\", "\x01", " ", "\ud800", "a\tb", "zz"]
_SCALARS = [
    "", "text", "ünï", "\x00\x1f", '"quoted"', "back\\slash", "😀", "\udfff",
    0, 1, -1, 2**63, -(2**70), 0.0, -0.0, 1.5, -2.25, 1e300, -1e-300, 5e-324, 1e16,
    True, False, None,
]


def _reference(value: object) -> str:
    """Spell `value` with the C encoder and the canonical options."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _random_value(rng: random.Random, depth: int) -> object:
    """Return a random value with `str` keys, nested at most `depth` containers deep."""
    shape = rng.randrange(4) if depth else 0
    if shape == 0:
        return rng.choice(_SCALARS)
    size = rng.randrange(5)
    if shape == 1:
        return [_random_value(rng, depth - 1) for _ in range(size)]
    if shape == 2:
        return tuple(_random_value(rng, depth - 1) for _ in range(size))
    return {rng.choice(_KEYS): _random_value(rng, depth - 1) for _ in range(size)}


def test_iterative_encoder_matches_the_c_encoder_on_random_values():
    rng = random.Random(20261016)
    for _ in range(2000):
        value = _random_value(rng, rng.randrange(7))
        assert _iterative_canonical_json(value) == _reference(value)


@pytest.mark.parametrize("path", _BUNDLED, ids=lambda path: path.name)
def test_iterative_encoder_matches_the_c_encoder_on_bundled_manifests(path):
    value = flow_manifest(load_flow(path.read_text(), search_paths=[path.parent]).compiled)
    assert _iterative_canonical_json(value) == _reference(value)


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
def test_iterative_encoder_rejects_non_finite_floats(number):
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        _reference({"a": [number]})
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        _iterative_canonical_json({"a": [number]})


def test_iterative_encoder_requires_str_keys():
    with pytest.raises(TypeError, match="keys must be str, not int"):
        _iterative_canonical_json({"a": {1: "one"}})


def test_a_value_past_the_c_encoders_depth_takes_the_iterative_encoder():
    value: list = []
    for _ in range(50_000):
        value = [value]
    with pytest.raises(RecursionError):
        _reference(value)
    assert _canonical_json(value) == "[" * 50_000 + "[]" + "]" * 50_000
    assert _canonical_json(value) == _iterative_canonical_json(value)


def test_a_shallow_value_never_takes_the_iterative_encoder(monkeypatch):
    def unexpected(value):
        raise AssertionError("the iterative encoder ran on a shallow value")

    monkeypatch.setattr(manifest, "_iterative_canonical_json", unexpected)
    value = {"é": [1, 2.5, None, True, {"b": "x", "a": ()}]}
    assert _canonical_json(value) == _reference(value)
