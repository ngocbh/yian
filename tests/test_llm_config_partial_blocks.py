"""Provider blocks that hold values: key paths and literal leaves checked, values left opaque."""

import pytest

SENTINEL = "sk-sentinel-5f0c2a9e"


class _Value:
    """Stands in for a value (`${...}`) in a block, resolved only later."""


def _is_value(leaf) -> bool:
    return isinstance(leaf, _Value)


def _problems(block, tree):
    from yian.llm_clients.provider_blocks import partial_block_problems

    return partial_block_problems(block, tree, _is_value)


def test_a_block_of_values_and_valid_literals_has_no_problems():
    tree = {"max_tokens": _Value(), "temperature": 0.5, "thinking": {"type": "enabled",
            "budget_tokens": 2048}}

    assert _problems("anthropic", tree) == []


def test_unknown_and_reserved_key_paths_are_reported_beside_values():
    tree = {"temprature": _Value(), "stream": True, "max_tokens": _Value()}

    problems = _problems("anthropic", tree)

    assert len(problems) == 2
    assert problems[0].startswith(
        "`anthropic.temprature` is not a supported field of the Anthropic Messages API request;"
    )
    assert problems[1] == "`anthropic.stream` is built by yian (streaming); remove it"


def test_a_reserved_nested_path_is_reported():
    problems = _problems("anthropic", {"output_config": {"format": _Value(), "effort": "high"}})

    assert problems == [
        "`anthropic.output_config.format` is built by yian (structured output); remove it"
    ]


def test_an_unknown_nested_key_is_reported_beside_a_value():
    problems = _problems("google", {"generationConfig": {"topK": _Value(), "topk": 3}})

    assert len(problems) == 1
    assert problems[0].startswith("`google.generationConfig.topk` is not a supported field")


def test_a_value_leaf_is_opaque_even_where_a_block_is_expected():
    assert _problems("anthropic", {"thinking": _Value(), "metadata": _Value()}) == []


def test_a_literal_leaf_is_checked_against_its_field():
    tree = {"thinking": {"type": "enabled", "budget_tokens": 10}, "temperature": _Value()}

    assert _problems("anthropic", tree) == [
        "`anthropic.thinking.budget_tokens`: Input should be greater than or equal to 1024"
    ]


def test_literal_leaves_are_checked_by_strict_type_and_literal():
    tree = {"max_tokens": "8000", "service_tier": "fastest", "top_k": _Value()}

    problems = _problems("anthropic", tree)

    assert problems == [
        "`anthropic.max_tokens`: Input should be a valid integer",
        "`anthropic.service_tier`: Input should be 'auto' or 'standard_only'",
    ]


def test_a_required_nested_field_given_as_a_value_counts_as_present():
    assert _problems(
        "anthropic", {"thinking": {"type": "enabled", "budget_tokens": _Value()}}
    ) == []


def test_a_missing_required_nested_field_is_reported():
    problems = _problems("anthropic", {"thinking": {"type": "enabled", "display": _Value()}})

    assert problems == ["`anthropic.thinking.budget_tokens` is required"]


def test_a_discriminator_given_as_a_value_skips_the_nested_keys():
    tree = {"thinking": {"type": _Value(), "budget_tokens": 10, "bogus": 1}}

    assert _problems("anthropic", tree) == []


def test_an_unknown_literal_discriminator_beside_a_value_is_reported():
    problems = _problems("anthropic", {"thinking": {"type": "sometimes", "display": _Value()}})

    assert problems == [
        "`anthropic.thinking`: `type` must be one of: adaptive, between_tools, disabled, enabled"
    ]


def test_list_items_are_walked_for_key_paths():
    tree = {"safetySettings": [
        {"category": _Value(), "threshold": "OFF"},
        {"category": _Value(), "treshold": "OFF"},
    ]}

    problems = _problems("google", tree)

    assert len(problems) == 2
    assert problems[0].startswith("`google.safetySettings[1].treshold` is not a supported field")
    assert problems[1] == "`google.safetySettings[1].threshold` is required"


def test_a_literal_list_is_checked_whole():
    problems = _problems("anthropic", {"stop_sequences": ["a", 1], "max_tokens": _Value()})

    assert problems == ["`anthropic.stop_sequences[1]`: Input should be a valid string"]


def test_messages_never_echo_values():
    tree = {
        "service_tier": SENTINEL,
        "metadata": {"user_id": SENTINEL * 40},
        "max_tokens": SENTINEL,
        "thinking": {"type": SENTINEL, "display": _Value()},
        "temperature": _Value(),
    }

    problems = _problems("anthropic", tree)

    assert len(problems) == 4
    assert not any(SENTINEL in problem for problem in problems)


def test_a_block_that_is_not_a_mapping_is_reported():
    assert _problems("anthropic", [1]) == ["`anthropic` must be a mapping"]


def test_the_annotation_of_a_bounded_integer_field():
    from yian.llm_clients.provider_blocks import BlockFieldType, block_field_annotation

    tree = {"thinking": {"type": "enabled", "budget_tokens": _Value()}}

    assert block_field_annotation("anthropic", ("thinking", "budget_tokens"), tree) == (
        BlockFieldType(base=int, ge=1024)
    )


def test_the_annotations_of_simple_fields():
    from yian.llm_clients.provider_blocks import BlockFieldType, block_field_annotation

    assert block_field_annotation("anthropic", ("temperature",), {}) == BlockFieldType(
        base=float, ge=0, le=1
    )
    assert block_field_annotation("anthropic", ("service_tier",), {}) == BlockFieldType(
        base=str, literals=("auto", "standard_only")
    )
    assert block_field_annotation("google", ("store",), {}) == BlockFieldType(base=bool)
    assert block_field_annotation("anthropic", ("inference_geo",), {}) == BlockFieldType(base=str)


@pytest.mark.parametrize(
    ("block", "path", "tree"),
    [
        ("openai", ("metadata",), {}),
        ("anthropic", ("thinking",), {}),
        ("anthropic", ("stop_sequences",), {}),
        ("ollama", ("think",), {}),
        ("anthropic", ("thinking", "budget_tokens"), {"thinking": {"type": _Value()}}),
        ("anthropic", ("thinking", "budget_tokens"), {}),
        ("anthropic", ("nope",), {}),
    ],
)
def test_no_annotation_for_fields_checked_when_the_value_resolves(block, path, tree):
    from yian.llm_clients.provider_blocks import block_field_annotation

    assert block_field_annotation(block, path, tree) is None
