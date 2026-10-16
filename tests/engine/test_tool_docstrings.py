"""Tool docstrings parse into a description and per-parameter descriptions like the Agents SDK."""

import inspect

import pytest

from yian.tools.docstrings import DocInfo, parse_docstring


def test_google_args_section():
    doc = (
        "Place an order.\n\nArgs:\n    symbol: The ticker.\n    qty (int): How many\n"
        "        shares.\n\nReturns:\n    The order id."
    )

    info = parse_docstring(doc)

    assert info.description == "Place an order."
    assert info.params == {"symbol": "The ticker.", "qty": "How many\nshares."}


@pytest.mark.parametrize("header", ["Arguments:", "Params:", "Parameters:", "ARGS:", "args:"])
def test_google_parameter_section_aliases(header):
    info = parse_docstring(f"Do it.\n\n{header}\n    x: The x.")

    assert info.description == "Do it."
    assert info.params == {"x": "The x."}


def test_google_args_without_blank_line_above():
    info = parse_docstring("Do it.\nArgs:\n    x: The x.\n    y: The y.")

    assert info.description == "Do it."
    assert info.params == {"x": "The x.", "y": "The y."}


def test_google_args_without_blank_line_after_another_section_body():
    info = parse_docstring("Do it.\n\nNote:\n    Careful.\nArgs:\n    x: The x.")

    assert info.description == "Do it."
    assert info.params == {"x": "The x."}


def test_yian_style_entries_with_description_below():
    doc = """Find a name.

    Args:
        name (`str`, *optional*, defaults to `None`):
            Name used in messages.
            `None` uses the default name.
        items (`dict[str, Item]`):
            Items keyed by node id.
    """

    info = parse_docstring(doc)

    assert info.description == "Find a name."
    assert info.params == {
        "name": "Name used in messages.\n`None` uses the default name.",
        "items": "Items keyed by node id.",
    }


def test_variadic_names_lose_their_stars():
    info = parse_docstring("Do it.\n\nArgs:\n    *args: Positional.\n    **kwargs: Named.")

    assert info.params == {"args": "Positional.", "kwargs": "Named."}


def test_no_section_gives_the_whole_docstring():
    info = parse_docstring("""Do it.

    More detail
    over two lines.
    """)

    assert info.description == "Do it.\n\nMore detail\nover two lines."
    assert info.params == {}


@pytest.mark.parametrize("header", ["Returns:", "Raises:", "Example:"])
def test_another_google_section_ends_the_description(header):
    info = parse_docstring(f"Do it.\n\n{header}\n    Something.\n\nArgs:\n    x: The x.")

    assert info.description == "Do it."
    assert info.params == {"x": "The x."}


def test_numpy_parameters():
    doc = "Add.\n\nParameters\n----------\na : int\n    First.\nb : int\n    Second.\n"

    info = parse_docstring(doc)

    assert info.description == "Add."
    assert info.params == {"a": "First.", "b": "Second."}


def test_numpy_parameters_end_at_the_next_section():
    doc = "Add.\n\nParameters\n----------\na : int\n    First.\n\nReturns\n-------\nint\n    Sum.\n"

    info = parse_docstring(doc)

    assert info.description == "Add."
    assert info.params == {"a": "First."}


def test_sphinx_params():
    info = parse_docstring("Add.\n\n:param a: First.\n:param b: Second.\n:returns: Sum.")

    assert info.description == "Add."
    assert info.params == {"a": "First.", "b": "Second."}


def test_sphinx_param_with_type_and_continuation():
    info = parse_docstring("Add.\n\n:param int a: First\n    number.\n:param b: Second.")

    assert info.params == {"a": "First\nnumber.", "b": "Second."}


def test_detection_tie_prefers_sphinx_over_numpy_over_google():
    # One sphinx marker and one google marker: sphinx wins the tie.
    sphinx_vs_google = parse_docstring("Do it.\n\n:param a: First.\n\nReturns:\n    Sum.")
    # One numpy marker and one google marker: numpy wins the tie.
    numpy_vs_google = parse_docstring(
        "Do it.\n\nRaises:\n    Error.\n\nParameters\n----------\na : int\n    First."
    )

    assert sphinx_vs_google.params == {"a": "First."}
    assert numpy_vs_google.params == {"a": "First."}


@pytest.mark.parametrize("doc", [None, "", "   \n  "])
def test_empty_docstring(doc):
    assert parse_docstring(doc) == DocInfo(None, {})


def test_header_with_title_ends_the_description():
    info = parse_docstring("Do it.\n\nExample: usage\n    do_it()\n\nArgs:\n    x: The x.")

    assert info.description == "Do it."
    assert info.params == {"x": "The x."}


def test_lower_case_header_ends_the_description():
    info = parse_docstring("Do it.\n\nnote:\n    Careful.")

    assert info.description == "Do it."


def test_colon_in_type_and_description():
    info = parse_docstring("Do it.\n\nArgs:\n    x (dict[str, int]): A map: of things.")

    assert info.params == {"x": "A map: of things."}


def test_colon_inside_parenthesized_type():
    info = parse_docstring('Do it.\n\nArgs:\n    x (Literal["a:b"]): The mode.')

    assert info.params == {"x": "The mode."}


def test_only_a_section_has_no_description():
    info = parse_docstring("Args:\n        x: The x.\n\n    Returns:\n        The y.\n    ")

    assert info.description is None
    assert info.params == {"x": "The x."}


def test_text_after_a_leading_section_is_the_description():
    info = parse_docstring("Args:\n    x: X.\n\nTrailing text.")

    assert info.description == "Trailing text."
    assert info.params == {"x": "X."}


def test_args_first_docstring_parses_raw_and_cleaned_alike():
    def place(x):
        """
        Args:
            x: The x.
        """

    assert parse_docstring(place.__doc__) == DocInfo(None, {"x": "The x."})
    assert parse_docstring(inspect.getdoc(place)) == DocInfo(None, {"x": "The x."})


def test_summary_first_docstring_parses_raw_and_cleaned_alike():
    def place(x):
        """Place it.

        Args:
            x: The x.
        """

    assert parse_docstring(place.__doc__) == DocInfo("Place it.", {"x": "The x."})
    assert parse_docstring(inspect.getdoc(place)) == DocInfo("Place it.", {"x": "The x."})


def test_whitespace_only_parameter_section_is_not_a_section():
    info = parse_docstring("Do it.\n\nArgs:\n    \nReturns:\n    y.")

    assert info.params == {}


def test_whitespace_only_parameter_section_in_a_function():
    def place(y):
        pass

    place.__doc__ = "Do it.\n\n    Args:\n        \n    Returns:\n        y.\n    "

    assert parse_docstring(place.__doc__).params == {}
    assert parse_docstring(inspect.getdoc(place)).params == {}


def test_numpy_multi_name_entry():
    info = parse_docstring("Add.\n\nParameters\n----------\na, b : int\n    Operands.\n")

    assert info.params == {"a": "Operands.", "b": "Operands."}


def test_sphinx_param_with_a_second_paragraph():
    info = parse_docstring("Add.\n\n:param a: First.\n\n    Para two.\n:param b: Second.")

    assert info.params == {"a": "First.\n\nPara two.", "b": "Second."}


def test_column_zero_colon_line_without_indented_block_is_text():
    info = parse_docstring("Do it.\n\nNote: this is text.\nStill text.")

    assert info.description == "Do it.\n\nNote: this is text.\nStill text."
