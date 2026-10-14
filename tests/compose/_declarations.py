"""Read authored `input:` or `variables:` declarations straight from YAML text."""

from yian.compose.helpers.types import read_source_parameters
from yian.compose.source import SourceDocument
from yian.compose.source.catalog import _SourceCatalogBuilder
from yian.nodes.contracts import Parameter
from yian.typesys.types import TypeRegistry


def parameters_from_yaml(
    text: str,
    registry: TypeRegistry,
    field: str = "input",
    require_default: bool = False,
) -> list[Parameter]:
    """Parse `text` and read its `field` section as located parameters.

    Args:
        text (`str`): A YAML document holding a `field:` mapping.
        registry (`TypeRegistry`): Typedefs the declarations resolve against.
        field (`str`, *optional*, defaults to `"input"`): The section to read.
        require_default (`bool`, *optional*, defaults to `False`): Read the section as
            agent variables, each needing a default.

    Returns:
        `list[Parameter]`: One parameter per declaration, in source order.
    """
    cursor = SourceDocument.parse(text).root.child(field)
    return read_source_parameters(
        cursor,
        registry,
        site_path=("test",),
        catalog_builder=_SourceCatalogBuilder(),
        field_name=field,
        declaration_label="variable" if require_default else "input",
        require_default=require_default,
    )
