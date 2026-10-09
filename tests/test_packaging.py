"""The wheel ships every runtime data file under `src/yian`."""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "yian"
NOT_SHIPPED = {"README.md"}


def test_package_data_covers_every_runtime_data_file():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"].get("package-data", {}).get("yian", [])
    shipped = {path for pattern in patterns for path in PACKAGE.glob(pattern)}
    data_files = {
        path
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and path.suffix not in {".py", ".pyc"}
        and "__pycache__" not in path.parts
        and path.name not in NOT_SHIPPED
    }

    assert sorted(p.relative_to(PACKAGE).as_posix() for p in data_files - shipped) == []
