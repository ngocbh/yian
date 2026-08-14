"""CLI coverage for durable run-log path selection and recovery options."""

from pathlib import Path


def test_cli_tests_use_an_isolated_home(isolated_cli_home: Path) -> None:
    """Default CLI state can never escape into the developer's real home."""
    assert Path.home() == isolated_cli_home
