"""The `ac` command-line interface — a thin host over the run/resume seam.

`ac run FLOW.yaml` loads a Compose flow, gathers its inputs (flags first, then an
interactive prompt for any missing required one), drives it to a terminal, resumes
any HUMAN_INPUT / WAIT pause interactively, and renders the output. The CLI owns no
engine logic: it calls `agent_composer.run` directly.
"""

from __future__ import annotations

import typer

from agent_composer.cli.chat.command import chat as _chat
from agent_composer.cli.run import run as _run

app = typer.Typer(
    name="ac",
    help="Agent Composer — run agent flows from the command line.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Keep `run` a named subcommand (a single-command Typer app otherwise collapses
    it, so `ac run flow.yaml` would read `run` as the flow path)."""


app.command("run")(_run)
app.command("chat")(_chat)


def main() -> None:
    """Console-script entry point (`ac = agent_composer.cli:main`)."""
    app()
