"""The `ac` command-line interface — a brokered host over the run boundary.

`ac run FLOW.yaml` loads a Compose flow, gathers its inputs (flags first, then an
interactive prompt for any missing required one), and drives it to a terminal.
The main thread owns terminal prompts/renders while a controller drives the flow;
blocking HUMAN_INPUT/``ask_user`` calls cross that broker, and the temporary WAIT
pause path is resumed there too. The CLI owns no engine scheduling logic.
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
