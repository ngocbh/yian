"""The `ac` command-line interface — provider setup and a brokered run host.

`ac auth openai` stages subscription or API-key authentication, presents available
models, and saves the selected method and default model together. It owns prompts;
provider services own the protocol and credential storage.

`ac run FLOW.yaml` loads a Compose flow, gathers fresh inputs from flags and an
available interactive prompt, selects and reports a durable run-log path, and drives
it to a terminal. Without an interactive stdin, optional/defaulted declarations stay
absent for START to resolve and missing required declarations fail before provider or
run-log setup.
``--resume`` instead recovers an existing path without gathering fresh inputs or
model configuration. The main thread owns terminal prompts/renders while a
controller drives the flow;
blocking HUMAN_INPUT/``ask_user`` calls cross that broker. WAIT remains live in
its worker until its deadline. The CLI owns path policy, not engine scheduling
or durable state-transition logic. `memory` discovers declared policies through
the compiled layout and lazily constructs built-in backends at the run boundary.
The controller owns their complete lifetime, after worker drain; optional
`--memory-storage POLICY=PATH` paths remain host-only. Terminal and historically
memory-dormant recovery never opens those backends.

Knows about: compose; compile metadata and LLM-cascade helpers; the package-root run
boundary; memory contracts and built-in backends; events; node host-stop signals;
typed values; tools; LLM API-key metadata;
process defaults in ``_settings``; and peer CLI modules. Never imports runtime scheduling
internals or provider SDKs; owns no graph mutation.
"""

from __future__ import annotations

import typer

from agent_composer.cli.chat.command import chat as _chat
from agent_composer.cli.auth import auth as _auth
from agent_composer.cli.run import run as _run

app = typer.Typer(
    name="composer",
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
app.command("auth")(_auth)


def main() -> None:
    """Console-script entry point (`composer = agent_composer.cli:main`)."""
    app()
