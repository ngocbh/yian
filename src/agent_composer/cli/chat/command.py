"""`ac chat` — an interactive REPL dogfooded as a flow.

Loads the bundled composer chat flow (or a user-supplied one), runs it, and drives the
per-turn human_input suspend/resume: print the assistant's latest reply, prompt the
user, resume with the message. `/exit` or EOF ends the session. The turn loop and
tools live here; the engine sees only an ordinary flow.

A single bad turn does not end the session. The reply agent can fail mid-turn (a raise,
or hitting its tool-iteration cap), which terminates the underlying flow *run* as
`failed`. Because the transcript is mirrored host-side off the turn body's terminal
(`__end__`) output, the loop surfaces the error and RESTARTS the flow seeded with the
accumulated transcript, so the conversation continues from where it left off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from agent_composer.cli.chat import tools as chat_tools
from agent_composer.cli.run import (  # reused helpers
    _ensure_cwd_importable,
    _ensure_provider_keys,
    _last_segment,
    _render_load_error,
)
from agent_composer.compose.errors import LoadError
from agent_composer.compose.loader import load_flow
from agent_composer.compose.run import resume_command, resume_flow, run_flow
from agent_composer.events import NodeSucceeded

console = Console()
err_console = Console(stderr=True)  # errors go to stderr, mirroring `ac run` (run.py:45-46)

# The composer chat flow shipped with the package — the default when no flow is given.
_BUNDLED = Path(__file__).parent / "chat.yaml"


def chat(
    flow: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False, readable=True,
        help="A chat flow .yaml (default: the bundled composer chat).",
    ),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", "-C",
        help="Directory the flow-op tools are confined to.",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override the LLM provider for agents that set none (cascade)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override the LLM model for agents that set none (cascade)."
    ),
) -> None:
    """Start an interactive chat session over a chat flow."""
    _ensure_cwd_importable()  # so a chat flow's `code: pkg.mod:fn` ref resolves (mirrors `ac run`)
    chat_tools.set_workspace(workspace)
    path = flow or _BUNDLED
    text = path.read_text()
    search = [path.parent, workspace]
    # The CLI flags supply the OUTERMOST cascade layer (fill-the-gap), not a hard override.
    cli_cfg = {k: v for k, v in {"provider": provider, "model": model}.items() if v}
    # Load and preflight once, mirroring `ac run` — both failure modes surface as a clean
    # stderr error + exit 1, never a raw traceback:
    #   - a malformed flow raises LoadError (a ValueError) -> a located `.yaml` source frame;
    #   - a missing key with no TTY raises RuntimeError    -> a red key message.
    try:
        loaded = load_flow(text, search_paths=search)
    except LoadError as err:
        _render_load_error(err, path, text)
        raise typer.Exit(code=1)
    try:
        _ensure_provider_keys(loaded, cli_cfg)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # Host-mirrored conversation state, updated off the flow's event stream each turn:
    #   reply      — the reply node's latest output, printed before the next prompt (`None`
    #                between turns, cleared once printed). The reply node lives inside the
    #                per-turn LOOP body, so its runtime id is namespaced (`chat/reply`,
    #                `chat~1/reply`) — `_last_segment` maps it back to the authored `reply`.
    #   transcript — the turn body's grown transcript, read off the body's terminal
    #                (`__end__`) record. Mirrored here so a FAILED turn (the agent raised
    #                or hit its tool cap, terminating the run) can restart the flow from
    #                the accumulated conversation instead of ending the session. The
    #                per-iteration body end emits the `{transcript, exited}` dict; the
    #                top-level flow end emits a bare str, so the `isinstance(..., dict)`
    #                guard keeps only the body's record.
    state: dict[str, Any] = {"reply": None, "transcript": ""}

    def on_event(ev: Any) -> None:
        if not isinstance(ev, NodeSucceeded):
            return
        seg = _last_segment(ev.node_id)
        if seg == "reply":
            state["reply"] = ev.output
        elif seg == "__end__" and isinstance(ev.output, dict):
            state["transcript"] = ev.output.get("transcript", state["transcript"])

    quit_session = False
    while not quit_session:
        # (Re)start the flow from the accumulated transcript. A FRESH `load_flow` is required per
        # (re)start: the engine grows the compiled graph in place as the LOOP spawns iterations, so
        # a graph that has already run cannot be re-run. It immediately pauses on the first
        # `human_input` (the turn's `ask`); the inner loop then drives turns.
        loaded = load_flow(text, search_paths=search)
        result = run_flow(
            loaded, {"opening": state["transcript"]}, on_event=on_event, llm_config=cli_cfg or None
        )
        while result.status == "paused":
            if state["reply"] is not None:
                console.print(Markdown(str(state["reply"])))
                state["reply"] = None
            try:
                msg = console.input("[bold cyan]You:[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D / Ctrl+C at the prompt echoes NO newline, leaving the cursor glued to
                # the `You:` line — emit one so the exit notice (and the shell prompt after) start
                # on a fresh line instead of needing a manual redraw.
                console.print()
                quit_session = True
                break
            if msg.strip() in {"/exit", "/quit"}:
                quit_session = True
                break
            # The pause is the turn's single `human_input`; deliver the typed line as its answer.
            reason = result.pause_reasons[0]
            cmd = resume_command(loaded, reason, msg)
            # Resume the SAME in-process run via the live engine. `llm_config` is not re-supplied
            # here: it is a no-op on the `engine=` path (the live graph already carries the baked
            # cascade from the initial run_flow; it only re-applies on the `checkpoint=` restore).
            result = resume_flow(loaded, engine=result.engine, commands=[cmd], on_event=on_event)

        if quit_session:
            break
        # The run left "paused" without the user quitting. A FAILED turn (the agent raised or hit
        # its tool cap) surfaces the error and restarts the flow from the mirrored transcript; any
        # other terminal (the flow reached its own end / max) simply ends the session.
        if result.status == "failed":
            console.print(f"[red]turn failed:[/red] {result.error}")
            console.print("[dim]continuing — your conversation so far is preserved.[/dim]")
            state["reply"] = None
            continue
        break

    console.print("[dim]session ended[/dim]")
