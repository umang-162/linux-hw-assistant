#!/usr/bin/env python3
"""Linux Hardware Q&A Assistant — CLI entry point.

Examples
--------
    python main.py                      # live hardware, Groq free cloud API
    python main.py --local              # live hardware, local Ollama (offline)
    python main.py --mock               # simulated bare-metal laptop (for VMs/WSL)
    python main.py --mock --mock-scenario laptop-nosudo
    python main.py --mock --inspect all # print tool JSON only — no LLM, no API key
    python main.py --ask "Is my SSD NVMe?"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from rich.console import Console, Group
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src import __version__
from src.config import MOCK_SCENARIOS, ConfigError, Settings, load_settings
from src.executor import AccessLog, build_source, detect_virtualization
from src.tools import ToolOutcome, ToolRegistry

console = Console()

STATUS_STYLE = {
    "ok": "green", "permission_required": "yellow", "virtualized_environment": "yellow",
    "unavailable": "yellow", "tool_missing": "yellow", "error": "red",
}

HELP_TEXT = """\
**Commands**

* `/tools` – list the read-only inspection tools
* `/debug` – toggle raw JSON tool output
* `/reset` – forget the conversation (and cached tool results)
* `/exit` – quit (or Ctrl-D)

**Try asking**

* Can I add more RAM, and how much?
* What about the GPU?
* Is my SSD NVMe? How much space is left?
* How healthy is my battery?
* How many cores and threads does my CPU have?
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Ask natural-language questions about this Linux machine's hardware.",
    )
    p.add_argument("--mock", action="store_true",
                   help="use realistic bare-metal fixtures instead of running real commands")
    p.add_argument("--mock-scenario", choices=MOCK_SCENARIOS, default=None,
                   help="which mock machine to simulate (default: laptop, or MOCK_SCENARIO)")
    p.add_argument("--local", action="store_true",
                   help="use a local Ollama model (offline, private) instead of Groq")
    p.add_argument("--debug", action="store_true",
                   help="show raw tool JSON and debug logs")
    p.add_argument("--model", default=None, help="override the model name")
    p.add_argument("--ram-slots", type=int, default=None, metavar="N",
                   help="real number of RAM slots on this board (for BIOSes that report "
                        "phantom slots); overrides PHYSICAL_RAM_SLOTS in .env")
    p.add_argument("--ask", metavar="QUESTION", default=None,
                   help="answer one question and exit (non-interactive)")
    p.add_argument("--inspect", metavar="TOOL", default=None,
                   help="run a tool (or 'all') and print its JSON — no LLM or API key needed")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_path=False,
                              rich_tracebacks=True)],
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.INFO if debug else logging.WARNING)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def render_tool_table(records) -> Table:
    table = Table(title="🔧 Under the hood", title_justify="left", show_lines=False,
                  expand=True, border_style="dim")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Commands / files accessed", style="dim")
    for r in records:
        status = Text(r.status, style=STATUS_STYLE.get(r.status, "white"))
        if r.cached:
            status.append(" (cached)", style="dim")
        table.add_row(r.name, status, "\n".join(r.accessed) or "-")
    return table


def render_json(payload: dict, title: str) -> Panel:
    return Panel(Syntax(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "json",
                        theme="monokai", word_wrap=True),
                 title=title, border_style="magenta")


def banner(settings: Settings, registry: ToolRegistry) -> Panel:
    virt = detect_virtualization(registry.ctx.source, AccessLog())
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("LLM", settings.provider_label)
    data_style = "yellow" if settings.mock else "green"
    grid.add_row("Hardware data", Text(settings.data_source_label, style=data_style))
    if virt["virtualized"]:
        grid.add_row("Environment", Text(f"virtualized ({virt['type']}) – physical slots/GPU "
                                         "may be hidden", style="yellow"))
    else:
        grid.add_row("Environment", "bare metal (no hypervisor detected)")
    grid.add_row("Privileges", "root" if registry.ctx.source.is_root()
                 else "normal user (RAM slot details need sudo -n dmidecode)")
    grid.add_row("Freshness", "every question re-runs the commands live – no saved results")
    if settings.physical_ram_slots:
        grid.add_row("RAM slots", f"{settings.physical_ram_slots} (your setting; overrides BIOS)")
    grid.add_row("Safety", "read-only allow-listed tools · shell=False · 5 s timeouts")
    grid.add_row("", Text("Type /help for commands, /exit to quit.", style="dim"))
    return Panel(grid, title=f"🐧 Linux Hardware Q&A Assistant v{__version__}",
                 border_style="cyan")


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def run_inspect(args: argparse.Namespace) -> int:
    settings = load_settings(local=args.local, mock=args.mock, mock_scenario=args.mock_scenario,
                             debug=args.debug, require_llm=False)
    if args.ram_slots:
        settings = dataclasses.replace(settings, physical_ram_slots=args.ram_slots)
    registry = ToolRegistry(build_source(settings.mock, settings.mock_scenario,
                                         settings.command_timeout),
                            physical_ram_slots=settings.physical_ram_slots)
    names = registry.names if args.inspect == "all" else [args.inspect]
    unknown = [n for n in names if n not in registry.names]
    if unknown:
        console.print(f"[red]Unknown tool:[/] {', '.join(unknown)}. "
                      f"Available: all, {', '.join(registry.names)}")
        return 2
    console.print(f"[dim]Data source: {settings.data_source_label}[/]")
    for name in names:
        outcome = registry.dispatch(name)
        console.print(render_json(outcome.payload, f"{name} · {outcome.status}"))
        console.print("[dim]  accessed: " + "; ".join(outcome.accessed) + "[/]\n")
    return 0


def answer(agent, question: str, debug: bool) -> bool:
    with console.status("[cyan]Thinking…", spinner="dots") as status:
        def on_start(name: str) -> None:
            status.update(f"[cyan]Running [bold]{name}[/bold] (read-only)…")

        def on_end(outcome: ToolOutcome) -> None:
            status.update("[cyan]Reading results…")

        agent.on_tool_start, agent.on_tool_end = on_start, on_end
        reply = agent.ask(question)

    if reply.tool_calls:
        console.print(render_tool_table(reply.tool_calls))
        if debug:
            for r in reply.tool_calls:
                console.print(render_json(r.payload, f"{r.name} → JSON sent to the model"))
    style = "red" if reply.error else "green"
    title = "⚠ Error" if reply.error else "🤖 Assistant"
    console.print(Panel(Markdown(reply.text), title=title, border_style=style))
    return not reply.error


def repl(agent, settings: Settings, registry: ToolRegistry) -> int:
    debug = settings.debug
    console.print(banner(settings, registry))
    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/]")
            return 0
        if not question:
            continue
        cmd = question.lower()
        if cmd in {"/exit", "/quit", "exit", "quit", ":q"}:
            console.print("[dim]Bye![/]")
            return 0
        if cmd in {"/help", "help", "?"}:
            console.print(Panel(Markdown(HELP_TEXT), border_style="dim"))
            continue
        if cmd == "/tools":
            t = Table(title="Allow-listed read-only tools", expand=True)
            t.add_column("Tool", style="cyan", no_wrap=True)
            t.add_column("What it inspects")
            for name, desc in registry.describe():
                t.add_row(name, desc)
            console.print(t)
            continue
        if cmd == "/reset":
            agent.reset()
            console.print("[dim]Conversation and cached results cleared.[/]")
            continue
        if cmd == "/debug":
            debug = not debug
            console.print(f"[dim]Raw JSON output {'on' if debug else 'off'}.[/]")
            continue
        try:
            answer(agent, question, debug)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/]")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mock_scenario:
        args.mock = True  # choosing a scenario implies mock mode
    setup_logging(args.debug)

    try:
        if args.inspect:
            return run_inspect(args)
        settings = load_settings(local=args.local, mock=args.mock,
                                 mock_scenario=args.mock_scenario, debug=args.debug,
                                 model_override=args.model)
        if args.ram_slots:
            settings = dataclasses.replace(settings, physical_ram_slots=args.ram_slots)
    except ConfigError as exc:
        console.print(Panel(str(exc), title="Configuration problem", border_style="red"))
        return 2

    from src.agent import HardwareAgent

    registry = ToolRegistry(build_source(settings.mock, settings.mock_scenario,
                                         settings.command_timeout),
                            physical_ram_slots=settings.physical_ram_slots)
    agent = HardwareAgent(settings, registry)

    if args.ask:
        console.print(Group(Text(f"LLM: {settings.provider_label}", style="dim"),
                            Text(f"Hardware data: {settings.data_source_label}", style="dim")))
        return 0 if answer(agent, args.ask, settings.debug) else 1
    return repl(agent, settings, registry)


if __name__ == "__main__":
    sys.exit(main())
