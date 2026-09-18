"""Click-based CLI entry point.

Commands:

    wh40k                       # overview + pointer to --help
    wh40k list                  # list available scenarios
    wh40k play <scenario-id>    # run a tutorial scenario
    wh40k demo                  # static preview of the three-panel UI
    wh40k version               # print the package version

`play` runs the scenario runner (phase 5) with the narrator (phase 6): the
player's side is driven by `HumanStrategy` prompts, the opponent by
`ScriptedStrategy`, and every volley is reported step by step straight from
the `AttackResult` record — each fact line followed by the rule that drove
it. After a volley the player can ask for the deeper rule behind any step,
and the rules panel of the final battlefield view carries the explanations
for the last volley.
"""

from __future__ import annotations

import random

import click
from rich.console import Console

from wh40k_tutorial import __version__
from wh40k_tutorial.core.scenario import (
    SIDES,
    ScenarioDataError,
    ScenarioTurn,
    available_scenarios,
    load_scenario_by_id,
    opposing_side,
)
from wh40k_tutorial.engine import BattleState, EngineError, MoveEvent, VolleyEvent, run_scenario
from wh40k_tutorial.narrator import StepNarration, narrate_volley
from wh40k_tutorial.strategies.base import Strategy
from wh40k_tutorial.strategies.heuristic import HeuristicStrategy
from wh40k_tutorial.strategies.human import HumanStrategy
from wh40k_tutorial.strategies.scripted import (
    ScriptedStrategy,
    ScriptExhaustedError,
    scripted_actions_for,
)
from wh40k_tutorial.ui.demo import run_demo
from wh40k_tutorial.ui.live import render_live_shell, volley_report_lines

# Explicit render heights so the layout never crops silently: the base value
# fits the battlefield (16 rows minimum) plus one full volley report; the
# final frame is taller because its narrow rules panel carries the narrator's
# five wrapped explanations for the last volley.
_SHELL_HEIGHT = 32
_FINAL_SHELL_HEIGHT = 46


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Warhammer 40,000 interactive tutorial."""
    if ctx.invoked_subcommand is None:
        click.echo("warhammer40k-tutorial — an interactive 40k tutorial.")
        click.echo("Run `wh40k list` to see scenarios, `wh40k play <id>` to play one.")
        click.echo("Run `wh40k --help` for all commands.")


@main.command()
def version() -> None:
    """Print the package version."""
    click.echo(__version__)


@main.command()
def demo() -> None:
    """Show a static preview of the three-panel tutorial interface."""
    run_demo()


@main.command(name="list")
def list_scenarios() -> None:
    """List available tutorial scenarios."""
    try:
        scenarios = available_scenarios()
    except ScenarioDataError as exc:
        raise click.ClickException(str(exc)) from exc
    for scenario in scenarios:
        click.echo(
            f"{scenario.scenario_id}  —  {scenario.title} (teaches {scenario.teaches.rstrip('.')})"
        )


def _why_loop(narrations: list[StepNarration]) -> None:
    """Offer the deeper rule behind any step of the volley just shown.

    Enter (or end-of-input, so piped runs never hang) continues the battle;
    a step name — optionally prefixed with "why" — prints its full rule.
    """
    by_step = {n.step: n for n in narrations}
    options = "/".join(n.step for n in narrations)
    while True:
        try:
            raw = click.prompt(
                f"Deeper rule? ({options} — Enter to continue)",
                default="",
                show_default=False,
            )
        except click.Abort:
            click.echo("")
            return
        choice = raw.strip().lower().removeprefix("why").strip().rstrip("?")
        if not choice:
            return
        narration = by_step.get(choice)
        if narration is None:
            click.echo(f"Pick one of: {options} — or press Enter to continue.")
            continue
        click.echo(f"\n{choice.upper()} — the full rule:")
        click.echo(narration.expansion + "\n")


@main.command()
@click.argument("scenario_id")
@click.option("--seed", type=int, default=None, help="Seed the dice for a reproducible battle.")
def play(scenario_id: str, seed: int | None) -> None:
    """Run a tutorial scenario end to end."""
    try:
        scenario = load_scenario_by_id(scenario_id)
    except ScenarioDataError as exc:
        raise click.ClickException(str(exc)) from exc

    console = Console()
    opponent = opposing_side(scenario.player_side)
    opponent_strategy: Strategy = (
        HeuristicStrategy()
        if scenario.opponent_strategy == "heuristic"
        else ScriptedStrategy(scripted_actions_for(scenario, opponent))
    )
    strategies: dict[str, Strategy] = {
        scenario.player_side: HumanStrategy(rng=random.Random(seed)),
        opponent: opponent_strategy,
    }

    click.echo(f"\n=== {scenario.title} ===")
    click.echo(f"This scenario teaches {scenario.teaches.rstrip('.')}.")
    click.echo(f"You play the {scenario.player_side}.\n")
    click.echo(scenario.intro + "\n")
    console.print(
        render_live_shell(
            BattleState.from_scenario(scenario).snapshot().units,
            ("The battle is about to begin.",),
        ),
        height=_SHELL_HEIGHT,
    )

    last_volley: list[str] = []
    last_narrations: list[StepNarration] = []

    def announce_turn(number: int, turn: ScenarioTurn) -> None:
        if turn.phase == "fight":
            click.echo(
                f"\n— Turn {number}: fight phase — both sides fight; "
                f"the {turn.active_side} picks first —"
            )
        elif turn.phase == "movement":
            click.echo(f"\n— Turn {number}: movement phase, the {turn.active_side} moves —")
        else:
            click.echo(f"\n— Turn {number}: {turn.phase} phase, the {turn.active_side} acts —")
        if turn.narrate_before:
            click.echo(turn.narrate_before + "\n")

    def show_move(event: MoveEvent) -> None:
        mt_name = event.move_type.replace("_", " ").title()
        if event.from_pos == event.to_pos:
            click.echo(
                f"   ↳ {event.action.attacker_unit_id} remained stationary at {event.to_pos}."
            )
        else:
            click.echo(
                f"   ↳ {event.action.attacker_unit_id} performed a {mt_name} move "
                f"from {event.from_pos} to {event.to_pos}."
            )

    def show_volley(event: VolleyEvent) -> None:
        last_volley[:] = volley_report_lines(event.result, turn=event.turn)
        last_narrations[:] = narrate_volley(event.result)
        header, fact_lines = last_volley[0], last_volley[1:]
        click.echo(header)
        for fact, narration in zip(fact_lines, last_narrations, strict=True):
            click.echo(fact)
            click.echo(click.style(f"   ↳ {narration.inline}", dim=True))
        click.echo("")
        _why_loop(last_narrations)

    try:
        final = run_scenario(
            scenario,
            strategies,
            rng=random.Random(seed),
            on_turn_start=announce_turn,
            on_volley=show_volley,
            on_move=show_move,
        )
    except (EngineError, ScriptExhaustedError) as exc:
        raise click.ClickException(str(exc)) from exc

    if last_narrations:
        rules_heading = "The rules behind that volley"
        rules_body = "\n\n".join(f"{n.step.upper()}: {n.inline}" for n in last_narrations)
        console.print(
            render_live_shell(
                final.snapshot().units,
                last_volley,
                rules_heading=rules_heading,
                rules_body=rules_body,
            ),
            height=_FINAL_SHELL_HEIGHT,
        )
    else:
        console.print(
            render_live_shell(
                final.snapshot().units,
                ("No shots were fired.",),
            ),
            height=_SHELL_HEIGHT,
        )
    for side in SIDES:
        if final.side_wiped(side):
            click.echo(f"The {side}'s force has been wiped out.")
    click.echo("\n" + scenario.outro)


if __name__ == "__main__":
    main()
