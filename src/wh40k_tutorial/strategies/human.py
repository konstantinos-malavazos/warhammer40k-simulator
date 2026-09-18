"""HumanStrategy: the player picks each shot and each fight through Click prompts.

The player IS the strategy — this class only translates battlefield snapshots
into numbered menus and the player's picks into an ``Action``. It offers only
legal choices (eligible shooters or fighters, weapons of the phase's type,
and — in melee — only engaged targets), so the engine's own legality
validation should never fire on a human decision. When
a menu has exactly one option it is announced and auto-picked: scenario 01
has one unit, one gun, and one target, and asking three one-option questions
would teach nothing.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

import click

from wh40k_tutorial.core.dice import roll_d6
from wh40k_tutorial.core.models import Weapon
from wh40k_tutorial.core.scenario import chebyshev_squares
from wh40k_tutorial.strategies.base import Action, GameState, UnitSnapshot

T = TypeVar("T")


def _describe_unit(unit: UnitSnapshot) -> str:
    return f"{unit.datasheet.display_name} ({unit.models} models)"


def _describe_weapon(weapon: Weapon) -> str:
    return (
        f"{weapon.display_name} — {weapon.attacks} attacks, {weapon.skill}+ to hit, "
        f"S{weapon.strength}, AP -{weapon.ap}, {weapon.damage} damage"
    )


class HumanStrategy:
    """Prompts the player for one action per activation — a shot, a fight, or a move."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def choose_action(self, state: GameState) -> Action:
        if state.phase == "fight":
            return self._choose_fight(state)
        if state.phase == "movement":
            return self._choose_move(state)
        return self._choose_shot(state)

    def _choose_move(self, state: GameState) -> Action:
        mover = _pick("unit to move", state.eligible_movers(), _describe_unit)
        move_types = state.legal_move_types(mover)

        def describe_move_type(mt: str) -> str:
            m = mover.datasheet.profile.movement
            if mt == "remain_stationary":
                return 'Remain Stationary (0" — no move penalties)'
            if mt == "normal":
                return f'Normal Move (up to {m}" — stay unengaged)'
            if mt == "advance":
                return f'Advance (up to {m}" + 1D6" — cannot shoot)'
            if mt == "fall_back":
                return f'Fall Back (up to {m}" — retreat, cannot shoot)'
            return mt

        move_type = _pick("move type", move_types, describe_move_type)
        if move_type == "remain_stationary":
            return Action(
                kind="move",
                attacker_unit_id=mover.unit_id,
                move_type="remain_stationary",
                destination=mover.position,
            )

        advance_roll = 0
        if move_type == "advance":
            result = roll_d6(1, rng=self.rng)
            advance_roll = result.raw_rolls[0]
            m = mover.datasheet.profile.movement
            click.echo(
                f"\n🎲 Advance roll for {mover.datasheet.display_name}: rolled a {advance_roll}! "
                f'Max move is {m + advance_roll}".'
            )
            dests = state.legal_destinations(mover, "advance", advance_roll=advance_roll)
        else:
            dests = state.legal_destinations(mover, move_type)

        if not dests:
            click.echo(f"No legal destinations available for {move_type}. Remaining stationary.")
            return Action(
                kind="move",
                attacker_unit_id=mover.unit_id,
                move_type="remain_stationary",
                destination=mover.position,
            )

        def describe_dest(dest: tuple[int, int]) -> str:
            col, row = dest
            dist_sq = chebyshev_squares(mover.position, dest)
            dist_in = dist_sq * 2
            if dest == mover.position:
                return f'({col}, {row}) — current square (0")'
            return f'({col}, {row}) — {dist_in}" ({dist_sq} sq)'

        destination = _pick("destination square", dests, describe_dest)
        return Action(
            kind="move",
            attacker_unit_id=mover.unit_id,
            move_type=move_type,
            destination=destination,
        )

    def _choose_shot(self, state: GameState) -> Action:
        shooter = _pick("unit to shoot with", state.eligible_shooters(), _describe_unit)
        # Only weapons that have a legal target (in range, unengaged — 04.02);
        # eligibility guarantees at least one, so the menu is never empty.
        weapons = tuple(w for w in shooter.ranged_weapons if state.shootable_targets(shooter, w))
        weapon = _pick("weapon", weapons, _describe_weapon)
        target = _pick("target", state.shootable_targets(shooter, weapon), _describe_unit)
        return Action(
            kind="shoot",
            attacker_unit_id=shooter.unit_id,
            weapon_key=weapon.name,
            target_unit_id=target.unit_id,
        )

    def _choose_fight(self, state: GameState) -> Action:
        def describe_fighter(unit: UnitSnapshot) -> str:
            # Two mobs of "Boyz (10 models)" are indistinguishable on a menu;
            # naming each unit's opponent is what makes the ordering decision
            # legible — WHICH fight, not just which unit.
            enemies = " and ".join(e.datasheet.display_name for e in state.engaged_enemies(unit))
            return f"{_describe_unit(unit)} — fighting {enemies}"

        fighter = _pick(
            "unit to fight with",
            state.eligible_fighters(state.active_side),
            describe_fighter,
        )
        weapon = _pick("melee weapon", fighter.melee_weapons, _describe_weapon)
        target = _pick("target", state.engaged_enemies(fighter), _describe_unit)
        return Action(
            kind="fight",
            attacker_unit_id=fighter.unit_id,
            weapon_key=weapon.name,
            target_unit_id=target.unit_id,
        )


def _pick(what: str, options: Sequence[T], describe: Callable[[T], str]) -> T:
    """Announce a single option, or show a numbered menu and prompt for a pick."""
    if not options:
        raise RuntimeError(f"no {what} available — the engine should not have asked for an action")
    if len(options) == 1:
        click.echo(f"{what.capitalize()}: {describe(options[0])}")
        return options[0]
    click.echo(f"Pick a {what}:")
    for i, option in enumerate(options, start=1):
        click.echo(f"  {i}. {describe(option)}")
    index = click.prompt("Your choice", type=click.IntRange(1, len(options)))
    return options[index - 1]
