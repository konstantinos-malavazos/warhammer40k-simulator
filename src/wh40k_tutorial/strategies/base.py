"""The Strategy protocol — the extension point for player input and AI opponents.

A Strategy is anything that, given the current game state, returns the next
action for a side. Three implementations are planned:

- HumanStrategy:     prompts the player via the CLI. The player IS the strategy.
- ScriptedStrategy:  replays a fixed sequence from the scenario file.
                     Used for the opponent in the teaching-ladder tutorials.
- HeuristicStrategy: picks the highest-expected-damage shot using the same
                     combat math the engine uses for dice resolution (the
                     estimator is Monte Carlo-tested against the pipeline).
                     Scenarios opt in with opponent_strategy: "heuristic".

The engine ONLY interacts with strategies through this protocol. Player input
logic does not leak into combat code. AI logic does not need to know how
the player decides things. This separation is what keeps the engine clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wh40k_tutorial.core.models import UnitDatasheet, Weapon, melee_weapons, shootable_weapons
from wh40k_tutorial.core.scenario import (
    in_engagement_range,
    in_weapon_range,
    legal_destinations,
    opposing_side,
)


@dataclass(frozen=True)
class UnitSnapshot:
    """One unit's battlefield state, frozen at decision time.

    The engine builds these from its mutable runtime state; strategies only
    ever see (and can never corrupt) an immutable snapshot.
    """

    unit_id: str
    side: str  # "attacker" or "defender"
    datasheet: UnitDatasheet
    position: tuple[int, int]
    models: int  # models still standing
    wounds_on_lead: int  # wounds left on the front model; 0 once destroyed
    has_shot: bool = False  # already activated in the current shooting phase
    has_fought: bool = False  # already selected to fight in the current fight phase
    moved: bool = False  # already moved in the current movement phase
    advanced: bool = False  # advanced this turn (cannot shoot under 10.04)
    fell_back: bool = False  # fell back this turn (cannot shoot under 10.04)
    # The scenario's loadout override for this unit; empty means "use the
    # datasheet's default_loadout". Set by the engine from ScenarioUnit.loadout.
    loadout: tuple[str, ...] = ()

    @property
    def destroyed(self) -> bool:
        return self.models == 0

    @property
    def ranged_weapons(self) -> tuple[Weapon, ...]:
        """The weapons this unit may shoot with, in loadout order.

        Defers to `core.models.shootable_weapons` — the scenario's loadout
        override when one exists, otherwise the datasheet's ``default_loadout``
        (or every ranged weapon on a sheet that declares no loadout at all).
        """
        return shootable_weapons(self.datasheet, self.loadout)

    @property
    def melee_weapons(self) -> tuple[Weapon, ...]:
        """The weapons this unit may fight with (see `core.models.melee_weapons`)."""
        return melee_weapons(self.datasheet, self.loadout)


@dataclass(frozen=True)
class GameState:
    """Snapshot of the battlefield at decision time."""

    turn: int
    phase: str
    active_side: str  # "attacker" or "defender"
    units: tuple[UnitSnapshot, ...] = ()

    def unit(self, unit_id: str) -> UnitSnapshot:
        for u in self.units:
            if u.unit_id == unit_id:
                return u
        raise KeyError(f"no unit {unit_id!r} on the battlefield")

    def units_on(self, side: str) -> tuple[UnitSnapshot, ...]:
        return tuple(u for u in self.units if u.side == side)

    def eligible_shooters(self) -> tuple[UnitSnapshot, ...]:
        """Active-side units that can still shoot this phase.

        Alive, not yet activated, unengaged (10.04: an engaged unit needs a
        [CLOSE-QUARTERS] weapon to shoot, and no unit in our data carries
        one), did not advance or fall back this turn (10.04), and able to make
        at least one legal shot — some carried ranged weapon has some legal
        target (in range and unengaged, 04.02). This is the single definition
        of shooting eligibility — the engine's turn loop and both strategies
        rely on it agreeing with itself.
        """
        return tuple(
            u
            for u in self.units_on(self.active_side)
            if not u.destroyed
            and not u.has_shot
            and not u.advanced
            and not u.fell_back
            and not self.engaged_enemies(u)
            and any(self.shootable_targets(u, w) for w in u.ranged_weapons)
        )

    def shootable_targets(self, shooter: UnitSnapshot, weapon: Weapon) -> tuple[UnitSnapshot, ...]:
        """Surviving enemy units ``shooter`` may target with ``weapon``.

        04.02: a shooting target must be within the weapon's range and
        unengaged (visibility is vacuous — no terrain). The single definition
        of legal shooting targets — the engine's validation and the
        strategies' target menus rely on it agreeing with itself.
        """
        return tuple(
            u
            for u in self.units_on(opposing_side(shooter.side))
            if not u.destroyed
            and in_weapon_range(shooter.position, u.position, weapon)
            and not self.engaged_enemies(u)
        )

    def surviving_enemies(self) -> tuple[UnitSnapshot, ...]:
        """Units of the non-active side that are still on the table."""
        return tuple(u for u in self.units_on(opposing_side(self.active_side)) if not u.destroyed)

    def engaged_enemies(self, unit: UnitSnapshot) -> tuple[UnitSnapshot, ...]:
        """Surviving enemy units within engagement range of ``unit``.

        These are the only legal melee targets for it (04.02: a melee weapon
        must target a unit engaged with its bearer).
        """
        return tuple(
            u
            for u in self.units_on(opposing_side(unit.side))
            if not u.destroyed and in_engagement_range(unit.position, u.position)
        )

    def eligible_fighters(self, side: str) -> tuple[UnitSnapshot, ...]:
        """``side``'s units that can still be selected to fight this phase.

        Alive, engaged with at least one surviving enemy, armed with a melee
        weapon, and not yet selected to fight. The single definition of fight
        eligibility — the engine's alternation loop and the strategies' menus
        rely on it agreeing with itself. (The rulebook also lets a unit whose
        combat ended mid-phase fight via an *overrun* move; without movement
        there is nothing for such a unit to reach, so it is not offered.)
        """
        return tuple(
            u
            for u in self.units_on(side)
            if not u.destroyed and not u.has_fought and u.melee_weapons and self.engaged_enemies(u)
        )

    def eligible_movers(self, side: str | None = None) -> tuple[UnitSnapshot, ...]:
        """Units on ``side`` (default active_side) that can still move this phase.

        Alive and not yet moved in the current movement phase.
        """
        target_side = self.active_side if side is None else side
        return tuple(u for u in self.units_on(target_side) if not u.destroyed and not u.moved)

    def legal_move_types(self, unit: UnitSnapshot) -> tuple[str, ...]:
        """Legal move types for ``unit`` based on its engagement status.

        Remain Stationary is always legal (09.04).
        If unengaged: Normal Move (09.05) and Advance Move (09.06).
        If engaged: Fall-back Move (09.07).
        """
        if self.engaged_enemies(unit):
            return ("remain_stationary", "fall_back")
        return ("remain_stationary", "normal", "advance")

    def legal_destinations(
        self,
        unit: UnitSnapshot,
        move_type: str,
        *,
        advance_roll: int = 0,
    ) -> tuple[tuple[int, int], ...]:
        """All legal destination coordinates for ``unit`` under ``move_type``."""
        occupied = {u.position for u in self.units if not u.destroyed}
        enemy_positions = {
            u.position for u in self.units_on(opposing_side(unit.side)) if not u.destroyed
        }
        return legal_destinations(
            unit.position,
            move_type,
            unit.datasheet.profile.movement,
            advance_roll=advance_roll,
            occupied=occupied,
            enemy_positions=enemy_positions,
        )


@dataclass(frozen=True)
class Action:
    """An action a strategy can take.

    Three kinds exist: "shoot" (a shooting-phase volley), "fight" (a
    fight-phase melee activation), and "move" (a movement-phase action:
    remain stationary, normal move, advance, or fall back).
    """

    kind: str  # "shoot", "fight", or "move"
    attacker_unit_id: str
    weapon_key: str = ""
    target_unit_id: str = ""
    move_type: str = ""  # "remain_stationary", "normal", "advance", "fall_back"
    destination: tuple[int, int] | None = None


class Strategy(Protocol):
    """Anything that can choose actions for a side."""

    def choose_action(self, state: GameState) -> Action:
        """Return the next action for the active side, given the current state."""
        ...
