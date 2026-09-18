"""The scenario runner: the only place battlefield state mutates.

`run_scenario` walks a scenario's turn entries. A shooting entry is one
shooting phase for one side: every eligible unit on that side activates once,
the side's `Strategy` chooses each shot, `core.combat.resolve_shooting`
resolves it, and the result's final defender state (normal damage plus any
mortal wounds) updates the target's runtime state. A fight entry is one Fight
phase, in which BOTH sides act: every engaged unit must fight exactly once,
players alternate picking which of their units fights next — the side whose
turn it is picks first — and casualties come off the table fight by fight, so
a unit selected later swings with whatever models it has left (ADR 0006).

Boundaries (see ADR 0005):

- Datasheets stay frozen value types; `UnitRuntime` here carries the mutable
  state (models left, wounds on the lead model).
- Strategies see frozen `GameState` snapshots and return `Action`s — they can
  never touch engine state. The engine validates every returned action and
  raises `EngineError` on an illegal one, so a buggy script or AI fails
  loudly instead of corrupting the battle.
- Presentation observes through optional callbacks receiving frozen
  `VolleyEvent`s; the engine itself never prints.

Distances follow ADR 0007 (1 square = 2", Chebyshev), and the shooting rules
they enable are enforced here on live state: a shot's target must be within
the weapon's range and unengaged, and the shooter must be unengaged (04.02,
10.04 — no [CLOSE-QUARTERS] weapons in our data). Engagement is the same
single definition the fight phase has always enforced
(`core.scenario.in_engagement_range` — adjacency, which under ADR 0007 IS
the 2" engagement range exactly). Remaining deliberate simplification: one
activation resolves one weapon profile. A unit's shootable weapons are its
effective loadout — the scenario's per-unit loadout override if one is
given, else the datasheet's default_loadout (see
`core.models.shootable_weapons`).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from wh40k_tutorial.core.combat import AttackResult, resolve_melee, resolve_shooting
from wh40k_tutorial.core.models import UnitDatasheet, Weapon, melee_weapons, shootable_weapons
from wh40k_tutorial.core.scenario import (
    BATTLEFIELD_HEIGHT,
    BATTLEFIELD_WIDTH,
    Scenario,
    ScenarioTurn,
    chebyshev_squares,
    distance_inches,
    in_engagement_range,
    in_weapon_range,
    opposing_side,
    reach_squares,
)
from wh40k_tutorial.strategies.base import Action, GameState, Strategy, UnitSnapshot


class EngineError(ValueError):
    """A strategy returned an action the rules of the battle don't allow."""


@dataclass
class UnitRuntime:
    """One unit's mutable battlefield state. Everything static lives on the datasheet."""

    unit_id: str
    side: str
    datasheet: UnitDatasheet
    position: tuple[int, int]
    models: int
    wounds_on_lead: int
    # The scenario's loadout override; empty = use the datasheet's default.
    loadout: tuple[str, ...] = ()
    moved: bool = False
    advanced: bool = False
    fell_back: bool = False

    @property
    def destroyed(self) -> bool:
        return self.models == 0

    def snapshot(
        self,
        *,
        has_shot: bool = False,
        has_fought: bool = False,
        moved: bool = False,
    ) -> UnitSnapshot:
        return UnitSnapshot(
            unit_id=self.unit_id,
            side=self.side,
            datasheet=self.datasheet,
            position=self.position,
            models=self.models,
            wounds_on_lead=self.wounds_on_lead,
            has_shot=has_shot,
            has_fought=has_fought,
            moved=moved or self.moved,
            advanced=self.advanced,
            fell_back=self.fell_back,
            loadout=self.loadout,
        )


@dataclass
class BattleState:
    """The whole battlefield's mutable state, plus where we are in the scenario."""

    units: dict[str, UnitRuntime]  # insertion order = scenario order
    turn: int = 0
    phase: str = ""
    active_side: str = ""
    shot_this_phase: set[str] = field(default_factory=set)
    fought_this_phase: set[str] = field(default_factory=set)
    moved_this_phase: set[str] = field(default_factory=set)

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> BattleState:
        units = {
            u.unit_id: UnitRuntime(
                unit_id=u.unit_id,
                side=side.name,
                datasheet=u.datasheet,
                position=u.position,
                models=u.models,
                wounds_on_lead=u.datasheet.profile.wounds,
                loadout=u.loadout,
            )
            for side in (scenario.attacker, scenario.defender)
            for u in side.units
        }
        return cls(units=units)

    def survivors(self, side: str) -> list[UnitRuntime]:
        return [u for u in self.units.values() if u.side == side and not u.destroyed]

    def side_wiped(self, side: str) -> bool:
        return not self.survivors(side)

    @property
    def battle_over(self) -> bool:
        return self.side_wiped("attacker") or self.side_wiped("defender")

    def reset_movement_flags(self, side: str) -> None:
        """Reset turn-level movement flags at the start of ``side``'s turn."""
        for u in self.units.values():
            if u.side == side:
                u.moved = False
                u.advanced = False
                u.fell_back = False

    def snapshot(self) -> GameState:
        return GameState(
            turn=self.turn,
            phase=self.phase,
            active_side=self.active_side,
            units=tuple(
                u.snapshot(
                    has_shot=u.unit_id in self.shot_this_phase,
                    has_fought=u.unit_id in self.fought_this_phase,
                    moved=u.unit_id in self.moved_this_phase,
                )
                for u in self.units.values()
            ),
        )


@dataclass(frozen=True)
class VolleyEvent:
    """One resolved attack — a shooting volley or a melee fight — for observers."""

    turn: int
    phase: str
    action: Action
    result: AttackResult


@dataclass(frozen=True)
class MoveEvent:
    """One resolved movement action for observers."""

    turn: int
    phase: str
    action: Action
    from_pos: tuple[int, int]
    to_pos: tuple[int, int]
    move_type: str


def run_scenario(
    scenario: Scenario,
    strategies: Mapping[str, Strategy],
    *,
    rng: random.Random | None = None,
    on_turn_start: Callable[[int, ScenarioTurn], None] | None = None,
    on_volley: Callable[[VolleyEvent], None] | None = None,
    on_move: Callable[[MoveEvent], None] | None = None,
) -> BattleState:
    """Play the scenario's turns and return the final battlefield state.

    ``strategies`` maps each side name to the `Strategy` that plays it; both
    "attacker" and "defender" must be present. ``rng`` is threaded into every
    dice roll for deterministic tests. The loop ends early the moment either
    side is wiped out.
    """
    for side in ("attacker", "defender"):
        if side not in strategies:
            raise EngineError(f"no strategy provided for the {side} side")
    dice = rng or random.Random()
    state = BattleState.from_scenario(scenario)

    for turn_number, turn in enumerate(scenario.turns, start=1):
        if state.battle_over:
            break
        state.turn = turn_number
        state.phase = turn.phase
        state.active_side = turn.active_side
        state.shot_this_phase = set()
        state.fought_this_phase = set()
        state.moved_this_phase = set()
        if turn.phase == "movement":
            state.reset_movement_flags(turn.active_side)
        if on_turn_start is not None:
            on_turn_start(turn_number, turn)
        if turn.phase == "fight":
            _run_fight_phase(state, strategies, turn_number, dice, on_volley)
        elif turn.phase == "movement":
            _run_movement_phase(state, strategies, turn_number, dice, on_move)
        else:
            _run_shooting_phase(state, strategies, turn_number, dice, on_volley)
    return state


def _run_shooting_phase(
    state: BattleState,
    strategies: Mapping[str, Strategy],
    turn_number: int,
    dice: random.Random,
    on_volley: Callable[[VolleyEvent], None] | None,
) -> None:
    """One side's shooting phase: every eligible unit activates once."""
    strategy = strategies[state.active_side]
    while True:
        snap = state.snapshot()
        if not snap.eligible_shooters() or not snap.surviving_enemies():
            break
        action = strategy.choose_action(snap)
        attacker, weapon, target = _validate_shoot(action, state)
        result = resolve_shooting(
            attacker.datasheet,
            attacker.models,
            weapon,
            target.datasheet,
            target.wounds_on_lead,
            target.models,
            rng=dice,
        )
        target.models = result.models_remaining
        target.wounds_on_lead = result.wounds_remaining_on_lead
        state.shot_this_phase.add(attacker.unit_id)
        if on_volley is not None:
            on_volley(
                VolleyEvent(turn=turn_number, phase=state.phase, action=action, result=result)
            )


def _run_fight_phase(
    state: BattleState,
    strategies: Mapping[str, Strategy],
    turn_number: int,
    dice: random.Random,
    on_volley: Callable[[VolleyEvent], None] | None,
) -> None:
    """One Fight phase: both sides act, alternating unit by unit (12.04).

    The side whose turn it is picks the first unit to fight; after each fight
    the pick passes to the other side; a side with nothing eligible passes
    back. Every eligible unit must fight (fighting is not optional), and
    casualties are applied fight by fight, so a unit picked later swings with
    only its surviving models — the phase's central lesson.

    No unit in the project's data has the Fights First ability, so the
    rulebook's Fights-First selection step is vacuously empty and this loop is
    the Resolve Remaining Combats step; the entry order matches the rule for
    that case (the active player, finding no Fights-First units, carries the
    first pick into remaining combats). Fights First is deferred with charges,
    which are what normally grants it (docs/design/fight-phase.md).
    """
    picker = state.active_side
    while not state.battle_over:
        overview = state.snapshot()
        if not overview.eligible_fighters("attacker") and not overview.eligible_fighters(
            "defender"
        ):
            break
        if not overview.eligible_fighters(picker):
            picker = opposing_side(picker)
            continue
        state.active_side = picker  # the side currently being asked to act
        action = strategies[picker].choose_action(state.snapshot())
        attacker, weapon, target = _validate_fight(action, state)
        result = resolve_melee(
            attacker.datasheet,
            attacker.models,
            weapon,
            target.datasheet,
            target.wounds_on_lead,
            target.models,
            rng=dice,
        )
        target.models = result.models_remaining
        target.wounds_on_lead = result.wounds_remaining_on_lead
        state.fought_this_phase.add(attacker.unit_id)
        if on_volley is not None:
            on_volley(
                VolleyEvent(turn=turn_number, phase=state.phase, action=action, result=result)
            )
        picker = opposing_side(picker)


def _run_movement_phase(
    state: BattleState,
    strategies: Mapping[str, Strategy],
    turn_number: int,
    dice: random.Random,
    on_move: Callable[[MoveEvent], None] | None,
) -> None:
    """One Movement phase: the active side moves its eligible units (09.04-09.07).

    The active side moves units one by one. Each unit picks one move type it is
    eligible for (Remain Stationary, Normal, Advance, or Fall Back). Moves update
    unit positions and set turn-level flags (moved, advanced, fell_back).
    """
    strategy = strategies[state.active_side]
    while not state.battle_over:
        overview = state.snapshot()
        if not overview.eligible_movers():
            break
        action = strategy.choose_action(overview)
        mover, move_type, destination = _validate_move(action, state)
        from_pos = mover.position
        mover.position = destination
        mover.moved = (destination != from_pos) or (move_type != "remain_stationary")
        if move_type == "advance":
            mover.advanced = True
        elif move_type == "fall_back":
            mover.fell_back = True
        state.moved_this_phase.add(mover.unit_id)
        if on_move is not None:
            on_move(
                MoveEvent(
                    turn=turn_number,
                    phase=state.phase,
                    action=action,
                    from_pos=from_pos,
                    to_pos=destination,
                    move_type=move_type,
                )
            )


def _is_engaged(unit: UnitRuntime, state: BattleState) -> bool:
    """True when any surviving enemy unit is within engagement range of ``unit``.

    Live-state engagement: destroyed units engage nothing, so a fight turn
    that wipes a combat frees its participants for later shooting turns.
    """
    return any(
        in_engagement_range(unit.position, enemy.position)
        for enemy in state.survivors(opposing_side(unit.side))
    )


def _acting_unit(action: Action, state: BattleState, verb: str) -> UnitRuntime:
    """The unit an action activates: must exist, belong to the acting side, and live."""
    attacker = state.units.get(action.attacker_unit_id)
    if attacker is None:
        raise EngineError(f"no unit {action.attacker_unit_id!r} on the battlefield")
    if attacker.side != state.active_side:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) belongs to the "
            f"{attacker.side}, who are not the active side"
        )
    if attacker.destroyed:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) is destroyed "
            f"and cannot {verb}"
        )
    return attacker


def _weapon_on(attacker: UnitRuntime, weapon_key: str) -> Weapon:
    """The named weapon on the acting unit's datasheet."""
    weapon = next((w for w in attacker.datasheet.weapons if w.name == weapon_key), None)
    if weapon is None:
        raise EngineError(f"{attacker.datasheet.display_name} has no weapon {weapon_key!r}")
    return weapon


def _enemy_unit(target_id: str, state: BattleState, attacks_name: str) -> UnitRuntime:
    """The action's target: must exist, be an enemy of the acting side, and live."""
    target = state.units.get(target_id)
    if target is None:
        raise EngineError(f"no unit {target_id!r} on the battlefield")
    if target.side != opposing_side(state.active_side):
        raise EngineError(
            f"{target.datasheet.display_name} ({target.unit_id!r}) is on your own "
            f"side — {attacks_name} target enemy units"
        )
    if target.destroyed:
        raise EngineError(
            f"{target.datasheet.display_name} ({target.unit_id!r}) is already "
            f"destroyed — pick a surviving target"
        )
    return target


def _validate_shoot(action: Action, state: BattleState) -> tuple[UnitRuntime, Weapon, UnitRuntime]:
    """Check a shooting action against the battle rules; return the resolved pieces."""
    if action.kind != "shoot":
        raise EngineError(
            f"action kind {action.kind!r} is not legal in a shooting phase — "
            f"activations here are 'shoot' actions"
        )
    attacker = _acting_unit(action, state, verb="shoot")
    if attacker.unit_id in state.shot_this_phase:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) already shot "
            f"this phase — each unit shoots once per shooting phase"
        )
    if attacker.advanced:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) advanced this turn "
            f"and cannot shoot (rule 10.04)"
        )
    if attacker.fell_back:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) fell back this turn "
            f"and cannot shoot (rule 10.04)"
        )
    weapon = _weapon_on(attacker, action.weapon_key)
    if weapon.type != "ranged":
        raise EngineError(
            f"{weapon.display_name} is a melee weapon and cannot be fired in the shooting phase"
        )
    carried = {w.name for w in shootable_weapons(attacker.datasheet, attacker.loadout)}
    if weapon.name not in carried:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) is not "
            f"carrying {weapon.display_name} in this scenario — its loadout is "
            f"{', '.join(sorted(carried))}"
        )
    target = _enemy_unit(action.target_unit_id, state, "shooting attacks")
    if _is_engaged(attacker, state):
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) is engaged "
            f"and cannot shoot (rule 10.04 — an engaged unit needs a "
            f"[CLOSE-QUARTERS] weapon, and none of this project's weapons is one)"
        )
    if _is_engaged(target, state):
        raise EngineError(
            f"{target.datasheet.display_name} ({target.unit_id!r}) is engaged — "
            f"shooting cannot target an engaged unit (rule 04.02)"
        )
    if not in_weapon_range(attacker.position, target.position, weapon):
        raise EngineError(
            f"{target.datasheet.display_name} ({target.unit_id!r}) is "
            f'{distance_inches(attacker.position, target.position)}" from '
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) — beyond "
            f"{weapon.display_name}'s {weapon.range}\" range (rule 04.02; "
            f'1 square = 2", ADR 0007)'
        )
    return attacker, weapon, target


def _validate_fight(action: Action, state: BattleState) -> tuple[UnitRuntime, Weapon, UnitRuntime]:
    """Check a fight action against the battle rules; return the resolved pieces."""
    if action.kind != "fight":
        raise EngineError(
            f"action kind {action.kind!r} is not legal in a fight phase — "
            f"activations here are 'fight' actions"
        )
    attacker = _acting_unit(action, state, verb="fight")
    if attacker.unit_id in state.fought_this_phase:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) already fought "
            f"this phase — each unit is selected to fight once per fight phase"
        )
    weapon = _weapon_on(attacker, action.weapon_key)
    if weapon.type != "melee":
        raise EngineError(
            f"{weapon.display_name} is a ranged weapon and cannot be swung in the "
            f"fight phase — fights are made with melee weapons"
        )
    carried = {w.name for w in melee_weapons(attacker.datasheet, attacker.loadout)}
    if weapon.name not in carried:
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) is not "
            f"carrying {weapon.display_name} in this scenario — its melee loadout "
            f"is {', '.join(sorted(carried))}"
        )
    target = _enemy_unit(action.target_unit_id, state, "melee attacks")
    if not in_engagement_range(attacker.position, target.position):
        raise EngineError(
            f"{attacker.datasheet.display_name} ({attacker.unit_id!r}) is not within "
            f"engagement range of {target.datasheet.display_name} ({target.unit_id!r}) "
            f"— melee attacks can only target a unit the attacker is engaged with "
            f"(adjacent squares, diagonals count)"
        )
    return attacker, weapon, target


def _validate_move(action: Action, state: BattleState) -> tuple[UnitRuntime, str, tuple[int, int]]:
    """Check a movement action against the rules; return mover, move_type, destination."""
    if action.kind != "move":
        raise EngineError(
            f"action kind {action.kind!r} is not legal in a movement phase — "
            f"activations here are 'move' actions"
        )
    mover = _acting_unit(action, state, verb="move")
    if mover.unit_id in state.moved_this_phase:
        raise EngineError(
            f"{mover.datasheet.display_name} ({mover.unit_id!r}) already moved "
            f"this phase — each unit moves once per movement phase"
        )
    move_type = action.move_type
    if move_type not in ("remain_stationary", "normal", "advance", "fall_back"):
        raise EngineError(
            f"move type {move_type!r} is not valid — must be remain_stationary, "
            f"normal, advance, or fall_back"
        )
    is_engaged = _is_engaged(mover, state)
    if move_type == "remain_stationary":
        destination = mover.position
        return mover, move_type, destination

    if move_type == "fall_back":
        if not is_engaged:
            raise EngineError(
                f"{mover.datasheet.display_name} ({mover.unit_id!r}) is not engaged "
                f"and cannot Fall Back"
            )
    elif is_engaged:
        raise EngineError(
            f"{mover.datasheet.display_name} ({mover.unit_id!r}) is engaged "
            f"and cannot make a {move_type.replace('_', ' ').title()} — must Fall Back "
            f"or Remain Stationary"
        )

    destination = action.destination
    if destination is None:
        raise EngineError(f"destination required for {move_type} move")
    col, row = destination
    if not (0 <= col < BATTLEFIELD_WIDTH and 0 <= row < BATTLEFIELD_HEIGHT):
        raise EngineError(f"destination {destination} is outside the battlefield bounds")
    for other in state.units.values():
        if not other.destroyed and other.unit_id != mover.unit_id and other.position == destination:
            raise EngineError(
                f"destination {destination} is already occupied by {other.datasheet.display_name}"
            )
    dist = chebyshev_squares(mover.position, destination)
    max_reach = (
        reach_squares(mover.datasheet.profile.movement + 6)
        if move_type == "advance"
        else reach_squares(mover.datasheet.profile.movement)
    )
    if dist > max_reach:
        raise EngineError(
            f"destination {destination} is {dist} squares from {mover.position} — "
            f"exceeds maximum reach of {max_reach} squares"
        )
    enemy_survivors = [u for u in state.units.values() if u.side != mover.side and not u.destroyed]
    if any(in_engagement_range(destination, e.position) for e in enemy_survivors):
        raise EngineError(
            f"{move_type} move to {destination} must end unengaged, but is within "
            f"engagement range of an enemy unit"
        )
    return mover, move_type, destination
