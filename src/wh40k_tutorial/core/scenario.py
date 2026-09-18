"""Scenario model: the pre-positioned battles that teach one concept each.

`load_scenario` / `load_scenario_by_id` deserialize `data/scenarios/*.json`
into frozen value types, validating eagerly in the same name-the-culprit
style as the faction loader: referenced factions are loaded, referenced
datasheets must exist, model counts must fit the datasheet's unit size,
positions must be on the battlefield grid and not collide, and any scripted
actions must be legal for the units on the board. Bad data fails at load
time, never mid-battle. Schema reference: `.claude/skills/add-scenario/SKILL.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from wh40k_tutorial.core.models import (
    FactionDataError,
    UnitDatasheet,
    Weapon,
    load_faction_by_name,
    melee_weapons,
    shootable_weapons,
)

# The battlefield the scenarios position their units on. The Rich UI draws
# this same grid (ui/shell.py defaults to these dimensions).
BATTLEFIELD_WIDTH = 12
BATTLEFIELD_HEIGHT = 8

SIDES = ("attacker", "defender")

# The phases a scenario turn may declare. Shooting arrived with v1; the
# fight phase is the first v2 mechanic (see docs/design/fight-phase.md).
_SUPPORTED_PHASES = ("shooting", "fight", "movement")
_VALID_MOVE_TYPES = ("remain_stationary", "normal", "advance", "fall_back")

# The grid's scale (ADR 0007): one square is 2 inches, and the distance
# between two squares is the Chebyshev distance — the number of king's moves —
# times the scale. Chosen so that adjacency (Chebyshev distance 1, diagonals
# count) is EXACTLY the 11th-edition 2" engagement range (03.04): the fight
# phase's long-standing adjacency convention stops being an approximation and
# becomes the theorem.
INCHES_PER_SQUARE = 2

# The 11th-edition engagement range: 2" horizontally (03.04; the 5" vertical
# arm is meaningless on a flat grid). Models mutually inside it — and their
# units — are engaged.
ENGAGEMENT_RANGE_INCHES = 2


def chebyshev_squares(a: tuple[int, int], b: tuple[int, int]) -> int:
    """The distance between two grid positions, in squares (king's moves)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def distance_inches(a: tuple[int, int], b: tuple[int, int]) -> int:
    """The distance between two grid positions on the tabletop scale."""
    return chebyshev_squares(a, b) * INCHES_PER_SQUARE


def reach_squares(inches: int) -> int:
    """How many squares an ``inches``-long reach covers — rounded DOWN.

    Flooring keeps ranges honest: a 9" reach covers 4 squares (8"), never a
    5th (10") the tabletop weapon could not touch. Applied to every distance
    the rules express in inches (weapon range, engagement range).
    """
    return inches // INCHES_PER_SQUARE


# Engagement in squares: reach_squares(2") = 1 — adjacency, diagonals count.
ENGAGEMENT_RANGE_SQUARES = reach_squares(ENGAGEMENT_RANGE_INCHES)


def in_engagement_range(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when two grid positions are within engagement range of each other.

    Chebyshev distance <= ENGAGEMENT_RANGE_SQUARES: orthogonal or diagonal
    neighbours count. The single definition of "engaged" — the loader's
    fight-turn check, the engine's eligibility and validation, and the
    strategies' target menus all defer here.
    """
    return chebyshev_squares(a, b) <= ENGAGEMENT_RANGE_SQUARES


def in_weapon_range(a: tuple[int, int], b: tuple[int, int], weapon: Weapon) -> bool:
    """True when a model at ``a`` can reach a target at ``b`` with ``weapon``.

    04.02: a shooting target must be within range of the weapon. The single
    definition of "in range" — the loader's script validation, the engine's
    shot validation, and the strategies' target menus all defer here.
    """
    return chebyshev_squares(a, b) <= reach_squares(weapon.range)


def legal_destinations(
    pos: tuple[int, int],
    move_type: str,
    movement_inches: int,
    *,
    advance_roll: int = 0,
    occupied: set[tuple[int, int]],
    enemy_positions: set[tuple[int, int]],
    width: int = BATTLEFIELD_WIDTH,
    height: int = BATTLEFIELD_HEIGHT,
) -> tuple[tuple[int, int], ...]:
    """All legal grid squares `pos` can reach under `move_type`.

    Rules 09.04-09.07:
    - remain_stationary: only `pos` itself
    - normal: within reach_squares(movement_inches), unoccupied, ending unengaged
    - advance: within reach_squares(movement_inches + advance_roll), unoccupied, ending unengaged
    - fall_back: within reach_squares(movement_inches), unoccupied, ending unengaged
    Sorted deterministically in reading order (row, then col).
    """
    if move_type == "remain_stationary":
        return (pos,)
    if move_type not in ("normal", "advance", "fall_back"):
        return ()
    max_reach = (
        reach_squares(movement_inches + advance_roll)
        if move_type == "advance"
        else reach_squares(movement_inches)
    )
    results: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            dest = (c, r)
            if dest in occupied and dest != pos:
                continue
            if chebyshev_squares(pos, dest) > max_reach:
                continue
            if any(in_engagement_range(dest, e) for e in enemy_positions):
                continue
            results.append(dest)
    return tuple(results)


# Who plays the non-player side: "scripted" replays the scenario's action
# lists; "heuristic" is the expected-damage AI (strategies/heuristic.py).
OPPONENT_STRATEGIES = ("scripted", "heuristic")


class ScenarioDataError(ValueError):
    """A scenario JSON file doesn't match the schema in the add-scenario skill."""


@dataclass(frozen=True)
class ScenarioUnit:
    """One unit as placed by the scenario, with its datasheet already resolved."""

    unit_id: str
    datasheet: UnitDatasheet
    position: tuple[int, int]  # (x, y) on the battlefield grid
    models: int
    # Per-scenario loadout override: the weapon keys every model carries in
    # THIS scenario instead of the datasheet's default_loadout. Empty means
    # "no override — use the datasheet's default". Same shape and rules as
    # default_loadout (see the add-scenario skill).
    loadout: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioSide:
    """Everything one side brings to the battle."""

    name: str  # "attacker" or "defender"
    faction: str
    units: tuple[ScenarioUnit, ...]


@dataclass(frozen=True)
class ScenarioAction:
    """One scripted action inside a turn entry — a shot, a fight, or a move.

    This is the *data-file* shape; `strategies.scripted` converts it into the
    protocol's `Action` at runtime (core stays free of the strategies layer).
    """

    attacker_unit_id: str
    # Weapon key on datasheet: ranged in shooting, melee in fights
    weapon: str = ""
    target_unit_id: str = ""
    move_type: str = ""  # "remain_stationary", "normal", "advance", "fall_back"
    destination: tuple[int, int] | None = None


@dataclass(frozen=True)
class ScenarioTurn:
    """One phase of one game turn.

    ``actions`` is the optional script for this turn: when the active side is
    played by ``ScriptedStrategy``, these are the shots it replays, in order.
    Turns the *player* acts in normally leave it empty — the human decides.
    """

    phase: str  # "shooting" or "fight"
    active_side: str  # whose turn it is; in a fight phase BOTH sides act — this side picks first
    narrate_before: str = ""
    actions: tuple[ScenarioAction, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """A whole tutorial scenario, loaded and cross-checked."""

    scenario_id: str
    title: str
    teaches: str
    intro: str
    player_side: str
    attacker: ScenarioSide
    defender: ScenarioSide
    turns: tuple[ScenarioTurn, ...]
    outro: str
    # How the non-player side is driven: "scripted" (default; replays the
    # turns' action lists) or "heuristic" (the expected-damage AI).
    opponent_strategy: str = "scripted"

    def side(self, name: str) -> ScenarioSide:
        if name == "attacker":
            return self.attacker
        if name == "defender":
            return self.defender
        raise KeyError(f"no side named {name!r}; sides are {SIDES}")

    @property
    def all_units(self) -> tuple[ScenarioUnit, ...]:
        return self.attacker.units + self.defender.units


def opposing_side(side: str) -> str:
    """The other side: attacker <-> defender."""
    if side == "attacker":
        return "defender"
    if side == "defender":
        return "attacker"
    raise KeyError(f"no side named {side!r}; sides are {SIDES}")


# ---------------------------------------------------------------------------
# Parsing helpers (same conventions as core/models.py)
# ---------------------------------------------------------------------------


def _get(mapping: dict, key: str, ctx: str) -> object:
    """Fetch a required field, failing with a message that names where it was missing."""
    try:
        return mapping[key]
    except (KeyError, TypeError):
        raise ScenarioDataError(f"missing required field {key!r} in {ctx}") from None


def _str_field(mapping: dict, key: str, ctx: str) -> str:
    value = _get(mapping, key, ctx)
    if not isinstance(value, str) or not value.strip():
        raise ScenarioDataError(f"{ctx}: field {key!r} must be a non-empty string, got {value!r}")
    return value


def _parse_position(raw: object, ctx: str) -> tuple[int, int]:
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in raw)
    ):
        raise ScenarioDataError(f"{ctx}: 'position' must be a two-integer [x, y], got {raw!r}")
    x, y = raw
    if not (0 <= x < BATTLEFIELD_WIDTH and 0 <= y < BATTLEFIELD_HEIGHT):
        raise ScenarioDataError(
            f"{ctx}: position ({x}, {y}) is off the "
            f"{BATTLEFIELD_WIDTH}x{BATTLEFIELD_HEIGHT} battlefield grid"
        )
    return (x, y)


def _parse_loadout_override(raw: object, sheet: UnitDatasheet, ctx: str) -> tuple[str, ...]:
    """Parse a unit's optional per-scenario loadout override.

    Mirrors the faction loader's ``default_loadout`` rules (keys must exist on
    the datasheet, coverage must be ``"all"`` in v1) plus one scenario-side
    requirement: the override must select at least one ranged weapon, because
    v1 units act only in the shooting phase and a shoot-nothing override is
    almost certainly an authoring mistake.
    """
    if not isinstance(raw, dict) or not raw:
        raise ScenarioDataError(
            f"{ctx}: 'loadout' must be a non-empty object of weapon-key -> 'all' — "
            f"omit the field entirely to use the datasheet's default loadout"
        )
    by_key = {w.name: w for w in sheet.weapons}
    for weapon_key, coverage in raw.items():
        if weapon_key not in by_key:
            raise ScenarioDataError(
                f"{ctx}.loadout: {sheet.display_name} has no weapon {weapon_key!r}; "
                f"available: {', '.join(sorted(by_key))}"
            )
        if coverage != "all":
            raise ScenarioDataError(
                f"{ctx}.loadout: only 'all' coverage is supported in v1, "
                f"got {coverage!r} for {weapon_key!r}"
            )
    override = tuple(raw)
    if all(by_key[key].type != "ranged" for key in override):
        raise ScenarioDataError(
            f"{ctx}.loadout: must include at least one ranged weapon — v1 units "
            f"act only in the shooting phase"
        )
    return override


def _parse_unit(
    raw: object, side_name: str, sheets: dict[str, UnitDatasheet], ctx: str
) -> ScenarioUnit:
    if not isinstance(raw, dict):
        raise ScenarioDataError(f"{ctx}: each unit must be an object, got {raw!r}")
    unit_id = _str_field(raw, "id", ctx)
    ctx = f"{ctx} (id {unit_id!r})"
    datasheet_key = _str_field(raw, "datasheet", ctx)
    sheet = sheets.get(datasheet_key)
    if sheet is None:
        raise ScenarioDataError(
            f"{ctx}: datasheet {datasheet_key!r} does not exist in the "
            f"{side_name} side's faction; available: {', '.join(sorted(sheets))}"
        )
    models = _get(raw, "models", ctx)
    if isinstance(models, bool) or not isinstance(models, int):
        raise ScenarioDataError(f"{ctx}: 'models' must be an integer, got {models!r}")
    hi = sheet.max_model_count
    if models < sheet.min_model_count or (hi is not None and models > hi):
        raise ScenarioDataError(
            f"{ctx}: {models} models does not fit {sheet.display_name}'s unit size "
            f"({sheet.min_model_count}..{hi})"
        )
    loadout_raw = raw.get("loadout")
    return ScenarioUnit(
        unit_id=unit_id,
        datasheet=sheet,
        position=_parse_position(_get(raw, "position", ctx), ctx),
        models=models,
        loadout=(() if loadout_raw is None else _parse_loadout_override(loadout_raw, sheet, ctx)),
    )


def _parse_side(data: dict, name: str, source: str) -> ScenarioSide:
    ctx = f"{source}.sides.{name}"
    raw = _get(data, name, f"{source}.sides")
    if not isinstance(raw, dict):
        raise ScenarioDataError(f"{ctx}: must be an object with 'faction' and 'units'")
    faction = _str_field(raw, "faction", ctx)
    try:
        sheets = load_faction_by_name(faction)
    except FactionDataError as exc:
        raise ScenarioDataError(f"{ctx}: cannot load faction {faction!r} — {exc}") from exc
    units_raw = _get(raw, "units", ctx)
    if not isinstance(units_raw, list) or not units_raw:
        raise ScenarioDataError(f"{ctx}: 'units' must be a non-empty list")
    units = tuple(
        _parse_unit(u, name, sheets, f"{ctx}.units[{i}]") for i, u in enumerate(units_raw)
    )
    return ScenarioSide(name=name, faction=faction, units=units)


def _cross_check_placement(scenario_units: tuple[ScenarioUnit, ...], source: str) -> None:
    seen_ids: dict[str, str] = {}
    seen_positions: dict[tuple[int, int], str] = {}
    for unit in scenario_units:
        if unit.unit_id in seen_ids:
            raise ScenarioDataError(
                f"{source}: unit id {unit.unit_id!r} is used more than once — "
                f"ids must be unique across both sides"
            )
        seen_ids[unit.unit_id] = unit.unit_id
        if unit.position in seen_positions:
            raise ScenarioDataError(
                f"{source}: units {seen_positions[unit.position]!r} and {unit.unit_id!r} "
                f"share position {unit.position} — one grid square holds one unit"
            )
        seen_positions[unit.position] = unit.unit_id


def _parse_action(
    raw: object,
    attacker_side: ScenarioSide,
    defender_side: ScenarioSide,
    active_side: str,
    phase: str,
    ctx: str,
) -> ScenarioAction:
    if not isinstance(raw, dict):
        raise ScenarioDataError(f"{ctx}: each action must be an object, got {raw!r}")
    if phase == "movement":
        unit_id = raw.get("unit") or raw.get("attacker")
        if not isinstance(unit_id, str) or not unit_id:
            raise ScenarioDataError(f"{ctx}: missing 'unit' field")
        move_type = _str_field(raw, "move_type", ctx)
        if move_type not in _VALID_MOVE_TYPES:
            raise ScenarioDataError(
                f"{ctx}: move_type {move_type!r} is not supported — "
                f"must be one of {', '.join(_VALID_MOVE_TYPES)}"
            )
        acting = attacker_side if active_side == "attacker" else defender_side
        enemies = defender_side if acting is attacker_side else attacker_side
        mover = next((u for u in acting.units if u.unit_id == unit_id), None)
        if mover is None:
            raise ScenarioDataError(
                f"{ctx}: unit {unit_id!r} is not a unit on the active ({acting.name}) side"
            )
        is_engaged = any(in_engagement_range(mover.position, e.position) for e in enemies.units)
        if move_type == "remain_stationary":
            destination = mover.position
        else:
            if move_type == "fall_back":
                if not is_engaged:
                    raise ScenarioDataError(
                        f"{ctx}: {mover.datasheet.display_name} ({unit_id!r}) cannot Fall Back "
                        f"because it is not engaged"
                    )
            elif is_engaged:
                name = mover.datasheet.display_name
                mt_title = move_type.replace("_", " ").title()
                raise ScenarioDataError(
                    f"{ctx}: {name} ({unit_id!r}) cannot make a {mt_title} "
                    f"because it is engaged — must Fall Back or Remain Stationary"
                )
            dest_raw = raw.get("destination")
            destination = _parse_position(dest_raw, f"{ctx}.destination")
            all_units = list(attacker_side.units) + list(defender_side.units)
            if any(u.position == destination and u.unit_id != unit_id for u in all_units):
                raise ScenarioDataError(
                    f"{ctx}: destination {destination} is already occupied by another unit"
                )
            dist = chebyshev_squares(mover.position, destination)
            max_reach = (
                reach_squares(mover.datasheet.profile.movement + 6)
                if move_type == "advance"
                else reach_squares(mover.datasheet.profile.movement)
            )
            if dist > max_reach:
                raise ScenarioDataError(
                    f"{ctx}: destination {destination} is {dist} squares from {mover.position} — "
                    f"exceeds max reach of {max_reach} squares for movement "
                    f'{mover.datasheet.profile.movement}"'
                    + (' + 6" advance' if move_type == "advance" else "")
                )
            if any(in_engagement_range(destination, e.position) for e in enemies.units):
                raise ScenarioDataError(
                    f"{ctx}: {move_type} move to {destination} must end unengaged, "
                    f"but is within engagement range of an enemy unit"
                )
        return ScenarioAction(
            attacker_unit_id=unit_id,
            move_type=move_type,
            destination=destination,
        )

    attacker_id = _str_field(raw, "attacker", ctx)
    weapon_key = _str_field(raw, "weapon", ctx)
    target_id = _str_field(raw, "target", ctx)
    if phase == "fight":
        # Both sides act in a fight turn, so a scripted fight belongs to
        # whichever side its acting unit is on.
        acting = next(
            (
                side
                for side in (attacker_side, defender_side)
                if any(u.unit_id == attacker_id for u in side.units)
            ),
            None,
        )
        if acting is None:
            raise ScenarioDataError(f"{ctx}: attacker {attacker_id!r} is not a unit on either side")
    else:
        acting = attacker_side if active_side == "attacker" else defender_side
    enemies = defender_side if acting is attacker_side else attacker_side
    shooter = next((u for u in acting.units if u.unit_id == attacker_id), None)
    if shooter is None:
        raise ScenarioDataError(
            f"{ctx}: attacker {attacker_id!r} is not a unit on the active ({acting.name}) side"
        )
    weapon = next((w for w in shooter.datasheet.weapons if w.name == weapon_key), None)
    if weapon is None:
        raise ScenarioDataError(
            f"{ctx}: {shooter.datasheet.display_name} has no weapon {weapon_key!r}"
        )
    needed = "melee" if phase == "fight" else "ranged"
    if weapon.type != needed:
        raise ScenarioDataError(
            f"{ctx}: {weapon.display_name} is a {weapon.type} weapon — scripted "
            f"actions in a {phase} turn need a {needed} weapon"
        )
    carried_weapons = (
        melee_weapons(shooter.datasheet, shooter.loadout)
        if phase == "fight"
        else shootable_weapons(shooter.datasheet, shooter.loadout)
    )
    carried = {w.name for w in carried_weapons}
    if weapon_key not in carried:
        raise ScenarioDataError(
            f"{ctx}: {shooter.datasheet.display_name} ({attacker_id!r}) is not "
            f"carrying {weapon.display_name} in this scenario — its loadout is "
            f"{', '.join(sorted(carried))}"
        )
    target = next((u for u in enemies.units if u.unit_id == target_id), None)
    if target is None:
        raise ScenarioDataError(
            f"{ctx}: target {target_id!r} is not a unit on the {enemies.name} side"
        )
    # Positions never change (no movement yet), so weapon range is a static
    # fact the loader can check outright (04.02 via ADR 0007): a scripted
    # shot at an out-of-reach target is an authoring bug that should fail
    # here, not mid-battle. Engagement, by contrast, is RUNTIME state — a
    # fight turn can destroy the engaging unit and free a legal shot — so
    # the engaged-shooting rules (10.04, 04.02) are the engine's to enforce,
    # on live positions and survivors.
    if phase != "fight" and not in_weapon_range(shooter.position, target.position, weapon):
        raise ScenarioDataError(
            f"{ctx}: {target.datasheet.display_name} ({target_id!r}) is "
            f'{distance_inches(shooter.position, target.position)}" from '
            f"{shooter.datasheet.display_name} ({attacker_id!r}) — beyond "
            f"{weapon.display_name}'s {weapon.range}\" range "
            f'(1 square = {INCHES_PER_SQUARE}", ADR 0007)'
        )
    return ScenarioAction(attacker_unit_id=attacker_id, weapon=weapon_key, target_unit_id=target_id)


def _parse_turn(
    raw: object, attacker: ScenarioSide, defender: ScenarioSide, ctx: str
) -> ScenarioTurn:
    if not isinstance(raw, dict):
        raise ScenarioDataError(f"{ctx}: each turn must be an object, got {raw!r}")
    phase = _str_field(raw, "phase", ctx)
    if phase not in _SUPPORTED_PHASES:
        raise ScenarioDataError(
            f"{ctx}: phase {phase!r} is not supported — v1 models only "
            f"{', '.join(_SUPPORTED_PHASES)} (see 'Scope discipline' in CLAUDE.md)"
        )
    active_side = _str_field(raw, "active_side", ctx)
    if active_side not in SIDES:
        raise ScenarioDataError(f"{ctx}: 'active_side' must be one of {SIDES}, got {active_side!r}")
    narrate = raw.get("narrate_before", "")
    if not isinstance(narrate, str):
        raise ScenarioDataError(f"{ctx}: 'narrate_before' must be a string, got {narrate!r}")
    actions_raw = raw.get("actions", [])
    if not isinstance(actions_raw, list):
        raise ScenarioDataError(f"{ctx}: 'actions' must be a list, got {actions_raw!r}")
    actions = tuple(
        _parse_action(a, attacker, defender, active_side, phase, f"{ctx}.actions[{i}]")
        for i, a in enumerate(actions_raw)
    )
    return ScenarioTurn(
        phase=phase, active_side=active_side, narrate_before=narrate, actions=actions
    )


def _parse_scenario(data: object, source: str) -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioDataError(f"{source}: top level must be a JSON object")
    scenario_id = _str_field(data, "id", source)
    player_side = _str_field(data, "player_side", source)
    if player_side not in SIDES:
        raise ScenarioDataError(
            f"{source}: 'player_side' must be one of {SIDES}, got {player_side!r}"
        )
    sides = _get(data, "sides", source)
    if not isinstance(sides, dict):
        raise ScenarioDataError(f"{source}: 'sides' must be an object with attacker and defender")
    attacker = _parse_side(sides, "attacker", source)
    defender = _parse_side(sides, "defender", source)
    _cross_check_placement(attacker.units + defender.units, source)
    turns_raw = _get(data, "turns", source)
    if not isinstance(turns_raw, list) or not turns_raw:
        raise ScenarioDataError(f"{source}: 'turns' must be a non-empty list")
    turns = tuple(
        _parse_turn(t, attacker, defender, f"{source}.turns[{i}]") for i, t in enumerate(turns_raw)
    )
    opponent_strategy = data.get("opponent_strategy", "scripted")
    if opponent_strategy not in OPPONENT_STRATEGIES:
        raise ScenarioDataError(
            f"{source}: 'opponent_strategy' must be one of {OPPONENT_STRATEGIES}, "
            f"got {opponent_strategy!r}"
        )
    if opponent_strategy == "heuristic":
        opponent = opposing_side(player_side)
        for i, turn in enumerate(turns):
            if not turn.actions:
                continue
            # A fight turn's actions always script the opponent (the loader
            # forbids player-side fight scripts below), so under a heuristic
            # opponent they are contradictory regardless of active_side.
            if turn.phase == "fight" or turn.active_side == opponent:
                raise ScenarioDataError(
                    f"{source}.turns[{i}]: scripted 'actions' for the {opponent} "
                    f"contradict opponent_strategy 'heuristic' — the AI picks its "
                    f"own shots and fights; remove the actions or drop the field"
                )
    fight_turns = [i for i, t in enumerate(turns) if t.phase == "fight"]
    if fight_turns:
        engaged = any(
            in_engagement_range(a.position, d.position)
            for a in attacker.units
            for d in defender.units
        )
        if not engaged:
            raise ScenarioDataError(
                f"{source}.turns[{fight_turns[0]}]: a fight turn needs at least one "
                f"engaged pair, but no attacker unit starts within engagement range "
                f"of a defender unit — place opposing units on adjacent squares "
                f"(within {ENGAGEMENT_RANGE_SQUARES} square, diagonals count)"
            )
        player_units = {
            u.unit_id for u in (attacker if player_side == "attacker" else defender).units
        }
        for i in fight_turns:
            for a in turns[i].actions:
                if a.attacker_unit_id in player_units:
                    raise ScenarioDataError(
                        f"{source}.turns[{i}]: fight action for "
                        f"{a.attacker_unit_id!r} scripts the player's own side — "
                        f"the player picks their own fights; script only the "
                        f"{opposing_side(player_side)}'s"
                    )
    return Scenario(
        scenario_id=scenario_id,
        title=_str_field(data, "title", source),
        teaches=_str_field(data, "teaches", source),
        intro=_str_field(data, "intro", source),
        player_side=player_side,
        attacker=attacker,
        defender=defender,
        turns=turns,
        outro=_str_field(data, "outro", source),
        opponent_strategy=opponent_strategy,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_scenario(path: str | Path) -> Scenario:
    """Load one scenario JSON file, validating it eagerly.

    Raises `ScenarioDataError` (a `ValueError`) naming the field, unit, or
    action at fault. The file's stem must match the scenario's ``id`` so
    ``wh40k play <id>`` always finds what ``wh40k list`` showed.
    """
    file = Path(path)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScenarioDataError(f"{file.name}: not valid JSON ({exc})") from exc
    scenario = _parse_scenario(data, source=file.name)
    if scenario.scenario_id != file.stem:
        raise ScenarioDataError(
            f"{file.name}: 'id' is {scenario.scenario_id!r} but must match the "
            f"file name ({file.stem!r})"
        )
    return scenario


def _packaged_scenarios_dir():
    return resources.files("wh40k_tutorial") / "data" / "scenarios"


def load_scenario_by_id(scenario_id: str) -> Scenario:
    """Load a packaged scenario by id, e.g. ``load_scenario_by_id("01_first_shots")``."""
    scenarios_dir = _packaged_scenarios_dir()
    candidate = scenarios_dir / f"{scenario_id}.json"
    if not candidate.is_file():
        available = sorted(
            p.name.removesuffix(".json")
            for p in scenarios_dir.iterdir()
            if p.name.endswith(".json")
        )
        raise ScenarioDataError(
            f"no scenario named {scenario_id!r}; available: {', '.join(available)}"
        )
    scenario = _parse_scenario(
        json.loads(candidate.read_text(encoding="utf-8")), source=f"{scenario_id}.json"
    )
    if scenario.scenario_id != scenario_id:
        raise ScenarioDataError(
            f"{scenario_id}.json: 'id' is {scenario.scenario_id!r} but must match the "
            f"file name ({scenario_id!r})"
        )
    return scenario


def available_scenarios() -> list[Scenario]:
    """Load every packaged scenario, in file-name (intended play) order.

    Packaged data is curated, so a malformed file fails loudly here rather
    than being silently skipped — `wh40k list` surfaces the loader's message.
    """
    scenarios_dir = _packaged_scenarios_dir()
    names = sorted(p.name for p in scenarios_dir.iterdir() if p.name.endswith(".json"))
    return [load_scenario_by_id(name.removesuffix(".json")) for name in names]
