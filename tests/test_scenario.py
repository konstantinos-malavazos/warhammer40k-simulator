"""Tests for the scenario loader (build phase 5).

Same philosophy as the faction-loader tests: a valid file round-trips into
frozen value types with datasheets resolved, and malformed data fails loudly
at load time with a message naming the culprit — never mid-battle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wh40k_tutorial.core.scenario import (
    BATTLEFIELD_HEIGHT,
    BATTLEFIELD_WIDTH,
    INCHES_PER_SQUARE,
    Scenario,
    ScenarioDataError,
    available_scenarios,
    chebyshev_squares,
    distance_inches,
    in_engagement_range,
    in_weapon_range,
    load_scenario,
    load_scenario_by_id,
    opposing_side,
    reach_squares,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _base() -> dict:
    """A minimal valid scenario against the packaged factions."""
    return {
        "id": "test_scenario",
        "title": "Test Scenario",
        "teaches": "testing the loader.",
        "intro": "An intro.",
        "player_side": "attacker",
        "sides": {
            "attacker": {
                "faction": "space_marines",
                "units": [
                    {
                        "id": "marines_1",
                        "datasheet": "intercessor_squad",
                        "position": [3, 4],
                        "models": 5,
                    }
                ],
            },
            "defender": {
                "faction": "tyranids",
                "units": [
                    {
                        "id": "termagants_1",
                        "datasheet": "termagants",
                        "position": [9, 4],
                        "models": 10,
                    }
                ],
            },
        },
        "turns": [{"phase": "shooting", "active_side": "attacker", "narrate_before": "Fire."}],
        "outro": "An outro.",
    }


def _load(tmp_path: Path, data: dict, filename: str | None = None) -> Scenario:
    name = filename or f"{data.get('id', 'broken')}.json"
    file = tmp_path / name
    file.write_text(json.dumps(data), encoding="utf-8")
    return load_scenario(file)


# ---------------------------------------------------------------------------
# Valid data round-trips
# ---------------------------------------------------------------------------


class TestValidScenario:
    def test_base_fixture_is_valid(self, tmp_path: Path) -> None:
        scenario = _load(tmp_path, _base())
        assert scenario.scenario_id == "test_scenario"
        assert scenario.title == "Test Scenario"
        assert scenario.player_side == "attacker"
        assert scenario.attacker.faction == "space_marines"
        assert scenario.defender.faction == "tyranids"

    def test_datasheets_are_resolved(self, tmp_path: Path) -> None:
        scenario = _load(tmp_path, _base())
        marines = scenario.attacker.units[0]
        gants = scenario.defender.units[0]
        assert marines.datasheet.display_name == "Intercessor Squad"
        assert marines.position == (3, 4)
        assert marines.models == 5
        assert gants.datasheet.display_name == "Termagants"
        assert gants.models == 10

    def test_turns_parse(self, tmp_path: Path) -> None:
        scenario = _load(tmp_path, _base())
        (turn,) = scenario.turns
        assert turn.phase == "shooting"
        assert turn.active_side == "attacker"
        assert turn.narrate_before == "Fire."
        assert turn.actions == ()

    def test_scripted_actions_parse_and_validate(self, tmp_path: Path) -> None:
        data = _base()
        data["turns"].append(
            {
                "phase": "shooting",
                "active_side": "defender",
                "actions": [
                    {"attacker": "termagants_1", "weapon": "fleshborer", "target": "marines_1"}
                ],
            }
        )
        scenario = _load(tmp_path, data)
        (action,) = scenario.turns[1].actions
        assert action.attacker_unit_id == "termagants_1"
        assert action.weapon == "fleshborer"
        assert action.target_unit_id == "marines_1"

    def test_side_lookup_and_opposing(self, tmp_path: Path) -> None:
        scenario = _load(tmp_path, _base())
        assert scenario.side("defender").faction == "tyranids"
        assert opposing_side("attacker") == "defender"
        assert opposing_side("defender") == "attacker"
        with pytest.raises(KeyError):
            scenario.side("bystander")
        with pytest.raises(KeyError):
            opposing_side("bystander")


class TestPackagedScenarios:
    def test_all_ten_scenarios_are_listed(self) -> None:
        assert [s.scenario_id for s in available_scenarios()] == [
            "01_first_shots",
            "02_tougher_targets",
            "03_piercing_armour",
            "04_lethal_hits",
            "05_sustained_hits",
            "06_return_fire",
            "07_devastating_wounds",
            "08_first_blood",
            "09_pick_your_fights",
            "10_position_and_fire",
        ]

    def test_pick_your_fights_is_two_disjoint_combats_vs_the_ai(self) -> None:
        scenario = load_scenario_by_id("09_pick_your_fights")
        assert scenario.opponent_strategy == "heuristic"
        assert len(scenario.turns) == 1
        assert scenario.turns[0].phase == "fight"
        assert scenario.turns[0].actions == ()  # the AI picks its own fights
        left, right = scenario.attacker.units
        imm, warr = scenario.defender.units
        # Two separate combats: each mob engages exactly one enemy unit.
        assert in_engagement_range(left.position, imm.position)
        assert in_engagement_range(right.position, warr.position)
        assert not in_engagement_range(left.position, warr.position)
        assert not in_engagement_range(right.position, imm.position)

    def test_first_blood_is_the_first_fight_scenario(self) -> None:
        scenario = load_scenario_by_id("08_first_blood")
        assert scenario.player_side == "attacker"
        assert scenario.attacker.faction == "orks"
        assert scenario.defender.units[0].datasheet.key == "intercessor_squad"
        turn = scenario.turns[0]
        assert len(scenario.turns) == 1
        assert turn.phase == "fight"
        assert turn.active_side == "attacker"  # the player picks first
        # The starting positions are engaged — the loader enforces this too.
        assert in_engagement_range(
            scenario.attacker.units[0].position, scenario.defender.units[0].position
        )
        # The script covers the opponent only; the player picks their own fight.
        assert [a.attacker_unit_id for a in turn.actions] == ["marines_1"]
        assert turn.actions[0].weapon == "close_combat_weapon"

    def test_devastating_wounds_arms_the_arc_rifle_override(self) -> None:
        scenario = load_scenario_by_id("07_devastating_wounds")
        rangers = scenario.attacker.units[0]
        assert rangers.loadout == ("arc_rifle",)
        arc = next(w for w in rangers.datasheet.weapons if w.name == "arc_rifle")
        assert "devastating_wounds" in arc.keywords
        assert arc.damage.is_variable  # D3 — the point of the lesson
        assert scenario.defender.units[0].datasheet.key == "intercessor_squad"

    def test_tougher_targets_offers_two_toughness_values(self) -> None:
        scenario = load_scenario_by_id("02_tougher_targets")
        defenders = scenario.defender.units
        toughness = sorted(u.datasheet.profile.toughness for u in defenders)
        assert toughness == [4, 5]  # the whole lesson is this contrast
        assert len(scenario.turns) == 2

    def test_piercing_armour_defender_has_the_invulnerable(self) -> None:
        scenario = load_scenario_by_id("03_piercing_armour")
        rangers = scenario.defender.units[0].datasheet
        assert rangers.profile.invulnerable_save == 5
        assert scenario.player_side == "attacker"

    def test_lethal_hits_attacker_carries_the_keyword(self) -> None:
        scenario = load_scenario_by_id("04_lethal_hits")
        flayer = scenario.attacker.units[0].datasheet.weapons[0]
        assert "lethal_hits" in flayer.keywords

    def test_sustained_hits_equips_the_tesla_override(self) -> None:
        scenario = load_scenario_by_id("05_sustained_hits")
        immortals = scenario.attacker.units[0]
        assert immortals.loadout == ("tesla_carbine",)
        tesla = next(w for w in immortals.datasheet.weapons if w.name == "tesla_carbine")
        assert "sustained_hits_2" in tesla.keywords
        assert scenario.player_side == "attacker"
        assert len(scenario.turns) == 2

    def test_return_fire_is_the_first_two_sided_scenario(self) -> None:
        scenario = load_scenario_by_id("06_return_fire")
        assert scenario.opponent_strategy == "heuristic"
        assert scenario.defender.faction == "tau_empire"
        assert len(scenario.attacker.units) == 2
        assert [t.active_side for t in scenario.turns] == [
            "attacker",
            "defender",
            "attacker",
            "defender",
        ]
        assert all(t.actions == () for t in scenario.turns)

    def test_first_shots_loads_by_id(self) -> None:
        scenario = load_scenario_by_id("01_first_shots")
        assert scenario.title == "First Shots"
        assert scenario.player_side == "attacker"
        assert scenario.attacker.units[0].datasheet.display_name == "Intercessor Squad"
        assert scenario.defender.units[0].models == 10
        assert scenario.turns[0].phase == "shooting"

    def test_unknown_id_names_the_available_ones(self) -> None:
        with pytest.raises(ScenarioDataError, match="01_first_shots"):
            load_scenario_by_id("99_does_not_exist")

    def test_available_scenarios_includes_first_shots(self) -> None:
        ids = [s.scenario_id for s in available_scenarios()]
        assert "01_first_shots" in ids
        assert ids == sorted(ids)  # file-name order = intended play order


# ---------------------------------------------------------------------------
# Malformed data fails loudly, naming the culprit
# ---------------------------------------------------------------------------


class TestMalformedScenario:
    def test_invalid_json(self, tmp_path: Path) -> None:
        file = tmp_path / "broken.json"
        file.write_text("{not json", encoding="utf-8")
        with pytest.raises(ScenarioDataError, match="not valid JSON"):
            load_scenario(file)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        data = _base()
        del data["title"]
        with pytest.raises(ScenarioDataError, match="'title'"):
            _load(tmp_path, data)

    def test_id_must_match_filename(self, tmp_path: Path) -> None:
        with pytest.raises(ScenarioDataError, match="must match the file name"):
            _load(tmp_path, _base(), filename="something_else.json")

    def test_bad_player_side(self, tmp_path: Path) -> None:
        data = _base()
        data["player_side"] = "spectator"
        with pytest.raises(ScenarioDataError, match="player_side"):
            _load(tmp_path, data)

    def test_missing_side(self, tmp_path: Path) -> None:
        data = _base()
        del data["sides"]["defender"]
        with pytest.raises(ScenarioDataError, match="defender"):
            _load(tmp_path, data)

    def test_unknown_faction(self, tmp_path: Path) -> None:
        data = _base()
        data["sides"]["defender"]["faction"] = "squats"
        with pytest.raises(ScenarioDataError, match="squats"):
            _load(tmp_path, data)

    def test_unknown_datasheet_names_available(self, tmp_path: Path) -> None:
        data = _base()
        data["sides"]["attacker"]["units"][0]["datasheet"] = "terminator_squad"
        with pytest.raises(ScenarioDataError, match=r"terminator_squad.*intercessor_squad"):
            _load(tmp_path, data)

    @pytest.mark.parametrize("models", [4, 11])  # Intercessors are 5..10
    def test_models_outside_unit_size(self, tmp_path: Path, models: int) -> None:
        data = _base()
        data["sides"]["attacker"]["units"][0]["models"] = models
        with pytest.raises(ScenarioDataError, match="unit size"):
            _load(tmp_path, data)

    def test_position_off_grid(self, tmp_path: Path) -> None:
        data = _base()
        data["sides"]["attacker"]["units"][0]["position"] = [BATTLEFIELD_WIDTH, 4]
        with pytest.raises(ScenarioDataError, match=f"{BATTLEFIELD_WIDTH}x{BATTLEFIELD_HEIGHT}"):
            _load(tmp_path, data)

    def test_position_malformed(self, tmp_path: Path) -> None:
        data = _base()
        data["sides"]["attacker"]["units"][0]["position"] = [3]
        with pytest.raises(ScenarioDataError, match="two-integer"):
            _load(tmp_path, data)

    def test_duplicate_unit_id_across_sides(self, tmp_path: Path) -> None:
        data = _base()
        data["sides"]["defender"]["units"][0]["id"] = "marines_1"
        with pytest.raises(ScenarioDataError, match="more than once"):
            _load(tmp_path, data)

    def test_duplicate_position(self, tmp_path: Path) -> None:
        data = _base()
        data["sides"]["defender"]["units"][0]["position"] = [3, 4]
        with pytest.raises(ScenarioDataError, match="share position"):
            _load(tmp_path, data)

    def test_empty_turns(self, tmp_path: Path) -> None:
        data = _base()
        data["turns"] = []
        with pytest.raises(ScenarioDataError, match="turns"):
            _load(tmp_path, data)

    def test_unsupported_phase_mentions_v1_scope(self, tmp_path: Path) -> None:
        data = _base()
        data["turns"][0]["phase"] = "charge"
        with pytest.raises(ScenarioDataError, match=r"charge.*Scope discipline"):
            _load(tmp_path, data)

    def test_bad_active_side(self, tmp_path: Path) -> None:
        data = _base()
        data["turns"][0]["active_side"] = "everyone"
        with pytest.raises(ScenarioDataError, match="active_side"):
            _load(tmp_path, data)


class TestMalformedScriptedActions:
    def _with_action(self, action: dict) -> dict:
        data = _base()
        data["turns"][0]["actions"] = [action]
        return data

    def test_attacker_not_on_active_side(self, tmp_path: Path) -> None:
        action = {"attacker": "termagants_1", "weapon": "fleshborer", "target": "marines_1"}
        with pytest.raises(ScenarioDataError, match="not a unit on the active"):
            _load(tmp_path, self._with_action(action))

    def test_unknown_weapon(self, tmp_path: Path) -> None:
        action = {"attacker": "marines_1", "weapon": "plasma_gun", "target": "termagants_1"}
        with pytest.raises(ScenarioDataError, match="plasma_gun"):
            _load(tmp_path, self._with_action(action))

    def test_melee_weapon_rejected(self, tmp_path: Path) -> None:
        action = {
            "attacker": "marines_1",
            "weapon": "close_combat_weapon",
            "target": "termagants_1",
        }
        with pytest.raises(ScenarioDataError, match="melee"):
            _load(tmp_path, self._with_action(action))

    def test_unknown_target(self, tmp_path: Path) -> None:
        action = {"attacker": "marines_1", "weapon": "bolt_rifle", "target": "hive_tyrant_1"}
        with pytest.raises(ScenarioDataError, match="hive_tyrant_1"):
            _load(tmp_path, self._with_action(action))

    def test_scripted_shot_beyond_weapon_range(self, tmp_path: Path) -> None:
        """Positions are static, so range is checked at load time (ADR 0007):
        a Fleshborer (18" = 9 squares) cannot script a shot across 11 squares."""
        data = _base()
        data["sides"]["attacker"]["units"][0]["position"] = [0, 0]
        data["sides"]["defender"]["units"][0]["position"] = [11, 7]
        data["turns"].append(
            {
                "phase": "shooting",
                "active_side": "defender",
                "actions": [
                    {
                        "attacker": "termagants_1",
                        "weapon": "fleshborer",
                        "target": "marines_1",
                    }
                ],
            }
        )
        with pytest.raises(ScenarioDataError, match=r'22" .*beyond.*18" range'):
            _load(tmp_path, data)


# ---------------------------------------------------------------------------
# Per-scenario loadout overrides
# ---------------------------------------------------------------------------


def _with_necron_attacker(data: dict) -> dict:
    """Swap the attacker for Immortals, who carry two ranged guns."""
    data["sides"]["attacker"] = {
        "faction": "necrons",
        "units": [{"id": "immortals_1", "datasheet": "immortals", "position": [3, 4], "models": 5}],
    }
    return data


class TestLoadoutOverride:
    def test_no_override_leaves_loadout_empty(self, tmp_path: Path) -> None:
        scenario = _load(tmp_path, _base())
        assert scenario.attacker.units[0].loadout == ()

    def test_valid_override_round_trips(self, tmp_path: Path) -> None:
        data = _with_necron_attacker(_base())
        data["sides"]["attacker"]["units"][0]["loadout"] = {"tesla_carbine": "all"}
        scenario = _load(tmp_path, data)
        assert scenario.attacker.units[0].loadout == ("tesla_carbine",)

    def test_unknown_weapon_names_the_available_ones(self, tmp_path: Path) -> None:
        data = _with_necron_attacker(_base())
        data["sides"]["attacker"]["units"][0]["loadout"] = {"doomsday_cannon": "all"}
        with pytest.raises(ScenarioDataError, match=r"doomsday_cannon.*tesla_carbine"):
            _load(tmp_path, data)

    def test_partial_coverage_rejected_in_v1(self, tmp_path: Path) -> None:
        data = _with_necron_attacker(_base())
        data["sides"]["attacker"]["units"][0]["loadout"] = {"tesla_carbine": "half"}
        with pytest.raises(ScenarioDataError, match="only 'all'"):
            _load(tmp_path, data)

    def test_empty_override_rejected(self, tmp_path: Path) -> None:
        data = _with_necron_attacker(_base())
        data["sides"]["attacker"]["units"][0]["loadout"] = {}
        with pytest.raises(ScenarioDataError, match="non-empty"):
            _load(tmp_path, data)

    def test_melee_only_override_rejected(self, tmp_path: Path) -> None:
        data = _with_necron_attacker(_base())
        data["sides"]["attacker"]["units"][0]["loadout"] = {"close_combat_weapon": "all"}
        with pytest.raises(ScenarioDataError, match="at least one ranged"):
            _load(tmp_path, data)

    def test_scripted_action_outside_the_override_rejected(self, tmp_path: Path) -> None:
        data = _with_necron_attacker(_base())
        data["sides"]["attacker"]["units"][0]["loadout"] = {"tesla_carbine": "all"}
        data["turns"][0]["actions"] = [
            {"attacker": "immortals_1", "weapon": "gauss_blaster", "target": "termagants_1"}
        ]
        with pytest.raises(ScenarioDataError, match=r"not\s+carrying.*tesla_carbine"):
            _load(tmp_path, data)

    def test_scripted_action_outside_the_default_loadout_rejected(self, tmp_path: Path) -> None:
        # No override: Immortals default to the gauss blaster, so a script
        # firing the tesla carbine is an authoring inconsistency.
        data = _with_necron_attacker(_base())
        data["turns"][0]["actions"] = [
            {"attacker": "immortals_1", "weapon": "tesla_carbine", "target": "termagants_1"}
        ]
        with pytest.raises(ScenarioDataError, match=r"not\s+carrying.*gauss_blaster"):
            _load(tmp_path, data)

    def test_scripted_action_inside_the_override_accepted(self, tmp_path: Path) -> None:
        data = _with_necron_attacker(_base())
        data["sides"]["attacker"]["units"][0]["loadout"] = {"tesla_carbine": "all"}
        data["turns"][0]["actions"] = [
            {"attacker": "immortals_1", "weapon": "tesla_carbine", "target": "termagants_1"}
        ]
        (action,) = _load(tmp_path, data).turns[0].actions
        assert action.weapon == "tesla_carbine"


# ---------------------------------------------------------------------------
# The opponent_strategy field
# ---------------------------------------------------------------------------


class TestOpponentStrategy:
    def test_defaults_to_scripted(self, tmp_path: Path) -> None:
        assert _load(tmp_path, _base()).opponent_strategy == "scripted"

    def test_heuristic_is_accepted(self, tmp_path: Path) -> None:
        data = _base()
        data["opponent_strategy"] = "heuristic"
        assert _load(tmp_path, data).opponent_strategy == "heuristic"

    def test_unknown_strategy_rejected(self, tmp_path: Path) -> None:
        data = _base()
        data["opponent_strategy"] = "psychic"
        with pytest.raises(ScenarioDataError, match="opponent_strategy"):
            _load(tmp_path, data)

    def test_heuristic_rejects_scripted_actions_for_the_opponent(self, tmp_path: Path) -> None:
        data = _base()
        data["opponent_strategy"] = "heuristic"
        data["turns"].append(
            {
                "phase": "shooting",
                "active_side": "defender",
                "actions": [
                    {"attacker": "termagants_1", "weapon": "fleshborer", "target": "marines_1"}
                ],
            }
        )
        with pytest.raises(ScenarioDataError, match="contradict"):
            _load(tmp_path, data)

    def test_heuristic_allows_player_side_actions(self, tmp_path: Path) -> None:
        # Actions on the *player's* turns are dead data but not a
        # contradiction — only the opponent's script conflicts with the AI.
        data = _base()
        data["opponent_strategy"] = "heuristic"
        data["turns"][0]["actions"] = [
            {"attacker": "marines_1", "weapon": "bolt_rifle", "target": "termagants_1"}
        ]
        assert _load(tmp_path, data).opponent_strategy == "heuristic"


# ---------------------------------------------------------------------------
# Fight turns: engagement, melee scripting, and the conventions around them
# ---------------------------------------------------------------------------


def _fight_base() -> dict:
    """The minimal scenario, repositioned into engagement and given a fight turn."""
    data = _base()
    data["sides"]["attacker"]["units"][0]["position"] = [4, 4]
    data["sides"]["defender"]["units"][0]["position"] = [5, 4]
    data["turns"] = [{"phase": "fight", "active_side": "attacker"}]
    return data


class TestDistanceModel:
    """ADR 0007: 1 square = 2", distance is Chebyshev x scale, reach floors."""

    @pytest.mark.parametrize(
        ("a", "b", "squares"),
        [
            ((0, 0), (0, 0), 0),
            ((3, 4), (9, 4), 6),  # scenario 01's shot: 6 squares = 12"
            ((2, 4), (9, 2), 7),  # diagonal-ish: the larger axis rules
            ((0, 0), (11, 7), 11),  # the grid's corner-to-corner maximum
            ((4, 4), (5, 5), 1),  # diagonal neighbour is ONE king's move
        ],
    )
    def test_chebyshev_and_inches(
        self, a: tuple[int, int], b: tuple[int, int], squares: int
    ) -> None:
        assert chebyshev_squares(a, b) == squares
        assert chebyshev_squares(b, a) == squares  # symmetric
        assert distance_inches(a, b) == squares * INCHES_PER_SQUARE

    @pytest.mark.parametrize(
        ("inches", "squares"),
        [
            (2, 1),  # engagement range: adjacency
            (3, 1),  # floors — an odd reach never gains a square
            (9, 4),  # a 9" pistol reaches 8", not 10"
            (12, 6),
            (18, 9),  # shoota / fleshborer
            (24, 12),  # bolt rifle covers the whole 12x8 grid
            (30, 15),
        ],
    )
    def test_reach_floors(self, inches: int, squares: int) -> None:
        assert reach_squares(inches) == squares

    def test_weapon_range_boundary(self) -> None:
        sheet = load_scenario_by_id("01_first_shots").defender.units[0].datasheet
        fleshborer = next(w for w in sheet.weapons if w.name == "fleshborer")  # 18"
        assert in_weapon_range((0, 4), (9, 4), fleshborer)  # 9 squares = 18": exactly in
        assert not in_weapon_range((0, 4), (10, 4), fleshborer)  # 10 squares = 20": out

    def test_engagement_is_the_two_inch_reach(self) -> None:
        """Adjacency IS the 2" engagement range — the theorem ADR 0007 buys."""
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                a, b = (5, 4), (5 + dx, 4 + dy)
                assert in_engagement_range(a, b) == (distance_inches(a, b) <= 2)


class TestEngagementGeometry:
    @pytest.mark.parametrize(
        ("a", "b", "engaged"),
        [
            ((4, 4), (5, 4), True),  # orthogonal neighbour
            ((4, 4), (5, 5), True),  # diagonal neighbour
            ((4, 4), (4, 3), True),
            ((4, 4), (6, 4), False),  # one square of daylight
            ((4, 4), (6, 6), False),
            ((4, 4), (4, 4), True),  # same square (placement forbids it anyway)
        ],
    )
    def test_adjacency_is_engagement(
        self, a: tuple[int, int], b: tuple[int, int], engaged: bool
    ) -> None:
        assert in_engagement_range(a, b) is engaged
        assert in_engagement_range(b, a) is engaged  # symmetric


class TestFightTurns:
    def test_fight_turn_with_an_engaged_pair_loads(self, tmp_path: Path) -> None:
        scenario = _load(tmp_path, _fight_base())
        assert scenario.turns[0].phase == "fight"

    def test_diagonal_engagement_counts(self, tmp_path: Path) -> None:
        data = _fight_base()
        data["sides"]["defender"]["units"][0]["position"] = [5, 5]
        assert _load(tmp_path, data).turns[0].phase == "fight"

    def test_fight_turn_without_an_engaged_pair_is_rejected(self, tmp_path: Path) -> None:
        data = _fight_base()
        data["sides"]["defender"]["units"][0]["position"] = [9, 4]
        with pytest.raises(ScenarioDataError, match="engaged pair"):
            _load(tmp_path, data)

    def test_scripted_fight_needs_a_melee_weapon(self, tmp_path: Path) -> None:
        data = _fight_base()
        data["turns"][0]["actions"] = [
            {"attacker": "termagants_1", "weapon": "fleshborer", "target": "marines_1"}
        ]
        with pytest.raises(ScenarioDataError, match="need a melee weapon"):
            _load(tmp_path, data)

    def test_scripted_ranged_action_still_needs_a_ranged_weapon(self, tmp_path: Path) -> None:
        data = _base()
        data["turns"][0]["actions"] = [
            {"attacker": "marines_1", "weapon": "close_combat_weapon", "target": "termagants_1"}
        ]
        with pytest.raises(ScenarioDataError, match="need a ranged weapon"):
            _load(tmp_path, data)

    def test_fight_turn_scripts_the_opponent_even_when_the_player_picks_first(
        self, tmp_path: Path
    ) -> None:
        """Both sides act in a fight turn: the actions array scripts the
        opponent's picks regardless of whose turn (active_side) it is."""
        data = _fight_base()
        data["turns"][0]["actions"] = [
            {"attacker": "termagants_1", "weapon": "claws_and_teeth", "target": "marines_1"}
        ]
        scenario = _load(tmp_path, data)
        action = scenario.turns[0].actions[0]
        assert action.attacker_unit_id == "termagants_1"
        assert action.weapon == "claws_and_teeth"

    def test_fight_action_may_not_script_the_players_side(self, tmp_path: Path) -> None:
        data = _fight_base()
        data["turns"][0]["actions"] = [
            {"attacker": "marines_1", "weapon": "close_combat_weapon", "target": "termagants_1"}
        ]
        with pytest.raises(ScenarioDataError, match="player picks their own fights"):
            _load(tmp_path, data)

    def test_heuristic_opponent_may_take_a_fight_turn(self, tmp_path: Path) -> None:
        data = _fight_base()
        data["opponent_strategy"] = "heuristic"
        scenario = _load(tmp_path, data)
        assert scenario.opponent_strategy == "heuristic"
        assert scenario.turns[0].phase == "fight"

    def test_heuristic_opponent_rejects_scripted_fight_actions(self, tmp_path: Path) -> None:
        """Fight-turn actions always script the opponent, so under a heuristic
        opponent they contradict it even on the player's own turn."""
        data = _fight_base()
        data["opponent_strategy"] = "heuristic"
        data["turns"][0]["actions"] = [
            {"attacker": "termagants_1", "weapon": "claws_and_teeth", "target": "marines_1"}
        ]
        with pytest.raises(ScenarioDataError, match="own shots and fights"):
            _load(tmp_path, data)

    def test_unknown_phase_is_still_rejected(self, tmp_path: Path) -> None:
        data = _base()
        data["turns"][0]["phase"] = "psychic"
        with pytest.raises(ScenarioDataError, match="'psychic'"):
            _load(tmp_path, data)
