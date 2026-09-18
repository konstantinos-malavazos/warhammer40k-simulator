from __future__ import annotations

import random

import pytest

from wh40k_tutorial.core.models import (
    load_faction_by_name,
)
from wh40k_tutorial.core.scenario import (
    Scenario,
    ScenarioAction,
    ScenarioSide,
    ScenarioTurn,
    ScenarioUnit,
    load_scenario_by_id,
)
from wh40k_tutorial.engine import EngineError, MoveEvent, run_scenario
from wh40k_tutorial.strategies.base import Action
from wh40k_tutorial.strategies.heuristic import HeuristicStrategy
from wh40k_tutorial.strategies.scripted import (
    ScriptedStrategy,
    scripted_actions_for,
)


def _marines(
    unit_id: str = "m1", pos: tuple[int, int] = (2, 2), models: int = 5
) -> ScenarioUnit:
    faction = load_faction_by_name("space_marines")
    return ScenarioUnit(
        unit_id=unit_id,
        datasheet=faction["intercessor_squad"],
        position=pos,
        models=models,
    )


def _gants(
    unit_id: str = "g1", pos: tuple[int, int] = (8, 2), models: int = 10
) -> ScenarioUnit:
    faction = load_faction_by_name("tyranids")
    return ScenarioUnit(
        unit_id=unit_id,
        datasheet=faction["termagants"],
        position=pos,
        models=models,
    )


def _boyz(
    unit_id: str = "b1", pos: tuple[int, int] = (3, 2), models: int = 10
) -> ScenarioUnit:
    faction = load_faction_by_name("orks")
    return ScenarioUnit(
        unit_id=unit_id,
        datasheet=faction["boyz"],
        position=pos,
        models=models,
    )


def _scenario(
    attackers: tuple[ScenarioUnit, ...],
    defenders: tuple[ScenarioUnit, ...],
    turns: tuple[ScenarioTurn, ...],
) -> Scenario:
    return Scenario(
        scenario_id="test_mov",
        title="Test Movement",
        teaches="movement",
        intro="intro",
        outro="outro",
        player_side="attacker",
        opponent_strategy="scripted",
        attacker=ScenarioSide(
            name="attacker", faction="space_marines", units=attackers
        ),
        defender=ScenarioSide(
            name="defender", faction="tyranids", units=defenders
        ),
        turns=turns,
    )


class TestMovementEngineRules:
    def test_remain_stationary_keeps_position(self) -> None:
        m1 = _marines(pos=(2, 2))
        g1 = _gants(pos=(8, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (g1,), (turn,))
        events: list[MoveEvent] = []
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="remain_stationary",
            destination=(2, 2),
        )
        strat = ScriptedStrategy([action])
        state = run_scenario(
            scen,
            {"attacker": strat, "defender": ScriptedStrategy([])},
            on_move=events.append,
        )
        assert len(events) == 1
        assert events[0].move_type == "remain_stationary"
        assert events[0].from_pos == (2, 2)
        assert events[0].to_pos == (2, 2)
        assert state.units["m1"].position == (2, 2)
        assert not state.units["m1"].advanced
        assert not state.units["m1"].fell_back

    def test_normal_move_updates_position_and_sets_moved(self) -> None:
        m1 = _marines(pos=(2, 2))
        g1 = _gants(pos=(8, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (g1,), (turn,))
        events: list[MoveEvent] = []
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="normal",
            destination=(4, 2),
        )
        strat = ScriptedStrategy([action])
        state = run_scenario(
            scen,
            {"attacker": strat, "defender": ScriptedStrategy([])},
            on_move=events.append,
        )
        assert len(events) == 1
        assert events[0].move_type == "normal"
        assert events[0].from_pos == (2, 2)
        assert events[0].to_pos == (4, 2)
        assert state.units["m1"].position == (4, 2)
        assert state.units["m1"].moved
        assert not state.units["m1"].advanced
        assert not state.units["m1"].fell_back

    def test_advance_move_sets_advanced_flag(self) -> None:
        m1 = _marines(pos=(2, 2))
        g1 = _gants(pos=(8, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (g1,), (turn,))
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="advance",
            destination=(5, 2),
        )
        strat = ScriptedStrategy([action])
        state = run_scenario(
            scen, {"attacker": strat, "defender": ScriptedStrategy([])}
        )
        assert state.units["m1"].position == (5, 2)
        assert state.units["m1"].advanced

    def test_fall_back_sets_fell_back_flag(self) -> None:
        m1 = _marines(pos=(2, 2))
        b1 = _boyz(pos=(3, 2))  # adjacent = engaged
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (b1,), (turn,))
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="fall_back",
            destination=(0, 2),
        )
        strat = ScriptedStrategy([action])
        state = run_scenario(
            scen, {"attacker": strat, "defender": ScriptedStrategy([])}
        )
        assert state.units["m1"].position == (0, 2)
        assert state.units["m1"].fell_back

    def test_normal_move_while_engaged_rejected(self) -> None:
        m1 = _marines(pos=(2, 2))
        b1 = _boyz(pos=(3, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (b1,), (turn,))
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="normal",
            destination=(0, 2),
        )
        strat = ScriptedStrategy([action])
        with pytest.raises(EngineError, match="must Fall Back"):
            run_scenario(
                scen, {"attacker": strat, "defender": ScriptedStrategy([])}
            )

    def test_advance_while_engaged_rejected(self) -> None:
        m1 = _marines(pos=(2, 2))
        b1 = _boyz(pos=(3, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (b1,), (turn,))
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="advance",
            destination=(0, 2),
        )
        strat = ScriptedStrategy([action])
        with pytest.raises(EngineError, match="must Fall Back"):
            run_scenario(
                scen, {"attacker": strat, "defender": ScriptedStrategy([])}
            )

    def test_fall_back_while_unengaged_rejected(self) -> None:
        m1 = _marines(pos=(2, 2))
        g1 = _gants(pos=(8, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (g1,), (turn,))
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="fall_back",
            destination=(0, 2),
        )
        strat = ScriptedStrategy([action])
        with pytest.raises(EngineError, match="not engaged"):
            run_scenario(
                scen, {"attacker": strat, "defender": ScriptedStrategy([])}
            )

    def test_move_ending_in_enemy_engagement_range_rejected(self) -> None:
        m1 = _marines(pos=(2, 2))
        g1 = _gants(pos=(6, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (g1,), (turn,))
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="normal",
            destination=(5, 2),
        )
        strat = ScriptedStrategy([action])
        with pytest.raises(EngineError, match="must end unengaged"):
            run_scenario(
                scen, {"attacker": strat, "defender": ScriptedStrategy([])}
            )

    def test_move_exceeding_reach_rejected(self) -> None:
        m1 = _marines(pos=(2, 2))
        g1 = _gants(pos=(10, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((m1,), (g1,), (turn,))
        # Intercessors M=6" -> reach_squares=3. (2, 2) to (6, 2) is 4 squares.
        action = Action(
            kind="move",
            attacker_unit_id="m1",
            move_type="normal",
            destination=(6, 2),
        )
        strat = ScriptedStrategy([action])
        with pytest.raises(EngineError, match="exceeds maximum reach"):
            run_scenario(
                scen, {"attacker": strat, "defender": ScriptedStrategy([])}
            )


class TestShootingEligibilityUnderRule1004:
    def test_advanced_unit_cannot_shoot(self) -> None:
        m1 = _marines("m1", pos=(2, 2))
        m2 = _marines("m2", pos=(0, 0))
        g1 = _gants("g1", pos=(6, 2))
        turns = (
            ScenarioTurn("movement", "attacker"),
            ScenarioTurn("shooting", "attacker"),
        )
        scen = _scenario((m1, m2), (g1,), turns)
        actions = [
            Action(
                kind="move",
                attacker_unit_id="m1",
                move_type="advance",
                destination=(4, 2),
            ),
            Action(
                kind="move",
                attacker_unit_id="m2",
                move_type="remain_stationary",
            ),
            Action(
                kind="shoot",
                attacker_unit_id="m1",
                weapon_key="bolt_rifle",
                target_unit_id="g1",
            ),
        ]
        strat = ScriptedStrategy(actions)
        with pytest.raises(EngineError, match="advanced this turn and cannot shoot"):
            run_scenario(
                scen, {"attacker": strat, "defender": ScriptedStrategy([])}
            )

    def test_fell_back_unit_cannot_shoot(self) -> None:
        m1 = _marines("m1", pos=(2, 2))
        m2 = _marines("m2", pos=(0, 0))
        b1 = _boyz("b1", pos=(3, 2))
        turns = (
            ScenarioTurn("movement", "attacker"),
            ScenarioTurn("shooting", "attacker"),
        )
        scen = _scenario((m1, m2), (b1,), turns)
        actions = [
            Action(
                kind="move",
                attacker_unit_id="m1",
                move_type="fall_back",
                destination=(0, 2),
            ),
            Action(
                kind="move",
                attacker_unit_id="m2",
                move_type="remain_stationary",
            ),
            Action(
                kind="shoot",
                attacker_unit_id="m1",
                weapon_key="bolt_rifle",
                target_unit_id="b1",
            ),
        ]
        strat = ScriptedStrategy(actions)
        with pytest.raises(EngineError, match="fell back this turn and cannot shoot"):
            run_scenario(
                scen, {"attacker": strat, "defender": ScriptedStrategy([])}
            )

    def test_normal_move_allows_shooting(self) -> None:
        m1 = _marines(pos=(2, 2))
        g1 = _gants(pos=(6, 2))
        turns = (
            ScenarioTurn("movement", "attacker"),
            ScenarioTurn("shooting", "attacker"),
        )
        scen = _scenario((m1,), (g1,), turns)
        actions = [
            Action(
                kind="move",
                attacker_unit_id="m1",
                move_type="normal",
                destination=(4, 2),
            ),
            Action(
                kind="shoot",
                attacker_unit_id="m1",
                weapon_key="bolt_rifle",
                target_unit_id="g1",
            ),
        ]
        strat = ScriptedStrategy(actions)
        state = run_scenario(
            scen,
            {"attacker": strat, "defender": ScriptedStrategy([])},
            rng=random.Random(42),
        )
        assert state.units["m1"].position == (4, 2)
        assert state.units["g1"].models < 10


class TestScriptedMoveValidation:
    def test_scripted_move_parsed_correctly(self) -> None:
        action = ScenarioAction(
            attacker_unit_id="m1",
            move_type="normal",
            destination=(4, 2),
        )
        turn = ScenarioTurn("movement", "attacker", actions=(action,))
        scen = _scenario((_marines(),), (_gants(),), (turn,))
        actions = scripted_actions_for(scen, "attacker")
        assert len(actions) == 1
        assert actions[0].kind == "move"
        assert actions[0].move_type == "normal"
        assert actions[0].destination == (4, 2)


class TestHeuristicMovement:
    def test_heuristic_moves_into_firing_range(self) -> None:
        gants = _gants(pos=(1, 3))
        marines = _marines(pos=(11, 3))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((gants,), (marines,), (turn,))
        ai = HeuristicStrategy()
        state = run_scenario(
            scen, {"attacker": ai, "defender": ScriptedStrategy([])}
        )
        new_pos = state.units["g1"].position
        assert new_pos[0] > 1
        assert (11 - new_pos[0]) <= 9

    def test_heuristic_falls_back_when_engaged(self) -> None:
        gants = _gants(pos=(2, 2))
        boyz = _boyz(pos=(3, 2))
        turn = ScenarioTurn("movement", "attacker")
        scen = _scenario((gants,), (boyz,), (turn,))
        ai = HeuristicStrategy()
        state = run_scenario(
            scen, {"attacker": ai, "defender": ScriptedStrategy([])}
        )
        new_pos = state.units["g1"].position
        assert new_pos != (2, 2)
        assert max(abs(new_pos[0] - 3), abs(new_pos[1] - 2)) > 1


class TestScenario10Execution:
    def test_scenario_10_loads_and_runs_cleanly(self) -> None:
        scen = load_scenario_by_id("10_position_and_fire")
        assert scen.scenario_id == "10_position_and_fire"
        assert scen.turns[0].phase == "movement"
        player = ScriptedStrategy([
            Action(
                kind="move",
                attacker_unit_id="termagants_1",
                move_type="normal",
                destination=(3, 3),
            ),
            Action(
                kind="shoot",
                attacker_unit_id="termagants_1",
                weapon_key="fleshborer",
                target_unit_id="intercessors_1",
            ),
        ])
        state = run_scenario(
            scen,
            {"attacker": player, "defender": HeuristicStrategy()},
            rng=random.Random(42),
        )
        assert state.units["termagants_1"].position == (3, 3)
