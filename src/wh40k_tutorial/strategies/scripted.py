"""ScriptedStrategy: replays a fixed sequence of actions from the scenario file.

Used for the non-player side in the teaching-ladder tutorials. The scenario
author writes the opponent's shots — and, in fight turns, its fights — into
the ``actions`` array of each turn entry (see the add-scenario skill);
``scripted_actions_for`` flattens that side's script and the strategy replays
it one action per activation, in order.

Running out of script is an authoring bug, not a fallback situation — the
strategy raises rather than inventing a move, so a scenario that under-scripts
its opponent fails loudly in testing instead of drifting from its lesson.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from wh40k_tutorial.core.scenario import Scenario
from wh40k_tutorial.strategies.base import Action, GameState


class ScriptExhaustedError(RuntimeError):
    """The scripted side was asked to act but has no scripted actions left."""


class ScriptedStrategy:
    """Replays a fixed sequence of actions, one per ``choose_action`` call."""

    def __init__(self, actions: Sequence[Action]) -> None:
        self._queue: deque[Action] = deque(actions)

    def choose_action(self, state: GameState) -> Action:
        if not self._queue:
            raise ScriptExhaustedError(
                f"the scripted {state.active_side} was asked for an action on turn "
                f"{state.turn} but its script is exhausted — add 'actions' to that "
                f"side's turn entries in the scenario file"
            )
        return self._queue.popleft()


def scripted_actions_for(scenario: Scenario, side: str) -> tuple[Action, ...]:
    """Flatten one side's scripted actions across the scenario's turns, in play order.

    A shooting or movement turn's actions belong to its active side. A fight turn is
    two-sided, so a fight action belongs to whichever side its acting unit
    is on — the loader guarantees that is never the player's side.
    """
    side_units = {u.unit_id for u in scenario.side(side).units}
    actions: list[Action] = []
    for turn in scenario.turns:
        for a in turn.actions:
            if turn.phase == "fight":
                if a.attacker_unit_id in side_units:
                    actions.append(
                        Action(
                            kind="fight",
                            attacker_unit_id=a.attacker_unit_id,
                            weapon_key=a.weapon,
                            target_unit_id=a.target_unit_id,
                        )
                    )
            elif turn.phase == "movement":
                if turn.active_side == side:
                    actions.append(
                        Action(
                            kind="move",
                            attacker_unit_id=a.attacker_unit_id,
                            move_type=a.move_type,
                            destination=a.destination,
                        )
                    )
            else:
                if turn.active_side == side:
                    actions.append(
                        Action(
                            kind="shoot",
                            attacker_unit_id=a.attacker_unit_id,
                            weapon_key=a.weapon,
                            target_unit_id=a.target_unit_id,
                        )
                    )
    return tuple(actions)
