"""HeuristicStrategy: the rule-based AI opponent (the v2 headliner).

Each activation it enumerates every legal action for the phase — in a
shooting phase, eligible shooter x carried ranged weapon x surviving enemy;
in a fight phase, eligible fighter x melee weapon x *engaged* enemy — scores
each by expected damage, and takes the best one. The score is
`core.expected.expected_damage` capped at the target's remaining total
wounds, so an attack is never "spent" on more unit than is left standing:
ten expected damage into a one-wound survivor scores one, and the fuller
enemy unit wins instead. The estimator never reads the weapon's type — WS
and BS are the same `skill` field — so one brain drives both phases.

Design notes:

- **Same math as the dice.** The estimator mirrors the combat pipeline's
  targets and keyword semantics and is tested against `resolve_shooting`'s
  Monte Carlo mean, honoring the promise in `strategies/base.py` that the AI
  scores actions with the combat math the engine resolves them with.
- **Deterministic.** Candidates are scored in snapshot order (scenario
  order) and only a strictly better score displaces the incumbent, so ties
  keep the earliest candidate and the same battlefield always produces the
  same action — the property every seeded test in this project leans on.
- **Greedy per activation.** Each `choose_action` call optimizes one action
  in isolation; it does not plan which of its units should act first. In a
  shooting phase that gap is cosmetic (every weapon locks to one target).
  In a fight phase greedy-per-pick *is* an ordering policy — the AI resolves
  its deadliest available fight first — but it does not anticipate the
  opponent's interleaved picks; that is the first thing a smarter successor
  would revisit.
- **Loadout-aware for free.** Candidates come from
  `UnitSnapshot.ranged_weapons` / `UnitSnapshot.melee_weapons`, the same
  single definitions the engine validates against, so the AI can never pick
  a weapon a scenario's loadout override took away — and it always keeps its
  melee arm, because overrides never disarm a unit in melee.
"""

from __future__ import annotations

from collections.abc import Iterable

from wh40k_tutorial.core.expected import expected_damage
from wh40k_tutorial.core.models import Weapon
from wh40k_tutorial.core.scenario import in_engagement_range, in_weapon_range
from wh40k_tutorial.strategies.base import Action, GameState, UnitSnapshot


def _remaining_wounds(unit: UnitSnapshot) -> int:
    """Total wounds the unit can still lose: full trailing models + the lead."""
    if unit.destroyed:
        return 0
    return (unit.models - 1) * unit.datasheet.profile.wounds + unit.wounds_on_lead


class HeuristicStrategy:
    """Picks the legal action with the highest capped expected damage."""

    def choose_action(self, state: GameState) -> Action:
        if state.phase == "fight":
            candidates = (
                (fighter, weapon, target)
                for fighter in state.eligible_fighters(state.active_side)
                for weapon in fighter.melee_weapons
                for target in state.engaged_enemies(fighter)
            )
            return self._best(candidates, kind="fight")
        if state.phase == "movement":
            return self._choose_move(state)
        candidates = (
            (shooter, weapon, target)
            for shooter in state.eligible_shooters()
            for weapon in shooter.ranged_weapons
            for target in state.shootable_targets(shooter, weapon)
        )
        return self._best(candidates, kind="shoot")

    def _choose_move(self, state: GameState) -> Action:
        movers = state.eligible_movers()
        if not movers:
            raise RuntimeError(
                "no legal move available — the engine should not have asked for an action"
            )
        mover = movers[0]  # deterministic scenario order
        enemies = state.surviving_enemies()

        def best_shooting_damage(from_pos: tuple[int, int]) -> float:
            best = 0.0
            if any(in_engagement_range(from_pos, e.position) for e in enemies):
                return 0.0
            for w in mover.ranged_weapons:
                for target in enemies:
                    if not in_weapon_range(from_pos, target.position, w):
                        continue
                    if state.engaged_enemies(target):
                        continue
                    dmg = min(
                        expected_damage(mover.models, w, target.datasheet.profile),
                        float(_remaining_wounds(target)),
                    )
                    if dmg > best:
                        best = dmg
            return best

        def incoming_damage(at_pos: tuple[int, int]) -> float:
            total = 0.0
            for enemy in enemies:
                if in_engagement_range(at_pos, enemy.position):
                    melee_best = max(
                        (
                            expected_damage(enemy.models, mw, mover.datasheet.profile)
                            for mw in enemy.melee_weapons
                        ),
                        default=0.0,
                    )
                    total += melee_best
                else:
                    if not state.engaged_enemies(enemy):
                        ranged_best = max(
                            (
                                expected_damage(enemy.models, rw, mover.datasheet.profile)
                                for rw in enemy.ranged_weapons
                                if in_weapon_range(enemy.position, at_pos, rw)
                            ),
                            default=0.0,
                        )
                        total += ranged_best
            return total

        def score_move(pos: tuple[int, int], move_type: str) -> float:
            dealt = 0.0 if move_type in ("advance", "fall_back") else best_shooting_damage(pos)
            taken = incoming_damage(pos)
            return dealt - taken

        baseline_score = score_move(mover.position, "remain_stationary")
        best_score = baseline_score
        best_action = Action(
            kind="move",
            attacker_unit_id=mover.unit_id,
            move_type="remain_stationary",
            destination=mover.position,
        )

        legal_types = state.legal_move_types(mover)
        for mt in legal_types:
            if mt == "remain_stationary":
                continue
            dests = state.legal_destinations(mover, mt)
            for dest in dests:
                s = score_move(dest, mt)
                if s > best_score:
                    best_score = s
                    best_action = Action(
                        kind="move",
                        attacker_unit_id=mover.unit_id,
                        move_type=mt,
                        destination=dest,
                    )
        return best_action

    def _best(
        self,
        candidates: Iterable[tuple[UnitSnapshot, Weapon, UnitSnapshot]],
        kind: str,
    ) -> Action:
        """The highest capped-expected-damage candidate, earliest on ties."""
        best_score = -1.0
        best: Action | None = None
        for attacker, weapon, target in candidates:
            score = min(
                expected_damage(attacker.models, weapon, target.datasheet.profile),
                float(_remaining_wounds(target)),
            )
            if score > best_score:
                best_score = score
                best = Action(
                    kind=kind,
                    attacker_unit_id=attacker.unit_id,
                    weapon_key=weapon.name,
                    target_unit_id=target.unit_id,
                )
        if best is None:
            raise RuntimeError(
                f"no legal {kind} available — the engine should not have asked for an action"
            )
        return best
