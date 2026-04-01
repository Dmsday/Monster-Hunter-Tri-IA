"""
MonsterZoneTracker — manages monster zone detection, combat state, and related rewards.

Extracted from MonsterHunterRewardCalculator.

Logic overview:
  - zone_has_monsters is activated ONLY when the player actually takes damage
    in a zone where monsters are present. Mere visual detection is not enough
    (avoids false positives when walking through empty zones).
  - Combat is active while damage was taken in the last combat_timeout seconds.
  - Left-monster-zone penalty: applied when zone_has_monsters transitions True→False
    outside the starting camp and without a zone change.
  - All state resets on zone change (prev_zone != current_zone).

Public API:
    tracker = MonsterZoneTracker()
    reward, in_combat = tracker.update(
        current_state, took_damage, zone_just_changed,
        current_time, reward, info,
        reward_breakdown, reward_breakdown_detailed,
        frames_stationary,        # from RewardCalculator (read-only)
        monsters_hit_count,       # from RewardCalculator (read-only)
    )
    tracker.reset()
    stats = tracker.get_stats()
"""

import time
from typing import Tuple

from info.module_logger import get_module_logger

logger = get_module_logger('monster_zone_tracker')


class MonsterZoneTracker:
    """
    Tracks monster zone presence, combat state, and zone-leave penalties.
    """

    def __init__(self):
        # Monster zone / combat state
        self.zone_has_monsters          = False   # Confirmed monster zone (damage-triggered)
        self.prev_zone_had_monsters     = False   # Previous step's value (for transition detection)
        self.prev_in_combat             = False   # Combat state from previous step
        self.frames_in_monster_zone     = 0       # Consecutive frames in a confirmed monster zone
        self.left_monster_zone_count    = 0       # How many times the agent left a monster zone

        # Combat timer
        self.last_damage_time           = 0.0     # Timestamp of last damage taken (0 = never)
        self.combat_timeout             = 10.0    # Seconds without damage → combat ends
        self._combat_damage_accumulated = 0       # Damage accumulated in current combat (for log summary)
        self._last_monster_count        = 0       # Monster count cached for GUI / stats

        # "Monster detected but not yet confirmed" flag (avoids log spam)
        self._monsters_detected_logged  = False

        # Constants (mirrors DEFAULT_REWARD_CONFIG; can be patched externally)
        self.BONUS_IN_MONSTER_ZONE          = 0.02
        self.PENALTY_LEFT_MONSTER_ZONE      = 1.5
        self.BONUS_MONSTER_ZONE_PERSISTENCE = 0.005

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all per-episode state. Call at the start of each episode."""
        self.zone_has_monsters          = False
        self.prev_zone_had_monsters     = False
        self.prev_in_combat             = False
        self.frames_in_monster_zone     = 0
        self.left_monster_zone_count    = 0
        self.last_damage_time           = 0.0
        self._combat_damage_accumulated = 0
        self._last_monster_count        = 0
        self._monsters_detected_logged  = False

    def update(
        self,
        current_state: dict,
        took_damage: bool,
        zone_just_changed: bool,
        current_time: float,
        reward: float,
        info: dict,
        reward_breakdown: dict,
        reward_breakdown_detailed: dict,
        frames_stationary: int,
        monsters_hit_count: int,
    ) -> Tuple[float, bool]:
        """
        Update monster zone / combat state for the current step.

        Args:
            current_state:             Full game state dict from MemoryReader.
            took_damage:               Whether the player took damage this step.
            zone_just_changed:         Whether the zone changed since last step.
            current_time:              time.time() value (passed in to avoid extra syscall).
            reward:                    Reward accumulated so far this step.
            info:                      Step info dict (modified in-place).
            reward_breakdown:          Top-level reward breakdown dict (modified in-place).
            reward_breakdown_detailed: Detailed breakdown dict (modified in-place).
            frames_stationary:         Consecutive stationary frames (from RewardCalculator).
            monsters_hit_count:        Cumulative monster hit count (from RewardCalculator).

        Returns:
            (adjusted_reward, in_combat)
              - adjusted_reward: reward after applying monster zone bonuses/penalties
              - in_combat:       whether the player is currently in combat
        """
        current_zone = current_state.get('current_zone', 0) or 0

        # ----------------------------------------------------------------
        # Detect monsters
        # ----------------------------------------------------------------
        monsters_present_now, monster_count_now = self._detect_monsters(current_state)
        if current_zone == 0:
            monsters_present_now = False
            monster_count_now    = 0
        self._last_monster_count = monster_count_now

        # ----------------------------------------------------------------
        # Combat state machine
        # ----------------------------------------------------------------
        if self.last_damage_time > 0:
            time_since_damage = current_time - self.last_damage_time
        else:
            time_since_damage = self.combat_timeout + 1.0   # Never took damage → not in combat

        if zone_just_changed:
            # ---- Zone changed : log combat summary, then reset ----
            if self.prev_in_combat and self.last_damage_time > 0 and self._combat_damage_accumulated > 0:
                time_in_combat = current_time - self.last_damage_time
                logger.info(
                    f"COMBAT ENDED (zone change → zone {current_zone}) | "
                    f"dmg taken: {self._combat_damage_accumulated} HP | "
                    f"monsters hit: {monsters_hit_count} | "
                    f"duration: {time_in_combat:.1f}s"
                )
                self._combat_damage_accumulated = 0
                info['combat_ended'] = True

            in_combat              = False
            self.last_damage_time  = 0.0

        elif took_damage:
            was_in_combat = self.prev_in_combat
            if current_zone != 0:
                # Start combat only outside the base camp
                in_combat = True
                self.last_damage_time = current_time
                if not was_in_combat:
                    info['combat_started'] = True
                    logger.info(f"COMBAT STARTED (zone {current_zone}, {monster_count_now} monsters)")

            else:
                # Damage in zone 0 (camp) is ignored for combat tracking
                in_combat = was_in_combat

        elif time_since_damage < self.combat_timeout:
            in_combat = True    # Still within timeout window

        else:
            was_in_combat = self.prev_in_combat
            in_combat     = False

            if was_in_combat and self.last_damage_time > 0:
                info['combat_ended'] = True
                logger.info(f"COMBAT ENDED (timeout {time_since_damage:.1f}s)")

                dmg = self._combat_damage_accumulated
                if dmg > 0:
                    logger.info(f"  Total damage taken: {dmg} HP")
                else:
                    logger.info(f"  No damage taken during this combat")
                logger.info(f"  Monsters hit: {monsters_hit_count} times")

                combat_duration = current_time - (self.last_damage_time - self.combat_timeout)
                if combat_duration > 0:
                    logger.info(f"  Combat duration: {combat_duration:.1f}s")

                self._combat_damage_accumulated = 0

            if in_combat != was_in_combat:
                self.last_damage_time = 0.0

        # Force no combat in camp
        if current_zone == 0:
            in_combat = False

        # Periodic debug log
        total_steps = info.get('total_steps', 0)
        if total_steps > 0 and total_steps % 300 == 0:
            logger.debug(
                f"Combat state step {total_steps}: "
                f"took_damage={took_damage} in_combat={in_combat} "
                f"prev_in_combat={self.prev_in_combat} "
                f"time_since_dmg={time_since_damage:.1f}s "
                f"zone_has_monsters={self.zone_has_monsters} "
                f"monsters_now={monsters_present_now}"
            )

        # ----------------------------------------------------------------
        # Activate / deactivate zone_has_monsters
        # ----------------------------------------------------------------
        if took_damage:
            was_in_monster_zone = self.zone_has_monsters
            if current_zone != 0:
                self.zone_has_monsters = True
                if not was_in_monster_zone and monster_count_now > 0:
                    logger.debug(
                        f"MONSTER ZONE CONFIRMED (took damage, {monster_count_now} monsters)"
                    )
            else:
                self.zone_has_monsters = False

        elif self.zone_has_monsters:
            # Already in monster zone: check 15s deactivation timeout
            if self.last_damage_time > 0 and time_since_damage > 15.0:
                logger.debug(
                    f"Monster zone DEACTIVATED (no damage for {time_since_damage:.1f}s)"
                )
                self.zone_has_monsters = False
                info['monster_zone_ignored_no_combat'] = True

        else:
            # Not yet in monster zone: detect but don't activate until damage
            if monsters_present_now:
                if not self._monsters_detected_logged:
                    self._monsters_detected_logged = True
                    logger.debug(
                        f"DETECTION: {monster_count_now} monsters present (waiting for damage)"
                    )
                info['monsters_detected_awaiting_damage'] = True
            else:
                self._monsters_detected_logged = False

        # ----------------------------------------------------------------
        # Monster zone reward / penalty
        # ----------------------------------------------------------------
        monster_reward = 0.0

        if self.zone_has_monsters and current_zone != 0:
            self.frames_in_monster_zone += 1

            # Base bonus (only if in combat or early in zone)
            if in_combat or self.frames_in_monster_zone < 100:
                monster_reward += self.BONUS_IN_MONSTER_ZONE
                reward_breakdown_detailed['monster_zone.in_zone_bonus'] = self.BONUS_IN_MONSTER_ZONE

            # Extra bonus if moving
            if frames_stationary < 30:
                monster_reward += self.BONUS_IN_MONSTER_ZONE
            else:
                info['monster_zone_bonus_blocked'] = True

            # Persistence bonus after ~3 seconds
            if self.frames_in_monster_zone > 100:
                persistence_bonus = self.BONUS_MONSTER_ZONE_PERSISTENCE
                monster_reward   += persistence_bonus
                reward_breakdown_detailed['monster_zone.persistence_bonus'] = persistence_bonus

            info['in_monster_zone']         = True
            info['monster_count']           = monster_count_now
            info['frames_in_monster_zone']  = self.frames_in_monster_zone

        # Left-monster-zone penalty: True→False transition, not caused by zone change
        if (
            self.prev_zone_had_monsters
            and not self.zone_has_monsters
            and current_zone != 0
            and not zone_just_changed
        ):
            monster_reward -= self.PENALTY_LEFT_MONSTER_ZONE
            self.left_monster_zone_count += 1
            reward_breakdown_detailed['monster_zone.left_zone_penalty'] = (
                -self.PENALTY_LEFT_MONSTER_ZONE
            )
            info['left_monster_zone']       = True
            info['left_monster_zone_count'] = self.left_monster_zone_count
            logger.debug(
                f"LEFT MONSTER ZONE! Penalty: -{self.PENALTY_LEFT_MONSTER_ZONE:.1f}"
            )
            self.frames_in_monster_zone = 0

        elif (
            not self.prev_zone_had_monsters
            and self.zone_has_monsters
            and not zone_just_changed
        ):
            self.frames_in_monster_zone = 0
            info['entered_monster_zone'] = True
            logger.info(
                f"ENTERED MONSTER ZONE! ({monster_count_now} monsters detected)"
            )

        # Save state for next step
        self.prev_zone_had_monsters = self.zone_has_monsters
        self.prev_in_combat         = in_combat

        # Accumulate combat damage (for summary logs)
        if in_combat and took_damage:
            # RewardCalculator already computed damage_taken; we get it from info
            self._combat_damage_accumulated += info.get('damage_taken', 0)

        # Push current state to info
        info['monsters_present']  = self.zone_has_monsters
        info['monster_count']     = monster_count_now
        info['in_monster_zone']   = self.zone_has_monsters
        info['in_combat']         = in_combat

        # Add to reward breakdown
        reward_breakdown['monster_zone'] = (
            reward_breakdown.get('monster_zone', 0.0) + monster_reward
        )
        reward += monster_reward

        return reward, in_combat

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_monsters(current_state: dict) -> Tuple[bool, int]:
        """
        Count monsters with HP > 0 in the current state.

        Returns:
            (monsters_present, monster_count)
        """
        count = 0
        for i in range(1, 5):      # Small monsters 1-4 (smonster5 uses same HP slot approach)
            hp = current_state.get(f'smonster{i}_hp')
            if hp is not None and hp > 0:
                count += 1
        for i in range(1, 2):      # Large monster 1
            hp = current_state.get(f'lmonster{i}_hp')
            if hp is not None and hp > 0:
                count += 1
        return count > 0, count

    # ------------------------------------------------------------------
    # Stats (used by RewardCalculator.get_stats)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a dict of stats for the current episode."""
        return {
            'left_monster_zone_count': self.left_monster_zone_count,
            'frames_in_monster_zone':  self.frames_in_monster_zone,
        }