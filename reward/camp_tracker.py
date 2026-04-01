"""
CampTracker — manages the starting camp (zone 0) stay penalties and exit bonuses.

Extracted from MonsterHunterRewardCalculator.

Camp logic:
  - If the agent stays in zone 0 longer than PENALTY_CAMP_THRESHOLD seconds,
    apply a continuous per-step penalty that grows progressively over time.
  - When the agent first exits zone 0 → BONUS_FIRST_CAMP_EXIT.
  - When the agent exits zone 0 after dying → BONUS_CAMP_EXIT_AFTER_DEATH.

Public API:
    tracker = CampTracker()
    reward  = tracker.calculate(current_zone, zone_just_changed, reward, info,
                                reward_breakdown, reward_breakdown_detailed)
    tracker.reset()

Note:
    camp_total_time is cumulative across the episode (not reset when leaving camp).
    It is exposed for stats/GUI via get_camp_stats().
"""

import time

from info.module_logger import get_module_logger

logger = get_module_logger('camp_tracker')


class CampTracker:
    """
    Tracks time spent in the starting camp and applies penalties / exit bonuses.
    """

    def __init__(self):
        # Per-episode mutable state
        self.camp_entry_time        = None   # Timestamp of most recent camp entry
        self.camp_total_time        = 0.0    # Total seconds spent in camp this episode
        self.camp_penalty_triggered = False  # Whether the threshold penalty was already triggered
        self.last_penalty_time      = None   # Timestamp of last per-step penalty application

        self.first_camp_exit        = False  # Has the agent left camp at least once?
        self.just_died              = False  # Did the agent die since last camp exit?
        self.camp_exit_after_death  = False  # Flag set when exiting camp after a death

        # Constants (mirrors DEFAULT_REWARD_CONFIG; can be patched externally)
        self.PENALTY_CAMP_THRESHOLD      = 30.0  # Seconds before penalty kicks in
        self.PENALTY_CAMP_PER_SECOND     = 0.02  # Continuous penalty per second past threshold
        self.PENALTY_CAMP_GROWTH_RATE    = 0.005 # Extra penalty per second² (progressive)
        self.PENALTY_CAMP_MAX_PER_SEC    = 0.15  # Cap per-second penalty
        self.BONUS_FIRST_CAMP_EXIT       = 3.0   # Bonus for first camp exit
        self.BONUS_CAMP_EXIT_AFTER_DEATH = 2.0   # Bonus for exiting camp after death

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all per-episode state. Call at the start of each episode."""
        self.camp_entry_time          = None
        self.camp_total_time          = 0.0
        self.camp_penalty_triggered   = False
        self.last_penalty_time        = None
        self.first_camp_exit          = False
        self.just_died                = False
        self.camp_exit_after_death    = False

    def calculate(
        self,
        current_zone: int,
        zone_just_changed: bool,
        reward: float,
        info: dict,
        reward_breakdown: dict,
        reward_breakdown_detailed: dict,
    ) -> float:
        """
        Apply camp-related penalties / bonuses and return the adjusted reward.

        Args:
            current_zone:             Current zone ID (0 = starting camp).
            zone_just_changed:        True if the zone changed this step.
            reward:                   Reward accumulated so far this step.
            info:                     Step info dict (modified in-place).
            reward_breakdown:         Top-level reward breakdown dict (modified in-place).
            reward_breakdown_detailed: Detailed breakdown dict (modified in-place).

        Returns:
            Adjusted reward (float).
        """
        current_time = time.time()

        # ----------------------------------------------------------------
        # IN CAMP (zone 0)
        # ----------------------------------------------------------------
        if current_zone == 0:
            if self.camp_entry_time is None:
                # Just entered the camp
                self.camp_entry_time   = current_time
                self.last_penalty_time = None
            else:
                time_in_camp         = current_time - self.camp_entry_time
                self.camp_total_time = time_in_camp

                if time_in_camp >= self.PENALTY_CAMP_THRESHOLD:
                    # Continuous per-step penalty — compute dt since last penalty
                    if self.last_penalty_time is None:
                        dt = 0.5  # first tick after threshold: assume ~0.5s
                    else:
                        dt = current_time - self.last_penalty_time

                    # Progressive: base + growth proportional to time over threshold
                    time_over = time_in_camp - self.PENALTY_CAMP_THRESHOLD
                    penalty_rate = self.PENALTY_CAMP_PER_SECOND + (
                        time_over * self.PENALTY_CAMP_GROWTH_RATE
                    )
                    penalty_rate = min(penalty_rate, self.PENALTY_CAMP_MAX_PER_SEC)

                    camp_penalty = penalty_rate * dt

                    reward -= camp_penalty
                    reward_breakdown['camp_penalty'] = (
                        reward_breakdown.get('camp_penalty', 0.0) - camp_penalty
                    )

                    reward_breakdown_detailed['penalties.camp_continuous'] = -camp_penalty
                    reward_breakdown_detailed['penalties.camp_rate_per_sec'] = -penalty_rate

                    info['camp_penalty']      = camp_penalty
                    info['camp_penalty_rate'] = penalty_rate
                    info['time_in_camp']      = time_in_camp
                    self.last_penalty_time    = current_time

        # ----------------------------------------------------------------
        # LEFT CAMP (zone != 0)
        # ----------------------------------------------------------------
        else:
            if self.camp_entry_time is not None:
                # Just left camp — apply exit bonuses if applicable
                zone_bonus = 0.0

                if not self.first_camp_exit:
                    # Very first time leaving camp this episode
                    zone_bonus            += self.BONUS_FIRST_CAMP_EXIT
                    self.first_camp_exit   = True
                    info['first_camp_exit'] = True
                    logger.info(f"FIRST CAMP EXIT! Bonus: +{zone_bonus:.1f}")

                elif self.just_died:
                    # Leaving camp after a death
                    zone_bonus                  += self.BONUS_CAMP_EXIT_AFTER_DEATH
                    self.just_died               = False
                    self.camp_exit_after_death   = True
                    info['camp_exit_after_death'] = True
                    logger.info(f"Camp exit after death! Bonus: +{zone_bonus:.1f}")

                if zone_bonus > 0:
                    reward += zone_bonus
                    reward_breakdown['zone_change'] = (
                        reward_breakdown.get('zone_change', 0.0) + zone_bonus
                    )

                    if self.camp_exit_after_death:
                        reward_breakdown_detailed['zone_change.exit_after_death'] = (
                            self.BONUS_CAMP_EXIT_AFTER_DEATH
                        )
                    else:
                        reward_breakdown_detailed['zone_change.first_exit'] = self.BONUS_FIRST_CAMP_EXIT

                # Reset camp timing state
                self.camp_entry_time          = None
                self.camp_penalty_triggered   = False
                self.last_penalty_time        = None

        return reward

    # ------------------------------------------------------------------
    # Stats (used by RewardCalculator.get_stats)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a dict of stats for the current episode."""
        return {
            'camp_total_time':  self.camp_total_time,
            'first_camp_exit':  self.first_camp_exit,
        }