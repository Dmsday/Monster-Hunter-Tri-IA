"""
OxygenTracker — manages oxygen / drowning penalties and recovery bonuses.

Extracted from MonsterHunterRewardCalculator to keep reward_calculator.py
focused on orchestration rather than per-system bookkeeping.

Public API:
    tracker = OxygenTracker()
    reward  = tracker.calculate(current_oxygen, reward, info, reward_breakdown, reward_breakdown_detailed)
    tracker.reset()
"""

import time

from info.module_logger import get_module_logger

logger = get_module_logger('oxygen_tracker')

# -------------------------------------------------------------------------
# Constants (can be overridden by passing reward_config to RewardCalculator)
# -------------------------------------------------------------------------
PENALTY_OXYGEN_INITIAL   = 0.7    # One-time penalty on first low-oxygen tick  (unused currently, progressive used instead)
PENALTY_OXYGEN_RECURRING = 0.2    # Recurring penalty while oxygen is low       (unused currently, progressive used instead)
BONUS_OXYGEN_RECOVERY    = 0.6    # Bonus granted when oxygen recovers to safe level
OXYGEN_LOW_THRESHOLD     = 25     # Below this → apply progressive penalty
OXYGEN_SAFE_THRESHOLD    = 50     # Above this → consider recovered
OXYGEN_PENALTY_DELAY     = 10.0   # Seconds before first penalty kicks in (not used in progressive mode)
OXYGEN_RECURRING_DELAY   = 2.0    # Seconds between recurring penalties       (not used in progressive mode)


class OxygenTracker:
    """
    Tracks oxygen level and applies progressive penalties / recovery bonuses.

    Penalty system:
      - Oxygen < OXYGEN_LOW_THRESHOLD → progressive penalty per second
        Formula: base * (1 + 2.3 * (1 - oxygen / threshold))
        i.e. the lower the oxygen, the bigger the penalty per second.
      - Oxygen >= OXYGEN_SAFE_THRESHOLD after being low → one-time recovery bonus.

    Note on water marker:
      The tracker no longer accesses exploration_tracker directly.
      The RewardCalculator handles that link in calculate().
    """

    def __init__(self):
        # Per-episode mutable state
        self.prev_oxygen           = None
        self.oxygen_low_start_time = None   # Timestamp when oxygen first went below threshold
        self.last_oxygen_penalty_time = None  # Timestamp of last applied penalty
        self.oxygen_penalty_count  = 0      # How many penalty ticks have been applied

        # Constants (mirrors DEFAULT_REWARD_CONFIG values; can be patched externally)
        self.PENALTY_OXYGEN_INITIAL   = PENALTY_OXYGEN_INITIAL
        self.PENALTY_OXYGEN_RECURRING = PENALTY_OXYGEN_RECURRING
        self.BONUS_OXYGEN_RECOVERY    = BONUS_OXYGEN_RECOVERY
        self.OXYGEN_LOW_THRESHOLD     = OXYGEN_LOW_THRESHOLD
        self.OXYGEN_SAFE_THRESHOLD    = OXYGEN_SAFE_THRESHOLD
        self.OXYGEN_PENALTY_DELAY     = OXYGEN_PENALTY_DELAY
        self.OXYGEN_RECURRING_DELAY   = OXYGEN_RECURRING_DELAY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all per-episode state. Call at the start of each episode."""
        self.prev_oxygen              = None
        self.oxygen_low_start_time    = None
        self.last_oxygen_penalty_time = None
        self.oxygen_penalty_count     = 0

    def calculate(
        self,
        current_oxygen,
        reward: float,
        info: dict,
        reward_breakdown: dict,
        reward_breakdown_detailed: dict,
    ) -> float:
        """
        Apply oxygen-related penalties / bonuses and return the adjusted reward.

        Args:
            current_oxygen:           Raw oxygen value from game state (int 0-100, or None).
            reward:                   Reward accumulated so far this step.
            info:                     Step info dict (modified in-place for GUI/logging).
            reward_breakdown:         Top-level reward breakdown dict (modified in-place).
            reward_breakdown_detailed: Detailed breakdown dict (modified in-place).

        Returns:
            Adjusted reward (float).
        """
        # Guard: nothing to do if oxygen not readable
        if current_oxygen is None:
            info['oxygen_status'] = 'none_detected'
            return reward

        # Validate value
        try:
            current_oxygen = int(current_oxygen)
        except (ValueError, TypeError):
            info['oxygen_status'] = 'invalid_value'
            return reward

        if current_oxygen < 0 or current_oxygen > 200:
            info['oxygen_status'] = f'out_of_range_{current_oxygen}'
            return reward

        current_time     = time.time()
        oxygen_penalty   = 0.0
        oxygen_bonus     = 0.0

        info['oxygen_level']  = current_oxygen
        info['oxygen_status'] = 'monitoring'

        # ----------------------------------------------------------------
        # LOW OXYGEN : progressive penalty per second
        # ----------------------------------------------------------------
        if current_oxygen < self.OXYGEN_LOW_THRESHOLD:

            # First time dropping below threshold this submersion
            if self.oxygen_low_start_time is None:
                self.oxygen_low_start_time    = current_time
                self.last_oxygen_penalty_time = None
                info['oxygen_low_started']    = True
                logger.info(f"LOW OXYGEN detected! Level: {current_oxygen}/100")

            # Progressive rate: higher as oxygen decreases
            # At 25 → x1.0, at 15 → x1.5, at 5 → x2.5, at 1 → x3.3
            base_penalty_per_second = 0.6
            oxygen_multiplier       = 1.0 + (2.3 * (1.0 - current_oxygen / self.OXYGEN_LOW_THRESHOLD))
            penalty_per_second      = base_penalty_per_second * oxygen_multiplier

            # Apply once per second
            if self.last_oxygen_penalty_time is None:
                self.last_oxygen_penalty_time = current_time

            time_since_last = current_time - self.last_oxygen_penalty_time
            if time_since_last >= 1.0:
                oxygen_penalty = penalty_per_second
                self.oxygen_penalty_count        += 1
                self.last_oxygen_penalty_time     = current_time

                reward_breakdown_detailed['oxygen.oxygen_progressive'] = -oxygen_penalty

                info['oxygen_progressive_penalty'] = True
                info['oxygen_penalty_count']        = self.oxygen_penalty_count
                info['oxygen_penalty_rate']         = penalty_per_second
                info['oxygen_status']               = f'low_oxygen_{self.oxygen_penalty_count}'

                logger.info(
                    f"CRITICAL OXYGEN ({current_oxygen}) "
                    f"— Penalty: -{penalty_per_second:.1f}/s (tick #{self.oxygen_penalty_count})"
                )

        # ----------------------------------------------------------------
        # RECOVERY : oxygen back above safe threshold
        # ----------------------------------------------------------------
        else:
            if self.oxygen_low_start_time is not None and current_oxygen >= self.OXYGEN_SAFE_THRESHOLD:
                # Give recovery bonus if penalties were ever applied
                if self.oxygen_penalty_count > 0:
                    oxygen_bonus += self.BONUS_OXYGEN_RECOVERY
                    reward_breakdown_detailed['oxygen.oxygen_recovery'] = self.BONUS_OXYGEN_RECOVERY
                    info['oxygen_recovery_bonus'] = True
                    info['oxygen_status']         = 'recovered'
                    logger.info(
                        f"OXYGEN RECOVERED! Level: {current_oxygen} "
                        f"— Bonus: +{self.BONUS_OXYGEN_RECOVERY:.1f}"
                    )

                # Reset submersion state
                self.oxygen_low_start_time    = None
                self.last_oxygen_penalty_time = None
                self.oxygen_penalty_count     = 0

            elif self.oxygen_low_start_time is not None:
                # Between low and safe threshold : recovering but not there yet
                info['oxygen_status'] = 'recovering'

        # ----------------------------------------------------------------
        # Apply to reward and breakdown
        # ----------------------------------------------------------------
        reward -= oxygen_penalty
        reward += oxygen_bonus

        if 'oxygen' not in reward_breakdown:
            reward_breakdown['oxygen'] = 0.0
        reward_breakdown['oxygen'] += oxygen_bonus - oxygen_penalty

        info['oxygen_penalty']       = oxygen_penalty
        info['oxygen_bonus']         = oxygen_bonus
        info['oxygen_penalty_count'] = self.oxygen_penalty_count

        self.prev_oxygen = current_oxygen
        return reward