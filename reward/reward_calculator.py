"""
Reward Calculator for Monster Hunter Tri RL agent.

Orchestrates the calculation of the reward signal at each environment step.
Delegates per-system bookkeeping to three tracker classes:
  - OxygenTracker       (environment/oxygen_tracker.py)
  - CampTracker         (environment/camp_tracker.py)
  - MonsterZoneTracker  (environment/monster_zone_tracker.py)

What stays here:
  - Global reward constants (DEFAULT_REWARD_CONFIG)
  - HP / stamina / death / sharpness / stationary / menu penalties
  - Monster damage detection (_check_monster_damage)
  - Exploration curiosity rewards (_check_curiosity_rewards)
  - Zone-change bonuses (orchestrated alongside CampTracker)
  - reset() / full_reset() / get_stats() / get_reward_breakdown_summary()
"""

import time
import numpy as np

from info.module_logger import get_module_logger

logger = get_module_logger('reward_calculator')

from reward.exploration_tracker import ExplorationTracker
from reward.oxygen_tracker import OxygenTracker
from reward.camp_tracker import CampTracker
from reward.monster_zone_tracker import MonsterZoneTracker

from core.controller.action_heads import (
    HEAD_MOVEMENT, HEAD_COMBAT, NUM_HEADS
)

# =============================================================================
# DEFAULT REWARD CONSTANTS
# Pass a partial dict to __init__ to override specific values without subclassing.
# =============================================================================
DEFAULT_REWARD_CONFIG = {
    # Player HP penalties
    "PENALTY_DAMAGE_BASE":          0.1,
    "PENALTY_BIG_HIT":              0.4,
    "PENALTY_LOW_HP":               0.04,
    "PENALTY_CRITICAL_HP":          0.1,
    "BONUS_GOOD_HEALTH":            0.02,

    # Stamina
    "PENALTY_LOW_STAMINA":          0.02,

    # Misc
    "PENALTY_IDLE":                 0.005,
    "REWARD_HIT_BASE":              1.0,
    "REWARD_HIT_MULTIPLIER":        0.02,
    "REWARD_ATTACK_ATTEMPT":        0.1,

    # Death / quest failure
    "PENALTY_DEATH_BASE":           30.0,
    "PENALTY_QUEST_FAILED":         60.0,
    "REWARD_VICTORY":               200.0,

    # Exploration curiosity
    "BONUS_NEW_ZONE_DISCOVERED":    2.0,
    "BONUS_NEW_AREA_DISCOVERED":    0.6,
    "PENALTY_REVISIT_AREA":         0.04,
    "REVISIT_THRESHOLD":            3,
    "PENALTY_STATIONARY":           0.004,

    # Small monster damage (balanced with health_loss: ~3 total for a full kill)
    "REWARD_SMONSTER_HIT": 0.5,
    "REWARD_SMONSTER_DAMAGE_MULT": 0.01,
    "BONUS_KILL_SMALL_MONSTER": 2.0,

    # Large monster damage (100% HP dealt = PENALTY_DEATH_BASE, kill = 3x)
    "REWARD_LMONSTER_HIT": 1.5,
    "REWARD_LMONSTER_DAMAGE_SCALE": 30.0,  # total reward for 100% HP = 1 death penalty
    "BONUS_KILL_LARGE_MONSTER": 90.0,  # 3 × PENALTY_DEATH_BASE

    # Zone change
    "BONUS_ZONE_CHANGE":            1.0,

    # Menu penalties
    "PENALTY_MENU_THRESHOLD":       5.0,
    "PENALTY_MENU_BASE":            0.6,
    "PENALTY_MENU_RECURRING":       0.2,
    "MENU_RECURRING_DELAY":         3.0,

    # Defensive actions
    "BONUS_BLOCK_ATTEMPT":          0.04,
    "BONUS_DODGE_ATTEMPT":          0.06,
    "BONUS_ATTACK_ATTEMPT":         0.05,

    # Weapon sharpness
    "PENALTY_LOW_SHARPNESS":        0.1,
    "PENALTY_BOUNCED":              1.0,
}


class MonsterHunterRewardCalculator:
    """
    Computes the reward signal for the Monster Hunter Tri RL agent.

    At each step, call:
        reward = calculator.calculate(prev_state, current_state, action, info, took_damage)

    At the end of each episode:
        calculator.reset()

    For a completely fresh start (new training run):
        calculator.full_reset()
    """

    def __init__(self, reward_config: dict = None):
        """
        Args:
            reward_config: Optional dict of constants to override.
                           Example: {'PENALTY_DEATH_BASE': 50.0}
                           Only supply the keys you want to change.
        """
        # ----------------------------------------------------------------
        # Apply reward constants from DEFAULT_REWARD_CONFIG
        # (with optional per-run overrides)
        # ----------------------------------------------------------------
        config = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}
        for attr_name, value in config.items():
            setattr(self, attr_name, value)

        # ----------------------------------------------------------------
        # Sub-trackers
        # ----------------------------------------------------------------
        self.oxygen_tracker       = OxygenTracker()
        self.camp_tracker         = CampTracker()
        self.monster_zone_tracker = MonsterZoneTracker()

        # Exploration spatial tracker
        self.cube_size = 650    # Edge length of one spatial cube
        self.max_cubes = 250    # Max cubes before compression
        self.exploration_tracker = ExplorationTracker(
            cube_size=self.cube_size,
            max_cubes=self.max_cubes,
            compression_target=0.85
        )

        # ----------------------------------------------------------------
        # Own mutable state (not delegated to a sub-tracker)
        # ----------------------------------------------------------------

        # HP / stamina tracking
        self.prev_hp                    = None
        self.prev_stamina               = None
        self.hp_recovery_given          = False   # Flag: HP recovery bonus already given
        self.stamina_recovery_given     = False
        self.hp_recovery_accumulated    = 0.0
        self.stamina_recovery_accumulated = 0.0

        # Damage flag tracking
        self.prev_damage_flag           = None    # Game's damage-receive flag (changes on hit)

        # Death tracking
        self.prev_death_count           = 0

        # Position / orientation / zone
        self.prev_position              = None
        self.prev_orientation           = None
        self.prev_zone                  = None

        # Sharpness
        self.prev_sharpness             = None

        # Oxygen value (tracked locally in addition to OxygenTracker)
        self.prev_oxygen                = None

        # Monster damage tracking
        self.prev_small_monsters_hp     = {}
        self.prev_large_monsters_hp     = {}
        self.lmonster_hp_max            = {}  # Tracked max HP per large monster
        self.lmonster_killed            = {}  # Flag: True once killed (ignore post-death garbage)
        self.monsters_hit_count         = 0
        self.total_monster_damage       = 0
        self.monsters_killed_count      = 0
        self.monster_damage_since_zone_change = 0

        # Stationary / idle tracking
        self.frames_stationary          = 0
        self.idle_start_time            = None

        # Distance tracking
        self.total_distance_traveled    = 0.0

        # Combo
        self.total_damage_dealt         = 0
        self.hit_count                  = 0
        self.consecutive_hits           = 0

        # In-game menu
        self.game_menu_entry_time       = None
        self.game_menu_total_time       = 0.0
        self.last_menu_penalty_time     = None
        self.game_menu_open_count       = 0
        self.prev_in_menu               = False
        # Debounce counters: require N consecutive frames to confirm a state change
        self._menu_open_streak = 0
        self._menu_closed_streak = 0

        # Zone change timing
        self.last_zone_change_time      = None
        self.zone_change_cooldown       = 7.0

        # Marker update timer
        self.last_marker_update         = time.time()
        self.marker_update_interval     = 1.0

        # Debug / internal counters
        self._combat_log_count          = 0
        self._debug_marker_check_count  = 0

        # Reward breakdown (rebuilt every step)
        self.reward_breakdown           = {}
        self.reward_breakdown_detailed  = {}
        self.last_reward_details        = {}

    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    def calculate(
            self,
            prev_state: dict,
            current_state: dict,
            action,  # <-- np.array(7,) or legacy int
            info: dict = None,
            took_damage: bool = False,
    ) -> float:
        """
        Compute the reward for one environment step.

        Args:
            prev_state:    Game state dict from the previous step (may be None on first step).
            current_state: Game state dict from the current step.
            action:        Action index executed (0-18).
            info:          Step info dict (modified in-place, forwarded to SB3 and GUI).
            took_damage:   True if the player took damage this step.

        Returns:
            reward (float)
        """
        reward = 0.0
        info   = info or {}

        # ----------------------------------------------------------------
        # Guard: ignore first step (no valid prev_state)
        # ----------------------------------------------------------------
        if info.get('episode_steps', 0) == 1:
            prev_state = None
            logger.debug("First step — prev_state ignored")

        # ----------------------------------------------------------------
        # Guard: only calculate while in quest (map 100 with running timer)
        # ----------------------------------------------------------------
        current_map  = current_state.get('current_map')
        death_count  = current_state.get('death_count', 0) or 0
        quest_time   = current_state.get('quest_time', 5400)

        is_valid_state = (current_map == 100 and quest_time is not None)
        if not is_valid_state:
            if current_map == 45:
                reason = "back at village (map=45, quest ended)"
            elif current_map == 100 and quest_time is None:
                reason = "quest end screen (map=100 but quest_time=None)"
            else:
                reason = f"unexpected map value (map={current_map})"

            logger.warning(f"Invalid state in reward_calculator: {reason}")
            logger.warning(f"  map={current_map}, deaths={death_count}, time={quest_time}")
            logger.warning("  → Returning zero reward, no calculation")
            info['invalid_state_detected'] = True
            info['invalid_state_reason']   = reason
            info['current_map']            = current_map
            return 0.0

        # Guard: quest_ended flags
        if current_state.get('quest_ended') or current_state.get('on_reward_screen'):
            logger.info("quest_ended flag detected in reward_calculator")
            info['quest_ended_flag_in_calc'] = True
            self.reset()
            return 0.0

        if prev_state is None:
            prev_state = {}

        current_time = time.time()

        # ----------------------------------------------------------------
        # Update dynamic cube markers (every second)
        # ----------------------------------------------------------------
        if current_time - self.last_marker_update >= self.marker_update_interval:
            self.exploration_tracker.marker_system.update_dynamic_markers()
            self.last_marker_update = current_time

        # Extract per-head branches from action vector
        if hasattr(action, '__len__') and len(action) >= NUM_HEADS:
            combat_branch = int(action[HEAD_COMBAT])  # 0=nothing,1=atk1,2=atk2,3=dodge,4=draw,5=ztarget
            movement_branch = int(action[HEAD_MOVEMENT])  # 0=nothing,1=fwd,2=back,3=strL,4=strR
            info['last_action'] = [int(a) for a in action]
        else:
            # Legacy int fallback
            combat_branch = 0
            movement_branch = 0
            info['last_action'] = action

        # ----------------------------------------------------------------
        # Reset per-step breakdown
        # ----------------------------------------------------------------
        self.reward_breakdown = {
            'combat': 0.0, 'health': 0.0,
            'exploration': 0.0, 'penalties': 0.0, 'zone_change': 0.0,
            'defensive_actions': 0.0, 'oxygen': 0.0, 'monster_zone': 0.0,
            'death': 0.0, 'damage_taken': 0.0, 'monster_hit': 0.0,
            'hit': 0.0, 'camp_penalty': 0.0, 'menu_penalty': 0.0,
            'sharpness_penalty': 0.0, 'other': 0.0,
        }
        self.reward_breakdown_detailed = {}

        # ================================================================
        # 1. SURVIVAL — removed: constant per-step bonus adds baseline
        #              noise without meaningful learning signal
        # ================================================================

        # ================================================================
        # 2. DAMAGE TAKEN (exponential HP-based penalty)
        # ================================================================
        current_hp = current_state.get('player_hp', 0) or 0

        # Save BEFORE any update — old_hp is reused in section 2B for recovery detection.
        # If prev_hp were updated first, hp_gain in 2B would always equal zero (bug fix).
        old_hp = self.prev_hp

        if prev_state and old_hp is not None:
            damage_taken = old_hp - current_hp

            # Discard aberrant deltas: >100 HP in a single step means a save-state reload
            # or a zone transition reset, not actual damage — ignore to avoid reward spikes.
            if abs(damage_taken) >= 100:
                logger.debug(
                    f"Aberrant HP delta ignored: {damage_taken:.1f} "
                    f"(prev={old_hp:.1f}, current={current_hp:.1f})"
                )
                damage_taken = 0

            if damage_taken > 0:
                damage_taken = min(damage_taken, 99)  # cap to avoid single-frame reward cliff
                hp_multiplier = self._calculate_hp_penalty_multiplier(current_hp)
                damage_penalty = damage_taken * self.PENALTY_DAMAGE_BASE * hp_multiplier
                reward -= damage_penalty
                self.reward_breakdown['damage_taken'] -= damage_penalty
                self.reward_breakdown_detailed['health.damage_penalty'] = -damage_penalty

                # Mark surrounding cubes as monster territory on any hit
                current_zone = current_state.get('current_zone', 0) or 0
                current_cube = getattr(self.exploration_tracker, 'current_cube', None)
                if current_cube:
                    zone_cubes = self.exploration_tracker.cubes_by_zone.get(current_zone, [])
                    self.exploration_tracker.marker_system.mark_monster_area(
                        current_cube, zone_cubes, max_distance=3.0
                    )

                # Extra penalty for large single hits (>20 HP lost in one frame)
                if damage_taken > 20:
                    big_hit_penalty = self.PENALTY_BIG_HIT * hp_multiplier
                    reward -= big_hit_penalty
                    self.reward_breakdown['damage_taken'] -= big_hit_penalty
                    self.reward_breakdown_detailed['health.big_hit_penalty'] = -big_hit_penalty

                info['damage_taken'] = damage_taken
                info['hp_multiplier'] = hp_multiplier

        # ================================================================
        # 2B. HP RECOVERY BONUS (one-time bonus per healing event)
        #
        # CRITICAL ORDER: this block runs BEFORE self.prev_hp is updated.
        # Using old_hp (previous frame) vs current_hp gives the true HP delta.
        # If self.prev_hp were updated above, current_hp - self.prev_hp == 0 always.
        # ================================================================
        if prev_state and old_hp is not None:
            hp_gain = current_hp - old_hp  # positive when the player healed this step

            # Discard aberrant gains: >=100 HP in a single step means a death-respawn
            # or save-state reload, not actual healing — skip to avoid reward spike.
            # Without this guard, respawning from death (0 → 150 HP) would give
            # a +120 bonus that completely offsets the death penalty.
            if hp_gain >= 100:
                logger.debug(
                    f"Aberrant HP gain ignored (respawn/reload): +{hp_gain:.1f} "
                    f"(prev={old_hp:.1f}, current={current_hp:.1f})"
                )
                hp_gain = 0

            if hp_gain > 0:
                if not self.hp_recovery_given:
                    # Cap bonus to avoid a single large heal dominating all other reward signals
                    hp_recovery_bonus = min(hp_gain * 0.8, 1.5)
                    reward += hp_recovery_bonus
                    self.reward_breakdown['health'] += hp_recovery_bonus
                    self.reward_breakdown_detailed['health.hp_recovery'] = hp_recovery_bonus
                    info['hp_recovered'] = hp_gain
                    self.hp_recovery_given = True
                    logger.debug(f"HP recovery bonus: +{hp_recovery_bonus:.2f} ({hp_gain:.1f} HP healed)")
            else:
                # No healing this step — reset flag so the bonus can trigger on next heal
                self.hp_recovery_given = False

        # Update after BOTH damage and recovery checks (order is intentional — see above)
        self.prev_hp = current_hp

        # ================================================================
        # 3. DAMAGE RECEIVED FLAG (game flag changes on each hit)
        # ================================================================
        damage_flag = current_state.get('damage_last_hit')
        if damage_flag is not None and self.prev_damage_flag is not None:
            if damage_flag != self.prev_damage_flag:
                flag_penalty = 1.0
                reward -= flag_penalty
                self.reward_breakdown['damage_taken'] -= flag_penalty
                self.reward_breakdown_detailed['health.hit_flag_penalty'] = -flag_penalty
                info['hit_received_flag'] = True
        self.prev_damage_flag = damage_flag

        # ================================================================
        # 4. MONSTER DAMAGE DEALT (player hitting monsters)
        # ================================================================
        monster_hit_reward, monster_damage = self._check_monster_damage(current_state)
        if monster_hit_reward > 0:
            reward += monster_hit_reward
            self.reward_breakdown['monster_hit'] += monster_hit_reward
            self.reward_breakdown_detailed['monster_hit.monster_damage'] = monster_hit_reward
            self.monsters_hit_count               += 1
            self.hit_count                        += 1
            self.total_monster_damage             += monster_damage
            self.monster_damage_since_zone_change += monster_damage
            info['monster_hit'] = True

        # ================================================================
        # 5. ZONE SETUP (compute once, used by multiple sections below)
        # ================================================================
        current_zone      = current_state.get('current_zone', 0) or 0
        zone_just_changed = (self.prev_zone is not None and current_zone != self.prev_zone)

        # ================================================================
        # 6. MONSTER ZONE + COMBAT STATE (MonsterZoneTracker)
        # ================================================================
        reward, in_combat = self.monster_zone_tracker.update(
            current_state          = current_state,
            took_damage            = took_damage,
            zone_just_changed      = zone_just_changed,
            current_time           = current_time,
            reward                 = reward,
            info                   = info,
            reward_breakdown       = self.reward_breakdown,
            reward_breakdown_detailed = self.reward_breakdown_detailed,
            monsters_hit_count     = self.monsters_hit_count,
        )

        # ================================================================
        # 7. ACTIONS (attack / block / dodge — only rewarded in combat)
        # ================================================================
        zone_has_monsters = self.monster_zone_tracker.zone_has_monsters

        # combat_branch: 1=attack1, 2=attack2, 3=dodge, 4=draw/sheath, 5=z_target
        if combat_branch in (1, 2) and zone_has_monsters:  # Attack

            if monster_hit_reward > 0:
                attack_success = 0.4 * (2.0 if in_combat else 1.0)
                reward += attack_success
                self.reward_breakdown['hit'] += attack_success
                self.reward_breakdown_detailed['hit.hit_success'] = attack_success
                info['attack_success'] = True
            elif in_combat:
                attack_reward = self.BONUS_ATTACK_ATTEMPT * 2.0
                reward += attack_reward
                self.reward_breakdown['defensive_actions'] += attack_reward
                self.reward_breakdown_detailed['defensive_actions.attack_attempt'] = attack_reward
                info['attack_attempt'] = True


        elif combat_branch == 3 and in_combat and zone_has_monsters:  # Dodge
            reward += self.BONUS_DODGE_ATTEMPT
            self.reward_breakdown['defensive_actions'] += self.BONUS_BLOCK_ATTEMPT
            self.reward_breakdown_detailed['defensive_actions.block'] = self.BONUS_BLOCK_ATTEMPT
            info['block_attempt'] = True


        elif combat_branch == 4 and in_combat and zone_has_monsters:  # Draw/sheath
            # (optional: add reward for weapon management)
            pass

        # ================================================================
        # 8. COMBO
        # ================================================================
        if self.consecutive_hits > 1:
            combo_bonus = min(self.consecutive_hits * 0.5, 5.0)
            reward += combo_bonus
            self.reward_breakdown['hit'] += combo_bonus
            self.reward_breakdown_detailed['hit.combo_bonus'] = combo_bonus
            info['combo'] = self.consecutive_hits

        # ================================================================
        # 9. POSITION TRACKING (distance traveled)
        # ================================================================
        x = current_state.get('player_x')
        y = current_state.get('player_y')
        z = current_state.get('player_z')

        if x is not None and y is not None and z is not None:
            if self.prev_position is not None:
                prev_x, prev_y, prev_z = self.prev_position
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)

                if distance > 5000:
                    logger.debug(f"Aberrant distance: {distance:.0f} — ignored (teleport/reset)")
                    distance = 0

                if distance < 1000:
                    self.total_distance_traveled += distance

            self.prev_position = (x, y, z)

        # ================================================================
        # 10. CURIOSITY / EXPLORATION REWARDS
        # ================================================================
        reward = self._check_curiosity_rewards(current_state, reward, info)

        # ================================================================
        # 11. ZONE CHANGE BONUS
        # ================================================================
        # Recompute zone_just_changed here (same value, but after curiosity rewards
        # may have updated internal state)
        zone_just_changed = (self.prev_zone is not None and current_zone != self.prev_zone)

        if zone_just_changed:
            self.exploration_tracker.pause_creation(duration=2.0)
            self.monster_damage_since_zone_change = 0
            # MonsterZoneTracker handles its own reset on zone change

            if hasattr(self.monster_zone_tracker, '_monsters_detected_logged'):
                self.monster_zone_tracker._monsters_detected_logged = False

            if self.last_zone_change_time is None:
                can_reward = True
            else:
                can_reward = (current_time - self.last_zone_change_time) >= self.zone_change_cooldown

            if can_reward:
                zone_bonus = self.BONUS_ZONE_CHANGE
                reward    += zone_bonus
                self.reward_breakdown['zone_change'] += zone_bonus
                self.reward_breakdown_detailed['zone_change.zone_bonus'] = zone_bonus
                info['zone_changed']          = True
                self.last_zone_change_time    = current_time

        self.prev_zone = current_zone

        # ================================================================
        # 12. CAMP PENALTIES / EXIT BONUSES (CampTracker)
        # ================================================================
        reward = self.camp_tracker.calculate(
            current_zone              = current_zone,
            zone_just_changed         = zone_just_changed,
            reward                    = reward,
            info                      = info,
            reward_breakdown          = self.reward_breakdown,
            reward_breakdown_detailed = self.reward_breakdown_detailed,
        )

        # Keep just_died in sync (MonsterZoneTracker → CampTracker needs it)
        if info.get('player_died'):
            self.camp_tracker.just_died = True

        # ================================================================
        # 13. OXYGEN (OxygenTracker)
        # ================================================================
        current_oxygen = current_state.get('time_underwater')
        reward = self.oxygen_tracker.calculate(
            current_oxygen            = current_oxygen,
            reward                    = reward,
            info                      = info,
            reward_breakdown          = self.reward_breakdown,
            reward_breakdown_detailed = self.reward_breakdown_detailed,
        )
        self.prev_oxygen = current_oxygen

        # ── WATER CUBE MARKING ───────────────────────────────────────────
        # Mark the current cube as water whenever the player is underwater (oxygen < 100)
        _oxy_val = current_state.get('time_underwater')
        if _oxy_val is not None and 0 <= int(_oxy_val) < 100:
            _water_cube = getattr(self.exploration_tracker, 'current_cube', None)
            if _water_cube is not None:
                self.exploration_tracker.marker_system.mark_water(_water_cube)
                logger.debug(f"Water cube marked (oxygen={_oxy_val})")

        # ================================================================
        # 14. HP STATE (low / critical / buffed)
        # ================================================================
        health_delta = 0.0
        if current_hp < 30:
            health_delta -= self.PENALTY_LOW_HP
            self.reward_breakdown_detailed['health.low_hp_penalty'] = -self.PENALTY_LOW_HP
        if current_hp < 15:
            health_delta -= self.PENALTY_CRITICAL_HP
            self.reward_breakdown_detailed['health.critical_hp_penalty'] = -self.PENALTY_CRITICAL_HP
        if current_hp > 100 and current_zone != 0:
            health_delta += self.BONUS_GOOD_HEALTH * 1.5
            self.reward_breakdown_detailed['health.buffed_hp_bonus'] = self.BONUS_GOOD_HEALTH * 1.5
        if current_hp > 80 and current_zone != 0:
            health_delta += self.BONUS_GOOD_HEALTH
            self.reward_breakdown_detailed['health.good_health_bonus'] = self.BONUS_GOOD_HEALTH

        reward += health_delta
        self.reward_breakdown['health'] = self.reward_breakdown.get('health', 0.0) + health_delta

        # ================================================================
        # 15. STAMINA
        # ================================================================
        stamina = current_state.get('player_stamina', 0) or 0
        if self.prev_stamina is not None:
            stamina_delta = stamina - self.prev_stamina
            if abs(stamina_delta) > 100:
                logger.warning(f"Aberrant stamina delta: {stamina_delta} — ignored")
        self.prev_stamina = stamina

        if stamina < 22:
            reward -= self.PENALTY_LOW_STAMINA
            self.reward_breakdown['penalties'] -= self.PENALTY_LOW_STAMINA
            self.reward_breakdown_detailed['penalties.stamina_low'] = -self.PENALTY_LOW_STAMINA
            info['stamina_low'] = True

        if stamina > 100:
            reward += 0.02
            self.reward_breakdown['other'] += 0.02
            self.reward_breakdown_detailed['other.buffed_stamina_bonus'] = 0.02
            info['stamina_buffed'] = True

        # ================================================================
        # 16. DEATH PENALTY
        # ================================================================
        if prev_state and self.prev_death_count is not None:
            if death_count > self.prev_death_count:
                base_penalty = self.PENALTY_DEATH_BASE * (1 + 0.5 * death_count)
                reduction    = self._calculate_death_penalty_reduction()
                final_penalty = base_penalty * (1.0 - reduction)

                reward -= final_penalty
                self.reward_breakdown['death'] -= final_penalty
                self.reward_breakdown_detailed['penalties.death_penalty'] = -final_penalty

                info['player_died']     = True
                info['death_number']    = death_count
                info['death_penalty']   = final_penalty
                info['death_count']     = death_count

                if death_count >= 3:
                    info['critical_death'] = True
                    logger.info(f"3RD DEATH — Episode will terminate. Final penalty: -{final_penalty:.2f}")
                else:
                    logger.info(
                        f"PLAYER DIED #{death_count} | base={base_penalty:.2f} "
                        f"reduction={reduction:.1%} final=-{final_penalty:.2f}"
                    )

                # Reset camp state on death
                self.camp_tracker.camp_entry_time          = None
                self.camp_tracker.camp_penalty_triggered   = False
                self.camp_tracker.last_penalty_time        = None
                self.camp_tracker.just_died                = True
        else:
            self.prev_death_count = death_count

        self.prev_death_count = death_count

        # ================================================================
        # 17. STATIONARY / IDLE PENALTY
        # ================================================================
        current_pos = (x, y, z)

        is_stationary = False
        if (
            current_pos[0] is not None
            and self.prev_position is not None
            and self.prev_position[0] is not None
        ):
            dx = current_pos[0] - self.prev_position[0]
            dy = current_pos[1] - self.prev_position[1]
            dz = current_pos[2] - self.prev_position[2]
            is_stationary = (np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) < 0.1
                             and movement_branch == 0 and combat_branch == 0)

        if is_stationary:
            self.frames_stationary += 1
            if self.frames_stationary > 90:
                idle_time       = (self.frames_stationary - 90) / 30.0
                idle_multiplier = 1.0 + (idle_time / 10.0)
                idle_penalty    = 0.6 * idle_multiplier / 30.0
                reward -= idle_penalty
                self.reward_breakdown['penalties'] -= idle_penalty
                self.reward_breakdown_detailed['penalties.idle'] = -idle_penalty
                info['idle_penalty'] = idle_penalty
                info['idle_time']    = idle_time
        else:
            self.frames_stationary = 0
            self.idle_start_time   = None

        # ================================================================
        # 18. ORIENTATION TRACKING
        # ================================================================
        self.prev_orientation = current_state.get('player_orientation')

        # ================================================================
        # 19. IN-GAME MENU PENALTY
        # ================================================================
        in_menu_raw = bool(current_state.get('in_game_menu', False))
        menu_penalty = 0.0

        # ── Debounce: 2 consecutive frames required to confirm any state change ──
        _MENU_DEBOUNCE = 2
        if in_menu_raw:
            self._menu_open_streak = min(self._menu_open_streak + 1, 10)
            self._menu_closed_streak = 0
        else:
            self._menu_closed_streak = min(self._menu_closed_streak + 1, 10)
            self._menu_open_streak = 0

        menu_just_opened = (self._menu_open_streak == _MENU_DEBOUNCE and not self.prev_in_menu)
        menu_just_closed = (self._menu_closed_streak == _MENU_DEBOUNCE and self.prev_in_menu)

        # Stable state follows prev_in_menu; update only on confirmed transitions
        in_menu = self.prev_in_menu
        if menu_just_opened:
            in_menu = True
        elif menu_just_closed:
            in_menu = False

        if menu_just_opened:
            self.game_menu_entry_time = current_time
            self.last_menu_penalty_time = None
            self.game_menu_open_count += 1
            info['game_menu_opened'] = True
            info['game_menu_count'] = self.game_menu_open_count
            logger.debug(f"Menu confirmed OPEN (streak={self._menu_open_streak}) — count={self.game_menu_open_count}")

        elif menu_just_closed:
            if self.game_menu_entry_time is not None:
                time_in_menu = current_time - self.game_menu_entry_time
                self.game_menu_total_time += time_in_menu
                info['game_menu_closed'] = True
                info['time_in_menu'] = time_in_menu
                self.game_menu_entry_time = None
            logger.debug(f"Menu confirmed CLOSED (streak={self._menu_closed_streak})")

        self.prev_in_menu = in_menu  # store debounced state, not raw read

        if in_menu and self.game_menu_entry_time is not None:
            time_in_menu          = current_time - self.game_menu_entry_time
            open_count_multiplier = 1.0 + (self.game_menu_open_count * 0.05)

            if time_in_menu >= self.PENALTY_MENU_THRESHOLD:
                if self.last_menu_penalty_time is None:
                    base_menu_penalty = self.PENALTY_MENU_BASE * open_count_multiplier
                    menu_penalty     += base_menu_penalty
                    self.last_menu_penalty_time = current_time
                    self.reward_breakdown_detailed['penalties.menu_initial'] = -base_menu_penalty
                    info['menu_initial_penalty'] = True
                    logger.info(
                        f"MENU open >5s (open #{self.game_menu_open_count}) "
                        f"— penalty: -{base_menu_penalty:.2f}"
                    )
                else:
                    time_since_last = current_time - self.last_menu_penalty_time
                    if time_since_last >= self.MENU_RECURRING_DELAY:
                        recurring = self.PENALTY_MENU_RECURRING * open_count_multiplier
                        menu_penalty += recurring
                        self.last_menu_penalty_time = current_time
                        self.reward_breakdown_detailed['penalties.menu_recurring'] = -recurring
                        info['menu_recurring_penalty'] = True

        reward -= menu_penalty
        self.reward_breakdown['menu_penalty'] -= menu_penalty
        info['in_game_menu']                    = in_menu
        info['game_menu_total_time']            = self.game_menu_total_time
        info['game_menu_open_count_multiplier'] = 1.0 + (self.game_menu_open_count * 0.5)

        # ================================================================
        # 20. STATS SNAPSHOT (for GUI / callbacks)
        # ================================================================
        info['total_hits']              = self.hit_count
        info['hit_count']               = self.hit_count
        info['total_damage_dealt']      = self.total_damage_dealt
        info['total_monster_damage']    = self.total_monster_damage
        info['monsters_killed_count']   = self.monsters_killed_count
        info['current_hp']              = current_hp
        info['current_stamina']         = stamina
        info['death_count']             = death_count
        info['total_distance']          = self.total_distance_traveled
        info['camp_total_time']         = self.camp_tracker.camp_total_time
        info['monsters_hit_count']      = self.monsters_hit_count
        info['quest_time']              = current_state.get('quest_time')
        info['oxygen_penalty_count']    = self.oxygen_tracker.oxygen_penalty_count
        info['left_monster_zone_count'] = self.monster_zone_tracker.left_monster_zone_count

        # Monster HP for GUI (small monsters + large monster current + max)
        for i in range(1, 6):
            k = f'smonster{i}_hp'
            info[k] = current_state.get(k, 0) or 0
        info['lmonster1_hp'] = current_state.get('lmonster1_hp', 0) or 0
        info['lmonster1_hp_max'] = current_state.get('lmonster1_hp_max', 0) or 0

        # ----------------------------------------------------------------
        # Ensure all breakdown categories exist (even if zero)
        # ----------------------------------------------------------------
        for cat in list(self.reward_breakdown.keys()):
            if cat not in self.reward_breakdown:
                self.reward_breakdown[cat] = 0.0

        self.last_reward_details           = self.reward_breakdown.copy()
        info['reward_breakdown']           = {
            k: float(v) if v is not None else 0.0
            for k, v in self.reward_breakdown.items()
        }
        info['reward_breakdown_detailed']  = {
            k: float(v) if v is not None else 0.0
            for k, v in self.reward_breakdown_detailed.items()
        }
        info['reward_breakdown_current']          = info['reward_breakdown'].copy()
        info['reward_breakdown_detailed_current'] = info['reward_breakdown_detailed'].copy()

        # ----------------------------------------------------------------
        # Periodic multi-marker debug check
        # ----------------------------------------------------------------
        self._debug_marker_check_count += 1
        if self._debug_marker_check_count % 1000 == 0:
            for zone_id, cubes in self.exploration_tracker.cubes_by_zone.items():
                for cube in cubes:
                    if len(cube.markers) > 1:
                        marker_names = [m.name for m in cube.markers.keys()]
                        logger.debug(
                            f"Multi-marker cube (step {self._debug_marker_check_count}): "
                            f"pos=({cube.center_x:.0f},{cube.center_y:.0f},{cube.center_z:.0f}) "
                            f"zone={cube.zone_id} markers={', '.join(marker_names)}"
                        )
                        break

        return reward

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    @staticmethod
    def _calculate_hp_penalty_multiplier(current_hp: float) -> float:
        """
        Exponential multiplier for damage penalties based on current HP.
        Lower HP → higher multiplier (more punishing to take damage when already hurt).
        Range: 0.1 – 2.0
        """
        if current_hp <= 0:
            return 2.0
        return float(np.clip(np.exp(-current_hp / 50.0), 0.1, 2.0))

    def _check_curiosity_rewards(self, current_state: dict, reward: float, info: dict) -> float:
        """
        Compute exploration / curiosity rewards using the ExplorationTracker.

        Rewards for discovering new cubes, penalties for revisiting known areas.
        Discovery is globally scaled down by 10×, then an additional /2 in camp (zone 0).
        """
        current_zone = current_state.get('current_zone', 0) or 0
        x = current_state.get('player_x')
        y = current_state.get('player_y')
        z = current_state.get('player_z')

        if x is None or y is None or z is None:
            return reward

        exploration_result = self.exploration_tracker.update_position(
            x, y, z,
            zone_id=current_zone,
            action=info.get('last_action'),
        )

        discovery_reward = exploration_result['discovery_reward']

        # Global scaling: reduce exploration reward by 10×
        discovery_reward /= 50.0

        if current_zone == 0:
            if self.camp_tracker.just_died:
                discovery_reward = 0.0
                info['camp_exploration_blocked'] = True
            else:
                # Additional /2 in camp (total = /20 vs base)
                discovery_reward /= 2.0
            if discovery_reward > 0:
                info['exploration_camp_penalty'] = True

        discovery_reward = min(discovery_reward, 2.0)

        if discovery_reward > 0:
            if exploration_result.get('new_cube'):
                self.reward_breakdown_detailed['exploration.new_cube_bonus'] = discovery_reward
            else:
                self.reward_breakdown_detailed['exploration.new_zone_bonus'] = discovery_reward

        revisit_penalty = exploration_result.get('revisit_penalty', 0.0)
        curiosity_reward = discovery_reward - revisit_penalty

        if revisit_penalty > 0:
            self.reward_breakdown_detailed['exploration.revisit_penalty'] = -revisit_penalty
            info['revisit_penalty'] = revisit_penalty

        if exploration_result['new_cube']:
            info['new_cube_discovered'] = True
        if exploration_result.get('cube_created'):
            info['cube_created'] = True
        if exploration_result['visit_count'] > 0:
            info['cube_visit_count'] = exploration_result['visit_count']

        tracker_stats = self.exploration_tracker.get_stats()
        info['total_cubes']         = tracker_stats['total_cubes']
        info['zones_discovered']    = tracker_stats['zones_discovered']
        info['exploration_visits']  = tracker_stats['total_visits']

        self.reward_breakdown['exploration'] = (
            self.reward_breakdown.get('exploration', 0.0) + curiosity_reward
        )
        return reward + curiosity_reward

    def _check_monster_damage(self, current_state: dict) -> tuple:
        """
        Compute reward for damage dealt to monsters (HP delta per step).

        Returns:
            (reward, total_damage_dealt)
        """
        reward       = 0.0
        total_damage = 0

        for i in range(1, 6):
            key = f'smonster{i}_hp'
            current_hp = current_state.get(key)
            if current_hp is None or current_hp < 0:
                continue

            prev_hp = self.prev_small_monsters_hp.get(i)
            if prev_hp is not None and prev_hp > current_hp:
                damage = prev_hp - current_hp
                if damage > 1000:
                    logger.warning(
                        f"Aberrant small monster HP delta: {damage} "
                        f"(monster {i}: {prev_hp} → {current_hp}) — ignored"
                    )
                    self.prev_small_monsters_hp[i] = current_hp
                    continue

                damage = min(damage, 50)
                total_damage += damage
                # Small monster: flat hit bonus + small damage multiplier
                m_reward = self.REWARD_SMONSTER_HIT + (damage * self.REWARD_SMONSTER_DAMAGE_MULT)

                if current_hp == 0 and prev_hp > 0:
                    m_reward += self.BONUS_KILL_SMALL_MONSTER
                    self.monsters_killed_count += 1
                    logger.info(f"Small monster {i} KILLED! Bonus: +{self.BONUS_KILL_SMALL_MONSTER:.1f}")

                reward += m_reward

            self.prev_small_monsters_hp[i] = current_hp

        for i in range(1, 2):
            # Skip if already killed (pointer chain returns garbage after death)
            if self.lmonster_killed.get(i, False):
                continue

            key = f'lmonster{i}_hp'
            current_hp = current_state.get(key)
            if current_hp is None or current_hp < 0:
                continue

            # Track max HP for %-based reward (read once when first seen > 0)
            max_hp_key = f'lmonster{i}_hp_max'
            hp_max = current_state.get(max_hp_key, 0) or 0
            if hp_max > 0 and i not in self.lmonster_hp_max:
                self.lmonster_hp_max[i] = hp_max
                logger.info(f"Large monster {i} max HP tracked: {hp_max}")
            elif hp_max > 0:
                # Update if game reports a higher value (shouldn't happen, safety net)
                self.lmonster_hp_max[i] = max(self.lmonster_hp_max[i], hp_max)

            effective_max = self.lmonster_hp_max.get(i, 0)

            prev_hp = self.prev_large_monsters_hp.get(i)
            if prev_hp is not None and prev_hp > current_hp:
                damage = prev_hp - current_hp
                if damage > 5000:
                    logger.warning(
                        f"Aberrant boss HP delta: {damage} "
                        f"(boss {i}: {prev_hp} → {current_hp}) — ignored"
                    )
                    self.prev_large_monsters_hp[i] = current_hp
                    continue

                # Large monster: %-based reward proportional to death penalty
                if effective_max > 0:
                    hp_pct = damage / effective_max
                    m_reward = self.REWARD_LMONSTER_HIT + (hp_pct * self.REWARD_LMONSTER_DAMAGE_SCALE)
                else:
                    # Fallback if max HP unknown: flat reward (should rarely happen)
                    m_reward = self.REWARD_LMONSTER_HIT + (damage * 0.01)
                    logger.debug(f"Large monster {i}: max HP unknown, using flat fallback")

                if current_hp == 0 and prev_hp > 0:
                    m_reward += self.BONUS_KILL_LARGE_MONSTER
                    self.monsters_killed_count += 1
                    self.lmonster_killed[i] = True  # Stop tracking post-death garbage
                    logger.info(
                        f"BOSS {i} DEFEATED! Kill bonus: +{self.BONUS_KILL_LARGE_MONSTER:.1f} "
                        f"(total damage reward ≈ {self.REWARD_LMONSTER_DAMAGE_SCALE:.1f})"
                    )

                reward += m_reward
                total_damage += damage

            self.prev_large_monsters_hp[i] = current_hp

        return reward, total_damage

    def _calculate_death_penalty_reduction(self) -> float:
        """
        Reduction factor for the death penalty based on how much damage was dealt.
        Up to 10% reduction for damage dealt, up to 20% extra for kills.
        Capped at 80% total reduction.
        """
        if self.monster_damage_since_zone_change == 0:
            return 0.0
        reduction = min(self.monster_damage_since_zone_change / 2000.0, 0.10)
        if self.monsters_killed_count > 0:
            reduction += 0.20
        return min(reduction, 0.80)

    # ==========================================================================
    # EPISODE LIFECYCLE
    # ==========================================================================

    def reset(self):
        """
        Reset all per-episode state for a new episode.
        Preserves exploration discoveries (cubes) across episodes.
        """
        # Delegate to sub-trackers
        self.oxygen_tracker.reset()
        self.camp_tracker.reset()
        self.monster_zone_tracker.reset()

        # Reset exploration episode counters (but keep discovered cubes)
        self.exploration_tracker.reset_episode()

        # Verify octree integrity after reset
        for zone_id in self.exploration_tracker.cubes_by_zone.keys():
            if not self.exploration_tracker.verify_octree_integrity(zone_id):
                logger.warning(f"Octree inconsistency after reset — zone {zone_id}")

        # Own state
        self.prev_hp                          = None
        self.prev_stamina                     = None
        self.hp_recovery_given                = False
        self.stamina_recovery_given           = False
        self.hp_recovery_accumulated          = 0.0
        self.stamina_recovery_accumulated     = 0.0
        self.prev_damage_flag                 = None
        self.prev_death_count                 = 0
        self.prev_position                    = None
        self.prev_orientation                 = None
        self.prev_zone                        = None
        self.prev_sharpness                   = None
        self.prev_oxygen                      = None

        self.prev_small_monsters_hp.clear()
        self.prev_large_monsters_hp.clear()
        self.lmonster_hp_max.clear()
        self.lmonster_killed.clear()
        self.monsters_hit_count               = 0
        self.total_monster_damage             = 0
        self.monsters_killed_count            = 0
        self.monster_damage_since_zone_change = 0

        self.frames_stationary                = 0
        self.idle_start_time                  = None
        self.total_distance_traveled          = 0.0
        self.total_damage_dealt               = 0
        self.hit_count                        = 0
        self.consecutive_hits                 = 0

        self.game_menu_entry_time             = None
        self.game_menu_total_time             = 0.0
        self.last_menu_penalty_time           = None
        self.game_menu_open_count             = 0
        self.prev_in_menu                     = False
        self._menu_open_streak                = 0
        self._menu_closed_streak              = 0

        self.last_zone_change_time            = None
        self._combat_log_count                = 0

        self.reward_breakdown                 = {}
        self.reward_breakdown_detailed        = {}
        self.last_reward_details              = {}

    def full_reset(self):
        """
        Complete reset including all exploration discoveries.
        Use at the start of a new training run (not between episodes).
        """
        self.reset()
        self.exploration_tracker = ExplorationTracker(
            cube_size          = self.cube_size,
            max_cubes          = self.max_cubes,
            compression_target = 0.85
        )

    # ==========================================================================
    # STATS & SUMMARY
    # ==========================================================================

    def get_stats(self) -> dict:
        """Return a snapshot of current episode statistics for the GUI / callbacks."""
        exploration_stats = self.exploration_tracker.get_stats()
        camp_stats        = self.camp_tracker.get_stats()
        mz_stats          = self.monster_zone_tracker.get_stats()

        return {
            'hit_count':               self.hit_count,
            'total_damage_dealt':      self.total_damage_dealt,
            'total_monster_damage':    self.total_monster_damage,
            'monsters_killed_count':   self.monsters_killed_count,
            'consecutive_hits':        self.consecutive_hits,
            'total_distance':          self.total_distance_traveled,
            'monsters_hit_count':      self.monsters_hit_count,
            'oxygen_penalty_count':    self.oxygen_tracker.oxygen_penalty_count,
            'game_menu_total_time':    self.game_menu_total_time,
            'game_menu_count':         self.game_menu_open_count,
            'zones_discovered':        exploration_stats['zones_discovered'],
            'areas_explored':          exploration_stats['total_cubes'],
            'total_cubes':             exploration_stats['total_cubes'],
            'exploration_visits':      exploration_stats['total_visits'],
            'left_monster_zone_count': mz_stats['left_monster_zone_count'],
            'camp_total_time':         camp_stats['camp_total_time'],
            'reward_breakdown':        self.last_reward_details.copy(),
            'exploration_cubes':       exploration_stats.get('exploration_cubes', {}),
        }

    def get_reward_breakdown_summary(self) -> str:
        """Return a human-readable summary of the last step's reward breakdown."""
        if not self.last_reward_details:
            return "No reward calculated yet"
        lines = [
            f"{cat:20s}: {'+' if val > 0 else ''}{val:+7.2f}"
            for cat, val in self.last_reward_details.items()
            if abs(val) > 0.01
        ]
        return "\n".join(lines)