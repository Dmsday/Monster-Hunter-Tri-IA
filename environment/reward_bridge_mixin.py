"""
reward_bridge_mixin.py — Interface between MonsterHunterEnv and RewardCalculator.

Reads game state from MemoryReader, detects damage, delegates to
RewardCalculator, and enriches step_info with GUI-friendly stats.

Mixin attributes expected on `self`:
    use_memory, memory, reward_calc, prev_raw_memory,
    episode_count, episode_steps, total_steps
"""

from typing import Dict, Tuple, Union, List

from info.module_logger import get_module_logger

logger = get_module_logger('reward_bridge')


class RewardBridgeMixin:
    """Reward calculation bridge for MonsterHunterEnv."""

    def _calculate_reward(self, action) -> Tuple[float, dict]:
        """
        Read game state, compute reward via RewardCalculator, enrich info.

        Args:
            action: The action just executed (0-18).

        Returns:
            (reward, reward_info) tuple.
        """
        reward_info: Dict[str, Union[int, float, bool, str, List, Dict]] = {
            'episode_num': self.episode_count,
            'episode_steps': self.episode_steps,
            'total_steps': self.total_steps,
        }

        if not (self.use_memory and self.memory):
            return 0.0, reward_info

        try:
            raw_memory = self.memory.read_game_state()

            # Detect damage taken since last step
            took_damage = self._detect_damage(raw_memory)

            # Delegate to RewardCalculator
            if self.reward_calc:
                reward = self.reward_calc.calculate(
                    self.prev_raw_memory,
                    raw_memory,
                    action,
                    reward_info,
                    took_damage=took_damage,
                )
                reward_info.update(self.reward_calc.get_stats())
            else:
                reward = 0.01

            # Update prev state for next step's delta
            self.prev_raw_memory = raw_memory

            # Enrich info with GUI-friendly stats
            self._enrich_info(reward_info, raw_memory)

            return reward, reward_info

        except Exception as exc:
            logger.error(f"Reward calculation error: {exc}")
            return 0.0, reward_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_damage(self, raw_memory: dict) -> bool:
        """Check if player took damage since the previous step."""
        if self.prev_raw_memory is None:
            return False
        prev_hp = self.prev_raw_memory.get('player_hp', 0) or 0
        curr_hp = raw_memory.get('player_hp', 0) or 0
        delta = prev_hp - curr_hp
        return 0 < delta < 100  # Ignore resets (>100 HP jump)

    def _enrich_info(self, info: dict, raw_memory: dict):
        """Add game state fields to info for GUI / callbacks."""
        info['hp'] = raw_memory.get('player_hp')
        info['stamina'] = raw_memory.get('player_stamina')
        info['damage_last_hit'] = raw_memory.get('damage_last_hit')
        info['death_count'] = raw_memory.get('death_count')
        info['current_zone'] = raw_memory.get('current_zone')
        info['player_x'] = raw_memory.get('player_x')
        info['player_y'] = raw_memory.get('player_y')
        info['player_z'] = raw_memory.get('player_z')
        info['orientation'] = raw_memory.get('player_orientation')
        info['money'] = raw_memory.get('money')
        info['quest_time'] = raw_memory.get('quest_time')
        info['sharpness'] = raw_memory.get('sharpness')
        info['inventory'] = self.memory.read_inventory()
        info['in_game_menu'] = raw_memory.get('in_game_menu', False)
        info['item_selected'] = raw_memory.get('item_selected', 24)
        info['game_menu_open_count'] = (
            getattr(self.reward_calc, 'game_menu_open_count', 0)
            if self.reward_calc else 0
        )
