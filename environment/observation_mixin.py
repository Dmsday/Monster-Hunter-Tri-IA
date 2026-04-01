"""
observation_mixin.py — Build, validate, and visualise observations.

Provides:
    _get_observation()                  — main observation builder
    _get_dummy_observation()            — zero-filled fallback
    _get_dummy_memory_state()           — default game state dict
    _save_observation_visualization()   — debug PNG every N steps

Mixin attributes expected on `self`:
    _vision_available, use_vision, use_memory, memory, state_fusion,
    preprocessor, reward_calc, observation_space, _obs_queue,
    total_steps, _frames_consumed, instance_id, _agent_id
"""

import os
import queue

import numpy as np
from gymnasium import spaces

from utils.memory_vector import build_memory_vector
from info.module_logger import get_module_logger

logger = get_module_logger('observation')

try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False
    plt = None


class ObservationMixin:
    """Observation building and validation for MonsterHunterEnv."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def _get_observation(self):
        """
        Build the current observation dict.

        Branches:
            1. Vision available  → pull frame from async queue, add memory/map
            2. Memory-only       → build memory vector, fill visual with zeros
        """
        observation = {}

        # ---- BRANCH 1: Vision hardware available ----
        if self._vision_available:
            try:
                if not hasattr(self, '_obs_queue'):
                    logger.error("obs_queue missing — capture thread not started?")
                    return self._get_dummy_observation()

                visual_state = self._obs_queue.get(timeout=0.1)
                self._frames_consumed += 1

                # Add memory / exploration map if reader available
                if self.memory is not None and self.state_fusion is not None:
                    raw_memory = self.memory.get_latest_state()
                    if raw_memory is None:
                        raw_memory = self._get_dummy_memory_state()

                    observation['visual'] = visual_state
                    observation['exploration_map'] = self._build_exploration_map(raw_memory)

                    if self.use_memory:
                        observation['memory'] = build_memory_vector(
                            raw_memory, reward_calc=self.reward_calc
                        )
                else:
                    observation['visual'] = visual_state

            except queue.Empty:
                logger.warning("Queue empty — frame missed")
                return self._get_dummy_observation()

        # ---- BRANCH 2: Memory only (or vision failed at init) ----
        elif self.use_memory and self.memory:
            if not hasattr(self, '_memory_only_logged'):
                if self.use_vision and not self._vision_available:
                    logger.warning(
                        "VISION FALLBACK — vision failed at init, "
                        "returning zero-filled visual observations"
                    )
                else:
                    logger.warning("MEMORY-ONLY MODE")
                self._memory_only_logged = True

            try:
                raw_memory = self.memory.read_game_state()
                observation['memory'] = build_memory_vector(
                    raw_memory, reward_calc=self.reward_calc
                )

                # Fill visual/map with zeros when vision was requested but failed
                if self.use_vision and not self._vision_available:
                    self._fill_missing_keys(observation)

            except Exception as exc:
                logger.error(f"Memory read error: {exc}")
                return self._get_dummy_observation()

        # ---- Consistency check (log once) ----
        self._ensure_observation_consistency(observation)

        # Periodic debug + visualization
        if self.total_steps % 1000 == 0:
            logger.debug(f"Observation keys: {list(observation.keys())}")
            if self._vision_available:
                try:
                    self._save_observation_visualization(observation)
                except KeyError as exc:
                    logger.warning(f"Visualization skipped (missing key): {exc}")

        return observation

    def _get_dummy_observation(self):
        """Return a zero-filled observation matching observation_space."""
        if isinstance(self.observation_space, spaces.Dict):
            dummy = {}
            for key, space in self.observation_space.spaces.items():
                dummy[key] = np.zeros(space.shape, dtype=space.dtype)
            return dummy
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    @staticmethod
    def _get_dummy_memory_state() -> dict:
        """Default game state when memory read fails."""
        return {
            'player_hp': 100.0,
            'player_hp_recoverable': 0.0,
            'player_stamina': 100.0,
            'player_hp_raw': 2516600000,
            'player_stamina_raw': 20000000,
            'player_x': 0.0, 'player_y': 0.0, 'player_z': 0.0,
            'player_orientation': 0.0,
            'current_zone': 0,
            'damage_last_hit': 0.0,
            'money': 0,
            'death_count': 0,
            'stamina_low': False,
            'quest_time': 5400,
            'attack_defense_value': 0,
            'sharpness': 150,
            'in_game_menu': False,
            'item_selected': 24,
            'smonster1_hp': 0, 'smonster2_hp': 0, 'smonster3_hp': 0,
            'smonster4_hp': 0, 'smonster5_hp': 0, 'lmonster1_hp': 0,
            'time_underwater': 0,
            'oxygen_valid': False,
            'inventory_items': [],
            'current_map': 0,
            'quest_ended': False,
            'on_reward_screen': False,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_exploration_map(self, raw_memory: dict) -> np.ndarray:
        """Create the 15x15x4 exploration minimap from raw memory."""
        px = raw_memory.get('player_x') or 0.0
        py = raw_memory.get('player_y') or 0.0
        pz = raw_memory.get('player_z') or 0.0
        zone = raw_memory.get('current_zone') or 0
        return self.state_fusion.create_exploration_map_with_channels(
            (px, py, pz), zone
        )

    def _fill_missing_keys(self, observation: dict):
        """Fill zero arrays for keys in observation_space but not in observation."""
        if not isinstance(self.observation_space, spaces.Dict):
            return
        for key in self.observation_space.spaces:
            if key not in observation:
                space = self.observation_space.spaces[key]
                observation[key] = np.zeros(space.shape, dtype=space.dtype)

    def _ensure_observation_consistency(self, observation: dict):
        """Check for missing/extra keys, wrong shapes, NaN/Inf. Log once."""
        if not isinstance(self.observation_space, spaces.Dict):
            return

        expected = set(self.observation_space.spaces.keys())
        actual = set(observation.keys())

        # Missing keys → fill zeros
        missing = expected - actual
        if missing:
            if not hasattr(self, '_missing_keys_logged'):
                logger.error(f"Missing observation keys: {missing} — filling zeros")
                self._missing_keys_logged = True
            for key in missing:
                space = self.observation_space.spaces[key]
                observation[key] = np.zeros(space.shape, dtype=space.dtype)

        # Extra keys → remove
        extra = actual - expected
        if extra:
            if not hasattr(self, '_extra_keys_logged'):
                logger.warning(f"Extra observation keys: {extra} — removing")
                self._extra_keys_logged = True
            for key in extra:
                del observation[key]

        # Shape + NaN/Inf validation
        for key in expected:
            if key not in observation:
                continue
            val = observation[key]
            exp_space = self.observation_space.spaces[key]

            if not isinstance(val, np.ndarray):
                logger.error(f"observation['{key}'] is not ndarray")
                observation[key] = np.zeros(exp_space.shape, dtype=exp_space.dtype)
                continue

            if val.shape != exp_space.shape:
                logger.error(
                    f"Shape mismatch for '{key}': {val.shape} vs {exp_space.shape}"
                )
                observation[key] = np.zeros(exp_space.shape, dtype=exp_space.dtype)

            if np.any(np.isnan(val)) or np.any(np.isinf(val)):
                logger.error(f"NaN/Inf in '{key}'")
                observation[key] = np.nan_to_num(val, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _save_observation_visualization(self, observation: dict):
        """
        Save a composite PNG showing visual frame, exploration map channels,
        and memory stats. Only called when use_vision=True.
        """
        if not _MPL:
            return

        required = ['visual', 'memory', 'exploration_map']
        missing = [k for k in required if k not in observation]
        if missing:
            logger.error(f"Cannot create visualization — missing keys: {missing}")
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'AI Vision — Step {self.total_steps}', fontsize=16, fontweight='bold')

            # --- Visual frame ---
            visual = observation['visual']
            if self.preprocessor.grayscale:
                last = np.stack([visual[:, :, -1]] * 3, axis=-1)
            else:
                last = visual[:, :, -3:]
            last = np.clip(last, 0.0, 1.0)
            axes[0, 0].imshow(last)
            axes[0, 0].set_title(f'Visual ({visual.shape[0]}x{visual.shape[1]})')
            axes[0, 0].axis('off')

            # --- Exploration map channels ---
            emap = observation['exploration_map']
            cmaps = ['viridis', 'hot', 'Blues', 'RdYlGn_r']
            labels = ['Ch0: Visits', 'Ch1: Position', 'Ch2: Recent', 'Ch3: Markers']
            positions = [(0, 1), (0, 2), (1, 0), (1, 1)]

            for i, (pos, cmap, label) in enumerate(zip(positions, cmaps, labels)):
                ch = emap[:, :, i]
                im = axes[pos].imshow(ch, cmap=cmap, vmin=ch.min(), vmax=ch.max())
                axes[pos].set_title(f'{label} [{ch.min():.2f}, {ch.max():.2f}]')
                axes[pos].axis('off')
                plt.colorbar(im, ax=axes[pos], fraction=0.046, pad=0.04)

            # --- Memory stats text ---
            mem = observation['memory']
            stats = (
                f"MEMORY VECTOR ({mem.shape[0]} features)\n\n"
                f"HP: {mem[0]:.0f}/100\n"
                f"Stamina: {mem[2]:.0f}/100\n"
                f"Pos: ({mem[3]:.0f}, {mem[4]:.0f}, {mem[5]:.0f})\n"
                f"Orientation: {mem[6]:.0f}\n"
                f"Zone: {int(mem[7])}\n"
                f"Deaths: {int(mem[10])}\n"
                f"Quest Time: {int(mem[61])}s\n"
                f"Monsters: {int(mem[63])}\n"
                f"Sharpness: {int(mem[65])}\n"
                f"In Menu: {'YES' if mem[66] > 0.5 else 'NO'}\n"
            )
            axes[1, 2].text(
                0.1, 0.5, stats, fontsize=9,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            )
            axes[1, 2].axis('off')

            debug_dir = os.path.join('.', 'vision', 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, f'ai_vision_step_{self.total_steps}.png')
            plt.tight_layout()
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Vision visualization saved: {path}")

        except Exception as exc:
            logger.error(f"Visualization error: {exc}")
