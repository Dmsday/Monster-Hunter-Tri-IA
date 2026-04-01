"""
gui_state.py — Shared data store for the Training GUI.
Holds all mutable state (agent data, episode history, config, progress)
so every tab / panel can read from a single source of truth.
"""

import os
import json
import time
from collections import deque
from typing import Dict, Any, Optional, List

from info.module_logger import get_module_logger

logger = get_module_logger('gui_state')

# Fixed ordered reward categories for the rewards chart.
ALL_REWARD_CATS: List[str] = [
    'exploration', 'monster_hit', 'monster_zone',
    'zone_change', 'hit', 'health', 'defensive_actions',
    'combat', 'oxygen', 'other',
    'damage_taken', 'death', 'camp_penalty',
    'menu_penalty', 'penalties', 'sharpness_penalty',
]

HISTORY_SIZE   = 300
UPDATE_RATE_MS = 300
CHART_RATE_MS  = 2000


class GuiState:
    """Central data store — no tkinter dependency."""

    def __init__(self):
        # ── Config persistence ─────────────────────────────────────
        _cfg_dir = os.path.join("config", "user")
        os.makedirs(_cfg_dir, exist_ok=True)
        self.config_file = os.path.join(_cfg_dir, "gui_config.json")
        self.config = self._load_config()

        # ── Lifecycle flags ────────────────────────────────────────
        self.running        = False
        self.stop_requested = False
        self._closing       = False
        self.start_time     = time.time()

        # ── Agent / instance selection ─────────────────────────────
        self._data: Dict[tuple, Dict]         = {}
        self._episode_data: Dict[tuple, List] = {}
        self._known_agents: List[int]         = [0]
        self._known_instances: Dict[int, List[int]] = {0: [0]}
        self.sel_agent   = 0
        self.sel_instance = 0

        # ── Progress ───────────────────────────────────────────────
        self.total_timesteps:     int   = 0
        self.training_start_time: float = time.time()
        self.global_total_steps:  int   = 0

        # ── Chart throttle ─────────────────────────────────────────
        self.last_chart_time = 0.0
        self.last_step_time  = time.time()
        self.last_step_count = 0

        # ── Reward breakdown cache (peak-hold) ─────────────────────
        self.rew_cache:    Dict[str, float] = {}
        self.rew_cache_ts: Dict[str, float] = {}
        self.rew_cache_ttl = 2.0

        # ── Public deques (written by callbacks) ───────────────────
        self.episode_history   = deque(maxlen=HISTORY_SIZE)
        self.reward_history    = deque(maxlen=HISTORY_SIZE)
        self.length_history    = deque(maxlen=HISTORY_SIZE)
        self.hits_history      = deque(maxlen=HISTORY_SIZE)
        self.reward_breakdown_history          = deque(maxlen=300)
        self.reward_breakdown_detailed_history = deque(maxlen=300)

        # ── External refs set by runner / callbacks ────────────────
        self.isolated_envs: List[int] = []
        self.disabled_heads: set      = set()

    # ── Agent / instance registration ──────────────────────────────

    def register_agent_instance(self, aid: int, iid: int):
        """Ensure agent and instance are tracked."""
        if aid not in self._known_agents:
            self._known_agents.append(aid)
            self._known_agents.sort()
        if aid not in self._known_instances:
            self._known_instances[aid] = []
        if iid not in self._known_instances[aid]:
            self._known_instances[aid].append(iid)
            self._known_instances[aid].sort()

    # ── Stats update ───────────────────────────────────────────────

    def update_stats(self, stats: Dict[str, Any]):
        """Thread-safe store for (agent_id, instance_id) stats."""
        aid = int(stats.get('agent_id', 0) or 0)
        iid = int(stats.get('instance_id', stats.get('env_idx', 0)) or 0)
        key = (aid, iid)
        self.register_agent_instance(aid, iid)

        if key not in self._data:
            self._data[key] = self.default_stats()
        entry = self._data[key]

        # Accumulate episode reward
        if 'reward' in stats:
            entry['episode_reward'] = entry.get('episode_reward', 0.0) + float(stats['reward'])
        entry.update(stats)

        # Breakdowns
        if 'reward_breakdown' in stats:
            self.reward_breakdown_history.append(stats['reward_breakdown'])
        if 'reward_breakdown_detailed' in stats:
            self.reward_breakdown_detailed_history.append(stats['reward_breakdown_detailed'])

        # Global step counter
        total = int(stats.get('total_steps', 0) or 0)
        if total > self.global_total_steps:
            self.global_total_steps = total

    def add_episode_data(self, episode: int, reward: float,
                         length: int, hits: int = 0,
                         agent_id: int = 0, instance_id: int = 0):
        try:
            episode = int(episode); reward = float(reward)
            length = int(length); hits = int(hits)
        except (ValueError, TypeError):
            return
        self.episode_history.append(episode)
        self.reward_history.append(reward)
        self.length_history.append(length)
        self.hits_history.append(hits)

        key = (agent_id, instance_id)
        if key not in self._episode_data:
            self._episode_data[key] = []
        self._episode_data[key].append({'ep': episode, 'r': reward, 'l': length, 'h': hits})
        if key in self._data:
            self._data[key]['episode_reward'] = 0.0

    def current_stats(self) -> Dict:
        """Return stats for currently selected agent/instance."""
        return self._data.get((self.sel_agent, self.sel_instance), self.default_stats())

    # ── Config persistence ─────────────────────────────────────────

    def _load_config(self) -> Dict:
        defaults = {'geometry': '1280x800+60+60'}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    defaults.update(json.load(f))
            except Exception:
                pass
        return defaults

    def save_config(self, geometry: Optional[str] = None):
        try:
            if geometry:
                self.config['geometry'] = geometry
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def default_stats() -> Dict:
        return {
            'episode': 0, 'step': 0, 'total_steps': 0,
            'reward': 0.0, 'episode_reward': 0.0,
            'hp': 100.0, 'stamina': 100.0,
            'hits': 0, 'deaths': 0, 'zone': 0, 'action': 0,
            'player_x': 0.0, 'player_y': 0.0, 'player_z': 0.0,
            'orientation': 0.0, 'money': 0, 'distance': 0.0,
            'sharpness': 150, 'quest_time': 5400,
            'in_combat': False, 'in_game_menu': False,
            'in_monster_zone': False, 'monsters_present': False,
            'monster_count': 0,
            'total_cubes': 0, 'zones_discovered': 0, 'exploration_visits': 0,
            'inventory': [], 'exploration_cubes': {},
            'reward_breakdown': {}, 'reward_breakdown_detailed': {},
            'resolved_action': None,
        }
