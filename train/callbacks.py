"""
callbacks.py — Stable-Baselines3 callbacks for the training loop.

Exports:
    GUIUpdateCallback        – pushes per-step stats to the training GUI
    ExplorationCheckpoint    – periodic exploration-map saves (JSON)
    ProgressWindowCallback   – floating progress bar for no-GUI mode
    ConsoleMessageManager    – helper to deduplicate console output
    build_callbacks(...)     – assemble the full callback list for learn()
"""

import os
import json
import time
import traceback

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from GUI.training_gui import TrainingGUI
from info.advanced_logging import TrainingLogger, LoggingCallback
from info.agent_context import AgentContext
from info.module_logger import get_module_logger
from reward.cube_markers import MarkerType

logger = get_module_logger('train.callbacks')


# ======================================================================
#  CONSOLE MESSAGE MANAGER
# ======================================================================

class ConsoleMessageManager:
    """Prevent the same console message from being printed every step."""

    def __init__(self):
        self.last_messages: dict = {}
        self.message_counts: dict = {}

    def print_grouped(self, key: str, message: str, update_same_line: bool = True):
        if key in self.last_messages and self.last_messages[key] == message:
            self.message_counts[key] = self.message_counts.get(key, 1) + 1
            if update_same_line:
                print(f"\r{message} (x{self.message_counts[key]})", end='', flush=True)
        else:
            if key in self.last_messages:
                print()  # newline after previous group
            print(message, flush=True)
            self.last_messages[key] = message
            self.message_counts[key] = 1

    def reset(self, key: str = None):
        if key:
            self.last_messages.pop(key, None)
            self.message_counts.pop(key, None)
        else:
            self.last_messages.clear()
            self.message_counts.clear()


# ======================================================================
#  GUI UPDATE CALLBACK
# ======================================================================

class GUIUpdateCallback(BaseCallback):
    """Push per-step game state and reward breakdown to the training GUI."""

    def __init__(self, gui: TrainingGUI, agent_id: int = None,
                 num_envs: int = 1, allocation: dict = None, verbose=0):
        super().__init__(verbose)
        self.gui = gui
        self.agent_id = agent_id
        self.num_envs = num_envs

        # Build env_idx -> owning agent_id lookup from allocation
        # allocation format: {agent_id: [instance_ids]}
        self._env_to_agent = {}
        if allocation:
            for aid, insts in allocation.items():
                for iid in insts:
                    self._env_to_agent[iid] = aid

        if self.agent_id is not None:
            AgentContext.set_current_agent(self.agent_id)

        self.episode_count = 0
        self.console = ConsoleMessageManager()
        self.last_item_selected = 24
        self.last_item_selected_name = "None"

    # -- SB3 hook ----------------------------------------------------------
    def _on_step(self) -> bool:
        return self.gui_update()

    # -- Core logic (also called directly by MultiAgentTrainer) ------------
    def gui_update(self) -> bool:
        if self.agent_id is not None:
            AgentContext.set_current_agent(self.agent_id)

        all_infos   = self.locals.get('infos', [])
        all_rewards = self.locals.get('rewards', [])
        all_dones   = self.locals.get('dones', [])

        info = all_infos[0] if all_infos else {}

        # -- Extract exploration cubes for the 3-D map ---------------------
        all_env_cubes: dict = {}
        try:
            vec_env = None
            if hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'env'):
                vec_env = self.model.env
            elif hasattr(self, '_env_ref') and self._env_ref is not None:
                vec_env = self._env_ref

            if vec_env is not None and hasattr(vec_env, 'get_attr'):
                for ei, rc in enumerate(vec_env.get_attr('reward_calc')):
                    if rc and hasattr(rc, 'exploration_tracker'):
                        all_env_cubes[ei] = _extract_cubes(rc.exploration_tracker)
        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        exploration_cubes = all_env_cubes.get(0, {})

        # -- Extract resolved actions for the action visualizer tab --------
        resolved_actions: dict = {}  # env_idx -> list (len NUM_HEADS)
        try:
            _ve = vec_env  # reuse ref from cubes block above
            if _ve is not None and hasattr(_ve, 'get_attr'):
                for ei, ra in enumerate(
                        _ve.get_attr('_last_resolved_action')):
                    if ra is not None:
                        resolved_actions[ei] = (
                            ra.tolist() if hasattr(ra, 'tolist') else list(ra))
        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        # -- Push stats for secondary environments ------------------------
        all_actions = self.locals.get('actions', [])
        for idx in range(1, len(all_infos)):
            ei = all_infos[idx]
            rew = float(all_rewards[idx]) if idx < len(all_rewards) else 0.0
            rb = ei.get('reward_breakdown', {}) or {}
            owner_agent = self._env_to_agent.get(idx, idx)
            # Extract per-env action data for the action tab
            raw_act = all_actions[idx] if idx < len(all_actions) else None
            res_act = resolved_actions.get(idx)
            self.gui.update_stats(_build_env_stats(
                env_idx=idx, agent_id=owner_agent,
                ei=ei, rew=rew, rb=rb,
                cubes=all_env_cubes.get(idx, {}),
                raw_action=raw_act,
                resolved_action=res_act,
            ))

        # -- Update isolation status in GUI --------------------------------
        _iso_list = [
            i for i in range(len(all_infos))
            if all_infos[i].get('isolated', False)
        ]
        if hasattr(self.gui, '_isolated_envs'):
            self.gui._isolated_envs = _iso_list

        # -- Log anomalous rewards ----------------------------------------
        current_reward = self.locals.get('rewards', [0])[0]
        if abs(current_reward) > 50:
            _log_reward_anomaly(info, current_reward)

        if info.get('quest_ended_screen') or info.get('quest_ended_after_action'):
            logger.info(f"QUEST END — Episode {info.get('episode_num', '?')}, "
                        f"Steps: {info.get('episode_steps', '?')}")

        # -- Collect primary environment fields ----------------------------
        episode_num   = info.get('episode_num', 0) or 0
        episode_steps = max(0, int(info.get('episode_steps', 0) or 0))
        total_steps   = info.get('total_steps', 0) or 0
        current_zone  = info.get('current_zone', 0) or 0

        reward_breakdown          = info.get('reward_breakdown', {})
        reward_breakdown_detailed = info.get('reward_breakdown_detailed', {})

        item_selected      = info.get('item_selected', 24) or 24
        item_selected_name = info.get('item_selected_name', 'None')

        if item_selected != self.last_item_selected:
            if item_selected == 24:
                logger.info("Item deselected")
            else:
                logger.info(f"Item selected: Slot {item_selected + 1} — {item_selected_name}")
            self.last_item_selected = item_selected
            self.last_item_selected_name = item_selected_name

        # -- Primary environment stats -> GUI ------------------------------
        self.gui.update_stats({
            'agent_id': self.agent_id,
            'episode': episode_num,
            'step': episode_steps,
            'total_steps': total_steps,
            'reward': self.locals.get('rewards', [0])[0],
            'hp': info.get('hp', 100),
            'stamina': info.get('stamina', 100),
            'hits': info.get('hit_count', 0),
            'deaths': info.get('death_count', 0),
            'zone': current_zone,
            'action': self.locals.get('actions', [None])[0],
            'resolved_action': resolved_actions.get(0),   # After conflict resolution (for action tab)
            'player_x': info.get('player_x', 0.0) or 0.0,
            'player_y': info.get('player_y', 0.0) or 0.0,
            'player_z': info.get('player_z', 0.0) or 0.0,
            'orientation': info.get('orientation', 0.0) or 0.0,
            'money': info.get('money', 0) or 0,
            'distance': info.get('total_distance', 0.0) or 0.0,
            'sharpness': info.get('sharpness', 100) or 100,
            'quest_time': info.get('quest_time', 0) or 0,
            'reward_breakdown': reward_breakdown,
            'reward_breakdown_detailed': reward_breakdown_detailed,
            'inventory': info.get('inventory', []),
            'total_cubes': info.get('total_cubes', 0) or 0,
            'zones_discovered': info.get('zones_discovered', 0) or 0,
            'exploration_visits': info.get('exploration_visits', 0) or 0,
            'left_monster_zone_count': info.get('left_monster_zone_count', 0) or 0,
            'monsters_present': bool(info.get('in_monster_zone', False)),
            'monster_count': info.get('monster_count', 0) or 0,
            'in_monster_zone': bool(info.get('in_monster_zone', False)),
            'in_combat': bool(info.get('in_combat', False)),
            'smonster1_hp': info.get('smonster1_hp', 0) or 0,
            'smonster2_hp': info.get('smonster2_hp', 0) or 0,
            'smonster3_hp': info.get('smonster3_hp', 0) or 0,
            'smonster4_hp': info.get('smonster4_hp', 0) or 0,
            'smonster5_hp': info.get('smonster5_hp', 0) or 0,
            'lmonster1_hp': info.get('lmonster1_hp', 0) or 0,   # Large monster HP for combat tab
            'zone_reward_total': (reward_breakdown.get('monster_zone', 0)
                                  + reward_breakdown.get('exploration', 0)),  # was 'curiosity' (doesn't exist)
            'exploration_cubes': exploration_cubes,
            'item_selected': item_selected,
            'item_selected_name': item_selected_name,
            'in_game_menu': info.get('in_game_menu', False),
            'game_menu_open_count': info.get('game_menu_open_count', 0),
            'game_menu_total_time': info.get('game_menu_total_time', 0.0),
        })

        # -- Zone change logging -------------------------------------------
        if info.get('zone_changed'):
            try:
                mdl = getattr(self, 'model', None)
                if mdl and hasattr(mdl, 'env') and hasattr(mdl.env, 'get_attr'):
                    rc = mdl.env.get_attr('reward_calc')[0]
                    if rc and hasattr(rc, 'exploration_tracker'):
                        n = len(rc.exploration_tracker.cubes_by_zone.get(current_zone, []))
                        logger.debug(f"Zone {current_zone}: {n} active cubes")
            except (AttributeError, KeyError, IndexError):
                pass

        if info.get('compression_triggered'):
            logger.debug(f"Cube compression triggered — total: {info.get('total_cubes', 0)}")

        # -- Episode end detection -----------------------------------------
        for eidx, (done, einfo) in enumerate(zip(all_dones, all_infos)):
            if not done:
                continue
            ep = einfo.get('episode', {})
            if ep:
                r = float(ep.get('r', 0.0))
                l = int(ep.get('l', 0))
                h = einfo.get('hit_count', 0)
                logger.info(f"Episode {self.episode_count} done (env {eidx}): "
                            f"reward={r:.2f} length={l} hits={h}")
                self.gui.add_episode_data(self.episode_count, r, l, h)
            else:
                self.gui.add_episode_data(self.episode_count, 0.0, 0, 0)
            self.episode_count += 1
            self.console.reset()

        # Periodic debug log
        if self.n_calls % 300 == 0:
            logger.debug(f"Step {self.n_calls} | Ep {episode_num} | "
                         f"Cumulative reward: {info.get('episode_reward', 0.0):.2f}")

        # Check GUI stop button
        if self.gui.should_stop():
            logger.warning("Stop requested via GUI")
            return False

        return True


# ======================================================================
#  EXPLORATION CHECKPOINT
# ======================================================================

class ExplorationCheckpoint(BaseCallback):
    """Periodically save the exploration map as a JSON file."""

    def __init__(self, save_path: str, save_freq: int = 50000,
                 advanced_logger: TrainingLogger = None):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq
        self.advanced_logger = advanced_logger

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        filepath = None
        try:
            if hasattr(self.model.env, 'get_attr'):
                rc = self.model.env.get_attr('reward_calc')[0]
                if rc and hasattr(rc, 'exploration_tracker'):
                    stats = _make_json_serializable(rc.exploration_tracker.get_stats())

                    filepath = os.path.join(self.save_path, f'exploration_{self.n_calls}.json')
                    with open(filepath, 'w') as f:
                        json.dump(stats, f, indent=2, ensure_ascii=False)

                    logger.info(f"Exploration map saved: {filepath} "
                                f"(cubes={stats['total_cubes']}, "
                                f"zones={stats['zones_discovered']})")

            if self.advanced_logger and filepath:
                self.advanced_logger.log_checkpoint(filepath, self.n_calls)

        except Exception as exc:
            if self.advanced_logger:
                self.advanced_logger.log_error(exc, context="Exploration checkpoint save")
            logger.error(f"Exploration save error: {exc}")
            traceback.print_exc()

        return True


# ======================================================================
#  PROGRESS WINDOW (NO-GUI MODE)
# ======================================================================

class ProgressWindowCallback(BaseCallback):
    """Floating progress bar shown when training without the main GUI."""

    def __init__(self, total_timesteps: int, num_envs: int, verbose: int = 0):
        super().__init__(verbose)
        from multi.multi_agent_trainer import TrainingProgressWindow
        self._win = TrainingProgressWindow(num_agents=1, total_timesteps=total_timesteps)
        self._num_envs = num_envs
        self._start = time.time()
        self._last_fps_steps = 0
        self._last_fps_time = time.time()

    def _on_step(self) -> bool:
        current = self.num_timesteps
        elapsed = time.time() - self._start

        dt = time.time() - self._last_fps_time
        if dt >= 2.0:
            fps = (current - self._last_fps_steps) / max(dt, 1e-9)
            self._last_fps_steps = current
            self._last_fps_time = time.time()
        else:
            fps = current / max(elapsed, 1e-9)

        total = max(self.locals.get('total_timesteps', current), 1)
        pct = min((current / total) * 100, 100.0)
        self._win.update(agent_id=0, steps=current, fps=fps, episodes=0, pct=pct)
        return True

    def _on_training_end(self):
        self._win.close()


# ======================================================================
#  CALLBACK ASSEMBLY
# ======================================================================

def build_callbacks(
    args,
    gui,
    env,
    models_dir: str,
    training_logger: TrainingLogger,
    training_loggers: list,
    allocation_result: dict,
):
    """
    Assemble the full callback list for SB3 learn().

    Returns:
        (callbacks, gui_callback, logging_callbacks)
    """
    callbacks = []
    logging_callbacks = []

    # -- Model checkpoint (every ~10 % of the run) -------------------------
    timesteps = args.debug_steps if args.debug_steps else args.timesteps
    n_envs = args.num_instances if args.num_instances > 1 else 1
    ckpt_freq = max(1, timesteps // (10 * n_envs))

    ckpt = CheckpointCallback(save_freq=ckpt_freq, save_path=models_dir, name_prefix="checkpoint")
    callbacks.append(ckpt)
    logger.info(f"Checkpoint every ~{ckpt_freq * n_envs:,} timesteps "
                f"({ckpt_freq} SB3 steps x {n_envs} envs)")

    # -- Logging callback(s) -----------------------------------------------
    alloc = allocation_result.get('allocation', {}) if allocation_result else {}

    if args.num_agents > 1 and training_loggers:
        # One logging callback per PPO agent
        agent_primary = {}
        for aid, insts in alloc.items():
            for iid in sorted(insts):
                if aid not in agent_primary and iid < len(training_loggers):
                    agent_primary[aid] = training_loggers[iid]
                    break

        for aid in range(args.num_agents):
            tl = agent_primary.get(aid, training_logger)
            alloc_envs = set(alloc.get(aid, []))
            env_loggers = [
                training_loggers[iid] if (iid in alloc_envs and iid < len(training_loggers)) else tl
                for iid in range(args.num_instances)
            ]
            lc = LoggingCallback(tl, agent_id=aid, all_env_loggers=env_loggers)
            logging_callbacks.append(lc)
    else:
        # Single-agent: one logging callback
        lc = LoggingCallback(training_logger, all_env_loggers=training_loggers)
        callbacks.append(lc)
        logging_callbacks.append(lc)

    # -- Exploration checkpoint --------------------------------------------
    callbacks.append(ExplorationCheckpoint(
        save_path=models_dir, save_freq=50000, advanced_logger=training_logger))

    # -- GUI callback ------------------------------------------------------
    gui_callback = None
    if gui:
        if args.num_instances > 1 and args.num_agents == 1:
            gui_callback = GUIUpdateCallback(
                gui, agent_id=None, num_envs=args.num_instances, allocation=alloc)
        elif args.num_agents > 1:
            gui_callback = GUIUpdateCallback(
                gui, agent_id=0, num_envs=args.num_instances, allocation=alloc)
        else:
            gui_callback = GUIUpdateCallback(
                gui, agent_id=None, num_envs=1, allocation=alloc)

        # Allow the callback (and the GUI stop button) to reach the env
        gui_callback._env_ref = env
        if hasattr(gui, 'stop_button'):
            gui._env_ref = env
        callbacks.append(gui_callback)

    return callbacks, gui_callback, logging_callbacks


# ======================================================================
#  PRIVATE HELPERS
# ======================================================================

def _extract_cubes(tracker) -> dict:
    """Snapshot cube data from an ExplorationTracker for the GUI 3-D map."""
    tracker.sync_all_markers_to_cubes()
    result = {}
    for zid, cube_list in tracker.cubes_by_zone.items():
        result[zid] = [
            {
                'center_x': c.center_x, 'center_y': c.center_y,
                'center_z': c.center_z, 'size_x': c.size_x,
                'size_y': c.size_y, 'size_z': c.size_z,
                'size': c.size,
                'avg_size': (c.size_x + c.size_y + c.size_z) / 3.0,
                'visit_count': c.visit_count, 'total_visits': c.total_visits,
                'effective_visit_count': c.effective_visit_count,
                'zone_id': c.zone_id,
                'blocked_directions': c.blocked_directions,
                'markers': c.markers,
            }
            for c in cube_list
        ]
    return result


def _build_env_stats(env_idx, agent_id, ei, rew, rb, cubes,
                     raw_action=None, resolved_action=None) -> dict:
    """
    Build the stats dict pushed to the GUI for a secondary environment.
    """
    zone_rew = rb.get('monster_zone', 0) + rb.get('exploration', 0)
    return {
        'agent_id': agent_id, 'instance_id': env_idx,
        'reward': rew,
        'action': raw_action,                # Raw agent output for action tab
        'resolved_action': resolved_action,  # After conflict resolution
        'hp': ei.get('hp', 0) or 0,
        'stamina': ei.get('stamina', 0) or 0,
        'zone': ei.get('current_zone', 0) or 0,
        'episode': ei.get('episode_num', 0) or 0,
        'step': ei.get('episode_steps', 0) or 0,
        'total_steps': ei.get('total_steps', 0) or 0,
        'deaths': ei.get('death_count', 0) or 0,
        'hits': ei.get('hit_count', 0) or 0,
        'in_combat': bool(ei.get('in_combat', False)),
        'in_monster_zone': bool(ei.get('in_monster_zone', False)),
        'monsters_present': bool(ei.get('in_monster_zone', False)),
        'monster_count': ei.get('monster_count', 0) or 0,
        'player_x': ei.get('player_x', 0.0) or 0.0,
        'player_y': ei.get('player_y', 0.0) or 0.0,
        'player_z': ei.get('player_z', 0.0) or 0.0,
        'orientation': ei.get('orientation', 0.0) or 0.0,
        'money': ei.get('money', 0) or 0,
        'distance': ei.get('total_distance', 0.0) or 0.0,
        'sharpness': ei.get('sharpness', 100) or 100,
        'quest_time': ei.get('quest_time', 0) or 0,
        'inventory': ei.get('inventory', []),
        'total_cubes': ei.get('total_cubes', 0) or 0,
        'zones_discovered': ei.get('zones_discovered', 0) or 0,
        'exploration_visits': ei.get('exploration_visits', 0) or 0,
        'left_monster_zone_count': ei.get('left_monster_zone_count', 0) or 0,
        'smonster1_hp': ei.get('smonster1_hp', 0) or 0,
        'smonster2_hp': ei.get('smonster2_hp', 0) or 0,
        'smonster3_hp': ei.get('smonster3_hp', 0) or 0,
        'smonster4_hp': ei.get('smonster4_hp', 0) or 0,
        'smonster5_hp': ei.get('smonster5_hp', 0) or 0,
        'lmonster1_hp': ei.get('lmonster1_hp', 0) or 0,   # Large monster HP for combat tab
        'in_game_menu': bool(ei.get('in_game_menu', False)),
        'game_menu_open_count': ei.get('game_menu_open_count', 0) or 0,
        'game_menu_total_time': ei.get('game_menu_total_time', 0.0) or 0.0,
        'item_selected': ei.get('item_selected', 24) or 24,
        'item_selected_name': ei.get('item_selected_name', 'None'),
        'zone_reward_total': zone_rew,
        'reward_breakdown': rb,
        'reward_breakdown_detailed': ei.get('reward_breakdown_detailed', {}),
        'exploration_cubes': cubes,
    }


def _log_reward_anomaly(info, reward):
    """Log details when reward exceeds the anomaly threshold."""
    logger.warning(f"REWARD ANOMALY: {reward:.2f}")
    logger.warning(f"  Episode: {info.get('episode_num', '?')}, "
                   f"Step: {info.get('episode_steps', '?')}")
    bd = info.get('reward_breakdown', {})
    if bd:
        for cat, val in sorted(bd.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(val) > 1.0:
                logger.warning(f"  {cat}: {val:+.2f}")
    logger.warning(f"  HP={info.get('hp', '?')} Zone={info.get('current_zone', '?')} "
                   f"Deaths={info.get('death_count', '?')}")


def _make_json_serializable(obj):
    """Recursively convert an object to JSON-safe types (handles MarkerType, numpy, etc.)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, MarkerType):
        return obj.value
    if hasattr(obj, '__class__') and 'Enum' in str(type(obj).__mro__):
        return obj.value
    if hasattr(obj, 'item'):   # numpy scalar
        return obj.item()
    if hasattr(obj, 'tolist'): # numpy array
        return obj.tolist()
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, MarkerType)
             else str(k) if not isinstance(k, (str, int, float, bool)) else k):
            _make_json_serializable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(i) for i in obj]
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"