"""
Trainer custom pour entraînement multi-agents PPO partageant des instances Dolphin.
Implémente la boucle collect/train avec scheduler.
"""

import os
import time
import copy
import traceback
from typing import List, Optional, Dict
from collections import defaultdict

import torch
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from multi.multi_agent_scheduler import MultiAgentScheduler
from info.module_logger import get_module_logger
from info.agent_context import AgentContext, EnvContext

logger = get_module_logger('multi_agent_trainer')

# ---------------------------------------------------------------------------
# Separate progress window — runs in its own daemon thread via tkinter
# ---------------------------------------------------------------------------
class TrainingProgressWindow:
    """
    Dark-themed training monitor with one progress card per agent.
    Runs in a background daemon thread — never blocks training.
    """

    _BG      = "#0f0f17"
    _CARD    = "#1a1a28"
    _BORDER  = "#2a2a40"
    _FG      = "#e2e8f0"
    _DIM     = "#64748b"
    _FONT    = "Segoe UI"

    _AGENT_PALETTE = [
        "#818cf8", "#34d399", "#fb923c", "#f472b6",
        "#38bdf8", "#a78bfa", "#facc15", "#4ade80",
    ]

    def __init__(self, num_agents: int, total_timesteps: int):
        self._num_agents  = num_agents
        self._total       = total_timesteps
        self._per_agent   = max(total_timesteps // max(num_agents, 1), 1)
        self._pending     = {}
        self._alive       = True
        self._start = time.time()
        self._isolated_envs = []  # Track isolated env indices for display

        # Widgets — set in _run(), read in _poll() (same thread via after())
        self._root        = None
        self._widgets     = {}   # aid -> {bar, pct_lbl, sub_lbl}
        self._global_bar  = None
        self._global_lbl  = None
        self._time_lbl    = None
        self._fps_lbl     = None
        self._eta_lbl     = None

        import threading
        threading.Thread(
            target=self._run, daemon=True, name="TrainingProgressWindow"
        ).start()

    # ------------------------------------------------------------------
    def _run(self):
        try:
            import tkinter as tk
            from tkinter import ttk

            root = self._root = tk.Tk()
            root.title("Training")
            root.configure(bg=self._BG)
            root.resizable(False, False)
            root.protocol("WM_DELETE_WINDOW", self._on_close)

            # ── ttk style ──────────────────────────────────────────────
            s = ttk.Style(root)
            s.theme_use("clam")

            for aid in range(self._num_agents):
                c = self._AGENT_PALETTE[aid % len(self._AGENT_PALETTE)]
                s.configure(f"A{aid}.Horizontal.TProgressbar",
                    troughcolor=self._BORDER, background=c,
                    bordercolor=self._BG, lightcolor=c, darkcolor=c,
                    thickness=6)

            s.configure("Global.Horizontal.TProgressbar",
                troughcolor=self._BORDER, background="#6366f1",
                bordercolor=self._BG, lightcolor="#6366f1",
                darkcolor="#6366f1", thickness=8)

            # ── Header ─────────────────────────────────────────────────
            hdr = tk.Frame(root, bg=self._BG)
            hdr.pack(fill="x", padx=20, pady=(16, 2))

            tk.Label(hdr, text="TRAINING", font=(self._FONT, 11, "bold"),
                     bg=self._BG, fg=self._FG).pack(side="left")

            self._time_lbl = tk.Label(hdr, text="0:00:00",
                font=(self._FONT, 9), bg=self._BG, fg=self._DIM)
            self._time_lbl.pack(side="right")

            tk.Label(root,
                text=f"{self._total:,} steps  ·  {self._num_agents} agent(s)",
                font=(self._FONT, 8), bg=self._BG, fg=self._DIM,
            ).pack(padx=20, anchor="w", pady=(0, 6))

            tk.Frame(root, bg=self._BORDER, height=1).pack(
                fill="x", padx=16, pady=4)

            # ── Agent cards ────────────────────────────────────────────
            for aid in range(self._num_agents):
                c = self._AGENT_PALETTE[aid % len(self._AGENT_PALETTE)]

                card = tk.Frame(root, bg=self._CARD,
                                highlightbackground=self._BORDER,
                                highlightthickness=1)
                card.pack(fill="x", padx=16, pady=3)

                row = tk.Frame(card, bg=self._CARD)
                row.pack(fill="x", padx=10, pady=(8, 2))

                tk.Label(row, text="●", font=(self._FONT, 7),
                         bg=self._CARD, fg=c).pack(side="left", padx=(0, 4))
                tk.Label(row, text=f"agent {aid}",
                         font=(self._FONT, 9, "bold"),
                         bg=self._CARD, fg=self._FG).pack(side="left")

                pct_lbl = tk.Label(row, text="0.0%",
                    font=(self._FONT, 9, "bold"),
                    bg=self._CARD, fg=self._DIM)
                pct_lbl.pack(side="right")

                bar = ttk.Progressbar(card,
                    style=f"A{aid}.Horizontal.TProgressbar",
                    length=420, maximum=self._per_agent,
                    mode="determinate")
                bar.pack(fill="x", padx=10, pady=2)

                sub_row = tk.Frame(card, bg=self._CARD)
                sub_row.pack(fill="x", padx=10, pady=(2, 8))
                sub_lbl = tk.Label(sub_row, text="waiting...",
                    font=(self._FONT, 8), bg=self._CARD, fg=self._DIM)
                sub_lbl.pack(side="left")

                self._widgets[aid] = dict(bar=bar, pct_lbl=pct_lbl,
                                          sub_lbl=sub_lbl)

            tk.Frame(root, bg=self._BORDER, height=1).pack(
                fill="x", padx=16, pady=4)

            # ── Global bar ─────────────────────────────────────────────
            gf = tk.Frame(root, bg=self._BG)
            gf.pack(fill="x", padx=16, pady=(2, 2))
            tk.Label(gf, text="total", font=(self._FONT, 8, "bold"),
                     bg=self._BG, fg=self._DIM).pack(side="left")
            self._global_lbl = tk.Label(gf,
                text=f"0 / {self._total:,}   0.0%",
                font=(self._FONT, 8), bg=self._BG, fg=self._DIM)
            self._global_lbl.pack(side="right")

            self._global_bar = ttk.Progressbar(root,
                style="Global.Horizontal.TProgressbar",
                length=420, maximum=self._total, mode="determinate")
            self._global_bar.pack(fill="x", padx=16, pady=(2, 4))

            foot = tk.Frame(root, bg=self._BG)
            foot.pack(fill="x", padx=16, pady=(0, 14))
            self._fps_lbl = tk.Label(foot, text="",
                font=(self._FONT, 8), bg=self._BG, fg=self._DIM)
            self._fps_lbl.pack(side="left")
            self._eta_lbl = tk.Label(foot, text="",
                font=(self._FONT, 8), bg=self._BG, fg=self._DIM)
            self._eta_lbl.pack(side="right")

            # Start polling loop
            root.after(500, self._poll) # noqa — PyCharm stub mismatch
            root.mainloop()

        except Exception as exc:
            logger.debug(f"TrainingProgressWindow could not open: {exc}")

    # ------------------------------------------------------------------
    def _poll(self):
        """Refresh widgets every 500 ms. Always reschedules even on error."""
        # Reschedule first — guarantees the loop never stops
        if self._alive and self._root is not None:
            try:
                self._root.after(500, self._poll)
            except Exception:
                return

        if not self._alive or self._root is None:
            return

        try:
            data    = dict(self._pending)
            total   = 0
            fps_sum = 0.0

            # Build isolation status string
            _iso_envs = getattr(self, '_isolated_envs', [])
            _iso_text = ""
            if _iso_envs:
                _iso_text = f"   ⚠ {len(_iso_envs)} env(s) isolated: {_iso_envs}"

            for aid, (steps, fps, episodes, pct) in data.items():
                w = self._widgets.get(aid)
                if w is None:
                    continue
                w["bar"]["value"] = steps
                w["pct_lbl"].config(text=f"{pct:5.1f}%")
                w["sub_lbl"].config(
                    text=f"{fps:.0f} fps   {steps:,} steps   {episodes} ep")
                total += steps
                fps_sum += fps

            # Show isolation warning in global label area
            if self._global_lbl is not None:
                gpct = (total / max(self._total, 1)) * 100
                base_text = f"{total:,} / {self._total:,}   {gpct:.1f}%"
                if _iso_text:
                    self._global_lbl.config(
                        text=base_text + _iso_text, fg="#fb923c")
                else:
                    self._global_lbl.config(
                        text=base_text, fg=self._DIM)

            # Global bar
            if self._global_bar is not None:
                self._global_bar["value"] = total

            # Elapsed time (always updated)
            elapsed = time.time() - self._start
            eh, rem = divmod(int(elapsed), 3600)
            em, es  = divmod(rem, 60)
            if self._time_lbl is not None:
                self._time_lbl.config(text=f"{eh}:{em:02d}:{es:02d}")

            # FPS + ETA
            if fps_sum > 0:
                if self._fps_lbl is not None:
                    self._fps_lbl.config(text=f"{fps_sum:.0f} fps total")
                if total > 0 and self._eta_lbl is not None:
                    eta_s = (self._total - total) / fps_sum
                    rh, rr = divmod(int(max(eta_s, 0)), 3600)
                    rm, rs  = divmod(rr, 60)
                    self._eta_lbl.config(text=f"eta {rh}:{rm:02d}:{rs:02d}")

        except Exception as exc:
            logger.debug(f"TrainingProgressWindow poll error: {exc}")

    # ------------------------------------------------------------------
    def update(self, agent_id: int, steps: int, fps: float,
               episodes: int, pct: float, isolated_envs: list = None):
        """Thread-safe update called from the trainer thread."""
        self._pending[agent_id] = (steps, fps, episodes, pct)
        if isolated_envs is not None:
            self._isolated_envs = isolated_envs

    def _on_close(self):
        self._alive = False
        if self._root:
            self._root.destroy()
            self._root = None

    def close(self):
        self._alive = False
        try:
            if self._root:
                self._root.quit()
        except Exception:
            pass

class MultiAgentTrainer:
    """
    Gère l'entraînement de plusieurs agents PPO partageant des instances Dolphin.

    Architecture :
    - N agents PPO (chacun avec son propre modèle, ses propres poids)
    - M instances Dolphin (M ≤ N, potentiellement M < N en instance-sharing)
    - Scheduler : décide quel agent joue sur quelle instance à chaque step
    - Boucle : collect N steps → update tous les agents → recommencer

    Note sur les buffers PPO :
    SB3's RolloutBuffer gère déjà le stockage des transitions (obs/action/reward/value/logprob).
    On l'alimente via rollout_buffer.add() et on déclenche l'optimisation manuellement.
    """

    def __init__(
            self,
            agents: List[PPO],
            env: VecEnv,
            scheduler: MultiAgentScheduler,
            steps_per_agent: int = 2048,
            callback: Optional[BaseCallback] = None,
            agent_callbacks: Optional[List[BaseCallback]] = None,
            scenario: str = "INSTANCE_SHARING",
            allocation: Optional[Dict[int, List[int]]] = None,
            models_dir: str = "models",
            weight_sync_every: int = 10,
    ):
        """
        Args:
            agents: Liste des agents PPO, indexés 0..N-1
            env: Environnement vectorisé (DummyVecEnv ou SubprocVecEnv)
            scheduler: Scheduler pour décider quel agent joue quand
            steps_per_agent: Nombre de steps à collecter par agent avant chaque update PPO
            callback: Callback SB3 partagé (optionnel, pour compatibilité)
            agent_callbacks: Liste de callbacks individuels (1 par agent, optionnel)
            scenario: Type de scénario pour le rééquilibrage weighted
                      ('INSTANCE_SHARING' ou 'AGENT_MULTIPLE_INSTANCES')
            allocation: Allocation initiale {agent_id: [instance_ids]} (pour référence)
        """
        self.agents = agents
        self.env = env
        self.scheduler = scheduler
        self.steps_per_agent = steps_per_agent
        self.callback = callback
        self.agent_callbacks = agent_callbacks
        self.scenario = scenario
        self.allocation = allocation
        self._models_dir = models_dir
        self.weight_sync_every = weight_sync_every

        self._env_consecutive_quest_end: Dict[int, int] = defaultdict(int)
        self._env_stuck_threshold = 10

        # Track previous done flag per agent - SB3 expects episode_start to be
        # True when the current obs is the FIRST of a new episode (i.e. when
        # the PREVIOUS step ended an episode), not when the current step ends one.
        self._last_done_per_agent: Dict[int, bool] = {
            aid: True for aid in range(len(agents))  # True = first step is an episode start
        }

        if self.agent_callbacks is None and self.callback is not None:
            self.agent_callbacks = [self.callback] * len(self.agents)

        self.total_timesteps = 0
        self.episode_counts = [0] * len(agents)
        self.episode_rewards: List[List[float]] = [[] for _ in range(len(agents))]

        logger.info("MultiAgentTrainer created")
        logger.info(f"   Agents      : {len(agents)}")
        logger.info(f"   Instances   : {env.num_envs}")
        logger.info(f"   Steps/agent : {steps_per_agent}")
        logger.info(f"   Mode        : {scheduler.mode}")
        logger.info(f"   Scenario    : {scenario}")

    def _save_agents(self, label: str = "checkpoint", models_dir: str = "models"):
        """
        Save all agents to disk. Called on interrupt or error.
        models_dir is best-effort — falls back to current directory.
        """
        saved = []
        for agent_id, agent in enumerate(self.agents):
            try:
                os.makedirs(models_dir, exist_ok=True)
                path = os.path.join(models_dir, f"{label}_agent_{agent_id}_{agent.num_timesteps}steps")
                agent.save(path)
                saved.append(f"agent {agent_id} -> {path}.zip ({agent.num_timesteps:,} steps)")
            except Exception as save_error:
                logger.error(f"Could not save agent {agent_id}: {save_error}")
        for msg in saved:
            logger.info(f"Saved: {msg}")

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _obs_to_tensor(obs, device: torch.device):
        """
        Convertit une observation (numpy ou dict) en tensor(s) PyTorch.

        Ajoute une dimension batch (unsqueeze(0)) car SB3 policy attend [B, ...].

        Args:
            obs: np.ndarray ou dict[str, np.ndarray]
            device: Device cible (cpu / cuda)

        Returns:
            torch.Tensor ou dict[str, torch.Tensor]
        """
        if isinstance(obs, dict):
            return {
                key: torch.as_tensor(value).unsqueeze(0).to(device)
                for key, value in obs.items()
            }
        return torch.as_tensor(obs).unsqueeze(0).to(device)

    @staticmethod
    def _get_env_obs(observations, env_idx: int):
        """
        Extract a single-env observation from VecEnv stacked format.
        Handles Dict obs space (returns {key: array[env_idx]})
        and Box obs space (returns array[env_idx]).
        """
        if isinstance(observations, dict):
            return {k: v[env_idx] for k, v in observations.items()}
        return observations[env_idx]

    def _store_transition(
            self,
            agent_id: int,
            obs,
            action,  # int (Discrete) or np.ndarray (MultiDiscrete)
            reward: float,
            done: bool,
    ):
        """
        Évalue la transition et l'ajoute au rollout buffer de l'agent.

        Cette méthode est appelée une fois par (instance, agent) concerné.
        Elle centralise la logique evaluate_actions + rollout_buffer.add
        qui était dupliquée entre le cas standard et le mode majority_vote.

        Args:
            agent_id: ID de l'agent dont le buffer doit recevoir la transition
            obs: Observation actuelle (numpy ou dict)
            action: Action exécutée (int)
            reward: Reward reçu (float)
            done: Vrai si l'épisode s'est terminé ce step
        """
        agent = self.agents[agent_id]

        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs, agent.device)
            action_array = np.asarray(action)
            action_tensor = torch.as_tensor(action_array).unsqueeze(0).to(agent.device)
            values, log_prob, _ = agent.policy.evaluate_actions(obs_tensor, action_tensor)
            values = values.flatten()
            log_prob = log_prob.flatten()

        # Use the PREVIOUS done flag as episode_start - SB3 convention:
        # episode_start=True means this obs is the first of a new episode
        episode_start = self._last_done_per_agent.get(agent_id, True)

        agent.rollout_buffer.add(
            obs=obs,
            action=action_array.reshape(1, -1),
            reward=np.array([reward]),
            episode_start=np.array([episode_start]),
            value=values,
            log_prob=log_prob
        )

        # Save current done for the NEXT call
        self._last_done_per_agent[agent_id] = done

    def _release_all_inputs(self) -> None:
        """Release all controller inputs before the PPO update phase to avoid frozen inputs."""
        try:
            for env in self.env.envs:
                ctrl = getattr(env, 'controller', None)
                if ctrl is not None and hasattr(ctrl, 'reset_all'):
                    ctrl.reset_all()
                if ctrl is not None and hasattr(ctrl, 'release_all_managed'):
                    ctrl.release_all_managed()
        except Exception as exc:
            logger.debug(f"Could not release inputs during update phase: {exc}")

    def _sync_weights(self, sync_every_n_cycles: int, current_cycle: int) -> None:
        """
        Periodically averages policy weights across all agents (FedAvg).
        Called after each update cycle. Agents keep learning independently
        between syncs, which maintains diversity while sharing improvements.
        """
        if current_cycle == 0 or current_cycle % sync_every_n_cycles != 0:
            return

        if len(self.agents) < 2:
            return

        logger.info(
            f"Weight sync cycle {current_cycle}: averaging policy across "
            f"{len(self.agents)} agents"
        )

        # Collect all state dicts
        state_dicts = [agent.policy.state_dict() for agent in self.agents]

        # Compute parameter-wise average
        avg_state = copy.deepcopy(state_dicts[0])
        for key in avg_state:
            avg_state[key] = torch.stack(
                [sd[key].float() for sd in state_dicts]
            ).mean(dim=0).to(avg_state[key].dtype)

        # Apply averaged weights to all agents
        for agent in self.agents:
            agent.policy.load_state_dict(avg_state)

        logger.info("Weight sync complete")

    # ================================================================
    # BOUCLE PRINCIPALE
    # ================================================================

    def train(self, total_timesteps: int, progress_bar: bool = True,
              show_progress_window: bool = True):
        """
        Boucle d'entraînement multi-agent : collect → update → repeat.

        Deux phases par itération :
        1. COLLECT  : chaque agent accumule steps_per_agent transitions dans son buffer
        2. UPDATE   : chaque agent lance ses epochs PPO sur son buffer, puis le vide

        Args:
            total_timesteps: Nombre total de steps à effectuer (répartis sur tous les agents)
            progress_bar: Affiche une barre de progression ASCII dans les logs toutes les 10s
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("STARTING MULTI-AGENT TRAINING")
        logger.info("=" * 70)
        logger.info(f"Total timesteps    : {total_timesteps:,}")
        logger.info(f"Timesteps/agent    : {total_timesteps // len(self.agents):,}")
        logger.info(f"Mode               : {self.scheduler.mode}")
        logger.info("")

        observations = self.env.reset()

        steps_collected: Dict[int, int] = defaultdict(int)
        total_steps_per_agent: Dict[int, int] = defaultdict(int)  # cumulative across cycles
        episodes_done: Dict[int, int] = defaultdict(int)
        start_time = time.time()
        last_log_time = start_time

        # show_progress_window=False when an external GUI (TrainingGUI) is active
        if show_progress_window:
            _progress_win = TrainingProgressWindow(
                num_agents=len(self.agents),
                total_timesteps=total_timesteps,
            )
        else:
            # Dummy object so the rest of the code can call _progress_win.update/close safely
            class _NullProgressWin:
                def update(self, *a, **kw): pass

                def close(self): pass

            _progress_win = _NullProgressWin()

        # ----------------------------------------------------------------
        # NOTE sur les buffers SB3 (RolloutBuffer) :
        # Chaque agent PPO a son propre RolloutBuffer de taille n_steps.
        # On alimente ce buffer via rollout_buffer.add() pendant COLLECT,
        # puis on déclenche train() → optimize_policy() pendant UPDATE.
        # Le buffer est réinitialisé (reset()) après chaque UPDATE.
        # ----------------------------------------------------------------

        try:
            _update_cycle = 0
            while self.total_timesteps < total_timesteps:

                # ============================================================
                # PHASE 1 : COLLECTION
                # Collecter steps_per_agent transitions pour chaque agent actif
                # ============================================================
                logger.debug("Collection phase starting...")

                # Use n_steps from the agent's rollout buffer as the hard limit.
                # steps_per_agent may exceed buffer size and cause IndexError on buffer.add().
                num_envs = self.env.num_envs
                num_agents = len(self.agents)
                # In instance-sharing mode, each agent gets num_envs/num_agents steps per outer iteration
                # so we need num_agents/num_envs more outer iterations to fill all buffers
                fill_ratio = max(1, (num_agents + num_envs - 1) // num_envs)
                steps_to_collect = self.agents[0].n_steps * fill_ratio

                # Pre-initialize to satisfy type checker - overwritten each iteration
                rewards = np.zeros(num_envs, dtype=np.float32)
                dones = np.zeros(num_envs, dtype=bool)
                infos: list = [{} for _ in range(num_envs)]

                for _ in range(steps_to_collect):
                    # Safety: skip if any buffer is already full
                    # only stop when ALL agents have full buffers
                    if all(
                            self.agents[aid].rollout_buffer.full
                            for aid in range(len(self.agents))
                    ):
                        logger.debug("All buffers full - stopping collection")
                        break

                    # --- Step each env individually to preserve AgentContext per agent ---
                    actions: List[np.ndarray] = []  # MultiDiscrete returns array of shape (NUM_HEADS,)
                    agents_used: List[int] = []
                    new_observations_list = []
                    rewards_list = []
                    dones_list = []
                    infos_list = []

                    for env_idx in range(self.env.num_envs):
                        _env = self.env.envs[env_idx]

                        # --- ISOLATION CHECK: skip envs with corrupted save states ---
                        if getattr(_env, '_isolated', False):
                            # Produce a dummy step so array shapes stay consistent
                            dummy_obs = self._get_env_obs(observations, env_idx)
                            new_observations_list.append(dummy_obs)
                            rewards_list.append(0.0)
                            dones_list.append(False)
                            infos_list.append({'isolated': True})
                            actions.append(np.zeros_like(
                                actions[0] if actions else
                                self.env.action_space.sample()
                            ))
                            agents_used.append(-2)  # -2 = isolated sentinel
                            continue

                        obs_for_env = self._get_env_obs(observations, env_idx)
                        action, agent_used = self.scheduler.get_action(env_idx, obs_for_env)

                        # Set context BEFORE stepping so all mh_* module logs are tagged correctly
                        if agent_used >= 0:
                            AgentContext.set_current_agent(agent_used)
                            EnvContext.set_current_env(env_idx)

                        actions.append(action)
                        agents_used.append(agent_used)

                        # Step this single env with the correct agent context active
                        step_result = _env.step(action)

                        # step_result is (obs, reward, terminated, truncated, info) from Gymnasium
                        if len(step_result) == 5:
                            s_obs, s_reward, s_terminated, s_truncated, s_info = step_result
                            s_done = s_terminated or s_truncated
                        else:
                            s_obs, s_reward, s_done, s_info = step_result

                        # Auto-reset when episode ends (mirrors DummyVecEnv behavior)
                        if s_done:
                            try:
                                _agent_for_reset = agent_used if agent_used >= 0 else env_idx
                                AgentContext.set_current_agent(_agent_for_reset)
                                _reset_result = _env.reset()
                                if isinstance(_reset_result, tuple):
                                    s_obs = _reset_result[0]
                                else:
                                    s_obs = _reset_result

                                # Check if reset triggered isolation
                                if getattr(_env, '_isolated', False):
                                    logger.error(
                                        f"Env {env_idx} became ISOLATED after reset. "
                                        f"Decoupling from training — other agents unaffected."
                                    )
                                    new_observations_list.append(s_obs)
                                    rewards_list.append(0.0)
                                    dones_list.append(False)
                                    infos_list.append({'isolated': True})
                                    continue

                                # Verify reset actually worked
                                if hasattr(_env, 'memory') and _env.memory is not None:
                                    _post_map = _env.memory.read_value('CURRENT_MAP')
                                    if _post_map == 45 or _post_map == 0:
                                        logger.warning(
                                            f"Env {env_idx}: reset returned MAP={_post_map} "
                                            f"— forcing extra reload with 3s wait"
                                        )
                                        import time as _time
                                        _time.sleep(3.0)
                                        _reset_result = _env.reset()
                                        if isinstance(_reset_result, tuple):
                                            s_obs = _reset_result[0]
                                        else:
                                            s_obs = _reset_result

                                        # Re-check isolation after second reset
                                        if getattr(_env, '_isolated', False):
                                            logger.error(
                                                f"Env {env_idx} ISOLATED after retry. "
                                                f"Decoupling from training."
                                            )
                                            new_observations_list.append(s_obs)
                                            rewards_list.append(0.0)
                                            dones_list.append(False)
                                            infos_list.append({'isolated': True})
                                            continue

                            except Exception as _reset_err:
                                logger.error(f"Auto-reset failed for env {env_idx}: {_reset_err}")

                        # Append observation regardless of done state.
                        # Isolated envs are already handled above with continue statements
                        # and never reach this point.
                        new_observations_list.append(s_obs)
                        rewards_list.append(s_reward)
                        dones_list.append(s_done)
                        infos_list.append(s_info)

                    # ================================================================
                    # All envs have been stepped — rebuild arrays once with full lists
                    # ================================================================

                    # Rebuild stacked arrays: infos_list now has num_envs entries
                    rewards = np.array(rewards_list, dtype=np.float32)
                    dones = np.array(dones_list, dtype=bool)
                    infos = infos_list

                    # Rebuild new_observations as stacked dict or array
                    if isinstance(new_observations_list[0], dict):
                        new_observations = {
                            key: np.stack([obs[key] for obs in new_observations_list])
                            for key in new_observations_list[0].keys()
                        }
                    else:
                        new_observations = np.stack(new_observations_list)

                        # --- Attempt recovery for isolated envs (lazy fallback) ---
                        _current_time_recovery = time.time()
                        for env_chk in range(self.env.num_envs):
                            _env_chk = self.env.envs[env_chk]
                            if not getattr(_env_chk, '_isolated', False):
                                continue

                            # Only attempt recovery every 5 seconds to avoid spamming
                            _last_attempt = getattr(_env_chk, '_last_recovery_attempt', 0.0)
                            if _current_time_recovery - _last_attempt < 5.0:
                                continue

                            # Check if fallback recovery is initialized
                            if not hasattr(_env_chk, '_fallback_queue'):
                                continue

                            result = _env_chk._attempt_next_fallback()
                            if result == 'recovered':
                                _env_chk._isolated = False
                                self._env_consecutive_quest_end[env_chk] = 0
                                logger.info(
                                    f"Env {env_chk} RECOVERED from isolation! "
                                    f"Resuming training."
                                )
                                # Force a fresh reset to get a clean observation
                                try:
                                    _reset_result = _env_chk.reset()
                                    if isinstance(_reset_result, tuple):
                                        _rec_obs = _reset_result[0]
                                    else:
                                        _rec_obs = _reset_result
                                    # Update the observation for this env in the stacked array
                                    if isinstance(observations, dict):
                                        for k in observations:
                                            observations[k][env_chk] = _rec_obs[k]
                                    else:
                                        observations[env_chk] = _rec_obs
                                except Exception as _rec_err:
                                    logger.error(
                                        f"Env {env_chk} recovery reset failed: {_rec_err}"
                                    )
                            elif result == 'exhausted':
                                logger.error(
                                    f"Env {env_chk}: all fallback recovery options exhausted. "
                                    f"Stays isolated until manual intervention."
                                )

                    # --- Attempt recovery for isolated envs (lazy fallback) ---
                    _current_time_recovery = time.time()
                    for env_chk in range(self.env.num_envs):
                        _env_chk = self.env.envs[env_chk]
                        if not getattr(_env_chk, '_isolated', False):
                            continue

                        # Only attempt recovery every 5 seconds to avoid spamming
                        _last_attempt = getattr(_env_chk, '_last_recovery_attempt', 0.0)
                        if _current_time_recovery - _last_attempt < 5.0:
                            continue

                        # Check if fallback recovery is initialized
                        if not hasattr(_env_chk, '_fallback_queue'):
                            continue

                        result = _env_chk._attempt_next_fallback()
                        if result == 'recovered':
                            _env_chk._isolated = False
                            self._env_consecutive_quest_end[env_chk] = 0
                            logger.info(
                                f"Env {env_chk} RECOVERED from isolation! "
                                f"Resuming training."
                            )
                            # Force a fresh reset to get a clean observation
                            try:
                                _reset_result = _env_chk.reset()
                                if isinstance(_reset_result, tuple):
                                    _rec_obs = _reset_result[0]
                                else:
                                    _rec_obs = _reset_result
                                # Update the observation for this env in the stacked array
                                if isinstance(observations, dict):
                                    for k in observations:
                                        observations[k][env_chk] = _rec_obs[k]
                                else:
                                    observations[env_chk] = _rec_obs
                            except Exception as _rec_err:
                                logger.error(
                                    f"Env {env_chk} recovery reset failed: {_rec_err}"
                                )
                        elif result == 'exhausted':
                            logger.error(
                                f"Env {env_chk}: all fallback recovery options exhausted. "
                                f"Stays isolated until manual intervention."
                            )

                    # --- Check if ALL envs are now isolated → stop training ---
                    active_envs = sum(
                        1 for i in range(self.env.num_envs)
                        if not getattr(self.env.envs[i], '_isolated', False)
                    )
                    if active_envs == 0:
                        logger.error(
                            "ALL environments are ISOLATED — no valid save states "
                            "remain across any user profile. Stopping training."
                        )
                        raise RuntimeError(
                            "Training stopped: all environments have corrupted "
                            "save states. Please re-save valid save states."
                        )

                    # --- Stuck detection: safe now, infos has num_envs entries ---
                    for env_chk in range(self.env.num_envs):
                        # Skip isolated envs for stuck detection
                        if getattr(self.env.envs[env_chk], '_isolated', False):
                            continue

                        # Safe: infos_list is complete (all envs stepped before this point)
                        chk_info = infos[env_chk]
                        if chk_info.get('quest_ended_screen') or chk_info.get('quest_ended_before_action'):
                            self._env_consecutive_quest_end[env_chk] += 1
                            if self._env_consecutive_quest_end[env_chk] >= self._env_stuck_threshold:
                                logger.error(
                                    f"Env {env_chk} stuck outside quest for "
                                    f"{self._env_consecutive_quest_end[env_chk]} steps "
                                    f"— ISOLATING (other agents continue)"
                                )
                                self.env.envs[env_chk]._isolated = True
                                # Initialize fallback recovery for stuck-detected isolation too
                                if not hasattr(self.env.envs[env_chk], '_fallback_queue'):
                                    self.env.envs[env_chk]._init_fallback_recovery()
                        else:
                            self._env_consecutive_quest_end[env_chk] = 0

                    # --- Store transitions (unchanged) ---
                    for env_idx in range(self.env.num_envs):
                        agent_used = agents_used[env_idx]

                        # Skip isolated environments — no data stored
                        if agent_used == -2:
                            continue

                        if agent_used >= 0:
                            # Skip if this agent's buffer is already full — prevents
                            # IndexError when writing at pos == buffer_size
                            if self.agents[agent_used].rollout_buffer.full:
                                continue

                            self._store_transition(
                                agent_id=agent_used,
                                obs=self._get_env_obs(observations, env_idx),
                                action=actions[env_idx],
                                reward=rewards[env_idx],
                                done=dones[env_idx],
                            )
                            steps_collected[agent_used] += 1
                            total_steps_per_agent[agent_used] += 1

                            # Log step data into this agent's training_data.jsonl
                            if (self.agent_callbacks is not None
                                    and agent_used < len(self.agent_callbacks)):
                                cb = self.agent_callbacks[agent_used]
                                if hasattr(cb, 'record_step'):
                                    cb.record_step(
                                        reward=float(rewards[env_idx]),
                                        action=actions[env_idx].tolist(),
                                        # MultiDiscrete: convert array to list of ints for JSON logging
                                        done=bool(dones[env_idx]),
                                        info=infos[env_idx] if infos else {},
                                        env_idx=env_idx,
                                    )

                            # Épisode terminé → mettre à jour le score pour le mode weighted
                            if dones[env_idx] and 'episode' in infos[env_idx]:
                                ep_info = infos[env_idx]['episode']
                                episode_reward = ep_info['r']
                                self.scheduler.update_agent_score(agent_used, episode_reward)
                                episodes_done[agent_used] += 1
                                # Note: log_episode_end is already called inside record_step()
                                # when done=True. Calling it again here would double-count episodes.

                        else:
                            _env_obs_majority = self._get_env_obs(observations, env_idx)
                            for aid in range(len(self.agents)):
                                # Skip agents whose buffer is already full
                                if self.agents[aid].rollout_buffer.full:
                                    continue
                                self._store_transition(
                                    agent_id=aid,
                                    obs=_env_obs_majority,
                                    action=actions[env_idx],
                                    reward=rewards[env_idx],
                                    done=dones[env_idx],
                                )
                                steps_collected[aid] += 1
                                total_steps_per_agent[aid] += 1

                                if dones[env_idx] and 'episode' in infos[env_idx]:
                                    ep_info = infos[env_idx]['episode']
                                    episode_reward = ep_info['r']
                                    self.scheduler.update_agent_score(agent_used, episode_reward)
                                    episodes_done[agent_used] += 1

                                    # Log episode end so EP counter advances in debug.log
                                    if (self.agent_callbacks is not None
                                            and agent_used < len(self.agent_callbacks)):
                                        cb = self.agent_callbacks[agent_used]
                                        if hasattr(cb, 'training_logger'):
                                            cb.training_logger.log_episode_end({
                                                'total_reward': float(ep_info['r']),
                                                'length': int(ep_info['l']),
                                                'hits': int(infos[env_idx].get('hit_count', 0)),
                                                'deaths': int(infos[env_idx].get('death_count', 0)),
                                                'zones_discovered': int(infos[env_idx].get('zones_discovered', 0)),
                                            })

                    observations = new_observations
                    self.total_timesteps += self.env.num_envs

                    # --- Shared SB3 callback (GUI update) ---
                    if self.callback:
                        self.callback.locals = {
                            'infos': infos_list,
                            'rewards': rewards_list,
                            'actions': actions,
                            'dones': dones_list,
                        }
                        self.callback.num_timesteps = self.total_timesteps
                        self.callback.n_calls += 1
                        # Call gui_update() directly — not on_step() which requires
                        # self.model (only set by SB3 learn(), not our custom loop),
                        # and not _on_step() which has the same confusing SB3 naming.
                        if not self.callback.gui_update():
                            logger.warning("Callback requested early stop")
                            return

                    # --- Progress window update (every 10s) ---
                    current_time = time.time()
                    if current_time - last_log_time > 10.0:
                        elapsed = current_time - start_time
                        fps_global = self.total_timesteps / max(elapsed, 1e-9)
                        steps_per_agent_target = total_timesteps // max(len(self.agents), 1)

                        # Count isolated envs for display
                        _isolated_envs = [
                            i for i in range(self.env.num_envs)
                            if getattr(self.env.envs[i], '_isolated', False)
                        ]

                        for _aid in range(len(self.agents)):
                            agent_total = total_steps_per_agent.get(_aid, 0)
                            agent_pct = (agent_total / max(steps_per_agent_target, 1)) * 100
                            _progress_win.update(
                                agent_id=_aid,
                                steps=agent_total,
                                fps=fps_global / max(len(self.agents), 1),
                                episodes=episodes_done.get(_aid, 0),
                                pct=min(agent_pct, 100.0),
                                isolated_envs=_isolated_envs,
                            )

                        last_log_time = current_time

                # ============================================================
                # PHASE 2 : UPDATE
                # Chaque agent lance ses epochs PPO si son buffer est plein
                # ============================================================
                logger.info("Update phase starting...")
                self._release_all_inputs()

                def _update_single_agent(agent_id_inner):
                    """Run PPO update for one agent — thread-safe because each agent
                    has independent parameters, buffer, and optimizer."""
                    agent = self.agents[agent_id_inner]
                    try:
                        buffer_is_full = agent.rollout_buffer.full
                        has_enough_steps = steps_collected[agent_id_inner] >= self.steps_per_agent

                        if not has_enough_steps and not buffer_is_full:
                            logger.debug(
                                f"Agent {agent_id_inner}: skipping update "
                                f"({steps_collected[agent_id_inner]}/{self.steps_per_agent} steps, buffer not full)"
                            )
                            agent.rollout_buffer.reset()
                            return

                        min_fill = agent.rollout_buffer.buffer_size // 2
                        if agent.rollout_buffer.pos < min_fill and not agent.rollout_buffer.full:
                            logger.debug(
                                f"Agent {agent_id_inner}: buffer too sparse "
                                f"({agent.rollout_buffer.pos}/{agent.rollout_buffer.buffer_size})"
                            )
                            agent.rollout_buffer.reset()
                            return

                        if not agent.rollout_buffer.full:
                            agent.rollout_buffer.full = True

                        last_valid_pos = agent.rollout_buffer.pos if agent.rollout_buffer.pos > 0 \
                            else agent.rollout_buffer.buffer_size

                        # --- Bootstrap value (GAE) ---
                        with torch.no_grad():
                            last_pos = (last_valid_pos - 1) % agent.rollout_buffer.buffer_size
                            if isinstance(agent.rollout_buffer.observations, dict):
                                last_obs_torch = {
                                    k: torch.as_tensor(v[last_pos]).to(agent.device)
                                    for k, v in agent.rollout_buffer.observations.items()
                                }
                            else:
                                last_obs_torch = torch.as_tensor(
                                    agent.rollout_buffer.observations[last_pos]
                                ).to(agent.device)
                            last_values = agent.policy.predict_values(last_obs_torch).flatten()

                        done_flag = agent.rollout_buffer.episode_starts[last_valid_pos - 1] \
                            if not agent.rollout_buffer.full \
                            else agent.rollout_buffer.episode_starts[-1]

                        agent.rollout_buffer.compute_returns_and_advantage(
                            last_values=last_values, dones=done_flag)

                        # --- PPO epochs ---
                        for epoch in range(agent.n_epochs):
                            for rollout_data in agent.rollout_buffer.get(agent.batch_size):
                                batch_actions = rollout_data.actions
                                if isinstance(agent.action_space, spaces.Discrete):
                                    batch_actions = batch_actions.long().flatten()
                                elif isinstance(agent.action_space, spaces.MultiDiscrete):
                                    batch_actions = batch_actions.long()

                                values, log_prob, entropy = agent.policy.evaluate_actions(
                                    rollout_data.observations, batch_actions)
                                values = values.flatten()

                                advantages = rollout_data.advantages
                                if agent.normalize_advantage:
                                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                                clip_range = agent.clip_range(1.0) if callable(agent.clip_range) else float(
                                    agent.clip_range)

                                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                                policy_loss = -torch.min(
                                    advantages * ratio,
                                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                                ).mean()

                                if agent.clip_range_vf is not None:
                                    clip_range_vf = agent.clip_range_vf(1.0) if callable(
                                        agent.clip_range_vf) else float(agent.clip_range_vf)
                                    values_pred = rollout_data.old_values + torch.clamp(
                                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                                else:
                                    values_pred = values
                                value_loss = (rollout_data.returns - values_pred).pow(2).mean()

                                entropy_loss = -torch.mean(entropy if entropy is not None else -log_prob)

                                loss = (policy_loss + agent.vf_coef * value_loss + agent.ent_coef * entropy_loss)

                                agent.policy.optimizer.zero_grad()
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
                                agent.policy.optimizer.step()

                        # Update agent step counter for LR schedule
                        agent.num_timesteps += steps_collected.get(agent_id_inner, 0)
                        logger.debug(f"Agent {agent_id_inner} update done")

                    except Exception as update_error:
                        logger.error(f"Agent {agent_id_inner} update failed: {update_error}")
                        traceback.print_exc()
                    finally:
                        agent.rollout_buffer.reset()

                # --- Run all agent updates in parallel ---
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                    futures = [executor.submit(_update_single_agent, aid)
                               for aid in range(len(self.agents))]
                    for f in futures:
                        f.result()  # Wait + propagate exceptions

                # --- Rééquilibrage weighted (si mode actif et assez d'épisodes) ---
                if self.scheduler.mode == 'weighted':
                    rebalanced = self.scheduler.rebalance_weighted_allocation(scenario=self.scenario)
                    if rebalanced:
                        logger.info("Allocation updated based on agent performance")

                _update_cycle += 1
                # Sync every 10 update cycles by default (configurable)
                sync_every = getattr(self, 'weight_sync_every', 10)
                self._sync_weights(sync_every, _update_cycle)

                # Deadlock guard: if ALL agents had 0 steps collected but some buffers are full,
                # collection immediately exited without doing anything → force-reset to break the cycle.
                all_zero = all(steps_collected[aid] == 0 for aid in range(len(self.agents)))
                if all_zero:
                    stuck_buffers = [(aid, self.agents[aid].rollout_buffer.pos)
                                     for aid in range(len(self.agents))
                                     if self.agents[aid].rollout_buffer.full or self.agents[aid].rollout_buffer.pos > 0]
                    if stuck_buffers:
                        logger.warning(
                            f"Deadlock detected: 0 steps collected but {len(stuck_buffers)} buffer(s) "
                            f"non-empty {[(aid, pos) for aid, pos in stuck_buffers]}. "
                            f"Force-resetting buffers to unblock collection."
                        )
                        for aid in range(len(self.agents)):
                            if self.agents[aid].rollout_buffer.full or self.agents[aid].rollout_buffer.pos > 0:
                                self.agents[aid].rollout_buffer.reset()

                # Reset per-cycle counters
                steps_collected = defaultdict(int)
                episodes_done = defaultdict(int)


        except KeyboardInterrupt:
            logger.warning("Training interrupted — saving all agents...")
            self._save_agents(label="interrupted",
                              models_dir=getattr(self, '_models_dir', 'models'))
            _progress_win.close()
            raise

        except Exception as training_loop_error:
            logger.error(f"Training loop error: {training_loop_error}")
            self._save_agents(label="error",
                              models_dir=getattr(self, '_models_dir', 'models'))
            _progress_win.close()
            raise

        # ============================================================
        # FIN DE L'ENTRAÎNEMENT
        # ============================================================
        total_time = time.time() - start_time
        fps_mean = self.total_timesteps / max(total_time, 1e-9)

        logger.info("")
        logger.info("=" * 70)
        logger.info("MULTI-AGENT TRAINING COMPLETE")
        logger.info("=" * 70)
        _progress_win.close()
        logger.info(f"Total timesteps : {self.total_timesteps:,}")
        logger.info(f"Total time      : {total_time:.1f}s")
        logger.info(f"Average FPS     : {fps_mean:.1f}")
        logger.info("Steps per agent :")
        for agent_id, agent in enumerate(self.agents):
            logger.info(f"   Agent {agent_id} : {agent.num_timesteps:,} steps")
        logger.info("=" * 70)
        logger.info("")