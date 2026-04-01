"""
training_gui.py — Monster Hunter Tri · RL Training Dashboard v6.0
Lightweight orchestrator: same public API as v5, delegates UI to sub-modules.

Public API (unchanged):
    gui = TrainingGUI(title="…")
    gui.start()
    gui.update_stats(stats_dict)
    gui.add_episode_data(episode, reward, length, hits, agent_id)
    gui.should_stop() → bool
    gui.set_total_timesteps(n)
    gui.set_disabled_heads(head_indices)
    gui.close()
"""

import time
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use('Agg')

from GUI.gui_theme import C, apply_dark_theme, make_separator
from GUI.gui_map3d import MapPanel
from GUI.gui_state import GuiState, UPDATE_RATE_MS, CHART_RATE_MS
from GUI.gui_header import HeaderBar
from GUI.gui_statusbar import StatusBar
from GUI.gui_tab_overview import OverviewTab
from GUI.gui_tab_charts import ChartsTab
from GUI.gui_tab_rewards import RewardsTab
from GUI.gui_tab_combat import CombatTab
from GUI.gui_tab_actions import ActionsTab
from info.module_logger import get_module_logger

logger = get_module_logger('training_gui')


class TrainingGUI:
    """Training dashboard — single window, delegates to tab modules."""

    def __init__(self, title: str = "Monster Hunter IA — Training"):
        self.title  = title
        self.window: Optional[tk.Tk] = None

        # Shared state (no tkinter dependency)
        self._state = GuiState()

        # UI components (created in _build_ui)
        self._header:       Optional[HeaderBar]   = None
        self._statusbar:    Optional[StatusBar]    = None
        self._tab_overview: Optional[OverviewTab]  = None
        self._tab_charts:   Optional[ChartsTab]    = None
        self._tab_rewards:  Optional[RewardsTab]   = None
        self._tab_combat:   Optional[CombatTab]    = None
        self._tab_actions:  Optional[ActionsTab]   = None
        self._map_panel:    Optional[MapPanel]     = None
        self._notebook:     Optional[ttk.Notebook] = None

        self._update_thread: Optional[threading.Thread] = None

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API  (identical signatures to v5)
    # ══════════════════════════════════════════════════════════════

    @property
    def running(self) -> bool:
        return self._state.running

    @running.setter
    def running(self, v: bool):
        self._state.running = v

    @property
    def stop_requested(self) -> bool:
        return self._state.stop_requested

    @stop_requested.setter
    def stop_requested(self, v: bool):
        self._state.stop_requested = v

    @property
    def stop_button(self):
        return self._header.stop_button if self._header else None

    @stop_button.setter
    def stop_button(self, _v):
        pass  # legacy setter — button is owned by HeaderBar now

    # Expose public deques for callbacks (backward compat)
    @property
    def episode_history(self):
        return self._state.episode_history

    @property
    def reward_history(self):
        return self._state.reward_history

    @property
    def length_history(self):
        return self._state.length_history

    @property
    def hits_history(self):
        return self._state.hits_history

    @property
    def reward_breakdown_history(self):
        return self._state.reward_breakdown_history

    @property
    def reward_breakdown_detailed_history(self):
        return self._state.reward_breakdown_detailed_history

    # Expose isolated_envs for callbacks
    @property
    def _isolated_envs(self) -> List[int]:
        return self._state.isolated_envs

    @_isolated_envs.setter
    def _isolated_envs(self, v: List[int]):
        self._state.isolated_envs = v

    def start(self):
        if self._state.running:
            return
        self._state.running = True
        self._state.stop_requested = False
        self._state.start_time = time.time()
        self._update_thread = threading.Thread(
            target=self._run_gui, daemon=True, name="TrainingGUI")
        self._update_thread.start()

    def set_total_timesteps(self, n: int):
        self._state.total_timesteps = n
        self._state.training_start_time = time.time()

    def set_disabled_heads(self, head_indices: set):
        self._state.disabled_heads = set(head_indices) if head_indices else set()
        logger.info(f"GUI: disabled heads set to {self._state.disabled_heads}")

    def update_stats(self, stats: Dict[str, Any]):
        self._state.update_stats(stats)

    def add_episode_data(self, episode: int, reward: float,
                         length: int, hits: int = 0,
                         agent_id: int = 0, instance_id: int = 0):
        self._state.add_episode_data(episode, reward, length, hits,
                                      agent_id, instance_id)

    def should_stop(self) -> bool:
        return self._state.stop_requested

    def close(self):
        s = self._state
        if s._closing:
            return
        s._closing = True
        s.running = False
        if self.window:
            try: s.save_config(self.window.geometry())
            except Exception: pass

        if self._map_panel:
            try: self._map_panel.stop()
            except Exception: pass

        if self.window:
            try:
                if self.window.winfo_exists():
                    self.window.after(0, self._destroy_window)
            except Exception:
                self.window = None

        current = threading.current_thread()
        if (self._update_thread is not None
                and self._update_thread.is_alive()
                and current is not self._update_thread):
            self._update_thread.join(timeout=2.0)

    def wait_until_closed(self):
        while self._state.running:
            time.sleep(0.1)

    # ══════════════════════════════════════════════════════════════
    #  GUI RUN LOOP
    # ══════════════════════════════════════════════════════════════

    def _run_gui(self):
        try:
            self.window = tk.Tk()
            self.window.title(self.title)
            self.window.configure(bg=C.BG)
            self.window.geometry(
                self._state.config.get('geometry', '1280x800+60+60'))
            self.window.minsize(900, 580)
            apply_dark_theme(self.window)
            self.window.protocol("WM_DELETE_WINDOW", self._on_close)
            self._build_ui()
            self._schedule_update()
            self.window.mainloop()
        except Exception as e:
            logger.error(f"GUI thread crashed: {e}")
        finally:
            self._state.running = False

    def _on_close(self):
        s = self._state
        if not s._closing:
            s.stop_requested = True
            s._closing = True
            s.running = False
            if self.window:
                try: s.save_config(self.window.geometry())
                except Exception: pass
            if self._map_panel:
                try: self._map_panel.stop()
                except Exception: pass
            try:
                if self.window and self.window.winfo_exists():
                    self.window.quit()
                    self.window.destroy()
            except Exception:
                pass
            self.window = None

    def _on_stop_click(self):
        """Release controller inputs then signal stop via should_stop() flag."""
        s = self._state
        if s.stop_requested:
            return
        s.stop_requested = True
        logger.warning("Stop requested via GUI button")

        # Release all controller inputs before stopping
        env_ref = getattr(self, '_env_ref', None)
        if env_ref is not None:
            try:
                if hasattr(env_ref, 'envs'):
                    for _e in env_ref.envs:
                        _ctrl = getattr(_e, 'controller', None)
                        if _ctrl and hasattr(_ctrl, 'reset_all'):
                            _ctrl.reset_all()
                elif hasattr(env_ref, 'controller'):
                    _ctrl = getattr(env_ref, 'controller', None)
                    if _ctrl and hasattr(_ctrl, 'reset_all'):
                        _ctrl.reset_all()
                logger.debug("All controller inputs released before stop")
            except Exception as _re:
                logger.debug(f"Could not release inputs on stop: {_re}")

        if self._header and self._header.stop_button:
            try:
                self._header.stop_button.configure(
                    text="⏳  STOPPING…", state="disabled",
                    bg="#2a1a00", fg=C.ORANGE)
            except Exception:
                pass

        if self._statusbar:
            self._statusbar.set_status(
                "Stop requested — releasing inputs and saving…", C.YELLOW)

        # Close Dolphin instances via stored PID ref
        dolphin_pids = getattr(self, '_dolphin_pids_ref', [])
        if dolphin_pids:
            try:
                import psutil
                for _pid in list(dolphin_pids):
                    if _pid is None or _pid <= 0:
                        continue
                    try:
                        if psutil.pid_exists(_pid):
                            psutil.Process(_pid).terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                logger.debug(f"GUI stop: terminated {len(dolphin_pids)} Dolphin(s)")
            except Exception as _dol_err:
                logger.debug(f"GUI stop: Dolphin cleanup error: {_dol_err}")

        logger.debug("GUI stop: training will stop cleanly via should_stop()")

    def _destroy_window(self):
        try:
            import matplotlib.pyplot as _plt
            _plt.close('all')
        except Exception:
            pass
        try:
            if self.window and self.window.winfo_exists():
                self.window.quit()
                self.window.destroy()
        except Exception:
            pass
        finally:
            self.window = None

    # ══════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        s = self._state

        # Header
        self._header = HeaderBar(self.window, s, self._on_stop_click)
        self._header.build()

        # Notebook (tabs)
        self._notebook = ttk.Notebook(self.window, style="Dark.TNotebook")
        self._notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self._tab_overview = OverviewTab(s)
        self._tab_charts   = ChartsTab(s)
        self._tab_rewards  = RewardsTab(s)
        self._tab_combat   = CombatTab(s)
        self._tab_actions  = ActionsTab(s)

        tabs = [
            ("  Overview  ", self._tab_overview),
            ("  Charts  ",   self._tab_charts),
            ("  Map  ",      None),              # special: MapPanel
            ("  Rewards  ",  self._tab_rewards),
            ("  Combat  ",   self._tab_combat),
            ("  Actions  ",  self._tab_actions),
        ]
        for label, tab_obj in tabs:
            frame = tk.Frame(self._notebook, bg=C.BG)
            self._notebook.add(frame, text=label)
            if tab_obj is not None:
                tab_obj.build(frame)
            else:
                # Map tab
                self._map_panel = MapPanel(frame)
                self._map_panel.pack(fill="both", expand=True)

        # Status bar
        self._statusbar = StatusBar(self.window, s)
        self._statusbar.build()

    # ══════════════════════════════════════════════════════════════
    #  REFRESH LOOP
    # ══════════════════════════════════════════════════════════════

    def _schedule_update(self):
        s = self._state
        if s._closing or not s.running:
            return
        if not self.window or not self.window.winfo_exists():
            s.running = False
            return
        try:
            self._refresh_ui()
        except tk.TclError:
            s.running = False
            return
        except Exception as e:
            logger.debug(f"GUI refresh error (non-fatal): {e}")
        finally:
            if s.running and not s._closing:
                try:
                    self.window.after(UPDATE_RATE_MS, self._schedule_update)
                except Exception:
                    s.running = False

    def _refresh_ui(self):
        s = self._state
        if s._closing:
            return

        stats = s.current_stats()
        total = int(stats.get('total_steps', 0) or 0)
        now   = time.time()

        # Header refresh (KPIs, progress, combos)
        if self._header:
            self._header.refresh(stats)

        # Status bar (FPS, isolation)
        if self._statusbar:
            self._statusbar.refresh(total)

        # Overview tab — always refresh (lightweight)
        if self._tab_overview:
            self._tab_overview.refresh(stats)

        # Map panel
        if self._map_panel:
            self._map_panel.update_data(stats)

        # Chart-throttled updates (only on visible tab)
        do_charts = (now - s.last_chart_time) >= (CHART_RATE_MS / 1000.0)
        if do_charts:
            s.last_chart_time = now

        if self._notebook:
            try:
                tab = self._notebook.index(self._notebook.select())
            except Exception:
                tab = 0
            # Tab indices: 0=Overview 1=Charts 2=Map 3=Rewards 4=Combat 5=Actions
            if tab == 1 and do_charts and self._tab_charts:
                self._tab_charts.refresh()
            elif tab == 3 and do_charts and self._tab_rewards:
                self._tab_rewards.refresh(stats)
            elif tab == 5 and self._tab_actions:
                self._tab_actions.refresh(stats)

        # Combat tab: always update (lightweight, no matplotlib)
        if self._tab_combat:
            self._tab_combat.refresh(stats)

    def _deferred_chart_redraw(self, delay_ms: int = 150):
        self._state.last_chart_time = 0.0
        if self.window and self.window.winfo_exists():
            try:
                self.window.after(delay_ms, self._force_chart_refresh)
            except Exception:
                pass

    def _force_chart_refresh(self):
        s = self._state
        if s._closing or not s.running:
            return
        try:
            stats = s.current_stats()
            if self._tab_charts:
                self._tab_charts.refresh()
            if self._tab_rewards:
                self._tab_rewards.refresh(stats)
        except Exception:
            pass