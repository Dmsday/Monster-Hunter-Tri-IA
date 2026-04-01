"""
gui_header.py — Top header bar: agent/instance selectors, live KPIs,
progress bar, badges, stop button.
"""

import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

from GUI.gui_theme import C, StatusBadge, make_separator
from GUI.gui_state import GuiState
from info.module_logger import get_module_logger

logger = get_module_logger('gui_header')


class HeaderBar:
    """Builds and refreshes the top header of the training dashboard."""

    def __init__(self, parent: tk.Tk, state: GuiState, on_stop_click):
        self.parent = parent
        self.state  = state
        self._on_stop_click = on_stop_click

        # Widget refs
        self._agent_combo:    Optional[ttk.Combobox] = None
        self._instance_combo: Optional[ttk.Combobox] = None
        self._agent_var:      Optional[tk.IntVar]    = None
        self._instance_var:   Optional[tk.IntVar]    = None
        self._ep_lbl:         Optional[tk.Label]     = None
        self._rew_lbl:        Optional[tk.Label]     = None
        self._steps_lbl:      Optional[tk.Label]     = None
        self._elapsed_lbl:    Optional[tk.Label]     = None
        self._prog_bar:       Optional[ttk.Progressbar] = None
        self._prog_lbl:       Optional[tk.Label]     = None
        self._eta_lbl:        Optional[tk.Label]     = None
        self._badge_combat:   Optional[StatusBadge]  = None
        self._badge_menu:     Optional[StatusBadge]  = None
        self.stop_button:     Optional[tk.Button]    = None

    def build(self):
        """Create header widgets. Call once after Tk root exists."""
        hdr = tk.Frame(self.parent, bg=C.SURFACE, pady=6)
        hdr.pack(fill="x")

        # ── Left: title + selectors ──────────────────────────────
        left = tk.Frame(hdr, bg=C.SURFACE)
        left.pack(side="left", padx=10)

        tk.Label(left, text="◈ MH TRI  RL", font=("Consolas", 12, "bold"),
                 bg=C.SURFACE, fg=C.ACCENT).pack(side="left", padx=(0, 16))

        tk.Label(left, text="AGENT", font=("Segoe UI", 9),
                 bg=C.SURFACE, fg=C.TEXT_DIM).pack(side="left")
        self._agent_var = tk.IntVar(value=0)
        self._agent_combo = ttk.Combobox(
            left, textvariable=self._agent_var,
            values=["0"], width=4, state="readonly", font=("Consolas", 10))
        self._agent_combo.pack(side="left", padx=(2, 10))
        self._agent_combo.bind("<<ComboboxSelected>>", self._on_agent_changed)

        tk.Label(left, text="INSTANCE", font=("Segoe UI", 9),
                 bg=C.SURFACE, fg=C.TEXT_DIM).pack(side="left")
        self._instance_var = tk.IntVar(value=0)
        self._instance_combo = ttk.Combobox(
            left, textvariable=self._instance_var,
            values=["0"], width=4, state="readonly", font=("Consolas", 10))
        self._instance_combo.pack(side="left", padx=(2, 0))
        self._instance_combo.bind("<<ComboboxSelected>>", self._on_instance_changed)

        # ── Centre: live KPIs ────────────────────────────────────
        centre = tk.Frame(hdr, bg=C.SURFACE)
        centre.pack(side="left", expand=True, fill="x", padx=20)

        self._ep_lbl    = self._make_metric(centre, "EPISODE",     C.BLUE)
        self._rew_lbl   = self._make_metric(centre, "EP REWARD",   C.GREEN)
        self._steps_lbl = self._make_metric(centre, "TOTAL STEPS", C.ACCENT)

        # Progress bar
        prog_f = tk.Frame(centre, bg=C.SURFACE)
        prog_f.pack(side="left", padx=16)
        self._prog_lbl = tk.Label(prog_f, text="0 / —  (0%)",
                                   font=("Consolas", 8), bg=C.SURFACE, fg=C.TEXT_DIMMER)
        self._prog_lbl.pack(anchor="w")
        self._prog_bar = ttk.Progressbar(
            prog_f, style="Accent.Horizontal.TProgressbar",
            length=180, maximum=100, mode="determinate")
        self._prog_bar.pack()
        self._eta_lbl = tk.Label(prog_f, text="ETA —",
                                  font=("Consolas", 8), bg=C.SURFACE, fg=C.TEXT_DIMMER)
        self._eta_lbl.pack(anchor="w")

        # ── Right: badges + elapsed + stop ───────────────────────
        right = tk.Frame(hdr, bg=C.SURFACE)
        right.pack(side="right", padx=10)

        self._elapsed_lbl = tk.Label(right, text="00:00:00",
                                      font=("Consolas", 11, "bold"),
                                      bg=C.SURFACE, fg=C.TEXT_DIM)
        self._elapsed_lbl.pack(side="right", padx=(8, 0))
        tk.Label(right, text="elapsed", font=("Segoe UI", 8),
                 bg=C.SURFACE, fg=C.TEXT_DIMMER).pack(side="right")

        self.stop_button = tk.Button(
            right, text="⏹  STOP",
            font=("Segoe UI", 10, "bold"),
            bg=C.RED_BG, fg=C.RED,
            activebackground="#4a0a0a", activeforeground=C.RED,
            relief="flat", cursor="hand2", padx=10, pady=4,
            command=self._on_stop_click)
        self.stop_button.pack(side="right", padx=(0, 12))

        badges = tk.Frame(right, bg=C.SURFACE)
        badges.pack(side="right", padx=6)
        self._badge_combat = StatusBadge(badges, "COMBAT",  False)
        self._badge_combat.pack(side="left", padx=2)
        self._badge_menu   = StatusBadge(badges, "IN MENU", False)
        self._badge_menu.pack(side="left", padx=2)

        make_separator(self.parent).pack(fill="x")

    # ── Refresh ──────────────────────────────────────────────────

    def refresh(self, stats: dict):
        """Called every UPDATE_RATE_MS from the main refresh loop."""
        s = self.state
        ep       = int(stats.get('episode', 0) or 0)
        ep_rew   = float(stats.get('episode_reward', 0.0) or 0.0)
        total    = int(stats.get('total_steps', 0) or 0)
        in_combat = bool(stats.get('in_combat', False))
        in_menu   = bool(stats.get('in_game_menu', False))

        elapsed = time.time() - s.start_time
        eh, rem = divmod(int(elapsed), 3600)
        em, es  = divmod(rem, 60)

        self._cfg(self._elapsed_lbl, text=f"{eh:02d}:{em:02d}:{es:02d}")
        self._cfg(self._ep_lbl,      text=str(ep))
        self._cfg(self._rew_lbl,     text=f"{ep_rew:+.2f}",
                  fg=C.GREEN if ep_rew >= 0 else C.RED)
        self._cfg(self._steps_lbl,   text=f"{total:,}")

        if self._badge_combat:
            try: self._badge_combat.set_state("COMBAT",  in_combat)
            except Exception: pass
        if self._badge_menu:
            try: self._badge_menu.set_state("IN MENU", in_menu)
            except Exception: pass

        # ── Progress bar ─────────────────────────────────────────
        if s.total_timesteps > 0 and self._prog_bar:
            pct = min(100.0, (s.global_total_steps / s.total_timesteps) * 100)
            try:
                self._prog_bar['value'] = pct
                self._cfg(self._prog_lbl,
                          text=f"{s.global_total_steps:,} / {s.total_timesteps:,}  ({pct:.1f}%)")
                if elapsed > 1 and pct > 0:
                    eta_s = (elapsed / pct) * (100 - pct)
                    rh, rr = divmod(int(eta_s), 3600)
                    rm, rs = divmod(rr, 60)
                    self._cfg(self._eta_lbl, text=f"ETA {rh}:{rm:02d}:{rs:02d}")
            except Exception:
                pass

        # ── Combo list sync ──────────────────────────────────────
        self._sync_combos()

    # ── Internal ─────────────────────────────────────────────────

    def _sync_combos(self):
        s = self.state
        if self._agent_combo:
            try:
                new = [str(a) for a in s._known_agents]
                if list(self._agent_combo['values']) != new:
                    self._agent_combo['values'] = new
            except Exception:
                pass
        if self._instance_combo:
            try:
                insts = s._known_instances.get(s.sel_agent, [0])
                new = [str(i) for i in insts]
                if list(self._instance_combo['values']) != new:
                    self._instance_combo['values'] = new
            except Exception:
                pass

    def _on_agent_changed(self, _event=None):
        try:
            self.state.sel_agent = int(self._agent_var.get())
        except Exception:
            pass
        instances = self.state._known_instances.get(self.state.sel_agent, [0])
        self._instance_combo['values'] = [str(i) for i in instances]
        self.state.sel_instance = instances[0]
        self._instance_var.set(self.state.sel_instance)
        self.state.last_chart_time = 0.0   # force chart redraw

    def _on_instance_changed(self, _event=None):
        try:
            self.state.sel_instance = int(self._instance_var.get())
        except Exception:
            pass
        self.state.last_chart_time = 0.0

    @staticmethod
    def _make_metric(parent, label, color):
        f = tk.Frame(parent, bg=C.SURFACE)
        f.pack(side="left", padx=12)
        tk.Label(f, text=label, font=("Segoe UI", 8),
                 bg=C.SURFACE, fg=C.TEXT_DIMMER).pack()
        lbl = tk.Label(f, text="—", font=("Consolas", 13, "bold"),
                       bg=C.SURFACE, fg=color)
        lbl.pack()
        return lbl

    @staticmethod
    def _cfg(widget, **kwargs):
        if widget:
            try: widget.configure(**kwargs)
            except Exception: pass
