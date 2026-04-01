"""
gui_tab_rewards.py — Rewards tab: per-category bar chart with peak-hold,
top gains / losses side panel.
"""

import time
import tkinter as tk
from typing import Dict, Optional

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from GUI.gui_theme import C, apply_matplotlib_dark_style, make_separator
from GUI.gui_state import GuiState, ALL_REWARD_CATS


class RewardsTab:
    """Fixed-axis reward breakdown chart with peak-hold cache."""

    def __init__(self, state: GuiState):
        self.state = state
        self._fig:            Optional[Figure]           = None
        self._ax              = None
        self._canvas          = None
        self._ep_total_lbl:   Optional[tk.Label]         = None
        self._top_gains_lbl:  Optional[tk.Label]         = None
        self._top_losses_lbl: Optional[tk.Label]         = None

    def build(self, parent: tk.Frame):
        # Episode total banner
        top = tk.Frame(parent, bg=C.SURFACE,
                       highlightbackground=C.BORDER, highlightthickness=1)
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="EPISODE TOTAL", font=("Segoe UI", 9, "bold"),
                 bg=C.SURFACE, fg=C.TEXT_DIMMER).pack(side="left", padx=10, pady=6)
        self._ep_total_lbl = tk.Label(top, text="+0.000",
                                       font=("Consolas", 18, "bold"),
                                       bg=C.SURFACE, fg=C.GREEN)
        self._ep_total_lbl.pack(side="right", padx=10)

        body = tk.Frame(parent, bg=C.BG)
        body.pack(fill="both", expand=True, padx=8)

        # Matplotlib chart
        self._fig = Figure(figsize=(7, 5), dpi=88)
        self._ax  = self._fig.add_subplot(111)
        apply_matplotlib_dark_style(self._fig, [self._ax])
        self._canvas = FigureCanvasTkAgg(self._fig, body)
        self._canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Side panel
        side = tk.Frame(body, bg=C.SURFACE,
                         highlightbackground=C.BORDER, highlightthickness=1, width=220)
        side.pack(side="right", fill="y", padx=(6, 0))
        side.pack_propagate(False)
        tk.Label(side, text="TOP GAINS", font=("Segoe UI", 9, "bold"),
                 bg=C.GREEN_BG, fg=C.GREEN).pack(fill="x", pady=2)
        self._top_gains_lbl = tk.Label(side, text="", font=("Consolas", 9),
                                        bg=C.SURFACE, fg=C.GREEN, justify="left")
        self._top_gains_lbl.pack(fill="x", padx=6, pady=4)
        make_separator(side, bg=C.BORDER).pack(fill="x", pady=2)
        tk.Label(side, text="TOP LOSSES", font=("Segoe UI", 9, "bold"),
                 bg=C.RED_BG, fg=C.RED).pack(fill="x", pady=2)
        self._top_losses_lbl = tk.Label(side, text="", font=("Consolas", 9),
                                         bg=C.SURFACE, fg=C.RED, justify="left")
        self._top_losses_lbl.pack(fill="x", padx=6, pady=4)

    def refresh(self, stats: dict):
        if not self._ep_total_lbl:
            return

        ep_total = float(stats.get('episode_reward', 0.0) or 0.0)
        try:
            self._ep_total_lbl.configure(
                text=f"{ep_total:+.3f}",
                fg=C.GREEN if ep_total >= 0 else C.RED)
        except Exception:
            pass

        if not self._ax or not self._canvas:
            return

        breakdown = stats.get('reward_breakdown', {}) or {}
        s = self.state
        now = time.time()

        # Peak-hold: retain non-zero values for a short period
        for cat in ALL_REWARD_CATS:
            v = float(breakdown.get(cat, 0.0))
            if abs(v) > 1e-5:
                s.rew_cache[cat] = v
                s.rew_cache_ts[cat] = now
            elif cat in s.rew_cache:
                age = now - s.rew_cache_ts.get(cat, 0)
                if age < s.rew_cache_ttl:
                    breakdown = dict(breakdown)
                    breakdown[cat] = s.rew_cache[cat]
                else:
                    s.rew_cache.pop(cat, None)
                    s.rew_cache_ts.pop(cat, None)

        vals = [float(breakdown.get(cat, 0.0)) for cat in ALL_REWARD_CATS]
        colors = [
            C.GREEN if v >  1e-5 else
            C.RED   if v < -1e-5 else
            C.SURFACE3
            for v in vals
        ]

        self._ax.cla()
        self._ax.set_facecolor(C.CHART_BG)
        for sp in self._ax.spines.values():
            sp.set_edgecolor(C.CHART_AXIS)
        self._ax.tick_params(colors=C.TEXT_DIM, labelsize=8)
        self._ax.grid(True, axis='x', color=C.CHART_GRID, linewidth=0.4, alpha=0.6)

        y_pos = range(len(ALL_REWARD_CATS))
        self._ax.barh(y_pos, vals, color=colors, alpha=0.85, height=0.55)
        self._ax.set_yticks(y_pos)
        self._ax.set_yticklabels(ALL_REWARD_CATS, fontsize=8, color=C.TEXT_DIM)
        self._ax.axvline(0, color=C.BORDER_LT, linewidth=0.9)

        max_abs = max((abs(v) for v in vals), default=0.0)
        if max_abs < 1e-5:
            max_abs = 1.0
        self._ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
        self._ax.set_title("Reward per step — all categories (fixed axis)",
                           fontsize=9, color=C.TEXT_DIM)
        self._fig.tight_layout(pad=1.0)
        try:
            self._canvas.draw()
        except Exception:
            pass

        # Side panel top gains / losses
        paired = list(zip(ALL_REWARD_CATS, vals))
        gains  = [(k, v) for k, v in paired if v >  1e-5][:5]
        losses = sorted([(k, v) for k, v in paired if v < -1e-5],
                        key=lambda x: x[1])[:5]
        self._cfg(self._top_gains_lbl,
                  text="\n".join(f"{k:<20} {v:+.4f}" for k, v in gains) or "—")
        self._cfg(self._top_losses_lbl,
                  text="\n".join(f"{k:<20} {v:+.4f}" for k, v in losses) or "—")

    @staticmethod
    def _cfg(widget, **kwargs):
        if widget:
            try: widget.configure(**kwargs)
            except Exception: pass
