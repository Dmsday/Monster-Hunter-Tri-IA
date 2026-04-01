"""
gui_tab_charts.py — Charts tab: episode reward, length, hits per episode.
"""

import numpy as np
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from GUI.gui_theme import C, apply_matplotlib_dark_style
from GUI.gui_state import GuiState


class ChartsTab:
    """Matplotlib-based line/bar charts for training progress."""

    def __init__(self, state: GuiState):
        self.state = state
        self._fig    = None
        self._axes   = []
        self._canvas = None

    def build(self, parent: tk.Frame):
        self._fig = Figure(figsize=(9, 6), dpi=88)
        self._axes = [self._fig.add_subplot(3, 1, i + 1) for i in range(3)]
        apply_matplotlib_dark_style(self._fig, self._axes)
        self._fig.tight_layout(pad=1.5)
        self._canvas = FigureCanvasTkAgg(self._fig, parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def refresh(self):
        if not self._fig or not self._canvas:
            return
        s = self.state
        if len(s.episode_history) < 2:
            return

        eps  = list(s.episode_history)
        rews = list(s.reward_history)
        lens = list(s.length_history)
        hits = list(s.hits_history)

        for ax in self._axes:
            ax.cla()
            ax.set_facecolor(C.CHART_BG)
            ax.grid(True, color=C.CHART_GRID, linewidth=0.4, alpha=0.8)
            ax.tick_params(colors=C.TEXT_DIM, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(C.CHART_AXIS)

        if rews:
            self._axes[0].plot(eps, rews, color=C.ACCENT, linewidth=1.2, alpha=0.7)
            self._axes[0].fill_between(eps, rews, alpha=0.1, color=C.ACCENT)
            if len(rews) >= 10:
                w = min(10, len(rews))
                mv = np.convolve(rews, np.ones(w) / w, mode='valid')
                self._axes[0].plot(eps[w-1:], mv, color=C.GREEN,
                                    linewidth=1.8, linestyle="--")
            self._axes[0].set_title("Episode Reward", color=C.TEXT_DIM, fontsize=9)

        if lens:
            self._axes[1].plot(eps, lens, color=C.BLUE, linewidth=1.2, alpha=0.7)
            self._axes[1].set_ylim(bottom=0)
            self._axes[1].set_title("Episode Length", color=C.TEXT_DIM, fontsize=9)

        if hits:
            self._axes[2].bar(eps, hits, color=C.PURPLE, alpha=0.65, width=0.7)
            self._axes[2].set_ylim(bottom=0)
            self._axes[2].set_title("Hits per Episode", color=C.TEXT_DIM, fontsize=9)

        self._fig.tight_layout(pad=1.5)
        try:
            self._canvas.draw()
        except Exception:
            pass
