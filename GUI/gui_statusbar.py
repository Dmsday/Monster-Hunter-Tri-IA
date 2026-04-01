"""
gui_statusbar.py — Bottom status bar: status text, FPS, isolation warnings.
"""

import time
import tkinter as tk
from typing import Optional

from GUI.gui_theme import C, make_separator
from GUI.gui_state import GuiState


class StatusBar:
    """Builds and refreshes the bottom status bar."""

    def __init__(self, parent: tk.Tk, state: GuiState):
        self.parent = parent
        self.state  = state
        self._status_lbl:    Optional[tk.Label] = None
        self._fps_lbl:       Optional[tk.Label] = None
        self._isolation_lbl: Optional[tk.Label] = None

    def build(self):
        make_separator(self.parent).pack(fill="x")
        bar = tk.Frame(self.parent, bg=C.SURFACE, height=22)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        self._status_lbl = tk.Label(bar, text="READY", font=("Consolas", 9),
                                     bg=C.SURFACE, fg=C.TEXT_DIM, anchor="w")
        self._status_lbl.pack(side="left", padx=8)
        self._isolation_lbl = tk.Label(bar, text="", font=("Consolas", 9, "bold"),
                                        bg=C.SURFACE, fg="#fb923c")
        self._isolation_lbl.pack(side="right", padx=8)
        self._fps_lbl = tk.Label(bar, text="", font=("Consolas", 9),
                                  bg=C.SURFACE, fg=C.TEXT_DIMMER)
        self._fps_lbl.pack(side="right", padx=8)

    def refresh(self, total_steps: int):
        """Called every UPDATE_RATE_MS."""
        s = self.state
        now = time.time()

        # FPS calculation
        if s.last_step_count == 0 and total_steps > 0:
            s.last_step_count = total_steps
            s.last_step_time  = now
        dt = now - s.last_step_time
        if dt >= 3.0:
            fps = (total_steps - s.last_step_count) / max(dt, 1e-9)
            self._cfg(self._fps_lbl, text=f"~{fps:.0f} steps/s")
            s.last_step_time  = now
            s.last_step_count = total_steps

        # Isolation warning
        try:
            if self._isolation_lbl and s.isolated_envs:
                self._isolation_lbl.configure(
                    text=f"⚠ {len(s.isolated_envs)} ENV(S) ISOLATED: {s.isolated_envs}")
            elif self._isolation_lbl:
                self._isolation_lbl.configure(text="")
        except Exception:
            pass

    def set_status(self, text: str, color: str = C.TEXT_DIM):
        self._cfg(self._status_lbl, text=text, fg=color)

    @staticmethod
    def _cfg(widget, **kwargs):
        if widget:
            try: widget.configure(**kwargs)
            except Exception: pass
