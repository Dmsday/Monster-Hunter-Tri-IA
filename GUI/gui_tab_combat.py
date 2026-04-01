"""
gui_tab_combat.py — Combat tab: zone info, combat state cards, monster HP bars.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional

from GUI.gui_theme import C, MetricCard, SectionHeader, make_separator
from GUI.gui_state import GuiState


class CombatTab:
    """Combat state display: zone, combat booleans, small/large monster HP."""

    def __init__(self, state: GuiState):
        self.state = state
        self._cbt_cards:    Dict[str, MetricCard] = {}
        self._monster_bars: Dict[str, Dict]       = {}
        self._zone_lbl:     Optional[tk.Label]    = None
        self._lm_bar        = None
        self._lm_lbl:       Optional[tk.Label]    = None

    def build(self, parent: tk.Frame):
        left = tk.Frame(parent, bg=C.BG, width=280)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)
        right = tk.Frame(parent, bg=C.BG)
        right.pack(side="left", fill="both", expand=True, pady=8)

        # Zone header
        zone_hdr = tk.Frame(left, bg=C.ACCENT_BG,
                             highlightbackground=C.ACCENT, highlightthickness=1)
        zone_hdr.pack(fill="x", pady=2)
        tk.Label(zone_hdr, text="CURRENT ZONE", font=("Segoe UI", 9),
                 bg=C.ACCENT_BG, fg=C.ACCENT).pack(side="left", padx=8, pady=5)
        self._zone_lbl = tk.Label(zone_hdr, text="—",
                                   font=("Consolas", 20, "bold"),
                                   bg=C.ACCENT_BG, fg=C.TEXT)
        self._zone_lbl.pack(side="right", padx=8)

        # Combat state cards
        SectionHeader(left, "Combat State", C.RED).pack(fill="x", pady=(8, 4))
        for key, label, color in [
            ("in_combat",              "In Combat",        C.RED),
            ("in_monster_zone",        "In Monster Zone",  C.YELLOW),
            ("monsters_present",       "Monsters Present", C.ORANGE),
            ("monster_count",          "Monster Count",    C.ORANGE),
            ("zone_reward_total",      "Zone Reward",      C.GREEN),
            ("left_monster_zone_count","Left Zone Count",  C.TEXT_DIM),
        ]:
            c = MetricCard(left, label, color=color)
            c.pack(fill="x", pady=2)
            self._cbt_cards[key] = c

        # Small monster HP bars
        SectionHeader(right, "Small Monsters HP", C.RED).pack(fill="x", pady=(0, 6))
        for i in range(1, 6):
            k = f'smonster{i}_hp'
            row = tk.Frame(right, bg=C.BG)
            row.pack(fill="x", pady=3, padx=8)
            tk.Label(row, text=f"M{i}", font=("Consolas", 9, "bold"),
                     bg=C.BG, fg=C.RED, width=3).pack(side="left")
            bar = ttk.Progressbar(row, style="Combat.Horizontal.TProgressbar",
                                   maximum=10000, mode="determinate")
            bar.pack(side="left", fill="x", expand=True, padx=4)
            lbl = tk.Label(row, text="—", font=("Consolas", 9),
                           bg=C.BG, fg=C.TEXT_DIM, width=8)
            lbl.pack(side="right")
            self._monster_bars[k] = {'bar': bar, 'label': lbl}

        # Large monster
        make_separator(right, bg=C.BORDER).pack(fill="x", pady=6, padx=8)
        SectionHeader(right, "Large Monster", C.MAGENTA).pack(fill="x", pady=(0, 6))
        lm_row = tk.Frame(right, bg=C.BG)
        lm_row.pack(fill="x", pady=3, padx=8)
        tk.Label(lm_row, text="LM1", font=("Consolas", 9, "bold"),
                 bg=C.BG, fg=C.MAGENTA, width=4).pack(side="left")
        self._lm_bar = ttk.Progressbar(lm_row, style="Combat.Horizontal.TProgressbar",
                                        maximum=50000, mode="determinate")
        self._lm_bar.pack(side="left", fill="x", expand=True, padx=4)
        self._lm_lbl = tk.Label(lm_row, text="—", font=("Consolas", 9),
                                  bg=C.BG, fg=C.TEXT_DIM, width=8)
        self._lm_lbl.pack(side="right")

    def refresh(self, stats: dict):
        zone = stats.get('zone', 0) or 0
        self._cfg(self._zone_lbl, text=str(zone))

        bool_keys = {'in_monster_zone', 'in_combat', 'monsters_present'}
        for key, card in self._cbt_cards.items():
            v = stats.get(key, 0)
            try:
                if key in bool_keys:
                    card.update("YES" if v else "NO",
                                color=C.GREEN if v else C.TEXT_DIM)
                elif key == 'zone_reward_total':
                    card.update(f"{float(v or 0):+.3f}",
                                color=C.GREEN if float(v or 0) >= 0 else C.RED)
                else:
                    card.update(str(v if v is not None else "—"))
            except Exception:
                pass

        for key, w in self._monster_bars.items():
            hp = int(stats.get(key, 0) or 0)
            try:
                w['bar']['value'] = hp
                w['label'].configure(text=f"{hp}" if hp > 0 else "—",
                                     fg=C.RED if hp > 0 else C.TEXT_DIMMER)
            except Exception:
                pass

        lm_hp = int(stats.get('lmonster1_hp', 0) or 0)
        if self._lm_bar:
            try: self._lm_bar['value'] = lm_hp
            except Exception: pass
        self._cfg(self._lm_lbl,
                  text=f"{lm_hp}" if lm_hp > 0 else "—",
                  fg=C.MAGENTA if lm_hp > 0 else C.TEXT_DIMMER)

    @staticmethod
    def _cfg(widget, **kwargs):
        if widget:
            try: widget.configure(**kwargs)
            except Exception: pass
