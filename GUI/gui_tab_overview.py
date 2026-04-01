"""
gui_tab_overview.py — Overview tab: health/quest, training/exploration, position/inventory.
"""

import tkinter as tk
from typing import Dict

from GUI.gui_theme import C, MetricCard, SectionHeader, ScrollableFrame, make_separator
from GUI.gui_state import GuiState


class OverviewTab:
    """Three-column merged tab: health/quest, training/exploration, position/inventory."""

    def __init__(self, state: GuiState):
        self.state = state
        self._stat_cards: Dict[str, MetricCard] = {}
        self._ply_cards:  Dict[str, MetricCard] = {}
        self._inv_slots:  list = []
        self._inv_scroll  = None

    def build(self, parent: tk.Frame):
        left = tk.Frame(parent, bg=C.BG, width=260)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        mid = tk.Frame(parent, bg=C.BG, width=220)
        mid.pack(side="left", fill="y", padx=4, pady=8)
        mid.pack_propagate(False)

        right = tk.Frame(parent, bg=C.BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        # ── LEFT: Health & Quest ─────────────────────────────────
        SectionHeader(left, "Health & Stamina", C.GREEN).pack(fill="x", pady=(0, 4))
        self._sc(left, "hp",        "HP",        C.GREEN,  bar=True, bar_style="HP.Horizontal.TProgressbar")
        self._sc(left, "stamina",   "Stamina",   C.YELLOW, bar=True, bar_style="Stamina.Horizontal.TProgressbar")
        self._sc(left, "deaths",    "Deaths",    C.RED)
        self._sc(left, "sharpness", "Sharpness", C.ORANGE)

        make_separator(left, bg=C.BORDER).pack(fill="x", pady=6)
        SectionHeader(left, "Quest", C.CYAN).pack(fill="x", pady=(0, 4))
        self._sc(left, "zone",       "Zone",       C.CYAN)
        self._sc(left, "quest_time", "Quest Time", C.TEXT_DIM)
        self._sc(left, "distance",   "Distance",   C.TEXT_DIM)

        make_separator(left, bg=C.BORDER).pack(fill="x", pady=6)
        SectionHeader(left, "Economy", C.YELLOW).pack(fill="x", pady=(0, 4))
        self._sc(left, "money", "Zenny", C.YELLOW)

        # ── MIDDLE: Training & Exploration ───────────────────────
        SectionHeader(mid, "Training", C.ACCENT).pack(fill="x", pady=(0, 4))
        self._sc(mid, "episode",     "Episode",      C.BLUE)
        self._sc(mid, "step",        "Episode Step", C.TEXT_DIM)
        self._sc(mid, "total_steps", "Total Steps",  C.ACCENT)
        self._sc(mid, "hits",        "Hits",         C.PURPLE)

        make_separator(mid, bg=C.BORDER).pack(fill="x", pady=6)
        SectionHeader(mid, "Exploration", C.CYAN).pack(fill="x", pady=(0, 4))
        self._sc(mid, "total_cubes",             "Cubes Explored",  C.BLUE)
        self._sc(mid, "zones_discovered",        "Zones Found",     C.CYAN)
        self._sc(mid, "exploration_visits",      "Total Visits",    C.TEXT_DIM)
        self._sc(mid, "left_monster_zone_count", "Left Zone Count", C.TEXT_DIM)

        make_separator(mid, bg=C.BORDER).pack(fill="x", pady=6)
        SectionHeader(mid, "Menu", C.TEXT_DIM).pack(fill="x", pady=(0, 4))
        self._pc(mid, "game_menu_open_count", "Menu Opens",     C.TEXT_DIM)
        self._pc(mid, "game_menu_total_time", "Menu Time (s)",  C.TEXT_DIM)

        # ── RIGHT: Position & Inventory ──────────────────────────
        SectionHeader(right, "Position & Orientation", C.CYAN).pack(fill="x", pady=(0, 4))
        self._pc(right, "player_x",    "X",           C.CYAN)
        self._pc(right, "player_y",    "Y",           C.CYAN)
        self._pc(right, "player_z",    "Z",           C.CYAN)
        self._pc(right, "orientation", "Orientation",  C.BLUE)

        make_separator(right, bg=C.BORDER).pack(fill="x", pady=6)
        SectionHeader(right, "Inventory", C.TEXT_DIM).pack(fill="x", pady=(0, 4))
        self._build_inventory(right)

    # ── Refresh ──────────────────────────────────────────────────

    def refresh(self, stats: dict):
        hp      = float(stats.get('hp', 0) or 0)
        stamina = float(stats.get('stamina', 0) or 0)
        ep      = int(stats.get('episode', 0) or 0)
        total   = int(stats.get('total_steps', 0) or 0)

        hp_color = C.GREEN if hp > 70 else (C.YELLOW if hp > 30 else C.RED)
        self._u('hp',      f"{hp:.1f}",      color=hp_color, bar_value=min(hp, 100))
        self._u('stamina', f"{stamina:.1f}",  bar_value=min(100, stamina))
        self._u('deaths',  str(stats.get('deaths', 0) or 0),
                color=C.RED if (stats.get('deaths', 0) or 0) > 0 else C.TEXT_DIM)
        self._u('sharpness', str(stats.get('sharpness', 0) or 0))
        self._u('zone',      str(stats.get('zone', 0) or 0))

        qt = int(stats.get('quest_time', 0) or 0)
        self._u('quest_time', f"{qt//60}:{qt%60:02d}" if qt else "—")
        self._u('money',   f"{int(stats.get('money', 0) or 0):,} z")
        dist = float(stats.get('distance', 0) or 0)
        self._u('distance', f"{dist/500:.1f} m")
        self._u('episode',     str(ep))
        self._u('step',        str(stats.get('step', 0) or 0))
        self._u('total_steps', f"{total:,}")
        self._u('hits',        str(stats.get('hits', 0) or 0))
        self._u('total_cubes',             str(stats.get('total_cubes', 0) or 0))
        self._u('zones_discovered',        str(stats.get('zones_discovered', 0) or 0))
        self._u('exploration_visits',      str(stats.get('exploration_visits', 0) or 0))
        self._u('left_monster_zone_count', str(stats.get('left_monster_zone_count', 0) or 0))

        # Player cards
        self._up('player_x',    f"{float(stats.get('player_x', 0.0) or 0.0):.1f}")
        self._up('player_y',    f"{float(stats.get('player_y', 0.0) or 0.0):.1f}")
        self._up('player_z',    f"{float(stats.get('player_z', 0.0) or 0.0):.1f}")
        self._up('orientation', f"{float(stats.get('orientation', 0.0) or 0.0):.1f}°")
        self._up('game_menu_open_count',
                 str(stats.get('game_menu_open_count', 0) or 0))
        self._up('game_menu_total_time',
                 f"{float(stats.get('game_menu_total_time', 0.0) or 0.0):.1f}")

        # Inventory
        self._refresh_inventory(stats)

    # ── Inventory ────────────────────────────────────────────────

    def _build_inventory(self, parent):
        self._inv_scroll = ScrollableFrame(parent, bg=C.BG)
        self._inv_scroll.pack(fill="both", expand=True)
        grid = tk.Frame(self._inv_scroll.inner, bg=C.BG)
        grid.pack(fill="x")
        for i in range(24):
            col, row = i % 6, i // 6
            slot_f = tk.Frame(grid, bg=C.SURFACE,
                              highlightbackground=C.BORDER, highlightthickness=1,
                              width=90, height=52)
            slot_f.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
            slot_f.pack_propagate(False)
            grid.columnconfigure(col, weight=1)
            tk.Label(slot_f, text=f"{i+1}", font=("Consolas", 7),
                     bg=C.SURFACE, fg=C.TEXT_DIMMER).place(x=3, y=2)
            name_l = tk.Label(slot_f, text="", font=("Segoe UI", 8),
                              bg=C.SURFACE, fg=C.TEXT_DIM, wraplength=82)
            name_l.place(relx=0.5, rely=0.45, anchor="center")
            qty_l = tk.Label(slot_f, text="", font=("Consolas", 8, "bold"),
                             bg=C.SURFACE, fg=C.ACCENT)
            qty_l.place(relx=1.0, rely=1.0, anchor="se", x=-3, y=-2)
            self._inv_slots.append({'frame': slot_f, 'name': name_l, 'qty': qty_l})

    def _refresh_inventory(self, stats: dict):
        inventory = stats.get('inventory', [])
        if not isinstance(inventory, list):
            return
        inv_by_slot = {
            item.get('slot', 0): item
            for item in inventory
            if isinstance(item, dict) and item.get('slot')
        }
        for i, slot_w in enumerate(self._inv_slots):
            slot_num = i + 1
            item = inv_by_slot.get(slot_num)
            try:
                if item and item.get('item_id', 0) > 0:
                    name = item.get('name', f"ID {item.get('item_id', '?')}")
                    qty = item.get('quantity', 0)
                    slot_w['name'].configure(
                        text=name[:16] if len(name) > 16 else name, fg=C.TEXT)
                    slot_w['qty'].configure(text=f"x{qty}")
                else:
                    slot_w['name'].configure(text="", fg=C.TEXT_DIMMER)
                    slot_w['qty'].configure(text="")
            except Exception:
                pass

    # ── Helpers ───────────────────────────────────────────────────

    def _sc(self, parent, key, label, color=C.TEXT, bar=False,
            bar_style="Accent.Horizontal.TProgressbar"):
        c = MetricCard(parent, label, color=color, bar=bar, bar_style=bar_style)
        c.pack(fill="x", pady=2)
        self._stat_cards[key] = c

    def _pc(self, parent, key, label, color=C.TEXT):
        c = MetricCard(parent, label, color=color)
        c.pack(fill="x", pady=2)
        self._ply_cards[key] = c

    def _u(self, key, value, **kw):
        card = self._stat_cards.get(key)
        if card:
            try: card.update(value, **kw)
            except Exception: pass

    def _up(self, key, value, **kw):
        card = self._ply_cards.get(key)
        if card:
            try: card.update(value, **kw)
            except Exception: pass
