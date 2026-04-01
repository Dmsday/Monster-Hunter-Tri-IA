"""
gui_tab_actions.py — Actions tab: action heads schematic with compatibility
lines, per-head activation rate bars, conflict display.
"""

import tkinter as tk
from collections import deque
from typing import Dict, Optional

from GUI.gui_theme import C, SectionHeader, make_separator
from GUI.gui_state import GuiState
from core.controller.action_heads import (
    NUM_HEADS, HEAD_NAMES, HEAD_PRIORITY, COMPAT,
    HEAD_MOVEMENT, HEAD_CAMERA, HEAD_COMBAT,
    HEAD_USE_ITEM, HEAD_SELECT_ITEM, HEAD_MENU, HEAD_SPRINT,
)

# Branch labels per head for display
_BRANCH_LABELS = {
    HEAD_MOVEMENT:    {0: '—', 1: 'Forward', 2: 'Backward', 3: 'Strafe L', 4: 'Strafe R'},
    HEAD_CAMERA:      {0: '—', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'},
    HEAD_COMBAT:      {0: '—', 1: 'Attack', 2: 'Dodge', 3: 'Draw/Sheath', 4: 'Kick'},
    HEAD_USE_ITEM:    {0: '—', 1: 'Use'},
    HEAD_SELECT_ITEM: {0: '—', 1: 'Sel Left', 2: 'Sel Right'},
    HEAD_MENU:        {0: '—', 1: 'Start', 2: 'Nav Up', 3: 'Nav Down',
                       4: 'Nav Left', 5: 'Nav Right', 6: 'Confirm', 7: 'Back'},
    HEAD_SPRINT:      {0: '—', 1: 'Sprint'},
}

# Visual layout: (head_idx, x_center, y_center)
_HEAD_LAYOUT = [
    (HEAD_MENU,        300,  40),
    (HEAD_USE_ITEM,    130, 120),
    (HEAD_COMBAT,      470, 120),
    (HEAD_SELECT_ITEM, 100, 230),
    (HEAD_SPRINT,      500, 230),
    (HEAD_CAMERA,      200, 340),
    (HEAD_MOVEMENT,    400, 340),
]

_HEAD_COLORS = {
    HEAD_MOVEMENT:    C.GREEN,
    HEAD_CAMERA:      C.BLUE,
    HEAD_COMBAT:      C.RED,
    HEAD_USE_ITEM:    C.YELLOW,
    HEAD_SELECT_ITEM: C.ORANGE,
    HEAD_MENU:        C.PURPLE,
    HEAD_SPRINT:      C.CYAN,
}


class ActionsTab:
    """Action heads schematic with live status and activation rates."""

    def __init__(self, state: GuiState):
        self.state = state
        self._canvas:       Optional[tk.Canvas] = None
        self._head_items:   Dict[int, Dict]     = {}
        self._compat_lines: list                = []
        self._info_lbls:    Dict[str, tk.Label] = {}
        self._history: Dict[int, deque] = {h: deque(maxlen=100) for h in range(NUM_HEADS)}

    def build(self, parent: tk.Frame):
        body = tk.Frame(parent, bg=C.BG)
        body.pack(fill="both", expand=True)

        # ── Left: canvas schematic ───────────────────────────────
        canvas_frame = tk.Frame(body, bg=C.BG)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self._canvas = tk.Canvas(canvas_frame, bg=C.BG, highlightthickness=0,
                                  width=600, height=440)
        self._canvas.pack(fill="both", expand=True)

        # Compatibility lines (drawn under the boxes)
        head_pos = {h: (x, y) for h, x, y in _HEAD_LAYOUT}
        drawn_pairs = set()
        for h_a, compat_set in COMPAT.items():
            for h_b in compat_set:
                if h_b == h_a:
                    continue
                pair = (min(h_a, h_b), max(h_a, h_b))
                if pair in drawn_pairs:
                    continue
                drawn_pairs.add(pair)
                x1, y1 = head_pos[h_a]
                x2, y2 = head_pos[h_b]
                lid = self._canvas.create_line(x1, y1, x2, y2,
                                                fill=C.BORDER, width=1, dash=(3, 4))
                self._compat_lines.append(lid)

        # Head boxes
        BOX_W, BOX_H = 120, 60
        for h_idx, cx, cy in _HEAD_LAYOUT:
            color = _HEAD_COLORS[h_idx]
            x0, y0 = cx - BOX_W // 2, cy - BOX_H // 2
            x1, y1 = cx + BOX_W // 2, cy + BOX_H // 2
            glow = self._canvas.create_rectangle(
                x0 - 2, y0 - 2, x1 + 2, y1 + 2, outline=C.BORDER, width=2)
            box = self._canvas.create_rectangle(
                x0, y0, x1, y1, fill=C.SURFACE, outline=C.BORDER, width=1)
            name_id = self._canvas.create_text(
                cx, cy - 10, text=HEAD_NAMES[h_idx].upper(),
                fill=C.TEXT_DIM, font=("Segoe UI", 9, "bold"))
            branch_id = self._canvas.create_text(
                cx, cy + 12, text="—", fill=color, font=("Consolas", 10, "bold"))
            self._canvas.create_text(
                x1 - 8, y0 + 8, text=str(HEAD_PRIORITY[h_idx]),
                fill=C.TEXT_DIMMER, font=("Consolas", 7))
            self._head_items[h_idx] = {
                'glow': glow, 'box': box,
                'name': name_id, 'branch': branch_id,
                'color': color, 'cx': cx, 'cy': cy,
            }

        self._canvas.create_text(
            300, 420,
            text="── compatible    ■ active    ■ suppressed    ■ idle    ■ disabled",
            fill=C.TEXT_DIMMER, font=("Segoe UI", 8))

        # ── Right: info side panel ───────────────────────────────
        side = tk.Frame(body, bg=C.SURFACE, width=260,
                        highlightbackground=C.BORDER, highlightthickness=1)
        side.pack(side="right", fill="y", padx=(0, 8), pady=8)
        side.pack_propagate(False)

        SectionHeader(side, "Action Details", C.ACCENT).pack(fill="x", padx=6, pady=(6, 4))
        self._info_row(side, "Active Heads:", "active_count",    C.GREEN)
        self._info_row(side, "Suppressed:",   "suppressed_count", C.RED)
        self._info_row(side, "Menu Open:",    "menu_state",       C.PURPLE)

        make_separator(side, bg=C.BORDER).pack(fill="x", pady=6, padx=6)
        SectionHeader(side, "Head Activation %", C.CYAN).pack(fill="x", padx=6, pady=(0, 4))

        for h_idx in range(NUM_HEADS):
            color = _HEAD_COLORS[h_idx]
            row = tk.Frame(side, bg=C.SURFACE)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=HEAD_NAMES[h_idx].upper(),
                     font=("Consolas", 8), bg=C.SURFACE, fg=color).pack(side="left")
            bar_cv = tk.Canvas(row, bg=C.SURFACE3, highlightthickness=0,
                               width=100, height=10)
            bar_cv.pack(side="left", padx=(6, 4))
            bar_fill = bar_cv.create_rectangle(0, 0, 0, 10, fill=color, outline="")
            pct_lbl = tk.Label(row, text="0%", font=("Consolas", 8),
                               bg=C.SURFACE, fg=C.TEXT_DIM)
            pct_lbl.pack(side="right")
            self._info_lbls[f'rate_{h_idx}']    = pct_lbl
            self._info_lbls[f'bar_{h_idx}']     = bar_cv
            self._info_lbls[f'barfill_{h_idx}'] = bar_fill

        make_separator(side, bg=C.BORDER).pack(fill="x", pady=6, padx=6)
        SectionHeader(side, "Conflicts This Step", C.RED).pack(fill="x", padx=6, pady=(0, 4))
        self._info_lbls['conflicts'] = tk.Label(
            side, text="—", font=("Consolas", 9),
            bg=C.SURFACE, fg=C.TEXT_DIM, justify="left", wraplength=230)
        self._info_lbls['conflicts'].pack(fill="x", padx=8, pady=4)

    # ── Refresh ──────────────────────────────────────────────────

    def refresh(self, stats: dict):
        if not self._canvas:
            return
        raw_action = stats.get('action')
        resolved   = stats.get('resolved_action')
        in_menu    = bool(stats.get('in_game_menu', False))

        raw = self._to_list(raw_action)
        res = self._to_list(resolved) if resolved is not None else list(raw)

        active_count = 0
        suppressed_count = 0
        conflict_parts = []

        for h_idx in range(NUM_HEADS):
            items = self._head_items.get(h_idx)
            if not items:
                continue
            r_val, s_val = int(raw[h_idx]), int(res[h_idx])
            color = items['color']

            # Disabled head
            if h_idx in self.state.disabled_heads:
                self._history[h_idx].append(0)
                self._set_head_visual(items, 'disabled')
                continue

            self._history[h_idx].append(1 if s_val != 0 else 0)

            if s_val != 0:
                active_count += 1
                label = _BRANCH_LABELS.get(h_idx, {}).get(s_val, f'?{s_val}')
                self._set_head_visual(items, 'active', label)
            elif r_val != 0:
                suppressed_count += 1
                wanted = _BRANCH_LABELS.get(h_idx, {}).get(r_val, f'?{r_val}')
                conflict_parts.append(f"{HEAD_NAMES[h_idx]}: {wanted} ✗")
                self._set_head_visual(items, 'suppressed', wanted)
            else:
                self._set_head_visual(items, 'idle')

        # Compatibility lines
        active_heads = {h for h in range(NUM_HEADS) if int(res[h]) != 0}
        drawn_pairs = set()
        line_idx = 0
        for h_a, compat_set in COMPAT.items():
            for h_b in compat_set:
                if h_b == h_a:
                    continue
                pair = (min(h_a, h_b), max(h_a, h_b))
                if pair in drawn_pairs:
                    continue
                drawn_pairs.add(pair)
                if line_idx < len(self._compat_lines):
                    lid = self._compat_lines[line_idx]
                    try:
                        if h_a in active_heads and h_b in active_heads:
                            self._canvas.itemconfigure(lid, fill=C.GREEN_DIM, width=2, dash=())
                        else:
                            self._canvas.itemconfigure(lid, fill=C.BORDER, width=1, dash=(3, 4))
                    except tk.TclError:
                        pass
                line_idx += 1

        # Side panel
        self._slbl('active_count',     str(active_count),
                    C.GREEN if active_count > 0 else C.TEXT_DIM)
        self._slbl('suppressed_count', str(suppressed_count),
                    C.RED if suppressed_count > 0 else C.TEXT_DIM)
        self._slbl('menu_state',       "YES" if in_menu else "NO",
                    C.PURPLE if in_menu else C.TEXT_DIM)

        # Activation rate bars
        for h_idx in range(NUM_HEADS):
            hist = self._history[h_idx]
            rate = sum(hist) / max(len(hist), 1) * 100
            self._slbl(f'rate_{h_idx}', f"{rate:.0f}%")
            bar_cv   = self._info_lbls.get(f'bar_{h_idx}')
            bar_fill = self._info_lbls.get(f'barfill_{h_idx}')
            if bar_cv and bar_fill:
                try: bar_cv.coords(bar_fill, 0, 0, int(rate), 10)
                except Exception: pass

        self._slbl('conflicts',
                   "\n".join(conflict_parts) if conflict_parts else "None",
                   C.RED if conflict_parts else C.TEXT_DIM)

    # ── Internal helpers ─────────────────────────────────────────

    def _set_head_visual(self, items: dict, mode: str, label: str = ""):
        """Set box appearance for a given mode (active/suppressed/idle/disabled)."""
        try:
            if mode == 'active':
                c = items['color']
                self._canvas.itemconfigure(items['glow'],   outline=c, width=3)
                self._canvas.itemconfigure(items['box'],    fill=C.SURFACE2, outline=c)
                self._canvas.itemconfigure(items['name'],   fill=C.TEXT)
                self._canvas.itemconfigure(items['branch'], text=label, fill=c)
            elif mode == 'suppressed':
                self._canvas.itemconfigure(items['glow'],   outline=C.RED_DIM, width=2)
                self._canvas.itemconfigure(items['box'],    fill=C.RED_BG, outline=C.RED_DIM)
                self._canvas.itemconfigure(items['name'],   fill=C.RED_DIM)
                self._canvas.itemconfigure(items['branch'], text=f"✗ {label}", fill=C.RED)
            elif mode == 'disabled':
                self._canvas.itemconfigure(items['glow'],   outline=C.TEXT_DIMMER, width=1)
                self._canvas.itemconfigure(items['box'],    fill=C.BG, outline=C.TEXT_DIMMER)
                self._canvas.itemconfigure(items['name'],   fill=C.TEXT_DIMMER)
                self._canvas.itemconfigure(items['branch'], text="DISABLED", fill=C.TEXT_DIMMER)
            else:  # idle
                self._canvas.itemconfigure(items['glow'],   outline=C.BORDER, width=1)
                self._canvas.itemconfigure(items['box'],    fill=C.SURFACE, outline=C.BORDER)
                self._canvas.itemconfigure(items['name'],   fill=C.TEXT_DIMMER)
                self._canvas.itemconfigure(items['branch'], text="—", fill=C.TEXT_DIMMER)
        except tk.TclError:
            pass

    def _slbl(self, key, text, color=None):
        lbl = self._info_lbls.get(key)
        if lbl:
            try:
                kw = {'text': text}
                if color:
                    kw['fg'] = color
                lbl.configure(**kw)
            except Exception:
                pass

    def _info_row(self, parent, label, key, color):
        row = tk.Frame(parent, bg=C.SURFACE)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label, font=("Segoe UI", 9),
                 bg=C.SURFACE, fg=C.TEXT_DIM).pack(side="left")
        lbl = tk.Label(row, text="—", font=("Consolas", 10, "bold"),
                       bg=C.SURFACE, fg=color)
        lbl.pack(side="right")
        self._info_lbls[key] = lbl

    @staticmethod
    def _to_list(action) -> list:
        if action is None:
            return [0] * NUM_HEADS
        try:
            lst = action.tolist() if hasattr(action, 'tolist') else list(action)
        except (TypeError, ValueError):
            lst = [0] * NUM_HEADS
        while len(lst) < NUM_HEADS:
            lst.append(0)
        return lst
