"""
gui_map3d.py — Self-contained 3D exploration map panel
Extracted from training_gui for cleanliness and maintainability.
"""
import tkinter as tk
import numpy as np
from typing import Optional, Dict, List

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from GUI.gui_theme import C, FONT_SMALL, FONT_TITLE

# Visit count → (colour, alpha)
_CUBE_STYLE = [
    (0,  "#0099cc", 0.30),   # 1 visit:  visits > 0
    (1,  "#00cc66", 0.35),   # 2-4:      visits > 1
    (4,  "#cccc00", 0.35),   # 5-6:      visits > 4
    (6,  "#ff9900", 0.40),   # 7-9:      visits > 6
    (9,  "#ff3333", 0.45),   # 10+:      visits > 9
]

def _cube_color(visits: int):
    # 0 visits = grey, handled before the loop to avoid threshold ambiguity
    if visits <= 0:
        return "#2a3a50", 0.15
    for threshold, color, alpha in reversed(_CUBE_STYLE):
        if visits > threshold:
            return color, alpha
    # 1 visit fallback (visits > 0 but no higher threshold matched)
    return _CUBE_STYLE[0][1], _CUBE_STYLE[0][2]


def _draw_cube(ax, cx, cy, cz, sx, sy, sz, color, alpha, draw_obstacles=False, blocked=None):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
    ]
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[0], verts[3], verts[7], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
    ]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=color, linewidths=0.3,
        edgecolors=C.BORDER, alpha=alpha, zorder=5))

    if draw_obstacles and blocked:
        offsets = {
            'north': (0, hz*0.9, 0),  'south': (0, -hz*0.9, 0),
            'east':  (hx*0.9, 0, 0),  'west':  (-hx*0.9, 0, 0),
            'up':    (0, 0, hy*0.9),  'down':  (0, 0, -hy*0.9),
        }
        for direction, count in blocked.items():
            if count >= 5:
                off = offsets.get(direction, (0, 0, 0))
                ax.scatter(cx+off[0], cz+off[2], cy+off[1],
                           c='#ff4444', marker='x', s=80,
                           linewidths=2, alpha=0.9, zorder=50, depthshade=False)


class MapPanel(tk.Frame):
    """
    Self-contained 3D exploration map panel.
    Manages its own Figure lifecycle — never recreates axes on resize.
    """

    MAX_CUBES_VISIBLE = 50
    VIEW_RADIUS       = 1800.0
    REFRESH_INTERVAL  = 1200   # ms

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C.BG, **kwargs)
        self._current_stats: Dict = {}
        self._auto_refresh  = True
        self._after_id      = None

        # label flags for legend deduplication
        self._lbl_water = self._lbl_monster = self._lbl_trans = self._lbl_obs = False

        # Hover state
        self._hovered_cube  = None
        self._cube_data_cache: List[Dict] = []

        # Store reference to canvas_frame for tooltip placement
        self._canvas_frame: Optional[tk.Frame] = None

        self._build_ui()

    # ── UI BUILD ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── toolbar ───────────────────────────────────────────────────
        bar = tk.Frame(self, bg=C.SURFACE, pady=4)
        bar.pack(fill="x")

        tk.Label(bar, text="EXPLORATION MAP", font=FONT_TITLE,
                 bg=C.SURFACE, fg=C.ACCENT).pack(side="left", padx=10)

        # Zone filter
        tk.Label(bar, text="zone:", font=FONT_SMALL,
                 bg=C.SURFACE, fg=C.TEXT_DIM).pack(side="left", padx=(12, 2))
        self._zone_var = tk.IntVar(value=0)
        tk.Spinbox(bar, from_=0, to=20, textvariable=self._zone_var,
                   width=4, bg=C.SURFACE2, fg=C.TEXT, insertbackground=C.TEXT,
                   buttonbackground=C.SURFACE3, relief="flat", font=FONT_SMALL
                   ).pack(side="left")

        # Snapshot button
        snap_btn = tk.Button(bar, text="⊞  FULL ZONE",
                             font=FONT_SMALL, bg=C.ACCENT_BG, fg=C.ACCENT,
                             activebackground=C.SURFACE3, relief="flat",
                             bd=0, padx=8, cursor="hand2",
                             command=self._open_full_snapshot)
        snap_btn.pack(side="right", padx=8)

        # Auto-refresh toggle
        self._refresh_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="auto", variable=self._refresh_var,
                       font=FONT_SMALL, bg=C.SURFACE, fg=C.TEXT_DIM,
                       activebackground=C.SURFACE, selectcolor=C.SURFACE3,
                       command=self._on_refresh_toggle).pack(side="right")

        # ── stats bar ─────────────────────────────────────────────────
        self._stats_lbl = tk.Label(self, font=("Consolas", 10),
                                    bg=C.BG, fg=C.TEXT_DIM, pady=2)
        self._stats_lbl.pack(fill="x", padx=4)

        # ── matplotlib canvas ─────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=C.BG)
        canvas_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._canvas_frame = canvas_frame

        self._fig = Figure(figsize=(7, 5), dpi=96)
        self._ax  = self._fig.add_subplot(111, projection='3d')
        self._style_axes()

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, canvas_frame)
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._mpl_canvas.mpl_connect("motion_notify_event", self._on_hover)

        # ── tooltip overlay — larger font, solid border, near cursor ──
        self._tooltip_lbl = tk.Label(
            canvas_frame,
            font=("Consolas", 11, "bold"),
            justify="left",
            bg="#0a0f18",
            fg=C.ACCENT,
            padx=10,
            pady=8,
            relief="solid",
            bd=1)
        self._tooltip_lbl.place_forget()

        # ── legend — larger font ──────────────────────────────────────
        leg = tk.Frame(self, bg=C.SURFACE)
        leg.pack(fill="x", padx=4, pady=2)
        items = [
            ("0 visits",   "#2a3a50"), ("1 visit",  "#0099cc"),
            ("2-4",        "#00cc66"), ("5-6",      "#cccc00"),
            ("7-9",        "#ff9900"), ("10+",      "#ff3333"),
            ("player",     C.ACCENT),  ("obstacle", "#ff4444"),
        ]
        for label, color in items:
            dot = tk.Frame(leg, bg=color, width=10, height=10)
            dot.pack(side="left", padx=(6, 1), pady=4)
            tk.Label(leg, text=label, font=("Segoe UI", 9, "bold"),
                     bg=C.SURFACE, fg=C.TEXT_DIM).pack(side="left", padx=(0, 6))

        self._schedule_refresh()

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(C.CHART_BG)
        self._fig.patch.set_facecolor(C.BG)
        ax.tick_params(colors=C.TEXT_DIMMER, labelsize=7)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(C.BORDER)
        ax.grid(True, color=C.CHART_GRID, linewidth=0.3, alpha=0.6)
        ax.set_xlabel("X", fontsize=8, color=C.TEXT_DIMMER)
        ax.set_ylabel("Z", fontsize=8, color=C.TEXT_DIMMER)
        ax.set_zlabel("Y", fontsize=8, color=C.TEXT_DIMMER)

    # ── DATA INPUT ───────────────────────────────────────────────────────

    def update_data(self, stats: Dict):
        self._current_stats = stats

    # ── REFRESH LOOP ────────────────────────────────────────────────────

    def _on_refresh_toggle(self):
        self._auto_refresh = self._refresh_var.get()
        if self._auto_refresh:
            self._schedule_refresh()

    def _schedule_refresh(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = self.after(self.REFRESH_INTERVAL, self._refresh)

    def _refresh(self):
        try:
            self._render()
        except Exception:
            pass
        if self._auto_refresh:
            self._schedule_refresh()

    def stop(self):
        self._auto_refresh = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass

    # ── RENDER ───────────────────────────────────────────────────────────

    def _render(self):
        if not self._current_stats:
            return

        stats = self._current_stats
        px = stats.get('player_x', 0.0) or 0.0
        py = stats.get('player_y', 0.0) or 0.0
        pz = stats.get('player_z', 0.0) or 0.0

        current_zone = stats.get('zone', 0) or 0
        zone_filter  = self._zone_var.get() or current_zone

        cubes_data: Dict = stats.get('exploration_cubes', {})
        zone_cubes: List = cubes_data.get(zone_filter, [])
        total_cubes = sum(len(v) for v in cubes_data.values())

        self._ax.cla()
        self._style_axes()
        self._cube_data_cache = []

        # Filter cubes to visible range
        if px or py or pz:
            visible = sorted(
                [c for c in zone_cubes
                 if (abs(c['center_x']-px)**2 +
                     abs(c['center_y']-py)**2 +
                     abs(c['center_z']-pz)**2) <= self.VIEW_RADIUS**2],
                key=lambda c: (c['center_x']-px)**2 + (c['center_y']-py)**2 + (c['center_z']-pz)**2
            )[:self.MAX_CUBES_VISIBLE]
        else:
            visible = zone_cubes[:self.MAX_CUBES_VISIBLE]

        # Draw cubes (coordinates relative to player for centred view)
        for cube in visible:
            rx = cube['center_x'] - px
            ry = cube['center_y'] - py
            rz = cube['center_z'] - pz
            color, alpha = _cube_color(cube.get('visit_count', 0))
            sx = cube.get('size_x', 650)
            sy = cube.get('size_y', 650)
            sz = cube.get('size_z', 650)
            _draw_cube(self._ax, rx, rz, ry, sx, sz, sy, color, alpha,
                       draw_obstacles=True,
                       blocked=cube.get('blocked_directions', {}))
            self._cube_data_cache.append({
                'rel': (rx, ry, rz),
                'cube': cube
            })
            self._draw_markers(cube, rx, rz, ry)

        # Player arrow
        ori = stats.get('orientation', 0.0) or 0.0
        ori_rad = np.radians(ori)
        alen = 350.0
        self._ax.quiver(0, 0, 0,
                        alen*np.sin(ori_rad), alen*np.cos(ori_rad), 0,
                        color=C.ACCENT, arrow_length_ratio=0.3,
                        linewidth=2.5, alpha=0.9, zorder=100)
        self._ax.scatter([0], [0], [0], c=C.ACCENT, s=60,
                         edgecolors='white', linewidths=1.5,
                         zorder=101, depthshade=False)

        # Axis limits
        vr = self.VIEW_RADIUS * 0.7
        self._ax.set_xlim(-vr, vr)
        self._ax.set_ylim(-vr, vr)
        self._ax.set_zlim(-vr*0.5, vr*0.5)
        self._ax.set_box_aspect([1, 1, 0.5])

        # Stats label
        total_visits = sum(c.get('visit_count', 0) for c in zone_cubes)
        self._stats_lbl.configure(
            text=f"zone {zone_filter} │ {len(zone_cubes)} cubes ({len(visible)} shown) "
                 f"│ {total_visits} visits │ global: {total_cubes} cubes")

        try:
            self._mpl_canvas.draw()
        except Exception:
            pass

    def _draw_markers(self, cube: Dict, rx: float, rz_ax: float, ry: float):
        """Draw cube markers (water, monster, transition)."""
        markers = cube.get('markers', {})
        if not markers:
            return
        from reward.cube_markers import MarkerType
        if MarkerType.WATER in markers:
            self._ax.scatter(rx, rz_ax, ry, c='#0099ff', marker='o', s=80,
                             alpha=0.8, edgecolors='white', linewidths=1,
                             zorder=80, depthshade=False)
        if MarkerType.MONSTER_LOCATION in markers:
            m = markers[MarkerType.MONSTER_LOCATION]
            self._ax.scatter(rx, rz_ax, ry, c='#ff4444', marker='X',
                             s=80 + m.strength*150, alpha=max(0.3, m.strength*0.9),
                             edgecolors='black', linewidths=1,
                             zorder=80, depthshade=False)
        if MarkerType.ZONE_TRANSITION in markers:
            self._ax.scatter(rx, rz_ax, ry, c='#ffd32a', marker='^', s=100,
                             alpha=0.85, edgecolors='#ff9900', linewidths=1.5,
                             zorder=80, depthshade=False)

    # ── HOVER TOOLTIP — follows mouse, large and readable ────────────────

    def _on_hover(self, event):
        if event.inaxes != self._ax or not self._cube_data_cache:
            self._tooltip_lbl.place_forget()
            return
        if event.xdata is None:
            self._tooltip_lbl.place_forget()
            return

        # Find closest cube via 2D projection
        best, best_dist = None, float('inf')
        try:
            proj = self._ax.get_proj()
            for item in self._cube_data_cache:
                rx, ry, rz = item['rel']
                pt = np.array([rx, rz, ry, 1.0])
                p2 = proj.dot(pt)
                if p2[3] != 0:
                    p2 = p2[:2] / p2[3]
                    inv = self._ax.transData.inverted()
                    dc = inv.transform(self._ax.transData.transform([[p2[0], p2[1]]]))[0]
                    d = (event.xdata - dc[0])**2 + (event.ydata - dc[1])**2
                    if d < best_dist:
                        best_dist = d
                        best = item
        except Exception:
            pass

        if best is None or best_dist > 1e6:
            self._tooltip_lbl.place_forget()
            self._hovered_cube = None
            return

        if best is self._hovered_cube:
            # Still same cube — update position in case mouse moved
            self._reposition_tooltip()
            return

        self._hovered_cube = best

        cube = best['cube']
        lines = [
            f"zone {cube['zone_id']} │ ({cube['center_x']:.0f}, {cube['center_y']:.0f}, {cube['center_z']:.0f})",
            f"visits  real:{cube.get('visit_count',0)}  eff:{cube.get('effective_visit_count',0)}  total:{cube.get('total_visits',0)}",
        ]
        markers = cube.get('markers', {})
        if markers:
            try:
                from reward.cube_markers import MarkerType
                names = {MarkerType.WATER: 'WATER', MarkerType.MONSTER_LOCATION: 'MONSTER',
                         MarkerType.ZONE_TRANSITION: 'TRANSITION'}
                lines.append("  ".join(names.get(k, str(k)) for k in markers.keys()))
            except Exception:
                pass

        self._tooltip_lbl.configure(text="\n".join(lines))
        self._reposition_tooltip()

    def _reposition_tooltip(self):
        """Place tooltip near mouse cursor, keeping it inside the canvas widget."""
        w = self._mpl_canvas.get_tk_widget()
        try:
            # Mouse position relative to the canvas widget
            mouse_x = w.winfo_pointerx() - w.winfo_rootx()
            mouse_y = w.winfo_pointery() - w.winfo_rooty()
            canvas_w = w.winfo_width()
            canvas_h = w.winfo_height()

            # Offset so cursor does not overlap the label
            offset_x = 16
            offset_y = -80  # above cursor by default

            tip_x = mouse_x + offset_x
            tip_y = mouse_y + offset_y

            # Keep tooltip inside canvas bounds (rough estimate: 200x80 px)
            tip_x = max(4, min(tip_x, canvas_w - 220))
            tip_y = max(4, min(tip_y, canvas_h - 90))

        except Exception:
            tip_x, tip_y = 8, 8

        self._tooltip_lbl.place(in_=w, x=tip_x, y=tip_y)

    # ── FULL SNAPSHOT WINDOW ─────────────────────────────────────────────

    def _open_full_snapshot(self):
        stats = self._current_stats
        if not stats:
            return
        zone = self._zone_var.get() or stats.get('zone', 0) or 0
        cubes = stats.get('exploration_cubes', {}).get(zone, [])
        if not cubes:
            return

        win = tk.Toplevel(self)
        win.title(f"Zone {zone} — Full Map Snapshot")
        win.configure(bg=C.BG)
        win.geometry("950x780")

        fig = Figure(figsize=(9, 6.5), dpi=96)
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(C.BG)
        ax.set_facecolor(C.CHART_BG)
        ax.set_title(f"Zone {zone} — {len(cubes)} cubes",
                     fontsize=10, color=C.TEXT, fontname="Consolas", pad=10)

        for cube in cubes:
            cx, cy, cz = cube['center_x'], cube['center_y'], cube['center_z']
            sx = cube.get('size_x', 650)
            sy = cube.get('size_y', 650)
            sz = cube.get('size_z', 650)
            color, alpha = _cube_color(cube.get('visit_count', 0))
            _draw_cube(ax, cx, cz, cy, sx, sz, sy, color, alpha)

        if cubes:
            xs = [c['center_x'] for c in cubes]
            ys = [c['center_y'] for c in cubes]
            zs = [c['center_z'] for c in cubes]
            cx_c = (min(xs)+max(xs))/2; cy_c = (min(ys)+max(ys))/2; cz_c = (min(zs)+max(zs))/2
            hr = max((max(xs)-min(xs)), (max(ys)-min(ys)), (max(zs)-min(zs)), 1000) * 0.6
            ax.set_xlim(cx_c-hr, cx_c+hr)
            ax.set_ylim(cz_c-hr, cz_c+hr)
            ax.set_zlim(cy_c-hr*0.5, cy_c+hr*0.5)
            ax.set_box_aspect([1, 1, 0.5])

        ax.set_xlabel("X", fontsize=8, color=C.TEXT_DIM)
        ax.set_ylabel("Z", fontsize=8, color=C.TEXT_DIM)
        ax.set_zlabel("Y", fontsize=8, color=C.TEXT_DIM)
        ax.tick_params(colors=C.TEXT_DIMMER, labelsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor(C.BORDER)

        c = FigureCanvasTkAgg(fig, win)
        c.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        c.draw()

        total_v = sum(c['visit_count'] for c in cubes)
        tk.Label(win, font=("Consolas", 10), bg=C.BG, fg=C.TEXT_DIM,
                 text=f"{len(cubes)} cubes │ {total_v} total visits"
                 ).pack(pady=(0, 6))