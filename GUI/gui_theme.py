"""
gui_theme.py — Dark terminal aesthetic for MH Tri Training Dashboard
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════

class C:
    """Centralized colour constants — deep-space terminal aesthetic."""
    # Backgrounds
    BG         = "#080b0f"   # main window
    SURFACE    = "#0e1117"   # cards
    SURFACE2   = "#151a23"   # slightly lighter
    SURFACE3   = "#1c2230"   # hover / active
    BORDER     = "#1e2d3d"   # borders
    BORDER_LT  = "#2a3f55"   # lighter border

    # Accents
    ACCENT     = "#00d4ff"   # electric cyan — primary
    ACCENT2    = "#0090b0"   # dimmer cyan
    ACCENT_BG  = "#003848"   # cyan tint bg

    # Status
    GREEN      = "#00ff88"
    GREEN_DIM  = "#00994d"
    GREEN_BG   = "#00261a"
    RED        = "#ff4757"
    RED_DIM    = "#c0392b"
    RED_BG     = "#2d0a0a"
    YELLOW     = "#ffd32a"
    YELLOW_DIM = "#b8960a"
    YELLOW_BG  = "#2a2000"
    ORANGE     = "#ff6b35"
    BLUE       = "#5585ff"
    CYAN       = "#00d4ff"   # alias for ACCENT — used in zone/player tabs
    PURPLE     = "#a855f7"
    MAGENTA    = "#ff4ecd"

    # Text
    TEXT       = "#dce6f0"
    TEXT_DIM   = "#6b7e90"
    TEXT_DIMMER= "#3d4f60"
    TEXT_CODE  = "#b0c8e0"

    # Chart
    CHART_BG   = "#080b0f"
    CHART_GRID = "#0e1825"
    CHART_AXIS = "#1e2d3d"

    # Agent palette (up to 8 agents)
    AGENTS = [
        "#00d4ff",  # 0 cyan
        "#00ff88",  # 1 green
        "#ffd32a",  # 2 yellow
        "#ff6b35",  # 3 orange
        "#a855f7",  # 4 purple
        "#ff4ecd",  # 5 magenta
        "#5585ff",  # 6 blue
        "#ff4757",  # 7 red
    ]


# Larger, bolder fonts — better readability on all screen sizes
FONT       = "Consolas"
FONT_UI    = "Segoe UI"
FONT_BOLD  = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 10)
FONT_MONO  = ("Consolas", 11)
FONT_METRIC= ("Consolas", 16, "bold")
FONT_TITLE = ("Segoe UI", 12, "bold")


# ═══════════════════════════════════════════════════════════════════════
#  TTK STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════

def apply_dark_theme(root: tk.Tk):
    """Apply the dark terminal theme to all ttk widgets."""
    s = ttk.Style(root)
    try:
        s.theme_use("clam")
    except Exception:
        pass

    # Notebook — larger tab font
    s.configure("Dark.TNotebook",
        background=C.BG, borderwidth=0, tabmargins=0)
    s.configure("Dark.TNotebook.Tab",
        background=C.SURFACE, foreground=C.TEXT_DIM,
        padding=[16, 8], font=FONT_TITLE, borderwidth=0)
    s.map("Dark.TNotebook.Tab",
        background=[("selected", C.SURFACE2), ("active", C.SURFACE3)],
        foreground=[("selected", C.ACCENT), ("active", C.TEXT)])

    # Frame
    s.configure("Dark.TFrame", background=C.BG)
    s.configure("Card.TFrame", background=C.SURFACE)
    s.configure("Card2.TFrame", background=C.SURFACE2)

    # Label
    s.configure("Dark.TLabel",
        background=C.BG, foreground=C.TEXT, font=FONT_SMALL)
    s.configure("Card.TLabel",
        background=C.SURFACE, foreground=C.TEXT, font=FONT_SMALL)
    s.configure("Dim.TLabel",
        background=C.SURFACE, foreground=C.TEXT_DIM, font=FONT_SMALL)

    # Scrollbar
    s.configure("Dark.Vertical.TScrollbar",
        background=C.SURFACE2, troughcolor=C.BG,
        arrowcolor=C.TEXT_DIM, borderwidth=0, arrowsize=12)
    s.map("Dark.Vertical.TScrollbar",
        background=[("active", C.BORDER_LT)])

    # Progressbar (used for HP/Stamina bars)
    for name, color in [
        ("HP.Horizontal.TProgressbar",      C.GREEN),
        ("Stamina.Horizontal.TProgressbar", C.YELLOW),
        ("Combat.Horizontal.TProgressbar",  C.RED),
        ("Blue.Horizontal.TProgressbar",    C.BLUE),
        ("Accent.Horizontal.TProgressbar",  C.ACCENT),
    ]:
        s.configure(name,
            background=color, troughcolor=C.SURFACE3,
            borderwidth=0, thickness=8,
            lightcolor=color, darkcolor=color)

    # Button
    s.configure("Dark.TButton",
        background=C.SURFACE2, foreground=C.TEXT,
        font=FONT_SMALL, borderwidth=1, relief="flat",
        padding=[10, 5])
    s.map("Dark.TButton",
        background=[("active", C.SURFACE3), ("pressed", C.BORDER)])

    s.configure("Stop.TButton",
        background=C.RED_BG, foreground=C.RED,
        font=("Segoe UI", 11, "bold"), borderwidth=1, relief="flat",
        padding=[10, 6])
    s.map("Stop.TButton",
        background=[("active", "#4a0a0a")])

    # Separator
    s.configure("Dark.TSeparator", background=C.BORDER)

    # Combobox
    s.configure("Dark.TCombobox",
        background=C.SURFACE2, foreground=C.TEXT,
        fieldbackground=C.SURFACE2, selectbackground=C.ACCENT_BG,
        arrowcolor=C.TEXT_DIM, font=FONT_SMALL)


# ═══════════════════════════════════════════════════════════════════════
#  REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════════

class MetricCard(tk.Frame):
    """
    Compact metric display: label + value + optional bar.
    Used in the left stats panel and overview grid.
    """
    def __init__(self, parent, label: str, value: str = "—",
                 color: str = C.ACCENT, bar: bool = False,
                 bar_style: str = "Accent.Horizontal.TProgressbar",
                 **kwargs):
        super().__init__(parent, bg=C.SURFACE, **kwargs)
        self.configure(highlightbackground=C.BORDER, highlightthickness=1)

        inner = tk.Frame(self, bg=C.SURFACE, padx=8, pady=5)
        inner.pack(fill="both", expand=True)

        # Larger, bolder label text
        tk.Label(inner, text=label.upper(), font=("Segoe UI", 9, "bold"),
                 bg=C.SURFACE, fg=C.TEXT_DIM).pack(anchor="w")

        # Larger value text
        self._val_lbl = tk.Label(inner, text=value, font=("Consolas", 14, "bold"),
                                  bg=C.SURFACE, fg=color)
        self._val_lbl.pack(anchor="w")

        if bar:
            self._bar = ttk.Progressbar(inner, style=bar_style, length=120,
                                         maximum=100, mode="determinate")
            self._bar.pack(fill="x", pady=(2, 0))
        else:
            self._bar = None

    def update(self, value: str, color: str = None, bar_value: float = None):
        self._val_lbl.configure(text=value)
        if color:
            self._val_lbl.configure(fg=color)
        if self._bar is not None and bar_value is not None:
            self._bar["value"] = max(0, min(100, bar_value))


class SectionHeader(tk.Frame):
    """Labelled section separator with accent line."""
    def __init__(self, parent, text: str, color: str = C.ACCENT, **kwargs):
        super().__init__(parent, bg=C.BG, **kwargs)
        tk.Label(self, text=text.upper(), font=("Segoe UI", 10, "bold"),
                 bg=C.BG, fg=color).pack(side="left", padx=(0, 8))
        tk.Frame(self, bg=color, height=1).pack(side="left", fill="x", expand=True)


class StatusBadge(tk.Label):
    """Small colored badge: YES/NO, ACTIVE, etc."""
    def __init__(self, parent, text="—", active=False, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(font=("Segoe UI", 9, "bold"), padx=6, pady=2)
        self.set_state(text, active)

    def set_state(self, text: str, active: bool):
        if active:
            self.configure(text=text, bg=C.GREEN_BG, fg=C.GREEN)
        else:
            self.configure(text=text, bg=C.SURFACE3, fg=C.TEXT_DIM)


class ScrollableFrame(tk.Frame):
    """Frame with vertical scrollbar."""
    def __init__(self, parent, bg=C.BG, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical",
                           command=canvas.yview, style="Dark.Vertical.TScrollbar")
        self.inner = tk.Frame(canvas, bg=bg)
        self.inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._canvas = canvas

    def refresh_scroll(self):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))


class AgentBadge(tk.Label):
    """Colored circular badge showing agent ID."""
    def __init__(self, parent, agent_id: int = 0, size: int = 18, **kwargs):
        color = C.AGENTS[agent_id % len(C.AGENTS)]
        super().__init__(parent, text=str(agent_id),
                         width=2, font=("Consolas", 10, "bold"),
                         bg=color, fg=C.BG, relief="flat", **kwargs)


def make_separator(parent, orient="h", bg=C.BORDER, size=1):
    """Thin separator line."""
    if orient == "h":
        return tk.Frame(parent, bg=bg, height=size)
    return tk.Frame(parent, bg=bg, width=size)


def apply_matplotlib_dark_style(fig, axes=None):
    """Apply dark background to a matplotlib figure."""
    fig.patch.set_facecolor(C.CHART_BG)
    if axes is None:
        axes = fig.get_axes()
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(C.CHART_BG)
        ax.tick_params(colors=C.TEXT_DIM, labelsize=8)
        ax.xaxis.label.set_color(C.TEXT_DIM)
        ax.yaxis.label.set_color(C.TEXT_DIM)
        for spine in ax.spines.values():
            spine.set_edgecolor(C.CHART_AXIS)
        ax.grid(True, color=C.CHART_GRID, linewidth=0.5, alpha=0.8)
    fig.tight_layout(pad=1.2)