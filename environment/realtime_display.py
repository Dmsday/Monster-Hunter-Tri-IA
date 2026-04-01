"""
realtime_display.py — Real-time OpenCV display for training monitoring.

Contains:
    SurveillanceWindow  — Multi-instance grid display (security-camera style)
    DisplayMixin        — Per-env display methods (_display_rt_vision, _display_rt_minimap_debug)
"""

import numpy as np

from info.module_logger import get_module_logger

logger = get_module_logger('rt_display')

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    cv2 = None


# ======================================================================
#  Surveillance window (multi-instance grid)
# ======================================================================

class SurveillanceWindow:
    """
    Single OpenCV window showing all instances' raw frames in a grid.
    Labels show "Agent X | Env Y" to distinguish ownership.
    """

    def __init__(self, num_agents: int, allocation: dict = None):
        self._num_agents = num_agents
        self._frames = {}
        self._lock = __import__('threading').Lock()
        self._win_name = "Surveillance - All Instances"

        # Build env_idx -> agent_id lookup
        self._env_to_agent = {}
        if allocation:
            for aid, insts in allocation.items():
                for iid in insts:
                    self._env_to_agent[iid] = aid

        # Grid layout
        cols = max(1, int(num_agents ** 0.5 + 0.5))
        self._cols = cols
        self._rows = (num_agents + cols - 1) // cols
        self._cell_w = 640
        self._cell_h = 360

        if _CV2:
            cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win_name,
                             self._cols * self._cell_w,
                             self._rows * self._cell_h)

    def _label_for(self, env_idx: int) -> str:
        agent_id = self._env_to_agent.get(env_idx, env_idx)
        return f"Agent {agent_id} | Env {env_idx}"

    def update_frame(self, instance_id: int, frame_bgr: np.ndarray):
        with self._lock:
            self._frames[instance_id] = frame_bgr

    def render(self):
        if not _CV2:
            return
        blank = np.zeros((self._cell_h, self._cell_w, 3), dtype=np.uint8)
        rows = []
        for r in range(self._rows):
            row_frames = []
            for c in range(self._cols):
                idx = r * self._cols + c
                label = self._label_for(idx)
                if idx < self._num_agents:
                    with self._lock:
                        f = self._frames.get(idx)
                    if f is not None and f.size > 0:
                        cell = cv2.resize(f, (self._cell_w, self._cell_h))
                    else:
                        cell = blank.copy()
                        cv2.putText(cell, f"{label} - waiting",
                                    (20, self._cell_h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
                else:
                    cell = blank.copy()
                cv2.rectangle(cell, (0, 0),
                              (self._cell_w - 1, self._cell_h - 1), (40, 40, 40), 2)
                cv2.putText(cell, label, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                row_frames.append(cell)
            rows.append(np.hstack(row_frames))
        grid = np.vstack(rows)
        cv2.imshow(self._win_name, grid)
        cv2.waitKey(1)

    def close(self):
        if _CV2:
            try:
                cv2.destroyWindow(self._win_name)
            except Exception:
                pass


# Module-level singleton — set by train/environment.py before env creation
_surveillance_win: "SurveillanceWindow | None" = None


# ======================================================================
#  Display mixin (mixed into MonsterHunterEnv)
# ======================================================================

class DisplayMixin:
    """Real-time display methods for MonsterHunterEnv."""

    def _display_rt_vision(self, observation: dict):
        """
        Show the current raw frame in an OpenCV window.
        Multi-instance mode routes to the shared SurveillanceWindow.
        """
        if not _CV2:
            return
        try:
            if not self.frame_capture:
                return
            frame = self.frame_capture.capture_frame()
            if frame is None or frame.size == 0 or frame.mean() < 5:
                return

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Overlay basic stats
            memory = observation.get('memory')
            if memory is not None:
                lines = [
                    f"Agent {self._agent_id} | Env {self.instance_id}",
                    f"Step: {self.total_steps}",
                    f"HP: {memory[0]:.0f}",
                    f"Zone: {int(memory[7])}",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(frame_bgr, text, (10, 24 + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

            # Multi-instance → surveillance grid
            global _surveillance_win
            if _surveillance_win is not None:
                _surveillance_win.update_frame(self.instance_id, frame_bgr)
                if self.instance_id == 0:
                    _surveillance_win.render()
                return

            # Single-instance → own window
            if self.rt_window_name:
                cv2.imshow(self.rt_window_name, frame_bgr)
                cv2.waitKey(1)

        except Exception as exc:
            if self.total_steps % 1000 == 0:
                logger.error(f"rt-vision display error: {exc}")

    def _display_rt_minimap_debug(self, observation: dict):
        """
        Full debug layout with visual frame + 4 exploration map channels + stats.

        Layout:
        ┌──────────┬──────────┬──────────┐
        │  Visual   │  Ch0     │  Ch1     │
        ├──────────┼──────────┼──────────┤
        │  Ch2     │  Ch3     │  Stats   │
        └──────────┴──────────┴──────────┘
        """
        if not _CV2 or self.rt_window_name is None:
            return

        try:
            visual = observation['visual']
            memory = observation['memory']
            emap = observation['exploration_map']

            # Prepare visual panel (84x84 → 300x300)
            if self.preprocessor.grayscale:
                last = (visual[:, :, -1] * 255).astype(np.uint8)
                vis_bgr = cv2.cvtColor(last, cv2.COLOR_GRAY2BGR)
            else:
                vis_bgr = (visual[:, :, -3:] * 255).astype(np.uint8)
            vis_panel = cv2.resize(vis_bgr, (300, 300), interpolation=cv2.INTER_NEAREST)

            # Prepare channel panels
            colormaps = [
                cv2.COLORMAP_VIRIDIS,
                cv2.COLORMAP_HOT,
                cv2.COLORMAP_OCEAN,
                cv2.COLORMAP_JET,
            ]
            ch_panels = []
            for i, cmap in enumerate(colormaps):
                ch = emap[:, :, i]
                lo, hi = ch.min(), ch.max()
                normed = ((ch - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else (ch * 0).astype(np.uint8)
                colored = cv2.applyColorMap(normed, cmap)
                panel = cv2.resize(colored, (300, 300), interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel, f"Ch{i}: {lo:.2f}-{hi:.2f}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                ch_panels.append(panel)

            # Stats panel
            stats_panel = np.zeros((300, 300, 3), dtype=np.uint8)
            lines = [
                f"Step: {self.total_steps}",
                f"Episode: {self.episode_count}",
                "",
                f"HP: {memory[0]:.0f}/100",
                f"Stamina: {memory[2]:.0f}/100",
                f"Zone: {int(memory[7])}",
                f"Deaths: {int(memory[10])}",
                "",
                f"Time: {int(memory[61])}s",
                f"Monsters: {int(memory[63])}",
                f"Sharpness: {int(memory[65])}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(stats_panel, line, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Assemble grid
            row1 = np.hstack([vis_panel, ch_panels[0], ch_panels[1]])
            row2 = np.hstack([ch_panels[2], ch_panels[3], stats_panel])
            grid = np.vstack([row1, row2])

            cv2.imshow(self.rt_window_name, grid)
            cv2.waitKey(1)

        except (AttributeError, KeyError, IndexError, ValueError) as exc:
            if self.total_steps % 1000 == 0:
                logger.error(f"Minimap debug display error: {exc}")
