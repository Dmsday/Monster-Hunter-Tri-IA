"""
capture_mixin.py — Asynchronous frame capture thread for the environment.

Runs a dedicated daemon thread that continuously captures and preprocesses
frames from Dolphin, depositing them in a bounded queue consumed by
_get_observation() in the main thread.

Mixin attributes expected on `self`:
    _agent_id, instance_id, _capture_running, _obs_queue,
    _vision_available, frame_capture, preprocessor,
    _frames_captured, _frames_dropped
"""

import time
import queue

from info.agent_context import AgentContext, EnvContext
from info.module_logger import get_module_logger

logger = get_module_logger('capture')


class CaptureMixin:
    """Provides the async capture loop for MonsterHunterEnv."""

    def _async_capture_loop(self):
        """
        Continuous frame capture running in a daemon thread.
        Captures a frame, preprocesses it, and pushes to the observation queue.
        Rate-limited to ~30 FPS.
        """
        AgentContext.set_current_agent(self._agent_id)
        EnvContext.set_current_env(self.instance_id)
        logger.info("Capture thread started")

        if self.frame_capture is None:
            logger.error(
                "Capture thread launched but frame_capture is None — "
                "this is expected in memory-only mode (use_vision=False)"
            )
            return

        while self._capture_running:
            try:
                # 1. Capture raw frame (~5 ms with reused GDI / DLL)
                frame = self.frame_capture.capture_frame()

                # 2. Preprocess + stack (~15 ms)
                if self._vision_available and self.preprocessor:
                    visual = self.preprocessor.process_and_stack_numpy(frame)
                else:
                    visual = None

                # 3. Enqueue (non-blocking)
                try:
                    self._obs_queue.put(visual, block=False)
                    self._frames_captured += 1
                except queue.Full:
                    self._frames_dropped += 1

                # Log drop rate only when significant
                if (self._frames_captured > 0
                        and self._frames_captured % 5000 == 0):
                    drop_pct = (self._frames_dropped / self._frames_captured) * 100
                    if drop_pct > 10.0:
                        logger.debug(f"Frame drop rate: {drop_pct:.1f}%")

                # 4. Rate limit (~30 FPS)
                time.sleep(1.0 / 30)

            except Exception as exc:
                logger.error(f"Capture thread error: {exc}")
                time.sleep(0.1)

        self._capture_running = False
        logger.debug("Capture thread stopped")
