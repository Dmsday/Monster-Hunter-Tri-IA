"""
mh_env.py — Gymnasium environment for Monster Hunter Tri RL training.

This is the main entry point. The class delegates to focused mixins:
    - EpisodeMixin          (episode_mixin.py)     — reset, reload, termination
    - ObservationMixin      (observation_mixin.py)  — observation building
    - RewardBridgeMixin     (reward_bridge_mixin.py)— reward calculation bridge
    - DisplayMixin          (realtime_display.py)   — OpenCV real-time display
    - CaptureMixin          (capture_mixin.py)      — async frame capture thread
"""

import time
import queue
import threading
import traceback
from typing import Dict, Optional, Union, List

import gymnasium as gym

# Optional visualization imports
try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    cv2 = None

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    _MPL = True
except ImportError:
    _MPL = False
    plt = None

# Pynput for save state reload fallback
try:
    from pynput.keyboard import Controller as KeyboardController, Key
    _PYNPUT = True
except ImportError:
    _PYNPUT = False

# ============================================================================
# Project modules
# ============================================================================
from core.dynamic_memory_reader import MemoryReader
from core.state_fusion import StateFusion
from core.controller import WiiController, find_dolphin_pid
from info.agent_context import AgentContext, EnvContext
from info.module_logger import get_module_logger
from vision.frame_capture import FrameCapture
from vision.preprocessing import FramePreprocessor
from reward.reward_calculator import MonsterHunterRewardCalculator

# ============================================================================
# Sub-modules (spaces, sanitizer, mixins)
# ============================================================================
from environment.spaces import build_action_space, build_observation_space
from environment.sanitizer import sanitize_info
from environment.episode_mixin import EpisodeMixin
from environment.observation_mixin import ObservationMixin
from environment.reward_bridge_mixin import RewardBridgeMixin
from environment.realtime_display import DisplayMixin
from environment.capture_mixin import CaptureMixin

logger = get_module_logger('mh_env')


class MonsterHunterEnv(
    EpisodeMixin,
    ObservationMixin,
    RewardBridgeMixin,
    DisplayMixin,
    CaptureMixin,
    gym.Env,
):
    """
    Hybrid RL environment for Monster Hunter Tri.

    Supports vision (CNN), memory (game state vector), and exploration map
    modalities. Designed for single-instance and multi-instance training
    with Dolphin emulator.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        use_vision=True,
        use_memory=True,
        frame_size=(84, 84),
        grayscale=False,
        frame_stack=4,
        action_repeat=4,
        render_mode=None,
        use_controller=True,       # Always True — DLL injection (legacy param kept for compat)
        controller_debug=False,
        use_advanced_rewards=True,
        auto_reload_save_state=True,
        save_state_slot=5,
        rt_vision=False,
        rt_minimap=False,
        instance_id=0,
        agent_id=None,
        disabled_heads=None,        # list of head names to disable, e.g. ['menu']
    ):
        super().__init__()

        # Store for _init_controller
        self._disabled_heads = disabled_heads if disabled_heads is not None else ['menu']

        # --- Identity ---
        self.instance_id = instance_id
        self._agent_id = agent_id if agent_id is not None else instance_id
        self._multi_instance_mode = (instance_id > 0)

        # -- Isolation flag: set to True if save state is unrecoverable --
        self._isolated = False

        # -- Dolphin base directory for cross-user save state recovery --
        self.dolphin_base_dir: str | None = None

        AgentContext.set_current_agent(self._agent_id)
        EnvContext.set_current_env(instance_id)

        # --- Feature flags ---
        self.use_vision = use_vision
        self._vision_available = use_vision  # Tracks hardware success
        self.use_memory = use_memory
        self.action_repeat = action_repeat
        self.render_mode = render_mode
        self.use_controller = use_controller
        self.auto_reload_save_state = auto_reload_save_state
        self.save_state_slot = save_state_slot
        self.rt_vision = rt_vision
        self.rt_minimap = rt_minimap

        # --- Real-time display windows ---
        self.rt_window_name: Optional[str] = None
        self._init_rt_windows()

        logger.info("=" * 70)
        logger.info("Initializing Monster Hunter environment")
        logger.info("=" * 70)

        # --- Keyboard for save state (pynput fallback) ---
        if self.auto_reload_save_state and _PYNPUT:
            self.keyboard = KeyboardController()
            self.save_state_reload_count = 0
        elif self.auto_reload_save_state:
            logger.warning("pynput missing — auto-reload may be limited")
            self.save_state_reload_count = 0

        # --- Initialize subsystems (order matters) ---
        self.frame_capture = None
        self.preprocessor = None
        self.memory = None
        self.state_fusion = None

        # 1. Memory reader (needed by state_fusion and rewards)
        self._init_memory_reader()

        # 2. Vision (frame capture + preprocessing)
        if use_vision:
            self._init_vision(frame_size, grayscale, frame_stack)

        # 3. Reward calculator
        self.reward_calc = None
        if use_advanced_rewards:
            self._init_reward_calculator()

        # 4. State fusion (needs vision + memory)
        self._init_state_fusion()

        # 5. Controller (DLL injection)
        self._init_controller(controller_debug)

        # --- Log multi-instance info ---
        if self.instance_id > 0:
            self._log_instance_info()

        # --- Build spaces ---
        self.action_space = build_action_space()
        self.observation_space = build_observation_space(
            use_vision=use_vision,
            use_memory=use_memory,
            has_memory_reader=(self.memory is not None),
            frame_size=frame_size,
            grayscale=grayscale,
            frame_stack=frame_stack,
        )

        # --- Episode state ---
        self.current_state = None
        self.prev_raw_memory = None
        self.episode_start_time = None
        self.episode_steps = 0
        self.total_steps = 0
        self.episode_count = 0
        self.total_reward = 0
        self._first_reset_done = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self._episode_ending = False

        # --- Async capture thread state ---
        self._obs_queue = queue.Queue(maxsize=8)
        self._capture_thread = None
        self._capture_running = False
        self._capture_lock = threading.Lock()
        self._frames_dropped = 0
        self._frames_captured = 0
        self._frames_consumed = 0

        logger.info("Environment ready!")

    # ==================================================================
    # step
    # ==================================================================

    def step(self, action):
        """
        Execute one action and return (obs, reward, terminated, truncated, info).
        """
        AgentContext.set_current_agent(self._agent_id)
        EnvContext.set_current_env(self.instance_id)

        self.episode_steps += 1
        self.total_steps += 1

        # --- Early exit: episode already ending ---
        if self._episode_ending:
            logger.warning("Episode already ending — step() ignored")
            obs = self.current_state or self._get_dummy_observation()
            return obs, 0.0, True, False, {'episode_already_ending': True}

        # --- Check for reward screen before action ---
        if self._is_on_reward_screen():
            self._episode_ending = True
            obs = self.current_state or self._get_dummy_observation()
            return obs, 0.0, True, False, {
                'quest_ended_before_action': True,
                'current_map': 45,
            }

        # --- Execute action ---
        self._execute_action(action)

        # --- Observation ---
        observation = self._get_observation()

        # Real-time display
        if isinstance(observation, dict) and (self.rt_vision or self.rt_minimap):
            if self.rt_minimap and self.rt_vision:
                self._display_rt_minimap_debug(observation)
            elif self.rt_vision:
                self._display_rt_vision(observation)

        # --- Reward ---
        if self.reward_calc and self.memory and self.episode_steps == 1:
            self.prev_raw_memory = None

        # Use resolved action (after compatibility masking) for reward calc
        resolved = getattr(self, '_last_resolved_action', action)
        reward, step_info = self._calculate_reward(resolved)

        # --- Check forced termination (3 deaths, time expired) ---
        terminated, end_reason = self._check_forced_termination(step_info)
        if terminated:
            return self._build_terminal_response(
                observation, reward, step_info, end_reason
            )

        # --- Normal continuation ---
        step_info['episode_num'] = int(self.episode_count)
        step_info['episode_steps'] = int(self.episode_steps)
        step_info['total_steps'] = int(self.total_steps)
        step_info['total_reward'] = self.total_reward

        self.episode_reward += reward
        self.episode_length += 1
        self.total_reward += reward

        # Check standard termination/truncation
        done = self._check_terminated(observation)
        truncated = self._check_truncated()

        if done or truncated:
            step_info['episode'] = {
                'r': float(self.episode_reward),
                'l': int(self.episode_length),
                't': float(time.time() - self.episode_start_time),
            }

        self.current_state = observation
        step_info = sanitize_info(step_info)

        return observation, reward, done, truncated, step_info

    # ==================================================================
    # Info / getters
    # ==================================================================

    def _get_info(self) -> dict:
        """Build the info dict for reset()."""
        info: Dict[str, Union[int, float, bool, str, List, Dict]] = {
            'episode_num': self.episode_count,
            'episode_steps': self.episode_steps,
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
        }

        if self.use_memory and self.memory:
            try:
                state = self.memory.read_game_state()
                info['hp'] = state.get('player_hp')
                info['stamina'] = state.get('player_stamina')
                info['death_count'] = state.get('death_count')
                info['current_zone'] = state.get('current_zone')
                info['player_x'] = state.get('player_x')
                info['player_y'] = state.get('player_y')
                info['player_z'] = state.get('player_z')
                info['orientation'] = state.get('player_orientation')
                info['money'] = state.get('money')
                info['quest_time'] = state.get('quest_time')
                info['sharpness'] = state.get('sharpness')
                info['inventory'] = self.memory.read_inventory()

                if self.reward_calc:
                    info.update(self.reward_calc.get_stats())
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        if self.auto_reload_save_state:
            info['save_state_reload_count'] = self.save_state_reload_count

        return info

    def render(self):
        if self.render_mode == "rgb_array" and self._vision_available:
            try:
                return self.frame_capture.capture_frame()
            except (AttributeError, RuntimeError):
                return None
        return None

    def get_controller(self):
        """Return the WiiController instance (used for HidHide config)."""
        return getattr(self, 'controller', None)

    def get_frame_capture(self):
        """Return the FrameCapture instance (used for cleanup)."""
        if hasattr(self, 'frame_capture') and self.frame_capture is not None:
            return self.frame_capture
        return None

    def get_window_title(self) -> str:
        """Return the title of the captured Dolphin window."""
        if self.frame_capture and self.frame_capture.hwnd:
            try:
                import win32gui
                return win32gui.GetWindowText(self.frame_capture.hwnd)
            except (OSError, AttributeError, ValueError):
                return ""
        return ""

    # ==================================================================
    # close
    # ==================================================================

    def close(self):
        """Clean up all resources."""
        AgentContext.set_current_agent(self._agent_id)
        EnvContext.set_current_env(self.instance_id)

        # Suppress frame_capture warnings during shutdown
        import logging as _logging
        _fc_logger = _logging.getLogger('mh_frame_capture')
        _prev_level = _fc_logger.level
        _fc_logger.setLevel(_logging.CRITICAL)

        # Stop capture thread
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        # Clean up frame capture GDI/DLL objects
        if self.frame_capture:
            self.frame_capture.close()

        _fc_logger.setLevel(_prev_level)

        # Controller cleanup
        if self.controller:
            try:
                self.controller.cleanup()
                self.controller = None
            except Exception as exc:
                logger.error(f"Controller cleanup error: {exc}")

        if self.auto_reload_save_state:
            logger.info(f"Save states reloaded: {self.save_state_reload_count} times")

        # Close OpenCV windows
        if self.rt_vision and self.rt_window_name and _CV2:
            try:
                cv2.destroyWindow(self.rt_window_name)
                cv2.waitKey(1)
            except Exception:
                pass

        # Close matplotlib windows
        if self.rt_minimap and self.rt_minimap_fig and _MPL:
            try:
                plt.close(self.rt_minimap_fig)
            except Exception:
                pass

        logger.info("Environment closed cleanly")
        EnvContext.clear()

    # ==================================================================
    # Private: initialization helpers
    # ==================================================================

    def _init_rt_windows(self):
        """Set up OpenCV and matplotlib windows for real-time display."""
        if self.rt_vision:
            if not _CV2:
                logger.warning("OpenCV unavailable — rt-vision disabled")
                self.rt_vision = False
            else:
                try:
                    self.rt_window_name = "AI Vision - Real Time"
                    cv2.namedWindow(self.rt_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.rt_window_name, 1200, 800)
                except Exception as exc:
                    logger.error(f"RT window init error: {exc}")
                    self.rt_vision = False
                    self.rt_window_name = None

        self.rt_minimap_window = None
        self.rt_minimap_fig = None
        self.rt_minimap_ax = None

        if self.rt_minimap and self.rt_vision:
            if not _MPL:
                logger.warning("Matplotlib unavailable — minimap disabled")
                self.rt_minimap = False
            else:
                try:
                    plt.ion()
                    self.rt_minimap_fig = plt.figure(figsize=(8, 6))
                    self.rt_minimap_ax = self.rt_minimap_fig.add_subplot(
                        111, projection='3d'
                    )
                    self.rt_minimap_window = True
                except Exception as exc:
                    logger.error(f"Minimap init error: {exc}")
                    self.rt_minimap = False

    def _init_memory_reader(self):
        """Initialize the Dolphin memory reader."""
        try:
            target_pid = find_dolphin_pid(f"MHTri-{self.instance_id}")
        except (ValueError, Exception):
            target_pid = None
            logger.warning(
                f"Instance #{self.instance_id}: could not resolve Dolphin PID "
                f"— will connect to first Dolphin found"
            )

        try:
            self.memory = MemoryReader(
                force_quest_mode=True,
                async_mode=True,
                read_frequency=100,
                target_pid=target_pid,
                instance_id=self.instance_id,
            )
            if self.memory is None:
                raise RuntimeError("MemoryReader returned None")

            if not self.use_memory:
                logger.info("Memory not used by agent, but active for rewards")

        except Exception as exc:
            logger.error(f"CRITICAL: MemoryReader unavailable: {exc}")
            self.memory = None
            raise RuntimeError(
                f"MemoryReader failed: {exc}. "
                "Check Dolphin is running as Administrator with MH Tri loaded."
            )

    def _init_vision(self, frame_size, grayscale, frame_stack):
        """Initialize frame capture and preprocessor."""
        try:
            force_pw = self.rt_vision or self.rt_minimap
            expected_title = (
                f"MHTri-{self.instance_id}" if self.instance_id > 0 else None
            )

            self.frame_capture = FrameCapture(
                target_fps=30,
                force_printwindow=force_pw,
                instance_id=self.instance_id,
                expected_window_title=expected_title,
                use_dll=True,
            )

            # Verify capture with brightness retries
            self._verify_frame_capture()

            self.preprocessor = FramePreprocessor(
                target_size=frame_size,
                grayscale=grayscale,
                frame_stack=frame_stack,
            )
            logger.info("Vision initialized successfully")

            # Verify window title
            self._verify_window_title()

        except Exception as exc:
            logger.error(f"CRITICAL: Vision init failed: {exc}")
            logger.error("FALLBACK: Training continues with MEMORY ONLY")
            self._vision_available = False
            self.frame_capture = None
            self.preprocessor = None

    def _verify_frame_capture(self):
        """Retry frame capture until a non-black frame is obtained."""
        max_retries = 20
        delay = 2.0 if (self._multi_instance_mode or self.instance_id > 0) else 1.0

        # Skip the first second: MH loading screen is always black,
        # avoids guaranteed "black frame" warnings at every training start
        time.sleep(1.5)

        for attempt in range(1, max_retries + 1):
            frame = self.frame_capture.capture_frame()
            if frame is None or frame.size == 0:
                logger.warning(
                    f"Frame capture attempt {attempt}/{max_retries}: "
                    f"invalid frame — retrying in {delay}s"
                )
                if attempt < max_retries:
                    time.sleep(delay)
                continue

            brightness = frame.mean()
            if brightness >= 5:
                logger.debug(
                    f"Frame OK on attempt {attempt} (brightness: {brightness:.1f})"
                )
                return

            logger.warning(
                f"Frame capture attempt {attempt}/{max_retries}: "
                f"black frame (brightness: {brightness:.1f})"
            )
            if attempt < max_retries:
                time.sleep(delay)

        raise RuntimeError(
            f"Frame capture failed after {max_retries} attempts"
        )

    def _verify_window_title(self):
        """Log a warning if the captured window doesn't match expected title."""
        if not self.frame_capture:
            return
        try:
            title = self.get_window_title()
            expected = f"MHTri-{self.instance_id}"
            if title and expected not in title:
                logger.warning(f"Window mismatch: expected '{expected}', got '{title}'")
                logger.warning("RISK OF COLLISION BETWEEN INSTANCES")
        except Exception:
            pass

    def _init_reward_calculator(self):
        """Create and attach the reward calculator."""
        self.reward_calc = MonsterHunterRewardCalculator()
        logger.debug("Reward calculator enabled")

        if self.memory is not None:
            self.memory.reward_calc = self.reward_calc

            if hasattr(self.reward_calc, 'exploration_tracker'):
                tracker = self.reward_calc.exploration_tracker
                total = sum(len(c) for c in tracker.cubes_by_zone.values())
                logger.info(
                    f"Exploration tracker: {total} cubes in "
                    f"{len(tracker.cubes_by_zone)} zones (both must be 0)"
                )
        else:
            raise RuntimeError(
                "MemoryReader is None — rewards require memory. "
                "Ensure Dolphin is running with MH Tri loaded."
            )

    def _init_state_fusion(self):
        """Initialize vision+memory state fusion."""
        if self.use_vision and self.memory is not None and self.preprocessor is not None:
            try:
                self.state_fusion = StateFusion(self.memory, self.preprocessor)
                logger.info("State fusion initialized")
            except Exception as exc:
                logger.error(f"State fusion error: {exc}")
                self.state_fusion = None
        else:
            self.state_fusion = None

    def _init_controller(self, debug: bool):
        """Initialize the DLL-injected Wii controller."""
        try:
            self.controller = WiiController(
                instance_id=self.instance_id,
                debug=debug,
                disabled_heads=self._disabled_heads,
            )
            if self.controller.is_connected:
                mode = (
                    "WGI gamepad"
                    if getattr(self.controller, '_use_wgi', False)
                    else "DInput keyboard"
                )
                logger.info(f"Controller initialized (DLL, mode={mode})")
            else:
                logger.warning("Controller not connected (DLL injection failed)")
                self.controller = None
        except Exception as exc:
            logger.error(f"Controller unavailable: {exc}")
            self.controller = None

    def _log_instance_info(self):
        """Log configuration for multi-instance environments."""
        logger.info("=" * 70)
        logger.info(f"ENVIRONMENT INSTANCE #{self.instance_id}")
        logger.info(f"  Vision: {self.use_vision}")
        logger.info(f"  Memory: {self.use_memory}")
        logger.info(f"  Controller: DLL injection (focus-free)")
        logger.info(f"  RT-vision: {self.rt_vision}")
        if self.frame_capture:
            title = self.get_window_title()
            if title:
                logger.info(f"  Window: '{title}'")
        logger.info("=" * 70)

    # ==================================================================
    # Private: step helpers
    # ==================================================================

    def _is_on_reward_screen(self) -> bool:
        """Quick check if the game is on the reward screen (MAP=45)."""
        if not (self.use_memory and self.memory):
            return False
        try:
            state = self.memory.read_game_state()
            return state.get('current_map') == 45
        except (AttributeError, KeyError, RuntimeError):
            return False

    STEP_DURATION_SINGLE = 0.080  # ~5 frames — single instance
    STEP_DURATION_MULTI = 0.033  # ~2 frames — multi instance

    def _execute_action(self, action):
        """Send multi-head action vector to the controller."""
        if self.controller is None:
            return
        try:
            if (self.frame_capture is not None
                    and getattr(self.frame_capture, '_shutdown', False)):
                time.sleep(0.016)
                return

            # Read menu state from memory for action masking
            menu_open = False
            if self.use_memory and self.memory:
                try:
                    state = self.memory.read_game_state()
                    menu_open = state.get('in_game_menu', False)
                except Exception:
                    pass

            dt = self.STEP_DURATION_MULTI if self._multi_instance_mode else self.STEP_DURATION_SINGLE
            self._last_resolved_action, _ = self.controller.execute_action(
                action,
                menu_open=menu_open,
                step_duration=dt,
            )
        except Exception as exc:
            logger.error(f"Action {action} execution error: {exc}")

    def _check_forced_termination(self, step_info: dict):
        """
        Check if the episode must end due to 3 deaths or time expired.

        Returns:
            (True, reason_str) if forced termination, (False, None) otherwise.
        """
        death_count = step_info.get('death_count', 0) or 0
        quest_time = step_info.get('quest_time', 5400)

        if death_count >= 3:
            logger.info(f"3 DEATHS DETECTED (count={death_count})")
            return True, 'three_deaths'

        if quest_time is not None and quest_time <= 1:
            logger.info("TIME EXPIRED")
            return True, 'time_expired'

        return False, None

    def _build_terminal_response(self, observation, reward, step_info, end_reason):
        """Build the return tuple for a forced episode termination."""
        self._episode_ending = True
        self.episode_reward += reward
        self.episode_length += 1

        step_info['episode'] = {
            'r': float(self.episode_reward),
            'l': int(self.episode_length),
            't': float(time.time() - self.episode_start_time),
        }
        step_info['end_reason'] = end_reason
        step_info['forced_termination'] = True

        return observation, reward, True, False, sanitize_info(step_info)


# ======================================================================
# Standalone test
# ======================================================================
if __name__ == "__main__":
    print("TEST ENVIRONMENT\n")
    try:
        env = MonsterHunterEnv(
            use_vision=True,
            use_memory=True,
            grayscale=False,
            frame_stack=4,
            use_controller=True,
            use_advanced_rewards=True,
            auto_reload_save_state=True,
        )
        print(f"Actions: {env.action_space}")
        obs, info = env.reset()
        print("Test passed!")
        env.close()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
