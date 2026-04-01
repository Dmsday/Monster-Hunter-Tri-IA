"""
wii_controller.py — Focus-free Dolphin input controller (multi-head edition).

Replaces the old single-action tap-and-release controller with a multi-head
hold/release system. The agent outputs a vector of branch choices (one per
head), and the controller:

    1. Resolves conflicts via ActionResolver (compatibility + menu state)
    2. Computes the key diff via KeyStateManager (hold/release)
    3. Waits for step_duration before the next agent decision

Keys that don't change between steps stay held — no interruption.
This produces smooth, continuous movement instead of jerky single-frame taps.

Save state injection (F1-F8) still uses PostMessage to the render window,
completely independent of the DInput key injection path.

Public API:
    execute_action(action_vector, menu_open, step_duration)  — multi-head
    execute_legacy_action(action_id, frames)                 — backward compat
    send_save_state_key(key_name)
    send_raw_key(key_name, duration)
    release_all_managed()
    reset_all()
    cleanup()
"""

import ctypes
import time

import win32api
import win32con
import win32gui

from info.module_logger import get_module_logger
from core.controller.constants import (
    DIK, VK_FKEYS, ALL_MANAGED_KEYS,
    FILE_MAP_ALL_ACCESS, SHARED_MEM_SIZE,
    MOUSE_LEFT, MOUSE_RIGHT,            # Mouse button shared memory offsets
)
from core.controller.dll_utils import (
    ensure_dll, find_dolphin_pid, inject_dll,
    open_shared_memory, close_shared_memory, is_dll_already_injected,
    kernel32,
)
from core.controller.action_resolver import ActionResolver
from core.controller.action_heads import HEAD_NAME_TO_IDX
from core.controller.key_state_manager import KeyStateManager

logger = get_module_logger('controller')


# Keys that must be tapped (press + release) every step they are active.
# Combat and item keys need a rising edge each step — holding them causes
# the game to ignore subsequent frames
EPHEMERAL_KEYS: frozenset = frozenset({
    'mouse_left',   # Attack (left click) — one tap = one swing | menu confirm
    'mouse_right',  # Dodge/evade (right click) — one tap = one roll | menu back
    'e',            # Open / close menu — one tap
    'q',            # Use selected item (a key on AZERTY) — one tap = one use
    'p',            # Draw / sheath weapon — one tap (each press toggles state)
    'm',            # Kick attack | menu zoom+close — one tap
})

class WiiController:
    """
    Focus-free Dolphin controller — multi-head hold/release via DInput injection.

    On construction:
        1. Auto-locates dolphin_input_hook.dll (builds if not found).
        2. Finds the Dolphin render window for this instance_id (MHTri-N).
        3. Injects the DLL if not already present.
        4. Opens the per-PID shared memory channel.
        5. Initializes ActionResolver + KeyStateManager.

    Two injection paths:
        DInput (shared memory bytes 0-255): in-game inputs, polled every frame.
        PostMessage WM_KEYDOWN → render window: save state hotkeys (F1-F8).
    """

    def __init__(
        self,
        instance_id:    int  = 0,
        debug:          bool = False,
        dll_path:       str  = None,
        use_controller: bool = True,   # legacy param, accepted but ignored
        disabled_heads: list = None,  # list of head name strings, e.g. ['menu']
    ):
        self.instance_id  = instance_id
        self.debug        = debug
        self.is_connected = False
        self._map_ptr     = None
        self.buf          = None
        self._handle      = None
        self._pid         = None
        self._render_hwnd = None

        # Multi-head subsystems (initialized after connection)
        self._key_manager = None  # Created after buf is available

        # Build frozenset of disabled head indices from names
        _names = disabled_heads if disabled_heads is not None else ['menu']
        _disabled = frozenset(
            HEAD_NAME_TO_IDX[n] for n in _names if n in HEAD_NAME_TO_IDX
        )
        self._resolver = ActionResolver(disabled_heads=_disabled)

        if debug:
            logger.info(f"Controller #{instance_id}: debug mode (no injection)")
            self.is_connected = True
            self._key_manager = KeyStateManager(
                press_fn=lambda k: None,
                release_fn=lambda k: None,
            )
            return

        # ── Step 1: Find or auto-build DLL ──
        try:
            resolved_dll = ensure_dll(dll_path)
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"Controller #{instance_id}: {e}")
            return

        # ── Step 2: Find render window (MHTri-N) → PID ──
        try:
            pid = find_dolphin_pid(f"MHTri-{instance_id}")
        except ValueError as e:
            logger.error(f"Controller #{instance_id}: {e}")
            return
        self._pid = pid

        # Store render window HWND for save state PostMessage
        render_hwnd = win32gui.FindWindow(None, f"MHTri-{instance_id}")
        if render_hwnd and win32gui.IsWindowVisible(render_hwnd):
            self._render_hwnd = render_hwnd
            logger.debug(
                f"Controller #{instance_id}: render HWND={render_hwnd} stored")
        else:
            logger.warning(
                f"Controller #{instance_id}: render window 'MHTri-{instance_id}' "
                f"not found at init — will retry later")

        # ── Step 3: Inject DLL (idempotent) ──
        if is_dll_already_injected(pid):
            logger.info(f"Controller #{instance_id}: DLL already injected (PID {pid})")
        else:
            try:
                inject_dll(pid, resolved_dll)
                time.sleep(0.5)  # Wait for DLL init
            except Exception as e:
                logger.error(f"Controller #{instance_id}: injection failed: {e}")
                return

        # ── Step 4: Open shared memory ──
        try:
            self._handle, self._map_ptr, self.buf = open_shared_memory(pid)
        except (TimeoutError, OSError) as e:
            logger.error(f"Controller #{instance_id}: {e}")
            return

        # ── Step 5: Initialize key state manager with real press/release ──
        self._key_manager = KeyStateManager(
            press_fn=self._press,
            release_fn=self._release,
        )

        self.is_connected = True
        logger.info(
            f"Controller #{instance_id}: ready "
            f"(PID={pid}, shm=0x{self._map_ptr:X}, "
            f"render_hwnd={self._render_hwnd}, mode=multi-head hold/release)")

    # =========================================================================
    # Low-level DInput key injection
    # =========================================================================

    def _set_key(self, dik: int, pressed: bool):
        """
        Set a single byte in shared memory by index.
        Handles both keyboard DIK scan codes (0-255) and mouse buttons (256+).
        Previously used 'dik & 0xFF' which capped at 255 — mouse buttons were silently ignored.
        """
        if self.buf and 0 <= dik < SHARED_MEM_SIZE:
            self.buf[dik] = 1 if pressed else 0

    def _press(self, key: str):
        """
        Press a key or mouse button by name.
        Supports 'mouse_left' (A button) and 'mouse_right' (B button)
        in addition to standard DIK keyboard key names.
        """
        if key == 'mouse_left':
            self._set_key(MOUSE_LEFT, True)
        elif key == 'mouse_right':
            self._set_key(MOUSE_RIGHT, True)
        elif key in DIK:
            self._set_key(DIK[key], True)

    def _release(self, key: str):
        """
        Release a key or mouse button by name.
        Supports 'mouse_left' (A button) and 'mouse_right' (B button).
        """
        if key == 'mouse_left':
            self._set_key(MOUSE_LEFT, False)
        elif key == 'mouse_right':
            self._set_key(MOUSE_RIGHT, False)
        elif key in DIK:
            self._set_key(DIK[key], False)

    def _tap(self, key: str, duration: float):
        """Press, hold for duration, then release a single key."""
        self._press(key)
        time.sleep(duration)
        self._release(key)

    # =========================================================================
    # Multi-head action execution (PRIMARY API)
    # =========================================================================

    def execute_action(
        self,
        action_vector,
        *,
        menu_open: bool = False,
        step_duration: float = 0.080,
    ) -> tuple:
        """
        Execute a multi-head action vector using hold/release semantics.

        The resolver applies menu-state gating and compatibility masking.
        The key state manager computes the diff and only presses/releases
        keys that changed since the previous step.

        Args:
            action_vector:  array-like, shape (NUM_HEADS,).
                            One branch choice per head from MultiDiscrete.
            menu_open:      True if in-game menu is currently open.
            step_duration:  Seconds to hold before the next agent decision.

        Returns:
            (resolved_action, step_duration) — the resolved vector after
            masking, useful for reward calculation / logging.
        """
        if self.debug:
            resolved, keys = self._resolver.resolve(action_vector, menu_open=menu_open)
            logger.debug(
                f"[DEBUG] raw={list(action_vector)} → resolved={list(resolved)} "
                f"keys={keys} menu={menu_open}")
            time.sleep(step_duration)
            return resolved, step_duration

        if not self.is_connected or not self.buf or not self._key_manager:
            time.sleep(step_duration)
            import numpy as np
            return np.zeros_like(action_vector), step_duration

        # 1) Resolve: apply menu gate + compatibility masking
        resolved, desired_keys = self._resolver.resolve(
            action_vector, menu_open=menu_open)

        # 2) Separate ephemeral (tap-only) keys from persistent hold keys
        tap_keys = desired_keys & EPHEMERAL_KEYS
        hold_keys = desired_keys - EPHEMERAL_KEYS

        # 3) Apply hold keys through state manager (persist across steps)
        self._key_manager.apply(hold_keys)

        # 4) Tap ephemeral keys: press briefly then release within this step
        # Adaptive tap duration: 70 % of the step window, between 25 ms and 80 ms.
        # In multi-instance mode (step=33 ms) this gives ~23 ms (clamped to 25 ms).
        # In single-instance mode (step=80 ms) this gives 56 ms
        tap_duration = max(0.025, min(0.080, step_duration * 0.70))
        if tap_keys:
            for key in tap_keys:
                self._press(key)
            time.sleep(tap_duration)
            for key in tap_keys:
                self._release(key)
            remaining = max(0.0, step_duration - tap_duration)
            if remaining > 0:
                time.sleep(remaining)
        else:
            time.sleep(step_duration)

        return resolved, step_duration

    # =========================================================================
    # Full release (episode reset / cleanup)
    # =========================================================================

    def release_all_managed(self):
        """
        Release all keys managed by the multi-head system.
        Called on episode reset to ensure clean state.
        """
        if self._key_manager:
            self._key_manager.force_sync(ALL_MANAGED_KEYS)

    # =========================================================================
    # Legacy single-action API (backward compatibility)
    # =========================================================================

    def execute_legacy_action(self, action_id: int, frames: int = 10) -> int:
        """
        Execute a single discrete action (0-18) using the old tap-and-release
        paradigm. Provided for backward compatibility during migration.

        This does NOT use the hold/release system. Each call taps the key
        for `frames * 0.016` seconds, then releases it.

        Args:
            action_id: Action index 0-18 (old Discrete(19) space).
            frames:    Number of frames to hold the key.

        Returns:
            Number of frames.
        """
        duration = frames * 0.016

        if self.debug:
            logger.debug(f"[LEGACY] action={action_id} duration={duration:.3f}s")
            time.sleep(duration)
            return frames

        if not self.is_connected or not self.buf:
            time.sleep(duration)
            return frames

        try:
            self._legacy_dispatch(action_id, duration)
        except Exception as e:
            logger.error(f"Controller #{self.instance_id}: legacy action {action_id} error: {e}")
            time.sleep(duration)

        return frames

    def _legacy_dispatch(self, action_id: int, duration: float):
        """Map old action_id (0-18) to key taps. Exact replica of old behavior."""
        if   action_id == 0:  time.sleep(duration)
        elif action_id == 1:  self._tap('w',     duration)
        elif action_id == 2:  self._tap('s',     duration)
        elif action_id == 3:  self._tap('a',     duration)
        elif action_id == 4:  self._tap('d',     duration)
        elif action_id == 5:  self._tap('up',    duration)
        elif action_id == 6:  self._tap('down',  duration)
        elif action_id == 7:  self._tap('left',  duration)
        elif action_id == 8:  self._tap('right', duration)
        elif action_id == 9:  self._tap('4',     duration)
        elif action_id == 10: self._tap('1',     duration)
        elif action_id == 11: self._tap('3',     duration)
        elif action_id == 12: self._tap('0',     duration)
        elif action_id == 13: self._tap('e',     duration)
        elif action_id == 14: self._tap('ctrl',  duration)
        elif action_id == 15: self._tap('shift', duration)
        elif action_id == 16: self._tap('q',     duration)
        elif action_id == 17:
            self._press('shift'); time.sleep(0.18)
            self._tap('left', 0.06); time.sleep(0.06)
            self._release('shift')
        elif action_id == 18:
            self._press('shift'); time.sleep(0.18)
            self._tap('right', 0.06); time.sleep(0.06)
            self._release('shift')
        else:
            logger.debug(f"Unknown legacy action_id={action_id}")
            time.sleep(duration)

    # =========================================================================
    # Raw key / save state injection
    # =========================================================================

    def send_raw_key(self, key_name: str, duration: float = 0.1):
        """
        Send an arbitrary key via DInput shared memory (tap-and-release).
        Only for non-gameplay uses (e.g. save state workarounds).
        """
        if self.debug or not self.buf:
            logger.debug(f"[DEBUG] send_raw_key({key_name}, {duration:.2f}s)")
            time.sleep(duration)
            return

        if key_name not in DIK:
            logger.warning(f"send_raw_key: unknown key '{key_name}'")
            return

        self._tap(key_name, duration)

    def send_hotkey(self, key_name: str) -> bool:
        """Trigger a save state hotkey via pynput (legacy fallback)."""
        if self.debug:
            logger.debug(f"[DEBUG] send_hotkey({key_name})")
            return True

        key_lower = key_name.lower()
        if key_lower not in VK_FKEYS:
            logger.warning(f"Controller #{self.instance_id}: unknown hotkey '{key_name}'")
            return False

        try:
            from pynput.keyboard import Controller as KB, Key
            _kb = KB()
            fkey_map = {
                'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
                'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            }
            key = fkey_map[key_lower]
            _kb.press(key)
            time.sleep(0.1)
            _kb.release(key)
            logger.debug(
                f"Controller #{self.instance_id}: {key_name.upper()} sent via pynput")
            return True
        except Exception as e:
            logger.error(f"Controller #{self.instance_id}: pynput failed: {e}")
            return False

    def send_save_state_key(self, key_name: str) -> bool:
        """
        Trigger a Dolphin save state hotkey (F1-F8) via PostMessage
        to the render window (MHTri-N).
        """
        if self.debug:
            logger.debug(f"[DEBUG] send_save_state_key({key_name})")
            return True

        key_lower = key_name.lower()
        if key_lower not in VK_FKEYS:
            logger.warning(
                f"Controller #{self.instance_id}: unknown save state key '{key_name}' "
                f"— supported: {list(VK_FKEYS.keys())}")
            return False

        vk_code = VK_FKEYS[key_lower]

        # Lazy-init render HWND if not found at construction
        if self._render_hwnd is None:
            hwnd = win32gui.FindWindow(None, f"MHTri-{self.instance_id}")
            if hwnd and win32gui.IsWindowVisible(hwnd):
                self._render_hwnd = hwnd
                logger.debug(
                    f"Controller #{self.instance_id}: render window found "
                    f"(late init) HWND={hwnd}")
            else:
                logger.error(
                    f"Controller #{self.instance_id}: render window "
                    f"'MHTri-{self.instance_id}' not found — "
                    f"cannot send {key_name.upper()}")
                return False

        render_title = win32gui.GetWindowText(self._render_hwnd)
        logger.debug(
            f"Controller #{self.instance_id}: posting {key_name.upper()} "
            f"(VK=0x{vk_code:02X}) to '{render_title}' HWND={self._render_hwnd}")

        win32api.PostMessage(self._render_hwnd, win32con.WM_KEYDOWN, vk_code, 0)
        time.sleep(0.2)
        win32api.PostMessage(self._render_hwnd, win32con.WM_KEYUP,   vk_code, 0)

        logger.debug(
            f"Controller #{self.instance_id}: {key_name.upper()} posted")
        return True

    # =========================================================================
    # Cleanup
    # =========================================================================

    def reset_all(self):
        """Release all injected DInput keys (zero out shared memory)."""
        if self.debug or not self.buf:
            return
        for i in range(256):
            self.buf[i] = 0

    def cleanup(self):
        """Release shared memory handles and clear all key state."""
        if self.debug:
            return
        self.release_all_managed()
        self.reset_all()
        close_shared_memory(self._handle, self._map_ptr)
        self._map_ptr     = None
        self.buf          = None
        self._handle      = None
        self.is_connected = False
        logger.debug(f"Controller #{self.instance_id}: cleaned up")

    def __del__(self):
        try:
            close_shared_memory(self._handle, self._map_ptr)
            self._map_ptr = None
            self._handle  = None
        except Exception:
            pass
