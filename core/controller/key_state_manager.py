"""
key_state_manager.py — Hold / release diff engine.

Instead of tap-and-release every step (which produces jerky movement),
this manager tracks which keys are currently held and only sends
press/release commands for CHANGES between steps.

If the agent outputs "forward" for 200 consecutive steps, the 'w' key
stays pressed the entire time with zero interruption.

Example:
    Step N:   desired = {w, shift}        currently_held = {w, up}
              → release 'up', press 'shift'   (w stays held)
    Step N+1: desired = {w, shift, 4}     currently_held = {w, shift}
              → press '4'                      (w and shift stay held)
    Step N+2: desired = {}                 currently_held = {w, shift, 4}
              → release all three
"""

from info.module_logger import get_module_logger

logger = get_module_logger('key_state_mgr')


class KeyStateManager:
    """
    Manages the diff between desired key state and current key state.

    Requires a press/release callback pair that talks to the DInput
    shared memory buffer.
    """

    def __init__(self, press_fn, release_fn):
        """
        Args:
            press_fn:   callable(key_name: str) — sets DIK byte to 1
            release_fn: callable(key_name: str) — sets DIK byte to 0
        """
        self._press = press_fn
        self._release = release_fn
        self._currently_held: set = set()

    @property
    def currently_held(self) -> frozenset:
        """Read-only view of keys currently held down."""
        return frozenset(self._currently_held)

    def apply(self, desired_keys: set):
        """
        Transition from current state to desired state.

        Only keys that CHANGE are touched:
            - Keys in current but NOT in desired → released
            - Keys in desired but NOT in current → pressed
            - Keys in both → untouched (stay held)

        Args:
            desired_keys: set of key name strings to hold this step.
                          Empty set = release everything.
        """
        to_release = self._currently_held - desired_keys
        to_press   = desired_keys - self._currently_held

        # Release first (avoid ghost combos from overlapping keys)
        for key in to_release:
            try:
                self._release(key)
            except Exception as exc:
                logger.error(f"Failed to release key '{key}': {exc}")

        # Then press new keys
        for key in to_press:
            try:
                self._press(key)
            except Exception as exc:
                logger.error(f"Failed to press key '{key}': {exc}")

        self._currently_held = set(desired_keys)

    def release_all(self):
        """Release every currently held key and clear internal state."""
        for key in list(self._currently_held):
            try:
                self._release(key)
            except Exception as exc:
                logger.error(f"Failed to release key '{key}' during release_all: {exc}")
        self._currently_held.clear()

    def force_sync(self, all_managed_keys: set):
        """
        Safety: release ALL managed keys at the hardware level,
        regardless of internal tracking state. Used on episode reset
        to guarantee a clean slate.

        Args:
            all_managed_keys: every key name the system can control.
        """
        for key in all_managed_keys:
            try:
                self._release(key)
            except Exception:
                pass
        self._currently_held.clear()
