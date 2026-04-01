"""
action_resolver.py — Resolves raw multi-head actions into a valid key set.

Pipeline:
    1. Menu state gate   — if menu is open, mask gameplay heads;
                           if menu is closed, mask menu navigation branches.
    2. Compatibility     — among remaining active heads, resolve conflicts
                           using priority (higher priority head wins).
    3. Key resolution    — map each surviving (head, branch) to its key(s).

The resolver never modifies the original action vector.
It returns a new "resolved" vector + the set of keys to press.
"""

import numpy as np

from info.module_logger import get_module_logger
from core.controller.action_heads import (
    NUM_HEADS, HEAD_KEYS, COMPAT, HEAD_PRIORITY, HEAD_NAMES,
    HEAD_MOVEMENT, HEAD_CAMERA, HEAD_COMBAT,
    HEAD_USE_ITEM, HEAD_SELECT_ITEM, HEAD_MENU, HEAD_SPRINT,
    HEAD_NAME_TO_IDX,
)

logger = get_module_logger('action_resolver')


class ActionResolver:
    """
    Stateless resolver: raw action vector + game context → valid key set.

    Usage:
        resolver = ActionResolver()
        resolved, keys = resolver.resolve(raw_action, menu_open=False)
    """

    # Heads that are allowed when the in-game menu IS open
    _MENU_OPEN_HEADS = frozenset({HEAD_MOVEMENT, HEAD_MENU})

    # Heads that are allowed when the in-game menu is NOT open
    _MENU_CLOSED_HEADS = frozenset({
        HEAD_MOVEMENT, HEAD_CAMERA, HEAD_COMBAT,
        HEAD_USE_ITEM, HEAD_SELECT_ITEM, HEAD_SPRINT,
        HEAD_MENU,  # only branch 0 or 1 (start) — enforced below
    })

    def __init__(self, disabled_heads: frozenset = None):
        """
        Args:
            disabled_heads: frozenset of head indices to disable.
                            Defaults to {HEAD_MENU} if None.
        """
        if disabled_heads is None:
            self._disabled_heads: frozenset = frozenset({HEAD_MENU})
        else:
            self._disabled_heads = frozenset(disabled_heads)

        if self._disabled_heads:
            disabled_names = [HEAD_NAMES.get(h, str(h)) for h in self._disabled_heads]
            logger.debug(f"ActionResolver: disabled heads = {disabled_names}")

    def resolve(self, raw_action, *, menu_open: bool = False):
        """
        Resolve a raw multi-head action vector into a valid key set.

        Args:
            raw_action:  array-like, shape (NUM_HEADS,).
                         Each element is the branch index chosen by the agent.
            menu_open:   True if the in-game menu is currently open
                         (read from memory address IN_GAME_MENU_IS_OPEN).

        Returns:
            (resolved_action, keys_to_press)
            - resolved_action: np.ndarray shape (NUM_HEADS,) after masking
            - keys_to_press:   set of key name strings (e.g. {'w', 'shift'})
        """
        resolved = np.array(raw_action, dtype=np.int32).copy()

        # Zero out disabled heads before any other processing
        for head_idx in self._disabled_heads:
            if 0 <= head_idx < NUM_HEADS:
                resolved[head_idx] = 0

        # ── Step 1: Menu state gate ──────────────────────────────────
        self._apply_menu_gate(resolved, menu_open)

        # ── Step 2: Compatibility conflict resolution ────────────────
        self._resolve_conflicts(resolved)

        # ── Step 3: Map to keys ──────────────────────────────────────
        keys = self._collect_keys(resolved)

        return resolved, keys

    def build_action_mask(self, *, menu_open: bool = False):
        from core.controller.action_heads import ACTION_BRANCHES
        masks = [np.ones(n, dtype=bool) for n in ACTION_BRANCHES]

        # Disabled heads: only branch 0 (nothing) is valid
        for h in self._disabled_heads:
            if 0 <= h < NUM_HEADS:
                masks[h][:] = False
                masks[h][0] = True

        if menu_open:
            for h in range(NUM_HEADS):
                if h not in self._MENU_OPEN_HEADS and h not in self._disabled_heads:
                    masks[h][:] = False
                    masks[h][0] = True
        else:
            masks[HEAD_MENU][:] = False
            masks[HEAD_MENU][0] = True
            masks[HEAD_MENU][1] = True

        return masks

    # ------------------------------------------------------------------
    # Step 1 — Menu state gate
    # ------------------------------------------------------------------

    def _apply_menu_gate(self, action, menu_open: bool):
        """
        Enforce menu-state rules IN PLACE on the action vector.

        When menu IS open:
            - Only Movement and Menu heads are active.
            - All other heads → 0.
            - Menu head: all branches (0-7) allowed.

        When menu is NOT open:
            - Menu head can only be 0 (nothing) or 1 (start).
            - All gameplay heads active.
        """
        if menu_open:
            if HEAD_MENU in self._disabled_heads:
                for h in range(NUM_HEADS):
                    action[h] = 0
                action[HEAD_MENU] = 1
            else:
                # Normal: only movement + menu allowed
                for h in range(NUM_HEADS):
                    if h not in self._MENU_OPEN_HEADS:
                        action[h] = 0
        else:
            # Menu head: only allow "nothing" (0) or "start" (1)
            if action[HEAD_MENU] > 1:
                action[HEAD_MENU] = 0

    # ------------------------------------------------------------------
    # Step 2 — Compatibility conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflicts(self, action):
        """
        Resolve mutual exclusion conflicts IN PLACE.

        Algorithm:
            1. Collect all active heads (branch != 0).
            2. Sort by descending priority.
            3. Walk through: for each active head, check if it conflicts
               with any higher-priority head already "locked in".
               If yes → zero it out.

        This is O(N²) on NUM_HEADS=7, so ~49 comparisons max — trivial.
        """
        # Collect active heads sorted by priority (highest first)
        active = [
            (HEAD_PRIORITY[h], h)
            for h in range(NUM_HEADS)
            if action[h] != 0
        ]
        active.sort(reverse=True)

        locked = set()  # Heads that are "accepted" and locked in

        for _priority, head in active:
            # Check compatibility with all already-locked heads
            compatible = True
            for locked_head in locked:
                if head not in COMPAT.get(locked_head, set()):
                    compatible = False
                    break

            if compatible:
                locked.add(head)
            else:
                action[head] = 0  # Conflict → zero out this lower-priority head

    # ------------------------------------------------------------------
    # Step 3 — Key collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_keys(resolved) -> set:
        """
        Map each (head, branch) to its physical key(s) and return the union.

        Returns:
            Set of key name strings, e.g. {'w', 'shift', 'left'}
        """
        keys = set()
        for head_idx in range(NUM_HEADS):
            branch = int(resolved[head_idx])
            if branch == 0:
                continue
            head_map = HEAD_KEYS.get(head_idx, {})
            key_list = head_map.get(branch, [])
            keys.update(key_list)
        return keys


