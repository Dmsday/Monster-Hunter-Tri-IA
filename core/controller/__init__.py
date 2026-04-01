"""
core.controller <--> Multi-head hold/release controller for Monster Hunter Tri.

Replaces the old monolithic core/controller.py with a clean package:

    core/controller/
    ├── __init__.py          ← you are here (re-exports)
    ├── constants.py         ← DIK codes, shared mem layout, VK codes
    ├── action_heads.py      ← Head definitions, branches, compatibility matrix
    ├── action_resolver.py   ← Menu-state gate + conflict resolution → key set
    ├── key_state_manager.py ← Hold/release diff engine
    ├── dll_utils.py         ← DLL find/build/inject, PID resolution
    └── wii_controller.py    ← Main WiiController class

Usage (backward-compatible import):
    from core.controller import WiiController, find_dolphin_pid

Usage (new multi-head imports):
    from core.controller.action_heads import ACTION_BRANCHES, NUM_HEADS, HEAD_NAMES
    from core.controller.action_heads import describe_action
    from core.controller.action_resolver import ActionResolver
"""

# ── Primary exports (backward-compatible with old `from core.controller import ...`) ──
from core.controller.wii_controller import WiiController
from core.controller.dll_utils import find_dolphin_pid

# ── Action space definition (used by environment/spaces.py) ──
from core.controller.action_heads import (
    ACTION_BRANCHES,
    NUM_HEADS,
    HEAD_MOVEMENT,
    HEAD_CAMERA,
    HEAD_COMBAT,
    HEAD_USE_ITEM,
    HEAD_SELECT_ITEM,
    HEAD_MENU,
    HEAD_SPRINT,
    HEAD_NAMES,
    HEAD_KEYS,
    describe_action,
)

# ── Make find_dolphin_pid available as _find_dolphin_pid for legacy code ──
_find_dolphin_pid = find_dolphin_pid

__all__ = [
    'WiiController',
    'find_dolphin_pid',
    '_find_dolphin_pid',
    'ACTION_BRANCHES',
    'NUM_HEADS',
    'HEAD_MOVEMENT',
    'HEAD_CAMERA',
    'HEAD_COMBAT',
    'HEAD_USE_ITEM',
    'HEAD_SELECT_ITEM',
    'HEAD_MENU',
    'HEAD_SPRINT',
    'HEAD_NAMES',
    'HEAD_KEYS',
    'describe_action',
]
