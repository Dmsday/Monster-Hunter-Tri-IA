"""
action_heads.py — Multi-head action space definition for Monster Hunter Tri.

Defines 7 independent action heads that the agent controls simultaneously.
Each head outputs a discrete choice; 0 always means "do nothing" for that group.
Keys STAY PRESSED between steps (hold/release paradigm).

Heads:
    0  Movement      [5]  nothing | forward | backward | strafe_L | strafe_R
    1  Camera        [5]  nothing | up | down | left | right
    2  Combat        [6]  nothing | attack1 | attack2 | dodge | draw_sheath | z_target
    3  Use Item      [2]  nothing | use
    4  Select Item   [3]  nothing | radial_left | radial_right
    5  Menu          [8]  nothing | start | nav_up | nav_down | nav_left | nav_right | confirm(A) | back(B)
    6  Sprint        [2]  nothing | sprint

Compatibility matrix (from game mechanics):
    When two heads are incompatible, the higher-priority one wins
    and the other is forced to 0 (release).

Menu state awareness:
    When in_game_menu == True  → only Menu + Movement heads are active
    When in_game_menu == False → Menu head can only output 0 (nothing) or 1 (start)
"""

from info.module_logger import get_module_logger

logger = get_module_logger('action_heads')

# ============================================================================
# Head indices
# ============================================================================
HEAD_MOVEMENT    = 0
HEAD_CAMERA      = 1
HEAD_COMBAT      = 2
HEAD_USE_ITEM    = 3
HEAD_SELECT_ITEM = 4
HEAD_MENU        = 5
HEAD_SPRINT      = 6

NUM_HEADS = 7

# ============================================================================
# Branch counts per head  →  gymnasium.spaces.MultiDiscrete(ACTION_BRANCHES)
# ============================================================================
ACTION_BRANCHES = [5, 5, 5, 2, 3, 8, 2]

# ============================================================================
# Key mappings: head_index → {branch_index: [key_names]}
# Branch 0 is always "nothing" (no keys) for every head.
# ============================================================================
HEAD_KEYS = {
    HEAD_MOVEMENT: {
        1: ['w'],                       # forward
        2: ['s'],                       # backward
        3: ['a'],                       # strafe left
        4: ['d'],                       # strafe right
    },
    HEAD_CAMERA: {
        1: ['up'],                      # camera up
        2: ['down'],                    # camera down
        3: ['left'],                    # camera left
        4: ['right'],                   # camera right
    },
    HEAD_COMBAT: {
        1: ['mouse_left'],              # Attack (left click - only effective when weapon drawn)
        2: ['mouse_right'],             # Dodge / crouch (right click)
        3: ['p'],                       # Draw / sheath weapon (p key)
        4: ['m'],                       # Kick attack (m key) | menu zoom+close
    },
    HEAD_USE_ITEM: {
        1: ['q'],                        # use selected item (Wii 1 button = Q key)
    },
    HEAD_SELECT_ITEM: {
        1: ['shift', 'left'],           # C-button + left  → radial select left
        2: ['shift', 'right'],          # C-button + right → radial select right
    },
    HEAD_MENU: {
        1: ['e'],                      # Open / close menu (e key)
        2: ['up'],                     # Menu navigate up
        3: ['down'],                   # Menu navigate down
        4: ['left'],                   # Menu navigate left
        5: ['right'],                  # Menu navigate right
        6: ['mouse_left'],             # Menu confirm (left click)
        7: ['mouse_right'],            # Menu back / cancel (right click)
    },
    HEAD_SPRINT: {
        1: ['ctrl'],                   # Run (ctrl held) | Block when weapon drawn (game decides)
    },
}

# ============================================================================
# Compatibility matrix
#
# COMPAT[head_a] is the set of heads that head_a is COMPATIBLE with.
# If head_a is active (branch != 0) and head_b is NOT in COMPAT[head_a],
# then head_b must be forced to 0.
#
# The matrix is symmetric:
#   if B in COMPAT[A], then A in COMPAT[B].
# ============================================================================
COMPAT = {
    HEAD_MOVEMENT:    {HEAD_MOVEMENT, HEAD_CAMERA, HEAD_COMBAT, HEAD_SELECT_ITEM, HEAD_MENU, HEAD_SPRINT},
    HEAD_CAMERA:      {HEAD_MOVEMENT, HEAD_CAMERA, HEAD_COMBAT, HEAD_USE_ITEM, HEAD_SPRINT},
    HEAD_COMBAT:      {HEAD_MOVEMENT, HEAD_CAMERA, HEAD_COMBAT},
    HEAD_USE_ITEM:    {HEAD_CAMERA, HEAD_USE_ITEM},
    HEAD_SELECT_ITEM: {HEAD_MOVEMENT, HEAD_SELECT_ITEM, HEAD_SPRINT},
    HEAD_MENU:        {HEAD_MOVEMENT, HEAD_MENU},
    HEAD_SPRINT:      {HEAD_MOVEMENT, HEAD_CAMERA, HEAD_SELECT_ITEM, HEAD_SPRINT},
}

# ============================================================================
# Conflict resolution priority (higher number = higher priority wins)
# When two active heads are incompatible, the one with higher priority
# keeps its value; the other is forced to 0.
# ============================================================================
HEAD_PRIORITY = {
    HEAD_MENU:        60,   # Menu overrides almost everything (when open)
    HEAD_USE_ITEM:    50,   # Use item is very restrictive
    HEAD_COMBAT:      40,   # Combat blocks sprint, items, menu
    HEAD_SELECT_ITEM: 30,   # Item radial blocks camera, combat
    HEAD_SPRINT:      20,   # Sprint blocks combat, menu
    HEAD_CAMERA:      10,   # Camera is nearly always allowed
    HEAD_MOVEMENT:     0,   # Movement is always allowed
}

# ============================================================================
# Human-readable head names (for logging / debugging)
# ============================================================================
HEAD_NAMES = {
    HEAD_MOVEMENT:    'movement',
    HEAD_CAMERA:      'camera',
    HEAD_COMBAT:      'combat',
    HEAD_USE_ITEM:    'use_item',
    HEAD_SELECT_ITEM: 'select_item',
    HEAD_MENU:        'menu',
    HEAD_SPRINT:      'sprint',
}

# Reverse mapping: name -> head index (used by CLI + ActionResolver)
HEAD_NAME_TO_IDX: dict = {name: idx for idx, name in HEAD_NAMES.items()}

# Head names accepted by CLI
ALL_HEAD_NAMES = list(HEAD_NAME_TO_IDX.keys())

def describe_action(action_vector) -> str:
    """
    Return a human-readable string for a resolved action vector.
    Useful for debug logging.

    Args:
        action_vector: array-like of shape (NUM_HEADS,)

    Returns:
        e.g. "movement=forward | camera=left | sprint=on"
    """
    branch_names = {
        HEAD_MOVEMENT:    {0: '-', 1: 'forward', 2: 'backward', 3: 'strafe_L', 4: 'strafe_R'},
        HEAD_CAMERA:      {0: '-', 1: 'up', 2: 'down', 3: 'left', 4: 'right'},
        HEAD_COMBAT:      {0: '-', 1: 'attack', 2: 'dodge', 3: 'draw_sheath', 4: 'kick'},
        HEAD_USE_ITEM:    {0: '-', 1: 'use'},
        HEAD_SELECT_ITEM: {0: '-', 1: 'sel_L', 2: 'sel_R'},
        HEAD_MENU:        {0: '-', 1: 'start', 2: 'nav_up', 3: 'nav_down',
                           4: 'nav_left', 5: 'nav_right', 6: 'confirm', 7: 'back'},
        HEAD_SPRINT:      {0: '-', 1: 'sprint'},
    }
    parts = []
    for h in range(NUM_HEADS):
        branch = int(action_vector[h])
        if branch != 0:
            name = HEAD_NAMES[h]
            label = branch_names.get(h, {}).get(branch, f'?{branch}')
            parts.append(f"{name}={label}")
    return ' | '.join(parts) if parts else 'NOOP'
