"""
constants.py — Low-level constants for Dolphin DInput injection.

Contains DIK scan codes, shared memory layout, and Windows VK codes.
These MUST match the Rust DLL (dolphin_input_hook) exactly.
"""

# ============================================================================
# Shared memory layout — must match the Rust DLL
# ============================================================================
FILE_MAP_ALL_ACCESS = 0xF001F
SHARED_MEM_SIZE     = 280   # 256 keyboard + 8 mouse + 16 reserved
HOTKEY_OFFSET       = 264   # Legacy hotkey byte (unused)

# ============================================================================
# VK codes for F1-F8 (Windows virtual key codes, for save state PostMessage)
# ============================================================================
VK_FKEYS = {
    'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
    'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
}

# ============================================================================
# DInput DIK scan codes — physical key positions, layout-independent.
# Match the Dolphin keybindings already configured for MH Tri.
# ============================================================================
DIK: dict = {
    # Movement (WASD)
    'w':     0x11,
    's':     0x1F,
    'a':     0x1E,
    'd':     0x20,
    # Camera / Menu navigation (arrow keys)
    'up':    0xC8,
    'down':  0xD0,
    'left':  0xCB,
    'right': 0xCD,
    # Action keys
    '1':     0x02,   # dodge
    '3':     0x04,   # draw / sheathe
    '4':     0x05,   # attack 1 / confirm (A button)
    '0':     0x0B,   # attack 2 / back (B button)
    'q':     0x10,   # use item
    'e':     0x12,   # menu / start
    'p':     0x19,   # Draw/sheath weapon | kick (sheath)
    'm':     0x32,   # Kick attack | menu zoom/close
    'ctrl':  0x1D,   # Z-target  (left ctrl)
    'shift': 0x2A,   # C-button / sprint (left shift)
    # Function keys (reference only — save states use PostMessage, not DInput)
    'f1': 0x3B, 'f2': 0x3C, 'f3': 0x3D, 'f4': 0x3E,
    'f5': 0x3F, 'f6': 0x40, 'f7': 0x41, 'f8': 0x42,
}

# ============================================================================
# Mouse button offsets in shared memory (after 256 keyboard bytes)
# Wii A button = mouse left click, Wii B button = mouse right click
# ============================================================================
MOUSE_LEFT  = 256   # Shared memory byte index for left mouse click (A button = attack)
MOUSE_RIGHT = 257   # Shared memory byte index for right mouse click (B button = dodge/evade)

# All keys that the multi-head system can control (keyboard names + mouse aliases)
ALL_MANAGED_KEYS = frozenset({
    'w', 's', 'a', 'd',
    'up', 'down', 'left', 'right',
    'ctrl', 'q', 'e', 'shift',
    'p', 'm',
    'mouse_left', 'mouse_right',
})