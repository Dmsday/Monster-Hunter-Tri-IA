"""
spaces.py — Build Gymnasium action and observation spaces.

Defines the shape of what the agent can do (action_space) and what it
sees (observation_space) based on the active modalities (vision, memory).

Public API:
    action_space                         = build_action_space()
    observation_space, modality_summary  = build_observation_space(...)
"""

import numpy as np
from gymnasium import spaces
from core.controller.action_heads import ACTION_BRANCHES # [5, 5, 6, 2, 3, 8, 2]

from info.module_logger import get_module_logger

logger = get_module_logger('spaces')

# Memory vector dimensionality
MEMORY_VECTOR_SIZE = 70

# Exploration minimap resolution and channels
MINIMAP_SIZE = 15
MINIMAP_CHANNELS = 4  # visits, player_pos, recent_cubes, markers


def build_action_space() -> spaces.MultiDiscrete:
    """Return the multi-head hold/release action space (7 heads)."""
    return spaces.MultiDiscrete(ACTION_BRANCHES)


def build_observation_space(
    *,
    use_vision: bool,
    use_memory: bool,
    has_memory_reader: bool,
    frame_size: tuple = (84, 84),
    grayscale: bool = False,
    frame_stack: int = 4,
) -> spaces.Dict:
    """
    Build the Dict observation space based on active modalities.

    Args:
        use_vision:        Whether the visual modality is enabled.
        use_memory:        Whether the memory modality is enabled.
        has_memory_reader: Whether a MemoryReader instance is available
                           (needed for exploration map even if use_memory=False).
        frame_size:        (H, W) of each visual frame.
        grayscale:         Use 1 channel per frame instead of 3.
        frame_stack:       Number of frames stacked along the channel axis.

    Returns:
        A gymnasium.spaces.Dict with the appropriate sub-spaces.
    """
    obs_spaces = {}

    # 1. Visual modality
    if use_vision:
        channels = (1 if grayscale else 3) * frame_stack
        obs_spaces['visual'] = spaces.Box(
            low=0.0, high=1.0,
            shape=(*frame_size, channels),
            dtype=np.float32,
        )

    # 2. Memory vector
    if use_memory:
        obs_spaces['memory'] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MEMORY_VECTOR_SIZE,),
            dtype=np.float32,
        )
        logger.info(f"Memory vector: {MEMORY_VECTOR_SIZE} features")

    # 3. Exploration minimap (requires vision + a memory reader)
    if use_vision and has_memory_reader:
        obs_spaces['exploration_map'] = spaces.Box(
            low=-1.0, high=1.0,
            shape=(MINIMAP_SIZE, MINIMAP_SIZE, MINIMAP_CHANNELS),
            dtype=np.float32,
        )
        logger.info(
            f"Exploration minimap enabled ({MINIMAP_SIZE}x{MINIMAP_SIZE}x{MINIMAP_CHANNELS})"
        )

    # Build final space
    if obs_spaces:
        obs_space = spaces.Dict(obs_spaces)
        modalities = list(obs_spaces.keys())
        logger.info(f"Observation space: Dict with {len(modalities)} modalities {modalities}")
    else:
        # Fallback — should never happen in practice
        obs_space = spaces.Dict({
            'fallback': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        })
        logger.warning("Observation space: fallback Box(10,) — no modality active")

    return obs_space
