"""
sanitizer.py — Cleans step/reset info dicts for VecNormalize compatibility.

Converts complex types (numpy, nested dicts, lists) into simple Python types
(int, float, bool, str) that VecNormalize and SB3 can handle safely.

Public API:
    cleaned = sanitize_info(raw_info)
"""

import numpy as np

from info.module_logger import get_module_logger

logger = get_module_logger('sanitizer')


def sanitize_info(info_dict: dict) -> dict:
    """
    Clean an info dictionary for VecNormalize / SB3 compatibility.

    Rules:
        1. Preserve 'episode' dict {r, l, t} for SB3 if valid
        2. Keep reward_breakdown dicts but sanitize None values
        3. Convert None -> default (0 / 0.0 / False)
        4. Convert numpy types -> native Python types
        5. Remove complex lists/dicts except inventory
        6. Guarantee critical keys exist

    Args:
        info_dict: Raw info dictionary from step/reset.

    Returns:
        Sanitized dictionary safe for VecNormalize.
    """
    sanitized = {}

    # --- 1. Handle 'episode' key (SB3 format {r, l, t}) ---
    episode_dict = _extract_episode_dict(info_dict)

    # --- 2. Process all other keys ---
    for key, value in info_dict.items():
        if key == 'episode':
            continue  # Already handled

        # Nested dicts: keep reward breakdowns, skip others
        if isinstance(value, dict):
            if 'reward' in key.lower() or 'breakdown' in key.lower():
                sanitized[key] = _sanitize_reward_dict(value)
            continue

        # None -> default value
        if value is None:
            sanitized[key] = _default_for_key(key)
            continue

        # Type conversion
        converted = _convert_value(key, value)
        if converted is not None:
            sanitized[key] = converted

    # --- 3. Re-attach episode dict ---
    if episode_dict is not None:
        sanitized['episode'] = episode_dict

    # --- 4. Guarantee critical keys ---
    _ensure_critical_keys(sanitized)

    return sanitized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_episode_dict(info_dict: dict):
    """Extract and validate the SB3 'episode' dict {r, l, t}."""
    if 'episode' not in info_dict:
        return None

    value = info_dict['episode']

    if isinstance(value, dict):
        if {'r', 'l', 't'}.issubset(value.keys()):
            try:
                return {
                    'r': float(value['r']),
                    'l': int(value['l']),
                    't': float(value['t']),
                }
            except (ValueError, TypeError) as exc:
                logger.error(f"Episode dict conversion error: {exc}")
        else:
            logger.warning(f"Incomplete 'episode' dict — keys: {value.keys()}")
    elif isinstance(value, (int, np.integer)):
        logger.warning(f"'episode' is int ({value}) — renamed to 'episode_num'")
    else:
        logger.warning(f"'episode' invalid type ({type(value)}) — ignored")

    return None


def _sanitize_reward_dict(d: dict) -> dict:
    """Sanitize a reward breakdown dict (None -> 0.0, numpy -> float)."""
    cleaned = {}
    for k, v in d.items():
        if v is None:
            cleaned[k] = 0.0
        elif isinstance(v, (int, float, np.integer, np.floating)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = 0.0
    return cleaned


def _default_for_key(key: str):
    """Return a sensible default for a None value based on key name."""
    key_lower = key.lower()
    if any(w in key_lower for w in ('count', 'num', 'steps', 'episode')):
        return 0
    if any(w in key_lower for w in ('hp', 'stamina', 'reward', 'distance', 'orientation')):
        return 0.0
    return False


def _convert_value(key: str, value):
    """Convert a value to a safe Python type. Returns None to skip."""
    try:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, (float, np.floating)):
            f = float(value)
            return 0.0 if (np.isnan(f) or np.isinf(f)) else f

        if isinstance(value, str):
            return value

        # Lists: only keep inventory
        if isinstance(value, (list, tuple, np.ndarray)):
            if key == 'inventory' and isinstance(value, list):
                return _clean_inventory(value)
            return None  # Skip other lists/arrays

        return None  # Unknown type — skip

    except (ValueError, TypeError, OverflowError) as exc:
        logger.warning(f"Cannot convert {key}={value}: {exc}")
        return None


def _clean_inventory(items: list) -> list:
    """Clean an inventory list of dicts."""
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entry = {}
        for k, v in item.items():
            if isinstance(v, (int, np.integer)):
                entry[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                entry[k] = float(v)
            elif isinstance(v, str):
                entry[k] = v
            elif v is None:
                entry[k] = None
        cleaned.append(entry)
    return cleaned


_CRITICAL_KEYS = {
    'episode_num': 0,
    'episode_steps': 0,
    'total_steps': 0,
    'hp': 0.0,
    'stamina': 0.0,
    'death_count': 0,
    'current_zone': 0,
}


def _ensure_critical_keys(sanitized: dict):
    """Make sure critical keys exist with safe defaults."""
    for key, default in _CRITICAL_KEYS.items():
        if key not in sanitized:
            sanitized[key] = default
