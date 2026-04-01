# utils/memory_vector.py
"""
Converts a raw_memory dict (from MemoryReader) into a normalized 70-feature numpy vector.
Single source of truth used by both mh_env.py and state_fusion.py.
"""
import numpy as np
from utils.safe_float import safe_float


def build_memory_vector(raw_memory: dict, reward_calc=None) -> np.ndarray:
    """
    Builds the 70-feature memory vector.

    Args:
        raw_memory: Dict from MemoryReader.read_game_state()
        reward_calc: Optional MonsterHunterRewardCalculator instance.
                     If provided, features 68 (in_combat) and 69 (in_monster_zone)
                     are read from it. If None, both are set to 0.0.

    Returns:
        np.ndarray of shape (70,), dtype float32

    Vector structure:
        [0-12]  13 base features  (HP, stamina, position, orientation, zone, etc.)
        [13-60] 48 inventory features (24 slots × 2: item_id + quantity)
        [61-64]  4 combat features (quest_time, attack/defense, monster_count, monsters_present)
        [65-69]  5 extra features (sharpness, in_menu, item_selected, in_combat, in_monster_zone)
    """
    features = []

    # ===== PART 1 : 13 BASE FEATURES =====
    features.extend([
        safe_float(raw_memory.get('player_hp'),              default=50.0,   min_val=0.0,     max_val=150.0),
        safe_float(raw_memory.get('player_hp_recoverable'),  default=0.0,    min_val=0.0,     max_val=150.0),
        safe_float(raw_memory.get('player_stamina'),         default=50.0,   min_val=0.0,     max_val=150.0),
        safe_float(raw_memory.get('player_x'),               default=0.0,    min_val=-10000.0, max_val=10000.0),
        safe_float(raw_memory.get('player_y'),               default=0.0,    min_val=-10000.0, max_val=10000.0),
        safe_float(raw_memory.get('player_z'),               default=0.0,    min_val=-10000.0, max_val=10000.0),
        safe_float(raw_memory.get('player_orientation'),     default=0.0,    min_val=0.0,     max_val=360.0),
        safe_float(raw_memory.get('current_zone'),           default=0.0,    min_val=0.0,     max_val=20.0),
        safe_float(raw_memory.get('damage_last_hit'),        default=0.0,    min_val=0.0,     max_val=10000.0),
        safe_float(raw_memory.get('money'),                  default=0.0,    min_val=0.0,     max_val=999999.0),
        safe_float(raw_memory.get('death_count'),            default=0.0,    min_val=0.0,     max_val=10.0),
        1.0 if raw_memory.get('stamina_low', False) else 0.0,
        safe_float(raw_memory.get('time_underwater'),        default=0.0,    min_val=0.0,     max_val=200.0),
    ])

    # ===== PART 2 : 48 INVENTORY FEATURES (24 slots × 2) =====
    inventory = raw_memory.get('inventory_items', [])
    inventory_dict = {
        item['slot']: item
        for item in inventory
        if item.get('slot') is not None and 1 <= item['slot'] <= 24
    }
    for slot_num in range(1, 25):
        if slot_num in inventory_dict:
            item = inventory_dict[slot_num]
            features.append(safe_float(item.get('item_id', 0),  default=0.0, min_val=0.0, max_val=746.0))
            features.append(safe_float(item.get('quantity', 0), default=0.0, min_val=0.0, max_val=99.0))
        else:
            features.extend([0.0, 0.0])

    # ===== PART 3 : 4 COMBAT FEATURES =====
    features.append(safe_float(raw_memory.get('quest_time'),          default=5400.0, min_val=0.0, max_val=5400.0))
    features.append(safe_float(raw_memory.get('attack_defense_value'), default=0.0,   min_val=0.0, max_val=10000.0))

    monster_count = sum(
        1 for i in range(1, 6)
        if raw_memory.get(f'smonster{i}_hp') is not None and raw_memory[f'smonster{i}_hp'] > 0
    )
    features.append(float(monster_count))
    features.append(1.0 if monster_count > 0 else 0.0)

    # ===== PART 4 : 5 EXTRA FEATURES =====
    features.append(safe_float(raw_memory.get('sharpness'),     default=150.0, min_val=-10.0, max_val=5000.0))
    features.append(1.0 if raw_memory.get('in_game_menu', False) else 0.0)
    features.append(safe_float(raw_memory.get('item_selected'), default=24.0,  min_val=0.0,  max_val=24.0))

    # Features 68-69 : need reward_calc to be accurate
    if reward_calc is not None:
        features.append(1.0 if getattr(reward_calc, 'prev_in_combat',    False) else 0.0)
        features.append(1.0 if getattr(reward_calc, 'zone_has_monsters', False) else 0.0)
    else:
        # No reward_calc available : always 0.0
        # (used in mh_env memory-only mode where reward_calc is not accessible here)
        features.extend([0.0, 0.0])

    assert len(features) == 70, f"Expected 70 features, got {len(features)}"

    arr = np.array(features, dtype=np.float32)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=10000.0, neginf=-10000.0)

    return arr