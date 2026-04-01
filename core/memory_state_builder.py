
import time as _time

# Imports required by build_game_state (normalization helpers and logger)
from core.memory_normalizer import normalize_stamina, normalize_hp, convert_orientation_to_degrees
from info.module_logger import get_module_logger

logger = get_module_logger('memory_state_builder')

_quest_end_log_timestamps: dict = {}   # id(reader) -> last log timestamp
_QUEST_END_LOG_INTERVAL = 10.0         # seconds between repeated logs

def build_game_state(reader: 'MemoryReader') -> dict:
    """
    État complet
    """
    state = {}

    # ===================================================================
    # PARTIE 0 : DÉTECTION FIN DE QUÊTE (PRIORITAIRE)
    # ===================================================================

    # Lire CURRENT_MAP pour détecter écran de fin
    current_map = reader.read_value('CURRENT_MAP')
    state['current_map'] = current_map

    # Si CURRENT_MAP = 45, on est sortie de la quete
    if current_map == 45:
        state['quest_ended'] = True
        state['on_reward_screen'] = True
        _now = _time.time()
        _key = id(reader)
        if _now - _quest_end_log_timestamps.get(_key, 0.0) >= _QUEST_END_LOG_INTERVAL:
            _quest_end_log_timestamps[_key] = _now
            logger.warning(
                f"Quest end screen detected (CURRENT_MAP=45) — "
                f"zone={reader.read_value('CURRENT_ZONE')}, "
                f"deaths={reader.read_value('DEATH_COUNTER')}"
            )
    else:
        state['quest_ended'] = False
        state['on_reward_screen'] = False

    # ===================================================================
    # PARTIE 1 : STATS DE BASE
    # ===================================================================

    stamina_raw = reader.read_value('PLAYER_CURRENT_STAMINA')
    hp_raw = reader.read_value('PLAYER_CURRENT_HP')
    hp_rec_raw = reader.read_value('PLAYER_RECOVERABLE_HP')
    orientation_ns_raw = reader.read_value('PLAYER_NS_ORIENTATION')
    orientation_ew_raw = reader.read_value('PLAYER_EW_ORIENTATION')

    state['player_stamina_raw'] = stamina_raw
    state['player_hp_raw'] = hp_raw
    state['player_hp_recoverable_raw'] = hp_rec_raw

    state['player_stamina'] = normalize_stamina(stamina_raw)
    state['player_hp'] = normalize_hp(hp_raw)
    state['player_hp_recoverable'] = normalize_hp(hp_rec_raw) if hp_rec_raw else 0.0

    state['stamina_low'] = state['player_stamina'] < 25 if state['player_stamina'] else False

    # Position
    player_x = reader.read_value('PLAYER_X')
    player_y = reader.read_value('PLAYER_Y')
    player_z = reader.read_value('PLAYER_Z')

    state['player_x'] = player_x
    state['player_y'] = player_y
    state['player_z'] = player_z

    # Orientation en degrés
    orientation_deg = convert_orientation_to_degrees(
        orientation_ns_raw,
        orientation_ew_raw
    )

    state['player_orientation'] = orientation_deg

    # Autres base
    state['current_zone'] = reader.read_value('CURRENT_ZONE')
    state['damage_last_hit'] = reader.read_value('DAMAGE_RECEIVE_LAST_HIT')
    state['money'] = reader.read_value('PLAYER_MONEY')
    state['death_count'] = reader.read_value('DEATH_COUNTER')
    state['player_stamina_max'] = reader.read_value('PLAYER_STAMINA_MAX')

    # ===================================================================
    # PARTIE 2 : FEATURES QUETES
    # ===================================================================

    # QUEST TIME (converti en secondes)
    quest_time_raw = reader.read_value('QUEST_TIME_SPENT')
    if quest_time_raw is not None:
        state['quest_time'] = int(quest_time_raw / 30)  # Frames → secondes
    else:
        state['quest_time'] = None

    # ATTACK & DEFENSE (valeur brute combinée)
    state['attack_defense_value'] = reader.read_value('ATTACK_AND_DEFENSE_VALUE')

    # SHARPNESS (valeur brute)
    state['sharpness'] = reader.read_value('SHARPNESS')

    # IN GAME MENU (valeur brute - 0 ou 1)
    in_menu_raw = reader.read_value('IN_GAME_MENU_IS_OPEN')
    state['in_game_menu'] = (in_menu_raw == 1) if in_menu_raw is not None else False

    # ITEM SELECTED (slot sélectionné 0-23, ou 24 si rien)
    item_selected_raw = reader.read_value('ITEM_SELECTED')
    if item_selected_raw is not None:
        # Valeur brute: 0-23 = slots 1-24, 24 = rien
        state['item_selected'] = item_selected_raw
    else:
        state['item_selected'] = 24  # Défaut = rien de sélectionné

    # ===================================================================
    # PARTIE 3 : HP MONSTRES (BRUT) + DÉTECTION
    # ===================================================================

    # Lire HP de tous les small monsters (1-5)
    state['smonster1_hp'] = reader.read_value('SMONSTER1_HP')
    state['smonster2_hp'] = reader.read_value('SMONSTER2_HP')
    state['smonster3_hp'] = reader.read_value('SMONSTER3_HP')
    state['smonster4_hp'] = reader.read_value('SMONSTER4_HP')
    state['smonster5_hp'] = reader.read_value('SMONSTER5_HP')

    # Large monster (si implémenté)
    state['lmonster1_hp'] = reader.read_value('LMONSTER1_HP')

    # ===================================================================
    # PARTIE 4 : OXYGÈNE
    # ===================================================================

    oxygen_raw = reader.read_value('TIME_SPENT_UNDERWATER')

    if oxygen_raw is not None:
        try:
            oxygen_value = int(oxygen_raw)

            if 0 <= oxygen_value <= 200:
                state['time_underwater'] = oxygen_value
                state['oxygen_valid'] = True
                state['oxygen_low_warning'] = oxygen_value < 25
                state['oxygen_critical_warning'] = oxygen_value < 10
            else:
                state['time_underwater'] = None
                state['oxygen_valid'] = False
                state['oxygen_error'] = f'out_of_range_{oxygen_value}'

        except (ValueError, TypeError) as conv_error:
            state['time_underwater'] = None
            state['oxygen_valid'] = False
            state['oxygen_error'] = f'conversion_error_{conv_error}'
    else:
        state['time_underwater'] = None
        state['oxygen_valid'] = False
        state['oxygen_error'] = 'read_failed'

    # ===================================================================
    # PARTIE 5 : INVENTAIRE COMPLET (24 SLOTS)
    # ===================================================================

    state['inventory_items'] = reader.read_inventory()

    return state