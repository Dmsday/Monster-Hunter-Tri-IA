"""
Memory Address Configuration for Monster Hunter Tri (Wii)
Precise data types definitions (2 bytes vs 4 bytes)

DATA TYPES:
- float : 4 bytes, big-endian, IEEE 754 standard
- int4  : 4 bytes, big-endian, signed or unsigned integer
- int2  : 2 bytes, big-endian, signed or unsigned integer
- byte  : 1 byte

IMPORTANT NOTE ABOUT VALUES:
    SOME VALUES READ MAY APPEAR STRANGE (3.4e37, etc.)
    This is NORMAL - The AI could learn patterns
    The neural network will focuse in this case on relative changes (↑/↓), not raw numbers
"""

# =============================================================================
# CATEGORY 1: PLAYER - STATISTICS
# =============================================================================

# Health Points
PLAYER_CURRENT_HP = 0x9014AEAF  # float (4 bytes) - Current HP (RAW VALUE)
PLAYER_RECOVERABLE_HP = 0x9014AEB3  # float (4 bytes) - Red HP (recoverable portion, RAW VALUE)

# Stamina System
PLAYER_CURRENT_STAMINA = 0x9014AEB8  # int4 (4 bytes) - Current stamina level (RAW VALUE)
PLAYER_STAMINA_MAX = 0x806C02A8  # float (4 bytes) - Maximum stamina capacity (RAW VALUE)

# Oxygen (underwater mechanics) - 2 bytes format
TIME_SPENT_UNDERWATER = 0x9014AEBE  # int2 (2 bytes) - Oxygen level: 100 = full, decreases over time; < 25 = warning alert

# Damage Detection - STATE CHANGE FLAG, not an actual damage value (different value = new hit)
DAMAGE_RECEIVE_LAST_HIT = 0x9014AED8  # float (4 bytes) - FLAG: if value changes = hit received

# Death Counter
DEATH_COUNTER = 0x9014AF86  # byte (1 byte) - Number of deaths in current quest (1, 2, 3...)

# Weapon and Armor Stats
ATTACK_AND_DEFENSE_VALUE = 0x9014AEF8  # int2 (2 bytes) - Combined attack/defense stat value
SHARPNESS = 0x9014B0AC  # int2 (2 bytes) - Weapon sharpness :
#   -1 = successful hit
#   -2 = bounce (attack deflected)

# In-Game Menu State
IN_GAME_MENU_IS_OPEN = 0x806AED56  # byte (1 byte) - Menu status: 0 = closed, 1 = open

# =============================================================================
# CATEGORY 2: PLAYER - POSITION & ORIENTATION
# =============================================================================

# 3D World Position - Raw values typically between -1 and 1
PLAYER_X = 0x900AD764  # float (4 bytes) - X coordinate
PLAYER_Y = 0x900AD75C  # float (4 bytes) - Y coordinate
PLAYER_Z = 0x900B00D8  # float (4 bytes) - Z coordinate
CAMERA_Z = 0x900AD748  # float (4 bytes) - Camera Z position

# Player Orientation (compass directions)
PLAYER_NS_ORIENTATION = 0x8066E410  # float (4 bytes) - North-South axis: -1 = north, +1 = south
PLAYER_EW_ORIENTATION = 0x8066E418  # float (4 bytes) - East-West axis: -1 = west, +1 = east

# Current Location Identifiers
CURRENT_ZONE = 0x806BAC64  # byte (1 byte) - Zone ID within current map
CURRENT_MAP = 0x92D1A80C  # int4 (4 bytes) - Current map identifier
REAL_MAP_ID = 0x90375E30  # int4 (4 bytes) - Real map ID: 1 = village (Moga Village)

# Controller Input
# (addresses to be added if possible for input injection)

# =============================================================================
# CATEGORY 3: INVENTORY & ECONOMY
# =============================================================================

# Economy System - mainly for yourself when you're playing, not for the AI lol
PLAYER_MONEY = 0x900E0588  # int4 (4 bytes) - Zenny (main currency)
PLAYER_MONEY_SPENT = 0x900E058B  # byte (1 byte) - Amount spent (transaction tracking)
MOGA_POINT = 0x900E4474  # int4 (4 bytes) - Moga Points (village resource points)

# Item Selection (currently active item)
ITEM_SELECTED = 0x9014AE44  # int2 (2 bytes) - Selected item slot number in pouch (slot - 1)
# Modulo 24: nothing = 24, slot 1 = 0, slot 2 = 1, etc.

# Inventory Pouch Layout (Village vs Quest contexts)
# Format: (village_address, quest_address) tuple for each slot
# Each slot has 2 memory locations: one for item ID, one for quantity

# Slot 1
ID_SLOT_1 = (0x900E0610, 0x9014ADB8)  # int2 (2 bytes) - Item ID in slot 1
NOITEM_SLOT_1 = (0x900E0612, 0x9014ADBA)  # int2 (2 bytes) - Quantity in slot 1

# Slot 2
ID_SLOT_2 = (0x900E0614, 0x9014ADBC)  # int2 (2 bytes) - Item ID in slot 2
NOITEM_SLOT_2 = (0x900E0616, 0x9014ADBE)  # int2 (2 bytes) - Quantity in slot 2

# Slot 3
ID_SLOT_3 = (0x900E0618, 0x9014ADC0)  # int2 (2 bytes) - Item ID in slot 3
NOITEM_SLOT_3 = (0x900E061A, 0x9014ADC2)  # int2 (2 bytes) - Quantity in slot 3

# Slot 4
ID_SLOT_4 = (0x900E061C, 0x9014ADC4)  # int2 (2 bytes) - Item ID in slot 4
NOITEM_SLOT_4 = (0x900E061E, 0x9014ADC6)  # int2 (2 bytes) - Quantity in slot 4

# Slot 5
ID_SLOT_5 = (0x900E0620, 0x9014ADC8)  # int2 (2 bytes) - Item ID in slot 5
NOITEM_SLOT_5 = (0x900E0622, 0x9014ADCA)  # int2 (2 bytes) - Quantity in slot 5

# Slot 6
ID_SLOT_6 = (0x900E0624, 0x9014ADCC)  # int2 (2 bytes) - Item ID in slot 6
NOITEM_SLOT_6 = (0x900E0626, 0x9014ADCE)  # int2 (2 bytes) - Quantity in slot 6

# Slot 7
ID_SLOT_7 = (0x900E0628, 0x9014ADD0)  # int2 (2 bytes) - Item ID in slot 7
NOITEM_SLOT_7 = (0x900E062A, 0x9014ADD2)  # int2 (2 bytes) - Quantity in slot 7

# Slot 8
ID_SLOT_8 = (0x900E062C, 0x9014ADD4)  # int2 (2 bytes) - Item ID in slot 8
NOITEM_SLOT_8 = (0x900E062E, 0x9014ADD6)  # int2 (2 bytes) - Quantity in slot 8

# Slot 9
ID_SLOT_9 = (0x900E0630, 0x9014ADD8)  # int2 (2 bytes) - Item ID in slot 9
NOITEM_SLOT_9 = (0x900E0632, 0x9014ADDA)  # int2 (2 bytes) - Quantity in slot 9

# Slot 10
ID_SLOT_10 = (0x900E0634, 0x9014ADDC)  # int2 (2 bytes) - Item ID in slot 10
NOITEM_SLOT_10 = (0x900E0636, 0x9014ADDE)  # int2 (2 bytes) - Quantity in slot 10

# Slot 11
ID_SLOT_11 = (0x900E0638, 0x9014ADE0)  # int2 (2 bytes) - Item ID in slot 11
NOITEM_SLOT_11 = (0x900E063A, 0x9014ADE2)  # int2 (2 bytes) - Quantity in slot 11

# Slot 12
ID_SLOT_12 = (0x900E063C, 0x9014ADE4)  # int2 (2 bytes) - Item ID in slot 12
NOITEM_SLOT_12 = (0x900E063E, 0x9014ADE6)  # int2 (2 bytes) - Quantity in slot 12

# Slot 13
ID_SLOT_13 = (0x900E0640, 0x9014ADE8)  # int2 (2 bytes) - Item ID in slot 13
NOITEM_SLOT_13 = (0x900E0642, 0x9014ADEA)  # int2 (2 bytes) - Quantity in slot 13

# Slot 14
ID_SLOT_14 = (0x900E0644, 0x9014ADEC)  # int2 (2 bytes) - Item ID in slot 14
NOITEM_SLOT_14 = (0x900E0646, 0x9014ADEE)  # int2 (2 bytes) - Quantity in slot 14

# Slot 15
ID_SLOT_15 = (0x900E0648, 0x9014ADF0)  # int2 (2 bytes) - Item ID in slot 15
NOITEM_SLOT_15 = (0x900E064A, 0x9014ADF2)  # int2 (2 bytes) - Quantity in slot 15

# Slot 16
ID_SLOT_16 = (0x900E064C, 0x9014ADF4)  # int2 (2 bytes) - Item ID in slot 16
NOITEM_SLOT_16 = (0x900E064E, 0x9014ADF6)  # int2 (2 bytes) - Quantity in slot 16

# Slot 17
ID_SLOT_17 = (0x900E0650, 0x9014ADF8)  # int2 (2 bytes) - Item ID in slot 17
NOITEM_SLOT_17 = (0x900E0652, 0x9014ADFA)  # int2 (2 bytes) - Quantity in slot 17

# Slot 18
ID_SLOT_18 = (0x900E0654, 0x9014ADFC)  # int2 (2 bytes) - Item ID in slot 18
NOITEM_SLOT_18 = (0x900E0656, 0x9014ADFE)  # int2 (2 bytes) - Quantity in slot 18

# Slot 19
ID_SLOT_19 = (0x900E0658, 0x9014AE00)  # int2 (2 bytes) - Item ID in slot 19
NOITEM_SLOT_19 = (0x900E065A, 0x9014AE02)  # int2 (2 bytes) - Quantity in slot 19

# Slot 20
ID_SLOT_20 = (0x900E065C, 0x9014AE04)  # int2 (2 bytes) - Item ID in slot 20
NOITEM_SLOT_20 = (0x900E065E, 0x9014AE06)  # int2 (2 bytes) - Quantity in slot 20

# Slot 21
ID_SLOT_21 = (0x900E0660, 0x9014AE08)  # int2 (2 bytes) - Item ID in slot 21
NOITEM_SLOT_21 = (0x900E0662, 0x9014AE0A)  # int2 (2 bytes) - Quantity in slot 21

# Slot 22
ID_SLOT_22 = (0x900E0664, 0x9014AE0C)  # int2 (2 bytes) - Item ID in slot 22
NOITEM_SLOT_22 = (0x900E0666, 0x9014AE0E)  # int2 (2 bytes) - Quantity in slot 22

# Slot 23
ID_SLOT_23 = (0x900E0668, 0x9014AE10)  # int2 (2 bytes) - Item ID in slot 23
NOITEM_SLOT_23 = (0x900E066A, 0x9014AE12)  # int2 (2 bytes) - Quantity in slot 23

# Slot 24
ID_SLOT_24 = (0x900E066C, 0x9014AE14)  # int2 (2 bytes) - Item ID in slot 24
NOITEM_SLOT_24 = (0x900E066E, 0x9014AE16)  # int2 (2 bytes) - Quantity in slot 24

# Currently Equipped Weapon
WEAPON_EQUIPED = None  # To find - equipped weapon identifier

# =============================================================================
# CATEGORY 4: QUEST & MONSTERS
# =============================================================================

# Quest Elements
QUEST_TIME_SPENT = 0x806C7CFC  # int4 (4 bytes) - Quest timer (raw value)
# Formula: raw_value / 30 = remaining time in seconds

# Small Monsters (minor enemies)
SMONSTER_NUMBER = None  # To find - total count of small monsters in zone
SMONSTER1_POSITION = None  # To find - position coordinates

# Small Monster HP Values
SMONSTER1_HP = 0x9014D710  # int4 (4 bytes) - Small monster 1 HP (raw value)
# Note: If unchanged when changing zones, it is possible that there is no monsters in zone
SMONSTER2_HP = 0x9014D714  # int4 (4 bytes) - Small monster 2 HP (raw value)
SMONSTER3_HP = 0x9014E228  # int4 (4 bytes) - Small monster 3 HP (raw value)
SMONSTER4_HP = 0x9014ED40  # int4 (4 bytes) - Small monster 4 HP (raw value)
SMONSTER5_HP = 0x9014CBF8  # int4 (4 bytes) - Small monster 5 HP (raw value)
SMONSTER6_HP = None  # To find - More Small monster HP

# Large Monsters (boss enemies, main hunt targets)
NUMBER_OF_LMONSTER = None  # To find - number of large monsters in quest
LMONSTER1_POS = None  # To find - large monster 1 position
LMONSTER1_HP = None  # To find - large monster 1 HP


# =============================================================================
# ADDRESS VALIDATION (optional utility)
# =============================================================================

def validate_addresses():
    """
    Verify that memory addresses are within valid Dolphin Memory Engine ranges.
    (can detect some typing errors).

    Valid Wii memory ranges:
    - MEM1: 0x80000000 - 0x81800000 (24 MB - main RAM)
    - MEM2: 0x90000000 - 0x94000000 (64 MB - extended RAM)

    This function checks each defined address and reports whether it falls
    within these valid ranges. Addresses outside these ranges will not work
    with Dolphin Memory Engine.

    Returns:
        bool: True if all addresses are valid, False otherwise
    """
    # Dictionary of addresses to validate
    # Only includes non-None single addresses and quest addresses from tuples
    addresses = {
        'PLAYER_CURRENT_HP': PLAYER_CURRENT_HP,
        'PLAYER_RECOVERABLE_HP': PLAYER_RECOVERABLE_HP,
        'PLAYER_CURRENT_STAMINA': PLAYER_CURRENT_STAMINA,
        'PLAYER_STAMINA_MAX': PLAYER_STAMINA_MAX,
        'TIME_SPENT_UNDERWATER': TIME_SPENT_UNDERWATER,
        'DAMAGE_RECEIVE_LAST_HIT': DAMAGE_RECEIVE_LAST_HIT,
        'PLAYER_X': PLAYER_X,
        'PLAYER_Y': PLAYER_Y,
        'PLAYER_Z': PLAYER_Z,
        'CURRENT_ZONE': CURRENT_ZONE,
        'SHARPNESS': SHARPNESS,
        'QUEST_TIME_SPENT': QUEST_TIME_SPENT,
    }

    print("🔍 Dolphin Memory Engine Address Validation:\n")

    all_valid = True
    for name, addr in addresses.items():
        # Handle tuple addresses (village, quest) - use quest address for validation
        if isinstance(addr, tuple):
            addr = addr[1]  # Use quest address (second element)

        # Check if address is in MEM1 range (main RAM)
        is_mem1 = 0x80000000 <= addr < 0x81800000

        # Check if address is in MEM2 range (extended RAM)
        is_mem2 = 0x90000000 <= addr < 0x94000000

        # Display validation result
        if is_mem1 or is_mem2:
            mem_type = "MEM1" if is_mem1 else "MEM2"
            print(f"✅ {name:30s} : {hex(addr)} ({mem_type})")
        else:
            print(f"❌ {name:30s} : {hex(addr)} (INVALID - outside DME ranges)")
            all_valid = False

    # Print summary
    if all_valid:
        print("\n✅ All addresses are within valid Dolphin Memory Engine ranges")
    else:
        print("\n⚠️ Some addresses are outside valid DME ranges and may not work")

    return all_valid


# =============================================================================
# MAIN EXECUTION (runs when script is executed directly)
# =============================================================================

if __name__ == "__main__":
    print("📋 MEMORY ADDRESS CONFIGURATION FOR MONSTER HUNTER TRI\n")
    print("=" * 70)

    # Run address validation
    validate_addresses()

    print("\n" + "=" * 70)
    print("💡 IMPORTANT REMINDERS:")
    print("=" * 70)
    print("1. Some values read may appear STRANGE (3.4e37, etc.)")
    print("2. This is NORMAL")
    print("3. No need to normalize for all- The neural network can handles raw values. The AI will learns patterns (↑/↓)")
    print("4. Run check_setup.py to verify that memory reading works correctly")
    print("=" * 70)