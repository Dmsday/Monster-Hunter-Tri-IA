"""
Configuration des adresses mémoire pour Monster Hunter Tri (Wii)
Types de données précis (2 bytes vs 4 bytes)

TYPES DE DONNÉES:
- float : 4 bytes, big-endian, IEEE 754
- int4 : 4 bytes, big-endian, signé ou non signé
- int2 : 2 bytes, big-endian, signé ou non signé
- byte : 1 byte

    LES VALEURS LUES PEUVENT ÊTRE BIZARRES (3.4e37, etc.)
    C'est NORMAL - l'IA apprend les patterns, pas les valeurs absolues
"""

# =============================================================================
# CATÉGORIE 1 : JOUEUR - STATISTIQUES
# =============================================================================

# Points de vie
PLAYER_CURRENT_HP = 0x9014AEAF  # float (4 bytes) - VALEUR BRUTE
PLAYER_RECOVERABLE_HP = 0x9014AEB3  # float (4 bytes) - HP rouges VALEUR BRUTE

# Stamina
PLAYER_CURRENT_STAMINA = 0x9014AEB8  # int4 (4 bytes) - VALEUR BRUTE
PLAYER_STAMINA_MAX = 0x806C02A8  # float (4 bytes) - VALEUR BRUTE

# Oxygène en 2 bytes
TIME_SPENT_UNDERWATER = 0x9014AEBE  # int2 (2 bytes) - 100 = max puis diminue ; < 25 = alerte

# Dégâts reçus - FLAG de changement d'état, pas une valeur
DAMAGE_RECEIVE_LAST_HIT = 0x9014AED8  # float (4 bytes) - FLAG : si valeur change = coup reçu
# ⚠️ NE PAS UTILISER LA VALEUR - juste détecter changement d'état

# Mort
DEATH_COUNTER = 0x9014AF86  # byte (1 byte) - Nombre de morts (1,2,3...)

# Arme et armure
ATTACK_AND_DEFENSE_VALUE = 0x9014AEF8  # int2 (2 bytes) - Combiné attaque/défense
SHARPNESS = 0x9014B0AC # int2 (2 bytes) - -1 = coup réussi ; -2 = rebond ; inférieur ou égal à 100 = baisse de tranchant, utiliser id item 98 pour augmenter

# In game menu
IN_GAME_MENU_IS_OPEN = 0x806AED56 # byte (1 byte) - fermé = 0, ouvert = 1

# =============================================================================
# CATÉGORIE 2 : JOUEUR - POSITION & ORIENTATION
# =============================================================================

# Position dans le monde 3D - valeurs brutes entre -1 et 1
PLAYER_X = 0x900AD764  # float
PLAYER_Y = 0x900AD75C  # float
PLAYER_Z = 0x900B00D8  # float
CAMERA_Z = 0x900AD748  # float
PLAYER_NS_ORIENTATION = 0x8066E410 # float entre -1 (nord) et 1 (sud)
PLAYER_EW_ORIENTATION = 0x8066E418 # float entre -1 (ouest) et 1 (est)

# Emplacement actuel
CURRENT_ZONE = 0x806BAC64  # byte (1 byte) - ID de la zone
CURRENT_MAP = 0x92D1A80C # int 4 bytes
REAL_MAP_ID = 0x90375E30 # int 4 bytes - 1 = village


# INPUT manette


# =============================================================================
# CATÉGORIE 3 : INVENTAIRE & ÉCONOMIE
# =============================================================================

# Économie
PLAYER_MONEY = 0x900E0588  # int4 (4 bytes) - Zennys
PLAYER_MONEY_SPENT = 0x900E058B  # byte (1 byte) - Dépensé
MOGA_POINT = 0x900E4474  # int4 (4 bytes) - Points Moga

#Objet sélectionner
ITEM_SELECTED = 0x9014AE44 # int 2 bytes - numero de slot dans la bourse -1 de l'item selectionné, mod24 (rien = 24, item slot 1 = 0, item slot 2 = 1...)

# Emplacement sacoche (village, quête)
# Format : (adresse_village, adresse_quête) pour chaque slot

# Slot 1
ID_SLOT_1 = (0x900E0610, 0x9014ADB8)  # int2 (2 bytes) - ID de l'item
NOITEM_SLOT_1 = (0x900E0612, 0x9014ADBA)  # int2 (2 bytes) - Quantité

# Slot 2
ID_SLOT_2 = (0x900E0614, 0x9014ADBC)
NOITEM_SLOT_2 = (0x900E0616, 0x9014ADBE)

# Slot 3
ID_SLOT_3 = (0x900E0618, 0x9014ADC0)
NOITEM_SLOT_3 = (0x900E061A, 0x9014ADC2)

# Slot 4
ID_SLOT_4 = (0x900E061C, 0x9014ADC4)
NOITEM_SLOT_4 = (0x900E061E, 0x9014ADC6)

# Slot 5
ID_SLOT_5 = (0x900E0620, 0x9014ADC8)
NOITEM_SLOT_5 = (0x900E0622, 0x9014ADCA)

# Slot 6
ID_SLOT_6 = (0x900E0624, 0x9014ADCC)
NOITEM_SLOT_6 = (0x900E0626, 0x9014ADCE)

# Slot 7
ID_SLOT_7 = (0x900E0628, 0x9014ADD0)
NOITEM_SLOT_7 = (0x900E062A, 0x9014ADD2)

# Slot 8
ID_SLOT_8 = (0x900E062C, 0x9014ADD4)
NOITEM_SLOT_8 = (0x900E062E, 0x9014ADD6)

# Slot 9
ID_SLOT_9 = (0x900E0630, 0x9014ADD8)
NOITEM_SLOT_9 = (0x900E0632, 0x9014ADDA)

# Slot 10
ID_SLOT_10 = (0x900E0634, 0x9014ADDC)
NOITEM_SLOT_10 = (0x900E0636, 0x9014ADDE)

# Slot 11
ID_SLOT_11 = (0x900E0638, 0x9014ADE0)
NOITEM_SLOT_11 = (0x900E063A, 0x9014ADE2)

# Slot 12
ID_SLOT_12 = (0x900E063C, 0x9014ADE4)
NOITEM_SLOT_12 = (0x900E063E, 0x9014ADE6)

# Slot 13
ID_SLOT_13 = (0x900E0640, 0x9014ADE8)
NOITEM_SLOT_13 = (0x900E0642, 0x9014ADEA)

# Slot 14
ID_SLOT_14 = (0x900E0644, 0x9014ADEC)
NOITEM_SLOT_14 = (0x900E0646, 0x9014ADEE)

# Slot 15
ID_SLOT_15 = (0x900E0648, 0x9014ADF0)
NOITEM_SLOT_15 = (0x900E064A, 0x9014ADF2)

# Slot 16
ID_SLOT_16 = (0x900E064C, 0x9014ADF4)
NOITEM_SLOT_16 = (0x900E064E, 0x9014ADF6)

# Slot 17
ID_SLOT_17 = (0x900E0650, 0x9014ADF8)
NOITEM_SLOT_17 = (0x900E0652, 0x9014ADFA)

# Slot 18
ID_SLOT_18 = (0x900E0654, 0x9014ADFC)
NOITEM_SLOT_18 = (0x900E0656, 0x9014ADFE)

# Slot 19
ID_SLOT_19 = (0x900E0658, 0x9014AE00)
NOITEM_SLOT_19 = (0x900E065A, 0x9014AE02)

# Slot 20
ID_SLOT_20 = (0x900E065C, 0x9014AE04)
NOITEM_SLOT_20 = (0x900E065E, 0x9014AE06)

# Slot 21
ID_SLOT_21 = (0x900E0660, 0x9014AE08)
NOITEM_SLOT_21 = (0x900E0662, 0x9014AE0A)

# Slot 22
ID_SLOT_22 = (0x900E0664, 0x9014AE0C)
NOITEM_SLOT_22 = (0x900E0666, 0x9014AE0E)

# Slot 23
ID_SLOT_23 = (0x900E0668, 0x9014AE10)
NOITEM_SLOT_23 = (0x900E066A, 0x9014AE12)

# Slot 24
ID_SLOT_24 = (0x900E066C, 0x9014AE14)
NOITEM_SLOT_24 = (0x900E066E, 0x9014AE16)

# Arme equipe
WEAPON_EQUIPED = None

# =============================================================================
# CATÉGORIE 4 : QUÊTE & MONSTRES
# =============================================================================

#Éléments de quête
QUEST_TIME_SPENT = 0x806C7CFC # valeur 4 bytes ; raw/30 = temps restant en seconde

#petits monstres
SMONSTER_NUMBER = None
SMONSTER1_POSITION = None
SMONSTER1_HP = 0x9014D710 # int 4 bytes brut - si ne change pas lorsque changement de zone alors aucun monstre dans la zone
SMONSTER2_HP = 0x9014D714
SMONSTER3_HP = 0x9014E228
SMONSTER4_HP = 0x9014ED40
SMONSTER5_HP = 0x9014CBF8
SMONSTER6_HP = None

#grand monstre
NUMBER_OF_LMONSTER = None
LMONSTER1_POS = None
LMONSTER1_HP = None

# =============================================================================
# VALIDATION DES ADRESSES (optionnel)
# =============================================================================

def validate_addresses():
    """
    Vérifie que les adresses sont dans des plages valides
    DME : 0x80000000 - 0x81800000 (MEM1) ou 0x90000000 - 0x94000000 (MEM2)
    """
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

    print("🔍 Validation des adresses DME:\n")

    all_valid = True
    for name, addr in addresses.items():
        # Gérer tuples (village, quête)
        if isinstance(addr, tuple):
            addr = addr[1]  # Prendre adresse quête

        # Vérifier range MEM1
        is_mem1 = 0x80000000 <= addr < 0x81800000
        # Vérifier range MEM2
        is_mem2 = 0x90000000 <= addr < 0x94000000

        if is_mem1 or is_mem2:
            mem_type = "MEM1" if is_mem1 else "MEM2"
            print(f"✅ {name:30s} : {hex(addr)} ({mem_type})")
        else:
            print(f"❌ {name:30s} : {hex(addr)} (INVALIDE)")
            all_valid = False

    if all_valid:
        print("\n✅ Toutes les adresses sont dans des ranges valides")
    else:
        print("\n⚠️ Certaines adresses sont hors des ranges DME")

    return all_valid


if __name__ == "__main__":
    print("📋 CONFIGURATION DES ADRESSES MÉMOIRE\n")
    print("=" * 70)

    validate_addresses()

    print("\n" + "=" * 70)
    print("💡 RAPPEL IMPORTANT:")
    print("=" * 70)
    print("1. Les valeurs lues peuvent être BIZARRES (3.4e37, etc.)")
    print("2. C'est NORMAL - l'IA apprend les patterns (↑/↓)")
    print("3. Pas besoin de normaliser - l'IA se débrouille")
    print("4. Lance check_setup.py pour vérifier que ça fonctionne")
    print("=" * 70)