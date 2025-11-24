"""
Script de calibration HP/Stamina
Détecte automatiquement les valeurs min/max

UTILISATION :
1. Lance ce script
2. Dans le jeu :
   - Attends que stamina soit PLEINE (100%)
   - Cours jusqu'à stamina VIDE (0%)
   - Utilise une potion pour HP max
   - Prends des dégâts pour HP min
3. Le script affichera les vraies valeurs
"""

import time
from core.dynamic_memory_reader import MemoryReader


def calibrate():
    """Calibre les valeurs HP/Stamina"""

    print("=" * 70)
    print("🔧 CALIBRATION HP/STAMINA")
    print("=" * 70)

    print("\n📡 Connexion à Dolphin...")
    reader = MemoryReader(force_quest_mode=True)

    print("\n✅ Connecté !")
    print("\n" + "=" * 70)
    print("📋 INSTRUCTIONS")
    print("=" * 70)
    print("""
    1. Va EN JEU dans une quête

    2. Pour STAMINA :
       - Attends que stamina soit PLEINE (100%)
       - Appuie sur ENTRÉE
       - Cours/esquive jusqu'à stamina VIDE (0%)
       - Appuie sur ENTRÉE

    3. Pour HP :
       - Utilise une potion pour HP max
       - Appuie sur ENTRÉE
       - Prends des dégâts (minimum possible)
       - Appuie sur ENTRÉE

    4. Le script donnera les vraies valeurs min/max
    """)

    input("\n⏸️  Prêt ? ENTRÉE pour commencer...")

    # ============================================================
    # CALIBRATION STAMINA
    # ============================================================
    print("\n" + "=" * 70)
    print("⚡ CALIBRATION STAMINA")
    print("=" * 70)

    print("\n1️⃣ Attends que stamina soit PLEINE (100%)")
    print("   💡 Reste immobile quelques secondes")
    input("   ✅ Stamina pleine ? ENTRÉE...")

    # Lire stamina max
    stamina_samples_max = []
    print("\n   📊 Lecture de 10 échantillons...")
    for i in range(10):
        state = reader.read_game_state()
        stam = state['player_stamina_raw']
        if stam is not None:
            stamina_samples_max.append(stam)
            print(f"      {i + 1}/10 : {stam}")
        time.sleep(0.1)

    stamina_max = max(stamina_samples_max) if stamina_samples_max else None
    print(f"\n   🔍 STAMINA MAX détecté : {stamina_max}")

    print("\n2️⃣ Cours/esquive jusqu'à stamina VIDE (0%)")
    print("   💡 Continue jusqu'à être essoufflé")
    input("   ✅ Stamina vide ? ENTRÉE...")

    # Lire stamina min
    stamina_samples_min = []
    print("\n   📊 Lecture de 10 échantillons...")
    for i in range(10):
        state = reader.read_game_state()
        stam = state['player_stamina_raw']
        if stam is not None:
            stamina_samples_min.append(stam)
            print(f"      {i + 1}/10 : {stam}")
        time.sleep(0.1)

    stamina_min = min(stamina_samples_min) if stamina_samples_min else None
    print(f"\n   🔍 STAMINA MIN détecté : {stamina_min}")

    # ============================================================
    # CALIBRATION HP
    # ============================================================
    print("\n" + "=" * 70)
    print("❤️ CALIBRATION HP")
    print("=" * 70)

    print("\n3️⃣ Utilise une potion pour HP MAX (100%)")
    input("   ✅ HP pleins ? ENTRÉE...")

    # Lire HP max
    hp_samples_max = []
    print("\n   📊 Lecture de 10 échantillons...")
    for i in range(10):
        state = reader.read_game_state()
        hp = state['player_hp_raw']
        if hp is not None:
            hp_samples_max.append(hp)
            print(f"      {i + 1}/10 : {hp}")
        time.sleep(0.1)

    hp_max = max(hp_samples_max) if hp_samples_max else None
    print(f"\n   🔍 HP MAX détecté : {hp_max}")

    print("\n4️⃣ Prends des dégâts (le moins possible)")
    print("   💡 Fais-toi attaquer 1-2 fois")
    input("   ✅ HP bas ? ENTRÉE...")

    # Lire HP après dégâts
    hp_samples_min = []
    print("\n   📊 Lecture de 10 échantillons...")
    for i in range(10):
        state = reader.read_game_state()
        hp = state['player_hp_raw']
        if hp is not None:
            hp_samples_min.append(hp)
            print(f"      {i + 1}/10 : {hp}")
        time.sleep(0.1)

    hp_min = min(hp_samples_min) if hp_samples_min else None
    print(f"\n   🔍 HP après dégâts : {hp_min}")

    # ============================================================
    # RÉSULTATS
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ RÉSULTATS DE CALIBRATION")
    print("=" * 70)

    if stamina_max and stamina_min:
        print(f"\n⚡ STAMINA :")
        print(f"   MAX (pleine) : {stamina_max}")
        print(f"   MIN (vide)   : {stamina_min}")
        print(f"   Range        : {stamina_max - stamina_min}")

        if stamina_max == stamina_min:
            print(f"\n   ⚠️ PROBLÈME : Min = Max !")
            print(f"      Les valeurs ne changent pas")
            print(f"      💡 Vérifie l'adresse mémoire")
        else:
            print(f"\n   ✅ Valeurs cohérentes")
    else:
        print(f"\n❌ STAMINA : Échec de lecture")

    if hp_max and hp_min:
        print(f"\n❤️ HP :")
        print(f"   MAX (pleins)  : {hp_max}")
        print(f"   Après dégâts  : {hp_min}")
        print(f"   Range estimé  : {abs(hp_max - hp_min)}")

        # Pour HP, les valeurs sont négatives
        if hp_max < hp_min:
            print(f"\n   ⚠️ ATTENTION : Valeurs inversées (négatives)")
            print(f"      C'est normal si les valeurs sont négatives")
            print(f"      MAX doit être MOINS négatif que MIN")

        if abs(hp_max - hp_min) < 100:
            print(f"\n   ⚠️ Range très petit")
            print(f"      Prends plus de dégâts pour HP MIN")
        else:
            print(f"\n   ✅ Valeurs cohérentes")
    else:
        print(f"\n❌ HP : Échec de lecture")

    # ============================================================
    # CODE À COPIER
    # ============================================================
    print("\n" + "=" * 70)
    print("📝 CODE À COPIER DANS dynamic_memory_reader.py")
    print("=" * 70)

    if stamina_max and stamina_min:
        print(f"""
# Dans normalize_stamina(), remplace les valeurs :

MIN_STAMINA = {stamina_min}  # Stamina vide (calibré)
MAX_STAMINA = {stamina_max}  # Stamina pleine (calibré)
""")

    if hp_max and hp_min:
        print(f"""
# Dans normalize_hp(), remplace les valeurs :

MAX_HP = {hp_max}  # HP pleins (calibré)
MIN_HP = {hp_min}  # HP après dégâts (calibré)

# ⚠️ IMPORTANT : Si les valeurs sont POSITIVES (comme {hp_max}) :
# - Ignore les valeurs négatives dans les exemples
# - Utilise directement tes valeurs calibrées
# - MAX_HP doit être PLUS GRAND que MIN_HP

# ⚠️ Si tu veux HP à 0 exact :
# - Fais-toi tuer en quête
# - Relance ce script juste avant de mourir
# - Note la valeur MIN_HP quand HP = 0
""")

    print("\n💡 CONSEILS :")
    print("   1. Copie ces valeurs dans dynamic_memory_reader.py")
    print("   2. Remplace MIN/MAX dans normalize_hp() et normalize_stamina()")
    print("   3. Teste avec : python core/dynamic_memory_reader.py")
    print("   4. Les valeurs normalisées doivent bouger de 0 à 100")
    print("\n   ⚠️ Si ça ne marche toujours pas :")
    print("   - Les adresses mémoire sont peut-être incorrectes")
    print("   - Vérifie memory_addresses.py")
    print("   - Utilise Cheat Engine pour trouver les bonnes adresses")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        calibrate()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback

        traceback.print_exc()