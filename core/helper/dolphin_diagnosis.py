"""
Diagnostic APPROFONDI pour dolphin-memory-engine
Teste différentes méthodes de connexion
"""

import sys
import os
import time

print("=" * 70)
print("🔬 DIAGNOSTIC APPROFONDI - DOLPHIN MEMORY ENGINE")
print("=" * 70)

# 1. Vérifier l'import
print("\n1️⃣ Import de dolphin-memory-engine...")
try:
    import dolphin_memory_engine as dme

    print(f"   ✅ Importé depuis: {dme.__file__}")

    # Afficher la version si disponible
    if hasattr(dme, '__version__'):
        print(f"   📦 Version: {dme.__version__}")
    else:
        print(f"   ⚠️ Version inconnue")

except ImportError as e:
    print(f"   ❌ Erreur import: {e}")
    sys.exit(1)

# 2. Vérifier les privilèges
print("\n2️⃣ Vérification des privilèges...")
try:
    import ctypes

    is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    print(f"   Admin: {'✅ OUI' if is_admin else '❌ NON'}")

    if not is_admin:
        print(f"   ⚠️ Python n'est PAS en admin!")
        print(f"   💡 Solution: Relance PyCharm/cmd en ADMINISTRATEUR")
except:
    print(f"   ⚠️ Impossible de vérifier les privilèges")

# 3. Vérifier Dolphin
print("\n3️⃣ Vérification de Dolphin...")
try:
    import psutil

    dolphin_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if 'dolphin' in proc.info['name'].lower():
                dolphin_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not dolphin_procs:
        print(f"   ❌ Dolphin non détecté!")
        sys.exit(1)

    print(f"   ✅ {len(dolphin_procs)} processus Dolphin détecté(s)")
    for proc in dolphin_procs:
        print(f"      - PID {proc.pid}: {proc.info['name']}")

except ImportError:
    print(f"   ⚠️ psutil non installé - impossible de vérifier")

# 4. Test de hook - MÉTHODE 1 (basique)
print("\n4️⃣ Test de connexion - MÉTHODE 1 (hook simple)...")
try:
    print(f"   🔌 Appel de dme.hook()...")
    result = dme.hook()
    print(f"   Résultat: {result} (type: {type(result)})")

    if result:
        print(f"   ✅ Hook réussi!")

        # Test de lecture basique
        print(f"\n   🧪 Test de lecture basique...")
        try:
            test_val = dme.read_byte(0x80000000)
            print(f"      ✅ Lecture à 0x80000000: {test_val}")
        except Exception as e:
            print(f"      ❌ Erreur lecture: {e}")
    else:
        print(f"   ❌ Hook échoué (retour False)")

except Exception as e:
    print(f"   ❌ Exception: {e}")
    import traceback

    traceback.print_exc()

# 5. Test de hook - MÉTHODE 2 (avec retry)
print("\n5️⃣ Test de connexion - MÉTHODE 2 (avec retry)...")
for attempt in range(3):
    print(f"   Tentative {attempt + 1}/3...")
    try:
        # Unhook au cas où
        try:
            dme.un_hook()
        except:
            pass

        time.sleep(0.5)

        # Hook
        result = dme.hook()

        if result:
            print(f"   ✅ Hook réussi à la tentative {attempt + 1}!")
            break
        else:
            print(f"   ❌ Échec tentative {attempt + 1}")

    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    if attempt < 2:
        time.sleep(1)

# 6. Vérifier si le jeu est chargé
print("\n6️⃣ Vérification que le jeu est chargé...")

if result:
    # Tester des adresses spécifiques au jeu
    test_addresses = [
        (0x80000000, "MEM1 start"),
        (0x806BAC64, "Current Zone"),
        (0x90000000, "MEM2 start"),
    ]

    success_count = 0

    for addr, name in test_addresses:
        try:
            val = dme.read_byte(addr)
            print(f"   ✅ {name} (0x{addr:X}): {val}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {name} (0x{addr:X}): {e}")

    if success_count == 0:
        print(f"\n   ⚠️ Aucune lecture réussie!")
        print(f"   💡 Le jeu est-il vraiment chargé et EN JEU?")
    else:
        print(f"\n   ✅ {success_count}/{len(test_addresses)} lectures OK")

# 7. Informations système
print("\n7️⃣ Informations système...")
print(f"   Python: {sys.version.split()[0]}")
print(f"   OS: {sys.platform}")
print(f"   Architecture: {sys.maxsize > 2 ** 32 and '64-bit' or '32-bit'}")

# 8. Vérifier les DLLs
print("\n8️⃣ Vérification des Dépendances...")
try:
    # dolphin-memory-engine nécessite certaines DLLs
    import ctypes

    # Vérifier si les DLLs Windows sont accessibles
    kernel32 = ctypes.windll.kernel32
    print(f"   ✅ kernel32.dll accessible")

except Exception as e:
    print(f"   ⚠️ Problème DLLs: {e}")

# RÉSUMÉ FINAL
print("\n" + "=" * 70)
print("📋 RÉSUMÉ & SOLUTIONS")
print("=" * 70)

if result:
    print("\n✅ Hook DME RÉUSSI!")
    print("\n💡 Si les lectures échouent quand même:")
    print("   1. Tu n'es peut-être pas EN JEU (menu/pause)")
    print("   2. Les adresses sont incorrectes pour ta version")
    print("   3. Lance: python core/dynamic_memory_reader.py")
else:
    print("\n❌ Hook DME ÉCHOUÉ")
    print("\n🔧 SOLUTIONS À ESSAYER:")
    print("\n   Solution 1: Vérifier les privilèges")
    print("   ============")
    print("   1. Ferme TOUT (Dolphin + Python/PyCharm)")
    print("   2. Clic droit Dolphin.exe → 'Exécuter en tant qu'admin'")
    print("   3. Clic droit PyCharm → 'Exécuter en tant qu'admin'")
    print("   4. Relance ce script")

    print("\n   Solution 2: Réinstaller dolphin-memory-engine")
    print("   ============")
    print("   pip uninstall dolphin-memory-engine")
    print("   pip install dolphin-memory-engine")

    print("\n   Solution 3: Vérifier la version de Dolphin")
    print("   ============")
    print("   dolphin-memory-engine fonctionne mieux avec:")
    print("   - Dolphin 5.0 (stable)")
    print("   - Dolphin Beta/Dev récentes")
    print("   Si tu utilises une vieille version, mets à jour!")

    print("\n   Solution 4: Tester avec un autre jeu")
    print("   ============")
    print("   Essaie de lire la mémoire d'un autre jeu Wii")
    print("   pour vérifier que DME fonctionne")

    print("\n   Solution 5: Utiliser l'outil DME standalone")
    print("   ============")
    print("   Télécharge DolphinMemoryEngine.exe depuis:")
    print("   https://github.com/aldelaro5/Dolphin-memory-engine")
    print("   Lance-le et vérifie qu'il peut se connecter")

print("\n" + "=" * 70)