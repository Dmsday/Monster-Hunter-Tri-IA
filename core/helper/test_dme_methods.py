"""
Teste TOUTES les méthodes possibles de dolphin-memory-engine
Pour identifier laquelle fonctionne
"""

import sys

print("=" * 70)
print("🧪 TEST DE TOUTES LES MÉTHODES DME")
print("=" * 70)

# Import
print("\n1️⃣ Import...")
try:
    import dolphin_memory_engine as dme

    print("   ✅ Importé")
except ImportError:
    print("   ❌ Non installé!")
    sys.exit(1)

# Lister toutes les fonctions disponibles
print("\n2️⃣ Fonctions disponibles dans DME:")
functions = [name for name in dir(dme) if not name.startswith('_')]
for func in functions:
    print(f"   - {func}")

# Test 1: hook()
print("\n3️⃣ Test: dme.hook()")
try:
    result = dme.hook()
    print(f"   Retour: {result} (type: {type(result)})")

    if result is None:
        print("   ⚠️ Retourne None (bug de version!)")
    elif result == True:
        print("   ✅ Retourne True (connecté!)")
    elif result == False:
        print("   ❌ Retourne False (échec)")
except Exception as e:
    print(f"   ❌ Exception: {e}")

# Test 2: is_hooked() (si existe)
print("\n4️⃣ Test: dme.is_hooked() (si existe)")
if hasattr(dme, 'is_hooked'):
    try:
        result = dme.is_hooked()
        print(f"   Retour: {result}")

        if result:
            print("   ✅ DME considère être connecté!")
        else:
            print("   ❌ Pas connecté selon is_hooked()")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
else:
    print("   ⚠️ Fonction is_hooked() n'existe pas")

# Test 3: Lecture directe (même si hook a échoué)
print("\n5️⃣ Test: Lecture directe (ignorer hook)")
try:
    # Essayer de lire quand même
    test_addr = 0x80000000
    print(f"   Tentative lecture à 0x{test_addr:X}...")

    value = dme.read_byte(test_addr)
    print(f"   ✅ Lecture réussie: {value}")
    print(f"   🎉 DME FONCTIONNE même si hook() retourne None!")

except Exception as e:
    print(f"   ❌ Lecture échoue: {e}")

# Test 4: Lecture d'une adresse de jeu
print("\n6️⃣ Test: Lecture d'adresse de jeu")
game_addresses = {
    'Current Zone': 0x806BAC64,
    'Player Money': 0x900E0588,
}

success_count = 0

for name, addr in game_addresses.items():
    try:
        value = dme.read_word(addr)
        print(f"   ✅ {name} (0x{addr:X}): {value}")
        success_count += 1
    except Exception as e:
        print(f"   ❌ {name} (0x{addr:X}): {e}")

if success_count > 0:
    print(f"\n   🎉 {success_count}/{len(game_addresses)} lectures réussies!")
    print(f"   ✅ DME FONCTIONNE RÉELLEMENT!")
else:
    print(f"\n   ❌ Aucune lecture réussie")
    print(f"   💡 Tu n'es peut-être pas EN JEU?")

# Test 5: Vérifier les processus
print("\n7️⃣ Processus Dolphin actifs:")
try:
    import psutil

    for proc in psutil.process_iter(['pid', 'name']):
        if 'dolphin' in proc.info['name'].lower():
            print(f"   - {proc.info['name']} (PID {proc.pid})")

            # Avertir si DolphinMemoryEngine.exe est ouvert
            if 'dolphinmemoryengine' in proc.info['name'].lower():
                print(f"      ⚠️ Ce processus peut INTERFÉRER!")
                print(f"         Ferme-le et garde seulement Dolphin.exe")
except:
    pass

# RÉSUMÉ
print("\n" + "=" * 70)
print("📋 RÉSUMÉ")
print("=" * 70)

# Détecter le cas spécial: hook() retourne None mais lectures fonctionnent
print("\n🔍 DIAGNOSTIC:")

if result is None:
    print("   ⚠️ dme.hook() retourne None (bug de version)")
    print("\n   💡 SOLUTIONS:")
    print("      1. Réinstalle depuis GitHub:")
    print("         pip uninstall dolphin-memory-engine")
    print("         pip install git+https://github.com/henriquegemignani/py-dolphin-memory-engine.git")
    print("\n      2. OU ignore hook() et utilise directement:")
    print("         # Ne PAS appeler dme.hook()")
    print("         # Juste utiliser dme.read_byte(), etc.")
    print("         # (peut fonctionner quand même!)")

if success_count > 0:
    print("\n   ✅ Les lectures FONCTIONNENT malgré le bug de hook()!")
    print("      → Solution: Modifier le code pour NE PAS vérifier hook()")
    print("      → Utiliser directement les fonctions read_*()")

print("\n" + "=" * 70)