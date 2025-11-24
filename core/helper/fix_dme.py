"""
Script pour réinstaller dolphin-memory-engine correctement
Avec la bonne version compatible Python 3.13
"""

import subprocess
import sys

print("=" * 70)
print("🔧 RÉINSTALLATION DE DOLPHIN-MEMORY-ENGINE")
print("=" * 70)

print("\n⚠️ PROBLÈME DÉTECTÉ:")
print("   - Python 3.13.2 détecté")
print("   - dolphin-memory-engine retourne None (bug de compatibilité)")
print("   - Besoin de la version corrigée")

print("\n1️⃣ Désinstallation de l'ancienne version...")
try:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "dolphin-memory-engine"
    ])
    print("   ✅ Désinstallation OK")
except Exception as e:
    print(f"   ⚠️ Erreur: {e}")

print("\n2️⃣ Installation de la version compatible...")

# Essayer plusieurs sources
sources = [
    # Source 1: Version GitHub la plus récente
    ("GitHub (recommandé)", "git+https://github.com/henriquegemignani/py-dolphin-memory-engine.git"),

    # Source 2: PyPI (peut être ancienne)
    ("PyPI (backup)", "dolphin-memory-engine"),
]

for name, source in sources:
    print(f"\n   Essai: {name}...")
    try:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            source
        ])
        print(f"   ✅ Installation réussie depuis {name}!")
        break
    except Exception as e:
        print(f"   ❌ Échec: {e}")
        continue
else:
    print("\n❌ Toutes les installations ont échoué!")
    print("\n💡 SOLUTION MANUELLE:")
    print("   1. Ouvre un terminal ADMIN")
    print("   2. Lance:")
    print("      pip uninstall dolphin-memory-engine")
    print("      pip install git+https://github.com/henriquegemignani/py-dolphin-memory-engine.git")
    sys.exit(1)

print("\n3️⃣ Vérification de l'installation...")
try:
    import dolphin_memory_engine as dme

    print("   ✅ Import OK")

    # Tester les fonctions
    if hasattr(dme, 'hook'):
        print("   ✅ Fonction hook() présente")
    else:
        print("   ❌ Fonction hook() manquante!")

    if hasattr(dme, 'is_hooked'):
        print("   ✅ Fonction is_hooked() présente")
    else:
        print("   ⚠️ Fonction is_hooked() manquante (peut être normal)")

except ImportError as e:
    print(f"   ❌ Import échoué: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ RÉINSTALLATION TERMINÉE")
print("=" * 70)

print("\n💡 PROCHAINES ÉTAPES:")
print("   1. Ferme DolphinMemoryEngine.exe standalone (si ouvert)")
print("   2. Garde uniquement Dolphin.exe EN ADMIN")
print("   3. Charge Monster Hunter Tri")
print("   4. Va EN JEU dans une quête")
print("   5. Relance: python diagnose_dme_advanced.py")

print("\n⚠️ IMPORTANT:")
print("   Ferme le processus 'DolphinMemoryEngine.exe' (PID 32468)")
print("   Il peut interférer avec py-dolphin-memory-engine")
print("   Garde SEULEMENT Dolphin.exe")