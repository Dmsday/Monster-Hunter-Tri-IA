"""
Scanner pour trouver MEM1 et MEM2 dans Dolphin
Détecte automatiquement les offsets corrects
"""

import pymem
import psutil
import struct


def find_dolphin_memory_regions():
    """
    Scanne toutes les régions mémoire de Dolphin
    pour trouver MEM1 et MEM2
    """
    print("=" * 70)
    print("🔍 SCANNER DE MÉMOIRE DOLPHIN")
    print("=" * 70)

    # Connexion
    print("\n1️⃣ Connexion à Dolphin...")
    try:
        pm = pymem.Pymem("Dolphin.exe")
        print(f"   ✅ Connecté (PID: {pm.process_id})")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return

    # Scanner les régions
    print("\n2️⃣ Scan des régions mémoire...")
    process = psutil.Process(pm.process_id)

    mem_regions = []

    for region in process.memory_maps():
        try:
            size_mb = region.rss / (1024 * 1024)

            # Ne garder que les grosses régions (>10 MB)
            if size_mb > 10:
                addr_str = region.addr
                if isinstance(addr_str, str):
                    # Format: "0x7FF123456-0x7FF789ABC"
                    start_addr = int(addr_str.split('-')[0], 16)
                else:
                    start_addr = addr_str

                mem_regions.append({
                    'start': start_addr,
                    'size_mb': size_mb,
                    'perms': getattr(region, 'perms', '???')
                })
        except Exception:
            continue

    # Trier par taille
    mem_regions.sort(key=lambda x: x['size_mb'], reverse=True)

    print(f"\n📊 {len(mem_regions)} régions >10MB trouvées:\n")

    # Afficher les plus grosses
    for i, region in enumerate(mem_regions[:10], 1):
        print(f"   {i:2d}. {region['size_mb']:6.1f} MB @ 0x{region['start']:016X} [{region['perms']}]")

    # Chercher MEM1 et MEM2
    print("\n3️⃣ Identification MEM1 et MEM2...")

    mem1_candidate = None
    mem2_candidate = None

    for region in mem_regions:
        size_mb = region['size_mb']

        # MEM1 : ~24 MB (20-30 MB)
        if 20 < size_mb < 30 and mem1_candidate is None:
            mem1_candidate = region
            mem1_offset = region['start'] - 0x80000000
            print(f"\n   🎯 MEM1 candidat:")
            print(f"      Taille: {size_mb:.1f} MB")
            print(f"      Adresse Windows: 0x{region['start']:X}")
            print(f"      Offset: 0x{mem1_offset:X}")

        # MEM2 : ~64 MB (55-70 MB)
        elif 55 < size_mb < 70 and mem2_candidate is None:
            mem2_candidate = region
            mem2_offset = region['start'] - 0x90000000
            print(f"\n   🎯 MEM2 candidat:")
            print(f"      Taille: {size_mb:.1f} MB")
            print(f"      Adresse Windows: 0x{region['start']:X}")
            print(f"      Offset: 0x{mem2_offset:X}")

    if not mem1_candidate:
        print("\n   ❌ MEM1 non trouvé!")
        print("      💡 Cherche manuellement une région de ~24 MB")
        return None, None

    if not mem2_candidate:
        print("\n   ⚠️ MEM2 non trouvé (pas critique)")

    # Test de lecture
    print("\n4️⃣ Test de lecture sur MEM1...")

    mem1_offset = mem1_candidate['start'] - 0x80000000

    # Tester quelques adresses connues
    test_addresses = {
        'Zone (0x806BAC64)': 0x806BAC64,
        'Money (0x900E0588)': 0x900E0588,  # MEM2
    }

    success_count = 0

    for name, dme_addr in test_addresses.items():
        try:
            # Convertir DME → Windows
            if 0x80000000 <= dme_addr < 0x81800000:
                # MEM1
                if mem1_candidate:
                    real_addr = dme_addr - 0x80000000 + mem1_candidate['start']
                else:
                    continue
            elif 0x90000000 <= dme_addr < 0x94000000:
                # MEM2
                if mem2_candidate:
                    real_addr = dme_addr - 0x90000000 + mem2_candidate['start']
                else:
                    continue
            else:
                continue

            # Lire 4 bytes
            bytes_data = pm.read_bytes(real_addr, 4)
            value = struct.unpack('>i', bytes_data)[0]

            print(f"   ✅ {name}: {value}")
            success_count += 1

        except Exception as e:
            print(f"   ❌ {name}: Erreur - {e}")

    if success_count == 0:
        print("\n   ⚠️ Aucune lecture réussie")
        print("      💡 Causes:")
        print("         1. Pas EN JEU (dans un menu)")
        print("         2. Offsets incorrects")
        print("         3. Version Dolphin incompatible")
    else:
        print(f"\n   ✅ {success_count}/{len(test_addresses)} lectures réussies!")

    # Résumé
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ")
    print("=" * 70)

    if mem1_candidate:
        mem1_offset = mem1_candidate['start'] - 0x80000000
        print(f"\n✅ MEM1 trouvé:")
        print(f"   Offset à utiliser: 0x{mem1_offset:X}")
        print(f"   (Adresse: 0x{mem1_candidate['start']:X})")

    if mem2_candidate:
        mem2_offset = mem2_candidate['start'] - 0x90000000
        print(f"\n✅ MEM2 trouvé:")
        print(f"   Offset à utiliser: 0x{mem2_offset:X}")
        print(f"   (Adresse: 0x{mem2_candidate['start']:X})")

    print("\n💡 Copie ces offsets dans dynamic_memory_reader.py")
    print("   ou utilise le scanner automatique intégré")

    return (
        mem1_candidate['start'] if mem1_candidate else None,
        mem2_candidate['start'] if mem2_candidate else None
    )


def scan_for_game_signature():
    """
    Cherche une signature dans la mémoire pour confirmer
    qu'on est bien dans Monster Hunter Tri
    """
    print("\n5️⃣ Recherche signature du jeu...")

    try:
        pm = pymem.Pymem("Dolphin.exe")

        # Scanner pour des valeurs typiques
        # Par exemple, l'ID du jeu : "R3MP08" (Monster Hunter Tri PAL)
        game_id = b"R3MP08"

        # Ou des valeurs fixes connues
        # À compléter avec des signatures spécifiques au jeu

        print("   🔍 Scan de signatures... (à implémenter)")

    except Exception as e:
        print(f"   ❌ Erreur: {e}")


if __name__ == "__main__":
    mem1, mem2 = find_dolphin_memory_regions()

    if mem1:
        print("\n🎉 Scanner terminé avec succès!")
    else:
        print("\n⚠️ Problème détecté - vérifications nécessaires")