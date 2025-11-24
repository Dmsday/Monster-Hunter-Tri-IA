# APRÈS (test_frame.py ligne 1-10)
"""
Test pour vérifier que les frames changent vraiment
"""
import sys
import os
import time


# AJOUT : Ajouter le dossier parent au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.frame_capture import FrameCapture

def test_frame_refresh():
    """Fonction de test pytest valide"""
    capturer = FrameCapture(target_fps=30)

    print("🔬 Test de refresh des frames (10 captures)\n")

    checksums = []
    for i in range(10):
        frame = capturer.capture_frame()
        checksum = frame.sum()
        checksums.append(checksum)

        print(f"Frame {i + 1}: Checksum = {checksum:.0f}")

        if i > 0:
            diff = abs(checksums[i] - checksums[i - 1])
            if diff == 0:
                print(f"   ⚠️ IDENTIQUE à la frame précédente !")
            else:
                print(f"   ✅ Différente (diff: {diff:.0f})")

        time.sleep(0.1)

    # Analyse
    unique_checksums = len(set(checksums))
    print(f"\n📊 Résultat : {unique_checksums} frames uniques sur 10")

    if unique_checksums == 1:
        print("❌ PROBLÈME : Toutes les frames sont identiques !")
        print("   Causes possibles :")
        print("   1. Dolphin en pause / pas de focus")
        print("   2. Rate limiting trop agressif")
        print("   3. Cache Windows BitBlt")
    elif unique_checksums < 5:
        print("⚠️ Peu de variation - vérifier Dolphin")
    else:
        print("✅ Les frames changent correctement")

    assert unique_checksums > 1, "PROBLÈME : Toutes les frames sont identiques !"

# Lance directement avec :
if __name__ == "__main__":
    test_frame_refresh()