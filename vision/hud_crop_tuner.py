"""
Outil interactif pour ajuster le crop des HUD
Utilise OpenCV pour afficher en temps réel et ajuster avec le clavier
"""

import cv2
import numpy as np
import json
import os
from vision.frame_capture import FrameCapture


class HUDCropTuner:
    """
    Outil interactif pour calibrer le crop des HUD

    Contrôles clavier:
    - W/S : Ajuster top crop (haut)
    - A/D : Ajuster left crop (gauche)
    - I/K : Ajuster bottom crop (bas)
    - J/L : Ajuster right crop (droite)
    - R : Reset aux valeurs par défaut
    - ESPACE : Capturer une nouvelle frame
    - ENTRÉE : Sauvegarder et quitter
    - ESC : Quitter sans sauvegarder
    """

    def __init__(self):
        self.capturer = None
        self.current_frame = None

        # Valeurs de crop (proportions 0.0 - 1.0)
        self.top_crop = 0.12
        self.bottom_crop = 0.15
        self.left_crop = 0.05
        self.right_crop = 0.05

        # Incrément d'ajustement
        self.step = 0.01  # 1%

        # Fenêtre
        self.window_name = "HUD Crop Tuner - Monster Hunter"

        # Config par défaut
        self.default_config = {
            'top_crop': 0.12,
            'bottom_crop': 0.15,
            'left_crop': 0.05,
            'right_crop': 0.05
        }

    def capture_frame_from_dolphin(self):
        """Capture une frame depuis Dolphin"""
        try:
            if self.capturer is None:
                print("\n📸 Connexion à Dolphin...")
                self.capturer = FrameCapture(window_name="Dolphin")

            frame = self.capturer.capture_frame()

            if frame is None or frame.size == 0:
                print("❌ Frame capturée vide!")
                return None

            print(f"✅ Frame capturée: {frame.shape}")
            return frame

        except ValueError as e:
            print(f"❌ Erreur: {e}")
            print("\n💡 Assure-toi que:")
            print("   - Dolphin est lancé")
            print("   - Un jeu est en cours")
            print("   - La fenêtre Dolphin est visible")
            return None
        except Exception as e:
            print(f"❌ Erreur inattendue: {e}")
            return None

    def draw_crop_overlay(self, frame):
        """
        Dessine l'overlay du crop sur la frame

        Args:
            frame: Image RGB

        Returns:
            Frame avec overlay dessiné
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Calculer les coordonnées du crop
        top = int(h * self.top_crop)
        bottom = int(h * (1 - self.bottom_crop))
        left = int(w * self.left_crop)
        right = int(w * (1 - self.right_crop))

        # Assombrir les zones à enlever
        overlay = display_frame.copy()

        # Zone haut (rouge foncé)
        cv2.rectangle(overlay, (0, 0), (w, top), (100, 0, 0), -1)

        # Zone bas (rouge foncé)
        cv2.rectangle(overlay, (0, bottom), (w, h), (100, 0, 0), -1)

        # Zone gauche (rouge foncé)
        cv2.rectangle(overlay, (0, top), (left, bottom), (100, 0, 0), -1)

        # Zone droite (rouge foncé)
        cv2.rectangle(overlay, (right, top), (w, bottom), (100, 0, 0), -1)

        # Appliquer transparence
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

        # Dessiner le rectangle de la zone conservée (vert)
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 3)

        # Ajouter les lignes de crop (jaunes pointillées)
        # Ligne haut
        for x in range(0, w, 20):
            cv2.line(display_frame, (x, top), (min(x + 10, w), top), (0, 255, 255), 2)

        # Ligne bas
        for x in range(0, w, 20):
            cv2.line(display_frame, (x, bottom), (min(x + 10, w), bottom), (0, 255, 255), 2)

        # Ligne gauche
        for y in range(top, bottom, 20):
            cv2.line(display_frame, (left, y), (left, min(y + 10, bottom)), (0, 255, 255), 2)

        # Ligne droite
        for y in range(top, bottom, 20):
            cv2.line(display_frame, (right, y), (right, min(y + 10, bottom)), (0, 255, 255), 2)

        # Ajouter texte avec les valeurs
        self._add_info_text(display_frame)

        return display_frame

    def _add_info_text(self, frame):
        """Ajoute les infos textuelles sur la frame"""
        h, w = frame.shape[:2]

        # Fond semi-transparent pour le texte
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 220), (400, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        y = h - 195
        line_height = 20

        texts = [
            "=== HUD CROP TUNER ===",
            f"Top:    {self.top_crop:.2f} (W/S)",
            f"Bottom: {self.bottom_crop:.2f} (I/K)",
            f"Left:   {self.left_crop:.2f} (A/D)",
            f"Right:  {self.right_crop:.2f} (J/L)",
            "",
            "R: Reset | ESPACE: Capturer",
            "ENTREE: Sauvegarder | ESC: Quitter"
        ]

        for text in texts:
            cv2.putText(frame, text, (20, y), font, font_scale, color, thickness)
            y += line_height

        # Afficher dimensions de la zone croppée
        top_px = int(h * self.top_crop)
        bottom_px = int(h * (1 - self.bottom_crop))
        left_px = int(w * self.left_crop)
        right_px = int(w * (1 - self.right_crop))

        crop_w = right_px - left_px
        crop_h = bottom_px - top_px

        cv2.putText(
            frame,
            f"Zone: {crop_w}x{crop_h}px",
            (w - 180, 30),
            font,
            0.6,
            (0, 255, 0),
            2
        )

    def handle_key(self, key):
        """
        Handles keyboard inputs

        Args:
            key: OpenCV keycode

        Returns:
            action: 'continue', 'save', 'quit'
        """
        # W/S - Top crop
        if key == ord('w') or key == ord('W'):
            self.top_crop = max(0.0, self.top_crop - self.step)
            print(f"Top crop: {self.top_crop:.2f}")

        elif key == ord('s') or key == ord('S'):
            self.top_crop = min(0.5, self.top_crop + self.step)
            print(f"Top crop: {self.top_crop:.2f}")

        # I/K - Bottom crop
        elif key == ord('i') or key == ord('I'):
            self.bottom_crop = max(0.0, self.bottom_crop - self.step)
            print(f"Bottom crop: {self.bottom_crop:.2f}")

        elif key == ord('k') or key == ord('K'):
            self.bottom_crop = min(0.5, self.bottom_crop + self.step)
            print(f"Bottom crop: {self.bottom_crop:.2f}")

        # A/D - Left crop
        elif key == ord('a') or key == ord('A'):
            self.left_crop = max(0.0, self.left_crop - self.step)
            print(f"Left crop: {self.left_crop:.2f}")

        elif key == ord('d') or key == ord('D'):
            self.left_crop = min(0.5, self.left_crop + self.step)
            print(f"Left crop: {self.left_crop:.2f}")

        # J/L - Right crop
        elif key == ord('j') or key == ord('J'):
            self.right_crop = max(0.0, self.right_crop - self.step)
            print(f"Right crop: {self.right_crop:.2f}")

        elif key == ord('l') or key == ord('L'):
            self.right_crop = min(0.5, self.right_crop + self.step)
            print(f"Right crop: {self.right_crop:.2f}")

        # R - Reset
        elif key == ord('r') or key == ord('R'):
            self.reset_to_default()
            print("🔄 Reset aux valeurs par défaut")

        # ESPACE - Nouvelle capture
        elif key == ord(' '):
            print("\n📸 Capture d'une nouvelle frame...")
            frame = self.capture_frame_from_dolphin()
            if frame is not None:
                self.current_frame = frame
                print("✅ Nouvelle frame capturée")
            return 'capture'

        # ENTRÉE - Sauvegarder
        elif key == 13 or key == 10:  # Enter
            return 'save'

        # ESC - Quitter
        elif key == 27:  # Escape
            return 'quit'

        return 'continue'

    def reset_to_default(self):
        """Reset aux valeurs par défaut"""
        self.top_crop = self.default_config['top_crop']
        self.bottom_crop = self.default_config['bottom_crop']
        self.left_crop = self.default_config['left_crop']
        self.right_crop = self.default_config['right_crop']

    def save_config(self, filepath='config/crop_config.json'):
        """Save crop configuration to config/ (tracked, not user-specific)"""
        config = {
            'top_crop': self.top_crop,
            'bottom_crop': self.bottom_crop,
            'left_crop': self.left_crop,
            'right_crop': self.right_crop
        }

        # Create config directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Configuration sauvegardée: {filepath}")
        print(f"   top_crop: {self.top_crop:.2f}")
        print(f"   bottom_crop: {self.bottom_crop:.2f}")
        print(f"   left_crop: {self.left_crop:.2f}")
        print(f"   right_crop: {self.right_crop:.2f}")

    def load_config(self, filepath='config/crop_config.json'):
        """Load crop configuration from config/ (tracked file)"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)

            self.top_crop = config['top_crop']
            self.bottom_crop = config['bottom_crop']
            self.left_crop = config['left_crop']
            self.right_crop = config['right_crop']

            print(f"📂 Configuration chargée: {filepath}")
            return True
        return False

    def run(self):
        """Lance l'outil interactif"""
        print("\n" + "=" * 70)
        print("🎯 HUD CROP TUNER - MONSTER HUNTER TRI")
        print("=" * 70)
        print("\n📋 INSTRUCTIONS:")
        print("   1. Assure-toi que Dolphin est lancé avec le jeu")
        print("   2. Va EN JEU (pas dans les menus)")
        print("   3. Utilise les touches pour ajuster le crop:")
        print("      - W/S : Haut")
        print("      - I/K : Bas")
        print("      - A/D : Gauche")
        print("      - J/L : Droite")
        print("      - R : Reset")
        print("      - ESPACE : Nouvelle capture")
        print("      - ENTRÉE : Sauvegarder et quitter")
        print("      - ESC : Quitter sans sauvegarder")
        print("\n💡 But: Ajuster pour enlever les HUD mais garder le monstre!\n")

        # Tenter de charger config existante
        self.load_config()

        input("Appuie sur ENTRÉE pour commencer...")

        # Capturer première frame
        print("\n📸 Capture de la première frame...")
        self.current_frame = self.capture_frame_from_dolphin()

        if self.current_frame is None:
            print("\n❌ Impossible de capturer une frame!")
            print("Vérifications:")
            print("   - Dolphin est-il lancé?")
            print("   - Le jeu est-il visible?")
            return

        # Créer fenêtre OpenCV
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        print("\n✅ Outil prêt! Ajuste le crop avec les touches...")
        print("   (Regarde la fenêtre OpenCV)\n")

        # Boucle principale
        while True:
            # Dessiner overlay
            display_frame = self.draw_crop_overlay(self.current_frame)

            # Convertir RGB -> BGR pour OpenCV
            display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

            # Afficher
            cv2.imshow(self.window_name, display_frame_bgr)

            # Attendre input (1ms refresh)
            key = cv2.waitKey(1) & 0xFF

            if key != 255:  # Si touche pressée
                action = self.handle_key(key)

                if action == 'save':
                    self.save_config()

                    # Save example image to vision/debug/
                    # Config folder should only contain configuration files
                    example_dir = os.path.join('vision', 'debug')
                    os.makedirs(example_dir, exist_ok=True)
                    example_path = os.path.join(example_dir, 'crop_example.png')
                    cv2.imwrite(example_path, display_frame_bgr)
                    print(f"📸 Example saved: {example_path}")

                    break

                elif action == 'quit':
                    print("\n👋 Annulé - configuration non sauvegardée")
                    break

        # Fermer fenêtre
        cv2.destroyAllWindows()

        print("\n✅ Fini!")


# ============================================================
# VERSION SIMPLIFIÉE SANS CAPTURE DOLPHIN (pour test)
# ============================================================

def run_with_test_image(image_path: str = None):
    """
    Version simplifiée avec une image de test
    Utile si tu ne peux pas capturer depuis Dolphin
    """
    run_tuner = HUDCropTuner()

    if image_path and os.path.exists(image_path):
        # Charger image fournie
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        run_tuner.current_frame = frame
        print(f"✅ Image chargée: {image_path}")
    else:
        # Créer image de test
        print("⚠️  Pas d'image fournie - création d'une image de test")
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Simuler des HUD
        # Barre de vie haut
        cv2.rectangle(frame, (50, 20), (300, 60), (255, 0, 0), -1)
        cv2.putText(frame, "HP: 100/150", (60, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Minimap gauche
        cv2.circle(frame, (80, 400), 60, (0, 255, 0), -1)
        cv2.putText(frame, "MAP", (55, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Items bas
        for i in range(5):
            x = 50 + i * 70
            cv2.rectangle(frame, (x, 650), (x + 50, 700), (100, 100, 100), -1)

        # Monstre au centre (à garder!)
        cv2.rectangle(frame, (500, 250), (800, 500), (150, 50, 200), -1)
        cv2.putText(frame, "MONSTRE", (550, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        run_tuner.current_frame = frame

    run_tuner.load_config()

    # Créer fenêtre
    cv2.namedWindow(run_tuner.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(run_tuner.window_name, 1280, 720)

    print("\n✅ Mode test actif - Ajuste avec les touches clavier")

    # Boucle
    while True:
        display_frame = run_tuner.draw_crop_overlay(run_tuner.current_frame)
        display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(run_tuner.window_name, display_frame_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            action = run_tuner.handle_key(key)

            if action == 'save':
                run_tuner.save_config()

                # Save to vision/debug/
                example_dir = os.path.join('vision', 'debug')
                os.makedirs(example_dir, exist_ok=True)
                example_path = os.path.join(example_dir, 'crop_example.png')
                cv2.imwrite(example_path, display_frame_bgr)
                print(f"📸 Example: {example_path}")
                break

            elif action == 'quit':
                print("\n👋 Annulé")
                break

    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    print("\n🎮 HUD CROP TUNER")
    print("\nChoisis un mode:")
    print("  1. Capture depuis Dolphin (recommandé)")
    print("  2. Mode test avec image simulée")

    if len(sys.argv) > 1:
        # Image fournie en argument
        run_with_test_image(sys.argv[1])
    else:
        choice = input("\nChoix (1 ou 2): ").strip()

        if choice == '1':
            tuner = HUDCropTuner()
            tuner.run()
        else:
            run_with_test_image()