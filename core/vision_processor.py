"""
Processeur vision avancé pour détecter des patterns visuels
Complémentaire au CNN pour extraire des features spécifiques
"""
from typing import Dict, Tuple

import cv2
import numpy as np

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('vision_processor')


class VisionProcessor:
    """
    Traitement vision spécialisé pour Monster Hunter
    Extrait des features que le CNN pourrait manquer:
    - Détection de monstre (couleur/mouvement)
    - Détection d'effets (explosions, coups)
    """

    def __init__(
            self,
            detect_monster: bool = True,
            detect_effects: bool = True,
            detect_ui_state: bool = True
    ):
        """
        Args:
            detect_monster: Activer détection de monstre
            detect_effects: Activer détection d'effets visuels
            detect_ui_state: Détecter l'état de l'UI (menu vs jeu)
        """
        self.detect_monster = detect_monster
        self.detect_effects = detect_effects
        self.detect_ui_state = detect_ui_state

        # Historique pour détection de mouvement
        self.prev_frame = None

        # Seuils de détection
        self.MOTION_THRESHOLD = 30
        self.BRIGHT_FLASH_THRESHOLD = 200

        logger.info("👁️  Vision Processor initialisé")

    def process_frame(self, process_frame: np.ndarray) -> Dict:
        """
        Traite une frame et extrait des features visuelles

        Args:
            process_frame: Image RGB (H, W, 3)

        Returns:
            Dict avec features détectées
        """
        frame_feature = {}

        if self.detect_monster:
            frame_feature['monster_detected'] = self._detect_monster(process_frame)
            frame_feature['monster_position'] = self._estimate_monster_position(process_frame)

        if self.detect_effects:
            frame_feature['hit_flash'] = self._detect_hit_flash(process_frame)
            frame_feature['motion_intensity'] = self._detect_motion(process_frame)

        if self.detect_ui_state:
            frame_feature['in_combat'] = self._detect_combat(process_frame)

        # Sauvegarder pour détection de mouvement
        self.prev_frame = process_frame.copy()

        return frame_feature

    @staticmethod
    def _detect_monster(monster_detection_frame: np.ndarray) -> bool:
        """
        Détecte la présence d'un monstre dans la frame

        Méthode simple: les monstres sont généralement des masses sombres
        ou colorées qui contrastent avec le décor

        Returns:
            True si monstre probablement présent
        """
        # Convertir en HSV pour meilleure détection de couleur
        hsv = cv2.cvtColor(monster_detection_frame, cv2.COLOR_RGB2HSV)

        # Masque pour détecter zones sombres/colorées (monstres)
        # Ajuster selon les monstres du jeu
        lower_dark = np.array([0, 30, 30])
        upper_dark = np.array([180, 255, 150])

        mask = cv2.inRange(hsv, lower_dark, upper_dark)

        # Calculer proportion de pixels détectés
        monster_pixels = np.sum(mask > 0)
        total_pixels = mask.size

        ratio = monster_pixels / total_pixels

        # Si plus de 10% de l'image correspond au profil = monstre présent
        return ratio > 0.1

    @staticmethod
    def _estimate_monster_position(monster_position_frame: np.ndarray) -> Tuple[float, float]:
        """
        Estime la position approximative du monstre

        Returns :
            (x, y) normalisés entre 0 et 1
        """
        # Convertir en grayscale
        gray = cv2.cvtColor(monster_position_frame, cv2.COLOR_RGB2GRAY)

        # Détecter contours
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.5, 0.5 # Centre par défaut

        # Trouver le plus gros contour (probablement le monstre)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculer centre de masse
        m = cv2.moments(largest_contour)
        if m['m00'] == 0:
            return 0.5, 0.5

        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']

        # Normaliser
        h, w = monster_position_frame.shape[:2]
        x_norm = cx / w
        y_norm = cy / h

        return x_norm, y_norm

    def _detect_hit_flash(self, flash_detection_frame: np.ndarray) -> bool:
        """
        Détecte un flash blanc (indicateur de coup réussi)

        Returns:
            True si flash détecté
        """
        # Convertir en grayscale
        gray = cv2.cvtColor(flash_detection_frame, cv2.COLOR_RGB2GRAY)

        # Calculer moyenne de luminosité
        mean_brightness = np.mean(gray)

        # Si très lumineux = flash
        return mean_brightness > self.BRIGHT_FLASH_THRESHOLD

    def _detect_motion(self, motion_detection_frame: np.ndarray) -> float:
        """
        Détecte l'intensité du mouvement dans la scène

        Returns:
            Intensité du mouvement (0.0 = statique, 1.0 = très dynamique)
        """
        if self.prev_frame is None:
            return 0.0

        # Convertir en grayscale
        gray_current = cv2.cvtColor(motion_detection_frame, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)

        # Différence entre frames
        frame_diff = cv2.absdiff(gray_current, gray_prev)

        # Calculer pourcentage de pixels en mouvement
        motion_pixels = np.sum(frame_diff > self.MOTION_THRESHOLD)
        total_pixels = frame_diff.size

        motion_ratio = motion_pixels / total_pixels

        # Normaliser entre 0 et 1
        return min(motion_ratio * 5, 1.0)  # x5 pour amplifier

    def _detect_combat(self, fight_detection_frame: np.ndarray) -> bool:
        """
        Détecte si on est en combat actif

        Indicateurs:
        - Mouvement élevé
        - Présence de monstre

        Returns:
            True si en combat
        """
        motion = self._detect_motion(fight_detection_frame)
        has_monster = self._detect_monster(fight_detection_frame)

        # Combat = mouvement + monstre
        return motion > 0.3 and has_monster

    def detect_hit_visual(self, success_hit_frame: np.ndarray) -> bool:
        """
        Détecte un coup réussi via analyse visuelle

        Combine plusieurs indicateurs:
        - Flash blanc
        - Particules (changement soudain de couleurs)
        - Secousse d'écran (mouvement brusque)

        Returns:
            True si coup probablement réussi
        """
        has_flash = self._detect_hit_flash(success_hit_frame)

        # Détecter changement soudain de couleur (particules)
        if self.prev_frame is not None:
            color_change = np.mean(np.abs(success_hit_frame.astype(float) - self.prev_frame.astype(float)))
            has_particles = color_change > 20
        else:
            has_particles = False

        return has_flash or has_particles

    def create_attention_map(self, heatmap_frame: np.ndarray) -> np.ndarray:
        """
        Crée une carte d'attention (heatmap) pour visualiser les zones importantes

        Args:
            heatmap_frame: Image RGB

        Returns:
            Heatmap (0-255, uint8)
        """
        h, w = heatmap_frame.shape[:2]
        attention_map = np.zeros((h, w), dtype=np.float32)

        # Convertir en grayscale
        gray = cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2GRAY)

        # Attention sur les contours (objets/monstres)
        edges = cv2.Canny(gray, 50, 150)
        attention_map += edges.astype(float)

        # Attention sur les zones de mouvement
        if self.prev_frame is not None:
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
            motion = cv2.absdiff(gray, gray_prev)
            attention_map += motion.astype(float)

        # Normaliser
        attention_map = (attention_map / attention_map.max() * 255).astype(np.uint8)

        # Appliquer colormap pour visualisation
        attention_colored = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

        return attention_colored

    def reset(self):
        """Reset pour un nouvel épisode"""
        self.prev_frame = None


# ============================================================
# INTÉGRATION AVEC L'ENVIRONNEMENT
# ============================================================

def integrate_vision_features(
        observation: Dict,
        vision_features: Dict
) -> Dict:
    """
    Ajoute les features visuelles à l'observation

    Args:
        observation: Observation actuelle (visual + memory)
        vision_features: Features extraites par VisionProcessor

    Returns:
        Observation augmentée
    """
    # Convertir features en vecteur
    feature_vector = [
        1.0 if vision_features.get('monster_detected', False) else 0.0,
        vision_features.get('monster_position', (0.5, 0.5))[0],
        vision_features.get('monster_position', (0.5, 0.5))[1],
        1.0 if vision_features.get('hit_flash', False) else 0.0,
        vision_features.get('motion_intensity', 0.0),
        1.0 if vision_features.get('in_menu', False) else 0.0,
        1.0 if vision_features.get('in_combat', False) else 0.0,
    ]

    # Ajouter au vecteur mémoire existant
    if 'memory' in observation:
        observation['memory'] = np.concatenate([
            observation['memory'],
            np.array(feature_vector, dtype=np.float32)
        ])

    return observation


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test du Vision Processor\n")

    processor = VisionProcessor(
        detect_monster=True,
        detect_effects=True,
        detect_ui_state=True
    )

    # Créer frames de test
    print("📸 Création de frames de test...\n")

    # Frame 1: Scène normale
    frame1 = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)

    # Frame 2: Flash blanc (coup)
    frame2 = np.full((480, 640, 3), 250, dtype=np.uint8)

    # Frame 3: Mouvement
    frame3 = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)

    # Tester chaque frame
    for i, frame in enumerate([frame1, frame2, frame3], 1):
        print(f"🖼️  Frame {i}:")
        frame_feature_test = processor.process_frame(frame)

        for key, value in frame_feature_test.items():
            print(f"   {key}: {value}")
        print()

    # Créer attention map
    print("🗺️  Création d'une attention map...")
    attention = processor.create_attention_map(frame1)
    print(f"   Shape: {attention.shape}")

    print("\n✅ Test réussi!")