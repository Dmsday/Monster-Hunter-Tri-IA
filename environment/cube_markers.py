"""
Système de marquage des cubes pour l'entraînement IA
Types de marqueurs :
- Statique : position fixe (changement zone, eau)
- Dynamique : évolue dans le temps (emplacement monstre)
"""

import time
import threading

from enum import Enum
from typing import Dict, Tuple

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('cube_markers')


# Ajouter ici tous les types de marqueurs
class MarkerType(Enum):
    """
    Types de marqueurs disponibles
    """
    ZONE_TRANSITION = "zone_transition"  # Changement de zone
    MONSTER_LOCATION = "monster_location"  # Emplacement monstre
    WATER = "water"  # Sous l'eau
    DANGER = "danger"  # Zone dangereuse (future utilisation)
    SAFE = "safe"  # Zone sans danger (future)
    RESOURCE = "resource"  # Ressource récoltable (future)
    LIMITE = "limite" # Bords de zone (futur)

class CubeMarker:
    """
    Marqueur sur un cube
    """

    def __init__(
            self,
            marker_type: MarkerType,
            strength: float = 1.0,
            metadata: Dict = None,
            is_static: bool = True
    ):
        """
        Args:
            marker_type: Type de marqueur
            strength: Force du marqueur (0.0 à 1.0)
            metadata: Données supplémentaires (ex: zone_id pour transition)
            is_static: True si marqueur permanent, False si décroît
        """
        self.marker_type = marker_type
        self.strength = strength
        self.metadata = metadata or {}
        self.is_static = is_static

        # Pour marqueurs dynamiques
        self.creation_time = time.time()
        self.last_update = time.time()

    def decay(self, decay_rate: float = 0.1):
        """
        Fait décroître un marqueur dynamique

        Args:
            decay_rate: Taux de décroissance par seconde
        """
        if self.is_static:
            return

        current_time = time.time()
        elapsed = current_time - self.last_update

        self.strength = max(0.0, self.strength - (decay_rate * elapsed))
        self.last_update = current_time

    def is_expired(self) -> bool:
        """Vérifie si le marqueur a expiré"""
        return not self.is_static and self.strength <= 0.0


class CubeMarkerSystem:
    """
    Gère tous les marqueurs sur les cubes
    """

    def __init__(self):
        # Dict[cube_id -> Dict[MarkerType -> CubeMarker]]
        self.markers: Dict[Tuple, Dict[MarkerType, CubeMarker]] = {}

        # Configuration décroissance pour chaque type
        self.decay_rates = {
            MarkerType.MONSTER_LOCATION: 0.20,  # Décroît progressivement (5s pour disparaître)
            MarkerType.DANGER: 0.5,  # Décroît rapidement (2s)
        }

        # Compteurs pour messages groupés
        self.new_markers_count = {}  # Dict[MarkerType, int]
        self.last_marker_message_time = 0.0
        self.marker_message_interval = 10.0  # 10 secondes

        # Securité threading
        self._lock = threading.Lock()

    def print_grouped_marker_messages(self):
        """
        Affiche les messages de nouveaux marqueurs groupés toutes les 10s
        """
        current_time = time.time()

        if current_time - self.last_marker_message_time >= self.marker_message_interval:
            # Afficher les compteurs non-nuls
            if self.new_markers_count:
                messages = []
                for marker_type, count in self.new_markers_count.items():
                    if count > 0:
                        emoji = {
                            MarkerType.MONSTER_LOCATION: "💹",
                            MarkerType.WATER: "💧",
                            MarkerType.ZONE_TRANSITION: "🚪",
                            MarkerType.DANGER: "⚠️",
                        }.get(marker_type, "📍")
                        messages.append(f"{emoji} {marker_type.value}: {count}")

                if messages:
                    logger.info(f"Nouveaux marqueurs (dernières {self.marker_message_interval:.0f}s): {', '.join(messages)}")

            # Reset
            self.new_markers_count.clear()
            self.last_marker_message_time = current_time

    @staticmethod
    def get_cube_id(cube) -> Tuple:
        """Génère un ID unique pour un cube"""
        return round(cube.center_x, 1), round(cube.center_y, 1), round(cube.center_z, 1)

    def add_marker(
            self,
            cube,
            marker_type: MarkerType,
            strength: float = 1.0,
            metadata: Dict = None,
            is_static: bool = True,
            force: bool = False,
    ):
        """
        Ajoute un marqueur sur un cube

        Args:
            cube: Cube à marquer
            marker_type: Type de marqueur
            strength: Force (0.0 à 1.0)
            metadata: Données supplémentaires
            is_static: Permanent ou temporaire
            force: Si True, remplace même si marqueur du même type existe
        """
        with self._lock:
            cube_id = self.get_cube_id(cube)

            if cube_id not in self.markers:
                self.markers[cube_id] = {}

            # VÉRIFIER SI UN MARQUEUR DU MÊME TYPE EXISTE DÉJÀ
            existing_markers = self.markers[cube_id]

            if marker_type in existing_markers and not force:
                # Un marqueur de ce type existe déjà
                existing_marker = existing_markers[marker_type]

                # Comparer la force pour décider si on remplace
                if strength > existing_marker.strength:
                    # Nouveau marqueur plus fort, remplacer
                    pass  # Le code continue ci-dessous
                else:
                    # Marqueur existant plus fort ou égal, ignorer
                    return  # Sortir sans ajouter

            marker = CubeMarker(marker_type, strength, metadata, is_static)
            self.markers[cube_id][marker_type] = marker

            # Incrémenter compteur au lieu de print immédiat
            if marker_type not in self.new_markers_count:
                self.new_markers_count[marker_type] = 0
            self.new_markers_count[marker_type] += 1

    def mark_zone_transition(
            self,
            cube_from_zone,
            cube_to_zone,
            from_zone_id: int,
            to_zone_id: int
    ):
        """
        Marque une transition de zone

        Args:
            cube_from_zone: Dernier cube de la zone d'origine
            cube_to_zone: Premier cube de la nouvelle zone
            from_zone_id: ID zone d'origine
            to_zone_id: ID nouvelle zone
        """
        # Marquer dernier cube zone d'origine
        self.add_marker(
            cube_from_zone,
            MarkerType.ZONE_TRANSITION,
            strength=1.0,
            metadata={'leads_to_zone': to_zone_id},
            is_static=True
        )

        # Marquer premier cube nouvelle zone
        self.add_marker(
            cube_to_zone,
            MarkerType.ZONE_TRANSITION,
            strength=1.0,
            metadata={'comes_from_zone': from_zone_id},
            is_static=True
        )

    def mark_monster_area(
            self,
            center_cube,
            surrounding_cubes: list,
            max_distance: float = 3.0
    ):
        """
        Marque une zone de monstre (dynamique, décroît avec le temps)

        Args:
            center_cube: Cube où le joueur a été frappé
            surrounding_cubes: Cubes environnants
            max_distance: Distance max pour marquage (en taille de cube)
        """
        # PROTECTION : Pas de marqueurs monstre dans le camp
        if center_cube.zone_id == 0:
            return  # Sortir immédiatement si c'est le camp

        # Pré-calculer pour éviter répétitions
        center_x, center_y, center_z = center_cube.center_x, center_cube.center_y, center_cube.center_z
        cube_size = center_cube.size
        max_dist_squared = (max_distance * cube_size) ** 2  # Comparer distance² (évite sqrt)

        # Marquer centre
        self.add_marker(
            center_cube,
            MarkerType.MONSTER_LOCATION,
            strength=1.0,
            metadata={'is_center': True},  # Metadata pour identifier le centre
            is_static=False)

        marked_count = 1
        for cube in surrounding_cubes:
            if cube == center_cube:
                continue

            # Distance au carré (plus rapide)
            dx = cube.center_x - center_x
            dy = cube.center_y - center_y
            dz = cube.center_z - center_z
            dist_squared = dx * dx + dy * dy + dz * dz

            if dist_squared <= max_dist_squared:
                # Calculer sqrt seulement si dans le rayon
                distance = (dist_squared ** 0.5) / cube_size
                strength = max(0.1, 1.0 - (distance / max_distance))

                self.add_marker(
                    cube,
                    MarkerType.MONSTER_LOCATION,
                    strength=strength,
                    is_static=False)

                marked_count += 1

    def mark_water(self, cube):
        """
        Marque un cube comme étant sous l'eau (statique)

        Args:
            cube: Cube sous l'eau
        """
        self.add_marker(
            cube,
            MarkerType.WATER,
            strength=1.0,
            is_static=True
        )

    def update_dynamic_markers(self):
        """
        Met à jour tous les marqueurs dynamiques (décroissance)
        """
        expired_cubes = []

        for cube_id, markers_dict in self.markers.items():
            expired_types = []

            for marker_type, marker in markers_dict.items():
                if not marker.is_static:
                    # Appliquer décroissance
                    decay_rate = self.decay_rates.get(marker_type, 0.05)
                    marker.decay(decay_rate)

                    # Marquer pour suppression si expiré
                    if marker.is_expired():
                        expired_types.append(marker_type)

            # Supprimer marqueurs expirés
            for marker_type in expired_types:
                del markers_dict[marker_type]

            # Si plus de marqueurs sur ce cube
            if len(markers_dict) == 0:
                expired_cubes.append(cube_id)

        # Supprimer cubes sans marqueurs
        for cube_id in expired_cubes:
            del self.markers[cube_id]

    def get_markers(self, cube) -> Dict[MarkerType, CubeMarker]:
        """
        Récupère tous les marqueurs d'un cube

        Returns:
            Dict des marqueurs actifs sur ce cube
        """
        cube_id = self.get_cube_id(cube)
        return self.markers.get(cube_id, {})

    def get_marker_vector(self, cube) -> list:
        """
        Convertit les marqueurs en vecteur pour l'IA

        Returns:
            Liste de valeurs [zone_transition, monster, water, ...]
            Chaque valeur = strength du marqueur (0.0 si absent)
        """
        markers = self.get_markers(cube)

        # Créer vecteur avec tous les types de marqueurs existants
        vector = [
            markers[MarkerType.ZONE_TRANSITION].strength if MarkerType.ZONE_TRANSITION in markers else 0.0,
            markers[MarkerType.MONSTER_LOCATION].strength if MarkerType.MONSTER_LOCATION in markers else 0.0,
            markers[MarkerType.WATER].strength if MarkerType.WATER in markers else 0.0,
            markers[MarkerType.DANGER].strength if MarkerType.DANGER in markers else 0.0,
            markers[MarkerType.SAFE].strength if MarkerType.SAFE in markers else 0.0,
            markers[MarkerType.RESOURCE].strength if MarkerType.RESOURCE in markers else 0.0,
            markers[MarkerType.LIMITE].strength if MarkerType.LIMITE in markers else 0.0,
        ]

        return vector

    def get_marker_count(self, cube) -> int:
        """
        Retourne le nombre de marqueurs actifs sur un cube

        Returns:
            Nombre de types de marqueurs différents (0-7)
        """
        markers = self.get_markers(cube)
        return len(markers)

    def has_marker_type(self, cube, marker_type: MarkerType) -> bool:
        """
        Vérifie si un cube a un marqueur d'un type donné

        Args:
            cube: Cube à vérifier
            marker_type: Type de marqueur recherché

        Returns:
            True si le cube a ce type de marqueur
        """
        markers = self.get_markers(cube)
        return marker_type in markers

    def get_all_markers_for_zone(self, zone_id: int, cubes_in_zone: list) -> Dict:
        """
        Récupère tous les marqueurs d'une zone (pour visualisation)

        Returns:
            Dict avec stats par type de marqueur
        """
        stats = {
            'zone_id': zone_id,
            'marker': {marker_type: 0 for marker_type in MarkerType},
        }

        for cube in cubes_in_zone:
            markers = self.get_markers(cube)
            for marker_type in markers:
                stats['markers'][marker_type] += 1

        return stats

    def reset_zone(self, zone_id: int, cubes_in_zone: list):
        """
        Reset tous les marqueurs dynamiques d'une zone

        Args:
            zone_id: ID de la zone (utilisé pour référence)
            cubes_in_zone: Liste des cubes à nettoyer
        """
        _ = zone_id  # Acknowledge parameter (gardé pour cohérence API)

        for cube in cubes_in_zone:
            cube_id = self.get_cube_id(cube)
            if cube_id in self.markers:
                # Garder seulement les marqueurs statiques
                static_markers = {
                    marker_type: marker
                    for marker_type, marker in self.markers[cube_id].items()
                    if marker.is_static
                }

                if static_markers:
                    self.markers[cube_id] = static_markers
                else:
                    del self.markers[cube_id]