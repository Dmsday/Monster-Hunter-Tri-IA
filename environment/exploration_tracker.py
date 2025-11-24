"""
Système de cartographie adaptative (Octree) pour Monster Hunter Tri
Gère les récompenses de découverte décroissantes et la compression mémoire

exploration_tracker.py
Version : 1.0
"""

import time
import math
import numpy as np

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from utils.module_logger import get_module_logger
logger = get_module_logger('exploration_tracker')

from environment.cube_markers import CubeMarkerSystem


class Cube:
    """
    Unité atomique de la carte - Cube 3D (parallélépipède)
    Représente un volume fixe dans l'espace (ex : 250×250×250)

    DERNIER AJOUT : Supporte maintenant des dimensions variables sur chaque axe (pour fusion)

    Attributes:
        center_x, center_y, center_z: Centre du volume
        size_x, size_y, size_z: Dimensions sur chaque axe
        visit_count: Visites durant l'épisode courant
        total_visits: Visites totales (permanent)
        blocked_directions: Obstacles détectés par direction
        needs_merge: Flag si le cube doit être fusionné
    """

    def __init__(self, center_x: float, center_y: float, center_z: float,
                 size_x: float = 200.0, size_y: float = 200.0, size_z: float = 200.0,
                 zone_id: int = 0):
        """
        Args:
            center_x, center_y, center_z: Centre du cube/parallélépipède
            size_x, size_y, size_z: Tailles sur chaque axe
            zone_id: ID de la zone du jeu
        """
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.zone_id = zone_id

        self.visit_count = 0 # Compteur de passages RÉEL (continue d'augmenter)
        self.effective_visit_count = 0  # Compteur EFFECTIF pour rewards (peut décroître)

        # Compteur permanent (conservé entre épisodes)
        self.total_visits = 0
        self.zones_discovered = set()

        # Obstacles détectés par direction
        self.blocked_directions = {
            'north': 0,  # +Z
            'south': 0,  # -Z
            'east': 0,  # +X
            'west': 0,  # -X
            'up': 0,  # +Y
            'down': 0  # -Y
        }

        # Historique des tentatives de mouvement bloquées
        self.block_attempts = defaultdict(int)

        # Flag pour fusion forcée (cube rogné devenu trop petit)
        self.needs_merge = False

        # Timer décroissance
        self.last_decay_time = 0.0  # Timestamp dernière décroissance

        # Marqueurs associés à ce cube (vide par défaut)
        self.markers = {}  # Dict[MarkerType, CubeMarker]

    # PROPRIÉTÉ de compatibilité : size = moyenne des 3 dimensions
    @property
    def size(self) -> float:
        """
        Taille moyenne (pour compatibilité avec ancien code)
        """
        return (self.size_x + self.size_y + self.size_z) / 3.0

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """
        Vérifie si un point est dans ce parallélépipède
        """
        half_x = self.size_x / 2
        half_y = self.size_y / 2
        half_z = self.size_z / 2

        return (abs(x - self.center_x) <= half_x and
                abs(y - self.center_y) <= half_y and
                abs(z - self.center_z) <= half_z)

    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Retourne les limites (min_x, max_x, min_y, max_y, min_z, max_z)
        """
        half_x = self.size_x / 2
        half_y = self.size_y / 2
        half_z = self.size_z / 2

        return (
            self.center_x - half_x, self.center_x + half_x,
            self.center_y - half_y, self.center_y + half_y,
            self.center_z - half_z, self.center_z + half_z
        )

    def get_volume(self) -> float:
        """
        Calcule le volume du parallélépipède
        """
        return self.size_x * self.size_y * self.size_z

    def increment_visit(self):
        """
        Incrémente les compteurs de visite
        """
        self.visit_count += 1  # Continue d'augmenter indéfiniment
        self.effective_visit_count += 1  # Aussi incrémenté (mais peut décroître)
        self.total_visits += 1

    def reset_episode_count(self):
        """
        Reset le compteur d'épisode (pas le total)
        """
        self.visit_count = 0  # Reset du compteur réel
        self.effective_visit_count = 0  # Reset du compteur effectif


class OctreeNode:
    """
    Nœud de l'octree - Peut contenir jusqu'à 8 enfants
    """

    def __init__(self, center_x: float, center_y: float, center_z: float,
                 size: float, max_depth: int = 5, depth: int = 0):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size = size
        self.depth = depth
        self.max_depth = max_depth

        # Enfants (8 maximum pour un octree)
        self.children: List[Optional[OctreeNode]] = [None] * 8

        # Cubes stockés dans ce nœud (si feuille)
        self.cubes: List[Cube] = []

        # Valeur d'exploration (somme des passages des cubes enfants)
        self.exploration_value = 0

    def is_leaf(self) -> bool:
        """Vérifie si c'est une feuille (pas d'enfants)"""
        return all(child is None for child in self.children)

    def get_octant(self, x: float, y: float, z: float) -> int:
        """
        Détermine dans quel octant (0-7) se trouve le point

        Octants :
        0 : -x, -y, -z
        1 : +x, -y, -z
        2 : -x, +y, -z
        3 : +x, +y, -z
        4 : -x, -y, +z
        5 : +x, -y, +z
        6 : -x, +y, +z
        7 : +x, +y, +z
        """
        octant = 0
        if x >= self.center_x: octant |= 1
        if y >= self.center_y: octant |= 2
        if z >= self.center_z: octant |= 4
        return octant

    def update_exploration_value(self):
        """
        Met à jour la valeur d'exploration (récursif)
        """
        if self.is_leaf():
            # Somme des visites des cubes
            self.exploration_value = sum(cube.visit_count for cube in self.cubes)
        else:
            # Somme des valeurs des enfants
            self.exploration_value = sum(
                child.exploration_value for child in self.children if child is not None
            )


class ExplorationTracker:
    """
    Système de cartographie adaptative avec octree

    Gère :
    - Création dynamique de cubes
    - Récompenses de découverte décroissantes
    - Compression mémoire (fusion de cubes)
    - Détection d'obstacles
    """

    def __init__(self,
                 cube_size: float = 650.0,
                 max_cubes: int = 250,
                 compression_target: float = 0.85):
        """
        Args:
            cube_size: Taille d'un cube de base
            max_cubes: Nombre max de cubes avant compression
            compression_target: Cible de compression (85% du max)
        """
        self.cube_size = cube_size
        self.max_cubes = max_cubes
        self.compression_target = compression_target

        # Dictionnaire des cubes par zone
        self.cubes_by_zone = {}

        # Octree par zone (structure hiérarchique pour fusion intelligente)
        self.octree_by_zone: Dict[int, OctreeNode] = {}

        # Paramètres octree
        self.octree_max_depth = 5  # Profondeur max (2^5 = 32 subdivisions)
        self.octree_world_size = 10000.0  # Taille du monde

        # Position actuelle du joueur
        self.current_position: Optional[Tuple[float, float, float]] = None
        self.current_cube: Optional[Cube] = None
        self.current_zone: int = 0

        # Historique des positions pour détecter blocages
        self.position_history: List[Tuple[float, float, float]] = []
        self.max_history_size = 10

        # Statistiques
        self.total_cubes_created = 0
        self.compression_count = 0

        # Configuration des récompenses
        self.reward_schedule = {
            1: 1.00,  # 100% pour 1ère visite
            2: 0.50,  # 50% pour 2ème visite
            3: 0.25,  # 25% pour 3ème visite
            4: 0.10,  # 10% pour 4ème visite
            5: 0.03,  # 3% pour 5ème visite
            6: 0.01,  # 1% pour 6ème visite
        }
        self.base_discovery_reward = 1.3 # cube_reward de base

        # Seuil de mouvement minimal (pour détecter immobilité)
        self.movement_threshold_creation = 0.0 # Pour création cube
        self.movement_threshold = 0.1  # Pour détection immobilité

        # système de marquage
        self.marker_system = CubeMarkerSystem()

        # Tracker pour transitions de zone
        self.last_zone = None
        self.last_cube_before_transition = None

        # Tracker le cube actuel pour détecter les changements
        self.current_cube_tracked = None  # Cube dans lequel on est actuellement

        # Pause système après événements
        self.paused_until = 0.0  # Timestamp jusqu'auquel le système est en pause
        self.pause_duration = 2.0  # Durée pause (secondes)

        # Config décroissance
        self.visit_decay_threshold = 11  # Décroît à partir de 10 visites
        self.visit_decay_interval = 180.0  # 3 minutes (120s)
        self.visit_decay_min = 6  # Minimum 6 visites

        # Compteurs pour messages groupés
        self.new_cubes_count = 0              # Compteur de nouveaux cubes
        self.last_cube_message_time = 0.0     # Timestamp dernier message
        self.cube_message_interval = 10.0     # Intervalle en secondes

        # Tracker dernière compression (pour cooldown)
        self.last_compression_time = {}  # Dict[zone_id, timestamp]
        self.compression_cooldown = 5.0  # 5 secondes après compression

        # Référence vers map_manager (sera injectée par StateFusion)
        self.map_manager = None  # Sera défini dans state_fusion.__init__()

    def pause_creation(self, duration: float = None):
        """
        Met en pause la création de cubes.
        A appeler durant temps de chargement et changements de zone

        Args:
            duration: Durée pause (secondes), None = utilise self.pause_duration
        """
        if duration is None:
            duration = self.pause_duration

        self.paused_until = time.time() + duration
        logger.debug(f"Création cubes PAUSÉE pour {duration:.1f}s")

    def _ensure_octree_for_zone(self, zone_id: int):
        """
        Crée l'octree pour une zone si elle n'existe pas encore

        Args:
            zone_id: ID de la zone
        """
        if zone_id not in self.octree_by_zone:
            # Créer la racine de l'octree centrée sur (0, 0, 0)
            # Taille = monde entier
            self.octree_by_zone[zone_id] = OctreeNode(
                center_x=0.0,
                center_y=0.0,
                center_z=0.0,
                size=self.octree_world_size,
                max_depth=self.octree_max_depth,
                depth=0
            )

    def _insert_cube_in_octree(self, cube: Cube, zone_id: int):
        """
        Insère un cube dans l'octree de sa zone

        Args:
            cube: Cube à insérer
            zone_id: ID de la zone
        """
        self._ensure_octree_for_zone(zone_id)
        root = self.octree_by_zone[zone_id]

        # Trouver le nœud approprié en descendant dans l'octree
        current_node = root

        while not current_node.is_leaf() and current_node.depth < current_node.max_depth:
            # Déterminer dans quel octant va le cube
            octant = current_node.get_octant(cube.center_x, cube.center_y, cube.center_z)

            # Créer le nœud enfant si nécessaire
            if current_node.children[octant] is None:
                # Calculer position et taille du nœud enfant
                half_size = current_node.size / 2
                quarter_size = current_node.size / 4

                # Offsets pour chaque octant (voir get_octant pour la numérotation)
                offsets = [
                    (-quarter_size, -quarter_size, -quarter_size),  # 0
                    (+quarter_size, -quarter_size, -quarter_size),  # 1
                    (-quarter_size, +quarter_size, -quarter_size),  # 2
                    (+quarter_size, +quarter_size, -quarter_size),  # 3
                    (-quarter_size, -quarter_size, +quarter_size),  # 4
                    (+quarter_size, -quarter_size, +quarter_size),  # 5
                    (-quarter_size, +quarter_size, +quarter_size),  # 6
                    (+quarter_size, +quarter_size, +quarter_size),  # 7
                ]

                offset = offsets[octant]

                current_node.children[octant] = OctreeNode(
                    center_x=current_node.center_x + offset[0],
                    center_y=current_node.center_y + offset[1],
                    center_z=current_node.center_z + offset[2],
                    size=half_size,
                    max_depth=current_node.max_depth,
                    depth=current_node.depth + 1
                )

            # Descendre dans l'octree
            current_node = current_node.children[octant]

        # Ajouter le cube au nœud feuille
        current_node.cubes.append(cube)

        # Mettre à jour les valeurs d'exploration en remontant
        self._update_octree_values(root)

    def _update_octree_values(self, node: OctreeNode):
        """
        Met à jour récursivement les valeurs d'exploration de l'octree

        Args:
            node: Nœud racine à partir duquel mettre à jour
        """
        if node.is_leaf():
            # Feuille : somme des visites des cubes
            node.exploration_value = sum(cube.visit_count for cube in node.cubes)
        else:
            # Nœud interne : somme des valeurs des enfants
            total = 0
            for child in node.children:
                if child is not None:
                    self._update_octree_values(child)  # Récursion
                    total += child.exploration_value
            node.exploration_value = total

    def visit_cube(self, cube):
        """
        Marque un cube comme visité ET notifie map_manager

        Cette méthode remplace les appels directs à cube.increment_visit()
        """
        # Incrémenter visite (code existant)
        cube.increment_visit()

        # Notifier map_manager pour update incrémental
        if self.map_manager is not None:
            self.map_manager.mark_cube_dirty(cube)

    def update_position(
            self,
            x: float,
            y: float,
            z: float,
            zone_id: int,
            action: Optional[int] = None,
    ) -> Dict:
        """
        Met à jour la position et crée des cubes uniformément

        DERNIER AJOUT : Système de marqueurs (cube_markers.py)
        """
        # Vérifier pause
        if time.time() < self.paused_until:
            # Système en pause, retourner résultat vide
            return {
                'discovery_reward': 0.0,
                'new_cube': False,
                'revisit_penalty': 0.0,
                'visit_count': 0,
                'cube_created': False,
                'total_cubes': sum(len(cubes) for cubes in self.cubes_by_zone.values()),
                'paused': True
            }

        # Gérer zone_id=None si un jour ça apparait
        if zone_id is None:
            zone_id = 0

        position_result = {
            'discovery_reward': 0.0,
            'new_cube': False,
            'revisit_penalty': 0.0,
            'visit_count': 0,
            'cube_created': False,
            'total_cubes': sum(len(cubes) for cubes in self.cubes_by_zone.values()),
            'obstacle_detected': False,
            'obstacle_direction_code': 0  # 0=aucun, 1=north, 2=south, 3=east, 4=west, 5=up, 6=down
        }

        # Verification si mouvement significatif
        if self.current_position is not None:
            prev_x, prev_y, prev_z = self.current_position
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2)

            # Si mouvement trop faible, ignorer
            if distance < self.movement_threshold_creation:
                return position_result

        # Decroissance des compteurs avant traitement
        current_time = time.time()
        self._apply_visit_decay(current_time)

        # CHERCHER CUBE EXISTANT
        cube = self._find_cube_at(x, y, z, zone_id)

        if cube is None:
            # Creer nouveau cube
            cube = self._create_cube_at(x, y, z, zone_id)
            self.visit_cube(cube)

            position_result['new_cube'] = True
            position_result['cube_created'] = True
            position_result['discovery_reward'] = self.base_discovery_reward
            position_result['visit_count'] = 1
        else:
            # Cube existant - REVISITER
            if self.current_cube_tracked is not cube:
                self.visit_cube(cube)
                visit_count_real = cube.visit_count  # Compteur réel (affichage)
                visit_count_effective = cube.effective_visit_count  # Compteur effectif

                position_result['visit_count'] = visit_count_real  # Pour l'affichage
                position_result['effective_visit_count'] = visit_count_effective  # Pour info

                # Système de reward décroissante
                if visit_count_effective == 1:
                    # 1ère visite après reset = reward complète
                    position_result['discovery_reward'] = self.base_discovery_reward
                    position_result['first_visit'] = True
                elif visit_count_effective <= 3:
                    # 2-3 visites = reward réduite
                    multiplier = self.reward_schedule.get(visit_count_effective, 0.5)
                    position_result['discovery_reward'] = self.base_discovery_reward * multiplier
                elif visit_count_effective <= 6:
                    # 4-6 visites = reward faible
                    multiplier = self.reward_schedule.get(visit_count_effective, 0.1)
                    position_result['discovery_reward'] = self.base_discovery_reward * multiplier
                elif visit_count_effective <= 10:
                    # 6-10 visites = pas de reward
                    position_result['discovery_reward'] = 0.0
                else:
                    # >10 visites = pas de reward + malus
                    position_result['discovery_reward'] = 0.0
                    position_result['revisit_penalty'] = 0.5
                    position_result['excessive_revisit'] = True

        # Sauvegarder ancien cube AVANT de vérifier les trous
        old_cube = self.current_cube_tracked

        # VÉRIFIER ET COMBLER LES TROUS
        if old_cube is not None and cube != old_cube:
            self._fill_missing_cubes_between(old_cube, cube, zone_id)

        # Mettre à jour le cube actuel
        self.current_cube_tracked = cube
        self.current_cube = cube
        self.current_zone = zone_id

        # DÉTECTER CHANGEMENT DE ZONE
        if self.last_zone is not None and zone_id != self.last_zone:
            # Transition de zone détectée
            if self.last_cube_before_transition is not None and cube is not None:
                self.marker_system.mark_zone_transition(
                    self.last_cube_before_transition,
                    cube,
                    self.last_zone,
                    zone_id
                )

        # Sauvegarder pour prochaine fois
        self.last_zone = zone_id
        self.last_cube_before_transition = cube

        # MARQUAGE EAU (si oxygène diminue)
        # À appeler depuis reward_calculator quand oxygène détecté

        # MAJ marqueurs dynamiques
        self.marker_system.update_dynamic_markers()

        # Synchroniser TOUS les marqueurs vers TOUS les cubes de TOUTES les zones
        for zone_id_sync, zone_cubes_sync in self.cubes_by_zone.items():
            for zone_cube in zone_cubes_sync:
                cube_id = self.marker_system.get_cube_id(zone_cube)
                zone_cube.markers = self.marker_system.markers.get(cube_id, {})

        # Détecter obstacles si action fournie
        if action is not None and old_cube is not None:
            obstacle_detected = self._detect_obstacles(action, x, y, z)
            if obstacle_detected:
                # Convertir la direction string en code numérique
                direction_codes = {
                    'north': 1, 'south': 2, 'east': 3,
                    'west': 4, 'up': 5, 'down': 6
                }
                position_result['obstacle_detected'] = True
                position_result['obstacle_direction_code'] = direction_codes.get(obstacle_detected, 0)

        self.current_position = (x, y, z)

        # Compression si nécessaire par zone
        if zone_id not in self.cubes_by_zone:
            self.cubes_by_zone[zone_id] = []

        zone_cubes_count = len(self.cubes_by_zone[zone_id])

        # Cooldown avant compression
        current_time = time.time()
        last_compression = self.last_compression_time.get(zone_id, 0)
        time_since_compression = current_time - last_compression

        # Si la zone actuelle dépasse le seuil, compresser CETTE zone
        if zone_cubes_count > self.max_cubes and time_since_compression >= 30.0:
            logger.info(f"Compression déclenchée pour zone {zone_id} ({zone_cubes_count} cubes)")
            self._compress_zone(zone_id)
            position_result['compression_triggered'] = True

            # LOG DIAGNOSTIC : Vérifier si tous les cubes sont présents
            octree_cube_count = sum(len(n.cubes) for n in self._collect_all_nodes(self.octree_by_zone[zone_id]))
            dict_cube_count = len(self.cubes_by_zone[zone_id])
            if octree_cube_count != dict_cube_count:
                logger.error(f"🚨 ERREUR SYNC: Octree={octree_cube_count} vs Dict={dict_cube_count}")

            logger.info(f"FUSION zone {zone_id}: {zone_cubes_count} → {len(self.cubes_by_zone[zone_id])} cubes")

        # Afficher messages groupés
        self._print_grouped_cube_messages() # message groupé pour la creation de cube
        self.marker_system.print_grouped_marker_messages() # message groupé pour l'ajout d'un marqueur

        return position_result

    def sync_all_markers_to_cubes(self):
        """
        Synchronise tous les marqueurs du marker_system vers les cubes
        À appeler avant d'envoyer les données au GUI

        DERNIER AJOUT : Vérifier qu'il n'y a pas de doublons d'un même type
        """
        for zone_id, cubes in self.cubes_by_zone.items():
            for cube in cubes:
                cube_id = self.marker_system.get_cube_id(cube)
                cube.markers = self.marker_system.markers.get(cube_id, {})

                # Compter les marqueurs de chaque type
                if len(cube.markers) > 0:
                    # Vérifier qu'il n'y a qu'1 marqueur de chaque type
                    type_counts = {}
                    for marker_type in cube.markers.keys():
                        type_counts[marker_type] = type_counts.get(marker_type, 0) + 1

                    # Si un type apparaît plusieurs fois, c'est une erreur
                    duplicates = {t: c for t, c in type_counts.items() if c > 1}
                    if duplicates:
                        logger.warning(f"🚨 ERREUR: Cube {cube_id} a des doublons de marqueurs!")
                        logger.warning(f"Doublons: {duplicates}")


    def _apply_visit_decay(self, current_time: float):
        """
        Fait décroître le compteur EFFECTIF (pour rewards)
        Le compteur RÉEL (visit_count) continue d'augmenter normalement
        """
        for zone_cubes in self.cubes_by_zone.values():
            for cube in zone_cubes:
                # Décroissance sur effective_visit_count uniquement
                if cube.effective_visit_count >= self.visit_decay_threshold:
                    if current_time - cube.last_decay_time >= self.visit_decay_interval:
                        if cube.effective_visit_count > self.visit_decay_min:
                            cube.effective_visit_count -= 1  # Seul ce compteur décroît
                            cube.last_decay_time = current_time
                            logger.debug(f"⬇ Décroissance cube effectif : {cube.effective_visit_count + 1} → {cube.effective_visit_count} (réel: {cube.visit_count})")

    def _compress_zone(self, zone_id: int):
        """
        Compresse les cubes pour une zone spécifique (pas au global)

        Args:
            zone_id: ID de la zone à compresser
        """
        if zone_id not in self.cubes_by_zone or zone_id not in self.octree_by_zone:
            return

        cubes_in_zone = len(self.cubes_by_zone[zone_id])
        target_count = int(self.max_cubes * self.compression_target)  # 85% du max
        cubes_to_remove = cubes_in_zone - target_count
        # Éviter fusion si aucun résultat attendu
        if cubes_to_remove <= 0:
            return

        logger.debug(f"Compression zone {zone_id} (seuil dépassé: {cubes_in_zone})")

        # Collecter les nœuds de CETTE zone uniquement
        root = self.octree_by_zone[zone_id]
        all_nodes = self._collect_all_nodes(root)
        nodes_with_cubes = [n for n in all_nodes if len(n.cubes) > 0]

        if len(nodes_with_cubes) == 0:
            logger.warning("Aucun nœud avec cubes à fusionner")
            return

        # Calculer score pour chaque nœud
        alpha = 0.001
        node_scores = []
        for node in nodes_with_cubes:
            volume = node.size ** 3
            score = node.exploration_value - (alpha * volume)
            node_scores.append((score, node))

        # Trier par score décroissant
        node_scores.sort(key=lambda x: x[0], reverse=True)

        # Sélectionner les 3-4 meilleurs nœuds
        top_nodes = [node for _, node in node_scores[:4]]

        logger.info(f"Sélection de {len(top_nodes)} nœuds à compresser")

        # Compter AVANT fusion pour vérification
        cubes_before_merge = sum(len(node.cubes) for node in all_nodes)

        # Fusionner localement dans chaque nœud
        total_removed = 0

        for node in top_nodes:
            # Fusionner même si 1 seul cube (avec voisins)
            if len(node.cubes) == 0:
                continue

            before = len(node.cubes)
            self._merge_cubes_in_node(node)
            after = len(node.cubes)

            removed = before - after
            total_removed += removed

            if removed > 0:
                logger.info(f"Nœud (taille={node.size:.0f}): {before} → {after} cubes (-{removed})")

            # Arrêter si on a assez enlevé
            if total_removed >= cubes_to_remove:
                break

        # Reconstruire cubes_by_zone depuis l'octree pour cette zone
        zone_cubes = []
        for node in all_nodes:
            zone_cubes.extend(node.cubes)

        # S'assurer qu'on n'a pas perdu de cubes
        cubes_after_merge = len(zone_cubes)
        expected_count = cubes_before_merge - total_removed

        # AJOUTER LOG DEBUG
        if cubes_after_merge != expected_count:
            logger.debug(f"🚨 ERREUR SYNCHRONISATION DÉTECTÉE !")
            logger.debug(f"Avant fusion: {cubes_before_merge}")
            logger.debug(f"Supprimés: {total_removed}")
            logger.debug(f"Attendu: {expected_count}")
            logger.debug(f"Obtenu: {cubes_after_merge}")
            logger.debug(f"→ CORRECTION EN COURS...")

            # Reconstruire en forçant la collecte de TOUS les cubes
            zone_cubes.clear()
            for node in self._collect_all_nodes(root):
                zone_cubes.extend(node.cubes)

            logger.debug(f"→ Après correction: {len(zone_cubes)} cubes")

            # Appliquer la mise à jour
        self.cubes_by_zone[zone_id] = zone_cubes

        self.compression_count += 1
        total_after = len(self.cubes_by_zone[zone_id])
        logger.info(f"Total après compression zone {zone_id}: {total_after} cubes (-{total_removed})")

        # Enregistrer timestamp pour cooldown
        self.last_compression_time[zone_id] = time.time()

        # VÉRIFICATION D'INTÉGRITÉ
        if not self.verify_octree_integrity(zone_id):
            logger.info("Tentative de correction automatique...")
            # Forcer reconstruction depuis l'octree
            zone_cubes = []
            for node in self._collect_all_nodes(self.octree_by_zone[zone_id]):
                zone_cubes.extend(node.cubes)
            self.cubes_by_zone[zone_id] = zone_cubes
            logger.info(f"→ Correction appliquée: {len(zone_cubes)} cubes")

    def verify_octree_integrity(self, zone_id: int) -> bool:
        """
        Vérifie que l'octree et cubes_by_zone sont synchronisés

        Returns:
            True si OK, False si incohérence détectée
        """
        if zone_id not in self.octree_by_zone or zone_id not in self.cubes_by_zone:
            return True  # Pas de données = pas d'erreur

        # Compter cubes dans l'octree
        root = self.octree_by_zone[zone_id]
        all_nodes = self._collect_all_nodes(root)
        octree_cubes = sum(len(node.cubes) for node in all_nodes)

        # Compter cubes dans le dict
        dict_cubes = len(self.cubes_by_zone[zone_id])

        if octree_cubes != dict_cubes:
            logger.warning(f"🚨 INCOHÉRENCE DÉTECTÉE - Zone {zone_id}")
            logger.warning(f"Octree: {octree_cubes} cubes")
            logger.warning(f"Dict: {dict_cubes} cubes")
            logger.warning(f"Différence: {abs(octree_cubes - dict_cubes)}")
            return False

        return True

    def _create_cube_at(self, x: float, y: float, z: float, zone_id: int) -> Cube:
        """
        Crée un nouveau cube centré sur la position donnée
        """
        # Aligner sur une grille pour éviter chevauchements
        grid_x = (x // self.cube_size) * self.cube_size + (self.cube_size / 2)
        grid_y = (y // self.cube_size) * self.cube_size + (self.cube_size / 2)
        grid_z = (z // self.cube_size) * self.cube_size + (self.cube_size / 2)

        # CREER avec dimensions égales initialement
        cube = Cube(grid_x, grid_y, grid_z, self.cube_size, self.cube_size, self.cube_size, zone_id)

        # Initialiser la liste si elle n'existe pas
        if zone_id not in self.cubes_by_zone:
            self.cubes_by_zone[zone_id] = []
            logger.debug(f"Zone {zone_id} initialisée dans cubes_by_zone")

        self.cubes_by_zone[zone_id].append(cube)
        self.total_cubes_created += 1

        # Incrémenter compteur au lieu de print immédiat
        self.new_cubes_count += 1

        # Insérer dans l'octree
        self._insert_cube_in_octree(cube, zone_id)

        return cube

    def _print_grouped_cube_messages(self):
        """
        Affiche les messages de nouveaux cubes groupés toutes les 10s

        - Si des cubes ont été créés : affiche le nombre
        - Sinon : ne fait rien (silence)
        """
        current_time = time.time()

        # Vérifier si 10 secondes se sont écoulées
        if current_time - self.last_cube_message_time >= self.cube_message_interval:
            # Si au moins 1 cube créé
            if self.new_cubes_count > 0:
                logger.info(f"{self.new_cubes_count} NOUVEAUX CUBES créés (dernières {self.cube_message_interval:.0f}s)")
                self.new_cubes_count = 0  # Reset le compteur

            # Mettre à jour le timestamp
            self.last_cube_message_time = current_time

    @staticmethod
    def _extend_cube_in_direction(cube: Cube, direction: str, extension_amount: float):
        """
        Étend un cube dans une direction donnée

        Args:
            cube: Cube à étendre
            direction: Direction ('north', 'south', 'east', 'west', 'up', 'down')
            extension_amount: Quantité d'extension (généralement cube_size)
        """
        # Calculer nouvelles dimensions et nouveau centre
        if direction == 'north':  # +Z
            cube.size_z += extension_amount
            cube.center_z += extension_amount / 2
        elif direction == 'south':  # -Z
            cube.size_z += extension_amount
            cube.center_z -= extension_amount / 2
        elif direction == 'east':  # +X
            cube.size_x += extension_amount
            cube.center_x += extension_amount / 2
        elif direction == 'west':  # -X
            cube.size_x += extension_amount
            cube.center_x -= extension_amount / 2
        elif direction == 'up':  # +Y
            cube.size_y += extension_amount
            cube.center_y += extension_amount / 2
        elif direction == 'down':  # -Y
            cube.size_y += extension_amount
            cube.center_y -= extension_amount / 2

        logger.info(f"📏 Cube étendu {direction}: {cube.size_x:.0f}×{cube.size_y:.0f}×{cube.size_z:.0f}")

    @staticmethod
    def _find_adjacent_cube(cube: Cube, direction: str, zone_cubes: list) -> Optional[Cube]:
        """
        Trouve le cube adjacent dans une direction donnée

        Args:
            cube: Cube de référence
            direction: Direction de recherche
            zone_cubes: Liste des cubes de la zone

        Returns:
            Cube adjacent ou None
        """
        # Calculer position attendue du cube adjacent
        expected_x = cube.center_x
        expected_y = cube.center_y
        expected_z = cube.center_z

        # Ajuster selon direction (utiliser size moyen du cube actuel)
        avg_size = (cube.size_x + cube.size_y + cube.size_z) / 3.0

        if direction == 'north':  # +Z
            expected_z += avg_size
        elif direction == 'south':  # -Z
            expected_z -= avg_size
        elif direction == 'east':  # +X
            expected_x += avg_size
        elif direction == 'west':  # -X
            expected_x -= avg_size
        elif direction == 'up':  # +Y
            expected_y += avg_size
        elif direction == 'down':  # -Y
            expected_y -= avg_size

        # Chercher le cube le plus proche de cette position
        closest_cube = None
        min_distance = float('inf')

        tolerance = avg_size * 0.6  # Tolérance de 60% de la taille

        for other_cube in zone_cubes:
            if other_cube == cube:
                continue

            # Distance à la position attendue
            dx = abs(other_cube.center_x - expected_x)
            dy = abs(other_cube.center_y - expected_y)
            dz = abs(other_cube.center_z - expected_z)
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # Doit être dans la tolérance
            if distance < tolerance and distance < min_distance:
                min_distance = distance
                closest_cube = other_cube

        return closest_cube

    def _find_cube_at(self, x: float, y: float, z: float, zone_id: int) -> Optional[Cube]:
        """Trouve le cube contenant cette position"""
        if zone_id not in self.cubes_by_zone:
            return None

        for cube in self.cubes_by_zone[zone_id]:
            if cube.contains_point(x, y, z):
                return cube

        return None

    def _fill_missing_cubes_between(self, prev_cube, current_cube, zone_id: int):
        """
        Crée les cubes manquants entre deux cubes non-adjacents
        --> comble les espaces découverts mais non créer

        LIMITES :
        - Maximum 3 cubes d'écart (sinon ignore)
        - Respecte la pause système (comme création normale)
        - Si cube existe déjà, incrémente juste la visite
        """
        if prev_cube is None or current_cube is None:
            return

        # VÉRIFIER PAUSE SYSTÈME (comme création normale)
        if time.time() < self.paused_until:
            return  # Système en pause, ne pas combler

        # VÉRIFIER COOLDOWN APRÈS COMPRESSION
        if zone_id in self.last_compression_time:
            time_since_compression = time.time() - self.last_compression_time[zone_id]
            if time_since_compression < self.compression_cooldown:
                return  # Trop tôt après compression, ne pas combler

        # Vérifier si les cubes sont adjacents
        dx = abs(current_cube.center_x - prev_cube.center_x)
        dy = abs(current_cube.center_y - prev_cube.center_y)
        dz = abs(current_cube.center_z - prev_cube.center_z)

        # Distance en "nombre de cubes"
        steps_x = int(round(dx / self.cube_size))
        steps_y = int(round(dy / self.cube_size))
        steps_z = int(round(dz / self.cube_size))

        max_steps = max(steps_x, steps_y, steps_z)

        # LIMITE : Ignorer si plus de 3 cubes d'écart (trop grand saut)
        if max_steps > 4:  # 4 car on remplit entre (1, 2, 3 intermédiaires)
            return

        # Si plus d'un cube d'écart, créer les intermédiaires
        if max_steps > 1:

            filled_count = 0
            visit_count = 0

            for i in range(1, max_steps):
                # Interpolation linéaire
                t = i / max_steps
                interp_x = prev_cube.center_x + t * (current_cube.center_x - prev_cube.center_x)
                interp_y = prev_cube.center_y + t * (current_cube.center_y - prev_cube.center_y)
                interp_z = prev_cube.center_z + t * (current_cube.center_z - prev_cube.center_z)

                # Vérifier si un cube existe déjà
                existing_cube = self._find_cube_at(interp_x, interp_y, interp_z, zone_id)

                if existing_cube is None:
                    # Créer cube intermédiaire
                    intermediate_cube = self._create_cube_at(interp_x, interp_y, interp_z, zone_id)
                    self.visit_cube(intermediate_cube)
                    filled_count += 1
                else:
                    # Marquer visite sur cube existant
                    self.visit_cube(existing_cube)
                    visit_count += 1

            # LOG - Afficher seulement si cubes créés
            if filled_count > 0:
                logger.debug(f"🔗 Comblé {filled_count} trous (distance: {max_steps} cubes = {dx:.0f}u)")

    def _update_position_history(self, x: float, y: float, z: float):
        """Met à jour l'historique des positions"""
        self.position_history.append((x, y, z))
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)

    def _detect_obstacles(self, action: int, new_x: float, new_y: float, new_z: float) -> Optional[str]:
        """
        Détecte les obstacles en comparant l'action effectuée et le mouvement réel

        Actions de déplacement (supposées) :
        1: Forward (+Z)
        2: Backward (-Z)
        3: Left (-X)
        4: Right (+X)
        """
        if self.current_position is None or self.current_cube is None:
            return None

        old_x, old_y, old_z = self.current_position

        # Calculer le mouvement réel
        dx = new_x - old_x
        dy = new_y - old_y
        dz = new_z - old_z

        movement = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Actions de mouvement selon controller.py
        movement_actions = [1, 2, 3, 4]  # Forward, Backward, Left, Right

        # Si mouvement très faible alors qu'on a demandé un déplacement
        if action in movement_actions and movement < self.movement_threshold:
            # Détecter la direction bloquée
            direction = None
            if action == 1:  # Forward
                direction = 'north'
            elif action == 2:  # Backward
                direction = 'south'
            elif action == 3:  # Left
                direction = 'west'
            elif action == 4:  # Right
                direction = 'east'

            if direction:
                self.current_cube.block_attempts[direction] += 1

                # Confirmer obstacle si bloqué plusieurs fois
                if self.current_cube.block_attempts[direction] >= 5:
                    self.current_cube.blocked_directions[direction] += 1
                    logger.info(f"🧱 OBSTACLE détecté - Direction: {direction}")

                    # Rogner le cube si obstacle confirmé
                    self._rogue_cube_on_direction(self.current_cube, direction)
                return direction

        return None

    @staticmethod
    def _rogue_cube_on_direction(cube: Cube, direction: str):
        """
        Rogne un cube dans une direction donnée (réduit sa taille)
        Si le cube devient trop petit, le fusionne avec un voisin

        Args:
            cube: Cube à rogner
            direction: Direction du mur ('north', 'south', 'east', 'west', 'up', 'down')
        """
        # Réduction de 20% dans la direction du mur
        reduction_factor = 0.8

        # Calculer la taille moyenne actuelle pour les ajustements de position
        avg_size = (cube.size_x + cube.size_y + cube.size_z) / 3.0

        # Ajuster la position du centre selon le mur ET réduire la dimension concernée
        if direction == 'north':  # +Z
            cube.center_z -= avg_size * 0.1
            cube.size_z *= reduction_factor
        elif direction == 'south':  # -Z
            cube.center_z += avg_size * 0.1
            cube.size_z *= reduction_factor
        elif direction == 'east':  # +X
            cube.center_x -= avg_size * 0.1
            cube.size_x *= reduction_factor
        elif direction == 'west':  # -X
            cube.center_x += avg_size * 0.1
            cube.size_x *= reduction_factor
        elif direction == 'up':  # +Y
            cube.center_y -= avg_size * 0.1
            cube.size_y *= reduction_factor
        elif direction == 'down':  # -Y
            cube.center_y += avg_size * 0.1
            cube.size_y *= reduction_factor

        # Vérifier si le cube est devenu trop petit (taille moyenne)
        min_avg_size = 30.0  # Taille moyenne minimum = 30
        if cube.size < min_avg_size:
            logger.info(f"🔧 Cube rogné trop petit (taille moy: {cube.size:.1f}) - Fusion nécessaire")
            cube.needs_merge = True

    def _collect_all_nodes(self, node: OctreeNode) -> List[OctreeNode]:
        """
        Collecte tous les nœuds de l'octree (récursif)

        Args:
            node: Nœud racine

        Returns:
            Liste de tous les nœuds
        """
        nodes = [node]

        if not node.is_leaf():
            for child in node.children:
                if child is not None:
                    nodes.extend(self._collect_all_nodes(child))

        return nodes

    def _merge_cubes_in_node(self, node: OctreeNode):
        """
        Fusionne les cubes dans un nœud = Extension des cubes adjacents très visités
        pour absorber les cubes peu visités

        Principe :
        1. Identifier cubes très visités (> trois visites)
        2. Trouver leur voisin le plus visité
        3. Étendre le voisin pour absorber le cube moins visité
        4. Transférer les visites (moyenne arrondie supérieure)
        5. Supprimer le cube absorbé
        """
        if len(node.cubes) == 0:
            return

        if len(node.cubes) <= 1:
            return

        # Séparer cubes rognés (trop petits = obstacles) et normaux
        cubes_to_remove = [c for c in node.cubes if hasattr(c, 'needs_merge') and c.needs_merge]
        # GARDER node.cubes comme référence (sera filtré à la fin)
        all_cubes = node.cubes  # Référence à tous les cubes du nœud
        normal_cubes = [c for c in all_cubes if c not in cubes_to_remove]

        # 1. SUPPRIMER cubes rognés (trop petits)
        if cubes_to_remove:
            logger.info(f"Suppression {len(cubes_to_remove)} cubes rognés (obstacles)")

        # 2. FUSION PAR EXTENSION
        if len(normal_cubes) >= 2:
            # Trier par nombre de visites (décroissant)
            normal_cubes.sort(key=lambda c: c.visit_count, reverse=True)

            cubes_absorbed = []  # Cubes qui seront absorbés

            # Parcourir une COPIE pour éviter modification pendant itération
            for cube in list(normal_cubes):  # list() crée une copie
                # Ignorer si déjà marqué pour absorption
                if cube in cubes_absorbed:
                    continue

                # Absorber seulement les cubes très visités (> 3 visites, moins en cas de fusion impossible)
                min_visits = 2 if len(normal_cubes) > 100 else 3
                if cube.visit_count <= min_visits:
                    continue

                # Chercher le voisin le plus visité dans chaque direction
                best_neighbor = None
                best_neighbor_visits = 0
                best_direction = None

                directions = ['north', 'south', 'east', 'west', 'up', 'down']

                for direction in directions:
                    neighbor = self._find_adjacent_cube(cube, direction, normal_cubes)

                    if neighbor and neighbor.visit_count > best_neighbor_visits:
                        # Ignorer si déjà absorbé
                        if neighbor in cubes_absorbed:
                            continue

                        best_neighbor = neighbor
                        best_neighbor_visits = neighbor.visit_count
                        best_direction = direction

                # Si on a trouvé un bon voisin
                if best_neighbor and best_neighbor_visits > cube.visit_count:
                    # ÉTENDRE le voisin pour absorber ce cube
                    # Direction opposée (si cube est au nord, étendre vers le nord)
                    opposite_direction = {
                        'north': 'north',
                        'south': 'south',
                        'east': 'east',
                        'west': 'west',
                        'up': 'up',
                        'down': 'down'
                    }[best_direction]

                    # Quantité d'extension = taille du cube dans cette dimension
                    if opposite_direction in ['north', 'south']:
                        extension = cube.size_z
                    elif opposite_direction in ['east', 'west']:
                        extension = cube.size_x
                    else:  # up, down
                        extension = cube.size_y

                    # Vérifier si extension est sûre
                    would_overlap = self._check_extension_overlap(
                        best_neighbor,
                        opposite_direction,
                        extension,
                        normal_cubes,
                        exclude=[cube, best_neighbor]  # Exclure cubes impliqués dans fusion
                    )

                    if not would_overlap:
                        # Extension sûre, procéder
                        self._extend_cube_in_direction(best_neighbor, opposite_direction, extension)

                        # TRANSFÉRER visites (moyenne arrondie supérieure) - DANS le if
                        combined_visits = best_neighbor.visit_count + cube.visit_count
                        avg_visits = combined_visits / 2.0
                        new_visit_count = math.ceil(avg_visits)

                        best_neighbor.visit_count = new_visit_count
                        best_neighbor.total_visits += cube.total_visits

                        logger.info(f"Fusion: Cube ({cube.visit_count} visites) absorbé par voisin")
                        logger.info(f"→ Nouveau visit_count: {new_visit_count} (moy. {avg_visits:.1f} arrondie)")

                        # Marquer pour suppression
                        cubes_absorbed.append(cube)
                    else:
                        # Overlap détecté, SKIP cette fusion
                        logger.debug(f"Fusion skippée (overlap détecté)")

            # Supprimer les cubes absorbés et les cubes rognés
            if cubes_absorbed or cubes_to_remove:
                logger.info(f"{len(cubes_absorbed)} cubes absorbés par extension")
                logger.info(f"{len(cubes_to_remove)} cubes rognés supprimés")

                # Garder SEULEMENT les cubes qui ne sont ni absorbés ni rognés
                node.cubes = [c for c in all_cubes
                              if c not in cubes_absorbed and c not in cubes_to_remove]
            else:
                node.cubes = normal_cubes

            # VÉRIFICATION FINALE
            logger.info(f"Nœud après fusion: {len(all_cubes)} → {len(node.cubes)} cubes restants")

    @staticmethod
    def _check_extension_overlap(
            cube: Cube,
            direction: str,
            extension_amount: float,
            all_cubes: List[Cube],
            exclude: List[Cube] = None
    ) -> bool:
        """
        Vérifie si étendre un cube causerait un overlap avec d'autres cubes

        Args:
            cube: Cube à étendre
            direction: Direction d'extension
            extension_amount: Quantité d'extension
            all_cubes: Tous les cubes de la zone
            exclude: Cubes à ignorer dans la vérification

        Returns:
            True si overlap détecté, False sinon
        """
        if exclude is None:
            exclude = []

        # Ajuster selon direction
        new_center_x = cube.center_x
        new_center_y = cube.center_y
        new_center_z = cube.center_z
        new_size_x = cube.size_x
        new_size_y = cube.size_y
        new_size_z = cube.size_z

        if direction == 'north':  # +Z
            new_center_z += extension_amount / 2
            new_size_z += extension_amount
        elif direction == 'south':  # -Z
            new_center_z -= extension_amount / 2
            new_size_z += extension_amount
        elif direction == 'east':  # +X
            new_center_x += extension_amount / 2
            new_size_x += extension_amount
        elif direction == 'west':  # -X
            new_center_x -= extension_amount / 2
            new_size_x += extension_amount
        elif direction == 'up':  # +Y
            new_center_y += extension_amount / 2
            new_size_y += extension_amount
        elif direction == 'down':  # -Y
            new_center_y -= extension_amount / 2
            new_size_y += extension_amount

        # Bounds du cube étendu
        new_min_x = new_center_x - new_size_x / 2
        new_max_x = new_center_x + new_size_x / 2
        new_min_y = new_center_y - new_size_y / 2
        new_max_y = new_center_y + new_size_y / 2
        new_min_z = new_center_z - new_size_z / 2
        new_max_z = new_center_z + new_size_z / 2

        # Vérifier overlap avec chaque cube
        for other_cube in all_cubes:
            if other_cube in exclude:
                continue

            # Bounds de l'autre cube
            other_min_x, other_max_x, other_min_y, other_max_y, other_min_z, other_max_z = other_cube.get_bounds()

            # Vérifier overlap sur les 3 axes
            overlap_x = not (new_max_x < other_min_x or new_min_x > other_max_x)
            overlap_y = not (new_max_y < other_min_y or new_min_y > other_max_y)
            overlap_z = not (new_max_z < other_min_z or new_min_z > other_max_z)

            if overlap_x and overlap_y and overlap_z:
                # Overlap détecté
                return True

        return False

    def _rebuild_cubes_from_octree(self):
        """
        Reconstruit cubes_by_zone depuis l'octree
        (après fusion, les cubes dans l'octree sont à jour)
        """
        # Vider cubes_by_zone
        self.cubes_by_zone.clear()

        # Parcourir tous les octrees et collecter les cubes
        for zone_id, root in self.octree_by_zone.items():
            all_nodes = self._collect_all_nodes(root)

            zone_cubes = []
            for node in all_nodes:
                zone_cubes.extend(node.cubes)

            self.cubes_by_zone[zone_id] = zone_cubes

    def _compress_cubes(self):
        """
        Compresse les cubes via l'octree (approche hiérarchique)

        Sélectionne les 3-4 nœuds avec le + de visites et fusionne localement
        """
        logger.info(f"🗜️ Compression des cubes via octree (seuil dépassé: {self.max_cubes})")

        target_count = int(self.max_cubes * self.compression_target)
        total_cubes = sum(len(cubes) for cubes in self.cubes_by_zone.values())
        cubes_to_remove = total_cubes - target_count

        if cubes_to_remove <= 0:
            return

        # ÉTAPE 1 : Collecter tous les nœuds non-racines de tous les octrees
        all_nodes = []

        for zone_id, root in self.octree_by_zone.items():
            nodes_in_zone = self._collect_all_nodes(root)
            # Filtrer : garder seulement les nœuds avec des cubes
            nodes_with_cubes = [n for n in nodes_in_zone if len(n.cubes) > 0]
            all_nodes.extend(nodes_with_cubes)

        if len(all_nodes) == 0:
            logger.warning("⚠️ Aucun nœud avec cubes à fusionner")
            return

        # ÉTAPE 2 : Calculer score pour chaque nœud
        # Score = exploration_value - α * volume du nœud
        alpha = 0.001  # Poids du volume (petit)

        node_scores = []
        for node in all_nodes:
            volume = node.size ** 3
            score = node.exploration_value - (alpha * volume)
            node_scores.append((score, node))

        # Trier par score décroissant (meilleurs candidats = + de visites, - de volume)
        node_scores.sort(key=lambda x: x[0], reverse=True)

        # ÉTAPE 3 : Sélectionner les 3-4 meilleurs nœuds
        top_nodes = [node for _, node in node_scores[:4]]

        logger.info(f"Sélection de {len(top_nodes)} nœuds à compresser")

        # ÉTAPE 4 : Fusionner localement dans chaque nœud
        total_removed = 0

        for node in top_nodes:
            if len(node.cubes) <= 1:
                continue  # Rien à fusionner

            before = len(node.cubes)
            self._merge_cubes_in_node(node)
            after = len(node.cubes)

            removed = before - after
            total_removed += removed

            if removed > 0:
                logger.info(f"Nœud (taille={node.size:.0f}): {before} → {after} cubes (-{removed})")
            else:
                # Logger si aucune fusion n'a eu lieu
                logger.warning(f"Nœud (taille={node.size:.0f}): Aucune fusion ({before} cubes)")

            # Arrêter si on a assez enlevé
            if total_removed >= cubes_to_remove:
                break

        # Détecter fusion bloquée
        if total_removed == 0 and cubes_to_remove > 0:
            # Calculer le total de cubes pour le seuil
            total_cubes_count = sum(len(cubes) for cubes in self.cubes_by_zone.values())
            min_visits_threshold = 2 if total_cubes_count > 100 else 3

            logger.error(f"FUSION BLOQUÉE : 0 cubes supprimés malgré objectif de {cubes_to_remove}")
            logger.info(f"Cause possible : Tous les cubes ont visit_count <= {min_visits_threshold}")
            logger.info(f"Solution : Augmenter max_cubes ou réduire min_visits dans _merge_cubes_in_node()")

            # Enregistrer cooldown pour éviter de spam + Empêcher nouvelle fusion pendant 30s
            current_time = time.time()
            for zone_id in self.cubes_by_zone.keys():
                self.last_compression_time[zone_id] = current_time + 30.0
                logger.info(f"COOLDOWN : Fusion désactivée 30s pour zone {zone_id}")

            return  # Sortir immédiatement

        # ÉTAPE 5 : Reconstruire cubes_by_zone depuis l'octree
        self._rebuild_cubes_from_octree()

        self.compression_count += 1
        total_after = sum(len(cubes) for cubes in self.cubes_by_zone.values())
        logger.info(f"Total après compression: {total_after} cubes (-{total_removed})")

    @staticmethod
    def _are_adjacent(cube1: Cube, cube2: Cube) -> bool:
        """Vérifie si deux cubes sont adjacents"""
        dx = abs(cube1.center_x - cube2.center_x)
        dy = abs(cube1.center_y - cube2.center_y)
        dz = abs(cube1.center_z - cube2.center_z)

        # Adjacent si distance = size sur un seul axe et 0 sur les autres
        size = cube1.size
        adjacent = (
                (abs(dx - size) < 1 and dy < 1 and dz < 1) or
                (dx < 1 and abs(dy - size) < 1 and dz < 1) or
                (dx < 1 and dy < 1 and abs(dz - size) < 1)
        )

        return adjacent

    @staticmethod
    def _merge_two_cubes(cube1: Cube, cube2: Cube) -> Cube:
        """
        Fusionne deux cubes en un seul (bounds englobante, compteurs combinés)
        """
        # Récupérer bounds des deux cubes
        min_x1, max_x1, min_y1, max_y1, min_z1, max_z1 = cube1.get_bounds()
        min_x2, max_x2, min_y2, max_y2, min_z2, max_z2 = cube2.get_bounds()

        # Calculer bounds englobantes
        min_x = min(min_x1, min_x2)
        max_x = max(max_x1, max_x2)
        min_y = min(min_y1, min_y2)
        max_y = max(max_y1, max_y2)
        min_z = min(min_z1, min_z2)
        max_z = max(max_z1, max_z2)

        # Centre du nouveau cube
        new_x = (min_x + max_x) / 2
        new_y = (min_y + max_y) / 2
        new_z = (min_z + max_z) / 2

        # Taille = max des dimensions englobantes
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        new_size = (size_x + size_y + size_z) / 3.0

        merged = Cube(new_x, new_y, new_z, new_size, cube1.zone_id)

        # Compteurs combinés
        merged.visit_count = (cube1.visit_count + cube2.visit_count) // 2
        merged.total_visits = cube1.total_visits + cube2.total_visits

        logger.debug(f"FUSION : ({cube1.size:.0f} + {cube2.size:.0f}) → {new_size:.0f} unités")

        return merged

    def reset_episode(self):
        """
        Reset les compteurs d'épisode (pas la structure de carte)
        """
        for zone_id, zone_cubes in self.cubes_by_zone.items():
            for cube in zone_cubes:
                cube.visit_count = 0  # Important pour couleur carte
                cube.reset_episode_count()
            # Vérifier intégrité
            logger.info(f"🔄 Reset épisode - Zone {zone_id}: {len(zone_cubes)} cubes conservés")

        self.position_history.clear()
        self.current_position = None
        self.current_cube = None
        self.current_cube_tracked = None

        # Reset timer camp
        if hasattr(self, '_camp_entry_time'):
            delattr(self, '_camp_entry_time')

    def get_stats(self) -> Dict:
        """
        Retourne les statistiques d'exploration
        """
        # Synchroniser les marqueurs avant de retourner les stats
        self.sync_all_markers_to_cubes()

        total_cubes = sum(len(cubes) for cubes in self.cubes_by_zone.values())
        zones_discovered = len(self.cubes_by_zone)

        total_visits = 0
        for zone_cubes in self.cubes_by_zone.values():
            total_visits += sum(cube.visit_count for cube in zone_cubes)

        # Ajouter camp_cubes pour les stats
        camp_cubes = len(self.cubes_by_zone.get(0, []))

        return {
            'total_cubes': total_cubes,
            'zones_discovered': zones_discovered,
            'total_visits': total_visits,
            'compression_count': self.compression_count,
            'cubes_created': self.total_cubes_created,
            'camp_cubes': camp_cubes,
            'exploration_cubes': {
                zone_id: [
                    {
                        'center_x': cube.center_x,
                        'center_y': cube.center_y,
                        'center_z': cube.center_z,
                        'size_x': cube.size_x,
                        'size_y': cube.size_y,
                        'size_z': cube.size_z,
                        'size': cube.size,  # Moyenne pour compatibilité
                        'visit_count': cube.visit_count,  # Compteur réel (pour affichage)
                        'effective_visit_count': cube.effective_visit_count, # Compteur effectif (pour rewards)
                        'total_visits': cube.total_visits,
                        'zone_id': cube.zone_id,
                        'markers': cube.markers,  # Marqueurs de cube
                        'blocked_directions': cube.blocked_directions
                    }
                    for cube in cubes
                ]
                for zone_id, cubes in self.cubes_by_zone.items()
            }
        }

    def get_detailed_map_info(self) -> str:
        """
        Retourne un résumé détaillé de la carte
        """
        lines = [
            "" + "=" * 70,
            "🗺️  CARTE D'EXPLORATION",
            "=" * 70
        ]

        map_info_stats = self.get_stats()
        lines.append(f"📊 Statistiques globales:")
        lines.append(f"Cubes actifs: {map_info_stats['total_cubes']}")
        lines.append(f"Zones découvertes: {map_info_stats['zones_discovered']}")
        lines.append(f"Visites totales: {map_info_stats['total_visits']}")
        lines.append(f"Compressions: {map_info_stats['compression_count']}")

        lines.append(f"🗺️  Détails par zone:")
        for zone_id, cubes in sorted(self.cubes_by_zone.items()):
            total_visits = sum(cube.visit_count for cube in cubes)
            avg_visits = total_visits / len(cubes) if cubes else 0
            lines.append(f"Zone {zone_id}: {len(cubes)} cubes, "
                         f"{total_visits} visites (moy: {avg_visits:.1f})")

        lines.append("=" * 70)
        return "".join(lines)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    print("🧪 Test création cubes dans le camp\n")

    tracker = ExplorationTracker(cube_size=650)

    # Test 1 : Premier cube dans le camp
    print("1️⃣ Test: Premier cube dans le camp (zone 0)")
    result = tracker.update_position(
        x=100.0, y=50.0, z=200.0,
        zone_id=0
    )

    assert result['new_cube'] == True, "❌ Le cube devrait être nouveau!"
    assert result['discovery_reward'] > 0, "❌ La reward devrait être > 0!"

    # Le résultat n'a PAS de 'cube_key', c'est interne
    print(f"✅ Nouveau cube créé")
    print(f"   Reward brute: {result['discovery_reward']:.2f}")
    print(f"   Visit count: {result['visit_count']}\n")

    # Test 2 : Revisiter le même cube
    print("2️⃣ Test: Revisiter le même cube")
    result2 = tracker.update_position(
        x=110.0, y=55.0, z=210.0,  # Même cube (delta < 200)
        zone_id=0
    )

    assert result2['new_cube'] == False, "❌ Le cube devrait exister!"
    # La reward peut être > 0
    print(f"✅ Cube existant revisité")
    print(f"   Visit count: {result2['visit_count']}")
    print(f"   Reward: {result2['discovery_reward']:.2f}\n")

    # Test 3 : Nouveau cube ailleurs dans le camp
    print("3️⃣ Test: Nouveau cube ailleurs dans le camp")
    result3 = tracker.update_position(
        x=500.0, y=100.0, z=600.0,  # Nouveau cube (delta > 200)
        zone_id=0
    )

    assert result3['new_cube'] == True, "❌ Le cube devrait être nouveau!"
    print(f"✅ Deuxième cube créé\n")

    # Stats finales
    print("4️⃣ Stats finales:")
    stats = tracker.get_stats()
    print(f"   Total cubes: {stats['total_cubes']}")
    print(f"   Cubes dans le camp: {stats['camp_cubes']}")
    print(f"   Zones découvertes: {stats['zones_discovered']}")

    assert stats['camp_cubes'] == 2, f"❌ Le camp devrait avoir 2 cubes (a {stats['camp_cubes']})!"
    assert 0 in tracker.cubes_by_zone, "❌ Zone 0 devrait être dans cubes_by_zone!"

    print("\n✅ Tous les tests réussis!")