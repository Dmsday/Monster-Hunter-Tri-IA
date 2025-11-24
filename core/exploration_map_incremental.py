"""
Optimisation de l'exploration map avec update incrémental

Au lieu de recalculer toute la map à chaque mouvement,
on ne met à jour que les cellules affectées.

Stratégie :
    1. Cache de la map complète
    2. Tracking des cubes "dirty" (modifiés depuis dernier calcul)
    3. Update seulement des cellules contenant ces cubes
    4. Update position joueur (toujours)
"""

import numpy as np
from typing import Tuple, Set, Optional

# Import MarkerType pour les marqueurs
try:
    from environment.cube_markers import MarkerType
except ImportError:
    # Fallback si module non trouvé
    class MarkerType:
        """
        Fallback pour MarkerType
        """
        DANGER = "danger"
        MONSTER_LOCATION = "monster_location"
        ZONE_TRANSITION = "zone_transition"
        WATER = "water"
        SAFE = "safe"
        RESOURCE = "resource"
        LIMITE = "limite"


class ExplorationMapIncremental:
    """
    Gestionnaire optimisé de l'exploration map

    Usage dans StateFusion :
        self.exp_map_manager = ExplorationMapIncremental(
            grid_size=15,
            radius=1000.0
        )

        map_result = self.exp_map_manager.update(
            player_pos=(px, py, pz),
            current_zone=zone,
            cubes=tracker.cubes_by_zone.get(zone, [])
        )
    """

    def __init__(
            self,
            grid_size: int = 15,
            radius: float = 1000.0
    ):
        self.grid_size = grid_size
        self.radius = radius
        self.cell_size = (radius * 2) / grid_size
        self.center = grid_size // 2

        # Cache
        self._current_map = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        self._last_player_pos: Optional[Tuple[float, float, float]] = None
        self._last_zone: Optional[int] = None

        # Tracking cubes modifiés
        self._dirty_cubes: Set[int] = set()  # Set d'IDs de cubes
        self._cube_positions: dict = {}  # Mapping cube_id → (grid_x, grid_z)

        # Stats
        self._total_updates = 0
        self._position_only_updates = 0
        self._incremental_updates = 0
        self._full_recalculations = 0

    def update(
            self,
            player_pos_update_method: Tuple[float, float, float],
            current_zone: int,
            cubes_update_method: list,
            force_full_recalc: bool = False
    ) -> np.ndarray:
        """
        Met à jour la map (incrémental si possible)

        Args:
            player_pos_update_method: (x, y, z) position joueur
            current_zone: ID zone actuelle
            cubes_update_method: Liste des cubes dans la zone
            force_full_recalc: Forcer recalcul complet

        Returns:
            np.ndarray (15, 15, 4) exploration map
        """
        self._total_updates += 1
        px, py, pz = player_pos_update_method

        # Détection changement de zone
        zone_changed = (current_zone != self._last_zone)

        if zone_changed or force_full_recalc:
            # Recalcul complet
            self._full_recalc(player_pos_update_method, current_zone, cubes_update_method)
            self._full_recalculations += 1
            return self._current_map.copy()

        # Détection mouvement joueur
        if self._last_player_pos is not None:
            last_px, _, last_pz = self._last_player_pos
            distance_moved = np.sqrt(
                (px - last_px) ** 2 + (pz - last_pz) ** 2
            )
        else:
            distance_moved = float('inf')

        # Stratégie update
        if distance_moved < 10.0 and not self._dirty_cubes:
            # Cas 1 : Mouvement minimal - juste position joueur
            self._update_player_position_only(px, pz)
            self._position_only_updates += 1
            return self._current_map.copy()

        elif distance_moved < 50.0 and len(self._dirty_cubes) > 0:
            # Cas 2 : Update incrémental
            self._incremental_update(player_pos_update_method, cubes_update_method)
            self._incremental_updates += 1
            return self._current_map.copy()

        else:
            # Cas 3 : Mouvement important - recalcul complet
            self._full_recalc(player_pos_update_method, current_zone, cubes_update_method)
            self._full_recalculations += 1
            return self._current_map.copy()

    def _update_player_position_only(self, px: float, pz: float):
        """
        Update seulement channel 1 (position joueur)
        Très rapide (~0.1ms)
        """
        # Effacer ancien channel 1
        self._current_map[:, :, 1] = 0.0

        # Gaussienne centrée
        for g in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((g - self.center) ** 2 + (j - self.center) ** 2)
                self._current_map[g, j, 1] = np.exp(-dist ** 2 / (2 * 2.0 ** 2))

        # Update last position
        self._last_player_pos = (px, 0, pz)

    def _incremental_update(
            self,
            player_pos_incremental_update_method: Tuple[float, float, float],
            cubes_incremental_update_method: list
    ):
        """
        Update incrémental : seulement les cellules avec cubes dirty
        Rapide (~0.5-1ms selon nombre de dirty cubes)
        """
        px, py, pz = player_pos_incremental_update_method

        # Identifier cellules à mettre à jour
        cells_to_update = set()

        for cube in cubes_incremental_update_method:
            cube_id = id(cube)

            # Si cube dirty OU nouveau
            if cube_id in self._dirty_cubes or cube_id not in self._cube_positions:
                # Calculer position grille
                dx = cube.center_x - px
                dz = cube.center_z - pz

                if abs(dx) <= self.radius and abs(dz) <= self.radius:
                    grid_x = int((dx + self.radius) / self.cell_size)
                    grid_z = int((dz + self.radius) / self.cell_size)

                    if 0 <= grid_x < self.grid_size and 0 <= grid_z < self.grid_size:
                        cells_to_update.add((grid_x, grid_z))
                        self._cube_positions[cube_id] = (grid_x, grid_z)

        # Recalculer seulement ces cellules
        for grid_x, grid_z in cells_to_update:
            # Reset cette cellule (channels 0, 2, 3)
            self._current_map[grid_z, grid_x, 0] = 0.0
            self._current_map[grid_z, grid_x, 2] = 0.0
            self._current_map[grid_z, grid_x, 3] = 0.0

            # Recalculer avec TOUS les cubes dans cette cellule
            self._compute_cell(grid_x, grid_z, px, pz, cubes_incremental_update_method)

        # Update position joueur
        self._update_player_position_only(px, pz)

        # Clear dirty cubes
        self._dirty_cubes.clear()

    def _compute_cell(
            self,
            grid_x: int,
            grid_z: int,
            px: float,
            pz: float,
            cubes_compute_cell_method: list
    ):
        """
        Calcule les valeurs d'UNE cellule
        """
        for cube in cubes_compute_cell_method:
            dx = cube.center_x - px
            dz = cube.center_z - pz

            # Vérifier si cube dans cette cellule
            cube_grid_x = int((dx + self.radius) / self.cell_size)
            cube_grid_z = int((dz + self.radius) / self.cell_size)

            if cube_grid_x != grid_x or cube_grid_z != grid_z:
                continue

            # CHANNEL 0 : Intensité de visite
            visit_intensity = min(cube.effective_visit_count / 10.0, 1.0)
            self._current_map[grid_z, grid_x, 0] = max(
                self._current_map[grid_z, grid_x, 0],
                visit_intensity
            )

            # CHANNEL 2 : Cubes récents
            if cube.effective_visit_count < 5:
                self._current_map[grid_z, grid_x, 2] = 1.0

            # CHANNEL 3 : Marqueurs (priorité max)
            markers = cube.markers
            marker_strength = 0.0
            highest_priority = -1

            for marker_type, marker in markers.items():
                priority = self._get_marker_priority(marker_type)
                weight = self._get_marker_weight(marker_type)

                if priority > highest_priority:
                    highest_priority = priority
                    marker_strength = marker.strength * weight

            if marker_strength > 0:
                self._current_map[grid_z, grid_x, 3] = min(
                    marker_strength,
                    1.0
                )

    @staticmethod
    def _get_marker_priority(marker_type) -> int:
        """Priorité du marqueur"""
        # Supporter à la fois l'enum et les strings
        if isinstance(marker_type, str):
            marker_str = marker_type
        elif hasattr(marker_type, 'value'):
            marker_str = marker_type.value
        else:
            marker_str = str(marker_type)

        priority_map = {
            'danger': 100,
            'monster_location': 90,
            'zone_transition': 70,
            'water': 50,
            'safe': 30,
            'resource': 20,
            'limite': 10
        }
        return priority_map.get(marker_str, 0)

    @staticmethod
    def _get_marker_weight(marker_type) -> float:
        """Poids du marqueur pour intensité"""
        # Supporter à la fois l'enum et les strings
        if isinstance(marker_type, str):
            marker_str = marker_type
        elif hasattr(marker_type, 'value'):
            marker_str = marker_type.value
        else:
            marker_str = str(marker_type)

        weight_map = {
            'danger': 1.0,
            'monster_location': 0.9,
            'zone_transition': 0.6,
            'water': 0.4,
            'safe': 0.3,
            'resource': 0.2,
            'limite': 0.1
        }
        return weight_map.get(marker_str, 0.0)

    def _full_recalc(
            self,
            player_pos_full_recalc_method: Tuple[float, float, float],
            current_zone: int,
            cubes_full_recalc_method: list
    ):
        """
        Recalcul complet de toute la map
        Utilisé lors changement de zone ou mouvement important
        """
        px, py, pz = player_pos_full_recalc_method

        # Reset map
        self._current_map.fill(0.0)
        self._cube_positions.clear()

        # CHANNEL 1 : Position joueur (TOUJOURS en premier)
        for gs in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((gs - self.center) ** 2 + (j - self.center) ** 2)
                self._current_map[gs, j, 1] = np.exp(-dist ** 2 / (2 * 2.0 ** 2))

        # CHANNELS 0, 2, 3 : Parcourir tous les cubes
        for cube in cubes_full_recalc_method:
            dx = cube.center_x - px
            dz = cube.center_z - pz

            if abs(dx) > self.radius or abs(dz) > self.radius:
                continue

            grid_x = int((dx + self.radius) / self.cell_size)
            grid_z = int((dz + self.radius) / self.cell_size)

            if 0 <= grid_x < self.grid_size and 0 <= grid_z < self.grid_size:
                # Sauvegarder position
                cube_id = id(cube)
                self._cube_positions[cube_id] = (grid_x, grid_z)

                # Calculer cette cellule
                self._compute_cell(grid_x, grid_z, px, pz, [cube])

        # Update trackers
        self._last_player_pos = player_pos_full_recalc_method
        self._last_zone = current_zone
        self._dirty_cubes.clear()

    def mark_cube_dirty(self, cube):
        """
        Marque un cube comme modifié (à recalculer)
        À appeler quand un cube est visité
        """
        self._dirty_cubes.add(id(cube))

    def get_stats(self) -> dict:
        """Stats de performance"""
        if self._total_updates == 0:
            return {
                'total_updates': 0,
                'position_only_updates': 0,
                'incremental_updates': 0,
                'full_recalculations': 0,
                'position_only_rate': 0.0,
                'incremental_rate': 0.0,
                'full_recalc_rate': 0.0,
                'dirty_cubes_current': 0
            }

        return {
            'total_updates': self._total_updates,
            'position_only_updates': self._position_only_updates,
            'incremental_updates': self._incremental_updates,
            'full_recalculations': self._full_recalculations,
            'position_only_rate': (self._position_only_updates / self._total_updates) * 100,
            'incremental_rate': (self._incremental_updates / self._total_updates) * 100,
            'full_recalc_rate': (self._full_recalculations / self._total_updates) * 100,
            'dirty_cubes_current': len(self._dirty_cubes)
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Test ExplorationMapIncremental\n")

    # Créer manager
    manager = ExplorationMapIncremental(grid_size=15, radius=1000.0)

    # Cube dummy pour test
    class DummyCube:
        def __init__(self, x, y, z):
            self.center_x = x
            self.center_y = y
            self.center_z = z
            self.effective_visit_count = 1
            self.markers = {}

    # Simuler 100 updates
    print("🔄 Simulation de 100 updates...")
    cubes = [
        DummyCube(100, 0, 100),
        DummyCube(200, 0, 200),
        DummyCube(300, 0, 300),
    ]

    for i in range(100):
        # Position joueur qui bouge légèrement
        player_pos = (i * 2, 0, i * 2)

        # Marquer un cube dirty de temps en temps
        if i % 10 == 0 and cubes:
            manager.mark_cube_dirty(cubes[0])

        # Update
        map_result = manager.update(
            player_pos_update_method=player_pos,
            current_zone=5,
            cubes_update_method=cubes
        )

        # Stats tous les 20 frames
        if (i + 1) % 20 == 0:
            stats = manager.get_stats()
            print(f"\nFrame {i+1}:")
            print(f"   Position-only: {stats['position_only_updates']} ({stats['position_only_rate']:.1f}%)")
            print(f"   Incrémentaux: {stats['incremental_updates']} ({stats['incremental_rate']:.1f}%)")
            print(f"   Recalculs: {stats['full_recalculations']} ({stats['full_recalc_rate']:.1f}%)")

    # Stats finales
    final_stats = manager.get_stats()
    print(f"\n📊 Résultat final:")
    print(f"   Taux position-only: {final_stats['position_only_rate']:.1f}%")
    print(f"   Taux incrémental: {final_stats['incremental_rate']:.1f}%")
    print(f"   Taux recalcul: {final_stats['full_recalc_rate']:.1f}%")
    
    # Critère de succès
    optimized_rate = final_stats['position_only_rate'] + final_stats['incremental_rate']
    if optimized_rate > 70:  # 70%
        print(f"Optimisation fonctionne ! ({optimized_rate:.1f}% optimisé)")
    else:
        print(f"Optimisation suboptimale ({optimized_rate:.1f}% < 70%)")