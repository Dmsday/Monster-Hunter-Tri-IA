"""
Memory and vision data merging
v1.5: Memory vector extended to 70 features
"""

import time
import numpy as np
from typing import Dict, Tuple

from utils.memory_vector import build_memory_vector
from core.exploration_map_incremental import ExplorationMapIncremental
from info.module_logger import get_module_logger
logger = get_module_logger('state_fusion')

class StateFusion:
    """
    Merges vision + memory
    """

    def __init__(
            self,
            memory_reader,
            frame_preprocessor
    ):
        self.memory = memory_reader
        self.preprocessor = frame_preprocessor

        self.last_valid_position = (0.0, 0.0, 0.0)
        self.last_valid_death_count = 0.0
        self._fusion_call_count = 0   # Initialized here, incremented in get_fused_state()
        self._map_debug_count = 0     # Initialized here, incremented in create_exploration_map_with_channels()

        # Gestionnaire d'exploration map optimisé
        self.exp_map_manager = ExplorationMapIncremental(
            grid_size=15,
            radius=1000.0
        )

        # Injecter map_manager dans tracker (si disponible)
        if self.memory is not None and hasattr(self.memory, 'reward_calc'):
            reward_calc = self.memory.reward_calc
            if reward_calc is not None and hasattr(reward_calc, 'exploration_tracker'):
                tracker = reward_calc.exploration_tracker
                tracker.map_manager = self.exp_map_manager
                logger.debug("map_manager injecte dans ExplorationTracker")

    def get_fused_state(self, frame: np.ndarray) -> Dict:
        """
        Creates the complete state combining vision and memory

        Returns:
            Dict containing:
                - 'visual': Preprocessed and stacked frames (0-1)
                - 'memory': Memory data vector (67 features)
                - 'exploration_map': Local exploration minimap (15x15x3)
                - 'raw_memory': Raw memory data
        """
        # 1. Traiter la frame
        visual_state = self.preprocessor.process_and_stack(frame)

        # 2. Lire la mémoire
        raw_memory = self.memory.read_game_state()

        # 3. Créer vecteur mémoire
        reward_calc = getattr(self.memory, 'reward_calc', None)
        memory_vector = build_memory_vector(raw_memory, reward_calc=reward_calc)

        # 4. Préparer position pour mini-carte
        player_x = raw_memory.get('player_x', 0.0)
        player_y = raw_memory.get('player_y', 0.0)
        player_z = raw_memory.get('player_z', 0.0)
        current_zone = raw_memory.get('current_zone', 0) or 0

        # Synchroniser marqueurs AVANT création map
        if hasattr(self.memory, 'reward_calc') and hasattr(self.memory.reward_calc, 'exploration_tracker'):
            tracker = self.memory.reward_calc.exploration_tracker
            tracker.sync_all_markers_to_cubes()

        # 5. Créer mini-carte d'exploration
        exploration_map = self.create_exploration_map_with_channels(
            player_pos=(player_x, player_y, player_z),
            current_zone=current_zone,
        )

        # VÉRIFICATION AVANT RETOUR (tous les 1000 steps)
        fused_state = {
            'visual': visual_state,
            'memory': memory_vector,
            'exploration_map': exploration_map,
            'raw_memory': raw_memory
        }

        self._fusion_call_count += 1

        if self._fusion_call_count % 5000 == 0:
            logger.debug(f"[DEBUG state_fusion] get_fused_state() call #{self._fusion_call_count}")
            logger.debug(f"   Visual: {visual_state.shape} | Range: [{visual_state.min():.3f}, {visual_state.max():.3f}]")
            logger.debug(f"   Memory: {memory_vector.shape} | Sample: HP={memory_vector[0]:.1f}, Zone={memory_vector[7]:.0f}")
            logger.debug(f"   Exploration map: {exploration_map.shape} | Channels non-nuls: {[i for i in range(4) if exploration_map[:, :, i].max() > 0.01]}")
            # Afficher stats d'optimisation
            stats = self.exp_map_manager.get_stats()
            logger.debug(f"   📊 Stats optimisation map:")
            logger.debug(f"      • Updates incrémentaux: {stats['incremental_rate']:.1f}%")
            logger.debug(f"      • Recalculs complets: {stats['full_recalc_rate']:.1f}%")
            logger.debug(f"      • Cubes dirty actuels: {stats['dirty_cubes_current']}")

        return fused_state

    def create_exploration_map_with_channels(
            self,
            player_pos: Tuple[float, float, float],
            current_zone: int,
    ) -> np.ndarray:
        """
        Crée une mini-carte d'exploration avec 4 channels
        Version optimisée avec update incremental

         Args:
            player_pos: Position du joueur (x, y, z)
            current_zone: ID de la zone actuelle

         Channels:
            0: Cubes visités (intensité de visite effective)
            1: Position du joueur (hot spot au centre)
            2: Cubes récents (effective_visit_count < 5)
            3: Marqueurs de cube

        Returns:
            np.ndarray: (15, 15, 4) - Grille 15x15 avec 4 channels
        """
        px, py, pz = player_pos

        # Récupérer tracker et cubes
        if not hasattr(self.memory, 'reward_calc'):
            return np.zeros((15, 15, 4), dtype=np.float32)

        reward_calc = self.memory.reward_calc
        if not hasattr(reward_calc, 'exploration_tracker'):
            return np.zeros((15, 15, 4), dtype=np.float32)

        tracker = self.memory.reward_calc.exploration_tracker
        cubes = tracker.cubes_by_zone.get(current_zone, [])

        # Utiliser exp_map_manager (optimisation automatique)
        exploration_map = self.exp_map_manager.update(
            player_pos_update_method=(px, py, pz),
            current_zone=current_zone,
            cubes_update_method=cubes,
            force_full_recalc=False
        )

        self._map_debug_count += 1

        # Debug périodique
        if self._map_debug_count % 500 == 0:
            stats = self.exp_map_manager.get_stats()
            logger.debug(f"🗺️ [DEBUG #{self._map_debug_count}] Exploration map - Zone {current_zone}")
            logger.debug(f"Position joueur: ({px:.0f}, {py:.0f}, {pz:.0f})")
            logger.debug(f"Nombre de cubes: {len(cubes)}")
            logger.debug(f"   📊 Optimisation:")
            logger.debug(f"Updates position-only: {stats['position_only_rate']:.1f}%")
            logger.debug(f"Updates incrémentaux: {stats['incremental_rate']:.1f}%")
            logger.debug(f"Recalculs complets: {stats['full_recalc_rate']:.1f}%")

        return exploration_map

    def get_state_shape(self, dummy_frame_shape: Tuple) -> Dict:
        """
        Retourne les shapes de l'état pour construire le modèle

        Args:
            dummy_frame_shape: Shape d'une frame d'exemple (H, W, C)

        Returns:
            Dict avec les shapes de chaque composant
        """
        # Créer frame dummy
        dummy_frame_test = np.zeros(dummy_frame_shape, dtype=np.uint8)
        dummy_state = self.get_fused_state(dummy_frame_test)

        return {
            'visual_shape': dummy_state['visual'].shape,
            'memory_shape': dummy_state['memory'].shape,
            'total_visual_dim': np.prod(dummy_state['visual'].shape),
            'total_memory_dim': len(dummy_state['memory'])
        }

    def is_game_running(self) -> bool:
        """
        Vérifie si le jeu tourne correctement
        """
        game_running_state = self.memory.read_game_state()
        hp = game_running_state.get('player_hp')

        if hp is None:
            return False

        pos = {
            'x': game_running_state.get('player_x'),
            'y': game_running_state.get('player_y'),
            'z': game_running_state.get('player_z')
        }

        if all(v is None for v in pos.values()):
            return False

        return True

    def wait_for_game_start(self, timeout=30, check_interval=1):
        """Attend que le jeu soit prêt"""
        start_time = time.time()

        logger.info("⏳ Attente du démarrage du jeu...")
        logger.info("   💡 Assure-toi d'être EN JEU dans une quête !")

        attempts = 0
        max_attempts = int(timeout / check_interval)

        while time.time() - start_time < timeout:
            if self.is_game_running():
                logger.info("Jeu détecté et actif!")
                return True

            attempts += 1
            if attempts % 5 == 0:
                logger.info(f"   ⏳ Tentative {attempts}/{max_attempts}...")

            time.sleep(check_interval)

        logger.error("❌ Timeout: le jeu n'a pas démarré")
        logger.error("💡 SOLUTION : Va dans le jeu et LANCE UNE QUÊTE")
        return False

    @staticmethod
    def print_fused_state_info(fused_state: Dict):
        """
        Debug: affiche les infos de l'état fusionné
        """
        logger.info("" + "=" * 60)
        logger.info("🧩 ÉTAT FUSIONNÉ")
        logger.info("=" * 60)

        logger.info(f"📸 Visual State:")
        logger.info(f"   Shape: {fused_state['visual'].shape}")
        logger.info(f"   Dtype: {fused_state['visual'].dtype}")
        logger.info(f"   Range: [{fused_state['visual'].min():.3f}, {fused_state['visual'].max():.3f}]")

        logger.info(f"🧠 Memory Vector :")
        logger.info(f"   Shape: {fused_state['memory'].shape}")

        logger.info(f"📋 Features détaillées:")
        feature_names = [
            # Base (13)
            "HP", "HP récupérable", "Stamina",
            "Player X", "Player Y", "Player Z",
            "Orientation", "Zone", "Damage last hit",
            "Money", "Death count", "Stamina low", "Time underwater",
            # Inventaire
            "Slot 1 ID", "Slot 1 Qty",
            "Slot 2 ID", "Slot 2 Qty",
            "Slot 3 ID", "Slot 3 Qty",
            "Slot 4 ID", "Slot 4 Qty",
        ]

        for i, (name, value) in enumerate(zip(feature_names, fused_state['memory'])):
            marker = " " if i >= 13 else "  "
            logger.info(f"{marker}[{i:2d}] {name:20s}: {value:10.2f}")

        # Afficher inventaire de façon lisible
        if len(fused_state['memory']) >= 21:
            logger.info(f"🎒 Inventaire (4 premiers slots) :")
            for slot in range(1, 5):
                idx_id = 13 + (slot - 1) * 2
                idx_qty = idx_id + 1

                item_id = int(fused_state['memory'][idx_id])
                quantity = int(fused_state['memory'][idx_qty])

                if item_id > 0:
                    logger.info(f"   Slot {slot}: Item ID {item_id:3d} x{quantity:2d}")
                else:
                    logger.info(f"   Slot {slot}: (vide)")

        #
        logger.info(f"⚔️  Features de Combat :")
        logger.info(f"   [61] Quest time      : {fused_state['memory'][61]:.0f}s")
        logger.info(f"   [62] Attack/Defense  : {fused_state['memory'][62]:.0f}")
        logger.info(f"   [63] Monster count   : {fused_state['memory'][63]:.0f}")
        logger.info(f"   [64] Monsters present: {fused_state['memory'][64]:.0f}")
        logger.info(f"   [65] Sharpness       : {fused_state['memory'][65]:.0f}")
        logger.info(f"   [66] In-game menu    : {fused_state['memory'][66]:.0f}")
        logger.info(f"   [67] Item selected   : {fused_state['memory'][67]:.0f}")
        logger.info(f"   [68] In combat       : {fused_state['memory'][68]:.0f}")
        logger.info(f"   [69] In monster zone : {fused_state['memory'][69]:.0f}")

        logger.info(f"📊 Raw Memory:")
        for key, value in fused_state['raw_memory'].items():
            if value is not None and key not in ['inventory_items']:
                logger.info(f"   {key:25s}: {value}")

        logger.info("=" * 60)


# Test
if __name__ == "__main__":
    print("🧪 Test State Fusion avec inventaire (67 features)\n")

    # Simuler
    class DummyMemoryForTest:
        @staticmethod
        def read_game_state():
            return {
                'player_hp': 100.0,
                'player_stamina': 80.0,
                'player_x': 10.5,
                'player_y': 5.2,
                'player_z': -3.7,
                'player_orientation': 45.0,
                'current_zone': 5,
                'damage_last_hit': 0.0,
                'money': 5000,
                'death_count': 0,
                'stamina_low': False,
                'time_underwater': 0,
                'sharpness': 150,
                'in_game_menu': False,
                'inventory_items': [
                    {'slot': 1, 'item_id': 13, 'quantity': 10},  # Potion
                    {'slot': 2, 'item_id': 98, 'quantity': 20},  # Aiguisoir
                    {'slot': 3, 'item_id': 42, 'quantity': 5},  # Carte
                    # Slot 4 vide
                ]
            }

    class DummyPreprocessorForTest:
        @staticmethod
        def process_and_stack(_frame):
            return np.zeros((84, 84, 4), dtype=np.float32)

    memory = DummyMemoryForTest()
    preprocessor = DummyPreprocessorForTest()

    fusion = StateFusion(memory, preprocessor)

    # Test
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = fusion.get_fused_state(dummy_frame)

    StateFusion.print_fused_state_info(state)

    print("\nTest réussi!")
    print(f"   Vecteur mémoire : {state['memory'].shape} (70 features)")