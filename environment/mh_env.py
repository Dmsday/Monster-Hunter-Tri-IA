"""
Environnement Gymnasium pour Monster Hunter Tri
"""

# ============================================================================
# IMPORTS STANDARD PYTHON
# ============================================================================
import time                    # Gestion des pauses et délais
import numpy as np            # Calculs numériques et arrays
import traceback              # Affichage détaillé des erreurs
from typing import Dict, Optional, Union, List       # Type hints pour les dictionnaires

# ============================================================================
# VISUALISATION TEMPS RÉEL
# ============================================================================
try:
    import cv2                # OpenCV pour affichage vision temps réel
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # Pour éviter les erreurs si utilise

try:
    import matplotlib.pyplot as plt  # Matplotlib pour minimap 3D
    from mpl_toolkits.mplot3d import Axes3D  # Axes 3D pour minimap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # Pour éviter les erreurs si utilise

# ============================================================================
# GYMNASIUM (REINFORCEMENT LEARNING)
# ============================================================================
import gymnasium as gym       # Framework RL (anciennement OpenAI Gym)
from gymnasium import spaces  # Définition des espaces d'actions/observations
import threading              # Parallelise les observations sur le GPU avec l'entrainement sur le CPU
import queue

# ============================================================================
# PYNPUT (CONTRÔLE CLAVIER)
# ============================================================================
from pynput.keyboard import Controller as KeyboardController, Key
# - KeyboardController : Simulation des touches clavier (pour F5)
# - Key : Touches spéciales (F1-F12, Ctrl, Alt, etc.)

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
# Lecture mémoire Dolphin
from core.dynamic_memory_reader import MemoryReader

# Utilitaires
from utils.safe_float import safe_float  # Conversion sécurisée float (évite NaN/Inf)
from utils.module_logger import get_module_logger
logger = get_module_logger('mh_env')

# Vision (capture et preprocessing)
from vision.frame_capture import FrameCapture        # Capture frames Dolphin
from vision.preprocessing import FramePreprocessor   # Preprocessing images

# Données de fusion (vision + mémoire)
from core.state_fusion import StateFusion

# Contrôleur Wii (clavier ou manette virtuelle)
from core.controller import WiiController

# Système de récompenses
from environment.reward_calculator import MonsterHunterRewardCalculator

# Vérification disponibilité pynput
try:
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput non installé (pip install pynput)")


class MonsterHunterEnv(gym.Env):
    """
    Environnement RL hybride pour Monster Hunter Tri
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            use_vision=True,
            use_memory=True,
            frame_size=(84, 84),
            grayscale=False,
            frame_stack=4,
            action_repeat=4,
            render_mode=None,
            use_controller=False,
            controller_debug=False,
            use_advanced_rewards=True,
            auto_reload_save_state=True,
            save_state_slot=5,
            rt_vision = False,
            rt_minimap = False,
            instance_id=0,
    ):
        super().__init__()

        self.instance_id = instance_id
        self.use_vision = use_vision
        self.use_memory = use_memory
        self.action_repeat = action_repeat
        self.render_mode = render_mode
        self.use_controller = use_controller
        self.auto_reload_save_state = auto_reload_save_state
        self.save_state_slot = save_state_slot
        self.rt_vision = rt_vision
        self.rt_minimap = rt_minimap

        # Fenêtre OpenCV pour vision temps réel
        self.rt_window_name: Optional[str] = None
        if self.rt_vision:
            if not CV2_AVAILABLE or cv2 is None:
                logger.warning("OpenCV non disponible - vision temps réel désactivée")
                self.rt_vision = False
            else:
                try:
                    self.rt_window_name = "IA Vision - Temps Réel"
                    cv2.namedWindow(self.rt_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.rt_window_name, 1200, 800)
                    logger.info("Fenêtre vision temps réel activée")
                except Exception as rt_init_error:
                    logger.error(f"Erreur initialisation fenêtre : {rt_init_error}")
                    self.rt_vision = False
                    self.rt_window_name = None

        # Fenêtre minimap 3D (si demandée)
        self.rt_minimap_window: Optional[bool] = None
        self.rt_minimap_fig = None
        self.rt_minimap_ax = None

        if self.rt_minimap and self.rt_vision:
            if not MATPLOTLIB_AVAILABLE or plt is None:
                logger.warning("Matplotlib non disponible - minimap désactivée")
                self.rt_minimap = False
            else:
                try:
                    plt.ion()  # Mode interactif
                    self.rt_minimap_fig = plt.figure(figsize=(8, 6))
                    self.rt_minimap_ax = self.rt_minimap_fig.add_subplot(111, projection='3d')
                    self.rt_minimap_window = True
                    logger.info("Fenêtre minimap 3D activée")
                except Exception as minimap_init_error:
                    logger.error(f"Erreur initialisation minimap : {minimap_init_error}")
                    self.rt_minimap = False
                    self.rt_minimap_window = None

        logger.info("🎮 Initialisation de l'environnement Monster Hunter v2.1.5...")

        # Clavier pour F5
        if self.auto_reload_save_state:
            if PYNPUT_AVAILABLE:
                self.keyboard = KeyboardController()
                self.save_state_reload_count = 0
                logger.info("Auto-reload save state activé (F5)")
            else:
                logger.warning("pynput manquant - auto-reload désactivé")
                self.auto_reload_save_state = False

        # Initialiser les attributs à None AVANT pour éviter AttributeError
        self.frame_capture = None
        self.preprocessor = None
        self.memory = None
        self.state_fusion = None

        # --- 1. MEMORY EN PREMIER (car state_fusion en dépend) ---
        # TOUJOURS INITIALISER MEMORY (pour rewards), même si use_memory=False
        if use_memory or True:  # Always init memory for rewards
            try:
                self.memory = MemoryReader(
                    force_quest_mode=True,
                    async_mode=True,
                    read_frequency=100,
                )

                # Check memory is actually available
                if self.memory is None:
                    logger.error("CRITICAL: MemoryReader returned None")
                    logger.error("Possible causes:")
                    logger.error("  1. Dolphin not running")
                    logger.error("  2. Dolphin not started as admin")
                    logger.error("  3. Game not loaded")
                    raise RuntimeError("MemoryReader initialization failed")

                logger.info("Memory reader en mode asynchrone (lecture rewards)")

                # Si use_memory=False, on lit quand meme pour les rewards
                if not use_memory:
                    logger.info("Memory non utilisee pour l'agent (mais active pour rewards)")

            except Exception as init_memory_error:
                logger.error(f"ERREUR critique: MemoryReader non disponible: {init_memory_error}")
                logger.error("Les rewards ne fonctionneront pas")
                self.memory = None
        else:
            # Cas theorique (jamais atteint grace au "or True")
            logger.warning("Memory desactivee completement (rewards non disponibles)")
            self.memory = None

        # --- 2. VISION ENSUITE ---
        if use_vision:
            try:
                # Force PrintWindow if rtvision active for stability
                force_pw = self.rt_vision or self.rt_minimap

                # MULTI-INSTANCE : Pass expected window title
                expected_title = f"MHTri-{self.instance_id}" if self.instance_id >= 0 else None

                self.frame_capture = FrameCapture(
                    target_fps=30,
                    force_printwindow=force_pw,
                    instance_id=self.instance_id,
                    expected_window_title=expected_title
                )

                self.preprocessor = FramePreprocessor(
                    target_size=frame_size,
                    grayscale=grayscale,
                    frame_stack=frame_stack,
                )

                logger.info("Vision initialisée")

            except Exception as vision_error:
                logger.error(f"ERREUR vision: {vision_error}")
                self.use_vision = False
                # Mettre à None si erreur
                self.frame_capture = None
                self.preprocessor = None

        # Verify window title matches expected format
        # Expected: MHTri-0, MHTri-1, MHTri-2...
        if self.frame_capture:
            try:
                window_title = self.get_window_title()
                expected_title = f"MHTri-{self.instance_id}"

                if window_title:
                    if expected_title not in window_title:
                        logger.error("=" * 70)
                        logger.error(f"ERREUR CRITIQUE INSTANCE #{self.instance_id}")
                        logger.error("=" * 70)
                        logger.error(f"Fenêtre capturée incorrecte")
                        logger.error(f"  Attendu : '{expected_title}'")
                        logger.error(f"  Trouvé : '{window_title}'")
                        logger.error("")
                        logger.error("RISQUE DE COLLISION ENTRE INSTANCES")
                        logger.error("")
                        logger.error("Solutions :")
                        logger.error("  1. Vérifier script PowerShell (renommage)")
                        logger.error("  2. Vérifier ordre lancement instances")
                        logger.error("  3. Augmenter délai initial (--dolphin-delay)")
                        logger.error("=" * 70)

                        # Ne pas raise car peut être faux positif
                        # Mais logger clairement pour investigation
                else:
                    logger.warning(f"Instance #{self.instance_id} : Titre fenêtre non détecté")

            except Exception as frame_capture_with_multi_instances_check_error:
                logger.debug(f"Erreur vérification fenêtre : {frame_capture_with_multi_instances_check_error}")

        # --- 3. REWARD CALCULATOR ---
        if use_advanced_rewards:
            self.reward_calc = MonsterHunterRewardCalculator()
            logger.info("Reward calculator avancé activé")

            # Attach to MemoryReader with validation
            if self.memory is not None:  # Check self.memory, not use_memory
                self.memory.reward_calc = self.reward_calc
                logger.info("RewardCalculator attached to MemoryReader")

                # Verify tracker exists
                if hasattr(self.reward_calc, 'exploration_tracker'):
                    tracker = self.reward_calc.exploration_tracker
                    total_cubes = sum(len(cubes) for cubes in tracker.cubes_by_zone.values())
                    logger.info(
                        f"Exploration tracker active: {total_cubes} cubes in {len(tracker.cubes_by_zone)} zones")
                else:
                    logger.warning("Exploration tracker not found in RewardCalculator!")
            else:
                logger.error("CRITICAL: MemoryReader is None - rewards will not work")
                logger.error("Possible causes:")
                logger.error("  1. Dolphin not running")
                logger.error("  2. Memory reading failed during initialization")
                logger.error("  3. Game not loaded")
                logger.error("")
                logger.error("SOLUTION: Start Dolphin and load Monster Hunter Tri before training")

                # Raise exception to stop initialization cleanly
                raise RuntimeError(
                    "MemoryReader initialization failed. "
                    "Ensure Dolphin is running with Monster Hunter Tri loaded."
                )

        else:
            self.reward_calc = None

        # --- STATE FUSION EN DERNIER (nécessite vision ET memory) ---
        # Verifier que les attributs existent ET ne sont pas None
        # Si use_vision=True mais use_memory=False, on initialise quand même
        # pour avoir la minimap (memory est disponible pour rewards)
        if use_vision and use_memory is not None:
            if self.preprocessor is not None:
                try:
                    self.state_fusion = StateFusion(self.memory, self.preprocessor)

                    if use_memory:
                        logger.info("State fusion initialisé (vision + memory pour agent)")
                    else:
                        logger.info("State fusion initialisé (vision seule pour agent, memory pour rewards)")

                except Exception as fusion_error:
                    logger.error(f"ERREUR state fusion: {fusion_error}")
                    traceback.print_exc()
                    self.state_fusion = None
            else:
                self.state_fusion = None
                logger.warning("State fusion NON initialisé:")
                logger.warning(f"   - preprocessor: {self.preprocessor is not None}")
                logger.warning(f"   - memory: {self.memory is not None}")
        else:
            self.state_fusion = None
            if use_vision or use_memory:
                logger.error(f"State fusion non créé (use_vision={use_vision}, use_memory={use_memory})")

        # Contrôleur
        if use_controller:
            try:
                self.controller = WiiController(
                    debug=controller_debug,
                    use_controller=use_controller,
                    instance_id=self.instance_id,
                )
                if self.controller.is_connected:
                    logger.info("Contrôleur initialisé")
                else:
                    logger.warning("Contrôleur non connecté")
                    self.use_controller = False
                    self.controller = None
            except Exception as other_controller_error:
                logger.error(f"Contrôleur non disponible: {other_controller_error}")
                self.use_controller = False
                self.controller = None
        else:
            # MODE CLAVIER PAR DÉFAUT
            try:
                self.controller = WiiController(debug=controller_debug, use_controller=False)
                if self.controller.is_connected:
                    logger.info("Contrôleur clavier initialisé")
                else:
                    logger.warning("Contrôleur clavier non connecté")
                    self.controller = None
            except Exception as defaut_controller_error:
                logger.error(f"Contrôleur clavier non disponible: {defaut_controller_error}")
                self.controller = None

        # Logs multi-instance
        if self.instance_id > 0:
            logger.info("=" * 70)
            logger.info(f"ENVIRONNEMENT INSTANCE #{self.instance_id}")
            logger.info("=" * 70)
            logger.info(f"  Vision : {self.use_vision}")
            logger.info(f"  Mémoire : {self.use_memory}")
            logger.info(f"  Contrôleur : {'Manette' if use_controller else 'Clavier'}")
            logger.info(f"  Vision temps réel : {self.rt_vision}")

            # Vérifier fenêtre capturée
            if self.frame_capture:
                window_title = self.get_window_title()
                if window_title:
                    logger.info(f"  Fenêtre capturée : '{window_title}'")

                    # Vérifier correspondance
                    expected = f"MHTri-{self.instance_id}"
                    if expected not in window_title:
                        logger.warning(f"ATTENTION : Attendu '{expected}', trouvé '{window_title}'")
                        logger.warning(f"RISQUE DE COLLISION ENTRE INSTANCES")
                else:
                    logger.warning(f"Fenêtre capturée : Non détectée")

            logger.info("=" * 70)
            logger.info("")

        # === SPACES ===
        self._setup_spaces(frame_size, grayscale, frame_stack)

        # État
        self.current_state = None
        self.prev_raw_memory = None
        self.episode_start_time = None

        # Compteurs
        self.episode_steps = 0
        self.total_steps = 0
        self.episode_count = 0
        self.total_reward = 0

        # Variables pour info['episode'] (nécessaires pour SB3)
        self.episode_reward = 0.0
        self.episode_length = 0

        # Flag fin d'épisode en cours
        self._episode_ending = False

        # Thread dédié à la capture
        self._obs_queue = queue.Queue(maxsize=8)  # Buffer de 8 frames
        self._capture_thread = None
        self._capture_running = False
        self._capture_lock = threading.Lock()

        # Compteurs pour monitoring
        self._frames_dropped = 0
        self._frames_captured = 0
        self._frames_consumed = 0

        logger.info("Environnement prêt!")

    def _setup_spaces(self, frame_size, grayscale, frame_stack):
        """
        Configure les espaces d'action et d'observation
        """
        # Action space : 19 actions
        self.action_space = spaces.Discrete(19)

        # Observation space
        obs_spaces = {}

        # 1. VISUAL (si activé)
        if self.use_vision:
            channels = 1 if grayscale else 3
            vision_shape = (*frame_size, channels * frame_stack)
            obs_spaces['visual'] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=vision_shape,
                dtype=np.float32
            )

        # 2. MEMORY (si activé)
        if self.use_memory:
            # 70 features
            obs_spaces['memory'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(70,),
                dtype=np.float32
            )
            logger.info("🧠 Vecteur mémoire : 70 features")

        # 3. EXPLORATION MAP (si vision ET memory activés)
        # Même si use_memory=False, on peut avoir la minimap pour l'agent
        if self.use_vision and self.memory is not None:
            # Mini-carte 15x15 avec 3 channels
            # Channel 0 : Intensité visite
            # Channel 1 : Position joueur
            # Channel 2 : Cubes récents
            # Channel 3 : Marqueurs
            obs_spaces['exploration_map'] = spaces.Box(
                low=-1.0,  # -1.0 pour marquer position joueur
                high=1.0,
                shape=(15, 15, 4),
                dtype=np.float32
            )

            logger.info("Mini-carte exploration activée (15x15x4)")
            logger.info("   • Channel 0 : Intensité visite")
            logger.info("   • Channel 1 : Position joueur")
            logger.info("   • Channel 2 : Cubes récents")
            logger.info("   • Channel 3 : Marqueurs (transitions/monstres/eau)")

        # Construire observation space final
        if len(obs_spaces) > 1:
            # Plusieurs modalités --> Dict
            self.observation_space = spaces.Dict(obs_spaces)
            logger.info(f"📦 Observation space : Dict avec {len(obs_spaces)} modalités")
        elif len(obs_spaces) == 1:
            # UNE SEULE MODALITÉ : Garder en Dict aussi pour cohérence
            self.observation_space = spaces.Dict(obs_spaces)
            logger.info(f"📦 Observation space : Dict avec 1 modalité ({list(obs_spaces.keys())[0]})")
        else:
            # Fallback si rien n'est activé
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32
            )
            logger.warning(f"Observation space : Box de fallback (aucune modalité active)")

    def _async_capture_loop(self):
        """
        Thread séparé qui capture en continue
        Tourne en parallèle du GPU qui entraîne le modèle

        NOTE : Ce thread ne devrait être lancé QUE si use_vision=True
        """
        logger.debug("🎥 Thread de capture demarre")

        # PROTECTION : Vérifier que frame_capture existe
        if self.frame_capture is None:
            logger.error("Thread capture lance mais frame_capture est None")
            logger.error("Cela devrait arriver si entrainement lance sur memoire seulement (use_vision=False)")
            return  # Sortie propre du thread

        while self._capture_running:
            try:
                # 1. Capturer frame (CPU - 5ms avec GDI réutilisé)
                frame = self.frame_capture.capture_frame()

                # 2. Preprocessing (CPU - 15ms)
                if self.use_vision and self.preprocessor:
                    visual = self.preprocessor.process_and_stack_numpy(frame)
                else:
                    visual = None

                # 3. Mettre dans queue (non-bloquant)
                try:
                    self._obs_queue.put(visual, block=False)
                    self._frames_captured += 1
                except queue.Full:
                    # Queue pleine = PPO trop lent, jeter frame
                    self._frames_dropped += 1

                # Log périodique
                if self._frames_captured % 500 == 0:
                    drop_rate = (self._frames_dropped / self._frames_captured) * 100
                    logger.debug(f"Frame drop rate: {drop_rate:.2f}%")

                # 4. Rate limiting (30 FPS)
                time.sleep(1.0 / 30)

            except Exception as thread_capture_error:
                logger.error(f"Erreur thread capture: {thread_capture_error}")
                time.sleep(0.1)  # Pause avant retry

        logger.debug("🛑 Thread de capture arrêté")

    def reset(self, seed=None, options=None):
        """
        Reset l'environnement
        """
        super().reset(seed=seed)

        logger.info(f"Reset épisode #{self.episode_count + 1}...")

        # PROTECTION : Forcer nettoyage AVANT tout calcul
        self.prev_raw_memory = None

        # Vérifier si on vient de détecter une fin de quête
        if hasattr(self, '_quest_end_cooldown_until'):
            logger.debug(f"Attente fin du cooldown détection...")
            current_time = time.time()
            wait_time = self._quest_end_cooldown_until - current_time
            if wait_time > 0:
                time.sleep(min(wait_time, 3.0))  # Max 3s d'attente
            # Supprimer le cooldown après attente
            delattr(self, '_quest_end_cooldown_until')
            logger.debug(f"Cooldown terminé")

        # RELOAD SAVE STATE
        # Check if memory exists (not use_memory flag, but actual self.memory instance)
        if self.auto_reload_save_state and self.memory is not None:
            logger.debug("Checking game state before reset...")
            try:
                # Read game state directly and extract values in one go
                game_state = self.memory.read_game_state()
                current_map = game_state.get('current_map')
                death_count = game_state.get('death_count', 0) or 0
                quest_time = game_state.get('quest_time', 5400)

                # Need reload if:
                # - On reward screen (MAP=45)
                # - Or 3 deaths
                # - Or time expired
                needs_reload = (
                        current_map == 45 or
                        death_count >= 3 or
                        (quest_time is not None and quest_time <= 1)
                )

                if needs_reload:
                    logger.debug(f"→ Reload nécessaire (MAP={current_map}, deaths={death_count}, time={quest_time})")
                    reload_success = self._reload_save_state()

                    if not reload_success:
                        raise RuntimeError("Échec reload save state dans reset()")
                else:
                    logger.debug(f"État OK (MAP={current_map}, deaths={death_count}, time={quest_time})")

            except Exception as reload_check_error:
                logger.error(f"Erreur vérification: {reload_check_error}")

        # Reset du contrôleur
        if self.use_controller and self.controller:
            self.controller.reset_all()

        # Pour éviter corruption si le reset survient après une fin de quête
        self.prev_raw_memory = None

        # Reset du reward calculator
        # Si reward_calc existe, forcer nettoyage complet
        if self.reward_calc:
            self.reward_calc.reset()

            # Pause création cubes pendant le reset
            if hasattr(self.reward_calc, 'exploration_tracker'):
                self.reward_calc.exploration_tracker.pause_creation(duration=1.5)
                logger.debug("Création cubes PAUSÉE (reset épisode)")

            # Double sécurité : réinitialiser tous les états internes
            self.reward_calc.prev_hp = None
            self.reward_calc.prev_stamina = None
            self.reward_calc.prev_damage_flag = None
            self.reward_calc.prev_position = None
            self.reward_calc.prev_zone = None
            self.reward_calc.prev_orientation = None
            self.reward_calc.prev_sharpness = None
            self.reward_calc.prev_oxygen = None
            self.reward_calc.prev_death_count = 0
            self.reward_calc.last_damage_time = 0.0
            logger.debug("États internes RewardCalculator nettoyés (sécurité)")

        logger.debug("prev_raw_memory nettoyé")

        # Attendre que le jeu soit prêt
        max_attempts = 3
        current_map = None

        #Vérifier que memory est disponible
        if self.memory is None:
            logger.warning("Memory non disponible - impossible de vérifier CURRENT_MAP")
            current_map = 0  # Assumer état valide
        else:
            try:
                for attempt in range(max_attempts):
                    current_map = self.memory.read_value('CURRENT_MAP')

                    # Vérifier que la valeur est valide
                    if current_map is None:
                        logger.warning(f"Tentative {attempt + 1}/{max_attempts} : Lecture CURRENT_MAP échouée")
                        time.sleep(1.0)
                        continue

                    if current_map != 45:
                        logger.debug(f"CURRENT_MAP = {current_map} (hors écran de fin)")
                        break
                    else:
                        logger.warning(f"Tentative {attempt + 1}/{max_attempts} : Détecté sur écran de fin (MAP=45)")

                        # SI ON EST SUR L'ÉCRAN DE FIN ET QU'ON A LE AUTO-RELOAD
                        if self.auto_reload_save_state and attempt < max_attempts - 1:
                            logger.warning(f"Tentative de reload automatique...")
                            try:
                                reload_success = self._reload_save_state()
                                if reload_success:
                                    # Attendre un peu plus après le reload
                                    time.sleep(1.0)
                                    # Revérifier
                                    current_map = self.memory.read_value('CURRENT_MAP')
                                    if current_map != 45:
                                        logger.warning(f"Reload réussi (MAP = {current_map})")
                                        break
                            except Exception as reload_in_reset_error:
                                logger.error(f"Reload échoué: {reload_in_reset_error}")

                        # Attendre avant prochaine tentative
                        time.sleep(1.0)

            except Exception as map_check_error:
                logger.error(f"Erreur lors de la vérification CURRENT_MAP: {map_check_error}")
                current_map = None

        # Vérifier le résultat final
        if current_map == 45:
            logger.error(f"🚨 CRITIQUE: Toujours sur écran de fin après {max_attempts} tentatives")
            logger.error(f"💡 SOLUTIONS:")
            logger.error(f"   1. Appuie MANUELLEMENT sur F5 dans Dolphin")
            logger.error(f"   2. 🎮 Ou relance une quête manuellement")
            logger.error(f"   3. 💾 Ou vérifie que la save state 5 existe EN QUÊTE")
            logger.error(f" L'entraînement va continuer mais les données seront invalides...")
        elif current_map is None:
            logger.error(f"Impossible de lire CURRENT_MAP après {max_attempts} tentatives")
            logger.error(f"💡 SOLUTIONS:")
            logger.error(f"   1. Vérifie que Dolphin est lancé EN ADMIN")
            logger.error(f"   2. Vérifie que Monster Hunter Tri est chargé")
            logger.error(f"   3. Assure-toi d'être EN JEU (pas dans un menu)")

        # Reset du frame buffer
        if self.use_vision and self.preprocessor is not None:
            self.preprocessor.reset_stack()

        # Reset stats épisode
        self.episode_start_time = time.time()
        self.episode_steps = 0
        self.total_reward = 0
        self.prev_raw_memory = None

        # Démarrer thread de capture
        if self.use_vision and not self._capture_running:
            self._capture_running = True
            self._capture_thread = threading.Thread(
                target=self._async_capture_loop,
                daemon=True,
                name="FrameCaptureThread"
            )
            self._capture_thread.start()
            logger.info("Thread de capture activé")
        elif not self.use_vision:
            logger.debug("Thread de capture non demarre (use_vision=False)")

        # Attendre première frame
        time.sleep(0.1)

        # Flag fin d'episode en cours
        self._episode_ending = False

        # NOTE: episode_count est incremente dans le callback, pas ici
        # Cela evite un double comptage (reset + done)

        # Obtenir l'état initial
        observation = self._get_observation()

        if observation is None:
            logger.warning("Observation invalide - utilisation d'un état dummy")
            observation = self._get_dummy_observation()

        # DEBUG : Afficher les clés retournées
        if isinstance(observation, dict):
            # Vérifier que 'observation_space' est bien un Dict
            if isinstance(self.observation_space, spaces.Dict):
                # Vérifier cohérence
                missing_keys = set(self.observation_space.spaces.keys()) - set(observation.keys())
                if missing_keys:
                    logger.debug(f"ERREUR: Clés manquantes dans observation: {missing_keys}")
            else:
                logger.debug(f"observation_space n'est pas un Dict (type: {type(self.observation_space)})")

        reset_info = self._get_info()
        self.current_state = observation

        # Nettoyer info
        reset_info = self._sanitize_info(reset_info)

        logger.info("Reset terminé")
        return observation, reset_info

    def step(self, action):
        """
        Exécute une action dans l'environnement
        """
        # Incrémenter les deux compteurs
        self.episode_steps += 1
        self.total_steps += 1

        # ===================================================================
        # PRIORITÉ 1 : DÉTECTER FIN DE QUÊTE AVANT ACTION
        # ===================================================================
        # Vérifier si l'épisode n'est pas déjà en train de se terminer
        if hasattr(self, '_episode_ending') and self._episode_ending:
            logger.warning("Épisode déjà en cours de fin - step() ignoré")
            observation = self.current_state or self._get_dummy_observation()
            return observation, 0.0, True, False, {'episode_already_ending': True}

        # Vérifier si on est sur l'écran de fin (protection légère)
        if self.use_memory and self.memory:
            try:
                pre_action_state = self.memory.read_game_state()
                current_map = pre_action_state.get('current_map')

                if current_map == 45:
                    logger.warning(f"🏁 ÉCRAN DE FIN DÉTECTÉ avant action")
                    self._episode_ending = True
                    observation = self.current_state or self._get_dummy_observation()
                    return observation, 0.0, True, False, {
                        'quest_ended_before_action': True,
                        'current_map': 45
                    }
            except (AttributeError, KeyError, RuntimeError):
                # Erreur lecture mémoire ou state invalide → continuer normalement
                pass  # Silencieux

        # ===================================================================
        # PRIORITÉ 2 : DÉTECTION FIN DE QUÊTE (cas où reload a échoué)
        # ===================================================================
        # Cette protection ne s'active QUE si le reload n'a pas fonctionné
        quest_ended_early = False

        if self.use_memory and self.memory:
            try:
                current_map = self.memory.read_value('CURRENT_MAP')

                if current_map == 45:
                    logger.warning(f"ÉCRAN DE FIN DÉTECTÉ MALGRÉ PROTECTION RELOAD!")
                    logger.warning(f"→ Le reload F5 n'a probablement pas fonctionné")
                    logger.warning(f"💡 Vérifie que Dolphin a le focus et que la save state 5 existe")
                    quest_ended_early = True

            except Exception as check_map_error:
                logger.error(f"Erreur vérification CURRENT_MAP: {check_map_error}")

        # SI FIN DÉTECTÉE : RETURN IMMÉDIAT
        if quest_ended_early:
            # Nettoyer
            self.prev_raw_memory = None
            logger.debug(f"prev_raw_memory nettoyé (early exit)")

            # Utiliser l'observation précédente
            observation = self.current_state
            if observation is None:
                observation = self._get_dummy_observation()

            # Accumuler stats
            self.episode_reward += 0.0
            self.episode_length += 1

            # noinspection PyDictCreation
            step_info = {
                'quest_ended_screen': True,
                'episode_num': int(self.episode_count),
                'episode_steps': int(self.episode_steps),
                'total_steps': int(self.total_steps),
                'current_map': 45,
                'forced_reset_reason': 'reward_screen_fallback',
            }

            # Format SB3
            step_info['episode'] = { # type: ignore[assignment]
                'r': float(self.episode_reward),
                'l': int(self.episode_length),
                't': float(time.time() - self.episode_start_time)
            }

            step_info = self._sanitize_info(step_info)

            # RETURN IMMÉDIAT
            return observation, 0.0, True, False, step_info

        # ===================================================================
        # EXÉCUTER L'ACTION
        # ===================================================================
        if self.controller is not None:
            try:
                self.controller.execute_action(action, frames=10)
            except Exception as controller_error:
                logger.error(f"Erreur exécution action {action}: {controller_error}")

        # ===================================================================
        # OBSERVATION (depuis queue asynchrone)
        # ===================================================================
        observation = self._get_observation()

        # Affichage temps réel (séparé selon le mode)
        if isinstance(observation, dict) and (self.rt_vision or self.rt_minimap):
            if self.rt_minimap and self.rt_vision:
                # Mode minimap = layout complet
                self._display_rt_minimap_debug(observation)
            elif self.rt_vision:
                # Mode vision seule = frame haute qualité
                self._display_rt_vision(observation)

        # ===================================================================
        # REWARD
        # ===================================================================
        # PROTECTION : Reset prev_raw_memory on first step of episode
        if self.reward_calc and self.memory:
            # On first step of new episode, ensure prev_raw_memory is clean
            if self.episode_steps == 1:
                self.prev_raw_memory = None
                logger.debug("First episode step - prev_raw_memory cleaned to prevent stale data")

        reward, step_info = self._calculate_reward(action)

        # ===================================================================
        # Check quest end (3 deaths or timeout)
        # ===================================================================
        death_count = step_info.get('death_count', 0) or 0
        quest_time = step_info.get('quest_time', 5400)

        # Episode termination conditions
        episode_should_end = False
        end_reason = None

        if death_count >= 3:
            episode_should_end = True
            end_reason = 'three_deaths'
            logger.info(f"💀💀💀 3 DEATHS DETECTED (count={death_count})")

        elif quest_time is not None and quest_time <= 1:
            episode_should_end = True
            end_reason = 'time_expired'
            logger.info(f"⏱️ TIME EXPIRED (≤1s)")

        # ===================================================================
        # SI FIN DÉTECTÉE : Marquer et terminer
        # ===================================================================
        if episode_should_end:
            self._episode_ending = True
            logger.debug(f"→ Fin d'épisode (raison: {end_reason})")
            logger.debug(f"→ reset() sera appelé par SB3")

            # Accumuler stats
            self.episode_reward += reward
            self.episode_length += 1

            # Infos de fin
            step_info['episode'] = {
                'r': float(self.episode_reward),
                'l': int(self.episode_length),
                't': float(time.time() - self.episode_start_time)
            }
            step_info['end_reason'] = end_reason
            step_info['forced_termination'] = True

            step_info = self._sanitize_info(step_info)

            # RETURN avec terminated=True
            return observation, reward, True, False, step_info

        # ===================================================================
        # CAS NORMAL : Continuer épisode
        # ===================================================================
        # Ajouter les compteurs dans step_info
        step_info['episode_num'] = int(self.episode_count)
        step_info['episode_steps'] = int(self.episode_steps)
        step_info['total_steps'] = int(self.total_steps)
        step_info['total_reward'] = self.total_reward

        # Accumuler les stats d'épisode
        self.episode_reward += reward
        self.episode_length += 1
        self.total_reward += reward

        # Vérifier fin épisode
        terminated = self._check_terminated(observation)
        truncated = self._check_truncated()

        # Si épisode terminé, ajouter format SB3
        if terminated or truncated:
            step_info['episode'] = {
                'r': float(self.episode_reward),
                'l': int(self.episode_length),
                't': float(time.time() - self.episode_start_time)
            }

        # Sauvegarder l'etat
        self.current_state = observation

        # Nettoyer info
        step_info = self._sanitize_info(step_info)

        return observation, reward, terminated, truncated, step_info

    def _get_observation(self):
        """
        Construit l'observation actuelle
        """
        observation = {}

        # ===================================================================
        # VISION : Récupérer depuis queue (non-bloquant)
        # ===================================================================
        if self.use_vision:
            try:
                # PROTECTION : Vérifier que la queue existe
                if not hasattr(self, '_obs_queue'):
                    logger.error("obs_queue non initialisee (thread capture non lance?)")
                    return self._get_dummy_observation()

                # Frame déjà capturée et preprocessed par thread !
                visual_state = self._obs_queue.get(timeout=0.1)
                self._frames_consumed += 1

                # ===================================================================
                # Si memory activée : ajouter memory + exploration map
                # ===================================================================
                # Si memory DISPONIBLE (pas forcément use_memory) : ajouter exploration map
                # Même si use_memory=False, on peut avoir la minimap
                if self.memory is not None and self.state_fusion is not None:
                    # Non-bloquant !
                    raw_memory = self.memory.get_latest_state()  # <0.1ms

                    # Si None (rare), utiliser dummy
                    if raw_memory is None:
                        raw_memory = self._get_dummy_memory_state()

                    # Créer memory_vector UNE SEULE FOIS
                    memory_vector = self._create_enhanced_memory_vector(raw_memory)

                    # Si use_memory=True, ajouter le vecteur à l'observation
                    if self.use_memory:
                        observation['memory'] = memory_vector

                    # Créer exploration map (avec cache - 5ms)
                    px = raw_memory.get('player_x', 0.0)
                    py = raw_memory.get('player_y', 0.0)
                    pz = raw_memory.get('player_z', 0.0)
                    zone = raw_memory.get('current_zone', 0) or 0

                    exploration_map = self.state_fusion.create_exploration_map_with_channels(
                        (px, py, pz), zone
                    )

                    # Ajouter à l'observation selon la config
                    observation['visual'] = visual_state
                    observation['exploration_map'] = exploration_map

                    # N'ajouter memory que si use_memory=True
                    if self.use_memory:
                        observation['memory'] = memory_vector

                # ===================================================================
                # Vision seule : juste visual
                # ===================================================================
                else:
                    observation['visual'] = visual_state

            except queue.Empty:
                # Fallback si queue vide
                logger.warning("Queue vide - frame manquée")
                return self._get_dummy_observation()

        # ===================================================================
        # MEMORY SEULE (cas rare, mais supporté)
        # ===================================================================
        elif self.use_memory and self.memory:
            logger.debug("Mode memory seule (pas de vision)")
            try:
                raw_memory = self.memory.read_game_state()
                memory_vector = self._create_enhanced_memory_vector(raw_memory)
                observation['memory'] = memory_vector

                logger.debug(f"Observation memory seule creee: {observation.keys()}")

            except Exception as read_memory_when_memory_only_error:
                logger.error(f"Erreur lecture memoire: {read_memory_when_memory_only_error}")
                import traceback
                traceback.print_exc()
                return self._get_dummy_observation()

        # ===================================================================
        # VÉRIFICATIONS (tous les 1000 steps)
        # ===================================================================
        if self.total_steps % 1000 == 0:
            logger.debug(f"Observation créée avec clés: {list(observation.keys())}")

            # Sauvegarder visualisation (seulement si vision activée)
            if self.use_vision:
                try:
                    self._save_observation_visualization(observation)
                except KeyError as viz_key_error:
                    logger.warning(f"Visualisation ignoree (cle manquante): {viz_key_error}")
            else:
                logger.debug("Visualisation ignoree (use_vision=False)")

        # ===================================================================
        # VÉRIFICATION COHÉRENCE
        # ===================================================================
        if isinstance(self.observation_space, spaces.Dict):
            expected_keys = set(self.observation_space.spaces.keys())
            actual_keys = set(observation.keys())

            # Clés manquantes
            missing_keys = expected_keys - actual_keys
            if missing_keys:
                logger.error(f"Clés manquantes: {missing_keys}")
                for key in missing_keys:
                    space = self.observation_space.spaces[key]
                    observation[key] = np.zeros(space.shape, dtype=space.dtype)

            # Clés en trop
            extra_keys = actual_keys - expected_keys
            if extra_keys:
                logger.warning(f"Clés en trop: {extra_keys}")
                for key in extra_keys:
                    del observation[key]

            # Vérifier shapes
            for key in expected_keys:
                if key not in observation:
                    continue

                obs_value = observation[key]
                expected_space = self.observation_space.spaces[key]

                if not isinstance(obs_value, np.ndarray):
                    logger.error(f"observation['{key}'] n'est pas un ndarray")
                    observation[key] = np.zeros(expected_space.shape, dtype=expected_space.dtype)
                    continue

                if obs_value.shape != expected_space.shape:
                    logger.error(f"Shape incorrecte pour '{key}': {obs_value.shape} vs {expected_space.shape}")
                    observation[key] = np.zeros(expected_space.shape, dtype=expected_space.dtype)

                if np.any(np.isnan(obs_value)) or np.any(np.isinf(obs_value)):
                    logger.error(f"NaN/Inf détecté dans '{key}'")
                    observation[key] = np.nan_to_num(obs_value, nan=0.0, posinf=1.0, neginf=-1.0)

        # Vérification périodique
        if self.total_steps % 1000 == 0:
            logger.debug(f"Observation créée avec clés: {list(observation.keys())}")
            self._save_observation_visualization(observation)

        return observation

    def _save_observation_visualization(self, observation: Dict):
        """
        Sauvegarde une visualisation de ce que voit l'IA

        Crée une image composite montrant :
        - Visual (première frame du stack)
        - Exploration map (3 premiers channels)
        - Stats mémoire importantes

        Sauvegarde : ./debug/ai_vision_step_{total_steps}.png
        """
        try:
            # VALIDATION : Vérifier que les clés nécessaires existent
            required_keys = ['visual', 'memory', 'exploration_map']
            missing_keys = [k for k in required_keys if k not in observation]

            if missing_keys:
                logger.warning(f"Visualisation impossible, cles manquantes: {missing_keys}")
                return

            # Créer figure avec subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'IA Vision - Step {self.total_steps}', fontsize=16, fontweight='bold')

            # ===== VISUAL (frame principale) =====
            visual = observation['visual']

            # Si grayscale : 1 channel par frame, sinon 3 (RGB)
            if self.preprocessor.grayscale:
                # Dernière frame = dernier channel
                last_frame = visual[:, :, -1]
                # Convertir en RGB pour affichage (répéter 3 fois)
                last_frame = np.stack([last_frame] * 3, axis=-1)
            else:
                # Dernière frame = 3 derniers channels
                last_frame = visual[:, :, -3:]

            # Vérifier que la frame est dans la bonne plage [0, 1]
            if last_frame.max() > 1.0:
                logger.warning(f" Frame hors range : max={last_frame.max()}")
                last_frame = np.clip(last_frame, 0.0, 1.0)

            axes[0, 0].imshow(last_frame)
            axes[0, 0].set_title(f'Visual - Frame actuelle ({visual.shape[0]}x{visual.shape[1]})')
            axes[0, 0].axis('off')

            # ===== EXPLORATION MAP - Channel 0 (visites) =====
            exploration_map = observation['exploration_map']
            ch0 = exploration_map[:, :, 0]
            ch0_min, ch0_max = ch0.min(), ch0.max()
            im0 = axes[0, 1].imshow(ch0, cmap='viridis', vmin=ch0_min, vmax=ch0_max)  # Adaptatif
            axes[0, 1].set_title(f'Ch0: Visites [{ch0_min:.2f}, {ch0_max:.2f}]')  # Afficher plage
            axes[0, 1].axis('off')
            plt.colorbar(im0, ax=axes[0, 1], fraction=0.046, pad=0.04)  # Colorbar

            # ===== EXPLORATION MAP - Channel 1 (position joueur) =====
            ch1 = exploration_map[:, :, 1]
            ch1_min, ch1_max = ch1.min(), ch1.max()
            im1 = axes[0, 2].imshow(ch1, cmap='hot', vmin=ch1_min, vmax=ch1_max)  # Adaptatif
            axes[0, 2].set_title(f'Ch1: Position [{ch1_min:.2f}, {ch1_max:.2f}]')
            axes[0, 2].axis('off')
            plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

            # ===== EXPLORATION MAP - Channel 2 (cubes récents) =====
            ch2 = exploration_map[:, :, 2]
            ch2_min, ch2_max = ch2.min(), ch2.max()
            im2 = axes[1, 0].imshow(ch2, cmap='Blues', vmin=ch2_min, vmax=ch2_max)  # Adaptatif
            axes[1, 0].set_title(f'Ch2: Récents [{ch2_min:.2f}, {ch2_max:.2f}]')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # ===== EXPLORATION MAP - Channel 3 (marqueurs) =====
            ch3 = exploration_map[:, :, 3]
            ch3_min, ch3_max = ch3.min(), ch3.max()
            im3 = axes[1, 1].imshow(ch3, cmap='RdYlGn_r', vmin=ch3_min, vmax=ch3_max)  # Adaptatif
            axes[1, 1].set_title(f'Ch3: Marqueurs [{ch3_min:.2f}, {ch3_max:.2f}]')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # Stats textuelles en dessous des images
            stats_text = f"""EXPLORATION MAP STATS (15x15x4)

            Channel 0 (Visites):
              Min: {ch0_min:.3f} | Max: {ch0_max:.3f}
              Non-zeros: {np.count_nonzero(ch0)} / 225

            Channel 1 (Position):
              Min: {ch1_min:.3f} | Max: {ch1_max:.3f}

            Channel 2 (Récents):
              Non-zeros: {np.count_nonzero(ch2)} / 225

            Channel 3 (Marqueurs):
              Min: {ch3_min:.3f} | Max: {ch3_max:.3f}
              Non-zeros: {np.count_nonzero(ch3)} / 225
            """

            # Ajouter texte en bas de la figure
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=8,
                     fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # ===== MEMORY STATS (texte) =====
            memory = observation['memory']
            stats_text = f"""MEMORY VECTOR (67 features)

    HP: {memory[0]:.1f}/100
    Stamina: {memory[2]:.1f}/100
    Position: ({memory[3]:.1f}, {memory[4]:.1f}, {memory[5]:.1f})
    Orientation: {memory[6]:.1f}°
    Zone: {int(memory[7])}
    Deaths: {int(memory[10])}
    Quest Time: {int(memory[61])}s
    Monster Count: {int(memory[63])}
    Sharpness: {int(memory[65])}
    In Menu: {'OUI' if memory[66] > 0.5 else 'NON'}
    Item Selected: {int(memory[67])}   

    Inventaire:
    Slot 1: ID {int(memory[13])} x{int(memory[14])}
    Slot 2: ID {int(memory[15])} x{int(memory[16])}
    Slot 3: ID {int(memory[17])} x{int(memory[18])}
    Slot 4: ID {int(memory[19])} x{int(memory[20])}
    """

            axes[1, 2].text(0.1, 0.5, stats_text,
                            fontsize=9,
                            verticalalignment='center',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 2].axis('off')

            # Sauvegarder
            import os
            os.makedirs('./debug', exist_ok=True)
            filepath = f'./debug/ai_vision_step_{self.total_steps}.png'
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            logger.debug(f"📸 Visualisation IA sauvegardée : {filepath}")

        except Exception as viz_error:
            logger.error(f"Erreur visualisation : {viz_error}")

    def _display_rt_vision(self, observation: Dict):
        """
        Affiche la frame actuellement capturée en haute qualité
        Mode --rtvision : Vision brute sans processing
        """
        if not CV2_AVAILABLE or cv2 is None:
            return

        if self.rt_window_name is None:
            return

        try:
            # Capturer frame originale (haute qualité)
            if self.frame_capture:
                frame = self.frame_capture.capture_frame()

                # VÉRIFIER QUE LA FRAME N'EST PAS NOIRE
                if frame is not None and frame.size > 0:
                    # Calculer luminosité moyenne
                    mean_brightness = frame.mean()

                    # Si frame trop sombre (< 5), probablement erreur capture
                    if mean_brightness < 5:
                        # Silencieux (pas de log tous les steps)
                        if self.total_steps % 100 == 0:
                            logger.debug(f"Frame capture sombre (luminosité: {mean_brightness:.1f})")
                        return  # Skip cette frame

                # Convertir RGB -> BGR pour OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Ajouter overlay avec infos basiques
                memory = observation.get('memory')
                if memory is not None:
                    info_text = [
                        f"Step: {self.total_steps}",
                        f"Episode: {self.episode_count}",
                        f"HP: {memory[0]:.0f}",
                        f"Zone: {int(memory[7])}",
                    ]

                    y_offset = 30
                    for text in info_text:
                        cv2.putText(frame_bgr, text, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 35

                # Afficher
                cv2.imshow(self.rt_window_name, frame_bgr)
                cv2.waitKey(1)

        except Exception as rt_error:
            if self.total_steps % 1000 == 0:
                logger.error(f"Erreur affichage vision temps réel : {rt_error}")

    def _display_rt_minimap_debug(self, observation: Dict):
        """
        Affiche vision debug complète avec minimap
        Mode --rtminimap : Layout complet avec tous les channels

        Layout :
        ┌─────────────┬─────────────┬─────────────┐
        │   Visual    │  Channel 0  │  Channel 1  │
        ├─────────────┼─────────────┼─────────────┤
        │  Channel 2  │  Channel 3  │    Stats    │
        └─────────────┴─────────────┴─────────────┘
        """
        if not CV2_AVAILABLE or cv2 is None:
            return

        if self.rt_window_name is None:
            return

        try:
            # Extraire données
            visual = observation['visual']
            memory = observation['memory']
            exploration_map = observation['exploration_map']

            # ===== PRÉPARER VISUAL =====
            if self.preprocessor.grayscale:
                last_frame = visual[:, :, -1]
                visual_rgb = (last_frame * 255).astype(np.uint8)
                visual_rgb = cv2.cvtColor(visual_rgb, cv2.COLOR_GRAY2BGR)
            else:
                last_frame = visual[:, :, -3:]
                visual_rgb = (last_frame * 255).astype(np.uint8)

            # Resize pour affichage (84x84 → 300x300)
            visual_display = cv2.resize(visual_rgb, (300, 300), interpolation=cv2.INTER_NEAREST)

            # ===== PRÉPARER CHANNELS EXPLORATION MAP =====
            channels_display = []

            for ch_idx in range(4):
                ch = exploration_map[:, :, ch_idx]

                # Normaliser pour visualisation
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    ch_norm = (ch - ch_min) / (ch_max - ch_min)
                else:
                    ch_norm = ch

                # Appliquer colormap
                ch_8bit = (ch_norm * 255).astype(np.uint8)

                if ch_idx == 0:
                    ch_color = cv2.applyColorMap(ch_8bit, cv2.COLORMAP_VIRIDIS)
                elif ch_idx == 1:
                    ch_color = cv2.applyColorMap(ch_8bit, cv2.COLORMAP_HOT)
                elif ch_idx == 2:
                    ch_color = cv2.applyColorMap(ch_8bit, cv2.COLORMAP_OCEAN)
                else:
                    ch_color = cv2.applyColorMap(ch_8bit, cv2.COLORMAP_JET)

                # Resize (15x15 → 300x300)
                ch_display = cv2.resize(ch_color, (300, 300), interpolation=cv2.INTER_NEAREST)

                # Ajouter label
                cv2.putText(ch_display, f"Ch{ch_idx}: {ch_min:.2f}-{ch_max:.2f}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                channels_display.append(ch_display)

            # ===== CRÉER PANNEAU STATS =====
            stats_panel = np.zeros((300, 300, 3), dtype=np.uint8)

            stats_text = [
                f"Step: {self.total_steps}",
                f"Episode: {self.episode_count}",
                f"",
                f"HP: {memory[0]:.0f}/100",
                f"Stamina: {memory[2]:.0f}/100",
                f"Zone: {int(memory[7])}",
                f"Deaths: {int(memory[10])}",
                f"",
                f"Time: {int(memory[61])}s",
                f"Monsters: {int(memory[63])}",
                f"Sharpness: {int(memory[65])}",
            ]

            y_offset = 30
            for line in stats_text:
                cv2.putText(stats_panel, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25

            # ===== ASSEMBLER LAYOUT =====
            # Ligne 1 : Visual | Ch0 | Ch1
            row1 = np.hstack([visual_display, channels_display[0], channels_display[1]])

            # Ligne 2 : Ch2 | Ch3 | Stats
            row2 = np.hstack([channels_display[2], channels_display[3], stats_panel])

            # Combiner
            final_display = np.vstack([row1, row2])

            # Afficher
            cv2.imshow(self.rt_window_name, final_display)
            cv2.waitKey(1)

        except (AttributeError, KeyError, IndexError, ValueError, cv2.error) as rt_error:
            if self.total_steps % 1000 == 0:
                logger.error(f"Erreur affichage minimap debug : {rt_error}")

    def _verify_observation_integrity(self, observation: Dict) -> bool:
        """
        Vérifie que l'observation est complète et valide

        Checks:
        - Toutes les clés présentes (visual, memory, exploration_map)
        - Shapes correctes
        - Pas de NaN/Inf
        - Valeurs dans les ranges attendus

        Returns:
            True si tout est OK, False sinon
        """
        issues = []

        # ===== CHECK 1 : Clés présentes =====
        expected_keys = {'visual', 'memory', 'exploration_map'}
        actual_keys = set(observation.keys())

        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            if missing:
                issues.append(f"Clés manquantes: {missing}")
            if extra:
                issues.append(f"Clés en trop: {extra}")

        # ===== CHECK 2 : Shapes =====
        if 'visual' in observation:
            visual = observation['visual']
            expected_visual_shape = (84, 84, 12)  # frame_stack=4, RGB=3 --> 12 channels
            if visual.shape != expected_visual_shape:
                issues.append(f"Visual shape incorrect: {visual.shape} (attendu: {expected_visual_shape})")

        if 'memory' in observation:
            memory = observation['memory']
            expected_memory_shape = (70,)
            if memory.shape != expected_memory_shape:
                issues.append(f"Memory shape incorrect: {memory.shape} (attendu: {expected_memory_shape})")

        if 'exploration_map' in observation:
            exp_map = observation['exploration_map']
            expected_map_shape = (15, 15, 4)
            if exp_map.shape != expected_map_shape:
                issues.append(f"Exploration map shape incorrect: {exp_map.shape} (attendu: {expected_map_shape})")

        # ===== CHECK 3 : NaN/Inf =====
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    issues.append(f"NaN détecté dans '{key}'")
                if np.any(np.isinf(value)):
                    issues.append(f"Inf détecté dans '{key}'")

        # ===== CHECK 4 : Ranges =====
        if 'visual' in observation:
            visual = observation['visual']
            if visual.min() < 0.0 or visual.max() > 1.0:
                issues.append(f"Visual hors range [0,1]: [{visual.min():.3f}, {visual.max():.3f}]")

        if 'exploration_map' in observation:
            exp_map = observation['exploration_map']
            if exp_map.min() < -1.0 or exp_map.max() > 1.0:
                issues.append(f"Exploration map hors range [-1,1]: [{exp_map.min():.3f}, {exp_map.max():.3f}]")

        # ===== CHECK 5 : Memory features critiques =====
        if 'memory' in observation:
            memory = observation['memory']

            # HP doit être entre 0 et 100
            hp = memory[0]
            if hp < 0 or hp > 150:
                issues.append(f"HP invalide: {hp} (attendu: 0-150)")

            # Zone doit être entre 0 et 20
            zone = memory[7]
            if zone < 0 or zone > 20:
                issues.append(f"Zone invalide: {zone} (attendu: 0-20)")

        # ===== RÉSULTAT =====
        if issues:
            logger.warning(f"PROBLÈMES DÉTECTÉS dans observation (step {self.total_steps}):")
            for issue in issues:
                logger.warning(f"• {issue}")
            return False

        return True

    def _create_enhanced_memory_vector(self, raw_memory):
        """
        v3 : Vecteur mémoire SÉCURISÉ contre valeurs NaN/Inf

        Version 67 features (utilisée quand use_memory=True mais use_vision=False)
        Utilise safe_float() pour éviter les NaN/Inf

        IDENTIQUE à _create_enhanced_memory_vector_full() de state_fusion.py
        mais sans dépendance à self.memory.reward_calc
        """
        features = []

        # === 13 FEATURES DE BASE (SÉCURISÉES) ===
        hp = safe_float(raw_memory.get('player_hp'), default=50.0, min_val=0.0, max_val=150.0)
        hp_rec = safe_float(raw_memory.get('player_hp_recoverable'), default=0.0, min_val=0.0, max_val=150.0)
        stamina = safe_float(raw_memory.get('player_stamina'), default=50.0, min_val=0.0, max_val=150.0)
        x = safe_float(raw_memory.get('player_x'), default=0.0, min_val=-10000.0, max_val=10000.0)
        y = safe_float(raw_memory.get('player_y'), default=0.0, min_val=-10000.0, max_val=10000.0)
        z = safe_float(raw_memory.get('player_z'), default=0.0, min_val=-10000.0, max_val=10000.0)
        orientation = safe_float(raw_memory.get('player_orientation'), default=0.0, min_val=0.0, max_val=360.0)
        zone = safe_float(raw_memory.get('current_zone'), default=0.0, min_val=0.0, max_val=20.0)
        damage = safe_float(raw_memory.get('damage_last_hit'), default=0.0, min_val=0.0, max_val=10000.0)
        money = safe_float(raw_memory.get('money'), default=0.0, min_val=0.0, max_val=999999.0)
        death_count = safe_float(raw_memory.get('death_count'), default=0.0, min_val=0.0, max_val=10.0)
        stamina_low = 1.0 if raw_memory.get('stamina_low', False) else 0.0
        time_underwater = safe_float(raw_memory.get('time_underwater'), default=0.0, min_val=0.0, max_val=200.0)

        features.extend([
            hp, hp_rec, stamina, x, y, z, orientation, zone, damage, money, death_count, stamina_low, time_underwater
        ])

        # === 44 FEATURES INVENTAIRE (24 SLOTS) ===
        inventory = raw_memory.get('inventory_items', [])

        inventory_dict = {}
        for item in inventory:
            slot = item.get('slot')
            if slot is not None and 1 <= slot <= 24:
                inventory_dict[slot] = item

        for slot_num in range(1, 25):
            if slot_num in inventory_dict:
                item = inventory_dict[slot_num]
                item_id = safe_float(item.get('item_id', 0), default=0.0, min_val=0.0, max_val=746.0)
                quantity = safe_float(item.get('quantity', 0), default=0.0, min_val=0.0, max_val=99.0)
            else:
                item_id = 0.0
                quantity = 0.0

            features.append(item_id)
            features.append(quantity)

        # === 4 FEATURES COMBAT ===

        # Quest time
        quest_time = safe_float(raw_memory.get('quest_time'), default=5400.0, min_val=0.0, max_val=5400.0)
        features.append(quest_time)

        # Attack & Defense reunies
        attack_defense = safe_float(raw_memory.get('attack_defense_value'), default=0.0, min_val=0.0,
                                        max_val=10000.0)
        features.append(attack_defense)

        # Number of monsters
        monster_count = 0
        monsters_present = False

        for i in range(1, 6):
            hp_key = f'smonster{i}_hp'
            monster_hp = raw_memory.get(hp_key)
            if monster_hp is not None and monster_hp > 0:
                monster_count += 1
                monsters_present = True

        features.append(float(monster_count))
        features.append(1.0 if monsters_present else 0.0)

        # 3 FEATURES SUPPLÉMENTAIRES

        # 1. SHARPNESS
        sharpness = safe_float(raw_memory.get('sharpness'), default=150.0, min_val=-10.0, max_val=400.0)
        features.append(sharpness)

        # 2. IN-GAME MENU
        in_menu = raw_memory.get('in_game_menu', False)
        features.append(1.0 if in_menu else 0.0)

        # 3. Item selected
        item_selected = safe_float(raw_memory.get('item_selected'), default=24.0, min_val=0.0, max_val=24.0)
        features.append(item_selected)

        # 4. IN_COMBAT (toujours False ici car pas de reward_calc accessible)
        features.append(0.0)

        # 5. IN_MONSTER_ZONE (toujours False ici car pas de reward_calc accessible)
        features.append(0.0)

        # NOTE pôur 4/5 : Dans mh_env.py, on ne peut pas accéder à reward_calc,
        # donc on met toujours 0.0. Ce n'est pas grave car cette méthode n'est
        # utilisée que dans des cas spéciaux (memory seule).

        # Vérification
        assert len(features) == 70, f"Expected 70 features, got {len(features)}"

        # VÉRIFICATION FINALE NaN/Inf
        features_array = np.array(features, dtype=np.float32)

        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            logger.warning("NaN/Inf détecté dans mh_env._create_enhanced_memory_vector !")
            logger.warning(f"Features problématiques: {features_array}")

            # Remplacer (sécurité ultime)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1000.0, neginf=-1000.0)
            logger.warning(f"Features corrigées: {features_array}")

        # Log périodique pour debug
        if self.total_steps % 1000 == 0:  # Tous les 1000 steps
            logger.debug(f"Memory vector sample (step {self.total_steps}):")
            logger.debug(f"   HP: {features_array[0]:.2f}, Stamina: {features_array[2]:.2f}")
            logger.debug(f"   Position: ({features_array[3]:.2f}, {features_array[4]:.2f}, {features_array[5]:.2f})")
            logger.debug(f"   Zone: {features_array[7]:.0f}")
            logger.debug(f"   Sharpness: {features_array[65]:.0f}")
            logger.debug(f"   In Menu: {features_array[66]:.0f}")
            logger.debug(f"   Item selected: {features_array[67]:.0f}")

        return features_array

    def _calculate_reward(self, action):
        """
            Interface de calcul de reward avec protections

            RÔLE :
            - Vérifier fin de quête (CURRENT_MAP = 45) avant calcul
            - Appeler RewardCalculator.calculate() pour calculs complexes
            - Gérer prev_raw_memory pour les deltas (HP, stamina)
            - Enrichir reward_info avec stats pour GUI
            - Gérer les erreurs de lecture mémoire

            Args:
                action: Action exécutée (0-18)

            Returns:
                tuple: (reward: float, reward_info: dict)

            Notes:
                - Les calculs réels sont dans RewardCalculator (reward_calculator.py)
                - Cette méthode sert d'interface + protection + enrichissement
            """
        reward_info: Dict[str, Union[int, float, bool, str, List, Dict]] = {
            'episode_num': self.episode_count,
            'episode_steps': self.episode_steps,
            'total_steps': self.total_steps,
        }

        if self.use_memory and self.memory:
            try:
                raw_memory = self.memory.read_game_state()

                # Détecter si dégâts pris avant d'appeler calculate()
                took_damage = False
                if self.prev_raw_memory is not None:
                    prev_hp = self.prev_raw_memory.get('player_hp', 0) or 0
                    current_hp = raw_memory.get('player_hp', 0) or 0

                    # Si delta HP > 0 et < 100 (pour éviter les resets)
                    hp_delta = prev_hp - current_hp
                    if 0 < hp_delta < 100:
                        took_damage = True
                        logger.info(f"🩸 Dégâts détectés: {hp_delta} HP")

                # Calcul reward
                if self.reward_calc:
                    reward = self.reward_calc.calculate(
                        self.prev_raw_memory,
                        raw_memory,
                        action,
                        reward_info,
                        took_damage=took_damage,
                    )

                    calc_stats = self.reward_calc.get_stats()
                    reward_info.update(calc_stats)
                else:
                    reward = 0.01

                # reset des reward
                self.prev_raw_memory = raw_memory

                # Infos pour GUI
                reward_info['hp'] = raw_memory.get('player_hp')
                reward_info['stamina'] = raw_memory.get('player_stamina')
                reward_info['damage_last_hit'] = raw_memory.get('damage_last_hit')
                reward_info['death_count'] = raw_memory.get('death_count')
                reward_info['current_zone'] = raw_memory.get('current_zone')
                reward_info['player_x'] = raw_memory.get('player_x')
                reward_info['player_y'] = raw_memory.get('player_y')
                reward_info['player_z'] = raw_memory.get('player_z')
                reward_info['orientation'] = raw_memory.get('player_orientation')
                reward_info['money'] = raw_memory.get('money')
                reward_info['quest_time'] = raw_memory.get('quest_time')
                reward_info['sharpness'] = raw_memory.get('sharpness')
                reward_info['inventory'] = self.memory.read_inventory()
                reward_info['in_game_menu'] = raw_memory.get('in_game_menu', False)
                reward_info['game_menu_open_count'] = getattr(self.reward_calc, 'game_menu_open_count', 0) if self.reward_calc else 0
                reward_info['item_selected'] = raw_memory.get('item_selected', 24)
                if self.reward_calc:
                    reward_info['in_combat'] = getattr(self.reward_calc, 'prev_in_combat', False)
                    reward_info['in_monster_zone'] = getattr(self.reward_calc, 'zone_has_monsters', False)
                    reward_info['monsters_present'] = getattr(self.reward_calc, 'zone_has_monsters', False)
                    reward_info['monster_count'] = getattr(self.reward_calc, '_last_monster_count', 0)

            except Exception as reward_calcul_error:
                logger.error(f"Erreur calcul reward: {reward_calcul_error}")
                reward = 0.0

            return reward, reward_info

        return 0.0, reward_info

    @staticmethod
    def _should_reload_save_state(current_state: dict) -> bool:
        """
        Vérifie si on doit recharger la save state
        """
        quest_time = current_state.get('quest_time')
        death_count = current_state.get('death_count', 0) or 0
        current_map = current_state.get('current_map')

        # ===================================================================
        # Vérifier CURRENT_MAP
        # ===================================================================
        if current_map == 45:
            logger.info(f"ÉCRAN DE FIN DÉTECTÉ (CURRENT_MAP = 45)")
            logger.info(f"→ Reload save state nécessaire")
            return True

        # Vérifier temps
        if quest_time is not None and quest_time <= 1:
            logger.info(f"TEMPS ÉCOULÉ (≤1s) - Quest time: {quest_time}s")
            return True

        # Vérifier morts
        if death_count >= 3:
            logger.info(f"3 MORTS ATTEINTES - Death count: {death_count}")
            return True

        return False

    def _reload_save_state(self):
        """
        Appuie sur F1-F8 pour recharger la save state configurée
        BLOQUE l'entraînement si le reload échoue après 3 tentatives
        """
        if not self.auto_reload_save_state or not PYNPUT_AVAILABLE:
            return False  # Retourne False si pas de reload possible

        # Mapping numéro -> touche
        key_mapping = {
            1: Key.f1, 2: Key.f2, 3: Key.f3, 4: Key.f4,
            5: Key.f5, 6: Key.f6, 7: Key.f7, 8: Key.f8
        }

        save_state_key = key_mapping.get(self.save_state_slot, Key.f5)

        max_reload_attempt = 3
        reload_wait_time = 3.0

        for attempt in range(1, max_reload_attempt + 1):
            logger.info(
                f"Rechargement save state {self.save_state_slot} (F{self.save_state_slot}) - Tentative {attempt}/{max_reload_attempt}...")

            try:
                # Appuyer sur F5
                self.keyboard.press(save_state_key)
                time.sleep(0.1)
                self.keyboard.release(save_state_key)

                self.save_state_reload_count += 1

                logger.info(f"Attente chargement ({reload_wait_time}s)...")
                time.sleep(reload_wait_time)

                # Vérifier que le chargement a fonctionné
                if self.use_memory and self.memory:
                    try:
                        verification_state = self.memory.read_game_state()
                        current_map = verification_state.get('current_map')

                        if current_map == 45:
                            logger.error(f"Échec: Toujours sur écran de fin (MAP=45)")

                            if attempt < max_reload_attempt:
                                logger.error(f"🔁 Nouvelle tentative dans 2s...")
                                time.sleep(2.0)
                                continue  # Réessayer
                            else:
                                # DERNIER ESSAI ÉCHOUÉ = BLOQUER
                                logger.error(
                                    f"🚨 ÉCHEC CRITIQUE: Impossible de charger la save state après {max_reload_attempt} tentatives")
                                logger.error(f"⛔ ENTRAÎNEMENT BLOQUÉ - Actions requises:")
                                logger.error(f"   1. VÉRIFIE QUE DOLPHIN A LE FOCUS")
                                logger.error(f"   2. 🎮 Lance Dolphin et charge Monster Hunter Tri")
                                logger.error(f"   3. 💾 Crée/vérifie la save state 5 EN QUÊTE (appuie F5 en jeu)")
                                logger.error(f"   4. Relance l'entraînement")
                                logger.error(f"La save state 5 doit être DANS UNE QUÊTE, pas au village ou écran de fin")

                                # LEVER UNE EXCEPTION POUR ARRÊTER L'ENTRAÎNEMENT
                                raise RuntimeError(
                                    f"Échec critique: Impossible de charger save state après {max_reload_attempt} tentatives. "
                                    f"Vérifie que Dolphin a le focus et que la save state 5 existe EN QUÊTE."
                                )

                        elif current_map is None:
                            logger.warning(f"Lecture CURRENT_MAP échouée")

                            if attempt < max_reload_attempt:
                                logger.warning(f"🔁 Nouvelle tentative...")
                                time.sleep(2.0)
                                continue
                            else:
                                raise RuntimeError("Impossible de lire CURRENT_MAP après reload")

                        else:
                            # SUCCÈS !
                            logger.debug(f"Reload confirmé (CURRENT_MAP = {current_map})")
                            logger.debug(f"Reprise de l'entraînement")
                            return True  # Reload réussi

                    except Exception as verify_error:
                        logger.error(f"Erreur vérification: {verify_error}")

                        if attempt < max_reload_attempt:
                            time.sleep(2.0)
                            continue
                        else:
                            raise RuntimeError(f"Erreur vérification reload: {verify_error}")

            except Exception as reload_error:
                logger.error(f"Erreur reload: {reload_error}")

                if attempt < max_reload_attempt:
                    time.sleep(2.0)
                    continue
                else:
                    raise RuntimeError(f"Erreur critique reload F5: {reload_error}")

        # Si on arrive ici, toutes les tentatives ont échoué
        return False

    def _check_terminated(self, _observation):
        """
        Vérifie si l'épisode est terminé (3 morts)
        """
        if self.use_memory and self.memory:
            try:
                current_state = self.memory.read_game_state()
                death_count = current_state.get('death_count', 0) or 0

                # Épisode terminé si 3 morts
                if death_count >= 3:
                    logger.info(f"💀💀💀 3 morts atteintes - Épisode terminé")
                    return True

            except (AttributeError, KeyError, TypeError):
                # Erreurs possibles lors de la lecture de la mémoire
                pass

        return False

    def _check_truncated(self):
        """Vérifie si l'épisode doit être tronqué (timeout)"""
        max_steps = 10000

        if self.episode_steps >= max_steps:
            logger.info(f"⏱️ Timeout après {max_steps} steps")
            return True

        return False

    def _get_info(self):
        """
        Informations supplémentaires
        """
        info_dict: Dict[str, Union[int, float, bool, str, List, Dict]] = {
            'episode_num': self.episode_count,
            'episode_steps': self.episode_steps,
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
        }
        # Note : la clé 'episode' est réservée pour le format SB3 {r, l, t}
        # et n'est ajoutée que lors de la fin d'épisode dans step()

        if self.use_memory and self.memory:
            try:
                state = self.memory.read_game_state()
                info_dict['hp'] = state.get('player_hp')
                info_dict['stamina'] = state.get('player_stamina')
                info_dict['death_count'] = state.get('death_count')
                info_dict['current_zone'] = state.get('current_zone')
                info_dict['player_x'] = state.get('player_x')
                info_dict['player_y'] = state.get('player_y')
                info_dict['player_z'] = state.get('player_z')
                info_dict['orientation'] = state.get('player_orientation')
                info_dict['money'] = state.get('money')
                info_dict['quest_time'] = state.get('quest_time')
                info_dict['sharpness'] = state.get('sharpness')
                info_dict['inventory'] = self.memory.read_inventory()

                if self.reward_calc:
                    calc_stats = self.reward_calc.get_stats()
                    info_dict.update(calc_stats)
            except (AttributeError, KeyError, TypeError, ValueError):
                # Erreurs possibles lors de la lecture de la mémoire ou des stats
                pass

        # Info sur auto-reload
        if self.auto_reload_save_state:
            info_dict['save_state_reload_count'] = self.save_state_reload_count

        return info_dict

    def _get_dummy_observation(self):
        """Crée une observation dummy"""
        if isinstance(self.observation_space, spaces.Dict):
            dummy_obs = {}

            if 'visual' in self.observation_space.spaces:
                visual_shape = self.observation_space['visual'].shape
                dummy_obs['visual'] = np.zeros(visual_shape, dtype=np.float32)

            if 'memory' in self.observation_space.spaces:
                memory_shape = self.observation_space['memory'].shape
                dummy_obs['memory'] = np.zeros(memory_shape, dtype=np.float32)

            if 'exploration_map' in self.observation_space.spaces:
                map_shape = self.observation_space['exploration_map'].shape
                dummy_obs['exploration_map'] = np.zeros(map_shape, dtype=np.float32)

            # DEBUG
            logger.debug(f"🔍 DEBUG _get_dummy_observation() - Clés créées: {list(dummy_obs.keys())}")

            return dummy_obs
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    @staticmethod
    def _get_dummy_memory_state() -> dict:
        """
        Crée une state mémoire dummy quand lecture échoue

        Returns:
            Dict avec valeurs par défaut sécurisées
        """
        return {
            # Base
            'player_hp': 100.0,
            'player_hp_recoverable': 0.0,
            'player_stamina': 100.0,
            'player_hp_raw': 2516600000,
            'player_stamina_raw': 20000000,
            'player_x': 0.0,
            'player_y': 0.0,
            'player_z': 0.0,
            'player_orientation': 0.0,
            'current_zone': 0,
            'damage_last_hit': 0.0,
            'money': 0,
            'death_count': 0,
            'stamina_low': False,

            # Quête
            'quest_time': 5400,
            'attack_defense_value': 0,
            'sharpness': 150,
            'in_game_menu': False,
            'item_selected': 24,

            # Monstres
            'smonster1_hp': 0,
            'smonster2_hp': 0,
            'smonster3_hp': 0,
            'smonster4_hp': 0,
            'smonster5_hp': 0,
            'lmonster1_hp': 0,

            # Oxygène
            'time_underwater': 0,
            'oxygen_valid': False,

            # Inventaire
            'inventory_items': [],

            # État
            'current_map': 0,
            'quest_ended': False,
            'on_reward_screen': False
        }

    def render(self):
        """Rendu"""
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            if self.use_vision:
                try:
                    return self.frame_capture.capture_frame()
                except (AttributeError, RuntimeError):
                    # Erreurs possibles lors de la capture de frame
                    return None
        return None

    def get_controller(self):
        """
        Retourne le contrôleur de cet environnement
        Utilisé pour la configuration HidHide

        Returns:
            WiiController ou None si non disponible
        """
        if hasattr(self, 'controller'):
            return self.controller
        else:
            logger.warning(f"Instance {self.instance_id} : Contrôleur non initialisé")
            return None

    def get_window_title(self) -> str:
        """
        Retourne le titre de la fenêtre capturée
        Utilisé pour debug/vérification en multi-instance

        Returns:
            str: Titre de la fenêtre ou chaîne vide

        Raises:
        Aucune - retourne une chaîne vide en cas d'erreur
        """
        if self.frame_capture and self.frame_capture.hwnd:
            try:
                import win32gui
                return win32gui.GetWindowText(self.frame_capture.hwnd)
            except (OSError, AttributeError, ValueError) as get_window_title_error:
                # OSError: Handle invalide ou fenêtre fermée
                # AttributeError: Problème avec l'objet hwnd
                # ValueError: Handle avec valeur invalide
                logger.debug(f"Impossible d'obtenir le titre de la fenêtre: {get_window_title_error}")
                return ""
            except ImportError:
                # win32gui non disponible (peu probable ici car import local)
                logger.warning("Module win32gui non disponible")
                return ""
        return ""

    def close(self):
        """
        Nettoyage
        """
        logger.info("🛑 Fermeture de l'environnement...")

        # Arrêter thread de capture
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            logger.info(f"Frames capturées: {self._frames_captured}")
            logger.info(f"Frames utilisées: {self._frames_consumed}")

        # Nettoyer objets GDI
        if self.frame_capture:
            self.frame_capture.close()

        # Reset du controller
        if self.controller:
            try:
                logger.info("🎮 Nettoyage contrôleur...")
                self.controller.cleanup()  # Appel explicite à cleanup()
                self.controller = None  # Libérer la référence
            except Exception as controller_cleanup_error:
                logger.error(f"Erreur nettoyage contrôleur: {controller_cleanup_error}")

        # Recharge de la save state
        if self.auto_reload_save_state:
            logger.info(f"Save states rechargées: {self.save_state_reload_count} fois")

        # Fermer fenêtre vision temps réel
        if self.rt_vision and self.rt_window_name and CV2_AVAILABLE and cv2 is not None:
            try:
                cv2.destroyWindow(self.rt_window_name)
                cv2.waitKey(1)  # Nécessaire pour fermeture propre
            except (cv2.error, AttributeError):
                # Fenêtre déjà fermée ou OpenCV non disponible
                pass

        # Fermer fenêtre minimap
        if self.rt_minimap and self.rt_minimap_fig and MATPLOTLIB_AVAILABLE and plt is not None:
            try:
                plt.close(self.rt_minimap_fig)
            except (AttributeError, RuntimeError):
                # Figure déjà fermée
                pass

        logger.info("L'environnement a été correctement fermé")

    @staticmethod
    def _sanitize_info(info_to_sanitized: dict) -> dict:
        """
        Nettoie le dict info pour éviter les problèmes avec VecNormalize
        Convertit tous les types complexes en types simples (int, float, bool, str)

        RÈGLES DE NETTOYAGE :
        1. Convertir None en valeurs par défaut (0, 0.0, False)
        2. Convertir types numpy en types Python (np.int64 --> int)
        3. Garder reward_breakdown mais nettoyer les None dedans
        4. Supprimer listes/dicts complexes (inventory, exploration_cubes)
        5. S'assurer que episode_steps/total_steps sont bien des int

        Args :
            info_to_sanitized : Dict d'informations potentiellement "sale"

        Returns :
            Dict nettoyé compatible VecNormalize
        """
        sanitized = {}

        # ========================================
        # RÈGLE SPÉCIALE : Clé 'episode' pour SB3
        # ========================================
        """
        SB3 s'attend à ce que 'episode' soit un dict avec {r, l, t}
        Si ce n'est pas le cas, on la supprime pour éviter les erreurs
        TRAITER EN PREMIER pour éviter qu'elle soit écrasée
        """
        episode_dict = None
        if 'episode' in info_to_sanitized:
            episode_value = info_to_sanitized['episode']

            if isinstance(episode_value, dict):
                # Vérifier que le dict contient les bonnes clés
                required_episode_keys = {'r', 'l', 't'}
                if required_episode_keys.issubset(episode_value.keys()):
                    # Format correct, nettoyer les valeurs
                    try:
                        episode_dict = {
                            'r': float(episode_value['r']),
                            'l': int(episode_value['l']),
                            't': float(episode_value['t'])
                        }
                    except (ValueError, TypeError) as episode_conversion_in_info_sanitize_error:
                        logger.error(f"Erreur conversion 'episode': {episode_conversion_in_info_sanitize_error}")
                        episode_dict = None
                else:
                    logger.warning(f"WARNING: 'episode' dict incomplet - clés: {episode_value.keys()}")
            elif isinstance(episode_value, (int, np.integer)):
                # C'est un entier (episode_num) - NE PAS L'UTILISER comme 'episode'
                logger.warning(f"WARNING: 'episode' est un int ({episode_value}) - renommé en 'episode_num'")
                sanitized['episode_num'] = int(episode_value)
            else:
                logger.warning(f"WARNING: 'episode' type invalide ({type(episode_value)}) - ignoré")

        for key, value in info_to_sanitized.items():
            # Ignorer 'episode' car déjà traité
            if key == 'episode':
                continue

            # ========================================
            # RÈGLE 1 : Dicts imbriqués
            # ========================================
            if isinstance(value, dict):
                # Exception : reward_breakdown peut rester dict
                if 'reward' in key.lower() or 'breakdown' in key.lower():
                    # Nettoyer les 'None' à l'intérieur
                    cleaned_dict = {}
                    for k, v in value.items():
                        if v is None:
                            cleaned_dict[k] = 0.0
                        elif isinstance(v, (int, np.integer)):
                            cleaned_dict[k] = float(v)
                        elif isinstance(v, (float, np.floating)):
                            cleaned_dict[k] = float(v)
                        else:
                            cleaned_dict[k] = 0.0
                    sanitized[key] = cleaned_dict
                    continue  # ← IMPORTANT : continue ICI pour passer à la clé suivante
                # Sinon ignorer (exploration_cubes, etc.)
                else:
                    continue  # ← Ignorer les autres dicts

            # ========================================
            # RÈGLE 2 : None → valeur par défaut
            # ========================================
            if value is None:
                # Compteurs --> 0
                if any(word in key.lower() for word in ['count', 'num', 'steps', 'episode']):
                    sanitized[key] = 0
                # Valeurs float (HP, stamina, etc.) --> 0.0
                elif any(word in key.lower() for word in ['hp', 'stamina', 'reward', 'distance', 'orientation']):
                    sanitized[key] = 0.0
                # Booléens --> False
                else:
                    sanitized[key] = False
                continue

            # ========================================
            # RÈGLE 3 : Convertir types
            # ========================================
            try:
                # Entiers (numpy ou Python)
                if isinstance(value, (int, np.integer)):
                    sanitized[key] = int(value)

                # Flottants (numpy ou Python)
                elif isinstance(value, (float, np.floating)):
                    # Vérifier NaN/Inf
                    float_value = float(value)
                    if np.isnan(float_value) or np.isinf(float_value):
                        sanitized[key] = 0.0
                    else:
                        sanitized[key] = float_value

                # Booléens
                elif isinstance(value, (bool, np.bool_)):
                    sanitized[key] = bool(value)

                # Strings
                elif isinstance(value, str):
                    sanitized[key] = str(value)

                # ========================================
                # RÈGLE 4 : Ignorer listes/tuples/arrays (sauf inventaire)
                # ========================================
                elif isinstance(value, (list, tuple, np.ndarray)):
                    # Garder 'inventory' pour le GUI
                    if key == 'inventory':
                        # Vérifier que c'est une liste de dicts valides
                        if isinstance(value, list):
                            # Nettoyer chaque item de l'inventaire
                            cleaned_inventory = []
                            for item in value:
                                if isinstance(item, dict):
                                    cleaned_item = {}
                                    for k, v in item.items():
                                        # Convertir en types Python natifs
                                        if isinstance(v, (int, np.integer)):
                                            cleaned_item[k] = int(v)
                                        elif isinstance(v, (float, np.floating)):
                                            cleaned_item[k] = float(v)
                                        elif isinstance(v, str):
                                            cleaned_item[k] = str(v)
                                        elif v is None:
                                            cleaned_item[k] = None
                                    cleaned_inventory.append(cleaned_item)
                            sanitized[key] = cleaned_inventory
                        continue

                    # Autres listes/arrays : ignorer (exploration_cubes, etc.)
                    continue

                # Type inconnu --> ignorer
                else:
                    continue

            except (ValueError, TypeError, OverflowError) as conversion_error:
                # En cas d'erreur de conversion, ignorer cette clé
                logger.warning(f"Impossible de convertir {key}={value}: {conversion_error}")
                continue

        # ========================================
        # AJOUTER 'episode' À LA FIN (si valide)
        # ========================================
        if episode_dict is not None:
            sanitized['episode'] = episode_dict

        # ========================================
        # VÉRIFICATION FINALE : Clés critiques
        # ========================================
        # S'assurer que les clés essentielles existent
        critical_keys = {
            'episode_num': 0,
            'episode_steps': 0,
            'total_steps': 0,
            'hp': 0.0,
            'stamina': 0.0,
            'death_count': 0,
            'current_zone': 0,
        }

        for critical_key, default_value in critical_keys.items():
            if critical_key not in sanitized:
                sanitized[critical_key] = default_value

        return sanitized

# Test
if __name__ == "__main__":
    print("🧪 TEST ENVIRONNEMENT v2.5.1\n")

    try:
        env = MonsterHunterEnv(
            use_vision=True,
            use_memory=True,
            grayscale=False,
            frame_stack=4,
            use_controller=True,
            use_advanced_rewards=True,
            auto_reload_save_state=True
        )

        print(f"\n📊 Environnement créé")
        print(f"   Actions: {env.action_space}")
        print(f"   Memory features: 67")

        # Reset
        obs, info = env.reset()

        print(f"\nTest réussi !")

        env.close()

    except Exception as e:
        logger.error(f"\nErreur: {e}")
        traceback.print_exc()