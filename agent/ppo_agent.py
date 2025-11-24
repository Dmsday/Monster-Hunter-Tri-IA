"""
Agent PPO avec support Vision + Mémoire
Utilise Stable-Baselines3 avec custom feature extractor
"""

# ============================================================================
# DEEP LEARNING (PYTORCH)
# ============================================================================
import torch                  # Framework Deep Learning principal
                             # - Utilisé pour les tenseurs (torch.Tensor)
                             # - Gestion du device (CPU/GPU)
                             # - Forward/backward pass pour l'entraînement

from torch import nn         # Modules de réseaux de neurones
                             # - nn.Module : Classe de base pour tous les réseaux
                             # - nn.Sequential : Empiler des couches séquentiellement
                             # - nn.Linear : Couches fully-connected (MLP)
                             # - nn.ReLU : Fonction d'activation
                             # - nn.Conv2d : Convolutions 2D pour les CNNs
                             # - nn.Flatten : Aplatir les tenseurs multi-dim en vecteurs

# ============================================================================
# GYMNASIUM (REINFORCEMENT LEARNING - ENVIRONMENT)
# ============================================================================
import gymnasium as gym      # Framework RL moderne (successeur d'OpenAI Gym)
                             # - Définit le standard des environnements RL
                             # - Fournit gym.spaces pour définir action/observation spaces
                             # - Interface reset(), step(), render()

# ============================================================================
# CALCUL NUMÉRIQUE
# ============================================================================
import numpy as np           # Calculs numériques et manipulation d'arrays
                             # - Utilisé pour calculer les dimensions des tenseurs
                             # - Conversion entre numpy arrays et torch tensors
                             # - np.prod() : Produit des éléments d'un array
                             #   (ex: calculer taille après Flatten)

# ============================================================================
# STABLE-BASELINES3 (ALGORITHMES RL)
# ============================================================================
from stable_baselines3 import PPO
# PPO (Proximal Policy Optimization)
# - Algorithme RL state-of-the-art pour l'apprentissage par renforcement
# - On-policy : utilise les données de la politique actuelle
# - Équilibre exploration/exploitation avec clipping
# - Méthodes principales :
#   • .learn(total_timesteps) : Entraîner l'agent
#   • .predict(obs) : Prédire une action
#   • .save(path) / .load(path) : Sauvegarder/charger le modèle

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# BaseFeaturesExtractor : Classe de base pour extracteurs de features personnalisés
# - À hériter pour créer un CNN custom
# - Remplace l'extracteur par défaut de SB3
# - Permet de traiter des observations complexes (Dict avec vision + mémoire)
# - Méthodes à implémenter :
#   • __init__() : Définir l'architecture (CNN, MLP, fusion)
#   • forward() : Extraire les features depuis les observations
# - Doit retourner un vecteur de features de dimension fixe

# ============================================================================
# TYPE HINTS (PYTHON 3.5+)
# ============================================================================
from typing import Dict      # Annotations de types pour les dictionnaires
                             # - Améliore la lisibilité du code
                             # - Permet aux IDE de détecter les erreurs
                             # - Exemple : Dict[str, torch.Tensor]
                             #   = dict avec clés string et valeurs tenseurs
                             # - Utilisé pour les observations multiples :
                             #   {'visual': tensor, 'memory': tensor, 'exploration_map': tensor}

# ============================================================================
# MODULES PERSONNALISÉS
# ============================================================================
from vision.feature_extractor import NatureCNN
# NatureCNN : Architecture CNN inspirée de Nature DQN (Mnih et al. 2015)
# - 3 couches de convolutions (8x8, 4x4, 3x3)
# - Utilisé pour extraire features des frames du jeu (vision principale)
# - Input : Frames stackées (84x84x4 en grayscale, ou 84x84x12 en RGB)
# - Output : Vecteur de features (256 dimensions par défaut)
# - Entraînement end-to-end avec PPO

from utils.module_logger import get_module_logger
logger = get_module_logger('ppo_agent')
# Logs par niveaux


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor pour observations Dict (vision + memory + exploration map)
    Compatible avec Stable-Baselines3

    SUPPORT 4 CHANNELS pour exploration_map (avec marqueurs)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int = 256,
            cnn_type: str = 'nature'
    ):
        """
        Initialise le feature extractor combiné

        Args:
            observation_space: Dict space contenant 'visual', 'memory', 'exploration_map'
            features_dim: Dimension finale des features après fusion
            cnn_type: Type de CNN ('nature', 'impala', 'minigrid')
        """
        # On doit passer features_dim à la classe parente
        super().__init__(observation_space, features_dim)

        # ========================================================================
        # 1. DÉTECTION DES MODALITÉS DISPONIBLES
        # ========================================================================
        has_visual = 'visual' in observation_space.spaces
        has_memory = 'memory' in observation_space.spaces
        has_exploration_map = 'exploration_map' in observation_space.spaces

        # Log de la configuration détectée
        logger.info(f"CustomCombinedExtractor configuration:")
        logger.info(f"   Vision : {'activée' if has_visual else 'désactivée'}")
        logger.info(f"   Mémoire : {'activée' if has_memory else 'désactivée'}")
        logger.info(f"   Exploration map : {'activée' if has_exploration_map else 'désactivée'}")

        # ========================================================================
        # 2. INITIALISATION DES DIMENSIONS
        # ========================================================================
        visual_features_dim = 0
        memory_features_dim = 0
        map_features_dim = 0

        # ========================================================================
        # 3. CNN POUR VISION (SI PRÉSENTE)
        # ========================================================================
        if has_visual:
            visual_shape = observation_space['visual'].shape
            visual_channels = visual_shape[-1]

            logger.info(f"📷 Configuration Vision:")
            logger.info(f"   Shape: {visual_shape}")
            logger.info(f"   Channels: {visual_channels}")

            # Sélection du type de CNN
            if cnn_type == 'nature':
                from vision.feature_extractor import NatureCNN
                self.visual_cnn = NatureCNN(
                    input_channels=visual_channels,
                    features_dim=256
                )
            elif cnn_type == 'impala':
                from vision.feature_extractor import ImpalaCNN
                self.visual_cnn = ImpalaCNN(
                    input_channels=visual_channels,
                    features_dim=256
                )
            elif cnn_type == 'minigrid':
                from vision.feature_extractor import MinigridCNN
                self.visual_cnn = MinigridCNN(
                    input_channels=visual_channels,
                    features_dim=256
                )
            else:
                logger.warning(f"CNN type '{cnn_type}' inconnu, fallback sur NatureCNN")
                from vision.feature_extractor import NatureCNN
                self.visual_cnn = NatureCNN(
                    input_channels=visual_channels,
                    features_dim=256
                )

            visual_features_dim = 256
            logger.info(f"   Features dim: {visual_features_dim}")
        else:
            self.visual_cnn = None
            logger.info(f"📷 Vision désactivée")

        # ========================================================================
        # 4. MLP POUR MÉMOIRE (SI PRÉSENTE)
        # ========================================================================
        if has_memory:
            memory_dim = observation_space['memory'].shape[0]

            logger.info(f"🧠 Configuration Mémoire:")
            logger.info(f"   Input dim: {memory_dim}")

            self.memory_mlp = nn.Sequential(
                nn.Linear(memory_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )

            memory_features_dim = 64
            logger.info(f"   Features dim: {memory_features_dim}")
        else:
            self.memory_mlp = None
            logger.info(f"🧠 Mémoire désactivée")

        # ========================================================================
        # 5. CNN POUR EXPLORATION MAP (SI DISPONIBLE)
        # ========================================================================
        self.has_exploration_map = has_exploration_map

        if has_exploration_map:
            map_shape = observation_space['exploration_map'].shape
            map_h, map_w, map_channels = map_shape

            logger.info(f"🗺️ Configuration Exploration Map:")
            logger.info(f"   Shape: {map_shape}")
            logger.info(f"   Dimensions: H={map_h}, W={map_w}, C={map_channels}")

            if map_channels == 4:
                logger.info("   • Marqueurs activés (Channel 3)")
            else:
                logger.warning(f"   • Attendu 4 channels, trouvé {map_channels}")

            # Créer les couches convolutionnelles
            map_conv_layers = nn.Sequential(
                nn.Conv2d(map_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Calculer dynamiquement la dimension après Flatten
            with torch.no_grad():
                dummy_map = torch.zeros(1, map_channels, map_h, map_w)
                map_flatten_dim = map_conv_layers(dummy_map).shape[1]

            logger.info(f"   Flatten dim: {map_flatten_dim}")

            # Créer le pipeline complet (conv + MLP)
            self.map_cnn = nn.Sequential(
                map_conv_layers,
                nn.Linear(map_flatten_dim, 64),
                nn.ReLU()
            )

            map_features_dim = 64
            logger.info(f"   Features dim: {map_features_dim}")
        else:
            self.map_cnn = None
            logger.info(f"🗺️ Exploration map désactivée")

        # ========================================================================
        # 6. COUCHE DE FUSION FINALE
        # ========================================================================
        combined_dim = visual_features_dim + memory_features_dim + map_features_dim

        logger.info(f"🔧 Fusion layer:")
        logger.info(f"   Visual features: {visual_features_dim}")
        logger.info(f"   Memory features: {memory_features_dim}")
        logger.info(f"   Map features: {map_features_dim}")
        logger.info(f"   Combined dim: {combined_dim} -> {features_dim}")

        # Vérifier qu'au moins une modalité est active
        if combined_dim == 0:
            raise ValueError(
                "Aucune modalité active ! Au moins une modalité (visual, memory, ou exploration_map) "
                "doit être présente dans l'observation space."
            )

        # Créer la couche de fusion
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

        # Stocker la configuration pour forward()
        self.has_visual = has_visual
        self.has_memory = has_memory

        logger.info(f"CustomCombinedExtractor initialisé avec succès")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass avec support modalités conditionnelles

        Args:
            observations: Dict pouvant contenir 'visual', 'memory', 'exploration_map'

        Returns:
            Features tensor fusionnées
        """
        features_list = []

        # 1. VISUAL
        if self.has_visual:
            visual = observations['visual']
            if visual.dim() == 4:
                visual = visual.permute(0, 3, 1, 2) # Permuter dimensions: (batch, H, W, C) -> (batch, C, H, W)
            visual_features = self.visual_cnn(visual)
            features_list.append(visual_features)

        # 2. MEMORY
        if self.has_memory:
            memory = observations['memory']
            memory_features = self.memory_mlp(memory)
            features_list.append(memory_features)

        # 3. EXPLORATION MAP
        if self.has_exploration_map:
            exploration_map = observations.get('exploration_map')
            if exploration_map is not None:
                if exploration_map.dim() == 4:
                    exploration_map = exploration_map.permute(0, 3, 1, 2)
                map_features = self.map_cnn(exploration_map)
                features_list.append(map_features)

        # 4. FUSION
        if len(features_list) == 0:
            raise RuntimeError("Aucune feature extraite ! Vérifier la configuration.")

        combined = torch.cat(features_list, dim=1)
        output = self.fusion(combined)

        return output

class CustomVisionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor pour vision seule
    """

    def __init__(
            self,
            observation_space: gym.spaces.Box,
            features_dim: int = 512,
            cnn_type: str = 'nature'
    ):
        super().__init__(observation_space, features_dim)

        visual_channels = observation_space.shape[-1]

        if cnn_type == 'nature':
            self.cnn = NatureCNN(visual_channels, features_dim)
        else:
            # Ajouter d'autres types si nécessaire
            self.cnn = NatureCNN(visual_channels, features_dim)

        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Permuter: (batch, H, W, C) -> (batch, C, H, W)
        if observations.dim() == 4:
            observations = observations.permute(0, 3, 1, 2)

        return self.cnn(observations)


def create_ppo_agent(
        environment_new,
        learning_rate: float = 1e-4,
        n_steps: int = 4096,
        batch_size: int = 512,
        n_epochs: int = 4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        features_dim: int = 256,
        cnn_type: str = 'nature',
        device: str = 'auto',
        verbose: int = 1,
        tensorboard_log: str = None
):
    """
    Crée un agent PPO configuré pour Monster Hunter

    Args:
        environment_new: Environnement Gym
        learning_rate: Taux d'apprentissage
        n_steps: Steps par rollout
        batch_size: Taille des batches
        n_epochs: Epochs d'optimisation par update
        gamma: Facteur de discount
        gae_lambda: Lambda pour GAE
        clip_range: Clip range pour PPO
        ent_coef: Coefficient d'entropie
        vf_coef: Coefficient value function
        max_grad_norm: Gradient clipping
        features_dim: Dimension des features
        cnn_type: Type de CNN ('nature', 'impala', 'minigrid')
        device: Device ('auto', 'cuda', 'cpu')
        verbose: Niveau de verbosité
        tensorboard_log: Path pour logs TensorBoard

    Returns:
        PPO agent
    """

    # Déterminer le feature extractor
    obs_space = environment_new.observation_space

    if isinstance(obs_space, gym.spaces.Dict):
        # Vision + Memory
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                cnn_type=cnn_type
            ),
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
        # Vision seule
        policy_kwargs = dict(
            features_extractor_class=CustomVisionExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                cnn_type=cnn_type
            ),
            net_arch=dict(pi=[256], vf=[256])
        )
    else:
        # Memory seule (MLP par défaut)
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )

    # Créer l'agent PPO
    new_agent = PPO(
        policy='MultiInputPolicy' if isinstance(obs_space, gym.spaces.Dict) else 'MlpPolicy',
        env=environment_new,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=verbose,
        tensorboard_log=tensorboard_log
    )

    # Log des hyperparamètres
    if verbose > 0:
        logger.info("📊 Hyperparamètres PPO:")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   N steps: {n_steps}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   N epochs: {n_epochs}")
        logger.info(f"   Gamma: {gamma}")
        logger.info(f"   GAE lambda: {gae_lambda}")
        logger.info(f"   Clip range: {clip_range}")
        logger.info(f"   Entropy coef: {ent_coef}")
        logger.info(f"   Value function coef: {vf_coef}")
        logger.info(f"   Max grad norm: {max_grad_norm}")
        logger.info(f"   Features dim: {features_dim}")
        logger.info(f"   CNN type: {cnn_type}")
        logger.info(f"   Device: {device}")

    return new_agent

# Fonctions utilitaires
def load_trained_agent(
        model_path: str,
        environment_load,
        device: str = 'auto'
):
    """
    Charge un agent entraîné

    Args:
        model_path: Chemin vers le modèle sauvegardé
        environment_load: Environnement
        device: Device

    Returns:
        Agent PPO chargé
    """
    existing_agent = PPO.load(model_path, env=environment_load, device=device)
    logger.info(f"✅ Modèle chargé: {model_path}")
    return existing_agent


def evaluate_agent(
        agent_evaluated,
        environment_evaluate,
        n_episodes: int = 10,
        render: bool = True
):
    """
    Évalue un agent sur plusieurs épisodes

    Args:
        agent_evaluated: Agent PPO
        environment_evaluate: Environnement
        n_episodes: Nombre d'épisodes
        render: Afficher le rendu

    Returns:
        Stats (mean_reward, std_reward, mean_length)
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs_eval, info = environment_evaluate.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            evaluation_action, _ = agent_evaluated.predict(obs_eval, deterministic=True)
            obs_eval, reward, terminated, truncated, info = environment_evaluate.step(evaluation_action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                environment_evaluate.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        logger.info(f"Épisode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, Length={episode_length}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    logger.info(f"📊 Résultats sur {n_episodes} épisodes:")
    logger.info(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"   Mean length: {mean_length:.0f} steps")

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


# Test
if __name__ == "__main__":
    print("🧪 Test de l'agent PPO\n")

    # Créer un env dummy pour test
    from gymnasium.spaces import Box, Dict as DictSpace


    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = DictSpace({
                'visual': Box(0, 1, shape=(84, 84, 4), dtype=np.float32),
                'memory': Box(0, 1, shape=(67,), dtype=np.float32),
                'exploration_map': Box(-1, 1, shape=(15, 15, 3), dtype=np.float32)
            })
            self.action_space = gym.spaces.Discrete(19)

        def reset(self, seed=None, options=None):
            # Dict keys in alphabetical order
            obs_reset = {
                'exploration_map': np.random.rand(15, 15, 3).astype(np.float32),
                'memory': np.random.rand(67).astype(np.float32),
                'visual': np.random.rand(84, 84, 4).astype(np.float32),
            }
            return obs_reset, {}

        def step(self, action_dummy):
            obs_step = {
                'exploration_map': np.random.rand(15, 15, 3).astype(np.float32),
                'memory': np.random.rand(67).astype(np.float32),
                'visual': np.random.rand(84, 84, 4).astype(np.float32),
            }
            return obs_step, 0.0, False, False, {}


    # Créer env
    env = DummyEnv()

    # Créer agent
    print("🤖 Création de l'agent PPO...")
    agent = create_ppo_agent(
        environment_new=env,
        features_dim=256,
        cnn_type='nature',
        verbose=1
    )

    print(f"\n📊 Agent créé!")
    print(f"   Policy: {type(agent.policy)}")
    print(f"   Device: {agent.device}")

    # Test predict
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    print(f"\n🎮 Test prediction:")
    print(f"   Action: {action}")

    print("\n✅ Test réussi!")