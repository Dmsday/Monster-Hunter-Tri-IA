"""
PPO agent with Vision + Memory support
Uses Stable-Baselines3 with custom feature extractor
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agent.extractors import CustomCombinedExtractor, CustomVisionExtractor
from info.module_logger import get_module_logger
from core.controller.action_heads import ACTION_BRANCHES

logger = get_module_logger('ppo_agent') # Logs per level


def create_ppo_agent(
        environment_new,
        learning_rate: float = 3e-4, # Faster initial convergence for 28K action combos
        n_steps: int = 4096,
        batch_size: int = 512,
        n_epochs: int = 4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.03, # Higher entropy for 7-head MultiDiscrete (28K combos), prevents early convergence to NOOP
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