"""
PPO agent with Vision + Memory support
Uses Stable-Baselines3 with custom feature extractor
Optionally uses Transformer-based action heads for cross-head coordination
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agent.extractors import CustomCombinedExtractor, CustomVisionExtractor
from agent.transformer_heads import get_transformer_policy_class
# Lazy init: SB3 is imported only when the class is first accessed
TransformerMultiInputPolicy = get_transformer_policy_class()

from info.module_logger import get_module_logger

logger = get_module_logger('ppo_agent') # Logs per level


def create_ppo_agent(
        environment_new,
        learning_rate: float = 3e-4,
        n_steps: int = 4096,
        batch_size: int = 512,
        n_epochs: int = 4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.03,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        features_dim: int = 256,
        cnn_type: str = 'nature',
        device: str = 'auto',
        verbose: int = 1,
        tensorboard_log: str = None,
        use_transformer_heads: bool = True,
        transformer_kwargs: dict = None,
):
    """
    Create a PPO agent configured for Monster Hunter

    Args:
        environment_new: Gym Environment
        learning_rate: Learning Rate
        n_steps: Steps per rollout
        batch_size: Batch Size
        n_epochs: Optimization Epochs per update
        gamma: Discount Factor
        gae_lambda: Lambda for GAE
        clip_range: Clip Range for PPO
        ent_coef: Entropy Coefficient
        vf_coef: Coefficient Value Function
        max_grad_norm: Gradient Clipping
        features_dim: Feature Dimensions
        cnn_type: CNN Type ('nature', 'impala', 'minigrid')
        device: Device ('auto', 'cuda', 'cpu')
        verbose: Verbosity Level
        tensorboard_log: Path for TensorBoard Logs
        use_transformer_heads: Use Transformer cross-attention between action heads
            instead of independent Linear layers (default: True)
        transformer_kwargs: Config dict for TransformerActionHead
            (d_head, n_layers, n_attn_heads). None uses defaults.

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

    # --- Create PPO agent ---
    # Select policy class — Transformer heads enabled by default for Dict obs
    if use_transformer_heads and isinstance(obs_space, gym.spaces.Dict):
        if TransformerMultiInputPolicy is not None:
            policy_class = TransformerMultiInputPolicy
            policy_kwargs['transformer_kwargs'] = transformer_kwargs or {
                'd_head': 48, 'n_layers': 2, 'n_attn_heads': 4,
            }
            logger.info("Using TransformerMultiInputPolicy (cross-attention action heads)")
        else:
            logger.warning("TransformerMultiInputPolicy not available, falling back to standard")
            policy_class = 'MultiInputPolicy'
    elif isinstance(obs_space, gym.spaces.Dict):
        policy_class = 'MultiInputPolicy'
    else:
        policy_class = 'MlpPolicy'

    new_agent = PPO(
        policy=policy_class,
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
        tensorboard_log=tensorboard_log,
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
        logger.info(f"   Transformer heads: {use_transformer_heads}")

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
                'memory': Box(0, 1, shape=(70,), dtype=np.float32),  # matches MEMORY_VECTOR_SIZE
                'exploration_map': Box(-1, 1, shape=(15, 15, 4), dtype=np.float32)
                # 4ch: visits, player, recent, markers
            })
            self.action_space = gym.spaces.MultiDiscrete([5, 5, 5, 2, 3, 8, 2])

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
                'exploration_map': np.random.rand(15, 15, 4).astype(np.float32),
                'memory': np.random.rand(70).astype(np.float32),
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
    head_names = ['move_x', 'move_y', 'combat', 'guard', 'menu', 'use_item', 'dodge']
    print(f"\n🎮 Test prediction (7 heads):")
    for i, (name, val) in enumerate(zip(head_names, action)):
        print(f"   Head {i} ({name}): {val}")

    # Verify Transformer is actually active
    has_transformer = hasattr(agent.policy.action_net, 'attn_layers')
    print(f"\n   Transformer action head active: {has_transformer}")
    if has_transformer:
        attn = agent.policy.action_net.get_head_attention_summary()
        if attn is not None:
            print(f"   Attention matrix shape: {attn.shape}")  # Expected: (7, 7)

    print("\n✅ Test réussi!")