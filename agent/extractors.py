from typing import Dict
import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from vision.feature_extractor import NatureCNN
from info.module_logger import get_module_logger

logger = get_module_logger('extractors')

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observations (vision + memory + exploration map)

    Compatible with Stable-Baselines3

    Supports 4 channels for exploration maps (with markers)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int = 256,
            cnn_type: str = 'nature'
    ):
        """
        Initializes the combined feature extractor

        Args:
            observation_space: Dict space contenant 'visual', 'memory', 'exploration_map'
            features_dim: Final dimensions of features after merging
            cnn_type: 'nature', 'impala', 'minigrid'
        """
        # On doit passer features_dim à la classe parente
        super().__init__(observation_space, features_dim)

        # ========================================================================
        # 1. DETECTION OF AVAILABLE MODALITIES
        # ========================================================================
        has_visual = 'visual' in observation_space.spaces
        has_memory = 'memory' in observation_space.spaces
        has_exploration_map = 'exploration_map' in observation_space.spaces

        # Log de la configuration détectée
        logger.info(f"CustomCombinedExtractor configuration:")
        logger.info(f"   Vision : {'enabled' if has_visual else 'disabled'}")
        logger.info(f"   Memory : {'enabled' if has_memory else 'disabled'}")
        logger.info(f"   Exploration map : {'enabled' if has_exploration_map else 'disabled'}")

        # ========================================================================
        # 2. INITIALIZING DIMENSIONS
        # ========================================================================
        visual_features_dim = 0
        memory_features_dim = 0
        map_features_dim = 0

        # ========================================================================
        # 3. CNN FOR VISION (IF PRESENT)
        # ========================================================================
        if has_visual:
            visual_shape = observation_space['visual'].shape
            visual_channels = visual_shape[-1]

            logger.info(f"Vision configuration:")
            logger.info(f"   Shape: {visual_shape}")
            logger.info(f"   Channels: {visual_channels}")

            # CNN type selection
            # for future implementation
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
                logger.warning(f"CNN type '{cnn_type}' unknown, fallback on NatureCNN")
                from vision.feature_extractor import NatureCNN
                self.visual_cnn = NatureCNN(
                    input_channels=visual_channels,
                    features_dim=256
                )

            visual_features_dim = 256
            logger.info(f"   Features dim: {visual_features_dim}")
        else:
            self.visual_cnn = None
            logger.info(f"Vision disabled")

        # ========================================================================
        # 4. MLP FOR THE RECORD (IF PRESENT)
        # ========================================================================
        if has_memory:
            memory_dim = observation_space['memory'].shape[0]

            logger.info(f"Memory configuration:")
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
            logger.info(f"Memory disabled")

        # ========================================================================
        # 5. CNN FOR EXPLORATION MAP (IF AVAILABLE)
        # ========================================================================
        self.has_exploration_map = has_exploration_map

        if has_exploration_map:
            map_shape = observation_space['exploration_map'].shape
            map_h, map_w, map_channels = map_shape

            logger.info(f"Exploration map configuration:")
            logger.info(f"   Shape: {map_shape}")
            logger.debug(f"   Dimensions: H={map_h}, W={map_w}, C={map_channels}")

            if map_channels == 4:
                logger.info("   Markers enabled (channel 3)")
            else:
                logger.warning(f"   Expected 4 channels, got {map_channels}")

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
            logger.info(f"Exploration map disabled")

        # ========================================================================
        # 6. FINAL FUSION LAYER
        # ========================================================================
        combined_dim = visual_features_dim + memory_features_dim + map_features_dim

        logger.info(f"Fusion layer:")
        logger.info(f"   Visual features: {visual_features_dim}")
        logger.info(f"   Memory features: {memory_features_dim}")
        logger.info(f"   Map features: {map_features_dim}")
        logger.info(f"   Combined dim: {combined_dim} -> {features_dim}")

        # Vérifier qu'au moins une modalité est active
        if combined_dim == 0:
            raise ValueError(
                "No modality active! At least one modality (visual, memory, or exploration_map)"
                "must be present in the observation space."
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

        logger.info(f"CustomCombinedExtractor successfully initialized !")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with support, subject to conditions

        Args:
            observations: Dict that may contain 'visual', 'memory', 'exploration_map'

        Returns:
            Tensor features fused
        """
        features_list = []

        # 1. VISUAL
        if self.has_visual:
            visual = observations['visual']
            if visual.dim() == 4:
                visual = visual.permute(0, 3, 1, 2) # Swap dimensions: (batch, H, W, C) -> (batch, C, H, W)
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
            raise RuntimeError("No features extracted ! Check the configuration.")

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
            logger.warning(f"CNN type '{cnn_type}' unknown in CustomVisionExtractor, falling back to NatureCNN")
            self.cnn = NatureCNN(visual_channels, features_dim)

        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Permutation: (batch, H, W, C) -> (batch, C, H, W)
        if observations.dim() == 4:
            observations = observations.permute(0, 3, 1, 2)

        return self.cnn(observations)