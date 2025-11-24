"""
Extracteur de features CNN pour les frames de jeu
Architecture optimisée pour RL (PPO)

NOTE : L'extracteur hybride principal est CustomCombinedExtractor dans ppo_agent.py
Ce fichier contient uniquement les architectures CNN de base (NatureCNN, ImpalaCNN, MinigridCNN)
"""

import torch
import torch.nn as nn
import numpy as np


class NatureCNN(nn.Module):
    """
    CNN inspiré de Nature DQN (Mnih et al. 2015)
    Standard pour les jeux et RL basé vision
    """

    def __init__(
            self,
            input_channels: int = 4,  # Frame stack
            features_dim: int = 512
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # Conv 1: 84x84xC -> 20x20x32
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),

            # Conv 2: 20x20x32 -> 9x9x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # Conv 3: 9x9x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten()
        )

        # Calculer la dimension après flatten
        with torch.no_grad():
            n_flatten = self._get_nature_output_size(input_channels)

        # FC final
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.features_dim = features_dim

    def _get_nature_output_size(self, input_channels):
        """
        Calcule la taille de sortie après les convolutions
        """
        test_input = torch.zeros(1, input_channels, 84, 84)
        test_output = self.cnn(test_input)
        return int(np.prod(test_output.shape))

    def forward(self, x):
        """
        Args:
            x: Tensor (batch, channels, height, width)
        Returns:
            features: Tensor (batch, features_dim)
        """
        cnn_output = self.cnn(x)
        nature_features  = self.linear(cnn_output)
        return nature_features


class ImpalaCNN(nn.Module):
    """
    Architecture IMPALA (Espeholt et al. 2018)
    Plus moderne et performante que Nature CNN
    """

    def __init__(
            self,
            input_channels: int = 4,
            features_dim: int = 256
    ):
        super().__init__()

        # Bloc résiduel IMPALA
        self.conv_seqs = nn.ModuleList([
            self._make_conv_sequence(input_channels, 16),
            self._make_conv_sequence(16, 32),
            self._make_conv_sequence(32, 32)
        ])

        # Pool et FC
        self.pool = nn.AdaptiveMaxPool2d((8, 8))

        # Calculer taille flatten
        with torch.no_grad():
            temp_tensor = torch.zeros(1, input_channels, 84, 84)
            for seq in self.conv_seqs:
                temp_tensor = seq(temp_tensor)
            temp_tensor = self.pool(temp_tensor)
            n_flatten = int(np.prod(temp_tensor.shape))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.features_dim = features_dim

    @staticmethod
    def _make_conv_sequence(in_channels, out_channels):
        """Crée une séquence Conv + MaxPool + Residual"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Residual block
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)

        pooled = self.pool(x)
        impala_features = self.fc(pooled)
        return impala_features


class MinigridCNN(nn.Module):
    """
    CNN léger inspiré de Minigrid
    Plus rapide à entraîner, bon pour prototypage
    """

    def __init__(
            self,
            input_channels: int = 4,
            features_dim: int = 128
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # 84x84 -> 40x40
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 40x40 -> 19x19
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 19x19 -> 9x9
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self._get_minigrid_output_size(input_channels)

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.features_dim = features_dim

    def _get_minigrid_output_size(self, input_channels):
        test_input = torch.zeros(1, input_channels, 84, 84)
        test_output = self.cnn(test_input)
        return int(np.prod(test_output.shape))

    def forward(self, x):
        cnn_out = self.cnn(x)
        mini_features = self.fc(cnn_out)
        return mini_features


# Test des CNNs individuels
if __name__ == "__main__":
    print("🧪 Test des extracteurs de features CNN de base\n")

    batch_size = 4

    # Test 1: Nature CNN
    print("1️⃣ Nature CNN")
    model = NatureCNN(input_channels=4, features_dim=512)
    dummy_input = torch.randn(batch_size, 4, 84, 84)
    output = model(dummy_input)
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Test 2: IMPALA CNN
    print("2️⃣ IMPALA CNN")
    model = ImpalaCNN(input_channels=4, features_dim=256)
    output = model(dummy_input)
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Test 3: Minigrid CNN
    print("3️⃣ Minigrid CNN")
    model = MinigridCNN(input_channels=4, features_dim=128)
    output = model(dummy_input)
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}\n")

    print("✅ Tous les tests réussis!")
    print("\n💡 Note : CustomCombinedExtractor (hybride) est dans ppo_agent.py")