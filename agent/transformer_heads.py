"""
transformer_heads.py — Transformer-based cross-attention action head for MultiDiscrete.

Replaces SB3's default independent Linear heads with a self-attention mechanism
that lets the 7 action heads coordinate their decisions BEFORE producing logits.

Architecture:
    latent_pi (256)
        │
    Per-head projection → 7 tokens (B, 7, d_head)
        │
    Self-Attention (2 layers) — heads see each other's intentions
        │
    Per-head logit projection → (B, sum(branches))
        │
    MultiCategoricalDistribution (unchanged SB3 mechanism)

Integration:
    Used via TransformerMultiInputPolicy in ppo_agent.py.
    Backward-compatible: old checkpoints load with standard policy,
    new training can opt-in with --transformer-heads flag.

Parameter cost: ~130K params (vs ~8K for standard Linear heads)
Inference cost: ~0.1ms added per step (negligible vs 33ms step_duration)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from info.module_logger import get_module_logger

logger = get_module_logger('transformer_heads')


# =====================================================================
#  BUILDING BLOCKS
# =====================================================================

class _HeadAttentionBlock(nn.Module):
    """
    Pre-norm Transformer block with explicit attention weight capture.

    Uses nn.MultiheadAttention directly (not TransformerEncoder) so we
    can extract the 7×7 attention matrix for GUI visualization.
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Stored after each forward — detached, (B, num_heads_attn, 7, 7)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attended, weights = self.attn(
            normed, normed, normed,
            need_weights=True, average_attn_weights=False,
        )
        # weights shape: (B, n_attn_heads, 7, 7)
        self.last_attn_weights = weights.detach()
        x = x + attended

        # Pre-norm feed-forward
        x = x + self.ffn(self.norm2(x))
        return x


# =====================================================================
#  MAIN MODULE — drop-in replacement for SB3's action_net
# =====================================================================

class TransformerActionHead(nn.Module):
    """
    Transformer-based action head for MultiDiscrete action spaces.

    Drop-in replacement for the nn.Linear that SB3 creates as action_net.
    Output shape is identical: (batch, sum(action_branches)).

    Instead of 7 independent Linear layers sharing the same input,
    each head gets its own token embedding, the tokens attend to each
    other (learning cross-head coordination), then each token produces
    its branch logits.

    The COMPAT dict in action_heads.py still applies post-hoc for safety,
    but the Transformer learns to avoid conflicts proactively — reducing
    wasted capacity on impossible action combinations.

    Args:
        latent_dim:     Dimension of latent_pi from SB3's MLP extractor (typically 256)
        action_branches: List of branch sizes per head, e.g. [5, 5, 5, 2, 3, 8, 2]
        d_head:         Embedding dimension per head token (default 48)
        n_layers:       Number of self-attention layers (default 2)
        n_attn_heads:   Number of attention heads in MHA (default 4, must divide d_head)
        dropout:        Dropout rate (default 0.0 — PPO is on-policy, dropout rarely helps)
    """

    def __init__(
        self,
        latent_dim: int,
        action_branches: List[int],
        d_head: int = 48,
        n_layers: int = 2,
        n_attn_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_action_heads = len(action_branches)
        self.action_branches = action_branches
        self.d_head = d_head
        self.n_layers = n_layers

        assert d_head % n_attn_heads == 0, (
            f"d_head ({d_head}) must be divisible by n_attn_heads ({n_attn_heads})"
        )

        # 1. Per-head input projection — each head starts with a different
        #    "question" derived from the shared latent vector
        self.head_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, d_head),
                nn.GELU(),
            )
            for _ in range(self.num_action_heads)
        ])

        # 2. Learned positional embedding (one per head)
        #    Encodes head identity so the Transformer knows which token
        #    is movement vs combat vs menu etc.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_action_heads, d_head) * 0.02
        )

        # 3. Self-attention layers (the core coordination mechanism)
        self.attn_layers = nn.ModuleList([
            _HeadAttentionBlock(
                d_model=d_head,
                n_heads=n_attn_heads,
                dim_ff=d_head * 2,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # 4. Final layer norm before logit projection
        self.final_norm = nn.LayerNorm(d_head)

        # 5. Per-head logit projection (each head → its own action space)
        self.logit_heads = nn.ModuleList([
            nn.Linear(d_head, n_branches)
            for n_branches in action_branches
        ])

        # Parameter count for logging
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"TransformerActionHead initialized:")
        logger.info(f"  Heads: {self.num_action_heads}, branches: {action_branches}")
        logger.info(f"  d_head={d_head}, layers={n_layers}, attn_heads={n_attn_heads}")
        logger.info(f"  Total parameters: {total_params:,}")

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_pi: (batch, latent_dim) from SB3's MLP extractor

        Returns:
            action_logits: (batch, sum(action_branches)) — same shape as
                          standard SB3 Linear action_net output
        """
        batch_size = latent_pi.shape[0]

        # 1. Project to per-head tokens: (B, 7, d_head)
        tokens = torch.stack([
            proj(latent_pi) for proj in self.head_projections
        ], dim=1)

        # 2. Add positional embedding (head identity)
        tokens = tokens + self.pos_embedding

        # 3. Self-attention: heads coordinate decisions
        for layer in self.attn_layers:
            tokens = layer(tokens)

        # 4. Final norm
        tokens = self.final_norm(tokens)

        # 5. Per-head logits → concatenate to match SB3's expected shape
        logits = [
            head_proj(tokens[:, i, :])
            for i, head_proj in enumerate(self.logit_heads)
        ]

        return torch.cat(logits, dim=1)  # (B, sum(branches))

    # -----------------------------------------------------------------
    #  Diagnostic API — for GUI / debugging
    # -----------------------------------------------------------------

    def get_attention_weights(self) -> List[Optional[torch.Tensor]]:
        """
        Return attention weights from each layer.

        Returns:
            List of tensors, one per layer.
            Each tensor shape: (batch, n_attn_heads, 7, 7)
            Returns None for a layer if no forward pass happened yet.
        """
        return [layer.last_attn_weights for layer in self.attn_layers]

    def get_head_attention_summary(self) -> Optional[torch.Tensor]:
        """
        Average attention across layers and attention heads.
        Returns a simple 7×7 matrix showing how much each action head
        "listens to" each other head.

        Returns:
            (7, 7) tensor averaged over batch/layers/attn_heads, or None.
        """
        weights = self.get_attention_weights()
        valid = [w for w in weights if w is not None]
        if not valid:
            return None

        # Stack layers: (n_layers, B, n_attn_heads, 7, 7)
        stacked = torch.stack(valid, dim=0)
        # Average over layers, batch, and attention heads → (7, 7)
        return stacked.mean(dim=(0, 1, 2))


# =====================================================================
#  CUSTOM SB3 POLICY — uses TransformerActionHead
# =====================================================================

def _make_transformer_policy():
    """
    Factory that imports SB3 and creates the policy class.
    Deferred import to avoid circular dependencies and allow the module
    to be imported even when SB3 is not installed (e.g. for testing).
    """
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy

    class TransformerMultiInputPolicy(MultiInputActorCriticPolicy):
        """
        SB3 MultiInputPolicy with Transformer-based action heads.

        Usage in ppo_agent.py:
            from agent.transformer_heads import TransformerMultiInputPolicy
            PPO(policy=TransformerMultiInputPolicy, env=env, policy_kwargs={
                'transformer_kwargs': {'d_head': 48, 'n_layers': 2},
                ...
            })

        The value function (critic) is unchanged — only the actor's
        action head is replaced with the Transformer module.
        """

        def __init__(self, *args, transformer_kwargs=None, **kwargs):
            # Store before super().__init__ calls _build()
            self._transformer_kwargs = transformer_kwargs or {}
            super().__init__(*args, **kwargs)

        def _build(self, lr_schedule):
            # Let SB3 build everything normally (extractor, mlp, action_net, value_net)
            super()._build(lr_schedule)

            # Now replace the standard Linear action_net with our Transformer
            action_branches = self.action_space.nvec.tolist()
            latent_dim = self.mlp_extractor.latent_dim_pi

            self.action_net = TransformerActionHead(
                latent_dim=latent_dim,
                action_branches=action_branches,
                **self._transformer_kwargs,
            )

            # Re-add transformer params to the optimizer
            # SB3's _build() already created the optimizer with the old action_net,
            # so we need to rebuild it with the new parameters
            self.optimizer = self.optimizer_class(
                self.parameters(),
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

            logger.info(
                f"TransformerMultiInputPolicy: replaced Linear action_net "
                f"with TransformerActionHead (latent_dim={latent_dim}, "
                f"branches={action_branches})"
            )

    return TransformerMultiInputPolicy


# Lazy singleton — created on first access
_TransformerMultiInputPolicy = None


def get_transformer_policy_class():
    """
    Get the TransformerMultiInputPolicy class (lazy import).
    Use this in ppo_agent.py instead of importing the class directly.
    """
    global _TransformerMultiInputPolicy
    if _TransformerMultiInputPolicy is None:
        _TransformerMultiInputPolicy = _make_transformer_policy()
    return _TransformerMultiInputPolicy
