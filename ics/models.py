"""ICS model: permutation-invariant context encoder (deep-sets, mean+max
pooling) conditioning a velocity head for flow matching in whitened DMAX-dim
space. Time embedding reuses jax_flows' sinusoidal embedding but with t
scaled x1000 (the audit found the raw t in [0,1] barely moves the embedding).
"""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from jax_flows.utils import sinusoidal_time_embedding

from .zoo import DMAX


class ContextEncoder(nn.Module):
    """Deep-sets encoder; with n_attn > 0, self-attention token-mixing blocks
    run before pooling (permutation-EQUIVARIANT, so the pooled output stays
    invariant — attempt-3b: token mixing is what lets the encoder relate
    energy readings across the landscape instead of pooling them blind).
    Optional mask (B, K) with 1 = keep supports short-context augmentation."""

    enc_dim: int = 128
    hidden: int = 256
    n_attn: int = 0

    @nn.compact
    def __call__(self, tokens, mask=None):  # (B, K, F) -> (B, enc_dim)
        h = nn.Dense(self.hidden)(tokens)
        h = nn.silu(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.silu(h)
        if mask is None:
            mask = jnp.ones(tokens.shape[:2], dtype=h.dtype)
        for _ in range(self.n_attn):
            attn_mask = (mask[:, None, None, :] > 0)  # (B, 1, 1, K)
            a = nn.SelfAttention(num_heads=4, qkv_features=self.hidden)(
                nn.LayerNorm()(h), mask=attn_mask)
            h = h + a
            f = nn.Dense(self.hidden)(nn.silu(nn.Dense(self.hidden)(nn.LayerNorm()(h))))
            h = h + f
        m = mask[:, :, None]
        mean_pool = (h * m).sum(axis=1) / jnp.maximum(m.sum(axis=1), 1.0)
        max_pool = jnp.where(m > 0, h, -jnp.inf).max(axis=1)
        pooled = jnp.concatenate([mean_pool, max_pool], axis=-1)
        out = nn.Dense(self.enc_dim)(pooled)
        return nn.LayerNorm()(out)


class VelocityHead(nn.Module):
    hidden: Sequence[int] = (256, 256, 256)
    time_embed_dim: int = 64

    @nn.compact
    def __call__(self, x, t, cond):  # (B, DMAX), (B,), (B, C) -> (B, DMAX)
        temb = sinusoidal_time_embedding(t * 1000.0, self.time_embed_dim)
        h = (
            nn.Dense(self.hidden[0])(x)
            + nn.Dense(self.hidden[0])(temb)
            + nn.Dense(self.hidden[0])(cond)
        )
        h = nn.silu(h)
        for w in self.hidden[1:]:
            h_res = h
            h = nn.silu(nn.Dense(w)(h))
            if h_res.shape == h.shape:
                h = h + h_res
        return nn.Dense(DMAX)(h)


class ICSModel(nn.Module):
    enc_dim: int = 128
    enc_hidden: int = 256
    head_hidden: Sequence[int] = (256, 256, 256)
    time_embed_dim: int = 64
    n_attn: int = 0

    def setup(self):
        self.encoder = ContextEncoder(enc_dim=self.enc_dim, hidden=self.enc_hidden,
                                      n_attn=self.n_attn)
        self.head = VelocityHead(
            hidden=self.head_hidden, time_embed_dim=self.time_embed_dim
        )

    def __call__(self, x, t, tokens, mask=None):
        return self.head(x, t, self.encoder(tokens, mask))

    def encode(self, tokens, mask=None):
        return self.encoder(tokens, mask)

    def velocity(self, x, t, cond):
        return self.head(x, t, cond)
