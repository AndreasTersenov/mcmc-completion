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
    enc_dim: int = 128
    hidden: int = 256

    @nn.compact
    def __call__(self, tokens):  # (B, K, F) -> (B, enc_dim)
        h = nn.Dense(self.hidden)(tokens)
        h = nn.silu(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.silu(h)
        pooled = jnp.concatenate([h.mean(axis=1), h.max(axis=1)], axis=-1)
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

    def setup(self):
        self.encoder = ContextEncoder(enc_dim=self.enc_dim, hidden=self.enc_hidden)
        self.head = VelocityHead(
            hidden=self.head_hidden, time_embed_dim=self.time_embed_dim
        )

    def __call__(self, x, t, tokens):
        return self.head(x, t, self.encoder(tokens))

    def encode(self, tokens):
        return self.encoder(tokens)

    def velocity(self, x, t, cond):
        return self.head(x, t, cond)
