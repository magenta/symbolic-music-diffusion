# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Autoregressive models."""
import jax.numpy as jnp
import jax

from flax import jax_utils
from flax import nn

from models.shared import TransformerPositionalEncoding, DenseResBlock, MDN


def shift_right(x):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = jnp.pad(x,
                   pad_widths,
                   mode='constant',
                   constant_values=x.dtype.type(0))
  return padded[:, :-1]


class TransformerMDN(nn.Module):
  """Transformer with continuous outputs."""

  def apply(self,
            inputs,
            shift=True,
            num_layers=6,
            num_heads=8,
            num_mlp_layers=2,
            mlp_dims=2048,
            mdn_mixtures=100):
    batch_size, seq_len, data_channels = inputs.shape
    x = inputs
    if shift:
      x = shift_right(x)
    embed_channels = 128
    temb = TransformerPositionalEncoding(jnp.arange(seq_len), embed_channels)
    temb = temb[None, :, :]
    assert temb.shape[1:] == (seq_len, embed_channels), temb.shape
    x = nn.Dense(x, embed_channels)

    x = x + temb
    for _ in range(num_layers):
      shortcut = x
      x = nn.LayerNorm(x)
      x = nn.SelfAttention(x, causal_mask=True, num_heads=num_heads)
      x = x + shortcut
      shortcut2 = x
      x = nn.LayerNorm(x)
      x = nn.Dense(x, mlp_dims)
      x = nn.gelu(x)
      x = nn.Dense(x, embed_channels)
      x = x + shortcut2

    x = nn.LayerNorm(x)
    x = nn.Dense(x, mlp_dims)

    for _ in range(num_mlp_layers):
      x = DenseResBlock(x, mlp_dims)

    x = nn.LayerNorm(x)
    mdn = MDN.partial(out_channels=data_channels,
                      num_components=mdn_mixtures,
                      name='mdn')
    pi, mu, log_sigma = mdn(x)
    return pi, mu, log_sigma
