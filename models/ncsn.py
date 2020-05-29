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
"""Noise-conditional iterative refinement networks."""
import jax
import jax.numpy as jnp
from flax import jax_utils
from flax import nn

from models.shared import TransformerPositionalEncoding, DenseResBlock


class NoiseEncoding(nn.Module):
  """Sinusoidal noise encoding block."""

  def apply(self, noise, channels):
    # noise.shape = (batch_size, 1)
    # channels.shape = ()
    noise = noise.squeeze(-1)
    assert len(noise.shape) == 1
    half_dim = channels // 2
    emb = jnp.log(10000) / float(half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = 5000 * noise[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if channels % 2 == 1:
      emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (noise.shape[0], channels)
    return emb


class DenseFiLM(nn.Module):
  """Feature-wise linear modulation (FiLM) generator."""

  def apply(self, position, embedding_channels, out_channels, sequence=False):
    # position.shape = (batch_size, 1)
    # embedding_channels.shape, out_channels.shape = (), ()
    assert len(position.shape) == 2
    pos_encoding = NoiseEncoding(position, embedding_channels)
    pos_encoding = nn.Dense(pos_encoding, embedding_channels * 4)
    pos_encoding = nn.swish(pos_encoding)
    pos_encoding = nn.Dense(pos_encoding, embedding_channels * 4)

    if sequence:
      pos_encoding = pos_encoding[:, None, :]

    scale = nn.Dense(pos_encoding, out_channels)
    shift = nn.Dense(pos_encoding, out_channels)
    return scale, shift


class ConvFiLM(nn.Module):
  """Convolutional FiLM generator."""

  def apply(self, position, embedding_channels, out_channels):
    # noise.shape = (batch_size, 1, 1)
    # embedding_channels.shape, out_channels.shape = (), ()
    assert len(position.shape) == 3
    position = position.squeeze(-1)
    pos_encoding = NoiseEncoding(position, embedding_channels)
    pos_encoding = nn.Dense(pos_encoding, embedding_channels * 4)
    pos_encoding = nn.swish(pos_encoding)
    pos_encoding = nn.Dense(pos_encoding, embedding_channels * 4)
    pos_encoding = pos_encoding[:, None, :]

    scale = nn.Conv(pos_encoding, out_channels, kernel_size=(3,), strides=(1,))
    shift = nn.Conv(pos_encoding, out_channels, kernel_size=(3,), strides=(1,))
    return scale, shift


class DenseNCSN(nn.Module):
  """Small fully-connected score network."""

  def apply(self, inputs, sigmas, num_layers=3, mlp_dims=2048):
    # inputs.shape = (batch_size, z_dims)
    # sigmas.shape = (batch_size, 1)
    x = inputs
    x = nn.Dense(x, mlp_dims)
    for _ in range(num_layers):
      scale, shift = DenseFiLM(t, 128, mlp_dims)
      x = DenseResBlock(x, mlp_dims, scale=scale, shift=shift)
    x = nn.LayerNorm(x)
    x = nn.Dense(x, inputs.shape[-1])

    output = x / sigmas
    return output


class ConvNCSN(nn.Module):
  """Convolutional score network for sequences."""

  def apply(self, inputs, sigmas):
    # inputs.shape = (batch_size, seq_len, z_dims)
    # sigmas.shape = (batch_size, 1, 1)
    input_channels = inputs.shape[-1]
    x = nn.Conv(inputs, 128, kernel_size=(2,), strides=(1,))

    for channels in (128, 256, 256, 128):
      x = ConvResBlock(x, channels)
      x = ConvResBlock(x, channels)

    x = nn.LayerNorm(x)
    x = nn.relu(x)
    x = nn.Conv(x, input_channels, kernel_size=(2,), strides=(1,))

    output = x / sigmas
    return output


class DenseDDPM(nn.Module):
  """Fully-connected diffusion network."""

  def apply(self, inputs, t, num_layers=3, mlp_dims=2048):
    # inputs.shape = (batch_size, z_dims)
    # t.shape = (batch_size, 1)
    x = inputs
    x = nn.Dense(x, mlp_dims)
    for _ in range(num_layers):
      scale, shift = DenseFiLM(t, 128, mlp_dims)
      x = DenseResBlock(x, mlp_dims, scale=scale, shift=shift)
    x = nn.LayerNorm(x)
    x = nn.Dense(x, inputs.shape[-1])
    return x


class TransformerDDPM(nn.Module):
  """Transformer-based diffusion model."""

  def apply(self,
            inputs,
            t,
            num_layers=6,
            num_heads=8,
            num_mlp_layers=2,
            mlp_dims=2048):
    batch_size, seq_len, data_channels = inputs.shape

    x = inputs
    embed_channels = 128
    temb = TransformerPositionalEncoding(jnp.arange(seq_len), embed_channels)
    temb = temb[None, :, :]
    assert temb.shape[1:] == (seq_len, embed_channels), temb.shape
    x = nn.Dense(x, embed_channels)

    x = x + temb
    for _ in range(num_layers):
      shortcut = x
      x = nn.LayerNorm(x)
      x = nn.SelfAttention(x, num_heads=num_heads)
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
      scale, shift = DenseFiLM(t.squeeze(-1), 128, mlp_dims, sequence=True)
      x = DenseResBlock(x, mlp_dims, scale=scale, shift=shift)

    x = nn.LayerNorm(x)
    x = nn.Dense(x, data_channels)
    return x
