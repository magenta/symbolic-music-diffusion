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
"""Shared neural network components."""
import jax.numpy as jnp
from flax import nn


class MDN(nn.Module):
  """Mixture density output layer."""

  def apply(self, inputs, out_channels=512, num_components=10):
    # inputs.shape = (batch_size, seq_len, channels)
    x = inputs
    mu = nn.Dense(x, out_channels * num_components)
    log_sigma = nn.Dense(x, out_channels * num_components)
    pi = nn.Dense(x, num_components)
    return pi, mu, log_sigma


class TransformerPositionalEncoding(nn.Module):
  """Transformer positional encoding block."""

  def apply(self, timesteps, channels):
    # timesteps.shape = (seq_len,)
    # channels.shape = ()
    assert len(timesteps.shape) == 1
    half_dim = channels // 2
    emb = jnp.log(10000) / float(half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if channels % 2 == 1:
      emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], channels)
    return emb


class FeaturewiseAffine(nn.Module):
  """Feature-wise affine layer."""

  def apply(self, x, scale, shift):
    return scale * x + shift


class DenseResBlock(nn.Module):
  """Fully-connected residual block."""

  def apply(self, inputs, output_size, scale=1., shift=0.):
    output = nn.LayerNorm(inputs)
    output = FeaturewiseAffine(output, scale, shift)
    output = nn.swish(output)
    output = nn.Dense(output, output_size)
    output = nn.LayerNorm(output)
    output = FeaturewiseAffine(output, scale, shift)
    output = nn.swish(output)
    output = nn.Dense(output, output_size)

    shortcut = inputs
    if inputs.shape[-1] != output_size:
      shortcut = nn.Dense(inputs, output_size)

    return output + shortcut


class ConvResBlock(nn.Module):
  """Convolutional residual block."""

  def apply(self, inputs, out_channels, scale=1., shift=0.):
    output = nn.Conv(inputs, out_channels, kernel_size=(3,), strides=(1,))
    output = nn.swish(output)
    shortcut = output
    output = nn.Conv(output, out_channels, kernel_size=(3,), strides=(1,))
    output = nn.GroupNorm(output)
    output = FeaturewiseAffine(output, scale, shift)
    output = nn.swish(output)
    assert shortcut.shape[-1] == out_channels
    return output + shortcut
