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
"""Sample from trained autoregressive MDN."""
import os
import time

from absl import app
from absl import flags
from absl import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from flax.metrics import tensorboard
from flax.training import checkpoints

import utils.data_utils as data_utils
import utils.train_utils as train_utils
import utils.losses as losses
import utils.metrics as metrics
import train_transformer
import input_pipeline

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

FLAGS = flags.FLAGS
AUTOTUNE = tf.data.experimental.AUTOTUNE

flags.DEFINE_integer('sample_seed', 1,
                     'Random number generator seed for sampling.')
flags.DEFINE_string('sampling_dir', 'sample', 'Sampling directory.')
flags.DEFINE_integer('sample_size', 1000, 'Number of samples.')
flags.DEFINE_boolean('flush', True, 'Flush generated samples to disk.')


def sample(num_samples=2400, steps=32, embedding_dims=42, rng_seed=1,
           real=None):
  """Generate samples using autoregressive decoding.
  
  Args:
    num_samples: The number of samples to generate.
    steps: Number of sampling steps.
    embedding_dims: Number of dimensions per embedding.
    rng_seed: Initialization seed.
 
  Returns:
    generated: An array of generated samples.
  """
  rng = jax.random.PRNGKey(rng_seed)
  rng, model_rng = jax.random.split(rng)

  # Create a model with dummy parameters and a dummy optimizer
  lm_kwargs = {
      'num_layers': FLAGS.num_layers,
      'num_heads': FLAGS.num_heads,
      'mdn_mixtures': FLAGS.mdn_components,
      'num_mlp_layers': FLAGS.num_mlp_layers,
      'mlp_dims': FLAGS.mlp_dims
  }
  model = train_transformer.create_model(model_rng, (steps, embedding_dims),
                                         lm_kwargs,
                                         batch_size=1,
                                         verbose=True)
  optimizer = train_transformer.create_optimizer(model, 0)
  early_stop = train_utils.EarlyStopping()

  # Load learned parameters
  optimizer, early_stop = checkpoints.restore_checkpoint(
      FLAGS.model_dir, (optimizer, early_stop))

  # Autoregressive decoding
  t0 = time.time()
  tokens = jnp.zeros((num_samples, steps, embedding_dims))

  for i in range(steps):
    pi, mu, log_sigma = optimizer.target(tokens, shift=False)

    channels = tokens.shape[-1]
    mdn_k = pi.shape[-1]
    out_pi = pi.reshape(-1, mdn_k)
    out_mu = mu.reshape(-1, channels * mdn_k)
    out_log_sigma = log_sigma.reshape(-1, channels * mdn_k)
    mix_dist = tfd.Categorical(logits=out_pi)
    mus = out_mu.reshape(-1, mdn_k, channels)
    log_sigmas = out_log_sigma.reshape(-1, mdn_k, channels)
    sigmas = jnp.exp(log_sigmas)
    component_dist = tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigmas)
    mixture = tfd.MixtureSameFamily(mixture_distribution=mix_dist,
                                    components_distribution=component_dist)

    rng, embed_rng = jax.random.split(rng)
    next_tokens = mixture.sample(seed=embed_rng).reshape(*tokens.shape)
    next_z = next_tokens[:, i]

    if i < steps - 1:
      tokens = jax.ops.index_update(tokens, jax.ops.index[:, i + 1], next_z)
    else:
      tokens = next_tokens  # remove start token

  logging.info('Generated samples in %f seconds', time.time() - t0)
  return tokens


def main(argv):
  del argv  # unused

  logging.info(FLAGS.flags_into_string())
  logging.info('Platform: %s', jax.lib.xla_bridge.get_backend().platform)

  # Make sure TensorFlow does not allocate GPU memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  log_dir = FLAGS.sampling_dir

  pca = data_utils.load(os.path.expanduser(
      FLAGS.pca_ckpt)) if FLAGS.pca_ckpt else None
  slice_idx = data_utils.load(os.path.expanduser(
      FLAGS.slice_ckpt)) if FLAGS.slice_ckpt else None
  dim_weights = data_utils.load(os.path.expanduser(
      FLAGS.dim_weights_ckpt)) if FLAGS.dim_weights_ckpt else None

  train_ds, eval_ds = input_pipeline.get_dataset(
      dataset=FLAGS.dataset,
      data_shape=FLAGS.data_shape,
      problem='vae',
      batch_size=FLAGS.batch_size,
      normalize=FLAGS.normalize,
      pca_ckpt=FLAGS.pca_ckpt,
      slice_ckpt=FLAGS.slice_ckpt,
      dim_weights_ckpt=FLAGS.dim_weights_ckpt,
      include_cardinality=False)
  eval_min, eval_max = eval_ds.min, eval_ds.max
  eval_ds = eval_ds.unbatch()
  if FLAGS.sample_size is not None:
    eval_ds = eval_ds.take(FLAGS.sample_size)
  real = np.stack([ex for ex in tfds.as_numpy(eval_ds)])
  shape = real[0].shape

  # Generate samples
  generated = sample(FLAGS.sample_size, shape[-2], shape[-1], FLAGS.sample_seed,
                     real)

  # Dump generated to CPU.
  generated = np.array(generated)

  # Write samples to disk (used for listening).
  if FLAGS.flush:
    # Inverse transform data back to listenable/unnormalized latent space.
    generated_t = input_pipeline.inverse_data_transform(generated,
                                                        FLAGS.normalize, pca,
                                                        train_ds.min,
                                                        train_ds.max, slice_idx,
                                                        dim_weights)
    real_t = input_pipeline.inverse_data_transform(real, FLAGS.normalize, pca,
                                                   eval_min, eval_max,
                                                   slice_idx, dim_weights)
    data_utils.save(real_t, os.path.join(log_dir, 'mdn/real.pkl'))
    data_utils.save(generated_t, os.path.join(log_dir, 'mdn/generated.pkl'))


if __name__ == '__main__':
  app.run(main)
