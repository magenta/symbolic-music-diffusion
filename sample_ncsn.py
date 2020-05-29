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
"""Sample from trained score network."""
import io
import os
import time
import warnings

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
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

import utils.data_utils as data_utils
import utils.ebm_utils as ebm_utils
import utils.train_utils as train_utils
import utils.plot_utils as plot_utils
import utils.losses as losses
import utils.metrics as metrics
import models.ncsn as ncsn
import train_ncsn
import input_pipeline

FLAGS = flags.FLAGS
AUTOTUNE = tf.data.experimental.AUTOTUNE

flags.DEFINE_integer('sample_seed', 1,
                     'Random number generator seed for sampling.')
flags.DEFINE_string('sampling_dir', 'samples', 'Sampling directory.')
flags.DEFINE_integer('sample_size', 1000, 'Number of samples.')

# Metrics.
flags.DEFINE_boolean('compute_metrics', False,
                     'Compute evaluation metrics for generated samples.')
flags.DEFINE_boolean('compute_final_only', False,
                     'Do not include metrics for intermediate samples.')

# Generation.
flags.DEFINE_boolean('flush', True, 'Flush generated samples to disk.')
flags.DEFINE_boolean('animate', False, 'Generate animation of samples.')
flags.DEFINE_boolean('infill', False, 'Infill.')
flags.DEFINE_boolean('interpolate', False, 'Interpolate.')


def evaluate(writer, real, collection, baseline, valid_real):
  """Evaluation metrics.

  NOTE: It is important for collection and real to be normalized for 
  accurate evaluation statistics.

  Args:
    writer: TensorBoard summary writer.
    real: An array of real data samples of shape [N, *data_shape].
    collection: Generated samples at varying timesteps. Each array
        is of shape [sampling_steps, N, *data_shape].
    baseline: Generated samples from baseline.
    valid_real: Real samples from training distribution.

  Returns:
    A dict of evaluation metrics.
  """
  assert collection.shape[1:] == real.shape

  logging.info(
      f'Generated sample range: [{collection[-1].min()}, {collection[-1].max()}]'
  )
  logging.info(f'Test sample range: [{real.min()}, {real.max()}]')

  if collection[-1].min() < -1. or collection[-1].max() > 1. \
    or real.min() < -1 or real.max() > 1.:
    warnings.warn(
        'Normalize test samples and generated samples to [-1, 1] range.')

  gen_test_points = collection[np.linspace(0,
                                           len(collection) - 1,
                                           20).astype(np.uint32)]

  if FLAGS.compute_final_only:
    gen_test_points = [gen_test_points[-1]]

  random_points = [np.random.randn(*collection[0].shape)]
  real_points = [valid_real]

  if collection.shape[-1] == 2:
    im_buf = plot_utils.scatter_2d(collection[0])
    im = tf.image.decode_png(im_buf.getvalue(), channels=4)
    writer.image('init', im, step=0)

  init = collection[0]
  prd_init = metrics.precision_recall_distribution(real, init)
  prd_perfect = metrics.precision_recall_distribution(real, real)

  for model, test_points in [('baseline', [baseline]),
                             ('ncsn', gen_test_points),
                             ('random', random_points), ('real', real_points)]:
    log_dir = f'{model}/'
    if any([point is None for point in test_points]):
      continue

    for i, samples in enumerate(test_points):
      # Render samples
      if samples.shape[-1] == 2:
        im_buf = plot_utils.scatter_2d(samples)
        im = tf.image.decode_png(im_buf.getvalue(), channels=4)
        writer.image(f'{log_dir}fake', im, step=i)

      # K-means histogram evaluation.
      prd_dist = metrics.precision_recall_distribution(real, samples)
      buf = io.BytesIO()
      metrics.prd.plot([prd_dist, prd_init, prd_perfect],
                       [model, 'noise', 'real'])
      plt.savefig(buf, format='png')
      plt.close()
      buf.seek(0)
      im = tf.image.decode_png(buf.getvalue(), channels=4)
      writer.image(f'{log_dir}prd', im, step=i)

      recall, precision = metrics.prd_f_beta_score(prd_dist)  # F8, F1/8 scores.
      f1 = metrics.f1_score(precision, recall)
      writer.scalar(f'{log_dir}precision', precision, step=i)
      writer.scalar(f'{log_dir}recall', recall, step=i)
      writer.scalar(f'{log_dir}f1', f1, step=i)

      # Nearest neighbor evaluation.
      improved_p, improved_r = metrics.precision_recall(real, samples)
      improved_f1 = metrics.f1_score(improved_p, improved_r)
      writer.scalar(f'{log_dir}improved_precision', improved_p, step=i)
      writer.scalar(f'{log_dir}improved_recall', improved_r, step=i)
      writer.scalar(f'{log_dir}improved_f1', improved_f1, step=i)

      realism_scores = metrics.realism_scores(real, samples)
      realism = realism_scores.mean()
      writer.scalar(f'{log_dir}ipr_realism', realism, step=i)

      ndb_over_k = metrics.ndb_score(real, samples, k=50)
      writer.scalar(f'{log_dir}ndb', ndb_over_k, step=i)

      # Distance evaluation.
      frechet_dist = metrics.frechet_distance(real, samples)
      writer.scalar(f'{log_dir}frechet_distance', frechet_dist, step=i)

      mmd_rbf = metrics.mmd_rbf(real, samples)
      writer.scalar(f'{log_dir}mmd_rbf', mmd_rbf, step=i)

      mmd_polynomial = metrics.mmd_polynomial(real, samples)
      writer.scalar(f'{log_dir}mmd_polynomial', mmd_polynomial, step=i)

  writer.flush()

  stats = {
      'precision': precision,
      'recall': recall,
      'f1': f1,
      'improved_precision': improved_p,
      'improved_recall': improved_r,
      'improved_f1': improved_f1,
      'realism': realism,
      'frechet_dist': frechet_dist,
      'mmd_rbf': mmd_rbf,
      'mmd_polynomial': mmd_polynomial
  }
  return stats


def infill_samples(samples, masks, rng_seed=1):
  rng = jax.random.PRNGKey(rng_seed)
  rng, model_rng = jax.random.split(rng)

  # Create a model with dummy parameters and a dummy optimizer
  model_kwargs = {
      'num_layers': FLAGS.num_layers,
      'num_heads': FLAGS.num_heads,
      'num_mlp_layers': FLAGS.num_mlp_layers,
      'mlp_dims': FLAGS.mlp_dims
  }
  model = train_ncsn.create_model(rng,
                                  samples.shape[1:],
                                  model_kwargs,
                                  batch_size=1,
                                  verbose=True)
  optimizer = train_ncsn.create_optimizer(model, 0)
  ema = train_utils.EMAHelper(mu=0, params=model.params)
  early_stop = train_utils.EarlyStopping()

  # Load learned parameters
  optimizer, ema, early_stop = checkpoints.restore_checkpoint(
      FLAGS.model_dir, (optimizer, ema, early_stop))

  # Create noise schedule
  sigmas = ebm_utils.create_noise_schedule(FLAGS.sigma_begin,
                                           FLAGS.sigma_end,
                                           FLAGS.num_sigmas,
                                           schedule=FLAGS.schedule_type)

  if FLAGS.sampling == 'ald':
    sampling_algorithm = ebm_utils.annealed_langevin_dynamics
  elif FLAGS.sampling == 'cas':
    sampling_algorithm = ebm_utils.consistent_langevin_dynamics
  elif FLAGS.sampling == 'ddpm':
    sampling_algorithm = ebm_utils.diffusion_dynamics
  else:
    raise ValueError(f'Unknown sampling algorithm: {FLAGS.sampling}')

  init_rng, ld_rng = jax.random.split(rng)
  init = jax.random.uniform(key=init_rng, shape=samples.shape)
  generated, collection, ld_metrics = sampling_algorithm(ld_rng,
                                                         optimizer.target,
                                                         sigmas,
                                                         init,
                                                         FLAGS.ld_epsilon,
                                                         FLAGS.ld_steps,
                                                         FLAGS.denoise,
                                                         True,
                                                         infill_samples=samples,
                                                         infill_masks=masks)
  ld_metrics = ebm_utils.collate_sampling_metrics(ld_metrics)

  return generated, collection, ld_metrics


def diffusion_stochastic_encoder(samples, rng_seed=1):
  """Stochastic encoder for diffusion process (DDPM models).
  
  Estimates q(x_T | x_0) given real samples (x_0) and a noise schedule.
  """
  assert FLAGS.sampling == 'ddpm'
  rng = jax.random.PRNGKey(rng_seed)
  betas = ebm_utils.create_noise_schedule(FLAGS.sigma_begin,
                                          FLAGS.sigma_end,
                                          FLAGS.num_sigmas,
                                          schedule=FLAGS.schedule_type)
  T = len(betas)
  alphas = 1. - betas
  alphas_prod = jnp.cumprod(alphas)

  rng, noise_rng = jax.random.split(rng)
  noise = jax.random.normal(key=rng, shape=samples.shape)
  mu = jnp.sqrt(alphas_prod[T]) * samples
  sigma = jnp.sqrt(1 - alphas_prod[T])
  z = mu + sigma * noise
  return z


def diffusion_decoder(z_list, rng_seed=1):
  """Generate samples given a list of latent z as an initialization."""
  assert FLAGS.sampling == 'ddpm'

  rng = jax.random.PRNGKey(rng_seed)
  rng, ld_rng, model_rng = jax.random.split(rng, num=3)
  betas = ebm_utils.create_noise_schedule(FLAGS.sigma_begin,
                                          FLAGS.sigma_end,
                                          FLAGS.num_sigmas,
                                          schedule=FLAGS.schedule_type)

  # Create a model with dummy parameters and a dummy optimizer
  model_kwargs = {
      'num_layers': FLAGS.num_layers,
      'num_heads': FLAGS.num_heads,
      'num_mlp_layers': FLAGS.num_mlp_layers,
      'mlp_dims': FLAGS.mlp_dims
  }
  model = train_ncsn.create_model(model_rng,
                                  z_list[0].shape[1:],
                                  model_kwargs,
                                  batch_size=1,
                                  verbose=True)
  optimizer = train_ncsn.create_optimizer(model, 0)
  ema = train_utils.EMAHelper(mu=0, params=model.params)
  early_stop = train_utils.EarlyStopping()

  # Load learned parameters
  optimizer, ema, early_stop = checkpoints.restore_checkpoint(
      FLAGS.model_dir, (optimizer, ema, early_stop))

  gen, collects, sampling_metrics = [], [], []
  for i, z in enumerate(z_list):
    generated, collection, ld_metrics = ebm_utils.diffusion_dynamics(
        ld_rng, optimizer.target, betas, z, FLAGS.ld_epsilon, FLAGS.ld_steps,
        FLAGS.denoise, False)
    ld_metrics = ebm_utils.collate_sampling_metrics(ld_metrics)
    gen.append(generated)
    collects.append(collection)
    sampling_metrics.append(ld_metrics)
    logging.info('Generated samples %i out of %i', i, len(z_list))

  return gen, collects, sampling_metrics


def generate_samples(sample_shape, num_samples, rng_seed=1):
  """Generate samples using pre-trained score network.

  Args:
    sample_shape: Shape of each sample.
    num_samples: Number of samples to generate.
    rng_seed: Random number generator for sampling.
  """
  rng = jax.random.PRNGKey(rng_seed)
  rng, model_rng = jax.random.split(rng)

  # Create a model with dummy parameters and a dummy optimizer
  model_kwargs = {
      'num_layers': FLAGS.num_layers,
      'num_heads': FLAGS.num_heads,
      'num_mlp_layers': FLAGS.num_mlp_layers,
      'mlp_dims': FLAGS.mlp_dims
  }
  model = train_ncsn.create_model(model_rng,
                                  sample_shape,
                                  model_kwargs,
                                  batch_size=1,
                                  verbose=True)
  optimizer = train_ncsn.create_optimizer(model, 0)
  ema = train_utils.EMAHelper(mu=0, params=model.params)
  early_stop = train_utils.EarlyStopping()

  # Load learned parameters
  optimizer, ema, early_stop = checkpoints.restore_checkpoint(
      FLAGS.model_dir, (optimizer, ema, early_stop))

  # Create noise schedule
  sigmas = ebm_utils.create_noise_schedule(FLAGS.sigma_begin,
                                           FLAGS.sigma_end,
                                           FLAGS.num_sigmas,
                                           schedule=FLAGS.schedule_type)

  rng, sample_rng = jax.random.split(rng)

  t0 = time.time()
  generated, collection, ld_metrics = train_ncsn.sample(
      optimizer.target,
      sigmas,
      sample_rng,
      sample_shape,
      num_samples=num_samples,
      sampling=FLAGS.sampling,
      epsilon=FLAGS.ld_epsilon,
      steps=FLAGS.ld_steps,
      denoise=FLAGS.denoise)
  logging.info('Generated samples in %f seconds', time.time() - t0)

  return generated, collection, ld_metrics


def main(argv):
  del argv  # unused

  logging.info(FLAGS.flags_into_string())
  logging.info('Platform: %s', jax.lib.xla_bridge.get_backend().platform)

  # Make sure TensorFlow does not allocate GPU memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  log_dir = FLAGS.sampling_dir
  writer = tensorboard.SummaryWriter(log_dir)

  pca = data_utils.load(os.path.expanduser(
      FLAGS.pca_ckpt)) if FLAGS.pca_ckpt else None
  slice_idx = data_utils.load(os.path.expanduser(
      FLAGS.slice_ckpt)) if FLAGS.slice_ckpt else None
  dim_weights = data_utils.load(os.path.expanduser(
      FLAGS.dim_weights_ckpt)) if FLAGS.dim_weights_ckpt else None

  train_ds, eval_ds = input_pipeline.get_dataset(
      dataset=FLAGS.dataset,
      data_shape=FLAGS.data_shape,
      problem=FLAGS.problem,
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

  # Generation.
  if FLAGS.infill:  # Infilling.
    if FLAGS.problem == 'toy' and real.shape[-1] == 2:
      samples = np.copy(real)
      samples[:, 1] = 0
      masks = np.zeros(samples.shape)
      masks[:, 0] = 1
    else:
      samples = np.copy(real)

      # Infill middle 16 latents
      idx = list(range(32))
      fixed_idx = idx[:8] + idx[-8:]
      infilled_idx = idx[8:-8]

      samples[:, infilled_idx, :] = 0  # infilled
      masks = np.zeros(samples.shape)
      masks[:, fixed_idx, :] = 1  # hold fixed

    generated, collection, ld_metrics = infill_samples(
        samples, masks, rng_seed=FLAGS.sample_seed)

  elif FLAGS.interpolate:  # Interpolation.
    starts = real
    goals = np.roll(starts, shift=1, axis=0)
    starts_z = diffusion_stochastic_encoder(starts, rng_seed=FLAGS.sample_seed)
    goals_z = diffusion_stochastic_encoder(goals, rng_seed=FLAGS.sample_seed)
    interp_zs = [(1 - alpha) * starts_z + alpha * goals_z
                 for alpha in np.linspace(0., 1., 9)]
    generated, collection, ld_metrics = diffusion_decoder(
        interp_zs, rng_seed=FLAGS.sample_seed)
    generated, collection = np.stack(generated), np.stack(collection)

  else:  # Unconditional generation.
    generated, collection, ld_metrics = generate_samples(
        shape, len(real), rng_seed=FLAGS.sample_seed)

  # Animation (for 2D samples).
  if FLAGS.animate and shape[-1] == 2:
    im_buf = plot_utils.animate_scatter_2d(collection[::2], fps=240)
    with open(os.path.join(log_dir, 'animated.gif'), 'wb') as f:
      f.write(im_buf.getvalue())
      f.close()

  # Dump generated to CPU.
  generated = np.array(generated)
  collection = np.array(collection)

  # Write samples to disk (used for listening).
  if FLAGS.flush:
    # Inverse transform data back to listenable/unnormalized latent space.
    generated_t = input_pipeline.inverse_data_transform(generated,
                                                        FLAGS.normalize, pca,
                                                        train_ds.min,
                                                        train_ds.max, slice_idx,
                                                        dim_weights)
    if not FLAGS.interpolate:
      collection_t = input_pipeline.inverse_data_transform(
          collection, FLAGS.normalize, pca, train_ds.min, train_ds.max,
          slice_idx, dim_weights)
      data_utils.save(collection_t, os.path.join(log_dir,
                                                 'ncsn/collection.pkl'))

    real_t = input_pipeline.inverse_data_transform(real, FLAGS.normalize, pca,
                                                   eval_min, eval_max,
                                                   slice_idx, dim_weights)
    data_utils.save(real_t, os.path.join(log_dir, 'ncsn/real.pkl'))
    data_utils.save(generated_t, os.path.join(log_dir, 'ncsn/generated.pkl'))

  # Run evaluation metrics.
  if FLAGS.compute_metrics:
    train_ncsn.log_langevin_dynamics(ld_metrics, 0, log_dir)
    metrics = evaluate(writer, real, collection, None, real)
    train_utils.log_metrics(metrics, 1, 1)


if __name__ == '__main__':
  app.run(main)
