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
"""Train iterative refinement networks (NCSN and DDPM)."""
import os
import time

from absl import app
from absl import flags
from absl import logging
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.optimizers
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from flax import nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import lr_schedule

import input_pipeline
import utils.ebm_utils as ebm_utils
import utils.plot_utils as plot_utils
import utils.train_utils as train_utils
from utils.losses import denoising_score_matching_loss, sliced_score_matching_loss, diffusion_loss
import utils.data_utils as data_utils
import models.ncsn as ncsn

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed for network initialization.')

# Training
flags.DEFINE_enum('loss', 'dsm', ['dsm', 'ssm', 'ddpm'], 'Loss function.')
flags.DEFINE_boolean('continuous_noise', True, 'Continuous noise conditioning.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate for optimizer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('epochs', 10, 'Number of training epochs.')
flags.DEFINE_integer('max_steps', None, 'Maximum number of training steps.')

# Training stability
flags.DEFINE_boolean('early_stopping', False,
                     'Use early stopping to prevent overfitting.')
flags.DEFINE_float('grad_clip', 1., 'Max gradient norm for training.')
flags.DEFINE_float('lr_gamma', 0.98, 'Gamma for learning rate scheduler.')
flags.DEFINE_integer('lr_schedule_interval', 10000,
                     'Number of steps between LR changes.')

# Model
flags.DEFINE_string('architecture', 'TransformerDDPM',
                    'Class name of model architecture.')
flags.DEFINE_integer('num_layers', 6, 'Number of encoder layers.')
flags.DEFINE_integer('num_heads', 8, 'Number of attention heads.')
flags.DEFINE_integer('num_mlp_layers', 2, 'Number of MLP layers.')
flags.DEFINE_integer('mlp_dims', 2048, 'Number of channels per MLP layer.')

# Noise schedule
flags.DEFINE_float('sigma_begin', 1.,
                   'Starting variance for noise schedule.')  # Technique 1
flags.DEFINE_float('sigma_end', 1e-2, 'Ending variance for noise schedule.')
flags.DEFINE_enum('schedule_type', 'geometric',
                  ['geometric', 'linear', 'fibonacci'],
                  'Noise schedule configuration.')
flags.DEFINE_integer(
    'num_sigmas', 15,
    'Number of sigma values (L) in noise schedule.')  # Technique 2

# Langevin dynamics (NCSN only)
flags.DEFINE_integer('ld_steps', 100,
                     'Number of steps for annealed Langevin dynamics.')
flags.DEFINE_float('ld_epsilon', 2e-6,
                   'Step size for annealed Langevin dynamics.')  # Technique 4

# Sampling
flags.DEFINE_enum('sampling', 'ald', ['ald', 'cas', 'ddpm'],
                  'Sampling algorithm to use.')
flags.DEFINE_boolean('ema', True,
                     'Exponential moving average smoothing.')  # Technique 5
flags.DEFINE_float('mu', 0.999, 'Momentum parameter for EMA.')
flags.DEFINE_boolean(
    'denoise', True,
    'Add additional denoising step during sampling (Song et al., 2020).')

# Data
flags.DEFINE_list('data_shape', [
    2,
], 'Shape of data.')
flags.DEFINE_enum('problem', 'toy', ['toy', 'mnist', 'vae'],
                  'Problem to solve.')
flags.DEFINE_string(
    'dataset', './output/mix2d',
    'Path to directory containing data as train/eval tfrecord files.')
flags.DEFINE_string('pca_ckpt', '', 'PCA transform.')
flags.DEFINE_string('slice_ckpt', '', 'Slice transform.')
flags.DEFINE_string('dim_weights_ckpt', '', 'Dimension scale transform.')
flags.DEFINE_boolean('normalize', True, 'Normalize dataset to [-1, 1].')

# Logging, checkpointing, and evaluation
flags.DEFINE_integer('logging_freq', 100, 'Logging frequency.')
flags.DEFINE_integer('snapshot_freq', 5000,
                     'Evaluation and checkpoint frequency.')
flags.DEFINE_boolean('snapshot_sampling', True,
                     'Sample from score network during evaluation.')
flags.DEFINE_integer('eval_samples', 3000, 'Number of samples to generate.')
flags.DEFINE_integer('checkpoints_to_keep', 50,
                     'Number of checkpoints to keep.')
flags.DEFINE_boolean('save_ckpt', True,
                     'Save model checkpoints at each evaluation step.')
flags.DEFINE_string('model_dir', './save/ncsn',
                    'Directory to store model data.')
flags.DEFINE_boolean('verbose', True, 'Toggle logging to stdout.')


def log_samples(writer,
                step,
                init,
                real,
                fake,
                display_fn=None,
                display_samples=10,
                flush=False,
                output_dir=None):
  """Log data samples.
  
  Args:
    writer: TensorBoard log writer.
    step: Current log step.
    init: Initial samples.
    real: Real samples.
    fake: Generated samples.
    display_fn: Function to generate image buffer.
    display_samples: Number of samples to post to TensorBoard. 
    flush: Flush samples to disk.
    output_dir: Directory where samples will be saved.
  """
  assert init.shape == real.shape == fake.shape

  for category, samples in (('init', init), ('real', real), ('fake', fake)):
    if display_fn is not None:
      buf = display_fn(samples[:display_samples])
      im = tf.image.decode_png(buf.getvalue(), channels=4)
      writer.image(category, im, step=step)

    if flush:
      data_utils.save(
          samples, os.path.join(output_dir, f'samples/{category}/{step}.pkl'))


def log_langevin_dynamics(ld_metrics, step, output_dir):
  """Log statistics for each step of Langevin dynamics sampling.

  Args:
    ld_metrics: Metrics collected for each noise level with shape (num_sigmas, T).
    step: Training step/epoch when sampling took place.
    output_dir: Output directory.
  """
  writer = tensorboard.SummaryWriter(
      os.path.join(output_dir, f'sampling_epoch{step}'))
  for i, sigma_metrics in enumerate(ld_metrics):
    for j, metric in enumerate(sigma_metrics):
      train_utils.log_metrics(metric,
                              j,
                              len(sigma_metrics),
                              epoch=i,
                              summary_writer=writer,
                              verbose=False)
  writer.flush()


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create(model)
  return optimizer


def create_model(rng, input_shape, model_kwargs, batch_size=32, verbose=False):
  clazz = getattr(ncsn, FLAGS.architecture)
  module = clazz.partial(**model_kwargs)
  output, initial_params = module.init_by_shape(
      rng, [((batch_size, *input_shape), jnp.float32),
            ((batch_size, *([1] * len(input_shape))), jnp.float32)])
  model = nn.Model(module, initial_params)

  if verbose:
    train_utils.report_model(model)
  return model


@partial(jax.jit, static_argnums=(0,))
def eval_step(objective, batch, model, sigmas, rng):
  """A single evaluation step.

  Args:
    objective; Objective function used for evaluation.
    batch: A batch of inputs.
    model: The model to be used for this evaluation step.
    sigmas: The noise scheduled used to train the model.
    rng: Random number generator for noise selection.
  
  Returns:
    loss: The summed loss on this batch.
  """
  loss = objective(batch, model, sigmas, rng, FLAGS.continuous_noise, 'sum')
  return loss


def evaluate(dataset, model, sigmas, rng):
  """Evaluates the model on a dataset.
  
  Args:
    dataset: A dataset to be used for the evaluation. Typically valid or test.
    model: A model to be evaluated.
    sigmas: The noise scheduled used to train the model.
    rng: Random number generator for noise selection.
  
  Returns:
    A dict with the evaluation results.
  """
  if FLAGS.loss == 'dsm':
    objective = denoising_score_matching_loss
  elif FLAGS.loss == 'ssm':
    objective = sliced_score_matching_loss
  elif FLAGS.loss == 'ddpm':
    objective = diffusion_loss
  else:
    raise ValueError(f'Unsupported objective {FLAGS.loss}')

  count = 0
  total_loss = 0.

  for inputs in tfds.as_numpy(dataset):
    count += inputs.shape[0]
    rng, eval_rng = jax.random.split(rng)
    loss = eval_step(objective, inputs, model, sigmas, eval_rng)
    total_loss += loss.item()

  loss = total_loss / count
  metrics = {'loss': loss}

  return metrics


@partial(jax.jit, static_argnums=(0,))
def train_step(objective, batch, optimizer, sigmas, rng, learning_rate):
  """Single optimized training step.

  Args:
    objective: Objective used for training.
    batch: A batch of inputs.
    optimizer: The optimizer to use to update the weights.
    sigmas: The noise schedule used to train the model.
    rng: Random number generator for noise selection.
    learning_rate: Current learning rate.

  Returns:
    optimizer: The optimizer in its new state.
    train_metrics: A dict with training statistics for the step.
  """

  def loss_fn(model):
    loss = objective(batch, model, sigmas, rng, FLAGS.continuous_noise, 'mean')
    train_metrics = {'loss': loss}
    return loss, train_metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, train_metrics), grad = grad_fn(optimizer.target)
  grad = jax.experimental.optimizers.clip_grads(grad, FLAGS.grad_clip)
  train_metrics['grad'] = jax.experimental.optimizers.l2_norm(grad)
  train_metrics['lr'] = learning_rate
  optimizer = optimizer.apply_gradient(grad, learning_rate=learning_rate)
  return optimizer, train_metrics


def train(train_batches, valid_batches, sigmas, output_dir=None, verbose=True):
  """Training loop.

  Args:
    train_batches: Training batches from tf.data.Dataset.
    valid_batches: Validation batches from tf.data.Dataset.
    sigmas: Noise schedule.
    output_dir: Output directory for checkpoints, logs, and samples.
    verbose: Logging verbosity.
  
  Returns:
    An optimizer object with final state.
  """
  train_writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'train'))
  eval_writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'eval'))

  pca = data_utils.load(os.path.expanduser(
      FLAGS.pca_ckpt)) if FLAGS.pca_ckpt else None
  slice_idx = data_utils.load(os.path.expanduser(
      FLAGS.slice_ckpt)) if FLAGS.slice_ckpt else None
  dim_weights = data_utils.load(os.path.expanduser(
      FLAGS.dim_weights_ckpt)) if FLAGS.dim_weights_ckpt else None

  tfds_batch = valid_batches.take(1)
  tfds_batch = list(valid_batches.as_numpy_iterator())[0]
  batch_size, *input_shape = tfds_batch.shape

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng, model_rng, sample_rng = jax.random.split(rng, num=3)

  model_kwargs = {
      'num_layers': FLAGS.num_layers,
      'num_heads': FLAGS.num_heads,
      'num_mlp_layers': FLAGS.num_mlp_layers,
      'mlp_dims': FLAGS.mlp_dims
  }
  model = create_model(model_rng,
                       input_shape,
                       model_kwargs,
                       batch_size,
                       verbose=verbose)
  optimizer = create_optimizer(model, FLAGS.learning_rate)
  early_stop = train_utils.EarlyStopping(patience=1)

  # Sampling
  ema = train_utils.EMAHelper(mu=FLAGS.mu, params=model.params)
  scorenet = create_model(sample_rng, input_shape, model_kwargs, batch_size)

  # Learning rate schedule
  lr_step_schedule = [(i, FLAGS.lr_gamma**i) for i in range(1000)]
  lr_scheduler = lr_schedule.create_stepped_learning_rate_schedule(
      FLAGS.learning_rate, FLAGS.lr_schedule_interval, lr_step_schedule)

  # Objective
  if FLAGS.loss == 'dsm':
    objective = denoising_score_matching_loss
  elif FLAGS.loss == 'ssm':
    objective = sliced_score_matching_loss
  elif FLAGS.loss == 'ddpm':
    objective = diffusion_loss
  else:
    raise ValueError(f'Unsupported objective {FLAGS.loss}')

  sampling_step = -1
  for epoch in range(FLAGS.epochs):
    start_time = time.time()
    for step, batch in enumerate(tfds.as_numpy(train_batches)):
      rng, train_rng = jax.random.split(rng)
      global_step = step + epoch * train_batches.examples
      optimizer, train_metrics = train_step(objective, batch, optimizer,
                                            sigmas, train_rng,
                                            lr_scheduler(global_step))

      if FLAGS.ema:
        ema = ema.update(optimizer.target)

      if step % FLAGS.logging_freq == 0:
        elapsed = time.time() - start_time
        batch_per_sec = (step + 1) / elapsed
        ms_per_batch = elapsed * 1000 / (step + 1)
        train_metrics['batch/s'] = batch_per_sec
        train_metrics['ms/batch'] = ms_per_batch
        train_utils.log_metrics(train_metrics,
                                step,
                                train_batches.examples,
                                epoch=epoch,
                                summary_writer=train_writer,
                                verbose=verbose)

      if (step % FLAGS.snapshot_freq == 0 and
          step > 0) or step == train_batches.examples - 1:

        sampling_step += 1

        rng, eval_rng = jax.random.split(rng)
        eval_metrics = evaluate(valid_batches, optimizer.target, sigmas,
                                eval_rng)
        train_utils.log_metrics(eval_metrics,
                                global_step,
                                train_batches.examples * FLAGS.epochs,
                                summary_writer=eval_writer,
                                verbose=verbose)
        improved, early_stop = early_stop.update(eval_metrics['loss'])

        if (not FLAGS.early_stopping and FLAGS.save_ckpt) or \
          (FLAGS.early_stopping and improved and FLAGS.save_ckpt):
          checkpoints.save_checkpoint(output_dir, (optimizer, ema, early_stop),
                                      sampling_step,
                                      keep=FLAGS.checkpoints_to_keep)

        if FLAGS.early_stopping and early_stop.should_stop:
          logging.info('EARLY STOP: Ended training after %s epochs.', epoch + 1)
          return

        if FLAGS.snapshot_sampling:
          scorenet = scorenet.replace(
              params=ema.params if FLAGS.ema else optimizer.target.params)
          rng, sample_rng = jax.random.split(rng)
          generated, collection, ld_metrics = sample(
              scorenet,
              sigmas,
              rng,
              input_shape,
              num_samples=FLAGS.eval_samples,
              sampling=FLAGS.sampling,
              epsilon=FLAGS.ld_epsilon,
              steps=FLAGS.ld_steps,
              denoise=FLAGS.denoise)
          log_langevin_dynamics(ld_metrics, sampling_step, output_dir)

          init = collection[0]
          real = valid_batches.unbatch().shuffle(8 * batch_size).take(
              FLAGS.eval_samples)
          real = np.stack([ex for ex in tfds.as_numpy(real)])
          real = input_pipeline.inverse_data_transform(real, FLAGS.normalize,
                                                       pca, valid_batches.min,
                                                       valid_batches.max,
                                                       slice_idx, dim_weights)
          init = input_pipeline.inverse_data_transform(init, FLAGS.normalize,
                                                       pca, train_batches.min,
                                                       train_batches.max,
                                                       slice_idx, dim_weights)
          generated = input_pipeline.inverse_data_transform(
              generated, FLAGS.normalize, pca, train_batches.min,
              train_batches.max, slice_idx, dim_weights)

          if FLAGS.problem == 'toy':
            init = init.reshape(-1, 2)
            real = real.reshape(-1, 2)
            generated = generated.reshape(-1, 2)

            display_fn = partial(plot_utils.scatter_2d, scale=8)
            log_samples(eval_writer,
                        global_step,
                        init,
                        real,
                        generated,
                        display_fn=display_fn,
                        display_samples=len(generated),
                        flush=False,
                        output_dir=output_dir)

            # Draw gradient field
            if len(input_shape) == 1 and FLAGS.sampling != 'ddpm':
              for sigma in sigmas:
                score_buf = plot_utils.score_field_2d(optimizer.target,
                                                      sigma=sigma,
                                                      scale=8)
                score_im = tf.image.decode_png(score_buf.getvalue(), channels=4)
                eval_writer.image('score_sigma={:.4f}'.format(sigma),
                                  score_im,
                                  step=global_step)

          elif FLAGS.problem == 'mnist':
            display_fn = partial(plot_utils.image_tiles, shape=(28, 28))
            log_samples(eval_writer,
                        global_step,
                        init,
                        real,
                        generated,
                        display_fn=display_fn,
                        display_samples=10,
                        flush=False,
                        output_dir=output_dir)

          elif FLAGS.problem == 'vae':
            display_fn = partial(plot_utils.image_tiles, shape=(16, 32))
            log_samples(eval_writer,
                        global_step,
                        init,
                        real,
                        generated,
                        display_fn=display_fn,
                        display_samples=10,
                        flush=True,
                        output_dir=output_dir)

        train_writer.flush()
        eval_writer.flush()

      # Early termination of training loop.
      if FLAGS.max_steps is not None and \
        global_step >= FLAGS.max_steps:
        return optimizer

  return optimizer


def sample(scorenet,
           sigmas,
           rng,
           sample_shape,
           num_samples=2400,
           sampling='ald',
           epsilon=1e-3,
           steps=100,
           denoise=True):
  """Generate samples via Langevin dynamics.
  
  Args:
    scorenet: Score model to use for sampling.
    sigmas: Noise schedule.
    rng: Random number generator for noise.
    sample_shape: Shape of each individual sample.
    num_samples: The number of samples to generate.
    sampling: Sampling algorithm to use.
    epsilon: Step size for Langevin dynamics.
    steps: Number of sampling steps.
    denoise: Apply an additional denoising step to yield
        the expected denoised sampled (EDS).
 
  Returns:
    generated: An array of generated samples.
    collection: An array with the samples at each step of sampling.
    ld_metrics: Sampling statistics for each step.
  """
  if sampling == 'ald':
    sampling_algorithm = ebm_utils.annealed_langevin_dynamics
  elif sampling == 'cas':
    sampling_algorithm = ebm_utils.consistent_langevin_dynamics
  elif sampling == 'ddpm':
    sampling_algorithm = ebm_utils.diffusion_dynamics
  else:
    raise ValueError(f'Unknown sampling algorithm: {sampling}')

  init_rng, ld_rng = jax.random.split(rng)

  # Initial state has mean=0, var=1.
  if sampling == 'ddpm':
    init = jax.random.normal(key=init_rng, shape=(num_samples, *sample_shape))
  else:
    rho = jnp.sqrt(12) / 2
    init = jax.random.uniform(key=init_rng,
                              shape=(num_samples, *sample_shape),
                              minval=-rho,
                              maxval=rho)

  generated, collection, ld_metrics = sampling_algorithm(
      ld_rng, scorenet, sigmas, init, epsilon, steps, denoise, False)
  ld_metrics = ebm_utils.collate_sampling_metrics(ld_metrics)
  return generated, collection, ld_metrics


def main(argv):
  del argv  # unused

  logging.info(FLAGS.flags_into_string())
  logging.info('Platform: %s', jax.lib.xla_bridge.get_backend().platform)

  # Make sure TensorFlow does not allocate GPU memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  train_ds, eval_ds = input_pipeline.get_dataset(
      dataset=FLAGS.dataset,
      data_shape=FLAGS.data_shape,
      problem=FLAGS.problem,
      batch_size=FLAGS.batch_size,
      normalize=FLAGS.normalize,
      pca_ckpt=FLAGS.pca_ckpt,
      slice_ckpt=FLAGS.slice_ckpt,
      dim_weights_ckpt=FLAGS.dim_weights_ckpt)

  # Noise schedule
  noise_schedule = ebm_utils.create_noise_schedule(FLAGS.sigma_begin,
                                                   FLAGS.sigma_end,
                                                   FLAGS.num_sigmas,
                                                   schedule=FLAGS.schedule_type)

  train(train_batches=train_ds,
        valid_batches=eval_ds,
        sigmas=noise_schedule,
        output_dir=FLAGS.model_dir,
        verbose=FLAGS.verbose)


if __name__ == '__main__':
  app.run(main)
