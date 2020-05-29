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
"""Train Transformer-based continuous language model."""
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
import utils.train_utils as train_utils
import utils.data_utils as data_utils
import models.autoregressive as ar
from utils.losses import reduce_fn

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed for network initialization.')

# Training
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate for optimizer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('epochs', 1000, 'Number of training epochs.')
flags.DEFINE_integer('max_steps', 100000, 'Maximum number of training steps.')

# Training stability
flags.DEFINE_boolean('early_stopping', False,
                     'Use early stopping to prevent overfitting.')
flags.DEFINE_float('grad_clip', 1., 'Max gradient norm for training.')
flags.DEFINE_float('lr_gamma', 0.98, 'Gamma for learning rate scheduler.')
flags.DEFINE_integer('lr_schedule_interval', 4000,
                     'Number of steps between LR changes.')
flags.DEFINE_float('lr_warmup', 0, 'Learning rate warmup (epochs).')

# Model
flags.DEFINE_string('architecture', 'TransformerMDN',
                    'Class name of model architecture.')
flags.DEFINE_integer('mdn_components', 100, 'Number of mixtures.')
flags.DEFINE_integer('num_heads', 8, 'Number of attention heads.')
flags.DEFINE_integer('num_layers', 6, 'Number of encoder layers.')
flags.DEFINE_integer('num_mlp_layers', 2, 'Number of output MLP layers.')
flags.DEFINE_integer('mlp_dims', 2048, 'Number of channels per MLP layer.')

# Data
flags.DEFINE_list('data_shape', [32, 512], 'Shape of data.')
flags.DEFINE_string(
    'dataset', './output/mel-32step-512',
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
flags.DEFINE_string('model_dir', './save/mdn', 'Directory to store model data.')
flags.DEFINE_boolean('verbose', True, 'Toggle logging to stdout.')


def mdn_loss(pi, mu, log_sigma, x, reduction='mean'):
  """Mixture density loss.
  
  Args:
    pi: Unnormalized component mixture distribution.
    mu: Mean vectors.
    log_sigma: Log standard deviation vectors.
    reduction: Type of reduction to apply to loss.

  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `data`;
    otherwise, it is scalar.
  """
  channels = x.shape[-1]
  mdn_k = pi.shape[-1]
  out_pi = pi.reshape(-1, mdn_k)
  out_mu = mu.reshape(-1, channels * mdn_k)
  out_log_sigma = log_sigma.reshape(-1, channels * mdn_k)

  # Create mixture distribution
  mix_dist = tfd.Categorical(logits=out_pi)

  # Create component distribution
  mus = out_mu.reshape(-1, mdn_k, channels)
  log_sigmas = out_log_sigma.reshape(-1, mdn_k, channels)
  sigmas = jnp.exp(log_sigmas)
  component_dist = tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigmas)

  # Compute loss
  mixture = tfd.MixtureSameFamily(mixture_distribution=mix_dist,
                                  components_distribution=component_dist)
  x = x.reshape(-1, channels)
  loss = -1 * mixture.log_prob(x)
  return reduce_fn(loss, reduction)


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create(model)
  return optimizer


def create_model(rng, input_shape, model_kwargs, batch_size=32, verbose=False):
  clazz = getattr(ar, FLAGS.architecture)
  module = clazz.partial(**model_kwargs)
  output, initial_params = module.init_by_shape(
      rng, [((batch_size, *input_shape), jnp.float32)])
  model = nn.Model(module, initial_params)

  if verbose:
    train_utils.report_model(model)
  return model


@jax.jit
def eval_step(batch, model):
  """A single evaluation step.

  Args:
    batch: A batch of inputs.
    model: The model to be used for this evaluation step.
  
  Returns:
    loss: The summed loss on this batch.
    examples: Number of examples in this batch.
  """
  pi, mu, log_sigma = model(batch)
  loss = mdn_loss(pi, mu, log_sigma, batch, 'none')
  return loss.sum(), loss.shape[0]


def evaluate(dataset, model):
  """Evaluates the model on a dataset.
  
  Args:
    dataset: A dataset to be used for the evaluation. Typically valid or test.
    model: A model to be evaluated.
  
  Returns:
    A dict with the evaluation results.
  """
  count = 0
  total_loss = 0.

  for inputs in tfds.as_numpy(dataset):
    loss, examples = eval_step(inputs, model)
    count += examples
    total_loss += loss.item()

  loss = total_loss / count
  metrics = {'loss': loss}

  return metrics


@jax.jit
def train_step(batch, optimizer, learning_rate):
  """Single optimized training step.

  Args:
    batch: A batch of inputs.
    optimizer: The optimizer to use to update the weights.
    learning_rate: Current learning rate.

  Returns:
    optimizer: The optimizer in its new state.
    train_metrics: A dict with training statistics for the step.
  """

  def loss_fn(model):
    pi, mu, log_sigma = model(batch)
    loss = mdn_loss(pi, mu, log_sigma, batch, 'mean')
    train_metrics = {'loss': loss}
    return loss, train_metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, train_metrics), grad = grad_fn(optimizer.target)
  grad = jax.experimental.optimizers.clip_grads(grad, FLAGS.grad_clip)
  train_metrics['grad'] = jax.experimental.optimizers.l2_norm(grad)
  train_metrics['lr'] = learning_rate
  optimizer = optimizer.apply_gradient(grad, learning_rate=learning_rate)
  return optimizer, train_metrics


def train(train_batches, valid_batches, output_dir=None, verbose=True):
  """Training loop.

  Args:
    train_batches: Training batches from tf.data.Dataset.
    valid_batches: Validation batches from tf.data.Dataset.
    output_dir: Output directory for checkpoints, logs, and samples.
    verbose: Logging verbosity.
  
  Returns:
    An optimizer object with final state.
  """
  train_writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'train'))
  eval_writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'eval'))

  tfds_batch = valid_batches.take(1)
  tfds_batch = list(valid_batches.as_numpy_iterator())[0]
  batch_size, *input_shape = tfds_batch.shape

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng, model_rng = jax.random.split(rng)

  lm_kwargs = {
      'num_layers': FLAGS.num_layers,
      'num_heads': FLAGS.num_heads,
      'mdn_mixtures': FLAGS.mdn_components,
      'num_mlp_layers': FLAGS.num_mlp_layers,
      'mlp_dims': FLAGS.mlp_dims
  }
  model = create_model(model_rng,
                       input_shape,
                       lm_kwargs,
                       batch_size,
                       verbose=verbose)
  optimizer = create_optimizer(model, FLAGS.learning_rate)
  early_stop = train_utils.EarlyStopping(patience=1)

  # Learning rate schedule
  lr_step_schedule = [(i, FLAGS.lr_gamma**i) for i in range(1000)]
  lr_scheduler = lr_schedule.create_stepped_learning_rate_schedule(
      FLAGS.learning_rate,
      FLAGS.lr_schedule_interval,
      lr_step_schedule,
      warmup_length=FLAGS.lr_warmup)

  sampling_step = -1
  for epoch in range(FLAGS.epochs):
    start_time = time.time()
    for step, batch in enumerate(tfds.as_numpy(train_batches)):
      global_step = step + epoch * train_batches.examples
      optimizer, train_metrics = train_step(batch, optimizer,
                                            lr_scheduler(global_step))

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

        eval_metrics = evaluate(valid_batches, optimizer.target)
        train_utils.log_metrics(eval_metrics,
                                global_step,
                                train_batches.examples * FLAGS.epochs,
                                summary_writer=eval_writer,
                                verbose=verbose)
        improved, early_stop = early_stop.update(eval_metrics['loss'])

        if (not FLAGS.early_stopping and FLAGS.save_ckpt) or \
          (FLAGS.early_stopping and improved and FLAGS.save_ckpt):
          checkpoints.save_checkpoint(output_dir, (optimizer, early_stop),
                                      sampling_step,
                                      keep=FLAGS.checkpoints_to_keep)

        if FLAGS.early_stopping and early_stop.should_stop:
          logging.info('EARLY STOP: Ended training after %s epochs.', epoch + 1)
          return

        train_writer.flush()
        eval_writer.flush()

      # Early termination of training loop.
      if FLAGS.max_steps is not None and \
        global_step >= FLAGS.max_steps:
        return optimizer

  return optimizer


def main(argv):
  del argv  # unused

  logging.info(FLAGS.flags_into_string())
  logging.info('Platform: %s', jax.lib.xla_bridge.get_backend().platform)

  # Make sure TensorFlow does not allocate GPU memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  train_ds, eval_ds = input_pipeline.get_dataset(
      dataset=FLAGS.dataset,
      data_shape=FLAGS.data_shape,
      problem='vae',
      batch_size=FLAGS.batch_size,
      normalize=FLAGS.normalize,
      pca_ckpt=FLAGS.pca_ckpt,
      slice_ckpt=FLAGS.slice_ckpt,
      dim_weights_ckpt=FLAGS.dim_weights_ckpt)

  train(train_batches=train_ds,
        valid_batches=eval_ds,
        output_dir=FLAGS.model_dir,
        verbose=FLAGS.verbose)


if __name__ == '__main__':
  app.run(main)
