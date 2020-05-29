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
"""Training utilities."""
import jax
import math
import numpy as np

from absl import logging
from flax import struct
from functools import partial


@struct.dataclass
class EarlyStopping:
  """Early stopping to avoid overfitting during training.
  
  Attributes:
    min_delta: Minimum delta between updates to be considered an
        improvement.
    patience: Number of steps of no improvement before stopping.
    best_metric: Current best metric value.
    patience_count: Number of steps since last improving update.
    should_stop: Whether the training loop should stop to avoid 
        overfitting.
  """
  min_delta: float = 0
  patience: int = 0
  best_metric: float = float('inf')
  patience_count: int = 0
  should_stop: bool = False

  def update(self, metric):
    """Update the state based on metric.
    
    Returns:
      Whether there was an improvement greater than min_delta from
          the previous best_metric and the updated EarlyStop object.
    """

    if math.isinf(
        self.best_metric) or self.best_metric - metric > self.min_delta:
      return True, self.replace(best_metric=metric, patience_count=0)
    else:
      should_stop = self.patience_count >= self.patience or self.should_stop
      return False, self.replace(patience_count=self.patience_count + 1,
                                 should_stop=should_stop)


@struct.dataclass
class EMAHelper:
  """Exponential moving average of model parameters.
  
  Attributes:
    mu: Momentum parameter.
    params: Flax network parameters to update.
  """
  mu: float
  params: any

  @jax.jit
  def update(self, model):
    ema_params = jax.tree_multimap(
        lambda p_ema, p: p_ema * self.mu + p * (1 - self.mu), self.params,
        model.params)
    return self.replace(mu=self.mu, params=ema_params)


def log_metrics(metrics,
                step,
                total_steps,
                epoch=None,
                summary_writer=None,
                verbose=True):
  """Log metrics.

  Args:
    metrics: A dictionary of scalar metrics.
    step: The current step.
    total_steps: The total number of steps.
    epoch: The current epoch.
    summary_writer: A TensorBoard summary writer.
    verbose: Whether to flush values to stdout.
  """
  metrics_str = ''
  for metric in metrics:
    value = metrics[metric]
    if metric == 'lr':
      metrics_str += '{} {:5.4f} | '.format(metric, value)
    else:
      metrics_str += '{} {:5.2f} | '.format(metric, value)

    if summary_writer is not None:
      writer_step = step
      if epoch is not None:
        writer_step = total_steps * epoch + step
      summary_writer.scalar(metric, value, writer_step)

  if epoch is not None:
    epoch_str = '| epoch {:3d} '.format(epoch)
  else:
    epoch_str = ''

  if verbose:
    logging.info('{}| {:5d}/{:5d} steps | {}'.format(epoch_str, step,
                                                     total_steps, metrics_str))


def report_model(model):
  """Log number of trainable parameters and their memory footprint."""
  trainable_params = np.sum(
      [param.size for param in jax.tree_leaves(model.params)])
  footprint_bytes = np.sum([
      param.size * param.dtype.itemsize
      for param in jax.tree_leaves(model.params)
  ])

  logging.info('Number of trainable paramters: {:,}'.format(trainable_params))
  logging.info('Memory footprint: %dMB', footprint_bytes / 2**20)
