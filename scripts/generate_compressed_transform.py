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
"""Fit a compression transform on chunk embedding data."""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

sys.path.append("{}/../".format(os.path.dirname(os.path.abspath(__file__))))
import utils.data_utils as data_utils

FLAGS = flags.FLAGS

# Input dataset.
flags.DEFINE_string('dataset', None, 'Path to input data (TFRecord format).')
flags.DEFINE_list('data_shape', [
    512,
], 'Shape of data.')
flags.DEFINE_integer('samples', int(2e6), 'Number of data samples to train on.')

# PCA.
flags.DEFINE_integer('dims', 200, 'Rank of compressed embedding.')
flags.DEFINE_enum('mode', 'pca', ['slice', 'pca'], 'Data generation mode.')
flags.DEFINE_boolean('normalize', True,
                     'Add normalization to transform pipeline.')
flags.DEFINE_string('ckpt', './output/pca.pkl',
                    'Path to file containing transform checkpoint.')

# Visualization.
flags.DEFINE_boolean('compute_dims', False,
                     'Compute the expected number of dimensions required.')
flags.DEFINE_float('var_threshold', .85,
                   'Explained variance threshold for computing dimensions.')


class SliceTransform(object):
  """Slice transform."""

  def __init__(self, component_idx, fill_idx, fill):
    self.orig_dims = len(component_idx) + len(fill)
    self.dims = len(component_idx)

    self.component_idx = component_idx  # important components
    self.fill_idx = fill_idx  # unimportant components
    self.fill = fill  # values to fill the unimportant components with

  def transform(self, x):
    # x.shape = (batch_size, self.orig_dims)
    compressed = x[:, self.component_idx]
    assert compressed.shape == (x.shape[0], self.dims)
    return compressed

  def inverse_transform(self, x):
    # x.shape = (batch_size, self.dims)
    recon = np.zeros((x.shape[0], self.orig_dims))
    recon[:, self.component_idx] = x
    recon[:, self.fill_idx] = self.fill
    assert recon.shape == (x.shape[0], self.orig_dims)
    return recon


def main(argv):
  del argv  # unused

  tf.config.experimental.set_visible_devices([], 'GPU')

  shape = tuple(map(int, FLAGS.data_shape))
  train_ds = data_utils.get_tf_record_dataset(
      file_pattern=f'{FLAGS.dataset}/train-*.tfrecord',
      shape=shape,
      batch_size=2048,
      shuffle=True)
  train_ds = train_ds.take(FLAGS.samples)
  train_ds = np.stack([ex['inputs'] for ex in tfds.as_numpy(train_ds)])

  if len(shape) == 2:
    assert shape[0] == 3
    means = train_ds[:, 1, :]
    sigmas = train_ds[:, 2, :]
    avg_mean = means.mean(0)
    avg_sigma = sigmas.mean(0)
    weights = 1 / avg_sigma**2
    # idx = np.where(avg_sigma < 0.98)[0]
    logging.info('Creating slice transform weights.')
    data_utils.save(weights, os.path.expanduser(FLAGS.ckpt))
    return -1

  singular_values = np.linalg.svd(train_ds,
                                  full_matrices=False,
                                  compute_uv=False)
  variance_gain = singular_values.cumsum() / singular_values.sum()

  if FLAGS.compute_dims:
    dims = np.where(variance_gain >= FLAGS.var_threshold)[0][0]
    variance = variance_gain[dims]
    plt.text(0, variance + 0.05, '{:.3f}'.format(variance), rotation=0)
    plt.text(dims + 0.1, 0.2, dims, rotation=0)
    plt.axhline(y=variance, color='r', linestyle='--')
    plt.axvline(x=dims, color='r', linestyle='--')
    plt.plot(variance_gain)
    plt.show()

    logging.info('Explained variance ratio: %f, Rank: %i.', variance, dims)

  else:
    logging.info('Creating %s transform with rank %i.', FLAGS.mode, FLAGS.dims)

    operations = []
    if FLAGS.normalize:
      logging.info('Adding normalization.')
      operations.append(('scaling', StandardScaler()))
    if FLAGS.mode == 'pca':
      operations.append(('pca', PCA(n_components=FLAGS.dims)))
    else:
      raise ValueError(f'Unsupported mode: {FLAGS.mode}')

    logging.info('Fitting transform.')
    pipeline = Pipeline(operations)
    pipeline = pipeline.fit(train_ds)
    data_utils.save(pipeline, os.path.expanduser(FLAGS.ckpt))


if __name__ == '__main__':
  app.run(main)
