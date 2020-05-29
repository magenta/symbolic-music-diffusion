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
"""Input data pipeline."""
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import logging
from functools import partial
import utils.data_utils as data_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE


def deconstruct_dict(batch_dict, problem):
  key = 'image' if problem == 'mnist' else 'inputs'
  return batch_dict[key]


def normalize_dataset(batch, data_min, data_max):
  """Normalize dataset to range [-1, 1]."""
  batch = (batch - data_min) / (data_max - data_min)
  batch = 2. * batch - 1.
  return batch


def slice_transform(batch, problem='vae', slice_idx=None, dim_weights=None):
  if dim_weights is not None:
    batch = batch * dim_weights
  if slice_idx is not None:
    batch = tf.gather(batch, slice_idx, axis=-1)
  return batch


def data_transform(batch, problem='vae', pca=None):
  """Data transform.

  Args:
    batch: A batch of data samples.
    pca: PCA transform object.
  
  Returns:
    Transformed batch array.
  """
  if problem == 'mnist':
    batch = tf.reshape(batch, (batch.shape[0], -1))
    batch = tf.cast(batch, tf.float32) / 255.
    batch = 2. * batch - 1.

  if pca is not None:
    if batch.ndim > 2:
      init_shape = batch.shape
      batch = batch.reshape(batch.shape[0], -1)
      batch = pca.transform(batch)
      batch = batch.reshape(*init_shape)
    else:
      batch = pca.transform(batch)

  return batch


def inverse_data_transform(batch,
                           normalize=True,
                           pca=None,
                           data_min=0.,
                           data_max=1.,
                           slice_idx=None,
                           dim_weights=None,
                           out_channels=512):
  """Inverse data transform.

  Args:
    batch: Transformed batch array.
    pca: PCA transform object.

  Returns:
    Original batch array.
  """
  if normalize:
    batch = (batch + 1.) / 2.
    batch = (data_max - data_min) * batch + data_min

  if pca is not None:
    batch = pca.inverse_transform(batch)

  if slice_idx is not None:
    transformed = np.random.randn(*batch.shape[:-1], out_channels)
    transformed[..., slice_idx] = batch
    batch = transformed

  if dim_weights is not None:
    batch = batch / dim_weights

  return batch


def get_dataset(dataset='',
                data_shape=(2,),
                problem='vae',
                batch_size=128,
                normalize=True,
                pca_ckpt='',
                slice_ckpt='',
                dim_weights_ckpt='',
                include_cardinality=True):
  if problem == 'mnist':
    train_ds = tfds.load('mnist', split='train', shuffle_files=True)
    eval_ds = tfds.load('mnist', split='test', shuffle_files=True)
  elif problem in ['vae', 'toy', 'tokens']:
    shape = tuple(map(int, data_shape))
    tokens = problem == 'tokens'
    train_ds = data_utils.get_tf_record_dataset(
        file_pattern=f'{dataset}/train-*.tfrecord',
        shape=shape,
        batch_size=batch_size,
        shuffle=True,
        tokens=tokens)
    eval_ds = data_utils.get_tf_record_dataset(
        file_pattern=f'{dataset}/eval-*.tfrecord',
        shape=shape,
        batch_size=batch_size,
        shuffle=True,
        tokens=tokens)
  else:
    raise ValueError(f'Unknown problem type: {problem}')

  # Dataset loading and transformation (PCA, Slice).
  pca = data_utils.load(os.path.expanduser(pca_ckpt)) if pca_ckpt else None
  slice_idx = data_utils.load(
      os.path.expanduser(slice_ckpt)) if slice_ckpt else None
  dim_weights = data_utils.load(
      os.path.expanduser(dim_weights_ckpt)) if dim_weights_ckpt else None

  # Batch.
  train_ds = train_ds.batch(batch_size, drop_remainder=True)
  eval_ds = eval_ds.batch(batch_size, drop_remainder=True)

  train_ds = train_ds.map(partial(deconstruct_dict, problem=problem),
                          num_parallel_calls=AUTOTUNE)
  eval_ds = eval_ds.map(partial(deconstruct_dict, problem=problem),
                        num_parallel_calls=AUTOTUNE)

  # PCA transform
  if problem != 'tokens':
    train_ds = train_ds.map(lambda example: tf.py_function(
        partial(data_transform, problem=problem, pca=pca), [example], tf.float32
    ),
                            num_parallel_calls=AUTOTUNE)
    eval_ds = eval_ds.map(lambda example: tf.py_function(
        partial(data_transform, problem=problem, pca=pca), [example], tf.float32
    ),
                          num_parallel_calls=AUTOTUNE)

    # Slice + weight transform
    train_ds = train_ds.map(partial(slice_transform,
                                    problem=problem,
                                    slice_idx=slice_idx,
                                    dim_weights=dim_weights),
                            num_parallel_calls=AUTOTUNE)
    eval_ds = eval_ds.map(partial(slice_transform,
                                  problem=problem,
                                  slice_idx=slice_idx,
                                  dim_weights=dim_weights),
                          num_parallel_calls=AUTOTUNE)

  # Dataset normalization.
  train_min, train_max = 0., 1.
  eval_min, eval_max = 0., 1.
  if normalize:
    logging.info('Normalizing dataset to have range [-1, 1].')
    config_name = pca_ckpt.split('/')[-1].split('.')[0]
    config_name += slice_ckpt.split('/')[-1].split('.')[0]
    config_name += dim_weights_ckpt.split('/')[-1].split('.')[0]
    train_min, train_max = data_utils.compute_dataset_min_max(
        train_ds,
        ds_split='train',
        cache=True,
        cache_dir=os.path.expanduser(dataset),
        config=config_name)
    eval_min, eval_max = data_utils.compute_dataset_min_max(
        eval_ds,
        ds_split='eval',
        cache=True,
        cache_dir=os.path.expanduser(dataset),
        config=config_name)
    train_ds = train_ds.map(lambda example: normalize_dataset(
        example, train_min, train_max),
                            num_parallel_calls=AUTOTUNE)
    eval_ds = eval_ds.map(lambda example: normalize_dataset(
        example, eval_min, eval_max),
                          num_parallel_calls=AUTOTUNE)

  train_ds = train_ds.prefetch(AUTOTUNE)
  eval_ds = eval_ds.prefetch(AUTOTUNE)
  eval_ds = eval_ds.cache()

  setattr(train_ds, 'min', train_min)
  setattr(train_ds, 'max', train_max)
  setattr(eval_ds, 'min', eval_min)
  setattr(eval_ds, 'max', eval_max)

  if include_cardinality:
    t0 = time.time()
    config_name = str(batch_size)
    data_utils.compute_dataset_cardinality(
        train_ds,
        ds_split='train',
        cache=True,
        cache_dir=os.path.expanduser(dataset),
        config=config_name)
    data_utils.compute_dataset_cardinality(
        eval_ds,
        ds_split='eval',
        cache=True,
        cache_dir=os.path.expanduser(dataset),
        config=config_name)
    logging.info('Computed dataset cardinality in %f seconds', time.time() - t0)

  return train_ds, eval_ds
