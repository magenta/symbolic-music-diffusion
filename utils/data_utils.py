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
"""Dataset utilities."""
import os
import pickle

import jax
import numpy as np
import tensorflow as tf

from absl import logging
from functools import reduce

AUTOTUNE = tf.data.experimental.AUTOTUNE


def save(obj, path):
  """Save an object to disk as a pickle file."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=4)
    logging.info('Saved to %s', path)


def load(path):
  """Load pickled object into memory."""
  with open(path, 'rb') as f:
    return pickle.load(f)


def _decode_record(record, flattened_shape, shape_len, tokens=False):
  if not tokens:
    input_parser = tf.io.FixedLenFeature([flattened_shape], tf.float32)
  else:
    input_parser = tf.io.FixedLenFeature((), tf.string)

  parsed = tf.io.parse_example(
      record, {
          'inputs': input_parser,
          'input_shape': tf.io.FixedLenFeature([shape_len], tf.int64)
      })

  if tokens:
    parsed['inputs'] = tf.io.parse_tensor(parsed['inputs'], out_type=np.bool)

  parsed['inputs'] = tf.reshape(parsed['inputs'], parsed['input_shape'])
  return parsed


def compute_dataset_cardinality(ds,
                                ds_split='train',
                                cache=False,
                                cache_dir=None,
                                config=''):
  """Computes and optionally caches cardinality of tf.data.Dataset."""
  card_cache_path = os.path.join(cache_dir,
                                 f'cache/{ds_split}_{config}_cardinality.pkl')

  if os.path.exists(card_cache_path):
    logging.info('Using cached dataset cardinality at %s', cache_dir)
    cardinality = load(card_cache_path)
  else:
    cardinality = -1
    if hasattr(ds, 'cardinality'):
      cardinality = ds.cardinality().numpy()
    if cardinality <= 0:
      cardinality = 0
      for e in ds.as_numpy_iterator():
        cardinality += 1

  if cache:
    assert cache_dir is not None
    assert not hasattr(ds, 'examples')
    setattr(ds, 'examples', cardinality)
    save(cardinality, card_cache_path)

  return cardinality


def compute_dataset_statistics(ds,
                               ds_split='train',
                               cache=False,
                               cache_dir=None,
                               config=''):
  """Computes the mean and standard deviation of tf.data.Dataset."""
  mean_cache_path = os.path.join(cache_dir,
                                 f'cache/{ds_split}_{config}_mean.pkl')
  stddev_cache_path = os.path.join(cache_dir,
                                   f'cache/{ds_split}_{config}_stddev.pkl')

  if os.path.exists(mean_cache_path) and os.path.exists(stddev_cache_path):
    logging.info('Using cached dataset statistics at %s', cache_dir)
    ds_mean = load(mean_cache_path)
    ds_std = load(stddev_cache_path)
  else:
    cardinality = compute_dataset_cardinality(ds, cache=False)
    ds_sum = ds.reduce(0., lambda x, y: x + y).numpy()
    ds_squared = ds.map(lambda x: x**2, num_parallel_calls=AUTOTUNE)
    ds_squared_sum = ds_squared.reduce(0., lambda x, y: x + y).numpy()
    ds_mean = ds_sum / cardinality
    ds_second_moment = ds_squared_sum / cardinality
    ds_std = np.sqrt(ds_second_moment - ds_mean**2)

  if cache:
    assert cache_dir is not None
    assert not hasattr(ds, 'mean') and not hasattr(ds, 'stddev')
    setattr(ds, 'mean', ds_mean)
    setattr(ds, 'stddev', ds_std)
    save(ds_mean, mean_cache_path)
    save(ds_std, stddev_cache_path)

  return ds_mean, ds_std


def compute_dataset_min_max(ds,
                            ds_split='train',
                            cache=False,
                            cache_dir=None,
                            config=''):
  """Computes the min and max of (batched) tf.data.Dataset."""
  min_cache_path = os.path.join(cache_dir, f'cache/{ds_split}_{config}_min.pkl')
  max_cache_path = os.path.join(cache_dir, f'cache/{ds_split}_{config}_max.pkl')

  if os.path.exists(min_cache_path) and os.path.exists(max_cache_path):
    logging.info('Using cached dataset min/max at %s', cache_dir)
    ds_min = load(min_cache_path)
    ds_max = load(max_cache_path)
  else:
    ds_maxes = ds.map(lambda x: tf.reduce_max(x), num_parallel_calls=AUTOTUNE)
    ds_mins = ds.map(lambda x: tf.reduce_min(x), num_parallel_calls=AUTOTUNE)
    ds_min = ds_mins.reduce(tf.float32.max, lambda x, y: tf.math.minimum(x, y))
    ds_max = ds_maxes.reduce(tf.float32.min, lambda x, y: tf.math.maximum(x, y))
    ds_min, ds_max = ds_min.numpy(), ds_max.numpy()

  if cache:
    assert cache_dir is not None
    assert not hasattr(ds, 'min') and not hasattr(ds, 'max')
    setattr(ds, 'min', ds_min)
    setattr(ds, 'max', ds_max)
    save(ds_min, min_cache_path)
    save(ds_max, max_cache_path)

  return ds_min, ds_max


def get_tf_record_dataset(file_pattern=None,
                          shape=(512,),
                          batch_size=512,
                          shuffle=True,
                          tokens=False):
  """Generates a TFRecord dataset given a file pattern.
  
  Args:
    file_pattern: TFRecord file pattern (e.g train-*.tfrecord)
    shape: Example shape.
    batch_size: Number of examples per batch during training. This 
      argument is used for setting the shuffle buffer size.
    shuffle: Whether to shuffle the data or not.
    tokens: Extract one-hot token dataset.
  
  Returns:
    A tf.data.Dataset iterator.
  """
  filenames = tf.data.Dataset.list_files(os.path.expanduser(file_pattern),
                                         shuffle=shuffle)
  dataset = filenames.interleave(map_func=tf.data.TFRecordDataset,
                                 cycle_length=40,
                                 num_parallel_calls=AUTOTUNE,
                                 deterministic=False)
  if shuffle:
    dataset = dataset.shuffle(8 * batch_size)

  prod = lambda a: reduce(lambda x, y: x * y, a)
  flattened_shape = prod(shape)
  shape_len = len(shape)
  decode_fn = lambda x: _decode_record(x, flattened_shape, shape_len, tokens)
  dataset = dataset.map(decode_fn, num_parallel_calls=AUTOTUNE)
  return dataset


def _truncate_embeddings(embeddings, length):
  """Truncate embedding matrix."""
  pad_length = length - len(embeddings)
  if pad_length <= 0:
    embeddings = embeddings[:length]
  else:
    padding = np.zeros((pad_length, embeddings.shape[-1]))
    embeddings = np.concatenate((embeddings, padding))

  assert len(embeddings) == length

  return embeddings


def self_similarity(embeddings, normalized=True, max_len=80):
  """Generates self-similarity matrix for sequence of embeddings."""
  embeddings = _truncate_embeddings(embeddings, max_len)

  self_sim = np.dot(embeddings, embeddings.T)
  if normalized:
    norm_embeddings = embeddings / np.linalg.norm(
        embeddings, ord=2, axis=1, keepdims=True)
    self_sim = np.dot(norm_embeddings, norm_embeddings.T)
    self_sim[np.isnan(self_sim)] = 0  # hack to overcome division by zero NaNs
  return self_sim


def unroll_upper_triangular(matrix):
  """Converts square matrix to vector by unrolling upper triangle."""
  rows, cols = matrix.shape
  assert rows == cols, 'Not a square matrix.'

  row_idx, col_idx = np.triu_indices(rows, 1)
  unrolled = []
  for i, j in zip(row_idx, col_idx):
    unrolled.append(matrix[i][j])
  assert len(unrolled) == rows * (rows - 1) // 2
  return unrolled


def roll_upper_triangular(vector, size):
  """Converts unrolled upper triangle into square matrix."""
  matrix = np.ones((size, size))
  offset = 0
  for i in range(size):
    stream = vector[offset:]
    row = stream[:size - (i + 1)]
    matrix[i, i + 1:size] = row
    matrix[i + 1:size, i] = row
    offset += len(row)
  assert offset == len(vector)
  return matrix


def erase_bars(embeddings, indices):
  """Erases vectors from a given a set of embeddings.

  Args:
    embeddings: A numpy matrix of embeddings.
    indices: A list of indices corresponding to vectors that will be erased.

  Returns:
    A modified embedding matrix.
  """
  return jax.ops.index_update(embeddings, jax.ops.index[indices], 0)


def infill_bars(embeddings, chunk_params, erased_chunk_indices):
  """Infills a partially erased embedding matrix with sampled embedding parameters.

  Args:
    embeddings: A partially incomplete embedding matrix.
    chunk_params: A list of sampled embedding vectors.
    erased_bar_indices: A list of indices corresponding to vector positions in the
        embedding matrix that will be replaced by sampled embeddings.

  Returns:
    A modified embedding matrix.
  """
  assert len(chunk_params) == len(erased_chunk_indices)
  return jax.ops.index_update(embeddings, jax.ops.index[erased_chunk_indices],
                              chunk_params)


def batches(data, labels=None, batch_size=32):
  """Generate batches of data.
  
  Args:
    data: A numpy matrix of data of shape [num_examples, *data_shape].
    labels: An optional matrix of corresponding labels for each entry in data of
        shape [num_examples, *label_shape]
    batch_size: Batch size.
  
  Returns:
    An iterator that yields batches of data with their labels.
  """
  num_batches = data.shape[0] // batch_size
  for i in range(num_batches):
    j, k = i * batch_size, (i + 1) * batch_size
    if labels is not None:
      assert len(data) == len(labels)
      batch = (data[j:k], labels[j:k])
    else:
      batch = data[j:k]
    yield batch


def shuffle(data, labels=None):
  """Shuffle dataset.
  
  Args:
    data: A numpy matrix of data of shape [num_examples, *data_shape].
    labels: An optional matrix of corresponding labels for each entry in data of
        shape [num_examples, *label_shape].

  Returns:
    Shuffled data and label matrices.
  """
  idx = np.random.permutation(len(data))
  shuffled_data = data[idx]
  if labels is not None:
    assert len(data) == len(labels)
    shuffled_labels = labels[idx]
    return shuffled_data, shuffled_labels
  else:
    return shuffled_data
