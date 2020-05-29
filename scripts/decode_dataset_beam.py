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
r"""Dataset generation."""

import functools
import pickle

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.metrics import Metrics
from magenta.models.music_vae import TrainedModel
import note_seq
import numpy as np
import tensorflow as tf

from .. import config
from ../utils/ import song_utils
from ../utils/ import data_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pipeline_options', '--runner=DirectRunner',
    'Command line flags to use in constructing the Beam pipeline options.')

# Model
flags.DEFINE_string('model', 'melody-2-big', 'Model configuration.')
flags.DEFINE_string('checkpoint', '~/checkpoints/cat-mel_2bar_big.tar',
                    'Model checkpoint.')
flags.DEFINE_boolean('melody', True, 'If True, decode melodies.')

# Dataset
flags.DEFINE_list('data_shape', [32, 512], 'Data shape.')
flags.DEFINE_string('input', './output/mel-32step-512',
                    'Path to TFRecord dataset.')
flags.DEFINE_string('output', './decoded', 'Output directory.')


class DecodeSong(beam.DoFn):
  """Decode MusicVAE embeddings into one-hot NoteSequence tensor."""

  def setup(self):
    logging.info('Loading pre-trained model %s', FLAGS.model)
    self.model_config = config.MUSIC_VAE_CONFIG[FLAGS.model]
    self.model = TrainedModel(self.model_config,
                              batch_size=1,
                              checkpoint_dir_or_path=FLAGS.checkpoint)

    shape = tuple(map(int, FLAGS.data_shape))
    prod = lambda a: functools.reduce(lambda x, y: x * y, a)
    flattened_shape = prod(shape)
    self.decode_fn = lambda x: data_utils._decode_record(x, flattened_shape, len(shape))

  def process(self, example):
    parsed = self.decode_fn(example)
    encoding = parsed['inputs']
    Metrics.counter('DecodeSong', 'decoding_song').inc()

    chunk_length = 2 if FLAGS.melody else 1
    chunks = song_utils.embeddings_to_chunks(encoding, self.model)
    song_utils.fix_instruments_for_concatenation(chunks)
    ns = note_seq.sequences_lib.concatenate_sequences(chunks)

    tensor = np.array(
        self.model_config.data_converter.to_tensors(ns).inputs[::chunk_length])
    tensor = tensor.reshape(-1, tensor.shape[-1])
    yield pickle.dumps(tensor)


def main(argv):
  del argv  # unused

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  with beam.Pipeline(options=pipeline_options) as p:
    p |= 'tfrecord_list' >> beam.Create(tf.io.gfile.glob(FLAGS.input))
    p |= 'read_tfrecord' >> beam.io.tfrecordio.ReadAllFromTFRecord()
    p |= 'shuffle_input' >> beam.Reshuffle()
    p |= 'decode_song' >> beam.ParDo(DecodeSong())
    p |= 'shuffle_output' >> beam.Reshuffle()
    p |= 'write' >> beam.io.WriteToTFRecord(FLAGS.output)


if __name__ == '__main__':
  app.run(main)
