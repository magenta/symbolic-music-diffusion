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
"""Generate wav files from samples."""
import os
import sys

import jax
import jax.numpy as jnp
import note_seq
import numpy as np
import ray
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
from flax import nn
from flax.training import checkpoints
from bokeh.io import export_png
from magenta.models.music_vae import TrainedModel
from pathlib import Path
from scipy.io import wavfile

sys.path.append("{}/../".format(os.path.dirname(os.path.abspath(__file__))))
import utils.data_utils as data_utils
import utils.song_utils as song_utils
import utils.train_utils as train_utils
import utils.metrics as metrics
import config
import train_lm

FLAGS = flags.FLAGS
SYNTH = note_seq.fluidsynth
SAMPLE_RATE = 44100
ray.init()

flags.DEFINE_integer('eval_seed', 42, 'Random number generator seed.')
flags.DEFINE_string('input', 'sample/ncsn', 'Sampling (input) directory.')
flags.DEFINE_string('output', './audio', 'Output directory.')
flags.DEFINE_integer('n_synth', 1000, 'Number of samples to decode.')
flags.DEFINE_boolean('include_wav', True, 'Include audio waveforms.')
flags.DEFINE_boolean('include_plots', True, 'Include Bokeh plots of MIDI.')
flags.DEFINE_boolean('gen_only', False, 'Only generate the fake audio.')

flags.DEFINE_boolean('melody', True, 'If True, decode melodies.')
flags.DEFINE_boolean('infill', False, 'Evaluate quality of infilled measures.')
flags.DEFINE_boolean('interpolate', False, 'Evaluate interpolations.')


def synthesize_ns(path, ns, synth=SYNTH, sample_rate=SAMPLE_RATE):
  """Synthesizes and saves NoteSequence to waveform file."""
  array_of_floats = synth(ns, sample_rate=sample_rate)
  normalizer = float(np.iinfo(np.int16).max)
  array_of_ints = np.array(np.asarray(array_of_floats) * normalizer,
                           dtype=np.int16)
  wavfile.write(path, sample_rate, array_of_ints)


def decode_emb(emb, model, data_converter, chunks_only=False):
  """Generates NoteSequence objects from set of embeddings.
  
  Args:
    emb: Embeddings of shape (n_seqs, seq_length, 512).
    model: Pre-trained MusicVAE model used for decoding.
    data_converter: Corresponding data converter for model.
    chunks_only: If True, assumes embeddings are of the shape (n_seqs, 512)
        where each generated NoteSequence corresponds to one embedding.
  
  Returns:
    A list of decoded NoteSequence objects.
  """
  if chunks_only:
    assert len(emb.shape) == 2
    samples = song_utils.embeddings_to_chunks(emb, model)
    samples = [
        song_utils.Song(sample, data_converter, reconstructed=True)
        for sample in samples
    ]
  else:
    samples = []
    count = 0
    for emb_sample in emb:
      if count % 100 == 0:
        logging.info(f'Decoded {count} sequences.')
      count += 1
      recon = song_utils.embeddings_to_song(emb_sample, model, data_converter)
      samples.append(recon)

  return samples


@ray.remote
def parallel_synth(song, i, ns_dir, audio_dir, image_dir, include_wav,
                   include_plots):
  """Synthesizes NoteSequences (and plots) in parallel."""
  audio_path = os.path.join(audio_dir, f'{i + 1}.wav')
  plot_path = os.path.join(image_dir, f'{i + 1}.png')
  ns_path = os.path.join(ns_dir, f'{i+1}.pkl')
  logging.info(audio_path)
  ns = song.play()

  if include_plots:
    fig = note_seq.plot_sequence(ns, show_figure=False)
    export_png(fig, filename=plot_path)

  if include_wav:
    synthesize_ns(audio_path, ns)

  data_utils.save(ns, ns_path)
  return ns


def main(argv):
  del argv  # unused

  # Get VAE model.
  if FLAGS.melody:
    model_config = config.MUSIC_VAE_CONFIG['melody-2-big']
    ckpt = os.path.expanduser('~/checkpoints/cat-mel_2bar_big.tar')
    vae_model = TrainedModel(model_config,
                             batch_size=1,
                             checkpoint_dir_or_path=ckpt)
  else:
    model_config = config.MUSIC_VAE_CONFIG['multi-0min-1-big']
    ckpt = os.path.expanduser(
        '~/checkpoints/multitrack/fb512_0trackmin/model.ckpt')
    vae_model = TrainedModel(model_config,
                             batch_size=1,
                             checkpoint_dir_or_path=ckpt)
  logging.info(f'Loaded {ckpt}')

  log_dir = FLAGS.input
  real = data_utils.load(os.path.join(log_dir, 'real.pkl'))
  generated = data_utils.load(os.path.join(log_dir, 'generated.pkl'))
  
  collection = data_utils.load(os.path.join(log_dir, 'collection.pkl'))
  idx = np.linspace(0, 40, 10).astype(np.int32)
  collection = collection[idx]


  # Get baselines.
  start_emb = real[:, 7, :]
  end_emb = real[:, 24, :]
  idx = list(range(32))
  if FLAGS.infill:
    fixed_idx = idx[:8] + idx[-8:]
    infilled_idx = idx[8:-8]

    # HACK: Since scaling of eval/train is different, re-add the real bars.
    generated[:, fixed_idx, :] = real[:, fixed_idx, :]

    # Prior baseline.
    prior = np.random.randn(*generated.shape)
    prior[:, fixed_idx, :] = real[:, fixed_idx, :]    
  else:
    prior = np.random.randn(*generated.shape)

  # Interpolation baseline.
  interp_baseline = [
      song_utils.spherical_interpolation(start_emb, end_emb, alpha)
      for alpha in np.linspace(0., 1., 16+2)
  ]
  interp_baseline = np.stack(interp_baseline).transpose(1, 0, 2)
  start_real = real[:, idx[:7], :]
  end_real = real[:, idx[-7:], :]
  interp_baseline = np.concatenate((start_real, interp_baseline, end_real), axis=1)
  assert interp_baseline.shape == generated.shape

  assert real.shape == generated.shape
  is_multi_bar = len(generated.shape) > 2

  logging.info('Decoding sequences.')
  eval_seqs = {}
  for sample_split, sample_emb in (('real', real), ('gen', generated),
                                   ('prior', prior), ('interp',
                                                      interp_baseline)):
    if FLAGS.gen_only and sample_split != 'gen':
      continue

    sample_split = str(sample_split)
    audio_dir = os.path.join(FLAGS.output, sample_split, 'audio')
    image_dir = os.path.join(FLAGS.output, sample_split, 'images')
    ns_dir = os.path.join(FLAGS.output, sample_split, 'ns')
    Path(audio_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    sequences = decode_emb(sample_emb[:FLAGS.n_synth],
                           vae_model,
                           model_config.data_converter,
                           chunks_only=not is_multi_bar)
    assert len(sequences) == FLAGS.n_synth

    futures = [
        parallel_synth.remote(song, i, ns_dir, audio_dir, image_dir,
                              FLAGS.include_wav, FLAGS.include_plots)
        for i, song in enumerate(sequences)
    ]
    ns = ray.get(futures)
    eval_seqs[sample_split] = ns

    logging.info(f'Sythesized {sample_split} at {audio_dir}')


if __name__ == '__main__':
  app.run(main)
