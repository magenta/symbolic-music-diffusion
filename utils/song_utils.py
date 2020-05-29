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
"""Utilities for manipulating multi-measure NoteSequences."""
import os
import sys

import note_seq
import numpy as np

sys.path.append("{}/../".format(os.path.dirname(os.path.abspath(__file__))))
from config import melody_2bar_converter


def spherical_interpolation(p0, p1, alpha):
  """Spherical linear interpolation."""
  assert p0.shape == p1.shape
  assert p0.ndim == 2 and p1.ndim == 2
  unit_p0 = p0 / np.linalg.norm(p0, axis=1, keepdims=1)
  unit_p1 = p1 / np.linalg.norm(p1, axis=1, keepdims=1)
  omega = np.arccos(np.diag(unit_p0.dot(unit_p1.T)))
  so = np.sin(omega)
  c1 = (np.sin((1.0 - alpha) * omega) / so)[:, np.newaxis]
  c2 = (np.sin(alpha * omega) / so)[:, np.newaxis]
  return c1 * p0 + c2 * p1


def count_measures(note_sequence):
  """Approximate number of measures in the song."""
  splits = note_seq.sequences_lib.split_note_sequence_on_time_changes(
      note_sequence)
  bars = 0
  for split in splits:
    time_signature = split.time_signatures[0]
    tempo = split.tempos[0]
    quarters_per_bar = 4 * time_signature.numerator / time_signature.denominator
    seconds_per_bar = 60 * quarters_per_bar / tempo.qpm
    num_bars = split.total_time / seconds_per_bar
    bars += num_bars
  return bars


def extract_melodies(note_sequence, keep_longest_split=False):
  """Extracts all melodies in a polyphonic note sequence.
  
  Args:
    note_sequence: A polyphonic NoteSequence object.
    keep_longest_split: Whether to discard all subsequences with tempo changes
        other than the longest one.
    
  Returns:
    List of monophonic NoteSequence objects.
  """
  splits = note_seq.sequences_lib.split_note_sequence_on_time_changes(
      note_sequence)

  if keep_longest_split:
    ns = max(splits, key=lambda x: len(x.notes))
    splits = [ns]

  melodies = []
  for split_ns in splits:
    qs = note_seq.sequences_lib.quantize_note_sequence(split_ns,
                                                       steps_per_quarter=4)

    instruments = list(set([note.instrument for note in qs.notes]))

    for instrument in instruments:
      melody = note_seq.melodies_lib.Melody()
      try:
        melody.from_quantized_sequence(qs,
                                       ignore_polyphonic_notes=True,
                                       instrument=instrument,
                                       gap_bars=np.inf)
      except note_seq.NonIntegerStepsPerBarError:
        continue
      melody_ns = melody.to_sequence()
      melodies.append(melody_ns)

  return melodies


def generate_shifted_sequences(song, resolution=1):
  """Generates shifted and overlapping versions of a Song.
  
  Args:
    song: A multitrack Song object.
    resolution: The number of shifted examples, with computed timing offsets
      uniformly spaced.

  Returns:
    A list of multitrack Song objects.
  """
  offset = 2.0 / resolution
  base = song.note_sequence
  dc = song.data_converter
  results = []
  for step in range(resolution):
    shift = note_seq.extract_subsequence(base, offset * step, base.total_time)
    results.append(Song(shift, dc, chunk_length=1))
  assert len(results) == resolution
  return results


def fix_instruments_for_concatenation(note_sequences):
  """Adjusts instruments for concatenating multitrack measures."""
  instruments = {}
  for i in range(len(note_sequences)):
    for note in note_sequences[i].notes:
      if not note.is_drum:
        if note.program not in instruments:
          if len(instruments) >= 8:
            instruments[note.program] = len(instruments) + 2
          else:
            instruments[note.program] = len(instruments) + 1
        note.instrument = instruments[note.program]
      else:
        note.instrument = 9


def fix_chunk_lengths_for_concatenation(note_sequences):
  """Adjusts the total_time of each tokenized chunk for concatenating
  multitrack measures.
  """
  max_chunk_time = max([ns.total_time for ns in note_sequences])
  for chunk in note_sequences:
    chunk.total_time = max_chunk_time


def chunks_to_embeddings(sequences, model, data_converter):
  """Convert NoteSequence objects into latent space embeddings.

  Args:
    sequences: A list of NoteSequence objects.
    model: A TrainedModel object used for inference.
    data_converter: A data converter (e.g. OneHotMelodyConverter, 
        TrioConverter) used to convert NoteSequence objects into
        tensor encodings for model inference.

  Returns:
    A numpy matrix of shape [len(sequences), latent_dims].
  """
  assert model is not None, 'No model provided.'

  latent_dims = model._z_input.shape[1]
  idx = []
  non_rest_chunks = []
  zs = np.zeros((len(sequences), latent_dims))
  mus = np.zeros((len(sequences), latent_dims))
  sigmas = np.zeros((len(sequences), latent_dims))
  for i, chunk in enumerate(sequences):
    if len(data_converter.to_tensors(chunk).inputs) > 0:
      idx.append(i)
      non_rest_chunks.append(chunk)
  if non_rest_chunks:
    z, mu, sigma = model.encode(non_rest_chunks)
    assert z.shape == mu.shape == sigma.shape
    for i, mean in enumerate(mu):
      zs[idx[i]] = z[i]
      mus[idx[i]] = mean
      sigmas[idx[i]] = sigma[i]
  return zs, mus, sigmas


def embeddings_to_chunks(embeddings, model, temperature=1e-3):
  """Decode latent embeddings as NoteSequences.

  Args:
    embeddings: A numpy array of latent embeddings.
    model: A TrainedModel object used for decoding embeddings.

  Returns:
    A list of NoteSequence objects.
  """
  assert model is not None, 'No model provided.'
  assert len(embeddings) > 0

  reconstructed_chunks = model.decode(embeddings,
                                      temperature=temperature,
                                      length=model._config.hparams.max_seq_len)
  assert len(reconstructed_chunks) == len(embeddings)

  embedding_norms = np.linalg.norm(embeddings, axis=1)
  rest_chunk_idx = np.where(
      embedding_norms == 0)[0]  # rests correspond to zero-length embeddings

  for idx in rest_chunk_idx:
    rest_ns = note_seq.NoteSequence()
    rest_ns.total_time = reconstructed_chunks[idx].total_time
    reconstructed_chunks[idx] = rest_ns
  return reconstructed_chunks


def embeddings_to_song(embeddings,
                       model,
                       data_converter,
                       fix_instruments=True,
                       temperature=1e-3):
  """Decode latent embeddings as a concatenated NoteSequence.

  Args:
    embeddings: A numpy array of latent embeddings.
    model: A TrainedModel object used for decoding.
    data_converter: A data converter used by the returned Song
        object.
    fix_instruments: A boolean determining whether instruments in 
        multitrack measures should be fixed before concatenation.

  Returns:
    A Song object.
  """
  chunks = embeddings_to_chunks(embeddings, model, temperature)
  if fix_instruments:
    fix_instruments_for_concatenation(chunks)
  concat_chunks = note_seq.sequences_lib.concatenate_sequences(chunks)
  return Song(concat_chunks, data_converter, reconstructed=True)


def encode_songs(model, songs, chunk_length=None, programs=None):
  """Generate embeddings for a batch of songs.

  Args:
    model: A TrainedModel object used for inference.
    songs: A list of Song objects.
    chunk_length: An integer describing the number of measures
        each chunk of each song should contain.
    programs: A list of integers specifying which MIDI programs to use.
        Default is to keep all available programs.

  Returns:
    A list of numpy matrices each with shape [3, len(song_chunks), latent_dims].
	"""
  assert model is not None, 'No model provided.'
  assert len(songs) > 0, 'No songs provided.'

  chunks, splits = [], []
  data_converter = songs[0].data_converter
  i = 0
  for song in songs:
    chunk_tensors, chunk_sequences = song.chunks(chunk_length=chunk_length,
                                                 programs=programs)
    del chunk_tensors
    chunks.extend(chunk_sequences)
    splits.append(i)
    i += len(chunk_sequences)

  z, mu, sigma = chunks_to_embeddings(chunks, model, data_converter)

  encoding = []
  for i in range(len(splits)):
    j, k = splits[i], None if i + 1 == len(splits) else splits[i + 1]
    song_encoding = [z[j:k], mu[j:k], sigma[j:k]]
    song_encoding = np.stack(song_encoding)
    encoding.append(song_encoding)

  assert len(encoding) == len(splits) == len(songs)
  return encoding


class Song(object):
  """Song object used to provide additional abstractions for NoteSequences.
  
  Attributes:
    note_sequence: A NoteSequence object holding the Song's MIDI data.
    data_converter: A data converter used for preprocessing and tokenization
        for a corresponding MusicVAE model.
    chunk_length: The number of measures in each tokenized chunk of MIDI
        (dependent on the model configuration).
    multitrack: Whether this Song is multitrack or not.
    reconstructed: A boolean describing whether this Song is reconstructed
        from the decoder of a MusicVAE model.
  """

  def __init__(self,
               note_sequence,
               data_converter,
               chunk_length=2,
               multitrack=False,
               reconstructed=False):
    self.note_sequence = note_sequence
    self.data_converter = data_converter
    self.chunk_length = chunk_length
    self.reconstructed = reconstructed
    self.multitrack = multitrack

  def encode(self, model, chunk_length=None, programs=None):
    """Encode song chunks (and full-chunk rests).
    
    Returns:
      z: (chunks, latent_dims), mu: (chunks, latent_dims), sigma: (chunks, latent_dims).
    """
    chunk_tensors, chunk_sequences = self.chunks(chunk_length=chunk_length,
                                                 programs=programs)
    z, means, sigmas = chunks_to_embeddings(chunk_sequences, model,
                                            self.data_converter)
    del chunk_tensors  # unused
    return z

  def chunks(self, chunk_length=None, programs=None, fix_instruments=True):
    """Split and featurize song into chunks of tensors and NoteSequences."""
    assert not self.reconstructed, 'Not safe to tokenize reconstructed Songs.'

    data = self.note_sequence
    step_size = self.chunk_length
    if chunk_length is not None:
      step_size = chunk_length
    if programs is not None:
      data = self.select_programs(programs)

    # Use the data converter to preprocess sequences
    tensors = self.data_converter.to_tensors(data).inputs[::step_size]
    sequences = self.data_converter.from_tensors(tensors)

    if fix_instruments and self.multitrack:
      fix_instruments_for_concatenation(sequences)

    return tensors, sequences

  def count_chunks(self, chunk_length=None):
    length = self.chunk_length if chunk_length is None else chunk_length
    return count_measures(self.note_sequence) // length

  @property
  def programs(self):
    """MIDI programs used in this song."""
    return list(set([note.program for note in self.note_sequence.notes]))

  def select_programs(self, programs):
    """Keeps selected programs of MIDI (e.g. melody program)."""
    assert len(programs) > 0
    assert all([program >= 0 for program in programs])

    ns = note_seq.NoteSequence()
    ns.CopyFrom(self.note_sequence)
    del ns.notes[:]

    for note in self.note_sequence.notes[:]:
      if note.program in programs:
        new_note = ns.notes.add()
        new_note.CopyFrom(note)
    return ns

  def truncate(self, chunks=0, offset=0):
    """Returns a truncated version of the song.

      Args:
        chunks: The number of chunks in the truncated sequence.
        offset: The offset in chunks to begin truncation.

      Returns:
        A truncated Song object.
    """
    tensors = self.data_converter.to_tensors(
        self.note_sequence).inputs[::self.chunk_length]
    sequences = self.data_converter.from_tensors(tensors)[offset:offset +
                                                          chunks]
    fix_instruments_for_concatenation(sequences)
    concat_chunks = note_seq.sequences_lib.concatenate_sequences(sequences)
    return Song(concat_chunks,
                self.data_converter,
                chunk_length=self.chunk_length)

  def _count_melody_chunks(self, program):
    """Determines the number of 2-measure chunks using the melody data pipeline."""
    ns = self.select_programs([program])
    tensors = melody_2bar_converter.to_tensors(ns).inputs[::2]
    sequences = melody_2bar_converter.from_tensors(tensors)
    return len(sequences)

  def find_programs(self):
    """Search for the most important MIDI programs in the song."""

    def heuristic(program):
      expected = self.count_chunks(chunk_length=2)
      extracted = self._count_melody_chunks(program)
      if extracted > 0 and abs(extracted - expected) < 0.5 * expected:
        return True
      return False

    midi_programs = self.programs
    top_programs = [p for p in midi_programs if heuristic(p)]
    return top_programs

  def stripped_song(self):
    """A stripped down version using programs found by a special heuristic."""
    top_programs = self.find_programs()
    ns = self.select_programs(top_programs)
    return Song(ns, self.data_converter, self.chunk_length)

  def download(self, filename, preprocessed=True, programs=None):
    """Download song as MIDI file."""
    assert filename is not None, 'No filename specified.'

    data = self.note_sequence
    if programs is not None:
      data = self.select_programs(programs)

    if not self.reconstructed and preprocessed:  # do not tokenize again if reconstructed
      tensors, chunks = self.chunks(programs=programs)
      del tensors  # unused
      data = note_seq.sequences_lib.concatenate_sequences(chunks)

    note_seq.sequence_proto_to_midi_file(data, filename)

  def play(self, preprocessed=True, programs=None):
    """Play a song with fluidsynth."""
    data = self.note_sequence
    if programs is not None:
      data = self.select_programs(programs)

    if not self.reconstructed and preprocessed:  # do not tokenize again if reconstructed
      tensors, chunks = self.chunks(programs=programs)
      del tensors  # unused
      data = note_seq.sequences_lib.concatenate_sequences(chunks)

    note_seq.play_sequence(data, synth=note_seq.fluidsynth)
    return data
