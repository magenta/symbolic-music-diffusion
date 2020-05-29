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
"""Model configurations."""
from magenta.models.music_vae import configs
from magenta.models.music_vae import data
from magenta.models.music_vae import data_hierarchical

MUSIC_VAE_CONFIG = {}

melody_2bar_converter = data.OneHotMelodyConverter(
    skip_polyphony=False,
    max_bars=100,  # Truncate long melodies before slicing.
    max_tensors_per_notesequence=None,
    slice_bars=2,
    gap_bars=None,
    steps_per_quarter=4,
    dedupe_event_lists=False)

mel_2bar_nopoly_converter = data.OneHotMelodyConverter(
    skip_polyphony=True,
    max_bars=100,  # Truncate long melodies before slicing.
    max_tensors_per_notesequence=None,
    slice_bars=2,
    gap_bars=None,
    steps_per_quarter=4,
    dedupe_event_lists=False)

melody_16bar_converter = data.OneHotMelodyConverter(
    skip_polyphony=False,
    max_bars=100,  # Truncate long melodies before slicing.
    slice_bars=16,
    gap_bars=16,
    max_tensors_per_notesequence=None,
    steps_per_quarter=4,
    dedupe_event_lists=False)

multitrack_default_1bar_converter = data_hierarchical.MultiInstrumentPerformanceConverter(
    num_velocity_bins=8,
    hop_size_bars=1,
    min_num_instruments=2,
    max_num_instruments=8,
    max_events_per_instrument=64)

multitrack_zero_1bar_converter = data_hierarchical.MultiInstrumentPerformanceConverter(
    num_velocity_bins=8,
    hop_size_bars=1,
    min_num_instruments=0,
    max_num_instruments=8,
    min_total_events=0,
    max_events_per_instrument=64,
    drop_tracks_and_truncate=True)

MUSIC_VAE_CONFIG['melody-2-big'] = configs.CONFIG_MAP[
    'cat-mel_2bar_big']._replace(data_converter=melody_2bar_converter)

MUSIC_VAE_CONFIG['melody-16-big'] = configs.CONFIG_MAP[
    'hierdec-mel_16bar']._replace(data_converter=melody_16bar_converter)

MUSIC_VAE_CONFIG['multi-1-big'] = configs.CONFIG_MAP[
    'hier-multiperf_vel_1bar_big']._replace(
        data_converter=multitrack_default_1bar_converter)

MUSIC_VAE_CONFIG['multi-0min-1-big'] = configs.CONFIG_MAP[
    'hier-multiperf_vel_1bar_big']._replace(
        data_converter=multitrack_zero_1bar_converter)

MUSIC_VAE_CONFIG['melody-2-big-nopoly'] = configs.Config(
    model=configs.MusicVAE(configs.lstm_models.BidirectionalLstmEncoder(),
                           configs.lstm_models.CategoricalLstmDecoder()),
    hparams=configs.merge_hparams(
        configs.lstm_models.get_default_hparams(),
        configs.HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=mel_2bar_nopoly_converter)
