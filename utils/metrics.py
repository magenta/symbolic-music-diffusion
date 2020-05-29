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
"""Metrics."""
import math
import note_seq
import numpy as np
import scipy
from sklearn import metrics


def frechet_distance(real, fake):
  """Frechet distance.
  
  Lower score is better.
  """
  mu1, sigma1 = np.mean(real, axis=0), np.cov(real, rowvar=False)
  mu2, sigma2 = np.mean(fake, axis=0), np.cov(fake, rowvar=False)
  diff = mu1 - mu2
  covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real

  assert np.isfinite(covmean).all() and not np.iscomplexobj(covmean)

  tr_covmean = np.trace(covmean)

  frechet_dist = diff.dot(diff)
  frechet_dist += np.trace(sigma1) + np.trace(sigma2)
  frechet_dist -= 2 * tr_covmean
  return frechet_dist


def mmd_rbf(real, fake, gamma=1.0):
  """(RBF) kernel distance.
  
  Lower score is better.
  """
  XX = metrics.pairwise.rbf_kernel(real, real, gamma)
  YY = metrics.pairwise.rbf_kernel(fake, fake, gamma)
  XY = metrics.pairwise.rbf_kernel(real, fake, gamma)
  return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_polynomial(real, fake, degree=2, gamma=1, coef0=0):
  """(Polynomial) kernel distance.
  
  Lower score is better.
  """
  XX = metrics.pairwise.polynomial_kernel(real, real, degree, gamma, coef0)
  YY = metrics.pairwise.polynomial_kernel(fake, fake, degree, gamma, coef0)
  XY = metrics.pairwise.polynomial_kernel(real, fake, degree, gamma, coef0)
  return XX.mean() + YY.mean() - 2 * XY.mean()


def framewise_statistic(ns, stat_fn, hop_size=1, frame_size=1):
  """Computes framewise MIDI statistic."""
  total_time = int(math.ceil(ns.total_time))
  frames = []
  trim = frame_size - hop_size
  for i in range(0, total_time - trim, hop_size):
    one_sec_chunk = note_seq.sequences_lib.trim_note_sequence(
        ns, i, i + frame_size)
    value = stat_fn(one_sec_chunk.notes)
    frames.append(value)
  return np.array(frames)


def note_density(ns, hop_size=1, frame_size=1):
  stat_fn = lambda notes: len(notes)
  return framewise_statistic(ns,
                             stat_fn,
                             hop_size=hop_size,
                             frame_size=frame_size)


def pitch_range(ns, hop_size=1, frame_size=1):

  def stat_fn(notes):
    pitches = [note.pitch for note in notes]
    return max(pitches) - min(pitches) if len(pitches) > 0 else 0

  return framewise_statistic(ns,
                             stat_fn,
                             hop_size=hop_size,
                             frame_size=frame_size)


def mean_pitch(ns, hop_size=1, frame_size=1):

  def stat_fn(notes):
    pitches = np.array([note.pitch for note in notes])
    return pitches.mean() if len(pitches) > 0 else 0

  return framewise_statistic(ns,
                             stat_fn,
                             hop_size=hop_size,
                             frame_size=frame_size)


def var_pitch(ns, hop_size=1, frame_size=1):

  def stat_fn(notes):
    pitches = np.array([note.pitch for note in notes])
    return pitches.var() if len(pitches) > 0 else 0

  return framewise_statistic(ns,
                             stat_fn,
                             hop_size=hop_size,
                             frame_size=frame_size)


def mean_note_duration(ns, hop_size=1, frame_size=1):

  def stat_fn(notes):
    durations = np.array([note.end_time - note.start_time for note in notes])
    return durations.mean() if len(durations) > 0 else 0

  return framewise_statistic(ns,
                             stat_fn,
                             hop_size=hop_size,
                             frame_size=frame_size)


def var_note_duration(ns, hop_size=1, frame_size=1):

  def stat_fn(notes):
    durations = np.array([note.end_time - note.start_time for note in notes])
    return durations.var() if len(durations) > 0 else 0

  return framewise_statistic(ns,
                             stat_fn,
                             hop_size=hop_size,
                             frame_size=frame_size)


def perceptual_midi_histograms(ns, interval=1):
  """Generates histograms for each MIDI feature."""
  return dict(
      nd=note_density(ns, interval=interval),
      pr=pitch_range(ns, interval=interval),
      mp=mean_pitch(ns, interval=interval),
      vp=var_pitch(ns, interval=interval),
      md=mean_note_duration(ns, interval=interval),
      vd=var_note_duration(ns, interval=interval),
  )


def perceptual_midi_statistics(ns, interval=1, vector=False):
  """Feature vector of means and variances of MIDI histograms.
  
  Args:
    ns: NoteSequence object.
    interval: Integer time interval (in seconds) for each histogram bin.
    vector: If True, returns statistics as a feature vector.
  """
  features = {}
  histograms = perceptual_midi_histograms(ns, interval=interval)
  for key in histograms:
    mu = histograms[key].mean()
    var = histograms[key].var()
    features[key] = (mu, var)

  if vector:
    vec = np.array(list(features.values()))
    return vec.reshape(-1)

  return features


def perceptual_similarity(ns1, ns2, interval=1):
  """Perceptual similarity as determined by Overlapping Area Metric.

  Determines pairwise similarity for two NoteSequence objects.

  Args:
    ns1: NoteSequence object.
    ns2: NoteSequence object.
    interval: Integer time interval (in seconds) for each histogram bin.
  """
  stats1 = perceptual_midi_statistics(ns1, interval, vector=False)
  stats2 = perceptual_midi_statistics(ns2, interval, vector=False)
  similarity = {}
  for key in stats1:
    mu1, var1 = stats1[key]
    mu2, var2 = stats2[key]
    similarity[key] = overlapping_area(mu1, mu2, var1, var2)
  return similarity


def overlapping_area(mu1, mu2, var1, var2):
  """Compute overlapping area of two Gaussians.
  
  Args:
    mu1: Mean of first Gaussian pdf.
    mu2: Mean of second Gaussian pdf.
    var1: Variance of first Gaussian pdf.
    var2: Variance of second Gaussian pdf.
  Returns:
    Overlapping area of the two density functions.
  """
  idx = mu2 < mu1
  mu_a = mu2 * idx + np.logical_not(idx) * mu1
  mu_b = mu1 * idx + np.logical_not(idx) * mu2
  var_a = var2 * idx + np.logical_not(idx) * var1
  var_b = var1 * idx + np.logical_not(idx) * var2

  c_sqrt_factor = (mu_a - mu_b)**2 + 2 * (var_a - var_b) * np.log(
      np.sqrt(var_a + 1e-6) / np.sqrt(var_b + 1e-6))
  c_sqrt_factor = np.sqrt(c_sqrt_factor)
  c = mu_b * var_a - np.sqrt(var_b) * (mu_a * np.sqrt(var_b) +
                                       np.sqrt(var_a) * c_sqrt_factor)
  c = c / (var_a - var_b + 1e-6)

  sqrt_2 = np.sqrt(2)
  oa = 1 - 0.5 * scipy.special.erf(
      (c - mu_a) / (sqrt_2 * np.sqrt(var_a + 1e-6)))
  oa = oa + 0.5 * scipy.special.erf(
      (c - mu_b) / (sqrt_2 * np.sqrt(var_b + 1e-6)))
  return oa
