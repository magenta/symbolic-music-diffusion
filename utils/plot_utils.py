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
"""Utilities for TensorBoard plotting."""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import utils.ebm_utils as ebm_utils


def scatter_2d(data, scale=8, show=False):
  """Create a plot to compare generated 2D samples.
  
  Args:
    data: An array of 2D points to draw.
    scale: A number that specifies the grid bounds.  
  """
  assert data.shape[-1] == 2
  x = data[:, 0]
  y = data[:, 1]

  buf = io.BytesIO()
  plt.figure()
  plt.scatter(x, y, s=0.1)
  plt.axis('square')
  plt.title('Samples')
  plt.xlim([-scale, scale])
  plt.ylim([-scale, scale])
  if show:
    plt.show()
  plt.savefig(buf, format='png')
  plt.close()
  buf.seek(0)
  return buf


def animate_scatter_2d(data, scale=8, show=False, fps=60):
  """Create an animation to compare generated 2D samples.
  
  Args:
    data: An array of 2D points to draw of shape [timesteps, N, 2].
    scale: A number that specifies the grid bounds.  
  """
  assert data.shape[-1] == 2 and data.ndim == 3

  buf = io.BytesIO()
  plt.figure()
  fig, ax = plt.subplots()
  sc = ax.scatter(data[0, :, 0], data[0, :, 1], s=0.1)
  title = ax.text(0.5,
                  0,
                  "",
                  bbox={
                      'facecolor': 'w',
                      'alpha': 0.5,
                      'pad': 5
                  },
                  transform=ax.transAxes,
                  ha="center")
  plt.axis('square')
  plt.title('Samples')
  plt.xlim([-scale, scale])
  plt.ylim([-scale, scale])

  def animate(i):
    title.set_text(f'frame: {i}')
    sc.set_offsets(data[i])

  anim = FuncAnimation(fig, animate, frames=len(data), interval=1000 / fps)

  anim.save('tmp.gif', writer='imagemagick')
  with open('tmp.gif', 'rb') as f:
    buf.write(f.read())
    f.close()
  os.remove('tmp.gif')

  if show:
    plt.draw()
    plt.show()

  plt.close()
  buf.seek(0)
  return buf


def energy_contour_2d(model, scale=5, show=False):
  """Create contour plot of the energy landscape.
  
  Args:
    model: An energy-based model (R^2 -> R).
    scale: A number that specifies the grid bounds.
  """
  x = np.arange(-scale, scale, 0.05)
  y = np.arange(-scale, scale, 0.05)
  xx, yy = np.meshgrid(x, y, sparse=False)
  coords = np.stack((xx, yy), axis=2)
  coords_flat = coords.reshape(-1, 2)
  z_flat = model(coords_flat)

  # highlight regions where energy is low (high density)
  z = -1 * z_flat.reshape(*coords.shape[:-1])

  buf = io.BytesIO()
  plt.figure()
  plt.contourf(x, y, z)
  if show:
    plt.show()
  plt.savefig(buf, format='png')
  plt.close()
  buf.seek(0)
  return buf


def score_field_2d(model, sigma=None, scale=8, show=False):
  """Create plot of the gradient field of the energy landscape.
  
  Args:
    model: An energy-based model (R^2 -> R) or a score network (R^2 -> R^2).
    sigma: A noise value for an NCSN. Only required if model is a score network.
    scale: A number that specifies the grid bounds.
  """
  mesh = []
  x = np.linspace(-scale, scale, 20)
  y = np.linspace(-scale, scale, 20)
  for i in x:
    for j in y:
      mesh.append(np.asarray([i, j]))
  mesh = np.stack(mesh, axis=0)

  if sigma is not None:
    sigma = sigma * np.ones((mesh.shape[0], 1))
    scores = model(mesh, sigma)
  else:
    scores = ebm_utils.vgrad(model, mesh)
  assert scores.shape == mesh.shape

  buf = io.BytesIO()
  plt.figure()
  plt.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=0.005)
  plt.title('Estimated scores', fontsize=16)
  plt.axis('square')
  if show:
    plt.show()
  plt.savefig(buf, format='png')
  plt.close()
  buf.seek(0)
  return buf


def image_tiles(images, shape=(28, 28), show=False):
  n = len(images)
  for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(images[i].reshape(*shape))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  if show:
    plt.show()
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close()
  buf.seek(0)
  return buf
