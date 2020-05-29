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
"""Loss functions."""
import jax
import jax.numpy as jnp
from flax import nn


def reduce_fn(x, mode):
  if mode == "none" or mode is None:
    return jnp.asarray(x)
  elif mode == "sum":
    return jnp.sum(x)
  elif mode == "mean":
    return jnp.mean(jnp.asarray(x))
  else:
    raise ValueError("Unsupported reduction option.")


def series_loss(context, true_target, pred_target, reduction="mean"):
  """Series loss (self-similarity + MSE loss).
  
  Compute the loss between a predicted target embedding and the true target embedding
  conditioned on a sequence of previous embeddings (context).

  Args:
    context: A sequence of continuous embeddings.
    true_target: The true next embedding in the sequence.
    pred_target: The predicted next embedding in the sequence.
    reduction: Type of reduction to apply to loss.

  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `data`;
    otherwise, it is scalar.
  """
  ss = context @ true_target.T
  ss_hat = context @ pred_target.T
  loss = mean_squared_error(ss.T, ss_hat.T) + mean_squared_error(
      true_target, pred_target)
  return reduce_fn(loss, reduction)


def _log_gaussian_pdf(y, mu, log_sigma):
  """The log probability density function of a Gaussian distribution."""
  norm_const = jnp.log(jnp.sqrt(2.0 * jnp.pi))
  return -0.5 * ((y - mu) / jnp.exp(log_sigma))**2 - log_sigma - norm_const


def gaussian_mixture_loss(log_pi, mu, log_sigma, data, reduction="mean"):
  """Mixture density network loss.
  
  Computes the negative log-likelihood of data under a Gaussian mixture.

  Args:
    log_pi: The log of the relative weights of each Gaussian.
    mu: The mean of the Gaussians.
    log_sigma: The log of the standard deviation of each Gaussian.
    data: A batch of data to compute the loss for.
    reduction: Type of reduction to apply to loss.
  
  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `data`;
    otherwise, it is scalar.
  """
  k = log_pi.shape[-1]
  data = jnp.repeat(data[:, jnp.newaxis, :,], k, axis=1)
  loglik = _log_gaussian_pdf(data, mu, log_sigma)  # dimension-wise density
  loglik = loglik.sum(axis=2)  # works because of diagonal covariance
  loss = jax.scipy.special.logsumexp(log_pi + loglik, axis=1)
  return -reduce_fn(loss, reduction)


def binary_cross_entropy_with_logits(logits, labels):
  """Numerically stable binary cross entropy loss."""
  return labels * nn.softplus(-logits) + (1 - labels) * nn.softplus(logits)


def sigmoid_cross_entropy(logits, labels, reduction="sum"):
  """Computes sigmoid cross entropy given logits and multiple class labels.

  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.

  `logits` and `labels` must have the same type and shape.

  Args:
    logits: Logit output values.
    labels: Ground truth integer labels in {0, 1}.
    reduction: Type of reduction to apply to loss.

  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `labels`;
    otherwise, it is scalar.

  Raises:
    ValueError: If the type of `reduction` is unsupported.
  """
  log_p = nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
  log_not_p = nn.log_sigmoid(-logits)
  loss = -labels * log_p - (1. - labels) * log_not_p
  return reduce_fn(loss, reduction)


def mean_squared_error(logits, labels, reduction="mean"):
  """Mean squared error."""
  loss = jnp.square(logits - labels).mean(axis=1)
  return reduce_fn(loss, reduction)


def kl_divergence(mu, var):
  """KL divergence from a standard unit Gaussian."""
  return 0.5 * jnp.sum(jnp.square(mu) + var - 1 - jnp.log(var))


def denoising_score_matching_loss(batch,
                                  model,
                                  sigmas,
                                  rng,
                                  continuous_noise=False,
                                  reduction="mean"):
  """Denoising score matching objective used to train NCSNs.
  
  Args:
    batch: A batch of data to compute the loss for.
    model: A noise-conditioned score network.
    sigmas: A noise schedule (list of standard deviations).
    rng: Random number generator key to sample sigmas.
    continuous_noise: If True, uses continuous noise conditioning.
    reduction: Type of reduction to apply to loss.
  
  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `data`;
    otherwise, it is scalar.
  """
  rng, label_rng, sample_rng = jax.random.split(rng, num=3)
  labels = jax.random.randint(key=label_rng,
                              shape=(batch.shape[0],),
                              minval=int(continuous_noise),
                              maxval=len(sigmas))

  if continuous_noise:
    rng, noise_rng = jax.random.split(rng)
    used_sigmas = jax.random.uniform(key=noise_rng,
                                     shape=labels.shape,
                                     minval=sigmas[labels - 1],
                                     maxval=sigmas[labels])
  else:
    used_sigmas = sigmas[labels]

  used_sigmas = used_sigmas.reshape(batch.shape[0],
                                    *([1] * len(batch.shape[1:])))
  noise = jax.random.normal(key=sample_rng, shape=batch.shape) * used_sigmas
  perturbed_samples = batch + noise
  target = -1 / (used_sigmas**2) * noise
  scores = model(perturbed_samples, used_sigmas)

  assert target.shape == batch.shape
  assert scores.shape == batch.shape

  # Compute loss
  target = target.reshape(target.shape[0], -1)
  scores = scores.reshape(scores.shape[0], -1)
  loss = 0.5 * jnp.sum(jnp.square(scores - target),
                       axis=-1) * used_sigmas.squeeze()**2
  return reduce_fn(loss, reduction)


def sliced_score_matching_loss(batch,
                               model,
                               sigmas,
                               rng,
                               continuous_noise=False,
                               reduction="mean"):
  """Sliced score matching objective used to train NCSNs.
  
  Args:
    batch: A batch of data to compute the loss for.
    model: A noise-conditioned score network.
    sigmas: A noise schedule (list of standard deviations).
    rng: Random number generator key to sample sigmas.
    continuous_noise: If True, uses continuous noise conditioning.
    reduction: Type of reduction to apply to loss.
  
  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `data`;
    otherwise, it is scalar.
  """
  n_particles = 1.  # TODO: does not support more than 1 particle.
  rng, label_rng, sample_rng, score_rng = jax.random.split(rng, num=4)
  labels = jax.random.randint(key=label_rng,
                              shape=(batch.shape[0],),
                              minval=int(continuous_noise),
                              maxval=len(sigmas))

  if continuous_noise:
    rng, noise_rng = jax.random.split(rng)
    used_sigmas = jax.random.uniform(key=noise_rng,
                                     shape=labels.shape,
                                     minval=sigmas[labels - 1],
                                     maxval=sigmas[labels])
  else:
    used_sigmas = sigmas[labels]

  used_sigmas = used_sigmas.reshape(batch.shape[0],
                                    *([1] * len(batch.shape[1:])))
  noise = jax.random.normal(key=sample_rng, shape=batch.shape) * used_sigmas
  perturbed_samples = batch + noise

  dup_samples = perturbed_samples[:]
  dup_sigmas = used_sigmas[:]

  vectors = jax.random.rademacher(key=score_rng, shape=dup_samples.shape)

  # Compute gradients.
  first_grad = model(dup_samples, dup_sigmas)
  score_fn = lambda x: jnp.sum(model(x, dup_sigmas) * vectors)
  first_grad_v, second_grad = jax.value_and_grad(score_fn)(dup_samples)
  assert second_grad.shape == first_grad.shape

  # Score loss.
  first_grad = first_grad.reshape(dup_samples.shape[0], -1)
  score_loss = 0.5 * jnp.sum(jnp.square(first_grad), axis=-1)

  # Hessian loss.
  hessian_loss = jnp.sum(
      (vectors * second_grad).reshape(dup_samples.shape[0], -1), axis=-1)

  # Compute loss.
  score_loss = score_loss.reshape(n_particles, -1).mean(axis=0)
  hessian_loss = hessian_loss.reshape(n_particles, -1).mean(axis=0)
  loss = (score_loss + hessian_loss) * (used_sigmas.squeeze()**2)

  return reduce_fn(loss, reduction)


def diffusion_loss(batch,
                   model,
                   betas,
                   rng,
                   continuous_noise=False,
                   reduction="mean"):
  """Diffusion denoising probabilistic model loss.
  
  Args:
    batch: A batch of data to compute the loss for.
    model: A diffusion probabilistic model.
    betas: A noise schedule.
    rng: Random number generator key to sample sigmas.
    continuous_noise: If True, uses continuous noise conditioning.
    reduction: Type of reduction to apply to loss.
  
  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `data`;
    otherwise, it is scalar.
  """
  T = len(betas)
  rng, label_rng, sample_rng = jax.random.split(rng, num=3)
  labels = jax.random.randint(key=label_rng,
                              shape=(batch.shape[0],),
                              minval=int(continuous_noise),
                              maxval=T + int(continuous_noise))

  alphas = 1. - betas
  alphas_prod = jnp.cumprod(alphas)

  # if continuous_noise:
  alphas_prod = jnp.concatenate([jnp.ones((1,)), alphas_prod])
  rng, noise_rng = jax.random.split(rng)
  used_alphas = jax.random.uniform(key=noise_rng,
                                   shape=labels.shape,
                                   minval=alphas_prod[labels - 1],
                                   maxval=alphas_prod[labels])
  # else:
  #   used_alphas = alphas_prod[labels]

  used_alphas = used_alphas.reshape(batch.shape[0],
                                    *([1] * len(batch.shape[1:])))
  t = labels.reshape(batch.shape[0], *([1] * len(batch.shape[1:])))

  eps = jax.random.normal(key=sample_rng, shape=batch.shape)
  perturbed_sample = jnp.sqrt(used_alphas) * batch + jnp.sqrt(1 -
                                                              used_alphas) * eps

  # if continuous_noise:
  pred = model(perturbed_sample,
               jnp.sqrt(used_alphas))  # condition on noise level.
  # else:
  # pred = model(perturbed_sample, t)  # condition on timestep.

  loss = jnp.square(eps - pred)
  loss = jnp.mean(loss, axis=tuple(range(1, len(loss.shape))))
  assert loss.shape == batch.shape[:1]

  return reduce_fn(loss, reduction)
