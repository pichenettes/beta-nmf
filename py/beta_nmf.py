#!/usr/bin/python2.6
#
# Copyright 2011 Emilie Gillet.

"""NMF with beta divergence."""

import logging
import numpy


TINY = numpy.finfo(float).tiny


def _beta_divergence_specialization_euclidian(x, y):
  """Compute the half euclidian distance between two matrices.
  
  Args:
    x, y: matrices to compare.
    
  Returns:
    Euclidian distance.
  """
  return ((x - y) ** 2).sum() / 2
  

def _beta_divergence_specialization_kullback_leibler(x, y):
  """Compute the KL divergence between two matrices.
  
  Args:
    x, y: matrices to compare.
    
  Returns:
    KL divergence.
  """
  small = x < TINY
  big = numpy.logical_not(small)
  return (x[big] * numpy.log(x[big] / y[big]) - x[big] + y[big]).sum() + \
      y[small].sum()

  
def _beta_divergence_specialization_itakura_saito(x, y):
  """Compute the Itakura-Saito divergence between two matrices.
  
  Args:
    x, y: matrices to compare.
    
  Returns:
    Itakura-Saito divergence.
  """
  d = (x / y - numpy.log(x / y)).sum() - numpy.prod(x.shape)
  return d


_BETA_DIVERGENCE_SPECIALIZATIONS = {
  2.0: _beta_divergence_specialization_euclidian,
  1.0: _beta_divergence_specialization_kullback_leibler,
  0.0: _beta_divergence_specialization_itakura_saito
}


def beta_divergence(x, y, beta):
  """Compute the generalized beta-divergence between two matrices.
  
  Args:
    x, y: matrices to compare.
    beta: value of the beta parameter. Particular cases:
      beta = 0: IS divergence
      beta = 1: KL divergence
      beta = 2: Euclidian distance.
    
  Returns:
    Beta-divergence.
  """
  if beta in _BETA_DIVERGENCE_SPECIALIZATIONS:
    return _BETA_DIVERGENCE_SPECIALIZATIONS[beta](x, y)
  else:
    d = (x ** beta + (beta - 1) * y ** beta - beta * x * y ** (beta - 1)).sum()
    return d / (beta * (beta - 1))


def _update(v, v_est, w, hT, beta):
  """Compute the multiplicative update for a NMF problem, using beta divergence.
  
  Args:
    v: matrix to approximate.
    v_est: current approximation of v.
    w: pattern dictionary matrix.
    h: transposed pattern activation matrix.
    beta: specify which divergence metric to minimize.
    
  Returns:
    Multiplicative update to the matrix w.
  """
  gamma = 1
  if beta == 1:
    num = numpy.dot(v / v_est, hT)
    # TODO(pichenettes): Optimize with sum and tile.
    den = numpy.dot(numpy.ones(v_est.shape), hT)
  elif beta == 2:
    num = numpy.dot(v, hT)
    den = numpy.dot(w, numpy.dot(hT.T, hT))
  else:
    num = numpy.dot(v * v_est ** (beta - 2), hT)
    den = numpy.dot(v_est ** (beta - 1), hT)
    if beta < 1:
      gamma = 1 / (2 - beta)
    elif beta > 2:
      gamma = 1 / (beta - 1)
  update = num / den
  if gamma:
    update = update ** gamma
  return update


def beta_nmf(v, w, h, beta, update=3, num_iterations=100):
  """Non-negative matrix factorization with beta divergence.
  
  Args:
    v: matrix to approximate.
    w: initial pattern dictionary matrix.
    h: initial pattern activation matrix.
    beta: specify which divergence metric to minimize.
      Beta can be a scalar, or a vector specifying the value of beta to use
      for each iteration.
    update: bitmask specifying what to optimize:
      1: estimate dictionary given activations.
      2: estimate activations given dictionary.
      3: estimate both dictionary and activations. w will be scaled so
         that ||w||_1 = 1
    num_iterations: number of iterations to run. When beta is a vector, this
      parameter is not used ; the size of beta is taken as the number of
      iterations.
      
  Returns:
    w: new value of the pattern dictionary matrix.
    h: new value of the pattern activation matrix.
    cost: value of the divergence at each iteration.
  """
    
  cost = numpy.zeros((num_iterations + 1,))
  f, n = v.shape
  r = w.shape[1]
  
  if not hasattr(beta, 'size'):
    beta = numpy.ones((num_iterations,)) * beta

  v_est = numpy.dot(w, h)
  cost[0] = beta_divergence(v, v_est, beta[0])
  
  if update == 0:
    cost[1:] = cost[0]
    return w, h, cost
  
  warning_displayed = False
  for i in xrange(len(beta)):
    if update & 1:
      w *= _update(v, v_est, w, h.T, beta[i])
      v_est = numpy.dot(w, h)
    if update & 3:
      h *= _update(v.T, v_est.T, h.T, w, beta[i]).T
      v_est = numpy.dot(w, h)
    # We need to apply scaling to account for gain indeterminacy.
    if update == 3:
      scale = w.sum(0)
      w /= scale
      h *= numpy.tile(scale.reshape((r, 1)), (1, n))
    cost[i + 1] = beta_divergence(v, v_est, beta[i])
    
    if cost[i + 1] > cost[i] and not warning_displayed:
      logging.warning('Non-increasing divergence')
      warning_displayed = True

  return w, h, cost
