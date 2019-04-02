#!/usr/bin/python2.6
#
# Copyright 2011 Emilie Gillet.

"""Command line tool to decompose a matrix stored in a text file."""

import logging
import matplotlib
import numpy
import optparse
import pylab
import sys

import beta_nmf


def main(args):
  parser = optparse.OptionParser()
  parser.add_option(
      '-b',
      '--beta',
      dest='beta',
      type=float,
      default=0.0,
      help='Value of the beta parameter')
  parser.add_option(
      '-r',
      '--num_components',
      dest='r',
      type=int,
      default=16,
      help='Number of components')
  parser.add_option(
      '-e',
      '--exponent',
      dest='exponent',
      type=float,
      default=2.0,
      help='Raise the input data to the e-th power')
  parser.add_option(
      '-i',
      '--iterations',
      dest='num_iterations',
      type=int,
      default=10,
      help='Number of iterations')
  parser.add_option(
      '-p',
      '--plot',
      dest='plot',
      action='store_true',
      default=True,
      help='Plot the results instead of saving them as text files')
  
  options, args = parser.parse_args(args)
  if len(args) != 1:
    logging.fatal('One, and only one input file must be provided')
    return
  r = options.r
  beta = options.beta
  x = numpy.loadtxt(args[0]).T
  x[x > 0] = x[x > 0] ** options.exponent
  x[x <= 1e-6] = 1e-6
  d, n = x.shape
  w = numpy.random.rand(d, r)
  h = numpy.random.rand(r, n)
  w, h, cost = beta_nmf.beta_nmf(x, w, h, beta, 3, options.num_iterations)
  if options.plot:
    for i in xrange(r):
      pylab.subplot(r, 2, i * 2 + 1)
      pylab.plot(10 * numpy.log10(w[:, i]))
      pylab.xlabel('Frequency')
      pylab.ylabel('Energy')
      pylab.subplot(r, 2, i * 2 + 2)
      pylab.plot(h[i, :])
      pylab.xlabel('Time')
      pylab.ylabel('Gain')
    pylab.show()


if __name__ == '__main__':
  main(sys.argv[1:])
