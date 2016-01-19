#   InventoryResearch - A library for the study of inventory management
#   Copyright (C) 2015-2016 Rui L. Lopes
#
#   This file is part of InventoryResearch.
#
#   InventoryResearch is free software: you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   InventoryResearch is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with InventoryResearch.  If not, see <http://www.gnu.org/licenses/>.
"""
Provides the cost functions for the discrete model.
"""
from __future__ import division
import os
import math
import logging
import warnings
import cPickle as pkl

from scipy.stats import poisson

log = logging.getLogger(__name__)

try:
    pdtable = pkl.load(open(os.path.dirname(__file__) + '\cumpoisson_int2M.pkl', 'rb'))
except IOError:
    warnings.warn("Error loading pre-loaded poisson cumulative distribution function table. Use scipy.stats.poisson.cdf instead!")

def pdcdf(x, lbd):
    '''
    Returns the pre-calculated value for the cumulative distribution
    function of a Poisson distribution.

    :param lbd: demand distribution expected value (lambda)
    :param x: point to calculate the cdf (integer)

    :returns: P(X <= x)
    :rtype: float

    :raises Warning: if the pair (x, lbd) does not exist in the table, and the return value is calculated on-the-fly using :mod:`scipy:scipy.stats`

    :examples:

    >>> round(pdcdf(1, 25),4) == round(poisson.cdf(1,25))
    True
    >>> round(pdcdf(1e6, 25),4) == round(poisson.cdf(1e6,25))
    True
    '''
    try:
        return pdtable[x, lbd]
    except IndexError:
        warnings.warn("Input too big for the pre-calculated table: %d %d" % (x, lbd))
        return poisson.cdf(x, lbd)

def G(y, lL, p, h, cdf):
    '''
    Aggregates the holding and backorder costs in the discrete model.

    :param y: the function y parameter
    :param lL: demand distribution expected value
    :param p: backorder penalty
    :param h: on-hold cost
    :param cdf: function used to calculate the cumulative distribution of the demand

    :returns: the aggregated on-hold and backordered inventory cost
    :rtype: float
    '''
    return (h + p)*sum(cdf(i, lL) for i in xrange(0, y)) + p*(lL - y)

def discreteTC(s, Q, lL, K, p, lbd, h, *args, **kwargs):
    '''
    Calculates the total cost using the discrete model.
    As described in [FedergruenZheng1992]_ and [Zheng1992]_.

    :param s: reorder point
    :param Q: reorder quantity
    :param lL: demand distribution expected value
    :param K: order setup cost
    :param p: backorder penalty
    :param lbd: total demand
    :param h: holding cost

    :returns: total cost
    :rtype: float

    :example:

    >>> round(discreteTC(50, 7, 50, 1, 25, 50, 10), 2)
    95.46
    >>> round(discreteTC(56, 7, 50, 1, 100, 50, 10), 2)
    142.81
    >>> round(discreteTC(46, 6, 50, 1, 25, 50, 25), 2)
    153.35
    '''
    if isinstance(s, float):
        sround = int(round(s))
        warnings.warn("s is float, converting to integer. %.2f: %d" % (s,sround))
        s = sround
    if isinstance(Q, float):
        qround = int(math.round(Q))
        warnings.warn("Q is float, converting to integer. %.2f: %d" % (Q,qround))
        Q = qround

    cdf = pdcdf if 'cdf' not in kwargs.keys() else kwargs['cdf']
    log.debug("Poisson cdf is: %s" % (cdf,))

    setup = K * lbd / Q
    try:
        hp =  sum(G(i, lL, p, h, cdf) for i in xrange(s + 1, s + Q + 1)) / Q
    except OverflowError:
        return float('nan')

    log.debug("Setup cost: %.2f; Holding&Stockout cost: %.2f" % (setup, hp))
    return setup + hp


def _discreteTC(s, Q, dL1, dL2, K, p, lbd, r, v, *args, **kwargs):
    '''
    This wrapper takes two distributions of demand over the lead time
    (dL1 and dL2, respectively demand of type 1 and 2, see XXX-REF),
    combines them in a single discrete distribution and calls discreteTC.
    It only serves testing purposes
    '''
    x1, std1 = dL1.args
    x2, std2 = dL2.args
    xL = x1 + x2
    stdL= math.sqrt(std1**2 + std2**2)
    globalD = poisson(xL)
    return  discreteTC(s, Q, globalD, K, p, lbd, r*v)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
