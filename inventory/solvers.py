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
Implements the heuristics and optimal algorithms.
"""
from __future__ import division
import logging
import warnings
import math
from functools import partial

from scipy.stats import norm, poisson
from scipy.optimize import minimize, minimize_scalar
import numpy as np

from .discrete import discreteTC

try:
    import matlab.engine
except ImportError:
    warnings.warn('matlab.engine not found. Calls to matlabSolver will result in error.')

from .discrete import pdcdf, G

log = logging.getLogger(__name__)

def _matlabSolver(samples):
    '''
    Derive and solve the ecommerce ETRC for s and Q (using Matlab). Requires Matlab engine
    and scripts.
    '''
    eng = matlab.engine.start_matlab()
    sQvalues = []

    for sample in samples:
        sortedData = (sample['xL1'], sample['sigma1'],
                sample['xL2'], sample['sigma2'],
                sample['A'], sample['B1'], sample['D'], sample['r'], sample['v'] )
        s,Q = eng.iteration(*sortedData, nargout = 2)
        sQvalues.append(sortedData + (s,Q))
    eng.quit()
    return sQvalues

def eoq(K, lbd, h):
    '''Calculates the order quantity according to the Economic Order Quantity (EOQ) model, without planned backorders.

    :param K: order setup cost
    :param lbd: total demand
    :param h: holding cost

    :returns: reorder quantity
    :rtype: float
    '''
    return math.sqrt(2*K*lbd/h)

def _sequentialQ(k, A, B1, D, rv, **kwargs):
    return eoq(A, D, rv)*math.sqrt(1 + (B1/A)*(norm.sf(k)))

def _sequentialK(Q, std, A, B1, D, rv, **kwargs):
    assert Q > 0, 'A negative order quantity is not possible!'
    try:
        return math.sqrt(2 * math.log(D * B1 / (math.sqrt(2 * math.pi) * Q * rv * std)))
    except ValueError:
        return -1

def alphaSolver(x, std, A, B1, D, rv, *args, **kwargs):
    '''
    Simultaneous (iterative) solver for alpha service level in traditional retail.

    :param x: demand distribution expected value
    :param std: demand distribution standard deviation
    :param A: setup cost (K)
    :param B1: backorder penalty (p)
    :param D: total demand (lambda)
    :param rv: on-hold cost (h)

    :returns: service factor, reorder quantity
    :rtype: tuple
    '''
    N = 100 if not 'N' in kwargs.keys() else kwargs['N']
    numQ = partial(_sequentialQ, A=A, B1=B1, D=D, rv=rv)
    numK = partial(_sequentialK, std=std, A=A, B1=B1, D=D, rv=rv)
    lastk = 0
    Q = eoq(A,D,rv)
    for i in xrange(N):
        k = numK(Q)
        log.debug( "%d - Q:%.3f; k: %.3f" % (i, Q, k))
        if k < 0:
            k = 0
            lastk = 0
        Q = numQ(k)
        if abs(k - lastk) < 1e-2:
            break
        lastk = k
    return k, Q

def betaSolver(lL, K,p,l,h):
    """
    Calculates (r, Q) parameters from [Zheng1992]_ heuristic.

    :param lL: demand distribution expected value
    :param K: order setup cost
    :param p: backorder penalty
    :param l: total demand
    :param h: holding cost

    :returns: reorder point, reorder quantity
    :rtype: tuple

    :examples:

    >>> import functools
    >>> map(functools.partial(round, ndigits = 1), betaSolver(50, 1, 25, 50, 10))
    [48.9, 3.7]
    >>> map(functools.partial(round, ndigits = 1), betaSolver(50, 5, 25, 50, 10))
    [47.6, 8.4]
    >>> map(functools.partial(round, ndigits = 1), betaSolver(50, 25, 25, 50, 10))
    [44.7, 18.7]
    >>> map(functools.partial(round, ndigits = 1), betaSolver(50, 100, 25, 50, 10))
    [39.3, 37.4]
    >>> map(functools.partial(round, ndigits = 1), betaSolver(50, 1000, 25, 50, 10))
    [16.2, 118.3]
    """
    Q = math.sqrt((2*l*K*(h + p)) / (h*p))
    r = lL - Q*(h/(h + p))
    return r, Q

def zhengNumeric(lL, K,p,l,h,L, costfun=discreteTC, minmethod='cobyla', roundmethod = round):
    """
    Calculates the reorder quantity (Q) parameter from [Zheng1992]_ heuristic,
    and finds the corresponding reorder point (r) through numerical optimisation
    (using constrained optimization by linear approximation).

    :param lL: demand distribution expected value
    :param K: order setup cost
    :param p: backorder penalty
    :param l: total demand
    :param h: holding cost
    :param L: lead time
    :param costfun: cost function to be minimised when calculating the reorder point
    :param minmethod: optimisation algorithm to use with :func:`~scipy:scipy.optimize.minimize`
    :param roundmethod: function used to round the floating point reorder quantity

    :returns: reorder point, reorder quantity
    :rtype: tuple
    """

    #methods that will not converge: CG, bfgs,
    rd, Qd = betaSolver(lL, K, p, l, h)
    Q = int(roundmethod(Qd))
    #if Q == 0: Q = 1
    cfun = partial(costfun, Q = Q, lL = lL, K = K, p = p, lbd = l, h=h)
    x0 = 100#int(round(rd))#np.array([1.3*rd, 0.7*rd, 0.8*rd, 1.9*rd, 1.2*rd])
    r = minimize(cfun, x0,  method=minmethod, tol=1e-16).x#,
                 #options={'xtol': 1e-8, 'disp': False}).x
    #r = minimize_scalar(cfun, method=minmethod).x
    #print len(r.x)
    return r, Q #r.x[0], Q
    #
    #return r,Q

def gallego(lL, K,p,l,h,L, costfun=discreteTC, minmethod='cobyla'):
    """
    Calculates the reorder quantity (Q) parameter from [Gallego1998]_ heuristic,
    and finds the corresponding reorder point (r) through numerical optimisation
    (using constrained optimization by linear approximation).

    :param lL: demand distribution expected value
    :param K: order setup cost
    :param p: backorder penalty
    :param l: total demand
    :param h: holding cost
    :param L: lead time
    :param costfun: cost function to be minimised when calculating the reorder point
    :param minmethod: optimisation algorithm to use with :func:`~scipy:scipy.optimize.minimize`
    :param roundmethod: function used to round the floating point reorder quantity

    :returns: reorder point, reorder quantity
    :rtype: tuple
    """
    #methods that will not converge: CG, bfgs,
    rd, Qd = betaSolver(lL, K, p, l, h)
    #Qd = eoq(K, l, h)
    Q = Qd * min(math.sqrt(2), (1 + (((h + p)*L)/(2*K))**2)**.25)
    #s, Q, lL, A, B1, D, rv,
    if minmethod.lower() == 'gallego':
        r = rd
    else:
        #must be ceil not round, inorder to reproduce the 0,32% reported in original
        Q = int(math.ceil(Q))
        #Q = int(round(Q)) #0.41% error
        if Q == 0: Q = 1
        cfun = partial(costfun, Q = Q, lL = lL, K = K, p = p, lbd = l, h=h)
        x0 = 100
        r = minimize(cfun, x0,  method=minmethod, tol=1e-16).x

    return r, Q

def kleinauGP(lL, K,p,l,h,L):
    """
    Calculates the reorder point and quantity parameters from [KleinauThonemann2004]_
    full Genetic Programming solution.

    :param lL: demand distribution expected value
    :param K: order setup cost
    :param p: backorder penalty
    :param l: total demand
    :param h: holding cost
    :param L: lead time

    :returns: reorder point, reorder quantity
    :rtype: tuple
    """
    r = poisson.ppf(1 - math.sqrt(h / p), mu = l*L)
    Q = math.sqrt(L + (K / h) + math.sqrt(2.1029*l*(K + h)*math.sqrt(K/h)))
    return r, Q

def kleinauNum(lL, K,p,l,h,L, minmethod='cobyla'):
    """
    Calculates the reorder quantity (Q) parameter from [KleinauThonemann2004]_ hybrid GP
    approach, and finds the corresponding reorder point (r) through numerical
    optimisation (using constrained optimization by linear approximation).

    :param lL: demand distribution expected value
    :param K: order setup cost
    :param p: backorder penalty
    :param l: total demand
    :param h: holding cost
    :param L: lead time
    :param costfun: cost function to be minimised when calculating the reorder point
    :param minmethod: optimisation algorithm to use with :func:`~scipy:scipy.optimize.minimize`
    :param roundmethod: function used to round the floating point reorder quantity

    :returns: reorder point, reorder quantity
    :rtype: tuple
    """
    Q = math.sqrt((l*K)/h) + (l*((l + L) + (math.sqrt(l)*(l + L))**(1/6)))**(1/6) + math.sqrt(l)*((((K*L)/p)*math.sqrt(math.sqrt(l)*(K/h)))**.25)
    Q = int(round(Q))
    cfun = partial(discreteTC, Q = Q, lL = lL, K = K, p = p, lbd = l, h=h)
    x0 = 100
    r = minimize(cfun, x0,  method=minmethod, tol=1e-16).x
    return r, Q

def _deltaG(y, lL, p, h, cdf, G):
    return G(y + 1) - G(y)

def discreteSolver(lL, K, p, l, h, cdf = pdcdf):
    """
    Calculates the optimal reorder point and quantity parameters as presented by [FedergruenZheng1992]_.

    :param lL: demand distribution expected value
    :param K: order setup cost
    :param p: backorder penalty
    :param l: total demand
    :param h: holding cost
    :param cdf: function used to calculate the cumulative distribution function of a Poisson

    :returns: reorder point, reorder quantity
    :rtype: tuple

    :example:

    >>> discreteSolver(50, 1, 25, 50, 10)
    (50, 7)
    >>> discreteSolver(50, 5, 25, 50, 10)
    (48, 12)
    >>> discreteSolver(50, 25, 25, 50, 10)
    (44, 23)
    >>> discreteSolver(50, 100, 25, 50, 10)
    (38, 40)
    >>> discreteSolver(50, 1000, 25, 50, 10)
    (15, 120)
    """
    L = 0
    _G = partial(G, lL = lL, p = p, h = h, cdf = cdf)
    dG = partial(_deltaG, lL = lL, p = p, h = h, cdf = cdf, G = _G)
    while dG(L) < 0:
        L = L + 1

    S = K*l + _G(L)
    Q = 1
    C = S
    r = L - 1
    R = L + 1

    while True:
        if _G(r) <= _G(R):
            if C <= _G(r):
                break
            else:
                S = S + _G(r)
                r = r - 1
        elif C <= _G(R):
                break
        else:
            S = S + _G(R)
            R = R + 1
        Q = Q + 1
        C = S / Q
    return r, Q



if __name__ == "__main__":
    import doctest
    doctest.testmod()
