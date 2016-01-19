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
Provides the cost functions for the continuous model.
"""
from __future__ import division
import logging
import math

from scipy.stats import norm, poisson, rv_continuous
from scipy.integrate import dblquad, quad
from numpy import vectorize

__all__ = ['ecomStockoutP','alphaTCecom','sfactorTCecom',
           'vectorizeTCecom', 'alphaTC', 'sfactorTC', 'betaTC']

log = logging.getLogger(__name__)

def _orderingCost(A, D, Q):
    return A * D / float(Q)

def _expectedOnHandCost(s, Q, dL1, v, r):
    xL1, sigma1 = dL1.args
    return (Q/2.0 + s - xL1) * v * r

def _stockoutCost(B1, D, Q, pso):
    return float(D) / Q * B1 * pso

def _ecomStockoutP2(s, Q, dL1, dL2):
    xL1, sigma1 = dL1.args
    xL2, sigma2 = dL2.args
    lb1 = lambda u2: (s + Q -xL2 -xL1 - u2*sigma2) / float(sigma1)
    lb2 = (Q - xL2) / float(sigma2)
    return dblquad(lambda u2,u1: (1 / (2 * math.pi)) *
                                 math.exp(- u1**2/2.0 - u2**2/2.0),
                   lb2, float('inf'),
                   lb1, lambda u2: float('inf'),
                   (), #extra args
                   1.49e-3, 1.49e-3 #epsabs, epsrel,increased by 10e5 compared to default
                   )[0]

def ecomStockoutP(dL1, dL2, s, Q):
    """
    Calculate the stockout probability for the e-commerce model.

    :param dL1: distribution of the demand of type 1 (must be continuous)
    :param dL2: distribution of the demand of type 2 (must be continuous)
    :param s: the reorder point
    :param Q: the reorder quantity

    :returns: stockout probability
    :rtype: float
    """
    assert (isinstance(dL1.dist, rv_continuous) and
            isinstance(dL2.dist, rv_continuous)), \
            "The distributions must be continuous"

    pso2 = _ecomStockoutP2(s, Q, dL1, dL2)
    p1s = dL1.sf(s)
    p2Q = dL2.cdf(Q)
    pso1 = p1s * p2Q

    log.debug("-ecomStockoutP- pStockout1: %.2E; pStockout2: %.2E" % (pso1, pso2))
    return pso1 + pso2

def alphaTCecom(s, Q, dL1, dL2, A, B1, D, r, v, *args, **kwargs):
    """
    Calculate the total cost for the e-commerce model
    (alpha service level, B1 items).

    :param dL1: distribution of the demand of type 1 (must be continuous)
    :param dL2: distribution of the demand of type 2 (must be continuous)
    :param s: the reorder point
    :param Q: the reorder quantity
    :param A: the setup cost
    :param B1: the backorder penalty
    :param D: the total demand
    :param r: inventory carrying charge
    :param v: unit variable cost

    :returns: total cost
    :rtype: float
    """
    assert (isinstance(dL1.dist, rv_continuous) and
            isinstance(dL2.dist, rv_continuous)), \
            "The distributions must be continuous"

    xL1, sigma1 = dL1.args
    xL2, sigma2 = dL2.args
    pso = ecomStockoutP(dL1, dL2, s, Q)

    OC = _orderingCost(A, D, Q)
    OH = _expectedOnHandCost(s, Q, dL1, v, r)
    SC = _stockoutCost(B1, D, Q, pso)

    log.debug("-alphaTCecom- pStockout: %.2E; OC: %.2f; HC: %.2f; SC: %.2f" % (pso, OC, OH, SC))
    return OC + OH + SC

def sfactorTCecom(k, Q, dL1, dL2, A, B1, D, r, v, *args, **kwargs):
    """
    Calculate the total cost for the e-commerce model based on the service factor
    (alpha service level, B1 items).

    :param dL1: distribution of the demand of type 1 (must be continuous)
    :param dL2: distribution of the demand of type 2 (must be continuous)
    :param k: the service factor
    :param Q: the reorder quantity
    :param A: the setup cost
    :param B1: the backorder penalty
    :param D: the total demand
    :param r: inventory carrying charge
    :param v: unit variable cost

    :returns: total cost
    :rtype: float
    """
    assert (isinstance(dL1.dist, rv_continuous) and
            isinstance(dL2.dist, rv_continuous)), \
            "The distributions must be continuous"

    xL1, sigma1 = dL1.args
    xL2, sigma2 = dL2.args
    #FIXME: is this s calculated correctly
    s = xL1 + k*sigma1
    pso = ecomStockoutP(dL1, dL2, s, Q)

    OC = _orderingCost(A, D, Q)
    OH = (Q/2.0 + k*sigma1) * r * v
    SC = _stockoutCost(B1, D, Q, pso)

    log.debug("-sfactorTCecom- s: %d; pStockout: %.2E; OC: %.2f; HC: %.2f; SC: %.2f" %
                (s, pso, OC, OH, SC))
    return OC + OH + SC

def vectorizeTCecom(xL1, sigma1, xL2, sigma2, tcOpt = 0, **kwargs):
    '''
    Vectorize the cost function :func:`.alphaTCecom`.

    :param xL1: the mean demand of type 1.
    :param xL2: the mean demand of type 2.
    :param sigma1: the standard deviation for the demand of type 1.
    :param sigma2: the standard deviation for the demand of type 2.

    :returns: vectorized total cost function
    :rtype: callable
    '''
    assert tcOpt >= 0, "Optimal total cost cannot be less than 0."

    denom = tcOpt if tcOpt > 0 else 1
    dL1 = norm(xL1, sigma1)
    dL2 = norm(xL2, sigma2)

    fun = vectorize(lambda s,Q: (alphaTCecom(s, Q, dL1, dL2, **kwargs) - tcOpt)/denom)
    log.debug(kwargs)
    return fun

def alphaTC(s, Q, dL, A, B1, D, h, **kwargs):
    """
    Calculate the total cost for an (s, Q) inventory control policy
    (alpha service level, B1 items).

    :param dL: distribution of the demand (must be continuous)
    :param s: reorder point
    :param Q: reorder quantity
    :param A: setup cost (K)
    :param B1: backorder penalty (p)
    :param D: total demand (lambda)
    :param h: on-hold cost

    :returns: total cost
    :rtype: float
    """
    assert isinstance(dL.dist, rv_continuous), "The distribution must be continuous"

    xL = dL.args[0]
    OC = _orderingCost(A, D, Q)
    HC = (Q/2.0 + s - xL) * h
    SC = ((D * B1) / Q) * dL.sf(s)

    log.debug("-alphaTC- OC: %.2f; HC: %.2f; SC: %.2f" % (OC, HC, SC))
    return OC + HC + SC

def sfactorTC(k, Q, dL, A, B1, D, h, **kwargs):
    """
    Calculate the total cost for an (s, Q) inventory control policy based
    on the service factor (alpha service level, B1 items).

    :param dL: distribution of the demand (must be continuous)
    :param k: service factor
    :param Q: reorder quantity
    :param A: setup cost (K)
    :param B1: backorder penalty (p)
    :param D: total demand (lambda)
    :param h: on-hold cost

    :returns: total cost
    :rtype: float
    """
    assert isinstance(dL.dist, rv_continuous), "The distribution must be continuous"

    xL,std  = dL.args
    OC = _orderingCost(A, D, Q)
    HC = (Q/2.0 + k*std) * h
    BC = ((D * B1) / Q) * norm.sf(k)

    log.debug("-sfactorTC- OC: %.2f; HC: %.2f; SC: %.2f" % (OC, HC, BC))
    return OC + HC + BC

def betaTC(s, Q, dL, A, B1, D, rv, *args, **kwargs):
    """
    Calculate the total cost for an (s, Q) inventory control policy
    (beta service level, continuous approzimation of the discrete model).

    :param dL: distribution of the demand (must be continuous)
    :param s: reorder point
    :param Q: reorder quantity
    :param A: setup cost (K)
    :param B1: backorder penalty (p)
    :param D: total demand (lambda)
    :param rv: on-hold cost (h)

    :returns: total cost
    :rtype: float
    """
    assert isinstance(dL.dist, rv_continuous), "The distribution must be continuous"

    def G(y):
        return (rv + B1)*quad(dL.cdf, 0, y)[0] + B1*(dL.args[0] - y)

    setup = _orderingCost(A, D, Q)
    hp =  quad(G, s + 1, s + Q + 1)[0] / Q

    log.debug("-betaTC- Setup cost: %.2f; Holding&Stockout cost: %.2f" % (setup, hp))
    return setup + hp

#######################################################
# TESTING HELPER                                      #
#######################################################
def _wrapdemand(fun, s, Q, dL1, dL2, A, B1, D, r, v, *args, **kwargs):
    '''
    This wrapper takes two distributions of demand over the lead time
    (dL1 and dL2, respectively demand of type 1 and 2, see XXX-REF),
    combines them in a single normal distribution and calls fun.
    It only serves testing purposes
    '''
    assert (isinstance(dL1.dist, rv_continuous) and
            isinstance(dL2.dist, rv_continuous)), \
            "The distributions must be continuous"

    x1, std1 = dL1.args
    x2, std2 = dL2.args
    xL = x1 + x2
    stdL= math.sqrt(std1**2 + std2**2)
    globalD = norm(xL, stdL)
    return  fun(s, Q, globalD, A, B1, D, r*v)
