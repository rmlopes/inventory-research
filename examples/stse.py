from __future__ import division
import random
import logging
import logging.config
import math,operator, sys
import cPickle as pkl
from math import isnan, isinf, ceil
from functools import partial

from deap import tools,base,creator,algorithms, gp
from deap.creator import *
from deap.gp import Primitive

import numpy as np
from scipy.stats import poisson as pd, norm

from inventory.utils import loadcmdargs, setupdeap, runCoEA, printIndTree, cartesian, getfitcases, KLEINAU_SET
from inventory.discrete import discreteTC
from inventory.solvers import discreteSolver

log = logging.getLogger(__name__)

def isInvalid(s, Q):
    return (Q < 0 or Q > 1000 or s > 1000 or
            math.isinf(Q) or math.isinf(s) or
            math.isnan(Q) or math.isnan(s))

def evaluate(inds, data, curspecies = 0, costfun = discreteTC):
    '''Returns average cost over the dataset as the result.'''
    solutions = []
    if curspecies == 0:
        sfunindex = 0
        qfunindex = 1
    else:
        sfunindex = 1
        qfunindex = 0

    sfun = toolbox.compile(expr=inds[sfunindex])
    qfun = toolbox.compile(expr=inds[qfunindex])
    fit = .0
    for dsample in data:
        l,h,p,k,L = dsample[:-3]
        dL = pd(l*L)
        toolbox.__dict__['dL'] = dL

        s = sfun(l,h,p,k,L)
        Q = qfun(l,h,p,k,L)

        if isInvalid(s, Q):
            return 1e6
        else:
            s = int(round(s))
            Q = int(round(Q))
            if Q == 0: Q = 1
            fit += costfun(s,Q, l*L, k, p, l, h)

    return fit

def evaluateGap(inds, data, curspecies = 0, costfun = discreteTC):
    if curspecies == 0:
        sfunindex = 0
        qfunindex = 1
    else:
        sfunindex = 1
        qfunindex = 0

    sfun = toolbox.compile(expr=inds[sfunindex])
    qfun = toolbox.compile(expr=inds[qfunindex])
    fit = .0
    for dsample in data:
        l,h,p,K,L,sOpt,qOpt,optcost = dsample
        #dL = pd(l*L)
        try:
            s = int(round(sfun(*dsample[:-3])))
            Q = int(round(qfun(*dsample[:-3])))
            if Q == 0: Q = 1
        except ValueError:
            log.exception('ValueError while executing individual (NaN). Skipping test')
            continue
        except OverflowError:
            log.exception('OverflowError while executing individual (inf). Skipping test')
            continue

        icost = costfun(s,Q, l*L, K, p, l, h)

        log.info("optimTC: %f; evoTC: %f; optS, optQ: %d, %d; s, Q: %d, %d, inputs: %s" % (optcost,
            icost, sOpt, qOpt, s, Q, dsample))
        fit += (icost - optcost) / optcost

    return fit/len(data)

def evaluateOptimal(func, inputs):
    costs = [getcost(func,t) for t in inputs]
    return sum(costs),

def customRoulette(individuals, k):
    """Enable roulette for minimization problems"""
    s_inds = sorted(individuals, key=operator.attrgetter("fitness"), reverse=True)
    sumfit = float(sum(ind.fitness.values[0] for ind in s_inds))
    probs = map(lambda i: i.fitness.values[0]/sumfit, s_inds)
    probs = map(lambda p: (1.0 - p) / (len(probs)-1), probs)

    chosen = []
    for i in xrange(k):
        u = random.random()
        sum_ = 0
        for ind,nf in zip(s_inds,probs):
            sum_ += nf
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen

def customMut(individual, expr, pset):
    '''To handle multiple mutation operators'''
    r = random.random()
    if r >= 0.4:
        individual = gp.mutUniform(individual, expr, pset)
    else:
        #apply shrink
        individual = gp.mutShrink(individual)
    return individual

if __name__ == '__main__':
    logging.config.dictConfig({'version': 1,
                               'level': logging.DEBUG,
                               'disable_existing_loggers': False,
                               'propagate': 1})
    args = loadcmdargs()
    fitcases = getfitcases(vlist = KLEINAU_SET, costfun = discreteTC, solver = discreteSolver)
    traind = random.sample(fitcases, 72)
    testd = fitcases

    toolbox, stats, logbook = setupdeap(
            args, traind,
            ['lbd', 'h', 'p', 'k', 'L'],
            evalfun = evaluate,
            discrete = True)
    best = runCoEA(toolbox, stats, logbook, **vars(args))

    gap = evaluateGap(best, testd)
    print "Average gap on the complete set: ", gap

    #Python 2.6 does not support pygraphviz out of the box
    if 'pygraphviz' in sys.modules:
        printIndTree(best[0], '%s%s_best_stree' % (args.outdir, args.id,))
        printIndTree(best[1], '%s%s_best_Qtree' % (args.outdir, args.id,))

    print "\nSaving the best individuals for future usage/treatment..."
    fh =  open('%s%s_bestinds.pkl' % (args.outdir, args.id,), 'wb')
    pkl.dump((str(best[0]),str(best[1])), fh)

    print "\nSaving the logbook for future reference/treatment..."
    pkl.dump(logbook, open('%s%s_logbook.pkl' % (args.outdir, args.id,), 'wb'))
