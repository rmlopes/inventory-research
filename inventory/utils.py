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
"""Implements utilities for data generation and plotting,
and helpers for the DEAP genetic programming algorithm."""

import logging
import warnings
import argparse
import functools
import operator
import random
import math
from datetime import datetime

import numpy as np

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:
    warnings.warn('matplotlib not found.')

try:
    import pygraphviz as pgv
except ImportError:
    warnings.warn('pygraphviz not found.')
    warnings.warn('use of inventory.utils.printIndTree will result in error.')

from deap import tools,base,creator,algorithms, gp
log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

###############################################################################
# Data helpers                                                                #
###############################################################################

def _p(i):
    def __p(h):
        return i*h
    return __p

ZHENG_SET  = {
    'l': [5,25,50],
    'h': [1,10,25],
    'p': [5,10,25,100], #no further transformation
    'k': [1,5,25,100,1000],
    'L': [1]
}

KLEINAU_SET  = {
    'l': [1,5,10],
    'h': [1,5,10,50,100],
    'p': [2,3,5,10], #[_p(2), _p(3), _p(10)], this cannot be handled by cartesian
    'k': [1,10,100,500],
    'L': [1,2,5]
}

FZ_SET = {
    'l': [1,10],
    'h': [1,5,10,100],
    'p': [2,3,10], #[_p(2), _p(3), _p(10)], this cannot be handled by cartesian
    'k': [1,10,500],
    'L': [1,3]
}

def getfitcases(vlist = None, costfun = None, solver = None, backorders = True):
    """Returns a list of problem instances, based on the cartesian product of `vlist`

    :param vlist: the base list to build the cartesian product
    :param costfun: function used to calculate the cost
    :param solver: heuristic used to calculate (r,Q) parameters for each instance
    :param backorders: if `False` and `r < 0`, then `r = 0`

    :returns: list of instances
    :rtype: list
    """
    #ensure correct order by providing ordered labels
    fc = cartesian([vlist[idx] for idx in ['l','h','p','k','L']])
    fcOpt = []
    for i in xrange(len(fc)):
        #Federgruen and Zheng make p function of h
        if vlist != ZHENG_SET:
            fc[i,2] *= fc[i,1]

        if vlist != ZHENG_SET or fc[i,1] <= fc[i,2]:
            fcOpt.append(fc[i].tolist())
            lL = fc[i,0]*fc[i,-1]

            if costfun.__name__ != 'discreteTC':
                lL = norm(lL, math.sqrt(lL))

            fcOpt[-1].extend(solver(fc[i,0]*fc[i,-1], fc[i,3],
                                    fc[i,2], fc[i,0], fc[i,1])) #lL, K, p, l, h,

            if not backorders and fcOpt[-1][-2] < 0:
                fcOpt[-1][-2] = 0

            fcOpt[-1].append(costfun(int(round(fcOpt[-1][-2])),int(fcOpt[-1][-1]), lL, fc[i,3],
                                    fc[i,2], fc[i,0], fc[i,1]))
    return fcOpt

def cartesian(arrays, out=None):
    """
    Generates a cartesian product of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

#Sample generation
def genSample(originData, lowerDev=20, upperDev=50):
    """Generates random samples from the originData example, with each variable transformed
    between lowerDev and upperDev."""
    rndfactors = [(random.random()*(upperDev + lowerDev) - lowerDev)/100
                    for _ in range(len(originData))]
    log.debug(rndfactors)

    krndf = dict(zip(originData.keys(), rndfactors))
    sampleData = dict()
    datav = np.array(originData.values())
    sampleData.update(zip(originData.keys(), datav + rndfactors*datav))

    for t in sampleData.items():
        log.info("key: %s, rndratio: %.2f, value: %.3f" % (t[0], krndf[t[0]], t[1]))
    log.debug(sampleData)

    return sampleData

def getLabels():
    """Returns the labels for the commerce variables."""
    return ['xL1', 'sigma1','xL2', 'sigma2', 'A', 'B1', 'D', 'r', 'v', 'sOpt','qOpt']

###############################################################################
# DEAP helpers - Genetic Programming                                          #
###############################################################################
def protectedDiv(left, right):
    """Safe version of the operator :py:func:`~operator.div` to use with GP."""
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    except OverflowError:
        return 1

def protectedSqrt(value):
    """Safe version of the operator :py:func:`~math.sqrt` to use with GP."""
    try:
        return math.sqrt(value)
    except ValueError:
        return 1

def statsfun(distr, statistic, toolbox, discrete = False):
    """Wrapper (decorator) to use statistical functions in the GP function set.

    :param distr: the name of the distribution variable
    :param statistic: the name of the statistical function (cdf, pdf, etc)
    :param toolbox: DEAP :py:class:`~deap:deap.base.Toolbox`
    :param discrete: used to force the casting of the input

    :returns: paramterized function
    :rtype: callable
    """
    def istatsfun(input):
        d = toolbox.__dict__[distr]
        try:
            inp = int(input)
        except ValueError:
            return input
        except OverflowError:
            return input
        return getattr(d, statistic)(input if not discrete else inp)
    return istatsfun

def setupdeap(args, dataset, inputs, evalfun, discrete = True):
    """
    Builds the evolutionary algorithm.

    :returns: :py:class:`~deap:deap.base.Toolbox`, :py:class:`~deap:deap.tools.Statistics`, :py:class:`~deap:deap.tools.Logbook`
    :rtype: tuple
    """
    evalfun = functools.partial(evalfun, data = dataset)
    #DEAP initialization
    #new primitive set with 4 inputs
    pset = gp.PrimitiveSet("MAIN", len(inputs))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(protectedSqrt, 1)

    #pset.addEphemeralConstant("rand101", lambda: random.randint(-2,2))#lambda: random.randint(-5,5))
    #pset.addEphemeralConstant("1", lambda:1.0)
    #define the terminals set labels

    #Python 2.6 does not have dictionary inclusion
    psetdict = dict([('ARG%d' % (idx,), inp) for idx, inp in enumerate(inputs)])
    #psetdict = {'ARG{}'.format(idx): inp for idx, inp in enumerate(inputs)}
    pset.renameArguments(**psetdict)

    IND_SIZE = 10
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    statswrap = functools.partial(statsfun, toolbox = toolbox, discrete = discrete)
    if args.statsf:
        if not discrete:
            pset.addPrimitive(statswrap('dL1','pdf'), 1, name='dL1pdf')
            pset.addPrimitive(statswrap('dL2','pdf'), 1, name='dL2pdf')
            pset.addPrimitive(statswrap('dL1','cdf'), 1, name='dL1cdf')
            pset.addPrimitive(statswrap('dL2','cdf'), 1, name='dL2cdf')
            #pset.addPrimitive(statsfun('dL1','sf'), 1, name='dL1sf')
            #pset.addPrimitive(statsfun('dL2','sf'), 1, name='dL2sf')
        else:
            #poisson stats funs
            #pset.addPrimitive(statswrap('dL','pmf'), 1, name='pmf')
            pset.addPrimitive(statswrap('dL','cdf'), 1, name='cdf')
            #pset.addPrimitive(statswrap('dL','ppf'), 1, name='ppf')
            pset.addPrimitive(statswrap('dL','isf'), 1, name='isf')

    #toolbox = coop_base.toolbox
    #initialize coevolution species and representatives
    toolbox.register("get_best", tools.selBest, k=1)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("species", tools.initRepeat, list, toolbox.individual, args.popSize)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalfun)
    #toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.5, fitness_first=False)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    #collect stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "species", "evals", "std", "min", "avg", "max"
    return toolbox, stats, logbook

def runCoEA(toolbox, stats, logbook, numGen = 100, rnd = False, cxp = .6, mutp = 1.0, **kwargs):
    """Initializes and runs the coevolutionary algorithm.

    :returns: the best individual of the run."""
    #RUN IT
    log.info('CxP: %f; MutP: %f' % (cxp, mutp))
    log.info("Creating the population...")
    NUMSPECIES = 2
    species_index = list(range(NUMSPECIES))
    species = [toolbox.species() for _ in range(NUMSPECIES)]
    representatives = [random.choice(species[i]) for i in range(NUMSPECIES)]
    representatives[0].fitness.values = (toolbox.evaluate(representatives, curspecies = 0),)
    representatives[1].fitness.values = (toolbox.evaluate(representatives[:: -1], curspecies = 1),)
    #pop = toolbox.population(n=4)
    bestsofar =  [random.choice(species[i]) for i in range(NUMSPECIES)]
    bestsofar[0].fitness.values = (toolbox.evaluate(bestsofar, curspecies = 0),)
    bestsofar[1].fitness.values = (toolbox.evaluate(bestsofar[:: -1], curspecies = 1),)
    log.info("Running the algorithm...")
    for g in xrange(numGen):
        # Initialize a container for the next generation representatives
        next_repr = [None] * len(species)
        for (i, s), j in zip(enumerate(species), species_index):
            # Vary the species individuals
            s = algorithms.varAnd(s, toolbox, cxp, mutp)
            # Get the representatives excluding the current species
            r = representatives[:i] + representatives[i+1:]
            for ind in s:
                log.debug("Evaluating species %d"%(j,))
                log.debug(r)
                # Evaluate and set the individual fitness
                ind.fitness.values = (toolbox.evaluate([ind] + r, curspecies = j),)

            # Select the individuals
            species[i] = toolbox.select(s, len(s))  # Tournament selection
            if not rnd:
                next_repr[i] = toolbox.get_best(species[i])[0]   # Best selection
            else:
                next_repr[i] = random.choice(species[i])

            #Update stats
            record = stats.compile(s)
            logbook.record(gen=g, species=j, evals=len(s), **record)
            log.info(logbook.stream)

        for i in species_index:
            if next_repr[i].fitness.values <= representatives[i].fitness.values:
                representatives[i] = next_repr[i]
        #for index in range(len(representatives)):
            if representatives[i].fitness.values < bestsofar[i].fitness.values:
                bestsofar[i] = representatives[i]

    log.info("Done!\n")
    log.info(bestsofar[0])
    log.info(bestsofar[1])
    return bestsofar

def printIndTree(ind, label):
    nodes, edges, labels = gp.graph(ind)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw("%s.pdf" % (label,))

def loadcmdargs():
    "Parses the command line arguments to use with scripts."
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--id', type=str,
                        default=datetime.now().strftime("%Y-%m-%d_%I-%M-%S"),
                        help='''Id for the filenames generated in the run.
                        Defaults to current date and time.''')
    parser.add_argument('-o', '--outdir', type=str, default='output/',
                        help='Specify ouput directory. (default=output)')
    parser.add_argument('-n', '--numGen', type=int, default=50,
                        help='GP: number of generations (default=50)')
    parser.add_argument('-s', '--popSize', type=int, default=100,
                        help='GP: population size (for each species, default = 100)')
    parser.add_argument('-sf', '--statsf', action='store_true',
                        help='GP: use statistical functions in the primitive set.')
    parser.add_argument('-rnd', '--rnd', action='store_true', default = False,
                        help='GP: choose representatives randomly')
    parser.add_argument('-cxp', '--cxp', type=float, default=.6,
                        help='GP: crossover probability')
    parser.add_argument('-mutp', '--mutp', type=float, default=1.0,
                        help='GP: mutation probability')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'),
                        default='data/rndSQdata30.pkl',
                        help='''Input filename (needed for both commands,
                              defaults to rndSQdata30.pkl)''')
    return parser.parse_args()

###############################################################################
# Plot helpers                                                                #
###############################################################################

def computePlot(data):
    "Computes the cost surface for every entry in data."
    labels = getLabels()
    alldata = [dict(zip(labels, ex)) for ex in data]
    log.debug(alldata)

    tosave = list()
    for d in alldata:
        #define the range based on the optimum
        optS, optQ = (d['sOpt'],d['qOpt'])
        u = np.linspace(.5 * optS, 2 * optS, 100)
        v = np.linspace(.5 * optQ, 2 * optQ, 100)
        x = len(u) * [u]
        y = np.column_stack(len(v) * (v,))

        fun = getVectorized(d)
        z = fun(x,y)
        tosave.append((x,y,z))
    return tosave

def plotETRC(t, show=True):
    """Plots a wireframe from surface data."""
    #FIXME: open multiple figures necessary
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.plot_wireframe(*t)
    if show:
        plt.show()
    return p

def plotETRCsurface(t, show=True, colormap = None):
    """Plots a surface for a data record."""
    #FIXME: open multiple figures necessary
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.plot_surface(*t, rstride = 1, cstride = 1,
                        cmap = colormap, linewidth = 0)
    if show:
        plt.show()
    return p
