.. example_discrete

Examples
========

The following code exemplifies how to use the library, in particular the functions regarding
the discrete model. The code can be run using the corresponding iPython notebook :file:`example_discrete.ipynb`.


.. code:: python

    from __future__ import division

    import logging
    from math import sqrt,ceil
    from operator import itemgetter

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import poisson, norm, rv_continuous

    from inventory.discrete import discreteTC
    from inventory.solvers import discreteSolver, gallego, kleinauGP,kleinauNum, zhengNumeric
    from inventory.utils import KLEINAU_SET as KS, ZHENG_SET as ZS, getfitcases

.. code:: python

    backorders = True
    fitcases = getfitcases(vlist = KS, costfun=discreteTC,
                           solver = discreteSolver, backorders = backorders)
    print len(fitcases)

.. code:: python

    #Heuristics and Gaps to optimal - GALLEGO
    gallego_results = []
    for l, h, p, K, L, sOpt, qOpt, costOpt in fitcases:
        sg, Qg = gallego(l*L, K, p, l, h, L, minmethod='cobyla')
        #sg = 0 if sg <0 else int(round(sg))
        sg = int(round(sg))
        if not backorders and sg < 0:
            sg = 0
        Qg = int(round(Qg))
        if Qg == 0: Qg = 1
        gcost = discreteTC(sg, Qg, l*L, K, p, l, h)
        gallego_results.append((l, h, p, K, L, sOpt, qOpt, sg,Qg, gcost,
                               ((gcost - costOpt) / costOpt)*100))

    #for g in gallego_results: print g

    print sum(abs(fc[-1]) for fc in gallego_results) / len(fitcases)


.. code:: python

    #Heuristics and Gaps to optimal - ZHENG numeric r
    zhengnum_results = []
    for l, h, p, K, L, sOpt, qOpt, costOpt in fitcases:
        sk, Qk = zhengNumeric(l*L, K, p, l, h,L, roundmethod=ceil)
        rd, Qd = betaSolver(l*L, K, p, l, h)
        if not backorders and sk < 0:
            sk = 0
        skr = int(round(sk))
        #with ceil, error is smaller
        #Qk is already rounded
        #Qdr = int(round(Qd))
        gcost = discreteTC(skr, Qk, l*L, K, p, l, h)
        zhengnum_results.append((l, h, p, K, L, sOpt, qOpt, costOpt, rd, sk,Qk, gcost,
                                ((gcost - costOpt) / costOpt) * 100))

    #for g in zhengnum_results: print g

    print sum(abs(fc[-1]) for fc in zhengnum_results) / len(fitcases)


.. code:: python

    #Heuristics and Gaps to optimal - KLEINAU GP
    kleinaugp_results = []
    for l, h, p, K, L, sOpt, qOpt, costOpt in fitcases:
        sk, Qk = kleinauGP(l*L, K, p, l, h,L)
        #sg = 0 if sg <0 else int(round(sg))
        if not backorders and sk < 0:
            sk = 0
        sk = int(round(sk))
        Qk = int(round(Qk))
        gcost = discreteTC(sk, Qk, l*L, K, p, l, h)
        kleinaugp_results.append((l, h, p, K, L, sOpt, qOpt, costOpt, sk,Qk, gcost,
                                 ((gcost - costOpt) / costOpt)*100))

    #for g in kleinaugp_results: print g

    print sum(fc[-1] for fc in kleinaugp_results) / len(fitcases)


.. code:: python

    #Heuristics and Gaps to optimal - KLEINAU NUMERIC
    kleinaunum_results = []
    for l, h, p, K, L, sOpt, qOpt, costOpt in fitcases:
        sk, Qk = kleinauNum(l*L, K, p, l, h,L)
        #sg = 0 if sg <0 else int(round(sg))
        if not backorders and sk < 0:
            sk = 0
        sk = int(round(sk))
        Qk = int(round(Qk))
        gcost = discreteTC(sk, Qk, l*L, K, p, l, h)
        kleinaunum_results.append((l, h, p, K, L, sOpt, qOpt, costOpt, sk,Qk, gcost,
                                  ((gcost - costOpt) / costOpt)*100))

    #for g in kleinaunum_results: print g

    print sum(fc[-1] for fc in kleinaunum_results) / len(fitcases)


.. code:: python

    # Test the best individual obtained with CCGP
    import cPickle as pkl
    from operator import add, mul, sub, div

    from inventory.utils import protectedDiv, protectedSqrt
    sfun, qfun = pkl.load(open('data/DTkleinau_n4000s400_cxp2mutp2_35_bestinds.pkl'))

    ccgp_results = []
    for l, h, p, K, L, sOpt, qOpt, costOpt in fitcases:
        lbd = l
        k = K
        sgp, Qgp = (eval(sfun, locals()), eval(qfun, locals()))

        sgp = int(round(sgp))
        if not backorders and sgp < 0: sgp = 0

        Qgp = int(round(Qgp))
        if Qgp == 0: Qgp = 1

        gcost = discreteTC(sgp, Qgp, l*L, K, p, l, h)
        ccgp_results.append((l, h, p, K, L, sOpt, qOpt, sgp,Qgp, gcost,
                            ((gcost - costOpt) / costOpt)*100))

    #for g in ccgp_results: print g

    print sum(fc[-1] for fc in ccgp_results) / len(fitcases)


.. code:: python

    # test differences in the resulting gap distributions
    from functools import partial
    from itertools import imap
    from scipy.stats import describe, ttest_ind

    def quantify(iterable, pred=bool):
        "Count how many times the predicate is true"
        return sum(imap(pred, iterable))

    def lt(a = 1, b = 1):
        return a < b

    def gt(a = 1, b = 1):
        return a > b

    def distribution(data):
        return [quantify(data, partial(lt, b = 1)),
                quantify(data, partial(lt, b = 2)),
                quantify(data, partial(lt, b = 3)),
                quantify(data, partial(gt, b = 5))]

    zz = [x[-1] for x in zhengnum_results]
    gg = [x[-1] for x in gallego_results]
    kk = [x[-1] for x in kleinaunum_results]
    ccgp = [x[-1] for x in ccgp_results]

    print distribution(zz)
    print distribution(gg)
    print distribution(kk)
    print distribution(ccgp)

    #Gallego vs ZhengNumeric
    print ttest_ind(zz, gg, equal_var = False)
    #ZhengNumeric vs CCGP
    print ttest_ind(zz, ccgp, equal_var = False)
    #Gallego vs CCGP
    print ttest_ind(gg, ccgp, equal_var = False)
    #KleinauNumeric vs CCGP
    print ttest_ind(kk, ccgp, equal_var = False)
