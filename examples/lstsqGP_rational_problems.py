import numpy as np
from SimpleGP import lstsqGP, GSGP
import types


def length(ind):
    if not isinstance(ind, types.ListType):
        return 1
    cnt = len(ind)
    if ind[0] == -1:
        return cnt + length(ind[1]) - 1
    elif ind[0] == -2:
        return cnt + length(ind[1]) + length(ind[2]) - 2

x = np.arange(-1, 1.1, 0.1)[:, np.newaxis]
l = np.array(map(lambda x: map(float, x.split()),
                 open('data/rational-problems.txt', 'r').readlines()))


def run(ins, pr):
    print "haciendo", pr
    gp = ins(seed=0, func=['+', '-', '*', '/'],
             popsize=100,
             verbose=True,  # fname_best='tmp.npy.gz', pxo=0.9,
             generations=2000).train(x, l[pr])
    gp.run()
    return gp
    # return gp._pop_eval[gp.best]

gp = run(lstsqGP, 0)
print "*"*10
gp1 = run(GSGP, 0)

