import numpy as np
from SimpleGP import lstsqGP
import types
from SimpleGP.lstsqGP import lstsqEval


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


def run(pr):
    print "haciendo", pr
    gp = lstsqGP(seed=0, func=['+', '-', '*', '/'],
                 verbose=True,  # fname_best='tmp.npy.gz', pxo=0.9,
                 generations=50).train(x, l[pr])
    gp.run()
    return gp
    # return gp._pop_eval[gp.best]

gp = run(0)
eval = lstsqEval(gp._pop_eval_mut, gp._history_ind, gp._history_coef)
print eval.eval(gp._pop_hist[gp.best])
# ind = gp._pop_hist[gp.best]
# print eval.eval(ind), length(ind)

