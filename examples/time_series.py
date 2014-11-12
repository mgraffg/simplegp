import numpy as np
from SimpleGP import TimeSeries, GPPDE


class TPDE(TimeSeries, GPPDE):
    pass


def rse(x, y):
    return ((x - y)**2).sum() / ((x - x.sum()/x.size)**2).sum()


ts = np.array(map(float, open('data/A.txt', 'r').readlines()))
nsteps = 100
gp = TPDE.run_cl(ts, nsteps=nsteps, verbose=True,
                 max_length=256,
                 update_best_w_rprop=True)
d = np.loadtxt('data/A.cont.txt')
print rse(d[:100], gp.predict_best())
