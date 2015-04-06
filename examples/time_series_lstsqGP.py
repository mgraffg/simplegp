import numpy as np
from SimpleGP import TimeSeries, lstsqGP


class TPDE(TimeSeries, lstsqGP):
    pass


def rse(x, y):
    return ((x - y)**2).sum() / ((x - x.sum()/x.size)**2).sum()


ts = np.array(map(float, open('data/A.txt', 'r').readlines()))
nsteps = 100
gp = TPDE.run_cl(ts, nsteps=nsteps, verbose=True, nrandom=0, generations=10)
d = np.loadtxt('data/A.cont.txt')
print rse(d[:100], gp.predict_best())
