import numpy as np
import SimpleGP
reload(SimpleGP)
from SimpleGP import TimeSeries, GPPDE, GPMAE, GPwRestart


def rse(x, y):
    return ((x - y)**2).sum() / ((x - x.sum()/x.size)**2).sum()


class TS(GPMAE, GPwRestart, GPPDE, TimeSeries):
    pass


seed = 0
ts = np.array(map(float, open('data/A.txt').readlines()))
nlags = int(np.ceil(np.log2(ts.shape[0])))
nsteps = 100
gp = TS(popsize=1000, generations=50,
        verbose=True, verbose_nind=1000,
        update_best_w_rprop=True,
        func=["+", "-", "*", "/", "sin", "sq", "sqrt",
              "cos", "exp", "ln"],
        nsteps=nsteps, min_depth=1,
        fname_best=None,
        seed=seed, max_length=100, nrandom=100,
        pxo=0.9, pgrow=0.5, nlags=nlags)
gp.create_random_constants()
x, y = TimeSeries.create_W(ts, nlags)
gp.train(x, y)
gp.run()
d = np.loadtxt('data/A.cont.txt')
print rse(d[:100], gp.predict_best())
















