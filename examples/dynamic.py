import numpy as np
import SimpleGP
reload(SimpleGP)
from SimpleGP import RGP, If, Sigmoid


class GP2(RGP):
    def create_random_constants(self):
        self._constants = np.random.uniform(-10,
                                            10,
                                            self._nrandom).astype(self._dtype)

if __name__ == '__main__':
    seed = 1  # if len(sys.argv) == 1 else int(sys.argv[1])
    ts = np.array(map(float, open('data/A.txt').readlines()))
    nlags = int(np.ceil(np.log2(ts.shape[0])))
    gp = GP2(popsize=10, generations=5000, verbose=True, verbose_nind=1000,
             func=[np.add, np.subtract, np.multiply, np.divide, np.fabs,
                   np.exp, np.sqrt, np.sin, np.cos, Sigmoid, If()],
             min_depth=1, nlags=nlags, fname_best=None,
             seed=seed, max_length=ts.shape[0]*10, nrandom=100, pxo=0.2,
             pgrow=0.5, walltime=None)
    gp.create_random_constants()
    x = np.zeros((ts.shape[0]-nlags, nlags+1))
    x[0, :nlags] = ts[:nlags][::-1]
    x[:, -1] = np.arange(ts.shape[0]-nlags)
    gp.train(x, ts[nlags:])
    gp.run()
