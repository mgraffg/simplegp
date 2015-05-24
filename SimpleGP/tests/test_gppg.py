import test_classification
import numpy as np
from SimpleGP import SparseGPPG, SparseArray
import os

cl = test_classification.cl
X = test_classification.X


def test_gppg():
    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    fname = 'gppg.npy.gz'
    gp = SparseGPPG.run_cl(x, cl, nprototypes=2,
                           fname_best=fname,
                           verbose=True, generations=2)
    assert len(gp.prototypes) == 2
    os.unlink(fname)


def test_tol():
    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    gp = SparseGPPG.run_cl(x, cl, nprototypes=3, tol=0.05, popsize=100,
                           verbose=True, generations=2)
    assert np.all(np.array(map(lambda x: len(x[-1]),
                               gp.prototypes)) == np.array([3, 1, 3]))


def test_sort_prototypes():
    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    gp = SparseGPPG.run_cl(x, cl, nprototypes=3, tol=0.05, popsize=100,
                           verbose=True, generations=2)
    _, perf = gp.prototypes_performance()
    ps = gp._prototypes_argsort
    assert np.all((ps[1:] - ps[:-1]) == 1)


def test_recall_d():
    class GPD(SparseGPPG):
        def distance(self, y, yh):
            return -self.recall_distance(y, yh).mean()

    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    gp = GPD.run_cl(x, cl, nprototypes=2,
                    verbose=True, generations=2)
    r = gp.recall_distance(gp._f, gp.eval())
    print r
    assert np.all(r <= 1)


def test_func_select():
    def func_select(x, y):
        raise Exception("!!!")
    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    try:
        SparseGPPG.run_cl(x, cl, nprototypes=2,
                          func_select=func_select,
                          verbose=True, generations=2)
    except Exception:
        return
    assert False
    
