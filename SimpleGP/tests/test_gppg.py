import test_classification
import numpy as np
from SimpleGP import SparseGPPG, SparseArray, SparseGPPGD
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


def test_gppgD():
    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    gp = SparseGPPGD.run_cl(x, cl, nprototypes=2,
                            verbose=True, generations=2)
    gp1 = SparseGPPG.run_cl(x, cl, nprototypes=2,
                            verbose=True, generations=2)
    d1 = gp._dist_matrix_W.min(axis=0)[gp.eval() == cl].sum()
    d2 = gp1._dist_matrix_W.min(axis=0)[gp1.eval() == cl].sum()
    assert d1 < d2
    
