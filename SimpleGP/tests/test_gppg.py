import test_classification
import numpy as np
from SimpleGP import SparseGPPG, SparseArray

cl = test_classification.cl
X = test_classification.X


def test_gppg():
    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    gp = SparseGPPG.run_cl(x, cl, nprototypes=2,
                           verbose=True, generations=2)
    assert len(gp.prototypes) == 2


def test_tol():
    x = map(lambda x: SparseArray.fromlist(X[x]), range(X.shape[0]))
    gp = SparseGPPG.run_cl(x, cl, nprototypes=3, tol=0.05, popsize=100,
                           verbose=True, generations=2)
    assert np.all(np.array(map(lambda x: len(x[-1]),
                               gp.prototypes)) == np.array([3, 1, 3]))
    
