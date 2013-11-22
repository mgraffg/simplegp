from SimpleGP import SimpleGA
import numpy as np


def test_SimpleGA():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA(generations=10000,
                 popsize=3, pm=0.1,
                 pxo=0.9,
                 fname_best=None,
                 verbose=False).train(X, f)
    s.run()
    assert np.fabs(pol - s._p[s.get_best()]).mean() < 0.1
