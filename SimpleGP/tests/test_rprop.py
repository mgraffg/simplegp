from SimpleGP import GP
import numpy as np


class TestRprop(object):
    def __init__(self):
        x = np.linspace(0, 1, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._gp = GP(compute_derivatives=True).train(x, y)
        self._gp.create_population()

    def test_rprop(self):
        nvar = self._gp._func.shape[0]
        self._gp._p[0] = np.array([0, 2, 14,
                                   nvar, nvar+1, 0,
                                   2, nvar, nvar+2, nvar+3])
        self._gp._p_constants[0] = self._pol * -1
        self._gp.eval(0)
        self._gp.rprop(0)
        c2 = self._gp._p_constants[0]
        assert np.fabs(self._pol - c2).mean() < 0.01
