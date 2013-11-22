from SimpleGP import GPMAE
import numpy as np


def test_gpmae():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPMAE(generations=30,
               max_length=1000).train(x, y)
    gp.run()
    assert gp.fitness(gp.get_best()) >= -0.0689764253988










