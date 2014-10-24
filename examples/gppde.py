import SimpleGP
reload(SimpleGP)
from SimpleGP import GPPDE
import numpy as np


x = np.linspace(-10, 10, 100)
pol = np.array([0.2, -0.3, 0.2])
X = np.vstack((x**2, x, np.ones(x.shape[0])))
y = (X.T * pol).sum(axis=1)
x = x[:, np.newaxis]
gp = GPPDE(generations=30, verbose=True,
           seed=2,
           # max_mem=10, verbose_nind=10,
           # update_best_w_rprop=True,
           max_length=1000).train(x, y)
gp.run()
assert gp.fitness(gp.best) >= -2.2399825722547702e-06
print gp.fitness(gp.best)
print gp.print_infix()
