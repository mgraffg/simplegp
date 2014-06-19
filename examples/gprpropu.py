import SimpleGP
reload(SimpleGP)
from SimpleGP import GPRPropU
import numpy as np


x = np.linspace(0, 10, 100)
pol = np.array([0.2, -0.3, 0.2])
X = np.vstack((x**2, x, np.ones(x.shape[0])))
y = (X.T * pol).sum(axis=1)
gp = GPRPropU(popsize=1000,
              generations=50*5,
              verbose=True,
              verbose_nind=1000,
              do_simplify=True,
              func=['+', '-', '*', "/", 'abs', 'exp', 'sqrt',
                    'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                    'ln', 'sq'],
              min_length=2,
              pleaf=None,
              min_depth=0, fname_best=None,
              seed=0, nrandom=100, pxo=1.0, pgrow=0.5, walltime=None)
x = x[:, np.newaxis]
gp.train(x, y)
gp.run()
print gp.fitness(gp.best)
print gp.print_infix()
