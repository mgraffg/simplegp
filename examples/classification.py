from SimpleGP import Classification
import numpy as np


fpt = open('data/iris.npy', 'rb')
X = np.load(fpt)
cl = np.load(fpt)
fpt.close()


gp = Classification(popsize=1000, generations=50, verbose=True,
                    verbose_nind=1000,
                    # type_xpoint_selection=2,
                    func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
                          'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                          'ln', 'sq'],
                    fname_best=None,
                    seed=0, nrandom=0,
                    pxo=0.9, pgrow=0.5,
                    walltime=None).fit(X, cl)
pr = gp.predict(X)
print gp.fitness(gp.best)
print (pr == cl).sum() / float(cl.shape[0])
print gp.print_infix()
