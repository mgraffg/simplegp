from SimpleGP import Classification, GPPDE
import numpy as np


class Cl2(Classification, GPPDE):
    pass

fpt = open('data/iris.npy')
X = np.load(fpt)
cl = np.load(fpt)
fpt.close()


gp = Cl2(popsize=1000, generations=50, verbose=True,
         verbose_nind=1000,
         func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
               'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
               'ln', 'sq'],
         fname_best=None,
         seed=0, nrandom=0,
         pxo=0.9, pgrow=0.5, walltime=None)
gp.train(X, cl)
gp.run()
gp.eval(gp.get_best())
print gp.fitness(gp.get_best())
print (gp._st[0] == cl).sum() / float(cl.shape[0])
print gp.print_infix()
