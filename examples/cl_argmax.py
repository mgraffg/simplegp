from SimpleGP import GPPDE
import numpy as np


class Cl2(GPPDE):
    def random_func(self, first_call=False):
        if first_call:
            # this is the argmax
            return 16
        return super(Cl2, self).random_func(first_call=first_call)

fpt = open('data/iris.npy')
X = np.load(fpt)
cl = np.load(fpt)
fpt.close()


gp = GPPDE(popsize=1000, generations=50, verbose=True,
           verbose_nind=1000,
           func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
                 'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                 'ln', 'sq', 'argmax'],
           fname_best=None,
           seed=0, nrandom=0,
           max_mem=500,
           argmax_nargs=np.unique(cl).shape[0],
           pxo=0.9, pgrow=0.5, walltime=None)

# Test
# cl = np.ones_like(cl) + 1
# gp.train(X, cl)
# gp.create_population()
# nfunc = gp._nop.shape[0]
# nvar = gp._x.shape[1]
# ind = np.array([nfunc-1, nfunc+nvar, nfunc+nvar+1, nfunc+nvar+2])
# gp._p[0] = ind
# gp._p_constants[0] = np.array([2.1, 18.5, 14.2])
# print gp.print_infix(0), gp.eval(0)[0]
# print gp.fitness(0)
# gp.rprop(0)
# print gp.print_infix(0)
# print gp.fitness(0)

gp.train(X, cl)
gp.run()
clh = np.round(gp.eval(gp.best))
print gp.print_infix()
print gp.fitness(gp.best)
print (clh == cl).sum() / float(cl.shape[0])

















