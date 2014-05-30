import SimpleGP
reload(SimpleGP)
from SimpleGP import GP
import numpy as np


class GP2(GP):
    """This class exemplifies the change of the distance function.
    In the example, the distance is MAE then the derivative of this
    function is computed in the method compute_error_pr"""
    def distance(self, y, yh):
        return np.fabs(y - yh).mean()

    def compute_error_pr(self, ind, pos=0, constants=None, epoch=0):
        if epoch == 0:
            g = self._st[self._output].T
        else:
            g = self.eval_ind(ind, pos=pos, constants=constants)
        e = self._f - g
        s = np.sign(e)
        e = -1 * s
        return e, g


def run(seed=0, pxo=0.9):
    # seed = 0  # if len(sys.argv) == 1 else int(sys.argv[1])
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -3.2, 1.3])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    gp = GP2(popsize=1000,
             generations=50,
             verbose=True,
             verbose_nind=100,
             min_length=1,
             pleaf=None,
             func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
                   'sin', 'cos', 'sigmoid', 'max', 'min',
                   'ln', 'sq'],
             min_depth=0, fname_best=None,
             seed=seed, nrandom=100, pxo=pxo, pgrow=0.5, walltime=None)
    gp.create_random_constants()
    x = x[:, np.newaxis]
    gp.train(x, y)
    gp.create_population()
    nvar = gp._func.shape[0]
    gp._p[0] = np.array([0, 2, 14, nvar, nvar+1, 0, 2, nvar, nvar+2, nvar+3])
    gp._p_constants[0] = pol * -1
    print gp._max_nargs
    print pol
    print "Fit", gp.distance(gp._f, gp.eval(0))
    print gp.print_infix(0)
    gp.rprop(0)
    print "Fit", gp.distance(gp._f, gp.eval(0))
    print gp.print_infix(0)
    # print gp._func[gp._func_allow]
    # gp.run()
    return gp

gp = run(0)













