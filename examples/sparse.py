import numpy as np
from SimpleGP import GP
from SimpleGP import SparseEval
from SimpleGP.Simplify import Simplify


class SparseGP(GP):
    def train(self, x, f):
        self._eval = SparseEval(self._nop)
        self._eval.X(x)
        self._x = x
        self._f = f
        self._st = None
        self._p_der_st = None
        self._error_st = None
        self._simplify = Simplify(x.shape[1], self._nop)
        self._simplify.set_constants(self._constants2)
        self._tree.set_nvar(self._x.shape[1])
        return self

    def eval_ind(self, ind, pos=0, constants=None):
        c = constants if constants is not None else self._constants
        self.nodes_evaluated += ind.shape[0]
        return self._eval.eval(ind, c)


def create_problem():
    x = np.linspace(-10, 10, 1000)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = np.vstack((x, np.sqrt(np.fabs(x)))).T
    return x, y


def run():
    x, y = create_problem()
    gp = SparseGP.run_cl(x, y,  func=['+', '-', '*', '/'],
                         max_length=256, popsize=10000, verbose=True,
                         generations=10)
    return gp

    
def run2():
    x, y = create_problem()
    gp = GP.run_cl(x, y,  func=['+', '-', '*', '/'],
                   max_length=256, popsize=10000, verbose=True,
                   generations=10)
    return gp


if __name__ == '__main__':
    run()
