import numpy as np
from SimpleGP import GP, SparseEval


def create_problem():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = np.vstack((x, np.sqrt(np.fabs(x)))).T
    return x, y


def run():
    x, y = create_problem()
    gp = GP.run_cl(x, y, func=['+', '-', '*', '/'],
                   max_length=256,
                   generations=2)
    sparse = SparseEval(gp._nop)
    sparse.X = x
    for i in range(gp.popsize):
        y = gp.predict(gp._x, ind=i)
        if not np.all(np.isfinite(y)):
            continue
        sparse.eval(gp.population[i], gp._p_constants[i])


if __name__ == '__main__':
    run()
