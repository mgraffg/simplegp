from SimpleGP import GP
import numpy as np


seed = 0  # if len(sys.argv) == 1 else int(sys.argv[1])
x = np.linspace(0, 1, 100)
pol = np.array([0.2, -0.3, 0.2])
X = np.vstack((x**2, x, np.ones(x.shape[0])))
y = (X.T * pol).sum(axis=1)
gp = GP(popsize=10,
        generations=100000,
        verbose=True,
        verbose_nind=1000,
        min_length=1,
        do_simplify=True,
        func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
              'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
              'ln', 'sq'],
        min_depth=0, fname_best='regression.npy',
        seed=seed, nrandom=100, pxo=0.2, pgrow=0.5, walltime=None)
gp.create_random_constants()
x = x[:, np.newaxis]
gp.train(x, y)
gp.create_population()

nvar = gp._nop.shape[0]
ind = np.array([2, 3, 0, 0, nvar, nvar, 1, nvar, nvar,
                0, 1, nvar, nvar, 2, nvar, nvar, 1, 3,
                nvar, nvar, 3, nvar, nvar], dtype=np.int)
print gp.print_infix(ind)
ind2 = gp.simplify(ind)
print gp.print_infix(ind2, constants=gp._ind_generated_c)

ind = np.array([1, 0, 3, nvar, nvar, 1, nvar, nvar,
                3, 2, nvar, nvar, 2, nvar, nvar], dtype=np.int)
print gp.print_infix(ind)
ind2 = gp.simplify(ind)
print gp.print_infix(ind2, constants=gp._ind_generated_c)
print ind2

ind = np.array([13, 5, 2, nvar, nvar], dtype=np.int)
print gp.print_infix(ind, constants=gp._ind_generated_c)
ind2 = gp.simplify(ind)
print gp.print_infix(ind2, constants=gp._ind_generated_c)

ind = np.array([5, 13, 2, nvar, nvar], dtype=np.int)
print gp.print_infix(ind, constants=gp._ind_generated_c)
ind2 = gp.simplify(ind)
print gp.print_infix(ind2, constants=gp._ind_generated_c)

ind = np.array([5, 13, 2, nvar, nvar], dtype=np.int)
print gp.print_infix(ind, constants=gp._ind_generated_c)
ind2 = gp.simplify(ind)
print gp.print_infix(ind2, constants=gp._ind_generated_c)


gp._p[0] = np.array([0, 2, nvar, nvar+2, nvar+1], dtype=np.int)
gp._p_constants[0] = np.array([0, 1.4])
print gp.print_infix(0)
gp.simplify(0)
print gp.print_infix(0) == "(X0 * 1.4)"

gp._p[0] = np.array([0, nvar+1, 2, nvar, nvar+2], dtype=np.int)
gp._p_constants[0] = np.array([0, 1.4])
print gp.print_infix(0)
gp.simplify(0)
print gp.print_infix(0) == "(X0 * 1.4)"

gp._p[0] = np.array([1, 0, 2, nvar, nvar+2, nvar+1,
                     2, nvar, nvar+2], dtype=np.int)
gp._p_constants[0] = np.array([0, 1.4])
print gp.print_infix(0)
gp.simplify(0)
print gp.print_infix(0)
