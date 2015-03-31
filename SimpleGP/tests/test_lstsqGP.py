from SimpleGP import lstsqGP
import numpy as np
from nose.tools import assert_almost_equals


def create_problem():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    return x, y


def test_lstsqGP():
    x, y = create_problem()
    gp = lstsqGP(generations=5, verbose=True).train(x, y)
    gp.run()

        
def test_create_population():
    x, y = create_problem()
    gp = lstsqGP().train(x, y)
    gp.create_population()
    for i in range(gp.popsize):
        assert np.all(gp._pop_eval[i] == gp.eval(i))
    assert np.all(np.isfinite(gp._fitness))


def test_crossover_mutation():
    x, y = create_problem()
    gp = lstsqGP(seed=0).train(x, y)
    gp.create_population()
    while not gp.pre_crossover():
        gp.pre_crossover()
    f1 = gp.population[gp._xo_father1]
    f2 = gp.population[gp._xo_father2]
    son = gp.crossover(f1, f2)
    print gp._ind_generated_c
    d = np.fabs(gp._ind_generated_c - np.array([-4.06569463e-01,
                                                -7.40067741e+04])).sum()
    assert_almost_equals(d, 0, 5)
    assert np.all(np.isfinite(son))
    assert gp._ind_generated_c.shape[0] == 2
    son = gp.mutation(f1)
    assert np.all(np.isfinite(son))
    assert gp._ind_generated_c.shape[0] == 2


def test_kill_ind():
    x, y = create_problem()
    gp = lstsqGP(seed=0).train(x, y)
    gp.create_population()
    while not gp.pre_crossover():
        gp.pre_crossover()
    f1 = gp._xo_father1
    f2 = gp._xo_father2
    f3 = gp.tournament(True)
    f4 = gp.tournament()
    while f1 == f3 or f2 == f3 or f3 == f4 or f4 == f1 or f4 == f2:
        f3 = gp.tournament(True)
    son = gp.crossover(f1, f2)
    gp.kill_ind(f3, son)
    assert gp._history_ind[gp._history_index-1][-1] == -1
    gp._xo_father1 = f3
    son1 = gp.mutation(f3)
    gp.kill_ind(f4, son1)
    assert gp._history_ind[gp._history_index-1][-1] > -1


def test_fitness():
    x, y = create_problem()
    gp = lstsqGP(seed=0).train(x, y)
    gp.create_population()
    while not gp.pre_crossover():
        gp.pre_crossover()
    f1 = gp._xo_father1
    f2 = gp._xo_father2
    f3 = gp.tournament(True)
    while f1 == f3 or f2 == f3:
        f3 = gp.tournament(True)
    son = gp.crossover(f1, f2)
    gp.kill_ind(f3, son)
    if max([gp.fitness(f1), gp.fitness(f2)]) > gp.fitness(f3):
        assert False
    print gp.fitness(f1), gp.fitness(f2), gp.fitness(f3)


def test_eval():
    from SimpleGP.lstsqGP import lstsqEval

    def predict(ind):
        return eval.eval(ind)
    x, y = create_problem()
    gp = lstsqGP(seed=0, popsize=100, generations=3).train(x, y)
    gp.run()
    eval = lstsqEval(gp._pop_eval_mut,
                     gp._history_ind,
                     gp._history_coef)
    inds = eval.inds_to_eval(gp._pop_hist[gp.best])
    print len(inds), inds[-1], gp._pop_hist[gp.best],\
        gp._history_ind[gp._pop_hist[gp.best]]
    for i in range(gp.popsize):
        pr = eval.eval(gp._pop_hist[i])
        assert np.all(pr == gp._pop_eval[i])

    
def test_save():
    import tempfile
    fname = tempfile.mktemp()
    # fname = fname + '.gz'
    print fname
    x, y = create_problem()
    gp = lstsqGP(generations=5, fname_best=fname,
                 seed=0,
                 verbose=True).train(x, y)
    gp.run()
    print gp.best
    fit = gp.fitness(gp.best)
    gp = lstsqGP(generations=5, fname_best=fname,
                 seed=0,
                 verbose=True).train(x, y)
    gp.run()
    print gp.best, gp._fitness[gp.best], fit
    assert fit == gp.fitness(gp.best)


def test_save_only_best():
    x, y = create_problem()
    try:
        lstsqGP(generations=3,
                seed=0,
                save_only_best=True,
                verbose=True).train(x, y)
        assert False
    except NotImplementedError:
        pass
        

def test_history():
    x, y = create_problem()
    gp = lstsqGP(generations=3,
                 seed=0,
                 verbose=True).train(x, y)
    gp.run()
    assert gp._history_coef[gp._history_index:].sum() == 0
    # print gp._history_index, gp._history_ind[:gp._history_index],
    gp._history_ind.max()
    print gp._pop_hist[gp.best], gp._history_ind[gp._pop_hist[gp.best]]
    assert np.all(gp._history_ind[gp._history_index:] == -1)
