from SimpleGP import lstsqGP
import numpy as np


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
    gp = lstsqGP().train(x, y)
    gp.create_population()
    while not gp.pre_crossover():
        gp.pre_crossover()
    f1 = gp.population[gp._xo_father1]
    f2 = gp.population[gp._xo_father2]
    son = gp.crossover(f1, f2)
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
    assert len(gp._pop_hist[f3]) == 3
    print "1xo", gp._pop_hist[f3]
    gp._xo_father1 = f3
    son1 = gp.mutation(f3)
    gp.kill_ind(f4, son1)
    print "1xo", gp._pop_hist[f3]
    hist = gp._pop_hist[f4]
    assert len(hist) == 2
    assert len(hist[0]) == 3
    print "1m", hist
    gp._xo_father2 = f4
    son = gp.crossover(f1, f4)
    gp.kill_ind(f1, son)
    hist = gp._pop_hist[f1]
    assert len(hist) == 3
    assert len(hist[1]) == 2
    for i in [f1, f2, f3, f4]:
        print gp._pop_hist[i]
        print "*"*10
    print son


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
        
