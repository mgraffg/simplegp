# Copyright 2013 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import numpy as np
from SimpleGP import GPPDE, GPMAE, GPwRestart
from pymock import use_pymock, override, returns, replay, verify
from nose.tools import assert_almost_equals


class TestSimpleGPPDE(object):
    def __init__(self):
        x = np.linspace(-10, 10, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        self._y = (X.T * pol).sum(axis=1)
        self._x = x[:, np.newaxis]
        self._gp = GPPDE.init_cl(training_size=self._x.shape[0],
                                 seed=0)
        self._gp.train(self._x, self._y)

    def test_pmutation_run(self):
        gp = GPPDE.run_cl(self._x, self._y, seed=0, verbose=True,
                          generations=5, ppm=1, pxo=0.5)
        fit = gp.fitness(gp.best)
        gp = GPPDE.run_cl(self._x, self._y, seed=0, verbose=True,
                          generations=5, ppm=0, pxo=0.5)
        assert fit != gp.fitness(gp.best)

    def test_pmutation(self):
        gp = self._gp
        gp._do_simplify = False
        gp.create_population()
        gp._ppm = 1.0
        var = gp.nfunc
        cons = var + 1
        gp.population[0] = np.array([0, 0, 1, 2, var, var, cons,
                                     2, var, cons+1, cons+2])
        gp._p_constants[0] = self._pol * -1
        gp.fitness(0)
        gp._xo_father1 = 0
        for i in range(100):
            ind = gp.mutation(gp.population[gp._xo_father1])
            print gp.population[0], ind
            assert (ind.shape[0] - np.sum(ind == gp.population[0])) <= 1

    def problem_three_variables(self):
        np.random.seed(0)
        nt = 100
        x = np.arange(nt)
        x = np.vstack((x, x[::-1], x+np.random.uniform(-1, 1, nt))).T
        # x = np.random.uniform(-1, 1, (100, 3))
        X = np.vstack((x[:, 0] * x[:, 1], x[:, 0], x[:, 2])).T
        coef = [0.2, -0.3, 0.2]
        y = (X * coef).sum(axis=1)
        return x, y

    @use_pymock
    def test_pmutation_constant(self):
        x, y = self.problem_three_variables()
        gp = GPPDE.run_cl(x, y, generations=2, seed=0)
        gp._fitness.fill(-np.inf)
        gp._ppm = 1.0
        var = gp.nfunc
        cons = var + 3
        gp.population[0] = np.array([0, 0, 2, cons, 2, var, var,
                                     2, var, cons+1,
                                     2, var+2, cons+2])
        gp._p_constants[0] = np.array([0.2, -0.1, 0.9]) * -1
        gp.fitness(0)
        gp._xo_father1 = 0
        gp._p_der.fill(0)
        ind = gp.population[0].copy()
        constants = gp._p_constants[0].copy()
        index = np.zeros_like(ind)
        gp.set_error_p_der()
        override(np.random, 'rand')
        for i in [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]:
            np.random.rand()
            returns(i)
        replay()
        c = gp._pde.compute_pdepm(ind, gp._p_st[0], index, 0.1)
        verify()
        st = gp._p_st[0]
        for i in index[:c]:
            e = np.sign(gp._p_der[i])
            de = gp._tree.pmutation_constant(ind, i, st, constants, 1, e)
            inc = gp._tree.pmutation_constant(ind, i, st, constants, -1, e)
            if gp.isconstant(ind[i]):
                index = ind[i] - gp.nfunc - gp._x.shape[1]
                assert_almost_equals((constants[index] - de) - 0.1, 0)
                assert_almost_equals((inc - constants[index]) - 0.1, 0)
            else:
                print de, inc
                m = e == 1
                assert_almost_equals((st[i, m].min() - de), 0)
                m = e == -1
                assert_almost_equals((st[i, m].max() - inc), 0)

    @use_pymock
    def test_pmutation_terminal_change(self):
        x, y = self.problem_three_variables()
        gp = GPPDE.run_cl(x, y, generations=2)
        gp._fitness.fill(-np.inf)
        gp._ppm = 1.0
        var = gp.nfunc
        cons = var + 3
        gp.population[0] = np.array([0, 0, 2, cons, 2, var, var+2,
                                     2, var, cons+1,
                                     2, var+2, cons+2])
        gp._p_constants[0] = np.array([0.2, -0.1, 0.9]) * -1
        gp.fitness(0)
        gp._xo_father1 = 0
        gp._p_der.fill(0)
        ind = gp.population[0].copy()
        constants = gp._p_constants[0].copy()
        index = np.zeros_like(ind)
        gp.set_error_p_der()
        override(np.random, 'rand')
        for i in [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]:
            np.random.rand()
            returns(i)
        replay()
        c = gp._pde.compute_pdepm(ind, gp._p_st[0], index, 0.1)
        verify()
        st = gp._p_st[0]
        # print index[:c]
        assert gp._tree.get_number_var_pm() == 10
        gp._tree.set_number_var_pm(3)
        for i in index[:c]:
            e = np.sign(gp._p_der[i])
            # print e, map(lambda x: (e == x).sum(), [-1, 0, 1])
            gp._tree.pmutation_terminal_change(ind,
                                               i, st,
                                               e,
                                               gp._x,
                                               constants,
                                               gp._eval)
            # l = map(lambda x: ((np.where(gp._x[:, x] > st[i],
            #                              -1, 1) * e) == 1).sum(), range(3))
            # print l, "*"
            print gp.isconstant(gp.population[0][i]), gp.population[0][i]
            if gp.isconstant(gp.population[0][i]):
                assert gp.population[0][i] == ind[i]
                print gp.population[0][i], ind[i], "cons",\
                    map(lambda x: (e == x).sum(), [-1, 0, 1])
            else:
                assert gp.population[0][i] != ind[i]
                print gp.population[0][i], ind[i], "var",\
                    map(lambda x: (e == x).sum(), [-1, 0, 1])
        assert False

    @use_pymock
    def test_pde_pm(self):
        gp = self._gp
        gp.create_population()
        gp._ppm = 1.0
        var = gp.nfunc
        cons = var + 1
        gp.population[0] = np.array([0, 0, 1, 2, var, var, cons,
                                     2, var, cons+1, cons+2])
        gp._p_constants[0] = self._pol * -1
        gp.fitness(0)
        gp._xo_father1 = 0
        gp._p_der.fill(0)
        ind = gp.population[0]
        index = np.zeros_like(ind)
        gp.set_error_p_der()
        override(np.random, 'rand')
        for i in [1, 0.2, 0.2]:
            np.random.rand()
            returns(i)
        replay()
        c = gp._pde.compute_pdepm(ind, gp._p_st[0], index, 0.3)
        verify()
        m = np.zeros(ind.shape, dtype=np.bool)
        m[index[:c]] = True
        assert np.all(gp._p_der[index] != 0)
        assert index[0] == 1 and index[1] == ind.shape[0]-1
        assert c == 2

    def test_pmutation_func_change(self):
        gp = self._gp
        gp.create_population()
        var = gp.nfunc
        cons = var + 1
        gp.population[0] = np.array([0, 0, 1, 2, var, var, cons,
                                     2, var, cons+1, cons+2])
        gp._p_constants[0] = self._pol
        gp._xo_father1 = 0
        gp.fitness(0)
        p1 = 2
        e = gp.get_error(p1)
        st = gp._p_st[gp._xo_father1]
        func = gp._tree.pmutation_func_change(gp.population[0],
                                              p1, st, e, gp._eval)
        print func
        assert func == 2

    def test_pmutation_eval(self):
        def if_func(x, y, z):
            s = 1 / (1 + np.exp(-100 * x))
            return s * (y - z) + z

        def argmax(x, y):
            x = np.exp(2.*x)
            y = np.exp(2.*y)
            xi = x + y
            return x*0 + y / xi
        gp = self._gp
        gp.create_population()
        map(gp.fitness, range(gp.popsize))
        np.random.seed(0)
        x = np.linspace(-10, 10, 100)
        args = np.vstack((x, x*np.random.random(100),
                          x*np.random.random(100)))
        func = np.array([np.add, np.subtract, np.multiply, np.divide,
                         np.absolute, np.exp, np.sqrt, np.sin, np.cos,
                         lambda x: 1 / (1 + np.exp(-x)), if_func,
                         lambda x, y: if_func(x-y, x, y),
                         lambda x, y: if_func(x-y, y, x),
                         lambda x: np.log(np.absolute(x)),
                         np.square, None, argmax])
        for nop in np.unique(gp._nop):
            if nop < 1:
                continue
            if nop == 3:
                a = np.vstack((args[0], args[1], np.zeros(100), args[2]))
                gp._eval.pmutation_eval(nop, a, np.array([0, 1, 3]))
            else:
                gp._eval.pmutation_eval(nop, args, np.arange(nop))
            if nop == 1:
                inputs = (args[0],)
            else:
                inputs = args[:nop]
            m = gp._nop == nop
            r = np.array(map(lambda x: x(*inputs), func[m]))
            assert gp._eval.pmutation_eval_test(nop, r)

    def test_type_xpoint_selection(self):
        gp = GPPDE.run_cl(self._x, self._y, type_xpoint_selection=1,
                          generations=5)
        gp2 = GPPDE.run_cl(self._x, self._y, type_xpoint_selection=0,
                           generations=5)
        b1 = gp.population[gp.get_best()]
        b2 = gp2.population[gp.get_best()]
        assert np.all(b1 != b2)

    def test_walltime(self):
        t = time.time()
        GPPDE.run_cl(self._x, self._y, walltime=1)
        assert time.time() - t < 1.1

    def test_time(self):
        t = time.time()
        gp = self._gp
        gp._gens = 10
        gp._verbose = True
        gp.run()
        assert (time.time() - t) < 1

    def test_parent(self):
        gp = self._gp
        gp.create_population()
        ind = np.argmax(map(lambda x: x.shape[0], gp._p))
        parent = np.empty_like(gp._p[ind])
        gp._tree.compute_parents(gp._p[ind],
                                 parent)
        end = gp._tree.traverse(gp._p[ind], pos=1)
        assert parent[end] == 0

    def test_path_to_root(self):
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           seed=1).train(self._x, self._y)
        gp.create_population()
        ind = filter(lambda x: (gp._p[x] == 11).sum(),
                     range(gp._p.shape[0]))
        ind = ind[np.argmax(map(lambda x: gp._p[x].shape[0], ind))]
        parent = np.empty_like(gp._p[ind])
        gp._tree.compute_parents(gp._p[ind],
                                 parent)
        end = gp._tree.traverse(gp._p[ind], pos=1)
        path = np.empty_like(gp._p[ind])
        gp._tree.path_to_root(parent,
                              path, end)
        print gp.print_infix(ind)
        assert (path[0] == 0) and (path[1] == end)

    def test_argument_position(self):
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        ind = filter(lambda x: (gp._p[x] == 11).sum(),
                     range(gp._p.shape[0]))
        ind = ind[np.argmax(map(lambda x: gp._p[x].shape[0], ind))]
        parent = np.empty_like(gp._p[ind])
        gp._tree.compute_parents(gp._p[ind],
                                 parent)
        ind = gp._p[ind]
        for i in range(ind.shape[0]):
            if ind[i] < gp._nop.shape[0]:
                for j in range(gp._nop[ind[i]]):
                    c = gp._tree.get_pos_arg(ind, i, j)
                    assert parent[c] == i

    def test_pde(self):
        from SimpleGP.pde import PDE
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        p_der = np.empty((gp._max_length, gp._x.shape[0]), dtype=gp._dtype)
        pde = PDE(gp._tree, p_der)
        for i in range(gp._p.shape[0]):
            gp.fitness(i)
            for pos in range(gp._p[i].shape[0]):
                pde.compute(gp._p[i], pos,
                            gp._p_st[i])

    def test_pde_add(self):
        from SimpleGP.pde import PDE
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        nfunc = gp._nop.shape[0]
        gp._p[0] = np.array([0, nfunc, nfunc])
        gp.fitness(0)
        print gp.print_infix(0)
        p_der = np.ones_like(gp._p_st[0])
        p_der[0] = gp.eval(0)
        pde = PDE(gp._tree, p_der)
        pde.compute(gp._p[0], 1,
                    gp._p_st[0])
        assert np.all(p_der[1] == p_der[0])
        assert np.all(p_der[2] == 1)
        pde.compute(gp._p[0], 2,
                    gp._p_st[0])
        assert np.all(p_der[2] == p_der[0])

    def test_pde_subtract(self):
        from SimpleGP.pde import PDE
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        nfunc = gp._nop.shape[0]
        gp._p[0] = np.array([1, nfunc, nfunc])
        gp.fitness(0)
        print gp.print_infix(0)
        p_der = np.ones_like(gp._p_st[0])
        p_der[0] = gp.eval(0)
        pde = PDE(gp._tree, p_der)
        pde.compute(gp._p[0], 1,
                    gp._p_st[0])
        assert np.all(p_der[1] == p_der[0])
        assert np.all(p_der[2] == 1)
        pde.compute(gp._p[0], 2,
                    gp._p_st[0])
        assert np.all(p_der[2] == -p_der[0])

    def test_pde_multiply(self):
        from SimpleGP.pde import PDE
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           update_best_w_rprop=False,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        nfunc = gp._nop.shape[0]
        gp._p[0] = np.array([2, nfunc, nfunc+1])
        gp._p_constants[0].fill(-1)
        gp.fitness(0)
        print gp.print_infix(0)
        p_der = np.ones_like(gp._p_st[0])
        p_der[0] = gp.eval(0)
        pde = PDE(gp._tree, p_der)
        pde.compute(gp._p[0], 1,
                    gp._p_st[0])
        assert np.all(p_der[1] == -p_der[0])
        assert np.all(p_der[2] == 1)
        pde.compute(gp._p[0], 2,
                    gp._p_st[0])
        assert np.all(p_der[2] == (p_der[0] * gp._p_st[0][1]))

    def test_pde_divide(self):
        from SimpleGP.pde import PDE
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           update_best_w_rprop=False,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        nfunc = gp._nop.shape[0]
        gp._p[0] = np.array([3, nfunc, nfunc+1])
        gp._p_constants[0].fill(-2)
        gp.fitness(0)
        print gp.print_infix(0)
        p_der = np.ones_like(gp._p_st[0])
        p_der[0] = gp.eval(0)
        pde = PDE(gp._tree, p_der)
        pde.compute(gp._p[0], 1,
                    gp._p_st[0])
        assert np.all(p_der[1] == (p_der[0] / gp._p_st[0][2]))
        assert np.all(p_der[2] == 1)
        pde.compute(gp._p[0], 2,
                    gp._p_st[0])
        r = - p_der[0] * gp._p_st[0][1] / gp._p_st[0][2]**2
        assert np.all(p_der[2] == r)

    def test_pde_fabs(self):
        from SimpleGP.pde import PDE
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           update_best_w_rprop=False,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        nfunc = gp._nop.shape[0]
        gp._p[0] = np.array([4, nfunc])
        gp.fitness(0)
        print gp.print_infix(0)
        p_der = np.ones_like(gp._p_st[0])
        p_der[0] = gp.eval(0)
        pde = PDE(gp._tree, p_der)
        pde.compute(gp._p[0], 1,
                    gp._p_st[0])
        assert np.all(p_der[1] == gp._p_st[0][1])

    def test_pde_exp(self):
        from SimpleGP.pde import PDE
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           update_best_w_rprop=False,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        nfunc = gp._nop.shape[0]
        gp._p[0] = np.array([5, nfunc])
        gp.fitness(0)
        print gp.print_infix(0)
        p_der = np.ones_like(gp._p_st[0])
        p_der[0] = 1
        pde = PDE(gp._tree, p_der)
        pde.compute(gp._p[0], 1,
                    gp._p_st[0])
        x = np.exp(gp._p_st[0][1])
        assert np.all(p_der[1] == x)

    def test_pde_constants(self):
        from SimpleGP.pde import PDE
        from SimpleGP.Rprop_mod import RPROP2
        gp = GPPDE.init_cl(training_size=self._x.shape[0], argmax_nargs=3,
                           update_best_w_rprop=False,
                           seed=0).train(self._x, self._y)
        gp.create_population()
        fit = gp.fitness(0)
        e, g = gp.compute_error_pr(None)
        print fit
        p_der = np.ones_like(gp._p_st[0])
        p_der[0] = e
        pde = PDE(gp._tree, p_der)
        cons = gp._p_constants[0].copy()
        pde.compute_constants(gp._p[0],
                              gp._p_st[0])
        rprop = RPROP2(gp._p[0], gp._p_constants[0],
                       p_der, gp._tree)
        rprop.update_constants_rprop()
        nfunc = gp._nop.shape[0]
        nvar = gp._x.shape[1]
        print "n cons", (gp._p[0] >= nfunc+nvar).sum(), cons.shape
        gp._fitness[0] = -np.inf
        assert gp.fitness(0) > fit


def test_gppde():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPPDE.run_cl(x, y, seed=0,
                      generations=5,
                      verbose=True)
    fit = gp.fitness(gp.get_best())
    assert not np.isnan(fit) and not np.isinf(fit)


def test_gppde_dosimplify():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPPDE.run_cl(x, y, generations=5, seed=0, do_simplify=False)
    fit = gp.fitness(gp.get_best())
    assert not np.isnan(fit) and not np.isinf(fit)


def test_gppde_crossover_length():
    # assert 0
    # the following test raises a bus error. The problem is that
    # do_simplify=False is not working
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPPDE.run_cl(x, y, generations=3, seed=0,
                      update_best_w_rprop=False,
                      do_simplify=False,
                      max_length=1024)
    for i in range(gp._p.shape[0]-1):
        gp._min_length = gp._p[i].shape[0]
        gp.tree_params()
        gp._xo_father1 = i
        gp._xo_father2 = i + 1
        f1l = gp._p[gp._xo_father1].shape[0]
        f2l = gp._p[gp._xo_father2].shape[0]
        if (f1l < f2l):
            for i in range(f1l):
                gp._tree.crossover_mask(gp._p[gp._xo_father1],
                                        gp._p[gp._xo_father2], i)
                npoints = gp._tree_mask[:f2l].sum()
                assert npoints > 0
                pos = gp._tree.father2_crossing_point(gp._p[gp._xo_father1],
                                                      gp._p[gp._xo_father2],
                                                      i)
                assert gp._tree_mask[pos]
            gp.fitness(gp._xo_father1)
            gp.fitness(gp._xo_father2)
            ind = gp.crossover(gp._p[gp._xo_father1],
                               gp._p[gp._xo_father2])
            assert ind.shape[0] >= gp._min_length


def test_gppde_mae():
    class G(GPMAE, GPPDE):
        pass
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = G.run_cl(x, y, generations=5, seed=0,
                  max_length=1000)
    fit = gp.fitness(gp.get_best())
    assert not np.isnan(fit) and not np.isinf(fit)


def test_gppde_test_set():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    x1 = np.linspace(-20, 20, 1000)[:, np.newaxis]
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPPDE.run_cl(x, y, test=x1, max_length=1024,
                      generations=5,
                      seed=0, verbose=True)
    assert gp is not None


def test_gppde_predict():
    class G(GPwRestart, GPMAE, GPPDE):
        pass
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    x1 = np.linspace(-20, 20, 1000)[:, np.newaxis]
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = G.init_cl(max_length=1024,
                   generations=3, ntimes=1,
                   seed=0, verbose=True).train(x, y)
    gp.run(exit_call=False)
    gp.predict(x1)
    print "*"*10
    gp._gens = 6
    gp.run()
    assert gp is not None
