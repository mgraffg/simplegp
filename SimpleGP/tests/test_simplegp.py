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
from SimpleGP import GP
import numpy as np


class TestSimpleGP(object):
    def __init__(self):
        x = np.linspace(-10, 10, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._x = x
        self._y = y
        self._gp = GP.init_cl(seed=0, generations=5).train(x, y)

    def test_seed(self):
        gp1 = GP.run_cl(self._x, self._y, generations=3, seed=0)
        gp2 = GP.run_cl(self._x, self._y, generations=3, seed=2)
        r = map(lambda x: np.all(gp1.population[x] == gp2.population[x]),
                range(gp1.popsize))
        print gp1.population[1], gp2.population[1]
        assert not np.all(r)
        gp1 = GP.run_cl(self._x, self._y, generations=3, seed=1)
        gp2 = GP.run_cl(self._x, self._y, generations=3, seed=1)
        r = map(lambda x: np.all(gp1.population[x] == gp2.population[x]),
                range(gp1.popsize))
        print gp1.population[1], gp2.population[1]
        assert np.all(r)


    def test_nop_only_func_allow(self):
        x = self._x
        y = self._y
        gp = GP.init_cl(seed=0, generations=5, func=['+', '-']).train(x, y)
        print gp._nop
        assert len(filter(lambda x: x != -1, gp._nop[2:])) == 0
        assert len(filter(lambda x: x != -1, gp._nop[:2])) == 2

    def test_get_sons(self):
        gp = self._gp
        gp.create_population()
        var = gp.nfunc
        cons = var + 1
        gp.population[0] = np.array([0, 0, 2, 2, var, var, cons,
                                     2, var, cons+1, cons+2])
        gp._p_constants[0] = self._pol
        sons = np.empty(2, dtype=np.int)
        gp._tree.get_sons_test(gp.population[0],
                               2, sons)
        assert np.all(sons == np.array([3, 6]))
        gp._tree.get_sons_test(gp.population[0],
                               7, sons)
        print sons
        assert np.all(sons == np.array([8, 9]))

    def test_best_length(self):
        gp = self._gp
        gp.create_population()
        var = gp.nfunc
        gp._p[0] = np.array([0, 0, 0, var, 1, var, var, var, var+1])
        gp._p_constants[0][0] = 1
        gp._p[1] = np.array([0, 0, var, var, var+1])
        gp._p_constants[1][0] = 1
        print gp.fitness(0)
        print gp.fitness(1)
        assert gp.best == 1

    def test_best(self):
        gp = self._gp
        gp.create_population()
        var = gp.nfunc
        gp._p_constants[1][0] = 1
        gp._p[1] = np.array([0, 0, var, var, var+1])
        gp.fitness(1)
        gp.best = 0
        assert gp.best == 1
        gp._fitness[0] = 0
        gp.best = 0
        assert gp.best == 0

    def test_new_best(self):
        gp = self._gp
        gp.set_test(np.array([[0]]))
        gp.create_population()
        var = gp.nfunc
        gp._p_constants[1][0] = 1
        gp._p[1] = np.array([0, 0, var, var, var+1])
        gp.fitness(1)
        # The following ind is invalid for the test set
        gp._p[0] = np.array([0, 0, var, var, 3, var, var])
        gp.fitness(0)
        gp._p[2] = np.array([0, var, 3, var, var])
        gp.fitness(2)
        # assert gp._best_fit is None
        print gp._best_fit, gp._fitness[:3], gp.best
        assert gp.best == 1

    def test_depth(self):
        gp = self._gp
        var = gp.nfunc
        ind = np.array([0, 0, var, var, 0, var, 1, var, var])
        depth = np.empty_like(ind)
        end = gp._tree.compute_depth(ind, depth)
        assert end == ind.shape[0]
        assert np.all(depth == np.array([0, 1, 2, 2, 1, 2, 2, 3, 3]))

    def test_depth_histogram(self):
        gp = self._gp
        var = gp.nfunc
        ind = np.array([0, 0, var, var, 0, var, 1, var, var])
        depth = np.empty_like(ind)
        end = gp._tree.compute_depth(ind, depth)
        hist = np.zeros_like(ind)
        e = gp._tree.compute_depth_histogram(depth, hist, end)
        assert e == 4
        assert np.all(hist[:4] == np.array([1, 2, 4, 2]))
        e = gp._tree.compute_depth_histogram(depth, hist, end)
        assert e == 4
        assert np.all(hist[:4] == np.array([1, 2, 4, 2]))

    def test_select_xpoint_depth(self):
        gp = self._gp
        var = gp.nfunc
        ind = np.array([0, 0, var, var, 0, var, 1, var, var])
        a = np.ones_like(ind, dtype=np.bool)
        while np.any(a):
            pos = gp._tree.select_xpoint_depth(ind)
            a[pos] = False
            assert pos >= 0 and pos < ind.shape[0]

    def test_select_xpoint_uniform(self):
        gp = self._gp
        var = gp.nfunc
        ind = np.array([0, 0, var, var, 0, var, 1, var, var])
        a = np.ones_like(ind, dtype=np.bool)
        while np.any(a):
            pos = gp._tree.select_xpoint_uniform(ind)
            a[pos] = False
            assert pos >= 0 and pos < ind.shape[0]

    def test_father1_crossing_point(self):
        gp = self._gp
        var = gp.nfunc
        ind = np.array([0, 0, var, var, 0, var, 1, var, var])
        a = np.ones_like(ind, dtype=np.bool)
        while np.any(a):
            pos = gp._tree.father1_crossing_point(ind)
            a[pos] = False
            assert pos >= 0 and pos < ind.shape[0]

    def test_type_xpoint_selection(self):
        ins = lambda x: GP.init_cl(seed=0, generations=5,
                                   type_xpoint_selection=x).train(self._x,
                                                                  self._y)
        gp = ins(0)
        var = gp.nfunc
        ind = np.array([0, 0, var, var, 0, var, 1, var, var])
        a = np.zeros_like(ind)
        cnt = 100
        for i in range(cnt):
            pos = gp._tree.father1_crossing_point(ind)
            a[pos] += 1
        gp = ins(0)
        for i in range(cnt):
            pos = gp._tree.select_xpoint_uniform(ind)
            a[pos] -= 1
        assert np.all(a == 0)
        gp = ins(1)
        for i in range(cnt):
            pos = gp._tree.father1_crossing_point(ind)
            a[pos] += 1
        gp = ins(1)
        for i in range(cnt):
            pos = gp._tree.select_xpoint_depth(ind)
            a[pos] -= 1
        assert np.all(a == 0)

    def test_create_population(self):
        self._gp.create_population()

    def test_popsize(self):
        gp = self._gp
        gp.create_population()
        fit = np.array(map(lambda x: gp.fitness,
                           range(gp.popsize)))
        s = fit.argsort()[::-1][:50]
        gp.popsize = 50
        fit2 = np.array(map(lambda x: gp.fitness,
                            range(gp.popsize)))
        assert np.all(fit[s] == fit2)
        gp.popsize = 100
        fit = np.array(map(lambda x: gp.fitness,
                           range(gp.popsize)))
        assert np.all(fit2 == fit[:50])

    def test_dosimplify(self):
        gp = GP.run_cl(self._x, self._y, generations=5,
                       seed=0, do_simplify=False)
        fit = gp.fitness(gp.best)
        assert not np.isnan(fit) and not np.isinf(fit)

    def test_save(self):
        import tempfile
        s = GP.run_cl(self._x, self._y, seed=0, generations=2)
        # p = s.population
        p = []
        for i in s.population:
            p.append(i.copy())
        p = np.array(p)
        cons = s._p_constants.copy()
        fname = tempfile.mktemp()
        s.save(fname)
        with open(fname, 'r') as fpt:
            p1 = np.load(fpt)
            assert np.all(map(lambda x: np.all(p[x] == p1[x]),
                              range(s.popsize)))
            cons1 = np.load(fpt)
            assert np.all(map(lambda x: np.all(cons[x] == cons1[x]),
                              range(s.popsize)))

    def test_save_run(self):
        import tempfile
        s = GP.run_cl(self._x, self._y, seed=0, generations=2)
        # p = s.population
        p = []
        for i in s.population:
            p.append(i.copy())
        p = np.array(p)
        cons = s._p_constants.copy()
        fname = tempfile.mktemp()
        s.save(fname)
        s1 = GP.run_cl(self._x, self._y, seed=1, generations=2,
                       fname_best=fname)
        p1 = s1.population
        cons1 = s1._p_constants
        assert np.all(map(lambda x: np.all(p[x] == p1[x]),
                          range(s.popsize)))
        assert np.all(map(lambda x: np.all(cons[x] == cons1[x]),
                          range(s.popsize)))

    def test_save_best(self):
        import tempfile
        s = GP.run_cl(self._x, self._y, seed=0, generations=2)
        bs = s.best
        p = s.population[bs].copy()
        cons = s._p_constants[bs].copy()
        fname = tempfile.mktemp()
        s.save_best(fname)
        s1 = GP.run_cl(self._x, self._y, fname_best=fname,
                       generations=2)
        s1.create_population()
        print p
        print s1.population[bs]
        assert np.all(s1.population[bs] == p)
        assert np.all(s1._p_constants[bs] == cons)

    def test_predict(self):
        gp = self._gp
        gp.run()
        best = gp.best
        x1 = np.linspace(-1, 1, 997)[:, np.newaxis]
        gp._x[:] = np.nan
        y = gp.predict(x1, ind=best)
        gp.train(x1, np.zeros(997))
        yh = gp.eval(best)
        print y - yh
        assert np.all(y == yh)
        x1 = np.linspace(-1, 1, 10)[:, np.newaxis]
        gp._x[:] = np.nan
        y = gp.predict(x1, ind=best)
        gp.train(x1, np.zeros(10))
        yh = gp.eval(best)
        assert np.all(y == yh)

    def test_type_of_nodes(self):
        gp = self._gp
        nvar = gp._x.shape[1]
        nfunc = gp._nop.shape[0]
        assert gp.isfunc(nfunc-1)
        assert not gp.isfunc(nfunc)
        assert gp.isvar(nfunc)
        assert not gp.isvar(0)
        assert not gp.isvar(nfunc+nvar)
        assert gp.isconstant(nfunc+nvar)

    def test_graphviz(self):
        import StringIO
        nvar = self._gp._func.shape[0]
        self._gp.create_population()
        self._gp._p[0] = np.array([0, 2, 14,
                                   nvar, nvar+1, 0,
                                   2, nvar, nvar+2, nvar+3])
        self._gp._p_constants[0] = self._pol * -1
        cdn = """digraph SimpleGP {
edge [dir="none"];
n0 [label="+"];
n1 [label="*"];
n2 [label="sq"];
n3 [label="X0"];
n2 -> n3;
n1 -> n2;
n4 [label="-0.2000"];
n1 -> n4;
n0 -> n1;
n5 [label="+"];
n6 [label="*"];
n7 [label="X0"];
n6 -> n7;
n8 [label="0.3000"];
n6 -> n8;
n5 -> n6;
n9 [label="-0.2000"];
n5 -> n9;
n0 -> n5;
}
"""
        s = StringIO.StringIO()
        self._gp.graphviz(0, fname=s)
        s.seek(0)
        cdn2 = s.read()
        assert cdn2.rstrip().lstrip() == cdn.rstrip().lstrip()

    def test_length(self):
        gp = self._gp
        gp.create_population()
        gp._do_simplify = False
        gp._p[0] = gp.create_random_ind_full(8)
        gp._p_constants[0] = gp._ind_generated_c
        l = gp.length(gp._p[0])
        for i in range(gp._p[0].shape[0]):
            _l = gp.traverse(gp._p[0],
                             pos=i) - i
            assert _l == l[i]

    def test_crossover_mask(self):
        """Missing test"""
        assert True

    def test_crossover(self):
        x = np.linspace(0, 1, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        gp = GP(seed=0, min_length=2).train(x, y)
        gp.create_population()
        lst = [np.array([3,2,2,3,1,0,20,0,17,21,0,0,22,17,1,17,23,0,24,3,25,17,3,2,3,1,26,17,0,27,17,3,2,17,28,1,29,17,0,30,0,1,31,17,1,32,17,2,0,3,33,1,0,17,17,2,17,17,0,2,0,34,17,0,17,35,1,36,1,37,17,1,2,38,1,2,39,17,1,40,17,1,2,41,0,17,42,2,43,3,17,44,3,0,3,3,1,2,17,45,3,17,46,2,2,17,47,3,48,17,1,1,0,17,17,49,0,2,50,17,2,17,17,1,2,3,51,2,17,52,2,53,17,2,3,0,54,17,2,17,55,2,1,56,17,0,17,57,0,0,0,58,3,59,17,0,1,0,17,60,1,17,61,1,62,17,3,3,3,3,17,63,64,1,0,65,17,1,66,17,0,0,17,17,1,3,17,2,2,67,17,3,17,68,69]),
np.array([3,0,1,3,3,2,17,20,0,17,21,2,1,22,17,0,17,23,1,24,1,25,2,26,17,2,3,2,2,27,17,3,17,28,3,29,2,17,30,1,2,31,2,32,17,2,0,17,17,2,17,33,1,0,2,34,1,0,17,35,36,2,0,37,2,38,17,0,2,17,17,3,17,39,2,1,1,40,2,41,17,0,2,42,17,0,17,43,3,2,17,17,3,44,17]),
np.array([0,2,1,20,1,21,17,2,3,17,22,1,17,23,2,3,0,0,17,17,0,17,17,1,0,17,17,0,17,24,2,1,2,25,2,17,26,0,17,17,1,0,17,17,0,17,27]),
np.array([2,3,17,0,0,20,17,21,1,2,2,17,22,2,17,23,2,0,24,17,25]),
np.array([2,17,3,17,20]),
np.array([0,2,0,1,20,17,2,21,17,3,2,17,17,0,17,22,0,3,23,17,1,3,17,25,0,17,26]),
np.array([3,0,3,3,20,17,0,17,17,0,21,0,17,17,3,2,1,22,17,25,1,2,17,17,0,17,17]),
np.array([0,0,0,3,2,3,3,17,20,2,21,17,2,3,17,22,23,0,1,0,17,24,0,3,17,25,17,1,0,26,17,27,2,0,1,2,28,17,3,17,29,1,2,17,30,3,31,17,0,3,1,32,17,2,17,17,1,1,17,33,2,34,17,3,1,3,35,2,1,36,17,0,37,17,1,3,0,38,17,2,17,39,1,0,17,17,2,17,17,1,1,1,1,40,17,3,17,41,0,3,42,17,2,17,17,3,43,0,3,17,44,45,0,3,0,0,0,46,2,47,17,1,2,17,48,49,2,1,0,17,17,50,3,51,0,17,52,0,3,1,3,17,53,2,17,54,0,2,17,17,55,0,1,3,17,56,57,58,3,0,1,2,0,17,59,0,60,17,61,62,1,1,3,63,0,17,17,3,64,0,17,17,3,0,0,17,65,66,3,67,1,68,17]),
np.array([1,3,17,20,2,1,17,21,0,22,0,17,23]),
np.array([3,0,1,22,17,25,26])]
        for i in range(10):
            gp._xo_father1 = i
            gp._xo_father2 = i + 1
            ind = gp.crossover(gp._p[gp._xo_father1],
                               gp._p[gp._xo_father2])
            assert (ind == lst[i]).sum()
            # print "np.array([" + ",".join(map(str, ind)) + "]),"

    def test_any_constant(self):
        gp = GP(nrandom=0, seed=0,
                do_simplify=False).train(self._x,
                                         self._y)
        gp.create_population()
        for i in gp._p:
            assert not gp.any_constant(i)
        gp = GP(min_length=8).train(self._x,
                                    self._y)
        gp.create_population()
        for i in gp._p:
            assert gp.any_constant(i)

















