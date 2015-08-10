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
from SimpleGP import ELMPDE, ELM
import numpy as np


class TestELM(object):
    def create_problem(self):
        x = np.linspace(-10, 10, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._x = x
        self._y = y
        return x, y

    def test_early_stopping(self):
        self.create_problem()
        gp = ELM(generations=3, popsize=100).fit(self._x, self._y,
                                                 test=self._x[::-1],
                                                 test_y=self._y[::-1])
        assert np.all(gp.predict(gp._x[::-1]) == gp.early_stopping[-1])
        print gp.early_stopping[0], gp.fitness(gp.best), gp.best
        gp.population[0] = gp.early_stopping[1]
        gp._p_constants[0] = gp.early_stopping[2]
        gp._elm_constants[0] = gp.early_stopping[3]
        assert np.all(gp.predict(gp._x, 0) == gp.predict(gp._x))

    def test_elm(self):
        x, y = self.create_problem()
        elm = ELM.run_cl(x, y, ntrees=3, generations=2)
        assert elm._output.shape[0] == 3

    def test_eval(self):
        x, y = self.create_problem()
        elm = ELMPDE(ntrees=3)
        elm.train(x, y)
        elm.create_population()
        print elm.eval(0)
        assert elm.eval(0).shape == y.shape

    def test_ntrees(self):
        x, y = self.create_problem()
        elm = ELMPDE.run_cl(x, y, ntrees=3, generations=2)
        assert elm._output.shape[0] == 3

    def test_predict(self):
        x, y = self.create_problem()
        elm = ELMPDE.run_cl(x, y, ntrees=3, generations=2)
        xp = np.linspace(-10, 10, 1000)[:, np.newaxis]
        assert np.all(np.isfinite(elm.predict(xp)))
        assert elm.predict(xp).shape[0] == 1000

    def test_difference(self):
        x, y = self.create_problem()
        elmpde = ELMPDE.run_cl(x, y, nrandom=100, ntrees=3, generations=2)
        elm = ELM.run_cl(x, y, nrandom=100, ntrees=3, generations=2)
        assert elm.fitness(elm.best) != elmpde.fitness(elmpde.best)

    def test_load_elm(self):
        import tempfile
        x, y = self.create_problem()
        elm = ELM.run_cl(x, y, ntrees=3, generations=2)
        yh = elm.predict(elm._x)
        fname = tempfile.mktemp()
        elm.save_best(fname)
        elm2 = ELM.run_cl(x, y, ntrees=3, generations=2, fname_best=fname)
        assert np.all(elm.predict(elm._x) == yh)
        r = elm._elm_constants[elm.best] == elm2._elm_constants[elm2.best]
        assert np.all(r)

    def test_elm_rational(self):
        def save(elm):
            import tempfile
            fname = tempfile.mktemp()
            elm.save_best(fname)
            return fname

        def cons(elm):
            bs = elm.best
            fit = elm.fitness(bs)
            print elm.distance(elm._f, elm.eval(bs)),\
                elm.population[bs],\
                elm._ntrees, bs
            elm._fitness[bs] = -np.inf
            cons = elm._elm_constants[bs].copy()
            elm._elm_constants[bs] = None
            print elm.fitness(bs), fit, elm._best_fit
            assert elm.fitness(bs) == fit
            r = elm._elm_constants[bs] == cons
            assert np.all(r)
        y = np.array([-1.173305, -1.210321, -1.234109, -1.245411,
                      -1.245645, -1.236385, -1.218823, -1.193283, -1.158863,
                      -1.11328, -1.053116, -0.974774, -0.876306, -0.759682,
                      -0.632008, -0.504144, -0.386973, -0.287819, -0.20919,
                      -0.149825, -0.106533])
        x = np.arange(-1, 1.1, 0.1)[:, np.newaxis]
        elm = ELM.run_cl(x, y, pxo=0.1, ppm=0, generations=5)
        assert elm._tree.get_select_root() == 0
        cons(elm)
        elm2 = ELM.run_cl(x, y, fname_best=save(elm), generations=2)
        assert np.all(elm.predict(elm._x) == elm.predict(elm._x))
        r = elm._elm_constants[elm.best] == elm2._elm_constants[elm2.best]
        print elm._elm_constants[elm.best], elm2._elm_constants[elm2.best]
        print r
        assert np.all(r)

    def test_subtree_mask_bug(self):
        x, y = self.create_problem()
        gp = ELM.run_cl(x, y, ntrees=2, generations=2)
        nvar = gp.nfunc
        ind = np.array([15, 0, nvar, nvar, nvar], dtype=np.int)
        ind2 = np.array([15, nvar, nvar], dtype=np.int)
        c = gp._tree.crossover_mask(ind, ind2, 1)
        assert gp._tree_mask[:ind2.shape[0]].sum() == c
