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
import numpy as np
from SimpleGP import Generational, GenerationalPDE


class TestGenerational(object):
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

    def test_create_population(self):
        x, y = self.create_problem()
        gp = Generational.run_cl(self._x, self._y, generations=2, verbose=True)
        assert gp._sons_p.shape[0] == gp.popsize
        assert gp._sons_p_constants.shape[0] == gp.popsize

    def test_kill_ind(self):
        x, y = self.create_problem()
        gp = Generational.run_cl(self._x, self._y, generations=2, verbose=True)
        assert gp._ngens == 2
        assert gp._sons_p[-1] is None

    def test_merge_population(self):
        x, y = self.create_problem()

        class G(Generational):
            def merge_population(self):
                self._b_ind = self.population[self.best].copy()
                super(G, self).merge_population()
                self._run = False
        gp = G.run_cl(self._x, self._y, generations=2, verbose=True)
        assert np.all(gp._b_ind == gp.population[gp.best])
        m = np.ones(gp.popsize, dtype=np.bool)
        m[gp.best] = False
        for k, v in enumerate(np.where(m)[0]):
            assert np.all(gp._sons_p[k] == gp._p[v])
        assert gp._fitness[gp.best] > -np.inf

    def test_generationalPDE(self):
        x, y = self.create_problem()
        gp = GenerationalPDE(generations=2, ppm=0, verbose=True)
        gp.train(x, y)
        gp.create_population()
        gp._xo_father1 = 0
        gp.fitness(gp._xo_father1)
        assert gp._nkills == 0
        gp.mutation(gp.population[gp._xo_father1])
        print gp._nkills
        assert gp._nkills == 0
