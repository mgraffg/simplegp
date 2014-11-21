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
        elmpde = ELMPDE.run_cl(x, y, ntrees=3, generations=2)
        elm = ELM.run_cl(x, y, ntrees=3, generations=2)
        assert elm.fitness(elm.best) != elmpde.fitness(elmpde.best)

    def test_load_elm(self):
        import tempfile
        x, y = self.create_problem()
        elm = ELM.run_cl(x, y, ntrees=3, generations=2)
        yh = elm.predict(elm._x)
        fname = tempfile.mktemp()
        elm.save_best(fname)
        elm = ELM.run_cl(x, y, ntrees=3, generations=2, fname_best=fname)
        assert np.all(elm.predict(elm._x) == yh)
