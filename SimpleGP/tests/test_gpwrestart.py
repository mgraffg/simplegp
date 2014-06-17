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
from SimpleGP import GPwRestart
import numpy as np


class TestGPwRestart(object):
    def __init__(self):
        x = np.linspace(0, 1, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._gp = GPwRestart(generations=25).train(x, y)
        self._x = x
        self._y = y

    def test_ntimes(self):
        gp = GPwRestart.run_cl(self._x, self._y, ntimes=3,
                               generations=3, popsize=5, verbose=False)
        assert gp._ntimes == 2
        assert gp.generations == 9

    def test_gpwrestart(self):
        self._gp._gens = 5
        flag = self._gp.run()
        pr = self._gp.predict(self._gp._x)
        print self._gp._best, self._gp._fitness
        assert not np.isinf(self._gp._fitness[self._gp.get_best()])
        assert self._gp.distance(self._gp._f,
                                 pr) < 0.1
        assert flag

    def test_walltime(self):
        import time
        t = time.time()
        GPwRestart.run_cl(self._x, self._y, generations=25,
                          walltime=1)
        assert time.time() - t < 1.1
