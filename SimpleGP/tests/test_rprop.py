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
from SimpleGP import GPPDE
import numpy as np


class TestRprop(object):
    def __init__(self):
        x = np.linspace(-1, 1, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._gp = GPPDE.init_cl().train(x, y)
        self._gp.create_population()

    def test_rprop(self):
        gp = self._gp
        nvar = gp.nfunc
        gp.population[0] = np.array([0, 2, 14,
                                     nvar, nvar+1, 0,
                                     2, nvar, nvar+2, nvar+3])
        gp._p_constants[0] = self._pol * -1
        fit = gp.fitness(0)
        gp.rprop(0)
        fit2 = gp.fitness(0)
        print fit, fit2, gp._nop[14], gp._func[14]
        assert fit2 > fit
        mudiff = np.fabs(self._pol - gp._p_constants[0]).mean()
        assert mudiff < 0.05

    def test_rprop_run(self):
        gp = self._gp
        nvar = gp.nfunc
        for i in range(2):
            gp.population[i] = np.array([0, 2, 14,
                                         nvar, nvar+1, 0,
                                         2, nvar, nvar+2, nvar+3])
            gp._p_constants[i] = self._pol * -1
        fit = gp.fitness(0)
        assert gp.best == 0
        gp._update_best_w_rprop = True
        fit2 = gp.fitness(1)
        mudiff = np.fabs(self._pol - gp._p_constants[1]).mean()
        assert mudiff < 0.05
        assert fit2 > fit
        assert gp.best == 1
