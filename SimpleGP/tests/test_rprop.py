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
        x = np.linspace(0, 1, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._gp = GPPDE().train(x, y)
        self._gp.create_population()

    def test_rprop(self):
        nvar = self._gp._func.shape[0]
        self._gp._p[0] = np.array([0, 2, 14,
                                   nvar, nvar+1, 0,
                                   2, nvar, nvar+2, nvar+3])
        self._gp._p_constants[0] = self._pol * -1
        fit = self._gp.fitness(0)
        self._gp.rprop(0)
        fit2 = self._gp.fitness(0)
        assert fit2 > fit
