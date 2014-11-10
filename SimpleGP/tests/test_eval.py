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


class TestEval(object):
    def __init__(self):
        x = np.linspace(-10, 10, 100)
        pol = np.array([0.2, -0.3, 0.2])
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._gp = GP(fname_best=None).train(x, y)
        self._gp.create_population()
        self._cons = 1.2
        self._gp._p_constants[0][0] = self._cons
        self._nfunc = self._gp._nop.shape[0]
        self._nvar = self._nfunc + self._gp._x.shape[1]

    def test_sum(self):
        self._gp._p[0] = np.array([0, self._nfunc, self._nvar], dtype=np.int)
        y = self._gp._x.flatten() + self._cons
        yh = self._gp.eval(0)
        assert np.fabs(y - yh).sum() == 0

    def test_subtract(self):
        self._gp._p[0] = np.array([1, self._nfunc, self._nvar], dtype=np.int)
        y = self._gp._x.flatten() - self._cons
        yh = self._gp.eval(0)
        assert np.fabs(y - yh).sum() == 0

    def test_multiply(self):
        self._gp._p[0] = np.array([2, self._nfunc, self._nvar], dtype=np.int)
        y = self._gp._x.flatten() * self._cons
        yh = self._gp.eval(0)
        assert np.fabs(y - yh).sum() == 0
