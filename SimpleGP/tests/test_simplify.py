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


class TestSimplify(object):
    def __init__(self):
        x = np.linspace(0, 1, 100)
        pol = np.array([0.2, -0.3, 0.2])
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        y = (X.T * pol).sum(axis=1)
        x = x[:, np.newaxis]
        self._gp = GP(fname_best=None).train(x, y)
        self._gp.create_population()

    def test_arithmetic(self):
        # ((((X0 + X0) + (X0 - X0)) / ((X0 - X0) + (X0 * X0))) *
        # ((X0 / X0) - (X0 / X0)))
        nvar = self._gp._nop.shape[0]
        ind = np.array([2, 3, 0, 0, nvar, nvar, 1, nvar, nvar,
                        0, 1, nvar, nvar, 2, nvar, nvar, 1, 3,
                        nvar, nvar, 3, nvar, nvar], dtype=np.int)
        ind2 = self._gp.simplify(ind)
        c2 = self._gp._ind_generated_c
        assert "0.0" == self._gp.print_infix(ind2,
                                             constants=c2)

    def test_arithmetic2(self):
        # (((X0 / X0) + (X0 - X0)) - ((X0 * X0) / (X0 * X0)))
        nvar = self._gp._nop.shape[0]
        ind = np.array([1, 0, 3, nvar, nvar, 1, nvar, nvar,
                        3, 2, nvar, nvar, 2, nvar, nvar], dtype=np.int)
        ind2 = self._gp.simplify(ind)
        c2 = self._gp._ind_generated_c
        assert "0.0" == self._gp.print_infix(ind2, constants=c2)

    def test_ln(self):
        # ln(exp((X0 * X0)))
        nvar = self._gp._nop.shape[0]
        ind = np.array([13, 5, 2, nvar, nvar], dtype=np.int)
        ind2 = self._gp.simplify(ind)
        c2 = self._gp._ind_generated_c
        assert "(X0 * X0)" == self._gp.print_infix(ind2, constants=c2)

    def test_exp(self):
        # exp(ln((X0 * X0)))
        nvar = self._gp._nop.shape[0]
        ind = np.array([5, 13, 2, nvar, nvar], dtype=np.int)
        ind2 = self._gp.simplify(ind)
        c2 = self._gp._ind_generated_c
        assert "(X0 * X0)" == self._gp.print_infix(ind2, constants=c2)

    def test_sqrt(self):
        # sqrt(sq((X0 * X0)))
        nvar = self._gp._nop.shape[0]
        ind = np.array([6, 14, 2, nvar, nvar], dtype=np.int)
        ind2 = self._gp.simplify(ind)
        c2 = self._gp._ind_generated_c
        assert "(X0 * X0)" == self._gp.print_infix(ind2, constants=c2)

    def test_sq(self):
        # sq(sqrt((X0 * X0)))
        nvar = self._gp._nop.shape[0]
        ind = np.array([14, 6, 2, nvar, nvar], dtype=np.int)
        ind2 = self._gp.simplify(ind)
        c2 = self._gp._ind_generated_c
        assert "(X0 * X0)" == self._gp.print_infix(ind2, constants=c2)

    def test_sum_0(self):
        gp = self._gp
        nvar = self._gp._nop.shape[0]
        gp._p[0] = np.array([0, 2, nvar, nvar+2, nvar+1], dtype=np.int)
        gp._p_constants[0] = np.array([0, 1.4])
        gp.simplify(0)
        assert gp.print_infix(0) == "(X0 * 1.4)"
        gp._p[0] = np.array([0, nvar+1, 2, nvar, nvar+2], dtype=np.int)
        gp._p_constants[0] = np.array([0, 1.4])
        gp.simplify(0)
        assert gp.print_infix(0) == "(X0 * 1.4)"

    def test_subtract_S_S(self):
        ## This test fails because there is a bug in simplify
        ## the problem is that simplify.equal does not test
        ## whether two constants are equal it relies on
        ## the index of the constant instead of its value
        gp = self._gp
        nvar = self._gp._nop.shape[0]
        gp._p[0] = np.array([1, 0, 2, nvar, nvar+2, nvar+1,
                             2, nvar, nvar+2], dtype=np.int)
        gp._p_constants[0] = np.array([0, 1.4])
        gp.simplify(0)
        assert gp.print_infix(0) == "0.0"
