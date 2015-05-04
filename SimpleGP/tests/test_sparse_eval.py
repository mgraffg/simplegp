# Copyright 2015 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from SimpleGP import SparseEval, GP
from nose.tools import assert_almost_equals
import numpy as np


def create_problem():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = np.vstack((x, np.sqrt(np.fabs(x)))).T
    return x, y


def test_eval():
    x, y = create_problem()
    gp = GP.run_cl(x, y, generations=2)
    sparse = SparseEval(gp._nop)
    sparse.X = x
    for i in range(gp.popsize):
        y = gp.predict(gp._x, ind=i)
        if not np.all(np.isfinite(y)):
            continue
        print gp.print_infix(gp.population[i],
                             constants=gp._p_constants[i])
        hy = sparse.eval(gp.population[i], gp._p_constants[i])
        # print hy[:10]
        # print y[:10]
        m = np.fabs(y - hy) > 0
        print hy[m]
        print y[m]
        print gp._x[m]
        assert_almost_equals(np.fabs(y - hy).sum(), 0)


def test_eval_simple_tree():
    x, y = create_problem()
    gp = GP().train(x, y)
    sparse = SparseEval(gp._nop)
    sparse.X = x
    assert sparse.nvar == x.shape[1]
    assert sparse.nfunc == gp._nop.shape[0]
    y = x[:, 0] + x[:, 1] * 12.1 + 12.1
    var = sparse.nfunc
    cons = sparse.nfunc + sparse.nvar
    ind = np.array([0, var, 0, 2, var+1, cons, cons], dtype=np.int)
    cons = np.array([12.1])
    print gp.print_infix(ind, constants=cons)
    hy = sparse.eval(ind, cons)
    print hy[:10]
    print y[:10]
    print np.fabs(y - hy).sum()
    assert_almost_equals(np.fabs(y - hy).sum(), 0)
