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
from SimpleGP import GPS, SparseArray
from pymock import use_pymock, override, returns, replay


def problem():
    np.random.seed(0)
    nt = 100
    x = np.linspace(-1, 1, nt)
    x = np.vstack((x, x[::-1], 0.2*x**3+x)).T
    X = np.vstack((x[:, 0] * x[:, 1], x[:, 0], x[:, 2]))
    coef = np.array([0.5, -1.5, 0.9])
    y = (X.T * coef).sum(axis=1)
    x = x.T
    for i in range(100):
        x = np.vstack((x, np.random.uniform(size=nt)))
    return map(lambda y: SparseArray.fromlist(y), x), y


@use_pymock
def test_random_leaf():
    from SimpleGP.individual import Individual
    nop = GPS()._nop
    X, y = problem()
    ind = Individual(X, nop=nop)
    override(np.random, 'randint')
    np.random.randint(len(X))
    returns(50)
    replay()
    x = ind.random_leaf()
    assert 50 + ind.nfunc == x


@use_pymock
def test_random_func():
    from SimpleGP.individual import Individual
    nop = GPS()._nop
    nop[15] = 2
    X, y = problem()
    ind = Individual(X, nop=nop)
    assert ind.nfunc == ind._nop.shape[0]
    override(np.random, 'randint')
    np.random.randint(ind._func_allow.shape[0])
    returns(3)
    np.random.randint(5)
    returns(4)
    replay()
    x = ind.random_func()
    assert x == 3
    x = ind.random_func(nop=2)
    assert x == 15


@use_pymock
def test_individual():
    from SimpleGP.individual import Individual
    nop = GPS()._nop
    X, y = problem()
    ind = Individual(X, nop=nop)
    override(ind, 'random_depth')
    override(ind, 'grow_or_full')
    override(ind, 'random_func')
    ind.random_depth()
    returns(2)
    ind.grow_or_full()
    returns(1)
    for i in [3, 2, 1]:
        ind.random_func()
        returns(i)
    replay()
    x, f = ind.random_ind()
    assert not ind._grow
    assert x[0] == 3 and x[1] == 2 and x[4] == 1
    assert x.shape[0] == 7
    assert f.isfinite()
    ind._eval.X(X)
    y = ind._eval.eval(x, ind._constants2, to_np_array=False)
    print f.SSE(y)
    assert f.SSE(y) == 0


@use_pymock
def test_individual_prod():
    from SimpleGP.individual import Individual
    nop = GPS()._nop
    X = [SparseArray.fromlist([0, 1.]), SparseArray.fromlist([1., 0.])]
    ind = Individual(X, nop=nop)
    override(ind, 'random_depth')
    override(ind, 'grow_or_full')
    override(ind, 'random_func')
    override(ind, 'random_leaf')
    ind.random_depth()
    returns(1)
    ind.grow_or_full()
    returns(1)
    ind.random_func()
    returns(2)
    for i in [ind.nfunc+1, ind.nfunc]:
        ind.random_leaf()
        returns(i)
    ind.random_func(nop=2)
    returns(0)
    replay()
    x, f = ind.random_ind()
    assert f.nele() > 0, x.shape[0] == 3


@use_pymock
def test_individual_div():
    from SimpleGP.individual import Individual, Infeasible
    nop = GPS()._nop
    X = [SparseArray.fromlist([2., np.inf]),
         SparseArray.fromlist([1., np.inf])]
    ind = Individual(X, nop=nop)
    override(ind, 'random_depth')
    override(ind, 'grow_or_full')
    override(ind, 'random_func')
    ind.random_depth()
    returns(1)
    ind.grow_or_full()
    returns(1)
    ind.random_func()
    returns(3)
    for i in range(2):
        ind.random_func(nop=2)
        returns(3)
    replay()
    try:
        x, f = ind.random_ind()
        assert False
    except Infeasible:
        pass


@use_pymock
def test_individual_grow():
    from SimpleGP.individual import Individual
    nop = GPS()._nop
    X = [SparseArray.fromlist([2., 4.]),
         SparseArray.fromlist([1., 5.])]
    ind = Individual(X, nop=nop)
    override(ind, 'random_depth')
    override(ind, 'grow_or_full')
    override(ind, 'random_func')
    ind.random_depth()
    returns(2)
    ind.grow_or_full()
    returns(0)
    ind.grow_or_full()
    returns(0)
    ind.random_func()
    returns(0)
    ind.grow_or_full()
    returns(1)
    ind.grow_or_full()
    returns(0)
    ind.random_func()
    returns(1)
    replay()
    x, f = ind.random_ind()
    assert x[0] == 0 and x[2] == 1
    
