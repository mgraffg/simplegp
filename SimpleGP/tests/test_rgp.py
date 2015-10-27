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

import numpy as np
from SimpleGP import RootGP
from nose.tools import assert_almost_equals
from pymock import use_pymock, override, returns, replay, verify


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
    return x.T, y


def test_rgp_random_leaf():
    X, y = problem()
    gp = RootGP().train(X, y)
    gp.set_test(X, y)
    a, c, e, test = gp.random_leaf()
    assert np.all(np.isfinite(c))
    assert e.isfinite()
    assert test.SSE(e) == 0


def test_rgp_population():
    X, y = problem()
    gp = RootGP().train(X, y)
    gp.set_test(X, y)
    gp.create_population()
    assert gp._fitness[gp.best] == gp._fitness.max()
    assert np.all(np.isfinite(gp._fitness))
    for k, v in enumerate(gp._pop_eval):
        assert_almost_equals(gp._fitness[k],
                             - gp.distance(gp._f,
                                           v))
    print gp.early_stopping[0], gp._fitness[gp.best]
    print gp.early_stopping[1], gp.population[gp.best]
    assert gp.early_stopping[0] == gp._fitness[gp.best]


def test_rgp_compute_coef():
    from SimpleGP import SparseArray
    X, y = problem()
    gp = RootGP().train(X, y)
    for x, xs in zip(X.T, gp._x):
        c = gp.compute_coef([xs])[0]
        a = np.linalg.lstsq(np.atleast_2d(x).T, y)[0][0]
        assert_almost_equals(c, a)
    c = gp.compute_coef(gp._x[3:])
    a = np.linalg.lstsq(X[:, 3:], y)[0]
    map(lambda (x, y): assert_almost_equals(x, y), zip(a, c))
    c = gp.compute_coef([gp._x[0], gp._x[0]])
    assert c is None
    x = X[:, 0]
    x[0] = np.nan
    a = np.linalg.lstsq(np.atleast_2d(x).T, y)[0][0]
    c = gp.compute_coef([SparseArray.fromlist(x)])[0]
    assert np.isnan(a) and np.isnan(c)


@use_pymock
def test_rgp_genetic_operators_linear_comb():
    X, y = problem()
    gp = RootGP(seed=0).train(X, y)
    gp.set_test(X, y)
    gp.create_population()
    override(gp, 'random_func')
    gp.random_func()
    returns(0)
    replay()
    son = gp.genetic_operators()
    assert son is not None
    assert gp._ind_generated_c is not None
    assert gp._ind_generated_f is not None


@use_pymock
def test_rgp_genetic_operators():
    X, y = problem()
    gp = RootGP(seed=0).train(X, y)
    gp.set_test(X, y)
    gp.create_population()
    override(gp, 'random_func')
    for i in range(1):
        gp.random_func()
        returns(5)
    replay()
    son = gp.genetic_operators()
    assert son is not None
    assert gp._ind_generated_c is not None
    assert gp._ind_generated_f is not None


def test_rgp_genetic_killind():
    X, y = problem()
    gp = RootGP(seed=0).train(X, y)
    gp.set_test(X, y)
    gp.create_population()
    son = gp.genetic_operators()
    fit = gp._fitness[0]
    gp.kill_ind(0, son)
    print fit, gp._fitness[0]
    assert fit != gp._fitness[0]


def test_RootGP_test():
    X, y = problem()
    gp = RootGP(seed=0, verbose=True,
                generations=3).fit(X, y, test=X, test_y=y)
    assert gp


def test_RootGP():
    X, y = problem()
    gp = RootGP(seed=0, verbose=True,
                generations=3).fit(X, y)
    assert gp


def test_genetic_operators():
    X, y = problem()
    gp = RootGP(seed=0).train(X, y)
    gp.set_test(X, y)
    gp.create_population()
    son = gp.genetic_operators()
    assert son[0] == 1000
        
