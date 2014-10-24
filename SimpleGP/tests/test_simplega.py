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
from SimpleGP import SimpleGA, BestNotFound
import numpy as np


def test_SimpleGA():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA.run_cl(X, f, generations=5)
    ind = s._p[s.get_best()]
    func = lambda x: not np.any(np.isnan(x)) and not np.any(np.isinf(x))
    assert func(ind)


def test_SimpleGA_run_cl():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    x1 = np.linspace(-1, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    X1 = np.vstack((x1**2, x1, np.ones(x1.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA().run_cl(X, f, test=X1, generations=5)
    ind = s._p[s.get_best()]
    func = lambda x: not np.any(np.isnan(x)) and not np.any(np.isinf(x))
    assert func(ind)


def test_SimpleGA_run_cl_error():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    x1 = np.linspace(-1, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    X1 = np.vstack((x1**2, x1, np.ones(x1.shape[0]))).T
    X1[:, 0] = np.inf
    f = (X * pol).sum(axis=1)
    try:
        SimpleGA().run_cl(X, f, generations=5,
                          test=X1)
    except BestNotFound:
        return
    assert False


def test_popsize_property():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA.init_cl(popsize=10, generations=5).train(X, f)
    s.create_population()
    s.popsize = 100
    assert s._p.shape[0] == 100


def test_save():
    import tempfile
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA.init_cl(popsize=10, generations=5).train(X, f)
    s.create_population()
    map(lambda x: s.fitness, range(s.popsize))
    fname = tempfile.mktemp()
    p = s.population
    s.save(fname)
    s1 = SimpleGA.init_cl(fname_best=fname,
                          popsize=10, generations=5).train(X, f)
    s1.create_population()
    assert np.all(p == s1.population)


def test_kill_ind_best():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA.run_cl(X, f, popsize=10, generations=5)
    try:
        s.kill_ind(s.best, s._p[0])
    except BestNotFound:
        return
    assert False


def test_create_population():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA()
    s.train(X, f)
    s.create_population()
    p = s.population.copy()
    s.create_population()
    print p[0]
    print s.population[0]
    assert np.all(map(lambda x: np.all(p[x] == s.population[x]),
                      range(s.popsize)))
