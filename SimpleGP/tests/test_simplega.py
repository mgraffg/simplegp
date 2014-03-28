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
from SimpleGP import SimpleGA
import numpy as np


def test_SimpleGA():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA.init_cl().train(X, f)
    s.run()
    assert np.fabs(pol - s._p[s.get_best()]).mean() < 0.1


def test_SimpleGA_run_cl():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    x1 = np.linspace(-1, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    X1 = np.vstack((x1**2, x1, np.ones(x1.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA().run_cl(X, f, test=X1)
    assert np.fabs(pol - s._p[s.get_best()]).mean() < 0.1


def test_SimpleGA_run_cl_error():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    x1 = np.linspace(-1, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    X1 = np.vstack((x1**2, x1, np.ones(x1.shape[0]))).T
    X1[:, 0] = np.inf
    f = (X * pol).sum(axis=1)
    s = SimpleGA().run_cl(X, f, test=X1, ntries=1)
    assert s is None
