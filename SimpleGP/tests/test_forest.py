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
from SimpleGP import SubTreeXO
import numpy as np


def test_get_subtree():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    pol1 = np.array([-0.2, 0.3, -0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = np.vstack(((X.T * pol).sum(axis=1),
                   (X.T * pol1).sum(axis=1))).T
    x = x[:, np.newaxis]
    gp = SubTreeXO(generations=30,
                   max_length=1000).train(x, y)
    nvar = gp._nop.shape[0]
    ind = np.array([0, 1, nvar, 1, nvar+1, nvar, 2, nvar, nvar])
    t = [0, 0, 0, 0, 0, 1, 1, 1]
    for k, v in enumerate(t):
        assert gp._tree.get_subtree(ind, k+1) == v


def test_subtree_mask():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    pol1 = np.array([-0.2, 0.3, -0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = np.vstack(((X.T * pol).sum(axis=1),
                   (X.T * pol1).sum(axis=1))).T
    x = x[:, np.newaxis]
    gp = SubTreeXO(generations=30,
                   max_length=1000).train(x, y)
    nvar = gp._nop.shape[0]
    ind = np.array([0, 1, nvar, 1, nvar+1, nvar, 2, nvar, nvar])
    ind2 = np.array([0, 1, nvar, nvar+1, nvar])
    m = np.array([[0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1]])
    for k in range(1, ind.shape[0]):
        j = gp._tree.get_subtree(ind, k)
        gp._tree.crossover_mask(ind, ind2, k)
        diff = gp._tree_mask[:ind2.shape[0]] - m[j]
        assert np.fabs(diff).sum() == 0
