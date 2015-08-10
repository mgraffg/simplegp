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
from SimpleGP import EGPS, SparseArray


class TestSGPS(object):
    def __init__(self):
        x = np.linspace(-10, 10, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        self._y = (X.T * pol).sum(axis=1)
        self._x = x[:, np.newaxis]

    def test_EGPS(self):
        gp = EGPS(generations=3).fit(self._x,
                                     self._y, test=self._x,
                                     test_y=self._y)
        assert isinstance(gp.early_stopping[-1], SparseArray)

    def test_cache(self):
        gp = EGPS(seed=0, nrandom=0,
                  use_st=1, generations=3).fit(self._x,
                                               self._y)
        while True:
            if gp.pre_crossover() and gp._xo_father1 != 0 and\
               gp._xo_father2 != 0:
                res = gp.crossover(gp.population[gp._xo_father1],
                                   gp.population[gp._xo_father2])
                if gp._ind_eval_st is not None:
                    break
        f1 = gp._xo_father1
        gp.kill_ind(0, res)
        assert np.all(gp._eval_A_st[f1] == gp._eval_A_st[0])
        f1_o = map(lambda x: gp._tree.get_pos_arg(gp.population[f1], 0, x),
                   range(gp._ntrees))
        trees = map(lambda x: gp._eval_st[f1][x], f1_o)
        s_o = map(lambda x: gp._tree.get_pos_arg(gp.population[0], 0, x),
                  range(gp._ntrees))
        s_trees = map(lambda x: gp._eval_st[0][x], s_o)
        assert len(filter(lambda x: x is None, s_trees)) == 1
        gp.eval(0)
        np.set_printoptions(precision=3)
        r = filter(lambda x: x[1] is None, enumerate(s_trees))[0][0]
        assert (gp._eval_A_st[f1] - gp._eval_A_st[0])[r, r] != 0
        trees2 = gp._eval.get_output()
        r = filter(lambda (x, y): x == y, zip(trees, trees2))
        assert len(r) == gp._ntrees - 1
