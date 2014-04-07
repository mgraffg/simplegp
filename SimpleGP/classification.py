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
from SimpleGP.forest import SubTreeXO
from SimpleGP.tree import PDEXO
from SimpleGP.gppde import GPPDE


class Classification(SubTreeXO):
    def train(self, x, f):
        y = np.zeros((f.shape[0], np.unique(f).shape[0]),
                     dtype=self._dtype)
        y[np.arange(y.shape[0]), f.astype(np.int)] = 1
        super(Classification, self).train(x, y)
        return self

    def predict(self, X, ind=None):
        pr = super(Classification, self).predict(X, ind=ind)
        r = pr.argmax(axis=1).astype(self._dtype)
        m = np.any(np.isnan(pr), axis=1) | np.any(np.isinf(pr), axis=1)
        r[m] = np.nan
        return r

    @classmethod
    def init_cl(cls, nrandom=0, **kwargs):
        ins = cls(nrandom=nrandom, **kwargs)
        return ins

    @staticmethod
    def BER(y, yh):
        u = np.unique(y)
        b = 0
        for cl in u:
            m = y == cl
            b += (~(y[m] == yh[m])).sum() / float(m.sum())
        return (b / float(u.shape[0])) * 100.

    @staticmethod
    def success(y, yh):
        return (y == yh).sum() / float(y.shape[0])


class ClassificationPDE(GPPDE, Classification):
    def train(self, x, f):
        y = np.zeros((f.shape[0], np.unique(f).shape[0]),
                     dtype=self._dtype)
        y[np.arange(y.shape[0]), f.astype(np.int)] = 1
        super(Classification, self).train(x, y)
        return self

    def tree_params(self):
        self._tree_length = np.empty(self._max_length,
                                     dtype=np.int)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=np.int)
        self._tree = PDEXO(self._nop,
                           self._tree_length,
                           self._tree_mask,
                           self._min_length,
                           self._max_length,
                           select_root=0)
