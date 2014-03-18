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
from .forest import SubTreeXO
from .tree import PDEXO
from .simplegp import GPPDE


class Classification(SubTreeXO):
    def train(self, x, f):
        y = np.zeros((f.shape[0], np.unique(f).shape[0]),
                     dtype=self._dtype)
        y[np.arange(y.shape[0]), f.astype(np.int)] = 1
        super(Classification, self).train(x, y)
        return self

    def predict(self, X):
        cnt = np.unique(self._f).shape[0]
        X = np.vstack((self._x[:cnt], np.atleast_2d(X)))
        x = self._x
        f = self._f
        dummy = np.zeros(X.shape[0])
        dummy[:cnt] = np.arange(cnt)
        self.train(X, dummy)
        pr = self.eval(self.get_best())[cnt:]
        self.train(x, f.argmax(axis=1))
        return pr.argmax(axis=1)


class ClassificationPDE(Classification, GPPDE):
    def train(self, x, f):
        super(ClassificationPDE, self).train(x, f)
        for i in range(self._popsize):
            self._p_st[i] = None
            self._p_error_st[i] = None
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
