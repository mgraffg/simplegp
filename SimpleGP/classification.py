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
from SimpleGP.forest import SubTreeXOPDE, SubTreeXO


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

    @staticmethod
    def balance(y, seed=0, nele=np.inf):
        cnt = min(map(lambda x: (y == x).sum(), np.unique(y)))
        cnt = min([cnt, nele])
        index = np.arange(y.shape[0])
        np.random.seed(seed)
        np.random.shuffle(index)
        mask = np.zeros_like(index, dtype=np.bool)
        for cl in np.unique(y):
            m = y[index] == cl
            mask[index[m][:cnt]] = True
        return np.where(mask)[0]

    @classmethod
    def f1(cls, y, yh):
        p = cls.precision(y, yh)
        r = cls.recall(y, yh)
        f1 = 2 * p * r / (p + r)
        f1[~np.isfinite(f1)] = -np.inf
        return f1

    @staticmethod
    def recall(y, yh):
        l = []
        for cl in np.unique(y):
            m = y == cl
            r = (yh[m] == cl).sum() / float(m.sum())
            l.append(r)
        return np.array(l)

    @staticmethod
    def precision(y, yh):
        l = []
        for cl in np.unique(y):
            m = yh == cl
            p = (y[m] == cl).sum() / float(m.sum())
            l.append(p)
        return np.array(l)


class ClassificationPDE(SubTreeXOPDE, Classification):
    pass


