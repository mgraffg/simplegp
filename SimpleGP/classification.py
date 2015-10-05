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
from SimpleGP.egp import EGPS
from sklearn.linear_model import LogisticRegression


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


class EGPSL(EGPS):
    def fitness_validation(self, k):
        """
        Fitness function used in the validation set.
        In this case it is the one used on the evolution
        """
        cnt = self._test_set_y.shape[0]
        fit_k = -self.distance(self._test_set_y,
                               self._pr_test_set[:cnt])
        return fit_k

    def train(self, *args, **kwargs):
        super(EGPSL, self).train(*args, **kwargs)
        self._f = self._f.tonparray()

    def distance(self, y,  yh):
        return -Classification.f1(y, yh).mean()

    def test_f(self, x):
        return np.all(np.isfinite(x))

    def set_test(self, x, y=None):
        """
        x is the set test, this is used to test, during the evolution, that
        the best individual does not produce nan or inf
        """
        super(EGPSL, self).set_test(x, y=y)
        if y is not None:
            self._test_set_y = self._test_set_y.tonparray()

    def compute_coef(self, k, r):
        m = LogisticRegression(fit_intercept=False)
        A = np.array(map(lambda x: x.tonparray(),
                         r)).T
        m.fit(A,
              self._f)
        return m.coef_

    def eval_ind(self, ind, **kwargs):
        if self._computing_fitness is None:
            cdn = "Use eval with the number of individual, instead"
            NotImplementedError(cdn)
        k = self._computing_fitness
        super(EGPS, self).eval_ind(ind, **kwargs)
        r = self._eval.get_output()
        if self._fitness[k] > -np.inf:
            coef, norm, index = self._elm_constants[k]
            r = map(lambda (y, x): (x - y[0]) / y[1], zip(norm, r))
            r = map(lambda x: r[x], index)
        else:
            norm = map(lambda x: (x.mean(), x.std()), r)
            r = map(lambda (y, x): (x - y[0]) / y[1], zip(norm, r))
            index = filter(lambda x: r[x].isfinite(), range(len(r)))
            if len(index) == 0:
                self._elm_constants[k] = ([1.0], norm, [0])
                return r[0]
            else:
                r = map(lambda x: r[x], index)
                coef = self.compute_coef(k, r)
        yh = []
        for c in coef:
            res = r[0] * c[0]
            for i in range(1, len(r)):
                res = res + (r[i] * c[i])
            yh.append(res.tonparray())
        yh = np.vstack(yh).argmax(axis=0)
        self._elm_constants[k] = (coef, norm, index)
        return yh
