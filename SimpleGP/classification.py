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
import array
import math
from SimpleGP.forest import SubTreeXOPDE, SubTreeXO
from SimpleGP.simplegp import GPS
from SimpleGP.sparse_array import SparseArray


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


class Bayes(GPS, SubTreeXO):
    def __init__(self, ntrees=5, nrandom=0, max_length=1024, ncl=None,
                 class_freq=None,
                 seed=0, **kwargs):
        super(Bayes, self).__init__(ntrees=ntrees, nrandom=nrandom,
                                    max_length=max_length, seed=seed,
                                    **kwargs)
        self._elm_constants = None
        self._ncl = ncl
        self._class_freq = class_freq
        if self._class_freq is not None:
            assert len(self._class_freq) == self._ncl

    def train(self, *args, **kwargs):
        super(Bayes, self).train(*args, **kwargs)
        self._nop[self._output_pos] = self._ntrees
        if self._ncl is None:
            self._ncl = np.unique(self._f.tonparray()).shape[0]
        if self._class_freq is None:
            self._class_freq = self._f.class_freq(self._ncl)
        tot = sum(self._class_freq)
        self._log_class_prior = array.array('d', map(lambda x:
                                                     math.log(x / tot),
                                                     self._class_freq))
        self._class_prior = array.array('d', map(lambda x:
                                                 x / tot,
                                                 self._class_freq))
        return self

    def create_population(self):
        if self._elm_constants is None or\
           self._elm_constants.shape[0] != self._popsize:
            self._elm_constants = np.empty(self._popsize,
                                           dtype=np.object)
        return super(Bayes, self).create_population()

    def early_stopping_save(self, k, fit_k=None):
        """
        Storing the best so far on the validation set.
        This funtion is called from early_stopping
        """
        assert fit_k is not None
        self._early_stopping = [fit_k,
                                self.population[k].copy(),
                                self._p_constants[k].copy(),
                                self._elm_constants[k],
                                self._class_freq,
                                self._pr_test_set.copy()]

    def set_early_stopping_ind(self, ind, k=0):
        if self._best == k:
            raise "Losing the best so far"
        self.population[k] = ind[1]
        self._p_constants[k] = ind[2]
        self._elm_constants[k] = ind[3]
        self._class_freq = ind[4]
        tot = sum(self._class_freq)
        self._log_class_prior = array.array('d', map(lambda x:
                                                     math.log(x / tot),
                                                     self._class_freq))
        self._class_prior = array.array('d', map(lambda x:
                                                 x / tot,
                                                 self._class_freq))

    def joint_log_likelihood(self, k):
        self._computing_fitness = k
        super(Bayes, self).eval_ind(self.population[k], pos=0,
                                    constants=self._p_constants[k])
        Xs = self._eval.get_output()
        y = self._f
        if self._fitness[k] > -np.inf:
            [mu, var, index] = self._elm_constants[k]
            if len(index) == 0:
                return None
            elif len(index) < len(Xs):
                Xs = map(lambda x: Xs[x], index)
        else:
            index = filter(lambda x: Xs[x].isfinite(), range(len(Xs)))
            if len(index) == 0:
                return None
            elif len(index) < len(Xs):
                Xs = map(lambda x: Xs[x], index)
            mu = y.mean_per_cl(Xs, self._class_freq)
            var = y.var_per_cl(Xs, mu, self._class_freq)
            self._elm_constants[k] = [mu, var, index]
        llh = y.joint_log_likelihood(Xs, mu, var, self._log_class_prior)
        return llh

    def eval_ind(self, ind, **kwargs):
        if self._computing_fitness is None:
            cdn = "Use eval with the number of individual, instead"
            NotImplementedError(cdn)
        k = self._computing_fitness
        llh = self.joint_log_likelihood(k)
        if llh is not None and np.all(np.isfinite(llh)):
            return SparseArray.fromlist(llh.argmax(axis=1))
        return SparseArray.fromlist(map(lambda x: np.inf,
                                        range(self._x[0].size())))

    def distance(self, y, yh):
        return y.BER(yh, self._class_freq)

