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
from SimpleGP.forest import SubTreeXO
from SimpleGP.simplegp import GPS
from SimpleGP.sparse_array import SparseArray
import numpy as np
import array
import math


class Bayes(GPS, SubTreeXO):
    def __init__(self, ntrees=5, nrandom=0, max_length=1024, ncl=None,
                 class_freq=None, use_st=0,
                 seed=0, **kwargs):
        super(Bayes, self).__init__(ntrees=ntrees, nrandom=nrandom,
                                    max_length=max_length, seed=seed,
                                    **kwargs)
        if use_st == 1:
            raise NotImplementedError('Cache is not implemented')
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
                                self._class_freq]

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

    def predict_log_proba(self, k, X=None):
        from sklearn.utils.extmath import logsumexp
        jll = self.joint_log_likelihood(k, X=X)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict(self, X, ind=None):
        if ind is not None:
            fit_k = self._fitness[ind]
            self._fitness[ind] = 1
        pr = super(Bayes, self).predict(X, ind=ind)
        if ind is not None:
            self._fitness[ind] = fit_k
        return pr

    def predict_proba(self, X=None, ind=None):
        if ind is None:
            ind = self.best
        fit_k = self._fitness[ind]
        self._fitness[ind] = 1
        pr = np.exp(self.predict_log_proba(ind, X=X))
        self._fitness[ind] = fit_k
        return pr

    def joint_log_likelihood(self, k, X=None):
        self._computing_fitness = k
        if X is not None:
            self._eval.X(X)
        super(Bayes, self).eval_ind(self.population[k], pos=0,
                                    constants=self._p_constants[k])
        if X is not None:
            self._eval.X(self._x)
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
