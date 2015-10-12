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
        self._save_ind = []
        self._ncl = ncl
        self._class_freq = class_freq
        if self._class_freq is not None:
            assert len(self._class_freq) == self._ncl
        self._class_freq_test = None

    def fitness_validation(self, k):
        """
        Fitness function used in the validation set.
        In this case it is the one used on the evolution
        """
        if self._class_freq_test is None:
            self._class_freq_test = self._test_set_y.class_freq(self._ncl)
        cnt = self._test_set_y.size()
        y = self._test_set_y
        return - y.BER(self._pr_test_set[:cnt], self._class_freq_test)

    def save_ind(self, k):
        self._save_ind = [self.population[k],
                          self._p_constants[k],
                          self._elm_constants[k],
                          self._fitness[k],
                          self._class_freq]

    def restore_ind(self, k):
        ind = self._save_ind
        self.population[k] = ind[0]
        self._p_constants[k] = ind[1]
        self._elm_constants[k] = ind[2]
        self._fitness[k] = ind[3]
        self._class_freq = ind[4]
    
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
                                        range(self._eval.get_X()[0].size())))

    def distance(self, y, yh):
        return y.BER(yh, self._class_freq)


class AdaBayes(Bayes):
    def __init__(self, ntimes=2, frac_ts=1.0, **kwargs):
        super(AdaBayes, self).__init__(**kwargs)
        self._inds = []
        self._ntimes = ntimes
        self._prob = None
        self._X_all = None
        self._y_all = None
        self._beta_constants = None
        self._frac_ts = frac_ts

    def create_population(self):
        if self._beta_constants is None or\
           self._beta_constants.shape[0] != self._popsize:
            self._beta_constants = np.empty(self._popsize,
                                            dtype=np.object)
        return super(AdaBayes, self).create_population()

    def compute_beta(self, ind):
        y = self._y_all.tonparray()
        yh = super(AdaBayes, self).predict(self._X_all, ind=ind).tonparray()
        self._beta_update = y == yh
        et = (~ self._beta_update).mean()
        self._et = et
        if et > 0.95:
            return False
        self._beta_constants[ind] = et / (1 - et)
        return True

    def save_ind(self, k):
        super(AdaBayes, self).save_ind(k)
        self._save_ind.append(self._beta_constants[k])

    def restore_ind(self, k):
        super(AdaBayes, self).restore_ind(k)
        self._beta_constants[k] = self._save_ind[5]

    def set_early_stopping_ind(self, ind, k=0):
        super(AdaBayes, self).set_early_stopping_ind(ind, k=k)
        self._beta_constants[k] = ind[5]

    def early_stopping_save(self, k, fit_k=None):
        super(AdaBayes, self).early_stopping_save(k, fit_k=fit_k)
        self._early_stopping.append(self._beta_constants[k])
        if self._et > 0.5:
            return
        if self._verbose:
            print "Best so far", self.gens_ind,\
                "%0.4f" % self.early_stopping[0]
        self._inds.append(self.early_stopping)
        # updating the distribution
        beta = self._beta_constants[k]
        mask = self._beta_update
        self._prob[mask] *= beta
        mu = self._prob.sum()
        self._prob = self._prob / mu
        # updating the training size
        self.train(self._X_all, self._y_all, prob=self._prob)
        self._fitness.fill(-np.inf)
        self._best = None
        self._best_fit = None

    def predict_test_set(self, ind):
        """Predicting the test set"""
        if not self.compute_beta(ind):
            return self._test_set[0].constant(np.inf,
                                              size=self._test_set[0].size())
        return self.predict(self._test_set, ind)

    def predict(self, X, ind=None):
        assert ind is not None
        assert self._beta_constants[ind] is not None
        score = np.zeros((X[0].size(), self._ncl))
        index = np.arange(X[0].size())
        k = 0
        if self._best is not None and self._best == k:
            k += 1
        hist = None
        for inds in self._inds:
            self.save_ind(k)
            self.set_early_stopping_ind(inds, k=k)
            pr = super(AdaBayes, self).predict(X,
                                               ind=k).tonparray()
            beta = math.log(1 / self._beta_constants[k])
            score[index, pr.astype(np.int)] += beta
            self.restore_ind(k)
            hist = [inds[1], inds[5]]
        if hist is None or (np.any(hist[0] != self.population[ind]) or
                            hist[1] != self._beta_constants[ind]):
            pr = super(AdaBayes, self).predict(X, ind=ind).tonparray()
            if not np.all(np.isfinite(pr)):
                return X[0].constant(np.inf,
                                     size=X[0].size())
            beta = math.log(1 / self._beta_constants[ind])
            score[index, pr.astype(np.int)] += beta
        self._score_yh = score
        return SparseArray.fromlist(score.argmax(axis=1))

    def select_ts(self):
        index = None
        cnt = int(self._frac_ts * self._X_all[0].size())
        while index is None or index.shape[0] < cnt:
            a = np.random.uniform(size=self._prob.shape[0])
            a = np.where(a < self._prob)[0]
            np.random.shuffle(a)
            if index is None:
                index = a[:cnt]
            else:
                index = np.concatenate((index, a))
        index = index[:cnt]
        index.sort()
        return index

    def train(self, X, y, prob=None, **kwargs):
        self._X_all = X
        self._y_all = y
        if prob is None:
            prob = np.empty(y.size())
            prob.fill(1. / y.size())
            self._prob = prob
        else:
            self._prob = prob
        index = self.select_ts()
        return super(AdaBayes, self).train(map(lambda x: x[index], X),
                                           y[index], **kwargs)

    def fit(self, X, y, test=None, callback=None, callback_args=None,
            test_y=None, **kwargs):
        if test is not None:
            self.set_test(test, y=test_y)
        ntimes = self._ntimes
        fit = -np.inf
        prob = None
        for i in range(ntimes):
            self._ntimes = i
            self.train(X, y, prob=prob)
            self.create_population()
            self.init()
            self.run()
            if self.early_stopping[0] <= fit:
                break
            prob = self._prob
            fit = self.early_stopping[0]
            if callback:
                if callback_args is None:
                    callback(self)
                else:
                    callback(self, *callback_args)
        self._ntimes = ntimes
        return self


class IBayes(Bayes):
    def __init__(self, ntimes=2, **kwargs):
        super(IBayes, self).__init__(**kwargs)
        self._inds = []
        self._prev_f = None
        self._prev_index = None
        self._ntimes = ntimes

    def prev_llh(self, llh):
        self._prev_index = llh.argmax(axis=1)
        self._prev_f = llh.max(axis=1)

    def predict_llh(self, X=None, ind=None):
        k = ind
        res = [self.joint_log_likelihood(k, X=X)]
        k = 0
        if self._best is not None and self._best == k:
            k += 1
        for ind in self._inds:
            self.save_ind(k)
            self.set_early_stopping_ind(ind, k=k)
            res.append(self.joint_log_likelihood(k, X=X))
            self.restore_ind(k)
        return np.concatenate(res, axis=1)

    def predict_proba(self, X=None, ind=None):
        res = [super(IBayes, self).predict_proba(X=X, ind=ind)]
        k = 0
        if self._best is not None and self._best == k:
            k += 1
        for ind in self._inds:
            self.save_ind(k)
            self.set_early_stopping_ind(ind, k=k)
            res.append(super(IBayes, self).predict_proba(X=X, ind=k))
            self.restore_ind(k)
        return np.concatenate(res, axis=1)

    def predict(self, X=None, ind=None):
        a = self.predict_proba(X=X, ind=ind)
        return SparseArray.fromlist(a.argmax(axis=1) % self._ncl)

    def eval_ind(self, ind, **kwargs):
        if self._computing_fitness is None:
            cdn = "Use eval with the number of individual, instead"
            NotImplementedError(cdn)
        k = self._computing_fitness
        llh = self.joint_log_likelihood(k)
        if self._prev_f is None:
            if llh is not None and np.all(np.isfinite(llh)):
                return SparseArray.fromlist(llh.argmax(axis=1))
            else:
                return SparseArray.fromlist(map(lambda x: np.inf,
                                                range(self._x[0].size())))
        if llh is None or not np.all(np.isfinite(llh)):
            yh = self._prev_index
            return SparseArray.fromlist(yh)
        a = np.vstack((self._prev_index, llh.argmax(axis=1)))
        f = np.vstack((self._prev_f, llh[np.arange(llh.shape[0]), a[1]]))
        yh = a[f.argmax(axis=0), np.arange(f.shape[1])]
        return SparseArray.fromlist(yh)

    def fit(self, X, y, test=None, callback=None, callback_args=None,
            test_y=None, **kwargs):
        if test is not None:
            self.set_test(test, y=test_y)
        ntimes = self._ntimes
        fit = -np.inf
        for i in range(ntimes):
            self._ntimes = i
            self.train(X, y)
            self.create_population()
            self.init()
            self.run()
            if self.early_stopping[0] <= fit:
                break
            fit = self.early_stopping[0]
            self.population[self.best] = self.early_stopping[1]
            self._p_constants[self.best] = self.early_stopping[2]
            self._elm_constants[self.best] = self.early_stopping[3]
            llh = self.predict_llh(self._x, ind=self.best)
            self.prev_llh(llh)
            self._inds.append(map(lambda x: x, self.early_stopping))
            if callback:
                if callback_args is None:
                    callback(self)
                else:
                    callback(self, *callback_args)
        self._ntimes = ntimes
        return self
