# Copyright 2015 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from SimpleGP import GP, BestNotFound
import numpy as np
import types
from numpy.linalg import lstsq


class PrGP(GP):
    def __init__(self, save_only_best=False, **kwargs):
        if save_only_best:
            raise NotImplementedError("This option is not implemented")
        super(PrGP, self).__init__(save_only_best=save_only_best,
                                   **kwargs)
        self._pop_eval_mut = None
        self._pop_eval = None
        self._pop_hist = None
        self._test_set_eval = None
        self._test_set_eval_mut = None
        self._nparents = 2

    def set_best(self):
        pass

    def save_extras(self, fpt):
        super(PrGP, self).save_extras(fpt)
        best = self.best
        np.save(fpt, best)
        np.save(fpt, self._pop_hist)
        np.save(fpt, self._pop_eval)
        np.save(fpt, self._history_coef)
        np.save(fpt, self._history_ind)
        np.save(fpt, self._history_index)
        np.save(fpt, self._test_set_eval)
        np.save(fpt, self._test_set_eval_mut)

    def load_extras(self, fpt):
        super(PrGP, self).load_extras(fpt)
        best = int(np.load(fpt))
        hist = np.load(fpt)
        eval = np.load(fpt)
        coef = np.load(fpt)
        ind = np.load(fpt)
        index = np.load(fpt)
        self._test_set_eval = np.load(fpt)
        self._test_set_eval_mut = np.load(fpt)
        self._load_tmp = [best, hist, eval, coef,
                          ind, index]

    def save_hist(self, kill):
        index = self._history_index
        self._history_coef[index] = self._ind_generated_c
        f1 = self._pop_hist[self._xo_father1]
        if self._nparents == 1:
            self._history_ind[index] = np.array([f1,
                                                 self._xo_father2[0],
                                                 self._xo_father2[1]])
        else:
            f2 = self._pop_hist[self._xo_father2]
            self._history_ind[index] = np.array([f1, f2, -1])
        self._pop_hist[kill] = index
        self._history_index += 1

    def new_best_comparison(self, k):
        """
        This function is called from new_best
        """
        if self._best_fit is None:
            return True
        f = self._fitness[k]
        if self._best_fit > f:
            return False
        if self._best_fit == f:
            return False
        return True

    def improve_fit(self, son):
        fit = - self.distance(self._f, son)
        if fit <= self.fitness(self._xo_father1):
            return fit, False
        if self._nparents == 2 and fit <= self.fitness(self._xo_father2):
            return fit, False
        if self._test_set is not None:
            if not np.all(np.isfinite(self._test_set_tmp)):
                return fit, False
        return fit, True

    def kill_ind(self, kill, son):
        """
        Replace the (kill)-th individual with son
        """
        if self._best == kill:
            raise BestNotFound("Killing the best so far!")
        fit, imp = self.improve_fit(son)
        if not imp:
            return
        self._pop_eval[kill] = son
        if self._test_set is not None:
            self._test_set_eval[kill] = self._test_set_tmp
        self.save_hist(kill)
        self._fitness[kill] = fit
        self.new_best(kill)

    def predict_test_set(self, ind):
        if self._test_set_eval is not None:
            return self._test_set_eval[ind]
        return self.predict(self._test_set, ind)

    def init_test_set(self):
        if self._test_set is None:
            return
        if self._test_set_eval is not None:
            return
        self._test_set_eval = np.array(map(lambda x:
                                           self.predict(self._test_set,
                                                        x),
                                           range(self.popsize)))
        self._test_set_eval_mut = self.sigmoid(self._test_set_eval)

    def create_population(self):
        def test_fitness():
            m = np.array(map(lambda x: self.fitness(x),
                             range(self.popsize)))
            m = np.isfinite(m)
            return m
        self._pop_hist = np.arange(self.popsize)
        flag = super(PrGP, self).create_population()
        if not flag:
            self._fitness.fill(-np.inf)
        m = test_fitness()
        while (~m).sum():
            index = np.where(~m)[0]
            generator = self.create_population_generator
            for i, ind, cons in generator(popsize=index.shape[0]):
                self._p[index[i]] = ind
                self._p_constants[index[i]] = cons
            m = test_fitness()
        self._pop_eval = np.empty((self.popsize, self._f.shape[0]))
        for i in range(self.popsize):
            self._pop_eval[i] = self.eval(i).copy()
        self._pop_eval_mut = self.sigmoid(self._pop_eval)
        if not flag:
            best, hist, eval, coef, ind, index = self._load_tmp
            self._pop_hist = hist
            self._pop_eval = eval
            for i in range(self.popsize):
                self._fitness[i] = - self.distance(self._f, eval[i])
            self._history_coef = coef
            self._history_ind = ind
            self._history_index = index
            self.new_best(int(best))
        else:
            self._history_coef = np.zeros((self.popsize *
                                           self.generations,
                                           2), dtype=self._dtype)
            self._history_ind = np.empty((self.popsize *
                                          self.generations,
                                          3), dtype=np.int)
            self._history_ind.fill(-1)
            self._history_index = self.popsize
        self.init_test_set()

    def compute_alpha_beta(self, X):
        return lstsq(X, self._f)[0]

    def crossover(self, father1, father2, *args):
        self._nparents = 2
        f1 = self._xo_father1
        f2 = self._xo_father2
        X = self._pop_eval[np.array([f1, f2])].T
        alpha_beta = self.compute_alpha_beta(X)
        y = X.dot(alpha_beta)
        self._ind_generated_c = alpha_beta
        if self._test_set is not None:
            ev = self._test_set_eval
            _ = ev[f1] * alpha_beta[0] + ev[f2] * alpha_beta[1]
            self._test_set_tmp = _
        return y

    def mutation(self, father1):
        self._nparents = 1
        f1 = self._xo_father1
        p1 = np.random.randint(self.popsize)
        p2 = np.random.randint(self.popsize)
        while p1 == p2:
            p2 = np.random.randint(self.popsize)
        self._xo_father2 = [p1, p2]
        X = np.vstack((self._pop_eval[f1],
                       self._pop_eval_mut[p1] - self._pop_eval_mut[p2])).T
        alpha_beta = self.compute_alpha_beta(X)
        y = X.dot(alpha_beta)
        self._ind_generated_c = alpha_beta
        if self._test_set is not None:
            ev = self._test_set_eval
            mut = self._test_set_eval_mut
            _ = ev[f1] * alpha_beta[0] + (mut[p1] - mut[p2]) * alpha_beta[1]
            self._test_set_tmp = _
        return y

    def eval(self, ind=None, **kwargs):
        if ind is None:
            return super(PrGP, self).eval(ind=ind, **kwargs)
        if not isinstance(ind, types.IntType):
            cdn = "The individual must be part of the population"
            raise NotImplementedError(cdn)
        if ind < self.popsize and self._pop_hist[ind] == ind:
            return super(PrGP, self).eval(ind=ind, **kwargs)
        eval = lstsqEval(None, self._history_ind, self._history_coef)
        if ind < self.popsize:
            ind = self._pop_hist[ind]
        inds = eval.inds_to_eval(ind)
        init = np.zeros((self.popsize, self._x.shape[0]))
        for i in inds:
            if i < self.popsize:
                init[i] = super(PrGP, self).eval(ind=i, **kwargs)
        eval.init = init
        pr = eval.eval(ind, inds=inds)
        return pr

    @staticmethod
    def sigmoid(x):
        # return x.copy()
        return 1 / (1 + np.exp(-x))


class lstsqEval(object):
    def __init__(self, init, hist_ind, hist_coef):
        self._init = init
        self._hist_ind = hist_ind
        self._hist_coef = hist_coef
        self._pos = 0

    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, init):
        self._init = init

    def inds_to_eval(self, ind):
        h = {ind: 1}
        lst = self._hist_ind[ind].tolist()
        while len(lst):
            k = lst.pop()
            if k in h:
                continue
            lst += self._hist_ind[k].tolist()
            h[k] = 1
        h = h.keys()
        h.sort()
        if h[0] == -1:
            return h[1:]
        return h

    def eval(self, ind, inds=None):
        def get_ev(_ind):
            if _ind < popsize:
                return init[_ind]
            return st[_m[_ind]]

        if inds is None:
            inds = self.inds_to_eval(ind)
        popsize = self._init.shape[0]
        coef = self._hist_coef
        init = self._init
        hist_ind = self._hist_ind
        st = []
        _m = {}
        c = 0
        for index in inds:
            if index < popsize:
                continue
            hist = hist_ind[index]
            c1, c2 = coef[index]
            f1 = get_ev(hist[0])
            if hist[-1] == -1:
                f2 = get_ev(hist[1])
            else:
                sigmoid = PrGP.sigmoid
                f2 = sigmoid(init[hist[1]]) - sigmoid(init[hist[2]])
            st.append(c1 * f1 + c2 * f2)
            _m[index] = c
            c += 1
        return get_ev(ind)


class GSGP(PrGP):
    def __init__(self, ms=1.0, **kwargs):
        super(GSGP, self).__init__(**kwargs)
        self._ms = ms

    def compute_alpha_beta(self, X):
        if self._nparents == 2:
            beta = np.random.rand()
            return np.array([beta, 1-beta])
        return np.array([1, self._ms])

    def improve_fit(self, son):
        fit = - self.distance(self._f, son)
        return fit, True
