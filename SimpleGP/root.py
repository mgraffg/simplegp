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


from SimpleGP import GPS
import numpy as np


class RootGP(GPS):
    def __init__(self, nrandom=0, ntrees=-1,
                 count_no_improvements=False,
                 greedy=True,
                 func=['+', '*', '/', 'abs',
                       'exp', 'sqrt', 'sin', 'cos',
                       'sigmoid', 'ln', 'sq', 'if'],
                 **kwargs):
        super(RootGP, self).__init__(nrandom=nrandom, func=func, **kwargs)
        self._ind_generated_f = None
        self._pop_eval = None
        self._test_set_eval = None
        self._count_no_improvements = count_no_improvements
        self._greedy = greedy
        self._hist = []
        if ntrees > 0:
            self._nop[self._output_pos] = ntrees
            self._func_allow = np.concatenate((self._func_allow,
                                               np.array([15])))

    def predict_test_set(self, ind):
        if self._test_set_eval is not None:
            return self._test_set_eval[ind]
        raise NotImplementedError("This function is not implemented yet")

    def random_leaf(self):
        c = None
        while c is None:
            a = super(RootGP, self).random_leaf()
            c = self.compute_coef([self._x[a - self.nfunc]])
        if self._test_set is not None:
            t = self._test_set[a - self.nfunc] * c[0]
        else:
            t = None
        return a, c, self._x[a - self.nfunc] * c[0], t

    def create_population(self):
        if self._p is not None:
            return False
        self._p_constants = np.empty(self._popsize, dtype=np.object)
        self._p = np.empty(self._popsize, dtype=np.object)
        self._pop_eval = np.empty(self._popsize, dtype=np.object)
        self._test_set_eval = np.empty(self._popsize, dtype=np.object)
        self._fitness = np.zeros(self._popsize)
        self._fitness[:] = -np.inf
        for i in range(self.popsize):
            f = -np.inf
            while f == -np.inf:
                a, c, e, t = self.random_leaf()
                f = - self.distance(self._f, e)
            self._p[i] = np.array([i])
            self._p_constants[i] = c
            self._fitness[i] = f
            self._pop_eval[i] = e
            self._test_set_eval[i] = t
            self.new_best(i)
            self._hist.append([a, c])
        return True

    def select_parents(self, nparents):
        lst = []
        while len(lst) < nparents:
            p = self.tournament()
            if p in lst:
                continue
            lst.append(p)
        return lst

    def cumsum(self, r):
        a = r[0]
        for x in r[1:]:
            a = a + x
        return a

    def genetic_operators_linear_comb(self, args):
        X = map(lambda x: self._pop_eval[x], args)
        coef = self.compute_coef(X)
        if coef is None:
            return None
        r = map(lambda (x, c): x * c, zip(X, coef))
        if self._test_set is not None:
            Xt = map(lambda x: self._test_set_eval[x], args)
            rt = map(lambda (x, c): x * c, zip(Xt, coef))
            rt = self.cumsum(rt)
            if not rt.isfinite():
                return None
        else:
            rt = None
        r = self.cumsum(r)
        return coef, r, rt

    def genetic_operators_inner(self):
        func = self.random_func()
        args = self.select_parents(self._nop[func])
        if func == 15 or func == 0:  # output o sum
            res = self.genetic_operators_linear_comb(args)
            if res is None:
                return None
        else:
            ind = np.array([func] + range(self.nfunc,
                                          self.nfunc + len(args)))
            self._eval.X(map(lambda x: self._pop_eval[x], args))
            e = self._eval.eval(ind, self._constants2, 0)
            coef = self.compute_coef([e])
            if coef is None:
                return None
            e = e * coef[0]
            et = None
            if self._test_set is not None:
                self._eval.X(map(lambda x: self._test_set_eval[x], args))
                et = self._eval.eval(ind, self._constants2, 0)
                et = et * coef[0]
                if not et.isfinite():
                    return None
            res = (coef, e, et)
        return func, args, res[0], res[1], res[2]

    def kill_ind(self, kill, son):
        super(RootGP, self).kill_ind(kill, son)
        yh, yht = self._ind_generated_f
        self._test_set_eval[kill] = yht
        self._pop_eval[kill] = yh
        self._fitness[kill] = - self.distance(self._f, yh)
        self.new_best(kill)

    def genetic_operators(self):
        while True:
            r = self.genetic_operators_inner()
            if r is None:
                continue
            func, args, coef, yh, yht = r
            f = - self.distance(self._f, yh)
            if self._greedy and f <= max(map(lambda x:
                                             self._fitness[x], args)):
                if self._count_no_improvements:
                    self.gens_ind += 1
                    if self.gens_ind < self.popsize * self.generations:
                        continue
                else:
                    continue
            self._ind_generated_f = yh, yht
            self._ind_generated_c = np.array([0.0])
            self._hist.append([map(lambda x: self._p[x][0],
                                   args), func, coef])
            return np.array([len(self._hist) - 1])

    def compute_coef(self, r):
        A = np.empty((len(r), len(r)))
        b = np.array(map(lambda f: (f * self._f).sum(), r))
        for i in range(len(r)):
            for j in range(i, len(r)):
                A[i, j] = (r[i] * r[j]).sum()
                A[j, i] = A[i, j]
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        return coef

    def use_hist(self, i, lst=None):
        if lst is None:
            lst = map(lambda x: False, self._hist)
        self._use_hist(i, lst)
        return lst

    def _use_hist(self, i, lst):
        if lst[i]:
            return
        lst[i] = True
        a = self._hist[i]
        if len(a) > 2:
            map(lambda x: self._use_hist(x, lst), a[0])
