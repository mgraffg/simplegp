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
from SimpleGP.simplegp import GPS
from SimpleGP.forest import GPForest
import numpy as np


class EGPS(GPS, GPForest):
    def __init__(self, ntrees=5, nrandom=0, **kwargs):
        super(EGPS, self).__init__(ntrees=ntrees, nrandom=nrandom,
                                   **kwargs)
        self._elm_constants = None

    def train(self, *args, **kwargs):
        super(EGPS, self).train(*args, **kwargs)
        self._nop[self._output_pos] = self._ntrees
        self._eval_A_st = map(lambda x: None, range(self.popsize))
        return self

    def early_stopping_save(self, k, fit_k=None):
        """
        Storing the best so far on the validation set.
        This funtion is called from early_stopping
        """
        assert fit_k
        self._early_stopping = [fit_k,
                                self.population[k].copy(),
                                self._p_constants[k].copy(),
                                self._elm_constants[k].copy(),
                                self._pr_test_set.copy()]

    def create_population(self):
        if self._elm_constants is None or\
           self._elm_constants.shape[0] != self._popsize:
            self._elm_constants = np.empty(self._popsize,
                                           dtype=np.object)
        return super(EGPS, self).create_population()

    def kill_ind(self, kill, son):
        super(EGPS, self).kill_ind(kill, son)
        self._eval_A_st[kill] = self._ind_eval_A_st

    def crossover(self, father1, father2, **kwargs):
        res = super(EGPS, self).crossover(father1, father2, **kwargs)
        if not self._use_st:
            return res
        if self._ind_eval_st is not None:
            A = self._eval_A_st[self._xo_father1]
            self._ind_eval_A_st = A
        return res

    def eval_ind(self, ind, **kwargs):
        if self._computing_fitness is None:
            cdn = "Use eval with the number of individual, instead"
            NotImplementedError(cdn)
        k = self._computing_fitness
        if self._eval_st[k] is not None and self._eval_A_st[k] is not None:
            st = self._eval_st[k]
            index = map(lambda x: self._tree.get_pos_arg(ind, 0, x),
                        range(self._ntrees))
            trees = map(lambda x: st[x], index)
        else:
            trees = map(lambda x: None, range(self._ntrees))
        super(EGPS, self).eval_ind(ind, **kwargs)
        r = filter(lambda x: x.isfinite(), self._eval.get_output())
        if len(r) == 0:
            return self._eval.get_output()[0]
        if self._fitness[k] > -np.inf:
            coef = self._elm_constants[k]
        else:
            A = np.empty((len(r), len(r)))
            Ap = self._eval_A_st[k]
            if Ap is not None and A.shape[0] != Ap.shape[0]:
                Ap = None
            b = np.array(map(lambda f: (f * self._f).sum(), r))
            for i in range(len(r)):
                for j in range(i, len(r)):
                    if trees[i] is None or trees[j] is None\
                       or Ap is None:
                        A[i, j] = (r[i] * r[j]).sum()
                    else:
                        A[i, j] = Ap[i, j]
                    A[j, i] = A[i, j]
            try:
                coef = np.linalg.solve(A, b)
                self._eval_A_st[k] = A
            except np.linalg.LinAlgError:
                coef = np.ones(len(r))
        res = r[0] * coef[0]
        for i in range(1, len(r)):
            res = res + (r[i] * coef[i])
        self._elm_constants[k] = coef
        return res
