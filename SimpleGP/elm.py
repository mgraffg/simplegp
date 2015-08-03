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

from .forest import GPForest, GPForestPDE
import numpy as np


class ELM(GPForest):
    def __init__(self, ntrees=2, nrandom=0, **kwargs):
        super(ELM, self).__init__(ntrees=ntrees, nrandom=nrandom,
                                  **kwargs)
        self._elm_constants = None

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
        super(ELM, self).create_population()

    def eval_ind(self, *args, **kwargs):
        r = super(ELM, self).eval_ind(*args, **kwargs)
        if self._computing_fitness is None:
            cdn = "Use eval with the number of individual, instead"
            NotImplementedError(cdn)
        k = self._computing_fitness
        if self._fitness[k] > -np.inf:
            coef = self._elm_constants[k]
        else:
            if np.all(np.isfinite(r)):
                coef = np.linalg.lstsq(r, self._f)[0]
            else:
                coef = np.ones(r.shape[1], dtype=self._dtype)
        self._elm_constants[k] = coef
        return np.dot(r, coef)


class ELMPDE(GPForestPDE, ELM):
    def compute_error_pr(self, ind, pos=0, constants=None, epoch=0):
        if epoch == 0:
            g = self._p_st[self._computing_fitness][self._output].T
            coef = self._elm_constants[self._computing_fitness]
            g = np.dot(g, coef)
        else:
            if ind is None:
                g = self.eval(self._computing_fitness)
                k = self._computing_fitness
                coef = self._elm_constants[k]
            else:
                g = self.eval_ind(ind, pos=pos, constants=constants)
        # e = - 2 * ( self._f - g)
        e = 2 * (g - self._f)
        e = np.repeat(coef[:, np.newaxis], e.shape[0], axis=1) * e
        return e.T, g
