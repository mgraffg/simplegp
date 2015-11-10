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
from SimpleGP.sparse_array import SparseEval


class Infeasible(Exception):
    pass


class Individual(object):
    def __init__(self, X, nop):
        self._X = X
        self._nop = nop
        self._eval = SparseEval(self._nop)
        self._nfunc = nop.shape[0]
        self._func_allow = np.where(nop > -1)[0]
        self._nvar = len(X)
        self._constants2 = np.zeros(3)
        self._grow = 0
        self._nop_map = {}
        for k, v in enumerate(nop):
            if v == -1:
                continue
            try:
                self._nop_map[v].append(k)
            except KeyError:
                self._nop_map[v] = [k]

    @property
    def nfunc(self):
        return self._nfunc

    @property
    def nvar(self):
        return self._nvar

    def random_func(self, nop=None):
        if nop is None:
            k = np.random.randint(self._func_allow.shape[0])
            return self._func_allow[k]
        f = self._nop_map[nop]
        k = np.random.randint(len(f))
        return f[k]

    def random_leaf(self):
        l = np.random.randint(self.nvar)
        return l + self.nfunc

    def random_depth(self):
        return np.random.randint(1, 8)

    def grow_or_full(self):
        return np.random.randint(2)

    def random_ind(self, depth=None):
        if depth is None:
            depth = self.random_depth()
        self._grow = False if self.grow_or_full() else True
        self._ind = []
        r = self.random_ind_inner(depth)
        return np.array(self._ind), r

    def random_ind_inner(self, depth):
        if depth == 0 or (self._grow and self.grow_or_full()):
            for _ in range(3):
                a = self.random_leaf()
                r = self._X[a - self.nfunc]
                if r.isfinite():
                    self._ind.append(a)
                    return r
        else:
            pos = len(self._ind)
            self._ind.append(None)
            f = self.random_func()
            args = map(lambda x: self.random_ind_inner(depth-1),
                       range(self._nop[f]))
            self._eval.X(args)
            for _ in range(3):
                ind = np.array([f] + range(self.nfunc,
                                           self.nfunc + len(args)),
                               dtype=np.int)
                r = self._eval.eval(ind, self._constants2, to_np_array=False,
                                    st=None)
                if r.isfinite() and r.nele() > 0:
                    self._ind[pos] = f
                    return r
                else:
                    f = self.random_func(nop=self._nop[f])
        raise Infeasible("Individual " + str(self._ind))
