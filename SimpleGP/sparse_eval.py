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


import numpy as np
from scipy.sparse import issparse
from scipy.sparse import csr_matrix as Smatrix
from scipy.sparse import isspmatrix_csr as isSmatrix


class SparseEval(object):
    def __init__(self, nop):
        self._nop = nop
        self._pos = 0
        self._x = None
        self._nvar = None
        self._nfunc = nop.shape[0]
        self._output = None

    @property
    def nfunc(self):
        return self._nfunc

    @property
    def nvar(self):
        return self._nvar

    @property
    def X(self):
        return self._x

    @X.setter
    def X(self, x):
        if not issparse(x):
            x = Smatrix(x)
        elif not isSmatrix(x):
            x = x.tocsc()
        self._x = x
        self._nvar = x.shape[1]

    def isfunc(self, a):
        return a < self.nfunc

    def isvar(self, a):
        nfunc = self.nfunc
        nvar = self.nvar
        return (a >= nfunc) and (a < nfunc+nvar)

    def isconstant(self, a):
        nfunc = self.nfunc
        nvar = self.nvar
        return a >= nfunc+nvar

    def eval(self, ind, constants):
        self._pos = 0
        self._ind = ind
        self._constants = constants
        hy = np.asarray(self._eval().todense())
        if hy.shape[1] == 1:
            return hy.flatten()
        return hy

    @staticmethod
    def one_arg_func(_x, func):
        x = _x.copy()
        x.data = func(x.data)
        return x

    @staticmethod
    def sigmoid(_x):
        x = -_x
        x.data = np.exp(x.data)
        one = Smatrix(np.ones(_x.shape[0])[:, np.newaxis])
        return one / (one + x)

    @staticmethod
    def ln(_x):
        x = SparseEval.one_arg_func(_x, np.fabs)
        x.data = np.log(x.data)
        return x

    @staticmethod
    def max_func(x, y):
        s = -100 * (x - y)
        s.data = np.exp(s.data)
        one = Smatrix(np.ones(x.shape[0])[:, np.newaxis])
        s = one / (one + s)
        return s.multiply(x - y) + y

    @staticmethod
    def min_func(x, y):
        s = -100 * (x - y)
        s.data = np.exp(s.data)
        one = Smatrix(np.ones(x.shape[0])[:, np.newaxis])
        s = one / (one + s)
        return s.multiply(y - x) + x

    @staticmethod
    def if_func(x, y, z):
        s = -100 * x
        s.data = np.exp(s.data)
        one = Smatrix(np.ones(x.shape[0])[:, np.newaxis])
        s = one / (one + s)
        return s.multiply(y - z) + z

    @staticmethod
    def argmax(*a):
        nargs = len(a)
        beta = 2.
        args = map(lambda x: None, range(nargs))
        for j, x in enumerate(a):
            x = beta * x
            x.data = np.exp(x.data)
            args[j] = x
        sum = args[0]
        for i in range(1, nargs):
            sum = sum + args[i]
        res = 0
        for i in range(1, nargs):
            res = res + (args[i] * i) / sum
        return res

    def output(self, *args):
        self._output = args
        return args[0]

    def _eval(self):
        pos = self._pos
        self._pos += 1
        node = self._ind[pos]
        if self.isfunc(node):
            args = map(lambda x: self._eval(), range(self._nop[node]))
            F = self.one_arg_func
            func = [np.add, np.subtract,
                    lambda x, y: x.multiply(y),
                    np.divide,
                    lambda x: F(x, np.fabs), lambda x: F(x, np.exp),
                    lambda x: F(x, np.sqrt), lambda x: F(x, np.sin),
                    lambda x: F(x, np.cos), self.sigmoid,
                    self.if_func, self.max_func, self.min_func,
                    self.ln, lambda x: x.multiply(x), self.output, self.argmax]
            return func[node](*args)
        elif self.isvar(node):
            return self.X[:, node - self.nfunc]
        else:
            v = self._constants[node - self.nfunc - self.nvar]
            return Smatrix(np.ones(self.X.shape[0])[:, np.newaxis]) * v
