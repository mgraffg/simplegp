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
from SimpleGP.sparse_array import SparseArray


class SparseEval(object):
    def __init__(self, nop):
        self._nop = nop
        self._pos = 0
        self._x = None
        self._nvar = None
        self._nfunc = nop.shape[0]
        self._output = None
        self._size = None

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
        self._x = map(lambda i: SparseArray.fromlist(x[:, i]),
                      range(x.shape[1]))
        self._nvar = x.shape[1]
        self._size = self._x[0].size()

    @property
    def size(self):
        return self._size

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
        hy = self._eval().tonparray()
        return hy

    @staticmethod
    def one_arg_func(_x, func):
        return func(_x)

    @staticmethod
    def sigmoid(_x):
        return 1. / (1. + np.exp(-_x))

    @staticmethod
    def ln(_x):
        return np.log(np.fabs(_x))

    @staticmethod
    def max_func(x, y):
        s = -100 * (x - y)
        s = np.exp(s)
        one = 1.
        s = one / (one + s)
        return s * (x - y) + y

    @staticmethod
    def min_func(x, y):
        s = -100 * (x - y)
        s = np.exp(s)
        one = 1.
        s = one / (one + s)
        return s * (y - x) + x

    @staticmethod
    def if_func(x, y, z):
        s = -100 * x
        s = np.exp(s)
        one = 1.
        s = one / (one + s)
        return s * (y - z) + z

    @staticmethod
    def argmax(*a):
        nargs = len(a)
        beta = 2.
        args = map(lambda x: None, range(nargs))
        for j, x in enumerate(a):
            x = beta * x
            x = np.exp(x)
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
            func = [lambda x, y: x + y,  # np.add,
                    lambda x, y: x - y,  # np.subtract,
                    lambda x, y: x * y,  # np.multiply,
                    lambda x, y: x / y,  # np.divide,
                    lambda x: F(x, np.fabs), lambda x: F(x, np.exp),
                    lambda x: F(x, np.sqrt), lambda x: F(x, np.sin),
                    lambda x: F(x, np.cos), self.sigmoid,
                    self.if_func, self.max_func, self.min_func,
                    self.ln, np.square, self.output, self.argmax]
            return func[node](*args)
        elif self.isvar(node):
            return self.X[node - self.nfunc]
        else:
            v = self._constants[node - self.nfunc - self.nvar]
            return self._x[0].constant(v, self.size)
