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
#cython: profile=True
#cython: nonecheck=False


from array import array as p_array
# cimport numpy as npc
cimport cython
import numpy as np
import types
from cpython.mem cimport PyMem_Malloc, PyMem_Free
# from cpython.ref cimport Py_INCREF

@cython.freelist(512)
cdef class SparseArray:
    def __cinit__(self):
        self._size = 0
        self._nele = 0
        self._usemem = 0

    cpdef int size(self):
        return self._size

    cpdef set_size(self, int s):
        self._size = s

    cpdef int nele(self):
        return self._nele

    cpdef set_nele(self, int s):
        self._nele = s

    cpdef init_ptr(self, array.array[int] index,
                   array.array[double] data):
        self._indexC = index.data.as_ints
        self._dataC = data.data.as_doubles

    cpdef init(self, int nele):
        if nele == 0:
            self._usemem = 1
            return
        if self._usemem == 0:
            # print "Poniendo memoria", self
            self._usemem = 1
            self._dataC = <double*> PyMem_Malloc(nele * sizeof(double))
            if not self._dataC:
                raise MemoryError("dataC")
            self._indexC = <int *> PyMem_Malloc(nele * sizeof(int))
            if not self._indexC:
                raise MemoryError("indexC")
            self.set_nele(nele)
        else:
            print "Error init", self

    def __dealloc__(self):
        if self._usemem and self.nele() == 0:
            return
        if self._usemem:
            # print "Borrando", self
            self._usemem = 0
            PyMem_Free(self._dataC)
            PyMem_Free(self._indexC)
            # self._dataC = null
            # self._indexC = null
        else:
            print "Error dealloc", self
            
    cpdef int nunion(self, SparseArray other):
        cdef int a=0, b=0, c=0
        cdef int anele = self.nele(), bnele=other.nele()
        while (a < anele) and (b < bnele):
            if self._indexC[a] == other._indexC[b]:
                a += 1
                b += 1
            elif self._indexC[a] < other._indexC[b]:
                a += 1
            else:
                b += 1
            c += 1
        if a == anele:
            return c + bnele - b
        else:
            return c + anele - a

    cpdef int nintersection(self, SparseArray other):
        cdef int a=0, b=0, c=0
        cdef int anele = self.nele(), bnele=other.nele()    
        while (a < anele) and (b < bnele):
            if self._indexC[a] == other._indexC[b]:
                a += 1
                b += 1
                c += 1
            elif self._indexC[a] < other._indexC[b]:
                a += 1
            else:
                b += 1                
        return c

    def __add__(self, other):
        res = self.empty(self.nunion(other), self.size())
        self.add(other, res)
        return res

    cpdef int add(self, SparseArray other, SparseArray res):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        cdef SparseArray last
        for c in range(rnele):
            if a >= anele:
                index = other._indexC[b]
                r = other._dataC[b]
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    r = self._dataC[a] + other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    a += 1
                else:
                    r = other._dataC[b]
                    index = other._indexC[b]
                    b += 1
            res._dataC[c] = r
            res._indexC[c] = index
        return 0

    def __sub__(self, other):
        res = self.empty(self.nunion(other), self.size())
        self.sub(other, res)
        return res
        
    cpdef int sub(self, SparseArray other, SparseArray res):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        cdef SparseArray last
        for c in range(rnele):
            if a >= anele:
                index = other._indexC[b]
                r = - other._dataC[b]
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    r = self._dataC[a] - other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    a += 1
                else:
                    r = - other._dataC[b]
                    index = other._indexC[b]
                    b += 1
            res._dataC[c] = r
            res._indexC[c] = index
        return 0

    def __mul__(self, other):
        res = self.empty(self.nintersection(other), self.size())
        self.mul(other, res)
        return res

    cpdef int mul(self, SparseArray other, SparseArray res):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        while c < rnele:
            if a >= anele:
                b += 1
            elif b >= bnele:
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    res._dataC[c] = self._dataC[a] * other._dataC[b]
                    res._indexC[c] = index
                    a += 1; b += 1; c += 1
                elif index < other._indexC[b]:
                    a += 1
                else:
                    b += 1
        return 0

    def __div__(self, other):
        cdef int s = self.nintersection(other)
        res = self.empty(s, self.size())
        self.div(other, res)
        return res

    @cython.cdivision(True)
    cpdef int div(self, SparseArray other, SparseArray res):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        while c < rnele:
            if a >= anele:
                b += 1
            elif b >= bnele:
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    res._dataC[c] = self._dataC[a] / other._dataC[b]
                    res._indexC[c] = index
                    a += 1; b += 1; c += 1
                elif index < other._indexC[b]:
                    a += 1
                else:
                    b += 1
        return 0

    cpdef double sum(self):
        cdef double res=0, *data = self._dataC
        cdef int i
        for i in xrange(self._nele):
            res += data[i]
        return res

    cpdef SparseArray fabs(self):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        for i in xrange(self.nele()):
            res._dataC[i] = math.fabs(self._dataC[i])
            res._indexC[i] = self._indexC[i]
        return res
            
    def tonparray(self):
        import numpy as np
        cdef npc.ndarray[double, ndim=1] res = np.zeros(self.size())
        cdef double * resC = <double *>res.data
        cdef int i, ele
        for i in range(self.nele()):
            ele = self._indexC[i]
            resC[ele] = self._dataC[i]
        return res

    def tolist(self):
        cdef int i, ele
        lst = map(lambda x: 0, xrange(self.size()))
        for i in range(self.nele()):
            ele = self._indexC[i]
            lst[ele] = self._dataC[i]
        return lst

    def set_data_index(self, data, index):
        cdef int c=0
        for d, i in zip(data, index):
            self._dataC[c] = d
            self._indexC[c] = i
            c += 1

    def print_data(self):
        cdef int i
        for i in range(self.nele()):
            print self._dataC[i]

    @classmethod
    def fromlist(cls, iter):
        self = cls()
        data = []
        index = []
        for k, v in enumerate(iter):
            if v == 0:
                continue
            data.append(v)
            index.append(k)
        self.init(len(data))
        self.set_data_index(data, index)    
        self.set_size(k + 1)
        return self

    cpdef SparseArray empty(self, int nele, int size=-1):
        cdef SparseArray res = SparseArray()
        res.init(nele)
        if size == -1:
            res.set_size(nele)
        else:
            res.set_size(size)
        return res

    cpdef SparseArray constant(self, double v, int size=-1):
        cdef int i
        cdef SparseArray res = SparseArray()
        if v == 0:
            res.init(0)
        else:
            res.init(size)
        if size == -1:
            size = self.size()
        res.set_size(size)
        for i in range(res.nele()):
            res._dataC[i] = v
            res._indexC[i] = i
        return res

cdef class SEval:
    def __cinit__(self, npc.ndarray[long, ndim=1] nop):
        self._nop = <long *> nop.data
        self._nfunc = nop.shape[0]
        self._pos = 0
        # self._st = []
        
    cdef int isfunc(self, int a):
        return a < self._nfunc

    cdef int isvar(self, int a):
        cdef int nfunc = self._nfunc
        cdef int nvar = self._nvar
        return (a >= nfunc) and (a < nfunc+nvar)

    cdef int isconstant(self, int a):
        cdef int nfunc = self._nfunc
        cdef int nvar = self._nvar
        return a >= nfunc+nvar
        
    cpdef set_size(self, int s):
        self._size = s
        
    cpdef set_nvar(self, int nvar):
        self._nvar = nvar
    
    def X(self, x):
        if isinstance(x, types.ListType):
            self._x = x
        else:
            self._x = map(lambda i: SparseArray.fromlist(x[:, i]),
                          range(x.shape[1]))
        self.set_nvar(len(self._x))
        self.set_size(self._x[0].size())

    cdef void init_st(self, int n):
        pass
        # cdef int i, size=self._size
        # cdef SparseArray tmp
        # cdef long *ind = self._ind
        # self._st = []
        # for i in range(n):
        #     if self.isfunc(ind[i]):
        #         tmp = SparseArray()
        #         tmp.set_size(size)
        #         print "init_st", tmp
        #         self._st.append(tmp)
            
    cpdef eval(self, npc.ndarray[long, ndim=1] ind,
               npc.ndarray[double, ndim=1] constants, bint to_np_array=1):
        self._pos = 0
        self._st_pos = 0
        self._ind = <long *> ind.data
        self._constants = <double *> constants.data
        self.init_st(ind.shape[0])
        if to_np_array:
            return self._eval().tonparray()
        else:
            return self._eval()

    cdef SparseArray two_args(self, int func, SparseArray first,
                              SparseArray second):
        cdef SparseArray res
        cdef int nele    
        if func == 0:  # add
            res = first.empty(first.nunion(second), self._size)
            first.add(second, res)
            return res
        elif func == 1:  # subtract
            res = first.empty(first.nunion(second), self._size)
            first.sub(second, res)
            return res
        elif func == 2:  # multiply
            nele = first.nintersection(second)
            res = first.empty(nele, self._size)
            first.mul(second, res)
            return res
        else:  # divide
            res = first.empty(first.nintersection(second), self._size)
            first.div(second, res)
            return res
            
    cdef SparseArray _eval(self):
        cdef SparseArray res, first, second
        cdef int pos = self._pos
        cdef double v
        self._pos += 1
        cdef int node = self._ind[pos]
        if self.isfunc(node):
            if self._nop[node] == 2:
                first = self._eval()
                second = self._eval()
                res = self.two_args(node, first, second)
        elif self.isvar(node):
            res = self._x[node - self._nfunc]
        else:
            v = self._constants[node - self._nfunc - self._nvar]
            res = self._x[0].constant(v, self._size)
        return res
            
        
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
        
