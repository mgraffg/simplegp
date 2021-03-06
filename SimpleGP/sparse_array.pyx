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
cdef extern from "numpy/npy_math.h":
    bint npy_isinf(double)
    bint npy_isnan(double)
    long double INFINITY "NPY_INFINITY"
    long double PI "NPY_PI"


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
        if self.size() == anele or self.size() == bnele:
            return self.size()
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
        cdef int a=0, b=0, c=0, *indexC = self._indexC, *oindexC = other._indexC
        cdef int anele = self.nele(), bnele=other.nele()
        if self.size() == anele or self.size() == bnele:
            if anele < bnele:
                return anele
            return bnele
        while (a < anele) and (b < bnele):
            if indexC[a] == oindexC[b]:
                a += 1
                b += 1
                c += 1
            elif indexC[a] < oindexC[b]:
                a += 1
            else:
                b += 1                
        return c

    def __add__(self, other):
        if isinstance(other, SparseArray):
            return self.add(other)
        return self.add2(other)

    cpdef SparseArray add(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nunion(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] + other._dataC[c]
                res._indexC[c] = c
            return res
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
        return res

    cpdef SparseArray add2(self, double other):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i
        for i in range(self.size()):
            res._indexC[i] = i
            res._dataC[i] = other
        for i in range(self.nele()):
            res._dataC[self._indexC[i]] = self._dataC[i] + other
        return res

    def __sub__(self, other):
        if isinstance(other, SparseArray):
            return self.sub(other)
        return self.sub2(other)
        
    cpdef SparseArray sub(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nunion(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] - other._dataC[c]
                res._indexC[c] = c
            return res        
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
        return res

    cpdef SparseArray sub2(self, double other):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i
        for i in range(self.size()):
            res._indexC[i] = i
            res._dataC[i] = - other
        for i in range(self.nele()):
            res._dataC[self._indexC[i]] = self._dataC[i] - other
        return res
        
    def __mul__(self, other):
        if isinstance(other, SparseArray):
            return self.mul(other)
        return self.mul2(other)

    cpdef SparseArray mul(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nintersection(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] * other._dataC[c]
                res._indexC[c] = c
            return res        
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
        return res

    cpdef SparseArray mul2(self, double other):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i] * other
        return res
                    
    def __div__(self, other):
        if isinstance(other, SparseArray):
            return self.div(other)
        return self.div2(other)

    @cython.cdivision(True)
    cpdef SparseArray div(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nintersection(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] / other._dataC[c]
                res._indexC[c] = c
            return res        
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
        return res

    @cython.cdivision(True)    
    cpdef SparseArray div2(self, double other):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i] / other
        return res
        
    cpdef double sum(self):
        cdef double res=0, *data = self._dataC
        cdef int i
        for i in xrange(self._nele):
            res += data[i]
        return res

    cpdef double mean(self):
        cdef double res=0
        res = self.sum()    
        return res / self.size()

    @cython.boundscheck(False)
    @cython.nonecheck(False)    
    def var_per_cl(self, list X, list mu,
                   array.array[double] kfreq):
        cdef int i, k=0, ncl = len(kfreq), j, xnele, ynele=self.nele()
        cdef SparseArray x
        cdef list m=[]
        cdef double epsilon = 1e-9
        cdef array.array[double] var, mu_x
        for x, mu_x in zip(X, mu):
            var = array.array('d', map(lambda x: 0, range(ncl)))
            k = 0
            i = 0
            xnele = x.nele()
            while i < xnele:
                if x._indexC[i] == self._indexC[k]:
                    j = int(self._dataC[k])
                    var[j] += math.pow(x._dataC[i] - mu_x[j], 2)
                    k += 1
                    i += 1
                elif self._indexC[k] < x._indexC[i]:
                    if k < ynele:
                        k += 1
                    else:
                        var[0] += math.pow(x._dataC[i] - mu_x[0], 2)
                        i += 1
                else:
                    var[0] += math.pow(x._dataC[i] - mu_x[0], 2)
                    i += 1
            for i in range(ncl):
                var[i] = var[i] / kfreq[i] + epsilon
            m.append(var)
        return m

    @cython.boundscheck(False)
    @cython.nonecheck(False)    
    def mean_per_cl(self, list X, array.array[double] kfreq):
        cdef int i, k=0, ncl = len(kfreq), j, xnele, ynele=self.nele()
        cdef SparseArray x
        cdef list m=[]
        cdef array.array[double] mu
        for x in X:
            mu = array.array('d', map(lambda x: 0, range(ncl)))
            k = 0
            i = 0
            xnele = x.nele()
            while i < xnele:
                if x._indexC[i] == self._indexC[k]:
                    j = int(self._dataC[k])
                    mu[j] += x._dataC[i]
                    k += 1
                    i += 1
                elif self._indexC[k] < x._indexC[i]:
                    if k < ynele:
                        k += 1
                    else:
                        mu[0] += x._dataC[i]
                        i += 1
                else:
                    mu[0] += x._dataC[i]
                    i += 1
            for i in range(ncl):
                mu[i] = mu[i] / kfreq[i]
            m.append(mu)
        return m

    @cython.boundscheck(False)
    @cython.nonecheck(False)        
    def class_freq(self, int ncl):
        cdef int i
        cdef list f
        cdef array.array[double] mu = array.array('d', map(lambda x: 0, range(ncl)))
        for i in range(self.nele()):
            mu[int(self._dataC[i])] += 1.0
        mu[0] = self.size() - sum(mu[1:])
        return mu    
        
    cpdef double std(self):
        cdef SparseArray res = self.sub2(self.mean())
        res = res.sq()
        return math.sqrt(res.sum() / res.size())

    cpdef SparseArray fabs(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        for i in xrange(self.nele()):
            res._dataC[i] = math.fabs(self._dataC[i])
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray exp(self):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i, j=0, nele=self.nele()
        for i in xrange(self.size()):
            if j < nele and self._indexC[j] == i:
                res._dataC[i] = math.exp(self._dataC[j])
                j += 1
            else:
                res._dataC[i] = 1
            res._indexC[i] = i
        return res
        
    cpdef SparseArray sin(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        for i in xrange(self.nele()):
            res._dataC[i] = math.sin(self._dataC[i])
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray cos(self):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i, j=0, nele=self.nele()
        for i in xrange(self.size()):
            if j < nele and self._indexC[j] == i:
                res._dataC[i] = math.cos(self._dataC[j])
                j += 1
            else:
                res._dataC[i] = 1
            res._indexC[i] = i
        return res
        
    cpdef SparseArray ln(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            r = self._dataC[i]
            res._dataC[i] = math.log(math.fabs(r))
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray sq(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            r = self._dataC[i]
            res._dataC[i] = r * r
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray sqrt(self):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            res._dataC[i] = math.sqrt(self._dataC[i])
            res._indexC[i] = self._indexC[i]
        return res

    @cython.cdivision(True)
    cpdef SparseArray sigmoid(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            res._dataC[i] = 1 / (1 + math.exp((-self._dataC[i] + 1) * 30))
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray if_func(self, SparseArray y, SparseArray z):
        cdef SparseArray s = self.sigmoid()
        cdef SparseArray sy, sz
        cdef SparseArray r
        cdef SparseArray res
        sy = s.mul(y)
        sz = s.mul(z)
        r = sy.sub(sz)
        res = r.add(z)
        return res
            
    cpdef double SAE(self, SparseArray other):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self._nele, bnele=other._nele
        cdef double r, res=0
        cdef SparseArray last
        while True:
            if a >= anele and b >= bnele:
                break
            elif a >= anele:
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
            res +=  math.fabs(r)
        if npy_isnan(res):
            return INFINITY
        return res
            
    cpdef double SSE(self, SparseArray other):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self._nele, bnele=other._nele
        cdef double r, res=0
        cdef SparseArray last
        while True:
            if a >= anele and b >= bnele:
                break
            elif a >= anele:
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
            res +=  r * r
        if npy_isnan(res):
            return INFINITY
        return res

    cpdef double pearsonr(self, SparseArray other):
        cdef double mx, my, up
        mx = self.sum() / self.size()
        my = other.sum() / other.size()
        up = ((self - mx) * (other - my)).sum()
        mx = math.sqrt((self - mx).sq().sum())
        my = math.sqrt((other - my).sq().sum())
        if mx == 0 or my == 0:
            return 1
        return up / (mx * my)
            
    cpdef bint isfinite(self):
        cdef int i
        cdef double r    
        for i in range(self.nele()):
            r = self._dataC[i]
            if npy_isnan(r) or npy_isinf(r):
                return 0
        return 1

    cdef SparseArray select(self, npc.ndarray[long, ndim=1] index):
        cdef long *indexC = <long *>index.data
        cdef list data = [], index2 = []
        cdef int anele=self.nele(), bnele=index.shape[0]
        cdef int a=0, b=0, c=0, i=0
        cdef SparseArray res
        for i in range(index.shape[0]-1):
            if index[i] > index[i+1]:
                raise NotImplementedError("The index must be in order")
        while (a < anele) and (b < bnele):
            if self._indexC[a] == indexC[b]:
                data.append(self._dataC[a])
                index2.append(c)
                b += 1
                c += 1
                if (b >= bnele) or self._indexC[a] != indexC[b]:
                    a += 1
            elif self._indexC[a] < indexC[b]:
                a += 1
            else:
                b += 1
                c += 1
        res = self.empty(len(data), index.shape[0])
        res.set_data_index(data, index2)        
        return res
                
    def __getitem__(self, value):
        cdef int i, init=-1, cnt=0
        cdef SparseArray res
        if isinstance(value, np.ndarray):
            return self.select(value)
        if not isinstance(value, slice):
            raise NotImplementedError("Not implemented yet %s" %(type(value)))
        start = value.start if value.start is not None else 0
        stop = value.stop if value.stop is not None else self.size()
        if stop > self.size():
            stop = self.size()
        for i in range(self.nele()):
            if self._indexC[i] >= start and init == -1:
                init = i
            if init > -1 and self._indexC[i] < stop:
                cnt += 1
            if self._indexC[i] >= stop:
                break
        res = self.empty(cnt, stop - start)
        for i in range(init, init+cnt):
            res._indexC[i - init] = self._indexC[i]
            res._dataC[i - init] = self._dataC[i]
        return res

    def concatenate(self, SparseArray dos):
        cdef SparseArray res = self.empty(self.nele() + dos.nele(),
                                          self.size() + dos.size())
        cdef int i, j=0, size=self.size()
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i]
        for i in range(self.nele(), res.nele()):
            res._indexC[i] = dos._indexC[j] + size
            res._dataC[i] = dos._dataC[j] 
            j += 1
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

    def get_data(self):
        cdef list d=[]
        cdef int i
        for i in range(self.nele()):
            d.append(self._dataC[i])
        return d

    def get_index(self):
        cdef list d=[]
        cdef int i
        for i in range(self.nele()):
            d.append(self._indexC[i])
        return d    
        
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
    def fromlist(cls, iter, bint force_finite=False):
        self = cls()
        data = []
        index = []
        k = -1
        for k, v in enumerate(iter):
            if force_finite and (npy_isnan(v) or npy_isinf(v)):
                continue
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

    cpdef SparseArray copy(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i]
        return res
            
    cpdef SparseArray constant(self, double v, int size=-1):
        cdef int i
        cdef SparseArray res = SparseArray()
        if size == -1:
            size = self.size()
        res.set_size(size)        
        if v == 0:
            res.init(0)
        else:
            res.init(size)
        for i in range(res.nele()):
            res._dataC[i] = v
            res._indexC[i] = i
        return res

    @cython.boundscheck(False)
    @cython.nonecheck(False)        
    def BER(self, SparseArray yh, array.array[double] class_freq):
        cdef array.array[double] err = array.array('d',
                                                   map(lambda x: 0,
                                                       range(len(class_freq))))
        cdef int i=0, j=0, k=0, ynele=self.nele(), yhnele=yh.nele()
        cdef int c1=0, c2=0
        cdef double res=0
        for k in range(self.size()):
            if self._indexC[i] == k:
                c1 = int(self._dataC[i])
                if i < ynele - 1:
                    i += 1
            else:
                c1 = 0
            if j < yhnele and yh._indexC[j] == k:
                c2 = int(yh._dataC[j])
                j += 1
            else:
                c2 = 0
            if c1 != c2:
                err[c1] += 1
        for i in range(len(class_freq)):
            res += err[i] / class_freq[i]
        return res / len(class_freq) * 100.
        
    @staticmethod
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def distance(list X, list ev, npc.ndarray[double, ndim=2] output):
        cdef SparseArray x
        cdef SparseArray y
        cdef double *data = <double *> output.data
        cdef int c = 0, i=0, j=0, len_X = len(X)
        for i in range(len(ev)):
            x = ev[i]
            for j in range(len_X):
                y = X[j]
                data[c] = x.SAE(y)
                c += 1

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)    
    cdef SparseArray joint_log_likelihood_sum(self, list Xs, list mu, list var, double n_ij, int cl):
        cdef double muj, varj, xj, tmp
        cdef int i, size=Xs[0].size(), nvar=len(Xs), j, k, xnele
        cdef SparseArray res = self.empty(size, size), x
        for i in range(size):
            res._dataC[i] = 0
            res._indexC[i] = i
        for j in range(nvar):
            k = 0
            x = Xs[j]
            muj = mu[j][cl]
            varj = var[j][cl]
            xnele = x.nele()
            for i in range(size):
                xj = 0
                if k < xnele and x._indexC[k] == i:
                    xj = x._dataC[k]
                    k += 1
                tmp = (xj - muj)
                res._dataC[i] = res._dataC[i] + (tmp * tmp) / varj
        for i in range(size):
            res._dataC[i] = -0.5 * res._dataC[i] - n_ij
        return res
                
    @staticmethod
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def joint_log_likelihood(list Xs, list mu, list var,
                             array.array[double] log_cl_prior):
        cdef int i
        cdef list llh = []
        cdef double n_ij
        cdef SparseArray sn_ij, x=Xs[0]
        cdef array.array[double] s
        for i in range(len(log_cl_prior)):
            n_ij = 0
            for s in var:
                n_ij += math.log(2. * s[i] * PI)
            n_ij = 0.5 * n_ij - log_cl_prior[i]
            sn_ij = x.joint_log_likelihood_sum(Xs, mu, var, n_ij, i)
            llh.append(sn_ij.tonparray())
        return np.array(llh).T
        
cdef class SparseEval:
    def __cinit__(self, npc.ndarray[long, ndim=1] nop):
        self._nop = <long *> nop.data
        self._nfunc = nop.shape[0]
        self._pos = 0
        self._st = None
        
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

    def get_X(self):
        return self._x

    def X(self, x):
        if isinstance(x, types.ListType):
            self._x = x
        else:
            self._x = map(lambda i: SparseArray.fromlist(x[:, i]),
                          range(x.shape[1]))
        self.set_nvar(len(self._x))
        self.set_size(self._x[0].size())
        self._x1 = self._x[0]

    cpdef eval(self, npc.ndarray[long, ndim=1] ind,
               npc.ndarray[double, ndim=1] constants, bint to_np_array=1,
               list st=None):
        self._pos = 0
        self._st_pos = 0
        self._ind = <long *> ind.data
        # self._st = 
        self._constants = <double *> constants.data
        self._st = st
        if st is not None:
            self._use_st = 1
        if to_np_array:
            return self._eval().tonparray()
        else:
            return self._eval()

    cdef SparseArray terminal(self, int node):
        cdef SparseArray res
        if self.isvar(node):
            res = self._x[node - self._nfunc]
        else:
            v = self._constants[node - self._nfunc - self._nvar]
            res = self._x1.constant(v)
        return res

    cdef SparseArray function_set(self, int node, list args):
        cdef int nargs = self._nop[node]    
        if node == 15:
            self._output = args
            return self._output[0]
        elif nargs == 1:
            return self.one_arg(node, args[0])
        elif nargs == 2:
            return self.two_args(node, args[0], args[1])
        elif nargs == 3:
            if node == 10:
                return args[0].if_func(args[1], args[2])
        raise NotImplementedError("%s" % node)
            
    cpdef eval2(self, npc.ndarray[long, ndim=1] ind,
                npc.ndarray[double, ndim=1] constants, bint to_np_array=1):
        cdef list func=[], args=[]
        cdef int node, nargs, pos=0
        cdef SparseArray res
        self._ind = <long *> ind.data
        self._constants = <double *> constants.data
        if not self.isfunc(self._ind[0]):
            args.append(self.terminal(self._ind[0]))
        else:
            func.append((pos, self._nop[self._ind[pos]]))
            while len(func):
                if func[-1][1] == 0:
                    pf, _ = func.pop()
                    node = self._ind[pf]
                    args2 = [args.pop() for _ in range(self._nop[node])]
                    args2.reverse()
                    res = self.function_set(node, args2)
                    args.append(res)
                    if len(func):
                        pf, nargs = func.pop()
                        func.append((pf, nargs-1))
                else:
                    pos += 1
                    node = self._ind[pos]
                    if not self.isfunc(node):
                        args.append(self.terminal(node))
                        pf, nargs = func.pop()
                        func.append((pf, nargs-1))
                    else:
                        func.append((pos, self._nop[node]))
        if to_np_array:
            return args[0].tonparray()
        else:
            return args[0]
            
    cdef SparseArray two_args(self, int func, SparseArray first,
                              SparseArray second):
        if func == 0:  # add
            return first.add(second)
        elif func == 1:  # subtract
            return first.sub(second)
        elif func == 2:  # multiply
            return first.mul(second)
        elif func == 3:  # divide
            return first.div(second)
        else:
            raise NotImplementedError("%s" % func)

    cdef SparseArray one_arg(self, int func, SparseArray first):
        if func == 4:
            return first.fabs()
        if func == 5:
            return first.exp()
        elif func == 6:
            return first.sqrt()
        elif func == 7:
            return first.sin()
        elif func == 8:
            return first.cos()
        elif func == 9:
            return first.sigmoid()
        elif func == 13:
            return first.ln()
        elif func == 14:
            return first.sq()
        else:
            raise NotImplementedError("%s" % func)

    def get_output(self):
        return self._output
        
    cdef SparseArray _eval(self):
        cdef SparseArray res, first, second, third
        cdef int pos = self._pos
        cdef double v
        self._pos += 1
        cdef int node = self._ind[pos], nop
        if self._use_st and self._st[pos] is not None:
            self._pos = self.traverse(pos)
            return self._st[pos]
        if self.isfunc(node):
            nop = self._nop[node]
            res = self.function_set(node,
                                    [self._eval() for x in range(nop)])
        elif self.isvar(node):
            res = self._x[node - self._nfunc]
        else:
            v = self._constants[node - self._nfunc - self._nvar]
            res = self._x1.constant(v)
        if self._use_st:
            self._st[pos] = res
        return res
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int traverse(self, int pos):
        cdef int nargs = 1
        cdef int cnt = 0
        cdef int pos_ini = pos
        cdef int ele
        cdef long *_funcC = self._nop
        cdef long *indC = self._ind
        while True:
            ele = indC[pos]
            pos += 1
            cnt += 1
            if self.isfunc(ele):
                nargs -= 1
                nargs += _funcC[ele]
            else:
                nargs -= 1
            if nargs == 0 :
                return cnt + pos_ini
        
