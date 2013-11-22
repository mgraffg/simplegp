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
cimport numpy as npc
import numpy as np
cimport cython
ctypedef npc.float_t FLOAT
ctypedef npc.int_t INT
ctypedef npc.uint8_t UINT8
cimport libc
cimport libc.stdlib as stdlib
from libc.string cimport memcpy
cimport libc.math as math
cdef extern from "math.h":
    int isinf(double)
    int isnan(double)
    float sqrt(double)

np.seterr(all='ignore')


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef eval_ind_inner(npc.ndarray[INT, ndim=1] ind, int pos,
                     npc.ndarray _func, npc.ndarray[FLOAT, ndim=2, mode="c"] _x,
                     npc.ndarray[FLOAT, ndim=1] _constants):
    cdef int shape = _func.shape[0]
    cdef int i=0, nargs
    if ind[pos] < shape:
        func = _func[ind[pos]]
        pos += 1
        nargs = func.nargs - 1
        args = []
        for i in range(nargs):
            r, pos = eval_ind_inner(ind, pos, _func, _x, _constants)
            args.append(r)
        return func(*args), pos
    elif ind[pos] < shape + _x.shape[1]:
        return _x[:, ind[pos] - shape], pos + 1
    else:
        c =  ind[pos] - shape - _x.shape[1]
        return _constants[c], pos + 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef traverse(npc.ndarray[INT, ndim=1, mode="c"] ind, INT pos, npc.ndarray[INT, ndim=1, mode="c"] _func):
    cdef INT nargs = 1
    cdef INT cnt = 0
    cdef INT pos_ini = pos
    cdef INT ele
    cdef INT shape = _func.shape[0]
    cdef INT *indC = <INT*> ind.data
    cdef INT *_funcC = <INT*> _func.data
    while True:
        ele = indC[pos]
        pos += 1
        cnt += 1
        if ele < shape: 
            nargs -= 1
            nargs += _funcC[ele]
        else:
            nargs -= 1
        if nargs == 0 :
            return cnt + pos_ini

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef copy(npc.ndarray[FLOAT, ndim=2, mode="c"] a,
           npc.ndarray[FLOAT, ndim=2, mode="c"] b,
           int l):
    cdef FLOAT *aC = <FLOAT *> a.data
    cdef FLOAT *bC = <FLOAT *> b.data
    memcpy(aC, bC, sizeof(FLOAT) * l)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sign(npc.ndarray[UINT8, ndim=2, mode="c"] a,
           npc.ndarray[FLOAT, ndim=2, mode="c"] b,
           int l):
    cdef UINT8 *aC = <UINT8 *> a.data
    cdef FLOAT *bC = <FLOAT *> b.data
    cdef int i
    for i in range(l):
        if bC[i] < 0:
            aC[i] = 1
        else:
            aC[i] = 0


cdef class Length:
    cdef INT *_ind
    cdef INT *_length
    cdef INT *_nop
    cdef INT _pos
    cdef int _nfunc
    def __cinit__(self, npc.ndarray[INT, ndim=1] ind,
                  npc.ndarray[INT, ndim=1] length,
                  npc.ndarray[INT, ndim=1] nop):
        self._ind = <INT *> ind.data
        self._length = <INT *> length.data
        self._nop = <INT *> nop.data
        self._pos = 0
        self._nfunc = nop.shape[0]
        self.compute()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int compute(self):
        cdef int node, i, l=1
        cdef int opos = self._pos
        node = self._ind[self._pos]
        self._pos += 1
        if node < self._nfunc:
            for i in range(self._nop[node]):
                l += self.compute()
        self._length[opos] = l
        return l












