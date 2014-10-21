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
cimport libc
cimport libc.stdlib as stdlib
cimport libc.math as math
cdef extern from "math.h":
    float sqrt(double)
    float sin(double)
    float cos(double)
cdef extern from "numpy/npy_math.h":
    bint npy_isinf(double)
    bint npy_isnan(double)


np.seterr(all='ignore')

cdef class Recursive:
    cdef INT end
    cdef INT ninputs
    cdef INT nmem
    cdef FLOAT *x
    cdef INT *nop
    cdef INT *indC
    cdef FLOAT *constantsC
    cdef FLOAT *resC
    cdef FLOAT *memC
    cdef INT nfunc
    cdef INT nlags
    cdef INT ncases
    cdef INT *cases
    def __cinit__(self, npc.ndarray[INT, ndim=1] _nop, int nfunc, npc.ndarray[FLOAT, ndim=2, mode="c"] _x,
                  npc.ndarray[FLOAT, ndim=1] mem,
                  npc.ndarray[FLOAT, ndim=1] res, npc.ndarray[INT, ndim=1] cases, int nlags=1):
        self.x = <FLOAT*> _x.data
        self.nop = <INT*> _nop.data
        self.resC = <FLOAT*> res.data
        self.memC = <FLOAT*> mem.data
        self.end = _x.shape[0]
        self.ninputs = _x.shape[1]
        self.nmem = mem.shape[0]
        self.nfunc = nfunc
        self.nlags = nlags
        self.ncases = cases.shape[0]
        self.cases = <INT *> cases.data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef update_x(self, int j):
        cdef int end = self.end, k=0, ninputs = self.ninputs
        cdef FLOAT *x = self.x
        cdef FLOAT *resC = self.resC
        if j < end and (self.ncases == 0 or self.cases[j] == 0):
            for k in range(self.nlags-1, 0, -1):
                x[j*ninputs+k] = x[(j-1)*ninputs+k-1]
            x[j*ninputs] = resC[j-1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef eval_ind_inner_iter(self, npc.ndarray[INT, ndim=1] ind,
                              npc.ndarray[FLOAT, ndim=1] _constants,
                              int pos):
        self.indC = <INT*> ind.data
        self.constantsC = <FLOAT*> _constants.data
        cdef int i=0, end=self.end, ninputs = self.ninputs, j, nmem = self.nmem, k=0
        cdef FLOAT *x = self.x
        cdef FLOAT *resC = self.resC
        for i in range(end):
            pos = 0
            resC[i] = self.eval_ind_inner2(&pos, &x[i*ninputs])
            j = i + 1
            self.update_x(j)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef FLOAT eval_ind_inner2(self, int *pos,
                               FLOAT *_x):
        cdef int nargs, func, c, ppos = pos[0]
        cdef FLOAT a, b, d
        cdef INT *ind = self.indC
        if ind[ppos] < self.nfunc:
            func = ind[ppos]
            nargs = self.nop[func]
            pos[0] += 1
            a = self.eval_ind_inner2(pos, _x)
            if nargs == 2:
                b = self.eval_ind_inner2(pos, _x)
            if nargs == 3:
                b = self.eval_ind_inner2(pos, _x)
                d = self.eval_ind_inner2(pos, _x)
            return self.apply_func(func, a, b, d)
        elif ind[ppos] < self.nfunc + self.ninputs:
            a = _x[ind[ppos] - self.nfunc]
            pos[0] += 1
            return a
        else:
            c =  ind[ppos] - self.nfunc - self.ninputs
            a = self.constantsC[c]
            pos[0] += 1
            return a

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef FLOAT apply_func(self, int func, FLOAT a, FLOAT b, FLOAT c):
        if func == 0:
            return a + b
        elif func == 1:
            return a - b
        elif func == 2:
            return a * b
        elif func == 3:
            return a / b
        elif func == 4:
            return math.fabs(a)
        elif func == 5:
            return math.exp(a)
        elif func == 6:
            return math.sqrt(a)
        elif func == 7:
            return math.sin(a)
        elif func == 8:
            return math.cos(a)
        elif func == 9:
            return 1 / (1 + math.exp(-a))
        elif func == 10:
            a = 1 / (1 + math.exp(-100 * a))
            return a * (b - c) + c
        elif func == 11:
            s = 1 / (1 + math.exp(-100 * (a-b)))
            return s * (a - b) + b
        elif func == 12:
            s = 1 / (1 + math.exp(-100 * (a-b)))
            return s * (b - a) + a
        elif func == 13:
            return math.log(math.fabs(a))
        elif func == 14:
            return a * a
        elif func == 15:
            if npy_isinf(a) or npy_isnan(a):
                return a
            return self.memC[int(a) % self.nmem]
        elif func == 16:
            if not (npy_isinf(a) or npy_isnan(a)) :
                self.memC[int(a) % self.nmem] = b
            return b


cdef class RProp(Recursive):
    cdef FLOAT *rprop
    cdef FLOAT *target
    cdef INT skip

    def set_rprop(self, npc.ndarray[FLOAT, ndim=1] r, INT max_op=2):
        self.rprop = <FLOAT *>r.data
        self.skip = r.shape[0] / max_op
        
    def set_target(self, npc.ndarray[FLOAT, ndim=1] target):
        self.target = <FLOAT *>target.data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef eval_ind_inner_iter(self, npc.ndarray[INT, ndim=1] ind,
                              npc.ndarray[FLOAT, ndim=1] _constants,
                              int pos):
        self.indC = <INT*> ind.data
        self.constantsC = <FLOAT*> _constants.data
        cdef int i=0, end=self.end, ninputs = self.ninputs, j, nmem = self.nmem, k=0
        cdef FLOAT *x = self.x
        cdef FLOAT *resC = self.resC
        cdef FLOAT error = 0
        for i in range(end):
            pos = 0
            resC[i] = self.eval_ind_inner(&pos, &x[i*ninputs], error)
            error = resC[i] - self.target[i]
            j = i + 1
            self.update_x(j)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef FLOAT eval_ind_inner(self, int *pos,
                               FLOAT *_x, float error):
        cdef int nargs, func, c, ppos = pos[0]
        cdef FLOAT a, b, d
        cdef INT *ind = self.indC
        if ind[ppos] < self.nfunc:
            func = ind[ppos]
            nargs = self.nop[func]
            pos[0] += 1
            a = self.eval_ind_inner(pos, _x, error * self.rprop[ppos])
            if nargs == 2:
                b = self.eval_ind_inner(pos, _x, error * self.rprop[ppos+self.skip])
            if nargs == 3:
                b = self.eval_ind_inner(pos, _x, error * self.rprop[ppos+self.skip])
                d = self.eval_ind_inner(pos, _x, error * self.rprop[ppos+2*self.skip])
            self.store(func, a, b, d, ppos)
            return self.apply_func(func, a, b, d)
        elif ind[ppos] < self.nfunc + self.ninputs:
            a = _x[ind[ppos] - self.nfunc]
            pos[0] += 1
            return a
        else:
            c =  ind[ppos] - self.nfunc - self.ninputs
            self.rprop[ppos] += error
            # if error > 0:
            #     self.rprop[ppos] += 1
            # else:
            #     self.rprop[ppos] -= 1
            a = self.constantsC[c]
            pos[0] += 1
            return a


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef FLOAT store(self, int func, FLOAT a, FLOAT b, FLOAT c, INT pos):
        cdef FLOAT s
        if func == 0:
            self.rprop[pos] = 1
            self.rprop[pos+self.skip] = 1
        elif func == 1:
            self.rprop[pos] = 1
            self.rprop[pos+self.skip] = -1
        elif func == 2:
            self.rprop[pos] = b
            self.rprop[pos+self.skip] = a
        elif func == 3:
            self.rprop[pos] = 1/b
            self.rprop[pos+self.skip] = -a/(b*b)
        elif func == 4:
            self.rprop[pos] = 1 if a >= 0 else -1
        elif func == 5:
            self.rprop[pos] = math.exp(a)
        elif func == 6:
            self.rprop[pos] = 1/ (2 * math.sqrt(a))
        elif func == 7:
            self.rprop[pos] = math.cos(a)
        elif func == 8:
            self.rprop[pos] = -math.sin(a)
        elif func == 9:
            a = 1 / ( 1 + math.exp(-a))
            self.rprop[pos] = a * (1 - a)
        elif func == 10:
            s = ( 1 / (1 + math.exp(-100 * a)))
            self.rprop[pos] = (b - c) * s * (1 - s)
            self.rprop[pos+self.skip] = s 
            self.rprop[pos+2*self.skip] = 1 - s
        elif func == 11:
            s = ( 1 / (1 + math.exp(-100 * (a - b))))
            self.rprop[pos] = (a - b) * s * (1 - s) + s
            self.rprop[pos+self.skip] = (a - b) * s * (s - 1) + 1 - s
        elif func == 12:
            s = ( 1 / (1 + math.exp(-100 * (a - b))))
            self.rprop[pos] = (b - a) * s * (1 - s) + 1 - s
            self.rprop[pos+self.skip] = (b - a) * s * (s - 1) + s
            
