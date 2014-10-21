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
cdef extern from "numpy/npy_math.h":
    bint npy_isinf(double)
    bint npy_isnan(double)
from .eval import Eval
from .eval cimport Eval

np.seterr(all='ignore')

cdef class Simplify:
    cdef INT *_nop
    cdef INT *_ind
    cdef INT _set_constants
    cdef FLOAT *_constants
    cdef FLOAT *_constants2
    cdef INT _nvar
    cdef INT _nfunc
    cdef INT _cntC
    cdef INT _pos
    cdef Eval _eval
    cdef npc.ndarray _x_eval
    cdef npc.ndarray _st_eval
    cdef npc.ndarray _ind_eval
    cdef npc.ndarray _c_eval
    def __cinit__(self, int nvar, npc.ndarray[INT, ndim=1] _nop):
        self._nvar = nvar
        self._nop = <INT *>_nop.data
        self._nfunc = _nop.shape[0]
        self._cntC = 2
        self._set_constants = 0
        self._pos = 0
        self._eval = Eval(0, nvar, _nop)
        self._x_eval = np.empty((2, 0), dtype=np.float)
        self._st_eval = np.empty((4, 1), dtype=np.float)
        self._ind_eval = np.empty(4, dtype=np.int)
        self._c_eval = np.empty(3, dtype=np.float)

    def get_nconstants(self):
        return self._cntC

    def get_constants(self):
        cdef npc.ndarray[FLOAT, ndim=1] c = np.zeros(self._cntC)
        cdef int i
        for i in range(self._cntC):
            c[i] = self._constants2[i]
        return c

    def set_constants(self, npc.ndarray[FLOAT, ndim=1] c):
        self._constants2 = <FLOAT *>c.data
        self._constants2[0] = 0
        self._constants2[1] = 1


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT clean(self, int pos):
        cdef INT *ind = self._ind
        cdef INT *nop = self._nop
        cdef int nargs = 1
        cdef int pos_ini = pos
        cdef int ele
        cdef int shape = self._nfunc
        while True:
            ele = ind[pos]
            if ele == -1:
                pos += 1
                continue
            ind[pos] = -1
            pos += 1
            if ele < shape: 
                nargs -= 1
                nargs += nop[ele]
            else:
                nargs -= 1
            if nargs == 0 :
                return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int equal_node(self, INT a, INT b):
        if self.isconstant(a) and self.isconstant(b):
            if self.constant_value(a) == self.constant_value(b):
                return 1
            else:
                return 0
        elif a == b:
            return 1
        return 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT equal(self, int a, int b):
        cdef INT *ind = self._ind
        cdef INT *nop = self._nop
        cdef int nargs = 1
        cdef int ele
        cdef int shape = self._nfunc
        while True:
            if ind[a] == -1:
                a += 1
                continue
            if ind[b] == -1:
                b += 1
                continue
            if self.equal_node(ind[a], ind[b]) == 0:
                return 0
            ele = ind[a]
            a += 1
            b += 1
            if ele < shape: 
                nargs -= 1
                nargs += nop[ele]
            else:
                nargs -= 1
            if nargs == 0 :
                return 1
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT generate_constant(self, FLOAT c):
        if npy_isnan(c) or npy_isinf(c): return self._nfunc + self._nvar  
        cdef int cntC = self._cntC
        self._cntC += 1
        self._constants2[cntC] = c
        return cntC + self._nfunc + self._nvar  

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT isconstant(self, INT a):
        if a < self._nfunc + self._nvar: return 0
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT isterminal(self, INT a):
        if a >= self._nfunc: return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef FLOAT constant_value(self, INT a):
        return self._constants2[a - self._nfunc - self._nvar]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT sum_reduce(self, INT pos, INT a, INT b):
        cdef INT *ind = self._ind
        if (self.isconstant(ind[a]) and
            self.constant_value(ind[a]) == 0):
            ind[a] = -1
            ind[pos] = -1 
            return b
        if (self.isconstant(ind[b]) and
            self.constant_value(ind[b]) == 0):
            ind[b] = -1
            ind[pos] = -1
            return a
        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT subtract_reduce(self, INT pos, INT a, INT b):
        cdef INT *ind = self._ind
        if self.equal(a, b):
            self.clean(a)
            self.clean(b)
            ind[pos] = self.generate_constant(0)
            return pos
        if (self.isconstant(ind[b]) and
            self.constant_value(ind[b]) == 0):
            ind[b] = -1
            ind[pos] = -1
            return a
        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT multiply_reduce(self, INT pos, INT a, INT b):
        cdef INT *ind = self._ind
        if (self.isconstant(ind[a]) and
            self.constant_value(ind[a]) == 1):
            ind[a] = -1
            ind[pos] = -1
            return b
        if (self.isconstant(ind[b]) and
            self.constant_value(ind[b]) == 1):
            ind[b] = -1
            ind[pos] = -1
            return a
        if ((self.isconstant(ind[a]) and
            self.constant_value(ind[a]) == 0) or
            (self.isconstant(ind[b]) and
             self.constant_value(ind[b]) == 0)):
            self.clean(a)
            self.clean(b)
            ind[pos] = self.generate_constant(0)
            return pos
        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT divide_reduce(self, INT pos, INT a, INT b):
        cdef INT *ind = self._ind
        if self.equal(a, b):
            self.clean(a)
            self.clean(b)
            ind[pos] = self.generate_constant(1)
            return pos
        if (self.isconstant(ind[a]) and
            self.constant_value(ind[a]) == 0):
            self.clean(a)
            self.clean(b)
            ind[pos] = self.generate_constant(0)
            return pos
        if (self.isconstant(ind[b]) and
            self.constant_value(ind[b]) == 1):
            ind[pos] = -1
            ind[b] = -1
            return a
        return -1


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT if_reduce(self, INT pos, INT a, INT b, INT c):
        cdef INT *ind = self._ind
        if self.isconstant(ind[a]):
            if self.constant_value(ind[a]) > 0.1:
                ind[pos] = -1
                self.clean(a)
                self.clean(c)
                return b
            elif self.constant_value(ind[a]) < -0.1:
                ind[pos] = -1
                self.clean(a)
                self.clean(b)
                return c
        return -1


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT arithmetic_reduce(self, INT pos, INT a, INT b):
        cdef INT *ind = self._ind
        cdef INT func = ind[pos]
        cdef INT tmp = -1
        if func == 0:
            tmp = self.sum_reduce(pos, a, b)
        elif func == 1:
            tmp = self.subtract_reduce(pos, a, b)
        elif func == 2:
            tmp = self.multiply_reduce(pos, a, b)
        elif func == 3:
            tmp = self.divide_reduce(pos, a, b)
        return tmp


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT next_pos(self, INT pos):
        cdef INT *ind = self._ind
        while True:
            if ind[pos] > -1:
                return pos
            pos += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT other_func_reduce(self, INT pos, INT a, INT b, INT c):
        cdef INT *ind = self._ind
        cdef INT func = ind[pos]
        cdef INT tmp = -1
        # fabs
        if func == 4:
            # in the case the next function is abs or exp or argmax
            if ind[pos] == ind[a] or ind[a] == 5 or ind[a] == 16:
                ind[pos] = -1
                tmp = a
        # sigmoid
        elif func == 9:
            if ind[pos] == ind[a]:
                ind[pos] = -1
                tmp = a
        elif func == 10:
            tmp = self.if_reduce(pos, a, b, c)
        # exp
        elif func == 5:
            if ind[a] == 13:
                ind[pos] = -1
                ind[a] = -1
                tmp = self.next_pos(a)
        # Ln
        elif func == 13:
            if ind[a] == 5:
                ind[pos] = -1
                ind[a] = -1
                tmp = self.next_pos(a)
        # Sq
        elif func == 14:
            if ind[a] == 6:
                ind[pos] = -1
                ind[a] = -1
                tmp = self.next_pos(a)
        # sqrt
        elif func == 6:
            if ind[a] == 14:
                ind[pos] = -1
                ind[a] = -1
                tmp = self.next_pos(a)
        return tmp


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT apply(self, INT pos, INT a, INT b, INT c):
        cdef FLOAT *_c_eval = <FLOAT *> self._c_eval.data
        cdef INT *_ind_eval = <INT *> self._ind_eval.data
        cdef FLOAT *_st_eval = <FLOAT *> self._st_eval.data
        cdef INT *ind = self._ind
        cdef INT tmp
        if b == -1 and self.isconstant(ind[a]) and ind[pos] != 15:
            _ind_eval[0] = ind[pos]
            _ind_eval[1] = self._nfunc + self._nvar
            _c_eval[0] = self.constant_value(ind[a])
            self._eval._pos = 0
            self._eval.eval_ind(self._ind_eval, self._x_eval,
                                self._st_eval, self._c_eval)
            ind[pos] = self.generate_constant(_st_eval[0])
            ind[a] = -1
            return pos
        elif (self.isconstant(ind[a]) and 
              self.isconstant(ind[b]) and c == -1 and ind[pos] != 15):
            _ind_eval[0] = ind[pos]
            _ind_eval[1] = self._nfunc + self._nvar
            _ind_eval[2] = self._nfunc + self._nvar + 1
            _c_eval[0] = self.constant_value(ind[a])
            _c_eval[1] = self.constant_value(ind[b])
            self._eval._pos = 0
            self._eval.eval_ind(self._ind_eval, self._x_eval,
                                self._st_eval, self._c_eval)
            ind[pos] = self.generate_constant(_st_eval[0])
            ind[a] = -1
            ind[b] = -1
            return pos
        elif (self.isconstant(ind[a]) and 
              self.isconstant(ind[b]) and
              self.isconstant(ind[c]) and ind[pos] != 15):
            _ind_eval[0] = ind[pos]
            _ind_eval[1] = self._nfunc + self._nvar
            _ind_eval[2] = self._nfunc + self._nvar + 1
            _ind_eval[3] = self._nfunc + self._nvar + 2
            _c_eval[0] = self.constant_value(ind[a])
            _c_eval[1] = self.constant_value(ind[b])
            _c_eval[2] = self.constant_value(ind[c])
            self._eval._pos = 0
            self._eval.eval_ind(self._ind_eval, self._x_eval,
                                self._st_eval, self._c_eval)
            ind[pos] = self.generate_constant(_st_eval[0])
            ind[a] = -1
            ind[b] = -1
            ind[c] = -1
            return pos
        tmp = self.arithmetic_reduce(pos, a, b)
        if tmp > -1:
            return tmp
        tmp = self.other_func_reduce(pos, a, b, c)
        if tmp > -1:
            return tmp
        return pos
    
    def simplify(self, npc.ndarray[INT, ndim=1] ind,
                 npc.ndarray[FLOAT, ndim=1] constants):
        self._ind = <INT *>ind.data
        self._constants = <FLOAT *>constants.data
        self._pos=0
        self._cntC = 2
        self.simplify_inner()
        return ind[ind > -1]
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT simplify_inner(self):
        cdef INT *ind = self._ind, a, b=-1, c=-1
        cdef int pos = self._pos, i=0, nargs
        self._pos += 1
        if ind[pos] < self._nfunc:
            nargs = self._nop[ind[pos]]
            a = self.simplify_inner()
            if nargs >= 2: 
                b = self.simplify_inner()
            if nargs >= 3:
                c = self.simplify_inner()
            if nargs > 3:
                for i in range(3, nargs):
                    self.simplify_inner()
            return self.apply(pos, a, b, c)
        if ind[pos] < self._nfunc + self._nvar:
            return pos
        res = self.generate_constant(self._constants[ind[pos] - self._nfunc - self._nvar])
        ind[pos] = res
        return pos








