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
cimport libc
cimport libc.math as math
cimport libc.stdlib as stdlib
cdef extern from "numpy/npy_math.h":
    bint npy_isinf(double)
    bint npy_isnan(double)

np.seterr(all='ignore')

cdef class Eval:
    def __cinit__(self, INT pos,
                  INT nvar,
                  npc.ndarray[INT, ndim=1] nop,
                  INT max_nargs=3):
        # number of operands
        self._nop = <INT *> nop.data
        # the initial point of the evaluation
        self._pos = pos
        # number of function
        self._nfunc = nop.shape[0]
        # number of variables
        self._nvar = nvar
        # setting the maximum number of arguments
        self._max_nargs = max_nargs
        # 
        self._pmut_eval_flag = 0
 
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_output_function(self, npc.ndarray[INT, ndim=1] _output):
        self._output = <INT *> _output.data
        self._n_output = _output.shape[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_pos(self, INT pos):
        self._pos = pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def eval_ind(self, npc.ndarray[INT, ndim=1] ind,
                 npc.ndarray[FLOAT, ndim=2, mode="c"] _x,
                 npc.ndarray[FLOAT, ndim=2, mode="c"] _st,
                 npc.ndarray[FLOAT, ndim=1] _constants):
        self._ind = <INT *> ind.data
        self._st  = <FLOAT *> _st.data
        self._l_st = _st.shape[1]
        self._x = <FLOAT *> _x.data
        self._cons = <FLOAT *> _constants.data
        cdef INT r = self.eval_ind_inner()
        return r
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT eval_ind_inner(self):
        cdef INT pos = self._pos, a, b, c, o_i, i
        cdef INT *ind = self._ind
        cdef INT t
        self._pos += 1
        cdef INT ele = ind[pos]
        cdef INT nvar = self._nvar
        cdef INT nfunc = self._nfunc, nargs = 0
        cdef FLOAT _c
        cdef FLOAT *_st = self._st, *_xC = self._x, *_cons = self._cons
        cdef INT *args
        if ele < nfunc:
            nargs = self._nop[ele]
            # output function
            if ele == 15:
                for i in range(nargs):
                    self._output[i] = self.eval_ind_inner()
                self.output_function()
                return pos
            # argmax function
            if ele == 16:
                args = <INT *> stdlib.malloc(sizeof(INT)*nargs)
                for i in range(nargs):
                    args[i] = self.eval_ind_inner()
                self.variable_args(ele, args, nargs, pos)
                stdlib.free(args)
                return pos
            if nargs == 1:
                a = self.eval_ind_inner()
                self.one_args(ele, a, pos)
                return pos
            elif nargs == 2:
                a = self.eval_ind_inner()
                b = self.eval_ind_inner()
                self.two_args(ele, a, b, pos)
                return pos
            elif nargs == 3:
                a = self.eval_ind_inner()
                b = self.eval_ind_inner()
                c = self.eval_ind_inner()
                self.three_args(ele, a, b, c, pos)
                return pos
        elif ele < nfunc + nvar:
            o_i = self._l_st * pos
            c = ele - nfunc
            for i in range(self._l_st):
                _st[o_i + i] = _xC[i*nvar + c]
            return pos
        else:
            o_i = self._l_st * pos
            _c = _cons[ele - nfunc - nvar]
            for i in range(self._l_st):
                _st[o_i + i] = _c
            return pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void one_args(self, 
                      INT func,
                      INT a,
                      INT pos):
        if func == 4:
            self.fabs(a, pos)
        elif func == 5:
            self.exp(a, pos)
        elif func == 6:
            self.sqrt(a, pos)
        elif func == 7:
            self.sin(a, pos)
        elif func == 8:
            self.cos(a, pos)
        elif func == 9:
            self.sigmoid(a, pos)
        elif func == 13:
            self.ln(a, pos)
        elif func == 14:
            self.sq(a, pos)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT fabs(self, 
                  INT a,
                  INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = math.fabs(st[a_i + i])
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT exp(self, 
                 INT a,
                 INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = math.exp(st[a_i + i])
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT sqrt(self, 
                  INT a,
                  INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = math.sqrt(st[a_i + i])
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT sin(self, 
                 INT a,
                 INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = math.sin(st[a_i + i])
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT cos(self, 
                 INT a,
                 INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = math.cos(st[a_i + i])
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT sigmoid(self, 
                     INT a,
                     INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = 1 / (1 + math.exp(-st[a_i + i]))
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT ln(self, 
                INT a,
                INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = math.log(math.fabs(st[a_i + i]))
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT sq(self, 
                INT a,
                INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = st[a_i + i] * st[a_i + i]
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void two_args(self, 
                       INT func,
                       INT a,
                       INT b,
                       INT pos):
        if func == 0:
            self.add(a, b, pos)
        elif func == 1:
            self.subtract(a, b, pos)
        elif func == 2:
            self.multiply(a, b, pos)
        elif func == 3:
            self.divide(a, b, pos)
        elif func == 11:
            self.max_func(a, b, pos)
        elif func == 12:
            self.min_func(a, b, pos)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT add(self, 
                 INT a,
                 INT b,
                 INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT b_i = l * b
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = st[a_i + i] + st[b_i + i]
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT subtract(self, 
                      INT a,
                      INT b,
                      INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT b_i = l * b
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = st[a_i + i] - st[b_i + i]
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT multiply(self, 
                      INT a,
                      INT b,
                      INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT b_i = l * b
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = st[a_i + i] * st[b_i + i]
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT divide(self, 
                    INT a,
                    INT b,
                    INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT b_i = l * b
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        for i in range(l):
            st[o_i + i] = st[a_i + i] / st[b_i + i]
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT max_func(self, 
                      INT a,
                      INT b,
                      INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT b_i = l * b
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        cdef FLOAT s, x, y
        for i in range(l):
            x = st[a_i + i]
            y = st[b_i + i]
            s = 1 / (1 + math.exp(-100 * (x-y)))
            st[o_i + i] = s * (x - y) + y
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT min_func(self, 
                      INT a,
                      INT b,
                      INT pos):
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT b_i = l * b
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        cdef FLOAT s, x, y
        for i in range(l):
            x = st[a_i + i]
            y = st[b_i + i]
            s = 1 / (1 + math.exp(-100 * (x-y)))
            st[o_i + i] = s * (y - x) + x
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void three_args(self, 
                         INT func,
                         INT a,
                         INT b,
                         INT c,
                         INT pos):
        # if
        cdef INT l = self._l_st
        cdef INT a_i = l * a
        cdef INT b_i = l * b
        cdef INT c_i = l * c
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0
        cdef FLOAT *st = self._st
        cdef FLOAT s, x, y, z
        for i in range(l):
            x = st[a_i + i]
            y = st[b_i + i]
            z = st[c_i + i]
            s = 1 / (1 + math.exp(-100 * x))
            st[o_i + i] = s * (y - z) + z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void output_function(self):
        cdef INT *output = self._output
        cdef INT i, j
        cdef INT l = self._l_st
        cdef INT nargs = self._n_output
        cdef FLOAT argmax
        cdef FLOAT max
        cdef FLOAT *st = self._st
        cdef INT po_i = 0
        for i in range(l):
            max = st[output[0] * l + i]
            argmax = 0
            for j in range(1, nargs):
                if st[output[j] * l + i] > max:
                    argmax = j
                    max = st[output[j] * l + i]
            st[i] = argmax

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void variable_args(self, INT func, INT *a, INT nargs, INT pos):
        if func == 16:
            self.argmax(a, nargs, pos)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void argmax(self, INT *a, INT nargs, INT pos):
        cdef INT l = self._l_st
        cdef INT o_i = l * pos
        cdef INT po_i = self._max_nargs * l * pos
        cdef INT i = 0, j = 0
        cdef FLOAT *st = self._st
        cdef FLOAT sup, sdown, up, down, beta, xi, c
        beta = 2.
        for i in range(l):
            sup = 0
            sdown = 0
            for j in range(nargs):
                xi = st[l * a[j] + i]
                down = math.exp(beta * xi)
                up = j * down
                sdown += down
                sup += up
            st[o_i + i] = sup / sdown

    def __dealloc__(self):
        if self._pmut_eval_flag == 1:
            stdlib.free(self._pmut_eval)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef pmutation_eval(self, int nop, npc.ndarray[FLOAT, ndim=2] st,
                         npc.ndarray[INT, ndim=1] index):
        self.pmutation_eval_inner(nop, <FLOAT *> st.data, <INT *> index.data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void pmutation_eval_inner(self, int nop, FLOAT *stC, INT *indexC):
        cdef int c=0, i, j, l_st=self._l_st, skip, nfunc=self._nfunc, nvar=self._nvar
        cdef INT *_nop=self._nop
        cdef FLOAT *pmut_eval
        self._nvar = nop
        if self._pmut_eval_flag == 0:
            self._pmut_eval = <FLOAT *>stdlib.malloc(sizeof(FLOAT) * l_st * nfunc)
            self._pmut_eval_flag = 1
        pmut_eval = self._pmut_eval
        self._x = <FLOAT *>stdlib.malloc(sizeof(FLOAT) * l_st * self._max_nargs)
        self._st = <FLOAT *>stdlib.malloc(sizeof(FLOAT) * l_st * (self._max_nargs + 1))
        self._ind = <INT *>stdlib.malloc(sizeof(INT) * (self._max_nargs + 1))
        for i in range(nop):
            self._ind[i + 1] = nfunc + i
            skip = indexC[i] * l_st
            for j in range(l_st):
                self._x[j*self._nvar + i] = stC[skip + j]
                c += 1
        c = 0
        stC = self._st
        for i in range(nfunc):
            if _nop[i] == nop:
                self._ind[0] = i
                self._pos = 0
                self.eval_ind_inner()
                for j in range(l_st):
                    pmut_eval[c] = stC[j]
                    c += 1
        self._nvar = nvar
        stdlib.free(self._x)
        stdlib.free(self._st)
        stdlib.free(self._ind)

    def pmutation_eval_test(self, int nop, npc.ndarray[FLOAT, ndim=2] st):
        cdef int i, j, c=0
        cdef FLOAT *stC = <FLOAT*>st.data, *pmut_eval=self._pmut_eval
        for i in range(self._nfunc):
            if self._nop[i] == nop:
                for j in range(self._l_st):
                    if npy_isinf(pmut_eval[c]) and npy_isinf(stC[c]):
                        c += 1
                        continue
                    if npy_isnan(pmut_eval[c]) and npy_isnan(stC[c]):
                        c += 1
                        continue
                    if (pmut_eval[c] - stC[c]) != 0:
                        print i, c, pmut_eval[c], stC[c]
                        return False
                    c += 1
        return True
