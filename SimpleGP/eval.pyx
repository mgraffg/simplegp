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
        # whether to compute the derivative
        self._p_der = 0
        # setting the maximum number of arguments
        self._max_nargs = max_nargs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_p_der_st(self, npc.ndarray[FLOAT, ndim=2, mode="c"] _st):
        """Set the partial derivative stack.
        Let (r, c) be the shape of self._st then the
        shape of self._p_der_st is (r, max_nargs * c)."""
        self._p_der_st = <FLOAT *> _st.data
        self._p_der = 1

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
        if self._p_der:
            for i in range(l):
                if st[a_i + i] == 0:
                    self._p_der_st[po_i + i] = 0
                elif st[a_i + i] < 0:
                    self._p_der_st[po_i + i] = -1
                else:
                    self._p_der_st[po_i + i] = 1
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = st[o_i + i]
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = 1 / (2 * st[o_i + i])
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = math.cos(st[a_i + i])
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = - math.sin(st[a_i + i])
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = st[o_i + i] * (1 - st[o_i + i])
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = 1 / st[a_i + i]
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = 2 * st[a_i + i]
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
            self.max(a, b, pos)
        elif func == 12:
            self.min(a, b, pos)

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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = 1
                self._p_der_st[po_i + i + l] = 1
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = 1
                self._p_der_st[po_i + i + l] = -1
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = st[b_i + i]
                self._p_der_st[po_i + i + l] = st[a_i + i]
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
        if self._p_der:
            for i in range(l):
                self._p_der_st[po_i + i] = 1 / st[b_i + i]
                self._p_der_st[po_i + i + l] = - st[a_i + i] / (st[b_i + i] * st[b_i + i])
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT max(self, 
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
        if self._p_der:
            for i in range(l):
                x = st[a_i + i]
                y = st[b_i + i]
                s = 1 / (1 + math.exp(-100 * (x-y)))
                self._p_der_st[po_i + i] = (x - y) * s * (1 - s) + s
                self._p_der_st[po_i + i + l] = (x - y) * s * (s - 1) + 1 - s
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef INT min(self, 
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
        if self._p_der:
            for i in range(l):
                x = st[a_i + i]
                y = st[b_i + i]
                s = 1 / (1 + math.exp(-100 * (x-y)))
                self._p_der_st[po_i + i] = (y - x) * s * (1 - s) + 1 - s
                self._p_der_st[po_i + i + l] = (y - x) * s * (s - 1) + s
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
        if self._p_der:
            for i in range(l):
                x = st[a_i + i]
                y = st[b_i + i]
                z = st[c_i + i]
                s = 1 / (1 + math.exp(-100 * x))
                self._p_der_st[po_i + i] = (y - z) * s * (1 - s)
                self._p_der_st[po_i + i + l] = s
                self._p_der_st[po_i + i + l + l] = 1-s

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
        if self._p_der:
            for j in range(nargs):
                po_i = j * l
                for i in range(l):
                    self._p_der_st[po_i + i] = 1

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
        if self._p_der:
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
                for j in range(nargs):
                    xi = st[l * a[j] + i]
                    down = math.exp(beta * xi)
                    up = (j * beta * beta * down) / sdown 
                    c = - sup / (sdown * sdown)
                    c = c * beta * beta * down
                    self._p_der_st[po_i + i + j*l] = up + c
        else:
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
    
