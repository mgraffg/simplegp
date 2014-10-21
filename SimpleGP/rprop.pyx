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
ctypedef npc.int8_t INT8
cimport libc
cimport libc.stdlib as stdlib
cimport libc.math as math
from .tree import Tree
from .tree cimport Tree
cdef extern from "math.h":
    float sqrt(double)

np.seterr(all='ignore')

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RPROP:
    cdef INT * _ind
    cdef FLOAT * _constants
    cdef INT * _nop
    cdef INT _nfunc
    cdef INT _nvar
    cdef FLOAT *_p_der_st
    cdef FLOAT *_error_st
    cdef FLOAT *_prev_step
    cdef FLOAT *_prev_slope
    cdef FLOAT _increase_factor
    cdef FLOAT _decrease_factor
    cdef FLOAT _delta_min
    cdef FLOAT _delta_max
    cdef INT _pos
    cdef INT _l_st
    cdef INT _update_constants
    cdef INT _max_nargs
    def __init__(self,
                 npc.ndarray[INT, ndim=1] ind,
                 npc.ndarray[FLOAT, ndim=1] constants,
                 npc.ndarray[INT, ndim=1] nop,
                 int nvar,
                 npc.ndarray[FLOAT, ndim=2, mode="c"] p_der_st,
                 int l_st,
                 npc.ndarray[FLOAT, ndim=1] prev_step,
                 npc.ndarray[FLOAT, ndim=1] prev_slope,
                 float increase_factor=1.2,
                 float decrease_factor=0.5,
                 float delta_min=0.0,
                 float delta_max=50.0,
                 int update_constants=1,
                 int max_nargs=3):
        self._ind = <INT *> ind.data
        self._constants = <FLOAT *> constants.data
        self._nop = <INT *> nop.data
        self._nfunc = nop.shape[0]
        self._nvar = nvar
        self._p_der_st = <FLOAT *> p_der_st.data
        self._l_st = l_st
        self._prev_step = <FLOAT *> prev_step.data
        self._prev_slope = <FLOAT *> prev_slope.data
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._delta_min = delta_min
        self._delta_max = delta_max
        self._update_constants = update_constants
        self._max_nargs = max_nargs

    def set_error_st(self, npc.ndarray[FLOAT, ndim=2, mode="c"] error_st):
        self._error_st = <FLOAT *> error_st.data

    def set_zero_pos(self):
        self._pos = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef INT update_constants_rprop(self):
        cdef int _l_st = self._l_st, i, j
        cdef int ele = self._ind[self._pos]
        cdef int pos = self._pos, init, end, c
        cdef int o_i, po_i = _l_st * pos
        cdef int ppo_i = self._max_nargs * _l_st * pos
        cdef float prev_step, prev_slope, slope, next_step, tmp
        self._pos += 1
        if ele < self._nfunc:
            nargs = self._nop[ele]
            init = 0
            for i in range(nargs):
                end = init + _l_st
                o_i = _l_st * self._pos
                for j in range(_l_st):
                    self._error_st[o_i + j] = self._error_st[po_i + j] *\
                                              self._p_der_st[ppo_i + init + j]
                self.update_constants_rprop()
                init = end
            return 0
        elif ele < self._nfunc + self._nvar:
            return 0
        else:
            if self._update_constants == 0:
                return 0
            c = ele - self._nfunc - self._nvar
            if self._prev_step[pos] > 0.0001:
                prev_step = self._prev_step[pos]
            else:
                prev_step = 0.0001
            prev_slope = self._prev_slope[pos]
            slope = 0
            for i in range(_l_st):
                slope += self._error_st[po_i + i]
            if (prev_slope * slope) >= 0:
                tmp = prev_step * self._increase_factor
                next_step = tmp if tmp < self._delta_max else self._delta_max
            else:
                tmp = prev_step * self._decrease_factor
                next_step = tmp if tmp > self._delta_min else self._delta_min
            if slope > 0:
                self._constants[c] -= next_step
            elif slope < 0:
                self._constants[c] += next_step
            self._prev_step[pos] = next_step
            self._prev_slope[pos] = slope
            return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RPROP2:
    cdef int _l_st
    cdef float *_prev_step
    cdef float *_prev_slope
    cdef float _increase_factor
    cdef float _decrease_factor
    cdef float _delta_min
    cdef float _delta_max
    cdef Tree _tree
    cdef int _ncons
    cdef int *_cons, *_pos
    cdef FLOAT *_p_st
    cdef FLOAT *_constants
    def __init__(self,
                 npc.ndarray[INT, ndim=1, mode="c"] ind,
                 npc.ndarray[FLOAT, ndim=1, mode="c"] constants,
                 npc.ndarray[FLOAT, ndim=2, mode="c"] p_st,
                 Tree tree,
                 float increase_factor=1.2,
                 float decrease_factor=0.5,
                 float delta_min=0.0,
                 float delta_max=50.0):
        cdef int i, ncons=0
        cdef INT *indC = <INT *> ind.data
        self._constants = <FLOAT *> constants.data
        self._p_st = <FLOAT *> p_st.data
        self._tree = tree
        self._l_st = p_st.shape[1]
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._delta_min = delta_min
        self._delta_max = delta_max
        for i in range(ind.shape[0]):
            if self._tree.isconstant(indC[i]):
                ncons += 1
        self._ncons = ncons
        self._cons = <int *> stdlib.malloc(sizeof(int)*ncons)
        self._pos = <int *> stdlib.malloc(sizeof(int)*ncons)
        ncons = 0
        for i in range(ind.shape[0]):
            if self._tree.isconstant(indC[i]):
                self._pos[ncons] = i
                self._cons[ncons] = indC[i] - self._tree._nfunc - self._tree._nvar
                ncons += 1
        self._prev_step = <float *> stdlib.malloc(sizeof(FLOAT)*ncons)
        self._prev_slope = <float *> stdlib.malloc(sizeof(FLOAT)*ncons)
        for i in range(self._ncons):
            self._prev_step[i] = 0
            self._prev_slope[i] = 0

    def __dealloc__(self):
        stdlib.free(self._cons)
        stdlib.free(self._pos)
        stdlib.free(self._prev_step)
        stdlib.free(self._prev_slope)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int update_constants_rprop(self):
        cdef int pos, ii, _l_st=self._l_st, c, i
        cdef float prev_slope, prev_step, slope, tmp, next_step
        for pos in range(self._ncons):
            if self._prev_step[pos] > 0.0001:
                prev_step = self._prev_step[pos]
            else:
                prev_step = 0.0001
            prev_slope = self._prev_slope[pos]
            slope = 0
            ii = self._pos[pos] * _l_st
            for i in range(_l_st):
                if self._p_st[ii+i] > 0:
                    slope += 1
                elif self._p_st[ii+i] < 0:
                    slope -= 1
            if (prev_slope * slope) >= 0:
                tmp = prev_step * self._increase_factor
                next_step = tmp if tmp < self._delta_max else self._delta_max
            else:
                tmp = prev_step * self._decrease_factor
                next_step = tmp if tmp > self._delta_min else self._delta_min
            c = self._cons[pos]
            if slope > 0:
                self._constants[c] = self._constants[c] - next_step
            elif slope < 0:
                self._constants[c] = self._constants[c] + next_step
            self._prev_step[pos] = next_step
            self._prev_slope[pos] = slope
        # print "antes de salir"
        return 1
