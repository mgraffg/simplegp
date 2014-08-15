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
# cython: profile=True
cimport numpy as npc
import numpy as np
cimport cython
cimport libc
cimport libc.math as math
cimport libc.stdlib as stdlib
from .tree import Tree
from .tree cimport Tree

np.seterr(all='ignore')

cdef class PDE:
    def __cinit__(self, Tree tree,
                  npc.ndarray[FLOAT, ndim=2, mode="c"] _p_st):
        cdef int max_length = tree._max_length
        self._tree = tree
        self._parent = <INT *>stdlib.malloc(sizeof(INT)*max_length)
        self._path = <INT *>stdlib.malloc(sizeof(INT)*max_length)
        self._p_st = <FLOAT *>_p_st.data

    def __dealloc__(self):
        stdlib.free(self._parent)
        stdlib.free(self._path)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int compute_pdepm(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                            npc.ndarray[FLOAT, ndim=2, mode="c"] _st,
                            npc.ndarray[INT, ndim=1, mode="c"] index,
                            float ppm,
                            int only_functions=0):
        cdef int pos=0, end=ind.shape[0], i, c=0
        cdef INT *indC = <INT *>ind.data, *indexC = <INT *> index.data
        self._l_st = _st.shape[1]
        self._ind = indC
        self._st = <FLOAT *>_st.data
        self._tree.set_pos(0)
        self._tree.compute_parents_inner(self._ind, self._parent, -1)
        while pos < end:
            if (1 == only_functions and indC[pos] >= self._tree._nfunc)\
               or np.random.rand() >= ppm:
                pos += 1
                continue
            self._end = self._tree.path_to_root_inner(self._parent, self._path,
                                                      pos)
            self.compute_inner()
            index[c] = pos
            c += 1
            pos = self._tree.traverse_inner(indC, pos)
        return c

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int compute_constants(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                                npc.ndarray[FLOAT, ndim=2, mode="c"] _st):
        cdef int pos, ind_size=ind.shape[0]
        cdef INT *indC = <INT *>ind.data
        self._l_st = _st.shape[1]
        self._ind = indC
        self._st = <FLOAT *>_st.data
        self._tree.set_pos(0)
        self._tree.compute_parents_inner(self._ind, self._parent, -1)
        for pos in range(ind_size):
            if self._tree.isconstant(indC[pos]):
                self._end = self._tree.path_to_root_inner(self._parent, self._path,
                                                          pos)
                self.compute_inner()
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int compute(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                      int pos,
                      npc.ndarray[FLOAT, ndim=2, mode="c"] _st):
        self._l_st = _st.shape[1]
        self._ind = <INT *>ind.data
        self._st = <FLOAT *>_st.data
        self._tree.set_pos(0)
        self._tree.compute_parents_inner(self._ind, self._parent, -1)
        self._end = self._tree.path_to_root_inner(self._parent, self._path,
                                                  pos)
        return self.compute_inner()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        for i in range(l_st):
            p_st[oi+i] = p_st[ii+i]
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void subtract(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        if self.is_first_var(_i):
            for i in range(l_st):
                p_st[oi+i] = p_st[ii+i]
        else:
            for i in range(l_st):
                p_st[oi+i] = -p_st[ii+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multiply(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        if self.is_first_var(_i):
            v = self._tree.get_pos_arg_inner(self._ind,
                                             path[_i],
                                             1) * l_st
        else:
            v = (path[_i] + 1) * l_st
        for i in range(l_st):
            p_st[oi+i] = p_st[ii+i] * st[v+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void divide(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        v = (path[_i] + 1) * l_st
        v2 = self._tree.get_pos_arg_inner(self._ind,
                                          path[_i],
                                          1) * l_st
        if self.is_first_var(_i):
            for i in range(l_st):
                p_st[oi+i] = p_st[ii+i] / st[v2+i]
        else:
            for i in range(l_st):
                p_st[oi+i] = - p_st[ii+i] * st[v+i] / math.pow(st[v2+i], 2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void fabs(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        v = (path[_i] + 1) * l_st
        for i in range(l_st):
            if st[v+i] < 0:
                p_st[oi+i] = -p_st[ii+i]
            else:
                p_st[oi+i] = p_st[ii+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void exp(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        for i in range(l_st):
            p_st[oi+i] = p_st[ii+i] * st[ii+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void sqrt(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        for i in range(l_st):
            p_st[oi+i] = 0.5 * p_st[ii+i] / st[ii+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void sin(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        v = (path[_i] + 1) * l_st
        for i in range(l_st):
            p_st[oi+i] = p_st[ii+i] * math.cos(st[v+i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cos(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        v = (path[_i] + 1) * l_st
        for i in range(l_st):
            p_st[oi+i] = - p_st[ii+i] * math.sin(st[v+i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void sigmoid(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        for i in range(l_st):
            p_st[oi+i] = p_st[ii+i] * st[ii+i] * (1 - st[ii+i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void if_func(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, a_i, b_i, c_i, pvar
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st, x, y, z, s
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        a_i = (path[_i] + 1) * l_st
        b_i = self._tree.get_pos_arg_inner(self._ind, path[_i], 1) * l_st
        c_i = self._tree.get_pos_arg_inner(self._ind, path[_i], 2) * l_st
        pvar = self.which_var(path[_i], path[_i+1])
        if pvar == 1:
            for i in range(l_st):
                x = st[a_i + i]
                y = st[b_i + i]
                z = st[c_i + i]
                s = 1 / (1 + math.exp(-100 * x))
                p_st[oi+i] = p_st[ii+i] * ((y - z) * s * (1 - s))
        elif pvar == 2:
            for i in range(l_st):
                x = st[a_i + i]
                s = 1 / (1 + math.exp(-100 * x))
                p_st[oi+i] = p_st[ii+i] * s
        elif pvar == 3:
            for i in range(l_st):
                x = st[a_i + i]
                s = 1 / (1 + math.exp(-100 * x))
                p_st[oi+i] = p_st[ii+i] * (1 - s)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void max(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, a_i, b_i
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st, x, y, s
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        a_i = (path[_i] + 1) * l_st
        b_i = self._tree.get_pos_arg_inner(self._ind, path[_i], 1) * l_st
        pvar = self.which_var(path[_i], path[_i+1])
        if pvar == 1:
            for i in range(l_st):
                x = st[a_i + i]
                y = st[b_i + i]
                s = 1 / (1 + math.exp(-100 * (x-y)))
                p_st[oi+i] = p_st[ii+i] * ((x - y) * s * (1 - s) + s)
        elif pvar == 2:
            for i in range(l_st):
                x = st[a_i + i]
                y = st[b_i + i]
                s = 1 / (1 + math.exp(-100 * (x-y)))
                p_st[oi+i] = p_st[ii+i] * ((x - y) * s * (s - 1) + 1 - s)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void min(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, a_i, b_i
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st, x, y, s
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        a_i = (path[_i] + 1) * l_st
        b_i = self._tree.get_pos_arg_inner(self._ind, path[_i], 1) * l_st
        pvar = self.which_var(path[_i], path[_i+1])
        if pvar == 1:
            for i in range(l_st):
                x = st[a_i + i]
                y = st[b_i + i]
                s = 1 / (1 + math.exp(-100 * (x-y)))
                p_st[oi+i] = p_st[ii+i] * ((y - x) * s * (1 - s) + 1 - s)
        elif pvar == 2:
            for i in range(l_st):
                x = st[a_i + i]
                y = st[b_i + i]
                s = 1 / (1 + math.exp(-100 * (x-y)))
                p_st[oi+i] = p_st[ii+i] * ((y - x) * s * (s - 1) + s)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void ln(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        v = (path[_i] + 1) * l_st
        for i in range(l_st):
            p_st[oi+i] = p_st[ii+i] / st[v+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void sq(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        v = (path[_i] + 1) * l_st
        for i in range(l_st):
            p_st[oi+i] = 2 * p_st[ii+i] * st[v+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void output(self, int _i):
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void argmax(self, int _i):
        cdef int i, ii, oi, l_st = self._l_st, v, v2, nargs, j
        cdef int *pos
        cdef INT *path = self._path
        cdef FLOAT *p_st = self._p_st, *st = self._st
        cdef FLOAT sup, sdown, up, down, beta, xi, c
        nargs = self._tree._nop[16]
        pos = <int *> stdlib.malloc(sizeof(int)*nargs)
        beta = 2.
        ii = path[_i] * l_st
        oi = path[_i+1] * l_st
        for j in range(nargs):
            pos[j] = self._tree.get_pos_arg_inner(self._ind, path[_i], j) * l_st
        for i in range(l_st):
            sup = 0
            sdown = 0
            for j in range(nargs):
                xi = st[pos[j]+i]
                down = math.exp(beta * xi)
                up = j * down
                sdown += down
                sup += up
            xi = st[oi+i]
            down = math.exp(beta * xi)
            up = (j * beta * beta * down) / sdown 
            c = - sup / (sdown * sdown)
            c = c * beta * beta * down
            p_st[oi+i] = p_st[ii+i] * (up + c)
        stdlib.free(pos)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int compute_inner(self):
        cdef int _i, node
        cdef INT *path = self._path, *ind=self._ind
        for _i in range(self._end-1):
            node = ind[path[_i]]
            if node == 0:
                self.add(_i)
            elif node == 1:
                self.subtract(_i)
            elif node == 2:
                self.multiply(_i)
            elif node == 3:
                self.divide(_i)
            elif node == 4:
                self.fabs(_i)
            elif node == 5:
                self.exp(_i)
            elif node == 6:
                self.sqrt(_i)
            elif node == 7:
                self.sin(_i)
            elif node == 8:
                self.cos(_i)
            elif node == 9:
                self.sigmoid(_i)
            elif node == 10:
                self.if_func(_i)
            elif node == 11:
                self.max(_i)
            elif node == 12:
                self.min(_i)
            elif node == 13:
                self.ln(_i)
            elif node == 14:
                self.sq(_i)
            elif node == 15:
                self.output(_i)
            elif node == 16:
                self.argmax(_i)
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int is_first_var(self, int pos):
        if self._path[pos] + 1 == self._path[pos + 1]:
            return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int which_var(self, int parent, int pos):
        cdef int var = 0
        cdef INT *p = self._parent
        while pos > parent:
            if p[pos] == parent:
                var += 1
            pos -= 1
        return var

