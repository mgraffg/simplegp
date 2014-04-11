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
cimport libc.math as math
cimport libc.stdlib as stdlib

np.seterr(all='ignore')

cdef class Tree:
    cdef INT *_father1
    cdef INT *_father2
    cdef INT *_length
    cdef INT *_nop
    cdef INT *_m
    cdef INT _p1
    cdef INT _p2
    cdef INT _ncons
    cdef INT _nfunc
    cdef INT _nvar
    cdef int _pos
    cdef int _min_length
    cdef int _max_length
    cdef int __length_p1
    cdef int __f1_end
    cdef int _select_root
    def __cinit__(self, npc.ndarray[INT, ndim=1, mode="c"] nop,
                  npc.ndarray[INT, ndim=1, mode="c"] _length,
                  npc.ndarray[INT, ndim=1, mode="c"] _mask,
                  int min_length, int max_length, int select_root=1):
        self._nop = <INT *>nop.data
        self._length = <INT *>_length.data
        self._m = <INT *> _mask.data
        self._nfunc = nop.shape[0]
        self._min_length = min_length
        self._max_length = max_length
        self._select_root = select_root

    def get_select_root(self):
        return self._select_root

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_nvar(self, int a):
        self._nvar = a

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT isfunc(self, INT a):
        if a < self._nfunc: return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT isvar(self, INT a):
        if a < self._nfunc: return 0
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT isconstant(self, INT a):
        if a < self._nfunc + self._nvar: return 0
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef INT equal_non_const(self,
                              npc.ndarray[INT, ndim=1, mode="c"] ind1,
                              npc.ndarray[INT, ndim=1, mode="c"] ind2):
        cdef INT pos = 0
        cdef INT *ind1C = <INT*> ind1.data
        cdef INT *ind2C = <INT*> ind2.data
        cdef INT end = ind1.shape[0]
        if ind2.shape[0] != end:
            return 0
        for pos in range(end):
            if self.isconstant(ind1C[pos]) and self.isconstant(ind2C[pos]):
                continue
            if ind1C[pos] != ind2C[pos]:
                return 0
        return 1

    def any_constant(self, npc.ndarray[INT, ndim=1, mode="c"] ind):
        cdef int i, l = ind.shape[0]
        cdef INT *c = <INT *>ind.data
        for i in range(l):
            if self.isconstant(c[i]):
                return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int fix(self, int node, int ncons):
        if self.isconstant(node):
            return node + ncons
        return node

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int crossover_mask(self,
                             npc.ndarray[INT, ndim=1, mode="c"] father1,
                             npc.ndarray[INT, ndim=1, mode="c"] father2,
                             int p1):
        cdef int f2_end, i, f1_end, l_p1, r, minl, maxl, c=0
        cdef INT *l, *m
        f1_end = self.traverse(father1, pos=p1)
        f2_end = self.length(father2)
        l_p1 = father1.shape[0] - (f1_end - p1)
        self.__length_p1 = l_p1
        self.__f1_end = f1_end
        l = self._length
        m = self._m
        minl = self._min_length
        maxl = self._max_length
        for i in range(f2_end):
            r = l_p1 + l[i] 
            if r >= minl and r <= maxl:
                m[i] = 1
                c += 1
            else:
                m[i] = 0
        if self._select_root == 0:
            m[0] = 0
        return c 

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int father2_crossing_point(self,
                                     npc.ndarray[INT, ndim=1, mode="c"] father1,
                                     npc.ndarray[INT, ndim=1, mode="c"] father2,
                                     int p1):
        cdef int c = self.crossover_mask(father1, father2, p1)
        cdef int f2_end = father2.shape[0]
        cdef INT *m
        m = self._m
        c = np.random.randint(c)
        for i in range(f2_end):
            if m[i] == 1:
                c -= 1
            if c == -1:
                return i
        return c

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef crossover(self, npc.ndarray[INT, ndim=1, mode="c"] father1,
                    npc.ndarray[INT, ndim=1, mode="c"] father2,
                    int ncons=2,
                    int p1=-1,
                    int p2=-2):
        cdef npc.ndarray[INT, ndim=1, mode="c"] son
        cdef INT *sonC, *f1, *f2
        cdef int l_p1, l_p2, i, c=0, f1_end
        f1 = <INT *> father1.data
        f2 = <INT *> father2.data
        if p1 < 0:
            if self._select_root:
                p1 = np.random.randint(father1.shape[0])
            else:
                p1 = np.random.randint(father1.shape[0]-1) + 1
        if p2 < 0:
            p2 = self.father2_crossing_point(father1, father2, p1)
            l_p1 = self.__length_p1
            f1_end = self.__f1_end
            l_p2 = self._length[p2]
        else:
            f1_end = self.traverse(father1, pos=p1)
            l_p1 = father1.shape[0] - (f1_end - p1)
            l_p2 = self.traverse(father2, pos=p2) - p2
        son = np.empty(l_p1 + l_p2, dtype=father1.dtype)
        sonC = <INT *> son.data
        for i in range(p1):
            sonC[c] = f1[i]
            c += 1
        for i in range(p2, p2+l_p2):
            sonC[c] = self.fix(f2[i], ncons)
            c += 1
        for i in range(f1_end, father1.shape[0]):
            sonC[c] = f1[i]
            c += 1            
        return son

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int traverse(self,
                       npc.ndarray[INT, ndim=1, mode="c"] ind, 
                       INT pos=0):
        cdef INT nargs = 1
        cdef INT cnt = 0
        cdef INT pos_ini = pos
        cdef INT ele
        cdef INT shape = self._nfunc
        cdef INT *indC = <INT*> ind.data
        cdef INT *_funcC = self._nop
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int length(self,
                     npc.ndarray[INT, ndim=1, mode="c"] ind):
        self._father1 = <INT *> ind.data
        self._pos = 0
        self.compute_length()
        return self._pos
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int compute_length(self):
        cdef int node, i, l=1
        cdef int opos = self._pos
        node = self._father1[self._pos]
        self._pos += 1
        if self.isfunc(node):
            for i in range(self._nop[node]):
                l += self.compute_length()
        self._length[opos] = l
        return l

cdef class SubTree(Tree):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int get_subtree(self, npc.ndarray[INT, ndim=1, mode="c"] father1,
                          int p1):
        cdef INT *fC = <INT *>father1.data, ele
        cdef INT *_nop = self._nop
        cdef int s = 0, i
        cdef int nargs = 1
        for i in range(1, p1+1):
            ele = fC[i]
            if self.isfunc(ele):
                nargs -= 1
                nargs += _nop[ele]
            else:
                nargs -= 1
            if nargs == 0 and i < p1:
                s += 1
                nargs = 1
        return s

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int crossover_mask(self,
                             npc.ndarray[INT, ndim=1, mode="c"] father1,
                             npc.ndarray[INT, ndim=1, mode="c"] father2,
                             int p1):
        cdef int f2_end, i, f1_end, l_p1, r, minl, maxl, c=0, subtree=0, nargs=1
        cdef int sub2=0
        cdef INT *l, *m, *fC = <INT *> father2.data, ele
        cdef INT *_nop = self._nop
        subtree = self.get_subtree(father1, p1)
        f1_end = self.traverse(father1, pos=p1)
        f2_end = self.length(father2)
        l_p1 = father1.shape[0] - (f1_end - p1)
        self.__length_p1 = l_p1
        self.__f1_end = f1_end
        l = self._length
        m = self._m
        minl = self._min_length
        maxl = self._max_length
        m[0] = 0
        for i in range(1, f2_end):
            if subtree != sub2:
                m[i] = 0
            else:
                r = l_p1 + l[i]
                if r >= minl and r <= maxl:
                    m[i] = 1
                    c += 1
                else:
                    m[i] = 0
            ele = fC[i]
            if self.isfunc(ele):
                nargs -= 1
                nargs += _nop[ele]
            else:
                nargs -= 1
            if nargs == 0:
                sub2 += 1
                nargs = 1
        return c 
    


cdef class PDEXO(Tree):
    cdef INT8 *_xo_error
    cdef FLOAT *_xo_x, *_xo_s
    cdef int _xo_c
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def father2_xp_extras(self,
                          npc.ndarray[INT8, ndim=2, mode="c"] error,
                          npc.ndarray[FLOAT, ndim=2, mode="c"] x,
                          npc.ndarray[FLOAT, ndim=2, mode="c"] s):
        self._xo_error = <INT8 *> error.data
        self._xo_x = <FLOAT *> x.data
        self._xo_s = <FLOAT *> s.data
        self._xo_c = s.shape[1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int father2_crossing_point(self,
                                     npc.ndarray[INT, ndim=1, mode="c"] father1,
                                     npc.ndarray[INT, ndim=1, mode="c"] father2,
                                     int p1):
        cdef int f2_end = father2.shape[0], i, j, j1, c
        cdef unsigned int flag, bflag=0, res=0
        cdef INT *m, *_length
        cdef INT8 *error
        cdef FLOAT *x, *s
        self.crossover_mask(father1, father2, p1)
        m = self._m
        c = self._xo_c
        error = & self._xo_error[p1 * c]
        x = & self._xo_x[p1 * c]
        s = self._xo_s
        _length = self._length
        for i in range(f2_end):
            if m[i] == 0:
                continue
            j1 = i * c
            flag = 0
            for j in range(c):
                # p2[m] = ((s[m] > x) == error).sum(axis=1)
                if s[j1] > x[j]:
                    if error[j] == -1:
                        flag += 1
                else:
                    if error[j] == 1:
                        flag += 1
                j1 += 1
            if flag > bflag:
                res = i
                bflag = flag
            elif flag == bflag and _length[i] < _length[res]:
                res = i
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int father2_xo_point_super(self,
                                     npc.ndarray[INT, ndim=1, mode="c"] father1,
                                     npc.ndarray[INT, ndim=1, mode="c"] father2,
                                     int p1):
        return super(PDEXO, self).father2_crossing_point(father1,
                                                         father2,
                                                         p1)


cdef class PDEXOSubtree(PDEXO):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int get_subtree(self, npc.ndarray[INT, ndim=1, mode="c"] father1,
                          int p1):
        cdef INT *fC = <INT *>father1.data, ele
        cdef INT *_nop = self._nop
        cdef int s = 0, i
        cdef int nargs = 1
        for i in range(1, p1+1):
            ele = fC[i]
            if self.isfunc(ele):
                nargs -= 1
                nargs += _nop[ele]
            else:
                nargs -= 1
            if nargs == 0 and i < p1:
                s += 1
                nargs = 1
        return s

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int crossover_mask(self,
                             npc.ndarray[INT, ndim=1, mode="c"] father1,
                             npc.ndarray[INT, ndim=1, mode="c"] father2,
                             int p1):
        cdef int f2_end, i, f1_end, l_p1, r, minl, maxl, c=0, subtree=0, nargs=1
        cdef int sub2=0
        cdef INT *l, *m, *fC = <INT *> father2.data, ele
        cdef INT *_nop = self._nop
        subtree = self.get_subtree(father1, p1)
        f1_end = self.traverse(father1, pos=p1)
        f2_end = self.length(father2)
        l_p1 = father1.shape[0] - (f1_end - p1)
        self.__length_p1 = l_p1
        self.__f1_end = f1_end
        l = self._length
        m = self._m
        minl = self._min_length
        maxl = self._max_length
        m[0] = 0
        for i in range(1, f2_end):
            if subtree != sub2:
                m[i] = 0
            else:
                r = l_p1 + l[i]
                if r >= minl and r <= maxl:
                    m[i] = 1
                    c += 1
                else:
                    m[i] = 0
            ele = fC[i]
            if self.isfunc(ele):
                nargs -= 1
                nargs += _nop[ele]
            else:
                nargs -= 1
            if nargs == 0:
                sub2 += 1
                nargs = 1
        return c 
