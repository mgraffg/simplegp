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

cdef class Tree:
    def __cinit__(self, npc.ndarray[INT, ndim=1, mode="c"] nop,
                  npc.ndarray[INT, ndim=1, mode="c"] _length,
                  npc.ndarray[INT, ndim=1, mode="c"] _mask,
                  int min_length, int max_length, int select_root=1,
                  int type_xpoint_selection=0):
        self._nop = <INT *>nop.data
        self._length = <INT *>_length.data
        self._m = <INT *> _mask.data
        self._nfunc = nop.shape[0]
        self._min_length = min_length
        self._max_length = max_length
        self._select_root = select_root
        self._depth = <INT *>stdlib.malloc(sizeof(INT)*max_length)
        self._hist = <INT *>stdlib.malloc(sizeof(INT)*max_length)
        self._type_xpoint_selection = type_xpoint_selection
        
    def __dealloc__(self):
        stdlib.free(self._depth)
        stdlib.free(self._hist)

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
        if a >= self._nfunc and a < (self._nfunc + self._nvar):
            return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT isconstant(self, INT a):
        if a < self._nfunc + self._nvar: return 0
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef set_pos(self, int pos):
        self._pos = pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int get_pos_arg(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                          int pos,
                          int narg):
        return self.get_pos_arg_inner(<INT *>ind.data, pos, narg)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int get_pos_arg_inner(self, INT *ind,
                               int pos,
                               int narg):
        cdef int i
        pos += 1
        for i in range(narg):
            pos = self.traverse_inner(ind, pos)
        return pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int path_to_root(self, npc.ndarray[INT, ndim=1, mode="c"] parent,
                           npc.ndarray[INT, ndim=1, mode="c"] path,
                           int pos):
        return self.path_to_root_inner(<INT*>parent.data,
                                       <INT*>path.data,
                                       pos)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int path_to_root_inner(self, INT *parentC,
                                INT *pathC,
                                int pos):
        cdef int c = 0, i, j, tmp
        while pos >= 0:
            pathC[c] = pos
            c += 1
            pos = parentC[pos]
        for i in range(c/2):
            tmp = pathC[i]
            pathC[i] = pathC[c -i -1]
            pathC[c -i -1] = tmp
        return c

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int compute_depth(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                            npc.ndarray[INT, ndim=1, mode="c"] depth):
        self._pos = 0
        self.compute_depth_inner(<INT*> ind.data,
                                 <INT*> depth.data,
                                 0)
        return self._pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void compute_depth_inner(self, INT *ind,
                                  INT *depth,
                                  int p):
        cdef int pos = self._pos, j, nop
        depth[pos] = p
        self._pos += 1
        if self.isfunc(ind[pos]):
            nop = self._nop[ind[pos]]
            for j in range(nop):
                self.compute_depth_inner(ind, depth, p+1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int compute_depth_histogram(self, npc.ndarray[INT, ndim=1, mode="c"] depth,
                                      npc.ndarray[INT, ndim=1, mode="c"] hist,
                                      int end):
        return self.compute_depth_histogram_inner(<INT*> depth.data,
                                                  <INT*> hist.data,
                                                  end)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int compute_depth_histogram_inner(self, INT *depthC,
                                           INT *histC,
                                           int end):
        cdef int max=-1, i, tmp
        for i in range(end):
            histC[i] = 0
        for i in range(end):
            tmp = depthC[i]
            if tmp > max:
                max = tmp
            histC[tmp] += 1
        return max + 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int select_xpoint_depth(self, npc.ndarray[INT, ndim=1, mode="c"] ind):
        cdef int end, p1, p2, i
        self._pos = 0
        cdef INT *depth
        depth = self._depth
        self.compute_depth_inner(<INT *>ind.data, depth, 0)
        end = self.compute_depth_histogram_inner(depth, self._hist, self._pos)
        if self._select_root:
            p1 = np.random.randint(end)
        else:
            p1 = np.random.randint(end-1) + 1
        p2 = np.random.randint(self._hist[p1])
        for i in range(self._pos):
            if depth[i] == p1:
                if p2 == 0:
                    return i
                else:
                    p2 -= 1
        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int select_xpoint_uniform(self, npc.ndarray[INT, ndim=1, mode="c"] ind):
        cdef int p1, s = ind.shape[0]
        if self._select_root:
            p1 = np.random.randint(s)
        else:
            p1 = np.random.randint(s - 1) + 1
        return p1

    def set_type_xpoint_selection(self, int t):
        self._type_xpoint_selection = t

    def get_type_xpoint_selection(self):
        return self._type_xpoint_selection

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int father1_crossing_point(self, npc.ndarray[INT, ndim=1, mode="c"] ind):
        if self._type_xpoint_selection == 0:
            return self.select_xpoint_uniform(ind)
        elif self._type_xpoint_selection == 1:
            return self.select_xpoint_depth(ind)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef compute_parents(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                          npc.ndarray[INT, ndim=1, mode="c"] parent):
        self._pos = 0
        self.compute_parents_inner(<INT*>ind.data,
                                   <INT*>parent.data, -1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef compute_parents_inner(self, INT *ind,
                               INT *parent,
                               int p):
        cdef int opos = self._pos, j
        cdef INT *nop = self._nop
        parent[opos] = p
        self._pos += 1
        if self.isfunc(ind[opos]):
            for j in range(nop[ind[opos]]):
                self.compute_parents_inner(ind, parent, opos)
    
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
            p1 = self.father1_crossing_point(father1)
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
        return self.traverse_inner(<INT *>ind.data, pos)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int traverse_inner(self, INT *indC, INT pos):
        cdef INT nargs = 1
        cdef INT cnt = 0
        cdef INT pos_ini = pos
        cdef INT ele
        cdef INT shape = self._nfunc
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

    def get_sons_test(self, npc.ndarray[INT, ndim=1, mode="c"] ind, int pos,
                 npc.ndarray[INT, ndim=1, mode="c"] sons):
        cdef INT *sonsC, *indC, i
        sonsC = self.get_sons_inner(<INT *>ind.data, pos)
        for i in range(sons.shape[0]):
            sons[i] = sonsC[i]
        stdlib.free(sonsC)

    cdef INT * get_sons_inner(self, INT *indC, int pos):
        cdef int nop=self._nop[indC[pos]]
        cdef INT *sonsC = <INT*> stdlib.malloc(sizeof(INT) * nop)
        sonsC[0] = pos+1
        for i in range(1, nop):
            sonsC[i] = self.traverse_inner(indC, sonsC[i-1])
        return sonsC

    cdef int count_func_cardinality(self, int nop):
        cdef int r=0, i
        for i in range(self._nfunc):
            if self._nop[i] == nop:
                r += 1
        return r

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
    def __init__(self, *args, **kwargs):
        super(PDEXO, self).__init__(*args, **kwargs)
        self._number_var_pm = 10

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def father2_xp_extras(self,
                          npc.ndarray[FLOAT, ndim=1, mode="c"] error,
                          npc.ndarray[FLOAT, ndim=1, mode="c"] x,
                          npc.ndarray[FLOAT, ndim=2, mode="c"] s):
        self._xo_error = <FLOAT *> error.data
        self._xo_x = <FLOAT *> x.data
        self._xo_s = <FLOAT *> s.data
        self._xo_c = s.shape[1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int count_error_value(self,
                               FLOAT *error,
                               FLOAT *x,
                               FLOAT *s,
                               int j1):
        cdef int c, j
        cdef int flag=0
        c = self._xo_c
        for j in range(c):
            # p2[m] = ((s[m] > x) == error).sum(axis=1)
            if s[j1] == x[j]:
                j1 += 1
                continue
            if s[j1] > x[j]:
                if error[j] == -1:
                    flag += 1
            else:
                if error[j] == 1:
                    flag += 1
            j1 += 1
        return flag


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int father2_crossing_point(self,
                                     npc.ndarray[INT, ndim=1, mode="c"] father1,
                                     npc.ndarray[INT, ndim=1, mode="c"] father2,
                                     int p1):
        cdef int f2_end = father2.shape[0], i, j, j1, c
        cdef int flag, bflag=0, res=0
        cdef INT *m, *_length
        cdef FLOAT *x, *s, *error
        self.crossover_mask(father1, father2, p1)
        m = self._m
        c = self._xo_c
        error = self._xo_error
        x = self._xo_x
        s = self._xo_s
        _length = self._length
        for i in range(f2_end):
            if m[i] == 0:
                continue
            j1 = i * c
            flag = self.count_error_value(error, x, s, j1)
            if flag > bflag:
                res = i
                bflag = flag
            elif flag == bflag and _length[i] < _length[res]:
                res = i
        self._bflag = bflag
        return res

    def get_xo_nmatch(self):
        return self._bflag

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int father2_xo_point_super(self,
                                     npc.ndarray[INT, ndim=1, mode="c"] father1,
                                     npc.ndarray[INT, ndim=1, mode="c"] father2,
                                     int p1):
        return super(PDEXO, self).father2_crossing_point(father1,
                                                         father2,
                                                         p1)

    def set_number_var_pm(self, int v):
        self._number_var_pm = v

    def get_number_var_pm(self):
        return self._number_var_pm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def pmutation_func_change(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                              int p1,
                              npc.ndarray[FLOAT, ndim=2, mode="c"] st,
                              npc.ndarray[FLOAT, ndim=1, mode="c"] e,
                              Eval eval):
        cdef FLOAT *stC, *eC, *pmut_eval, *x
        cdef INT *indC = <INT *> ind.data, *index
        cdef int nop = self._nop[indC[p1]], i=0, c=st.shape[1], _i
        cdef int bflag=-1, res=-1, flag=-1
        self._xo_c = c
        stC = <FLOAT *> st.data
        x = stC + p1*c
        eC = <FLOAT *> e.data
        index = self.get_sons_inner(indC, p1)
        eval.pmutation_eval_inner(nop, stC, index)
        pmut_eval = eval._pmut_eval
        for _i in range(self._nfunc):
            if nop != self._nop[_i]:
                continue
            j1 = i * c
            i += 1
            flag = self.count_error_value(eC, x, pmut_eval, j1)
            if flag > bflag:
                res = _i
                bflag = flag
        # print nop, self._nop[res]
        if res < 0:
            print res, flag, bflag
        stdlib.free(index)
        self._bflag = bflag
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float pmutation_constant(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                                   int p1,
                                   npc.ndarray[FLOAT, ndim=2, mode="c"] st,
                                   npc.ndarray[FLOAT, ndim=1, mode="c"] cons,
                                   int kind,
                                   npc.ndarray[FLOAT, ndim=1, mode="c"] e):
        cdef INT *indC = <INT *> ind.data
        cdef int i, j, end=e.shape[0]
        cdef float v
        cdef FLOAT *stC = <FLOAT *> st.data, *eC = <FLOAT *> e.data
        cdef FLOAT *consC = <FLOAT *> cons.data
        if self.isconstant(indC[p1]):
            i = indC[p1] - self._nfunc - self._nvar
            if kind == -1:
                return consC[i] + 0.1
            else:
                return consC[i] - 0.1
        else:
            j = end * p1            
            if kind == -1:
                v = -np.inf
                for i in range(end):
                    if eC[i] == -1 and stC[j] > v:
                        v = stC[j]
                    j += 1
                return v
            else:
                v = np.inf
                for i in range(end):
                    if eC[i] == 1 and stC[j] < v:
                        v = stC[j]
                    j += 1
                return v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def pmutation_terminal_change(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                                  int p1,
                                  npc.ndarray[FLOAT, ndim=2, mode="c"] st,
                                  npc.ndarray[FLOAT, ndim=1, mode="c"] error,
                                  npc.ndarray[FLOAT, ndim=2, mode="c"] x,
                                  npc.ndarray[FLOAT, ndim=1, mode="c"] cons,
                                  int ncons,
                                  Eval eval):
        cdef int *flag
        cdef int i, j=0, k=0, kk, end = st.shape[1], argmax, maxv, nvar = self._nvar
        cdef INT *var, *indC = <INT *> ind.data
        cdef FLOAT *errorC = <FLOAT *> error.data, *consC = <FLOAT *> cons.data
        cdef FLOAT *xC = <FLOAT *> x.data, *stC = <FLOAT *> st.data
        cdef npc.ndarray d = np.arange(self._number_var_pm)
        if self.isvar(indC[p1]):
            flag = <int *> stdlib.malloc(sizeof(int)*self._number_var_pm)
            for i in range(self._number_var_pm):
                flag[i] = 0
            np.random.shuffle(d)
            var = <INT *> d.data
            l = p1 * end
            for i in range(end):
                k = i * nvar
                for j in range(self._number_var_pm):
                    kk = k + var[j]
                    if stC[l] == xC[kk]:
                        continue
                    if xC[kk] > stC[l] and errorC[i] == -1:
                        flag[j] += 1
                    if xC[kk] < stC[l] and errorC[i] == 1:
                        flag[j] += 1
                l += 1
            argmax = 0
            maxv = flag[0]
            for i in range(1, self._number_var_pm):
                if flag[i] > maxv:
                    maxv = flag[i]
                    argmax = i
            self._bflag = maxv
            stdlib.free(flag)
            # print ind, d, var[argmax], var[argmax] + self._nfunc
            indC[p1] = var[argmax] + self._nfunc
            return 0
        for i in range(end):
            if errorC[i] == -1:
                j += 1
            elif errorC[i] == 1:
                k += 1 
        if j > k:
            j = -1
            self._bflag = j
        elif j < k:
            j = 1
            self._bflag = k
        else:
            self._bflag = j
            return 0
        i = indC[p1] - self._nfunc - self._nvar
        consC[i] = self.pmutation_constant(ind, p1, st, cons, j, error)
        return 0

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
