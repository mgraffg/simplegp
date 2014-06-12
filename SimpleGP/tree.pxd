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

cdef class Tree:
    cdef INT *_father1
    cdef INT *_father2
    cdef INT *_length
    cdef INT *_nop
    cdef INT *_depth
    cdef INT *_hist
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
    cdef int _type_xpoint_selection
 
    cdef INT isfunc(self, INT a)

    cdef INT isvar(self, INT a)

    cdef INT isconstant(self, INT a)

    cpdef set_pos(self, int pos)

    cpdef int get_pos_arg(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                          int pos,
                          int narg)

    cdef int get_pos_arg_inner(self, INT *ind,
                               int pos,
                               int narg)

    cpdef int path_to_root(self, npc.ndarray[INT, ndim=1, mode="c"] parent,
                           npc.ndarray[INT, ndim=1, mode="c"] path,
                           int pos)

    cdef int path_to_root_inner(self, INT *parent,
                                INT *path,
                                int pos)

    cpdef int compute_depth(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                            npc.ndarray[INT, ndim=1, mode="c"] depth)

    cdef void compute_depth_inner(self, INT *ind,
                                 INT *depth,
                                 int p)

    cpdef int compute_depth_histogram(self, npc.ndarray[INT, ndim=1, mode="c"] depth,
                                      npc.ndarray[INT, ndim=1, mode="c"] hist,
                                      int end)

    cdef int compute_depth_histogram_inner(self, INT *depth,
                                           INT *hist,
                                           int end)

    cpdef int select_xpoint_depth(self, npc.ndarray[INT, ndim=1, mode="c"] ind)

    cpdef int select_xpoint_uniform(self, npc.ndarray[INT, ndim=1, mode="c"] ind)

    cpdef int father1_crossing_point(self, npc.ndarray[INT, ndim=1, mode="c"] ind)

    cpdef compute_parents(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                          npc.ndarray[INT, ndim=1, mode="c"] parent)

    cdef compute_parents_inner(self, INT *ind,
                               INT *parent,
                               int p)
    
    cpdef INT equal_non_const(self,
                              npc.ndarray[INT, ndim=1, mode="c"] ind1,
                              npc.ndarray[INT, ndim=1, mode="c"] ind2)

    cdef int fix(self, int node, int ncons)

    cpdef int crossover_mask(self,
                             npc.ndarray[INT, ndim=1, mode="c"] father1,
                             npc.ndarray[INT, ndim=1, mode="c"] father2,
                             int p1)

    cpdef int father2_crossing_point(self,
                                     npc.ndarray[INT, ndim=1, mode="c"] father1,
                                     npc.ndarray[INT, ndim=1, mode="c"] father2,
                                     int p1)

    cpdef crossover(self, npc.ndarray[INT, ndim=1, mode="c"] father1,
                    npc.ndarray[INT, ndim=1, mode="c"] father2,
                    int ncons=?,
                    int p1=?,
                    int p2=?)

    cpdef int traverse(self,
                       npc.ndarray[INT, ndim=1, mode="c"] ind, 
                       INT pos=?)

    cdef int traverse_inner(self, INT *indC, INT pos)

    cpdef int length(self,
                     npc.ndarray[INT, ndim=1, mode="c"] ind)
    
    cdef int compute_length(self)

cdef class SubTree(Tree):
    cpdef int get_subtree(self, npc.ndarray[INT, ndim=1, mode="c"] father1,
                          int p1)


cdef class PDEXO(Tree):
    cdef FLOAT *_xo_x, *_xo_s, *_xo_error
    cdef int _xo_c

    cpdef int father2_xo_point_super(self,
                                     npc.ndarray[INT, ndim=1, mode="c"] father1,
                                     npc.ndarray[INT, ndim=1, mode="c"] father2,
                                     int p1)


cdef class PDEXOSubtree(PDEXO):
    cpdef int get_subtree(self, npc.ndarray[INT, ndim=1, mode="c"] father1,
                          int p1)
