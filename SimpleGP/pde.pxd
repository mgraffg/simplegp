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
cimport cython
from .tree cimport Tree
ctypedef npc.int_t INT
ctypedef npc.float_t FLOAT


cdef class PDE:
    cdef Tree _tree
    cdef INT *_ind
    cdef INT *_parent
    cdef INT *_path
    cdef int _l_st
    cdef FLOAT *_st
    cdef FLOAT *_p_st
    cdef int _end

    cpdef int compute_constants(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                                npc.ndarray[FLOAT, ndim=2, mode="c"] _st)

    cpdef int compute_pdepm(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                            npc.ndarray[FLOAT, ndim=2, mode="c"] _st,
                            npc.ndarray[INT, ndim=1, mode="c"] index,
                            float ppm)

    cpdef int compute(self, npc.ndarray[INT, ndim=1, mode="c"] ind,
                      int pos,
                      npc.ndarray[FLOAT, ndim=2, mode="c"] _st)

    cdef int compute_inner(self)

    cdef int is_first_var(self, int pos)

    cdef int which_var(self, int parent, int pos)

    cdef void add(self, int _i)
    
    cdef void subtract(self, int _i)

    cdef void multiply(self, int _i)

    cdef void divide(self, int _i)

    cdef void fabs(self, int _i)

    cdef void exp(self, int _i)
    
    cdef void sqrt(self, int _i)

    cdef void sin(self, int _i)

    cdef void cos(self, int _i)

    cdef void sigmoid(self, int _i)

    cdef void if_func(self, int _i)

    cdef void max(self, int _i)

    cdef void min(self, int _i)

    cdef void ln(self, int _i)

    cdef void sq(self, int _i)

    cdef void output(self, int _i)

    cdef void argmax(self, int _i)
