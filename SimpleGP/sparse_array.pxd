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
#cython: nonecheck=True

from cpython cimport array
cimport numpy as npc
cimport cython
cimport libc.math as math


cdef class SparseArray:
    cdef int _nele
    cdef int _size
    cdef int * _indexC
    cdef double * _dataC
    cdef bint _usemem
    cpdef int size(self)
    cpdef set_size(self, int s)
    cpdef int nele(self)
    cpdef set_nele(self, int s)
    cpdef init_ptr(self, array.array[int] index, array.array[double] data)
    cpdef int nunion(self, SparseArray b)
    cpdef int nintersection(self, SparseArray other)
    cpdef SparseArray add(self, SparseArray other)
    cpdef SparseArray sub(self, SparseArray other)
    cpdef SparseArray mul(self, SparseArray other)
    cpdef SparseArray div(self, SparseArray other)
    cpdef double sum(self)
    cpdef SparseArray fabs(self)
    cpdef SparseArray exp(self)
    cpdef SparseArray sqrt(self)
    cpdef SparseArray sin(self)
    cpdef SparseArray cos(self)
    cpdef SparseArray ln(self)
    cpdef SparseArray sq(self)
    cpdef SparseArray sigmoid(self)
    cpdef SparseArray if_func(self, SparseArray y, SparseArray z)
    cpdef double SAE(self, SparseArray other)
    cpdef double SSE(self, SparseArray other)
    cpdef init(self, int nele)
    cpdef SparseArray empty(self, int nele, int size=?)
    cpdef SparseArray constant(self, double v, int size=?)

cdef class SparseEval:
    cdef list _st
    cdef list _x
    cdef list _output
    cdef SparseArray _x1
    cdef long *_nop
    cdef int _nvar
    cdef int _nfunc
    cdef int _pos
    cdef int _st_pos
    cdef int _size
    cdef long *_ind
    cdef double *_constants
    cpdef set_size(self, int s)
    cpdef set_nvar(self, int nvar)
    cdef int isfunc(self, int a)
    cdef int isvar(self, int a)
    cdef int isconstant(self, int a)
    cpdef eval(self, npc.ndarray[long, ndim=1] ind,
               npc.ndarray[double, ndim=1] constants,
               bint to_np_array=?)
    cdef SparseArray _eval(self)
    cdef SparseArray two_args(self, int func, SparseArray first, SparseArray second)
    cdef SparseArray one_arg(self, int func, SparseArray first)
