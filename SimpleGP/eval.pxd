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
ctypedef npc.float_t FLOAT
ctypedef npc.int_t INT

cdef class Eval:
    cdef INT _pos
    cdef INT *_ind
    cdef INT _nvar
    cdef INT _nfunc
    cdef FLOAT *_st
    cdef INT _l_st
    cdef INT *_nop
    cdef INT *_output
    cdef INT _n_output
    cdef INT _max_nargs
    cdef FLOAT *_x
    cdef FLOAT *_cons

    cdef INT eval_ind_inner(self)

    cdef void one_args(self, INT func, INT a, INT pos)

    cdef INT fabs(self, 
                  INT a,
                  INT pos)

    cdef INT exp(self, 
                 INT a,
                 INT pos)

    cdef INT sqrt(self, 
                  INT a,
                  INT pos)

    cdef INT sin(self, 
                 INT a,
                 INT pos)

    cdef INT cos(self, 
                 INT a,
                 INT pos)

    cdef INT sigmoid(self, 
                 INT a,
                 INT pos)

    cdef INT ln(self, 
                INT a,
                INT pos)

    cdef INT sq(self, 
                INT a,
                INT pos)

    cdef void two_args(self, INT func, INT a, INT b, INT pos)

    cdef INT add(self, 
                 INT a,
                 INT b,
                 INT pos)

    cdef INT subtract(self, 
                      INT a,
                      INT b,
                      INT pos)

    cdef INT multiply(self, 
                      INT a,
                      INT b,
                      INT pos)

    cdef INT divide(self, 
                    INT a,
                    INT b,
                    INT pos)

    cdef INT max(self, 
                 INT a,
                 INT b,
                 INT pos)

    cdef INT min(self, 
                 INT a,
                 INT b,
                 INT pos)

    cdef void three_args(self, INT func, INT a, INT b, INT c, INT pos)

    cdef void output_function(self)

    cdef void variable_args(self, INT func, INT *a, INT nargs, INT pos)

    cdef void argmax(self, INT *a, INT nargs, INT pos)












