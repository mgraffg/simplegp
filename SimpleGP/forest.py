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
import numpy as np
from .simplegp import GP, GPPDE
from .tree import Tree, SubTree, PDEXO


class GPForest(GP):
    def __init__(self, ntrees=None, **kwargs):
        self._ntrees = None
        super(GPForest, self).__init__(**kwargs)

    def new_best(self, k):
        output = self._output
        super(GPForest, self).new_best(k)
        self._output = output
        self._eval.set_output_function(self._output)

    def train(self, x, f):
        super(GPForest, self).train(x, f)
        if self._ntrees is not None:
            self._output = np.empty(self._ntrees, dtype=np.int)
        else:
            if f.ndim == 1:
                self._output = np.empty(np.unique(f).shape[0], dtype=np.int)
            else:
                self._output = np.empty(f.shape[1], dtype=np.int)
            self._ntrees = self._output.shape[0]
        self._eval.set_output_function(self._output)
        self._nop[self._output_pos] = self._ntrees
        return self

    def min_max_length_params(self, minimum=None, maximum=None):
        if minimum is not None:
            self._min_length = minimum
        if self._min_length == 0:
            self._min_length = 1
        if maximum is not None:
            self._max_length = maximum

    def tree_params(self):
        self._tree_length = np.empty(self._max_length,
                                     dtype=np.int)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=np.int)
        self._tree = Tree(self._nop,
                          self._tree_length,
                          self._tree_mask,
                          self._min_length,
                          self._max_length,
                          select_root=0)

    def random_func(self, first_call=False):
        if first_call:
            return self._output_pos
        return super(GPForest, self).random_func(first_call=first_call)


class SubTreeXO(GPForest):
    def tree_params(self):
        self._tree_length = np.empty(self._max_length,
                                     dtype=np.int)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=np.int)
        self._tree = SubTree(self._nop,
                             self._tree_length,
                             self._tree_mask,
                             self._min_length,
                             self._max_length,
                             select_root=0)


class SubTreeXOPDE(GPForest, GPPDE):
    def new_best(self, k):
        p_st = map(lambda x: self._p_st[x],
                   range(self._popsize))
        p_error_st = map(lambda x: self._p_error_st[x],
                         range(self._popsize))
        super(SubTreeXOPDE, self).new_best(k)
        for i in range(self._popsize):
            self._p_st[i] = p_st[i]
            self._p_error_st[i] = p_error_st[i]

    def train(self, x, f):
        super(SubTreeXOPDE, self).train(x, f)
        for i in range(self._popsize):
            self._p_st[i] = None
            self._p_error_st[i] = None
        return self

    def tree_params(self):
        self._tree_length = np.empty(self._max_length,
                                     dtype=np.int)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=np.int)
        self._tree = PDEXO(self._nop,
                           self._tree_length,
                           self._tree_mask,
                           self._min_length,
                           self._max_length,
                           select_root=0)









