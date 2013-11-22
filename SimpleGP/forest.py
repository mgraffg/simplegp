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
from .simplegp import GP


class GPForest(GP):
    def __init__(self, ntrees=None, **kwargs):
        self._ntrees = None
        super(GPForest, self).__init__(**kwargs)

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

    def function_set(self, func, *args):
        if len(filter(lambda x: x == 'output', func)) == 0:
            func.append('output')
        super(GPForest, self).function_set(func, *args)
        self._output_pos = 15

    def subtrees_points(self, ind):
        points = np.zeros(self._ntrees+1, dtype=np.int)
        pos = 1
        points[0] = 1
        for i in range(1, self._ntrees+1):
            tmp = self.traverse(ind, pos=pos)
            points[i] = tmp
            pos = tmp
        return points

    def subtrees_length(self, ind):
        lengths = np.zeros(self._ntrees, dtype=np.int)
        pos = 1
        for i in range(self._ntrees):
            tmp = self.traverse(ind, pos=pos)
            lengths[i] = tmp - pos
            pos = tmp
        return lengths

    def subtree_selection(self, father):
        point = super(GPForest, self).subtree_selection(father)
        if father.shape[0] == 0:
            raise Exception("Inds of lenght 1 are forbidden")
        if point == 0:
            return 1
        return point

    def select_subtree(self):
        return np.random.randint(self._ntrees)

    def random_func(self, first_call=False):
        if first_call:
            return self._output_pos
        return super(GPForest, self).random_func(first_call=first_call)


class SubTreeXO(GPForest):
    def subtree_selection_fathers(self, father1, father2):
        ntree = self.select_subtree()
        pointsP1 = self.subtrees_points(father1)
        pointsP2 = self.subtrees_points(father2)
        f1 = father1[pointsP1[ntree]:pointsP1[ntree+1]]
        p1 = GP.subtree_selection(self,
                                  f1) + pointsP1[ntree]
        f2 = father2[pointsP2[ntree]:pointsP2[ntree+1]]
        p2 = GP.subtree_selection(self,
                                  f2) + pointsP2[ntree]
        return p1, p2
