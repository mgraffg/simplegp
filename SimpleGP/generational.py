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

from SimpleGP.simplegp import GP


class Generational(GP):
    def __init__(self, **kwargs):
        super(Generational, self).__init__(**kwargs)
        self._ngens = 1
        self._nkills = 0

    def create_population(self):
        super(Generational, self).create_population()
        self._sons_p = np.empty(self.popsize, dtype=np.object)
        self._sons_p_constants = np.empty(self.popsize, dtype=np.object)

    def kill_ind(self, kill, son):
        nk = self._nkills
        self._sons_p[nk] = son
        self._sons_p_constants[nk] = self._ind_generated_c
        self._nkills += 1
        if self._nkills == (self.popsize - 1):
            self._ngens += 1
            self._nkills = 0
            self.merge_population()

    def merge_population(self):
        m = np.ones(self.popsize, dtype=np.bool)
        m[self.best] = False
        self._fitness[m] = -np.inf
        for k, v in enumerate(np.where(m)[0]):
            self._p[v] = self._sons_p[k].copy()
            self._p_constants[v] = self._sons_p_constants[k].copy()
