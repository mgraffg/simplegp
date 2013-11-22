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
from .RecursiveGP_mod import Recursive
from .simplegp import GP


class RecursiveGP(GP):
    def __init__(self, nlags=1, cases=None, **kwargs):
        super(RecursiveGP, self).__init__(**kwargs)
        self._nlags = nlags
        self._cases = np.empty(0, dtype=np.int) if cases is None else cases

    def train(self, x, f):
        super(RecursiveGP, self).train(x, f)
        self._g = np.zeros(self._x.shape[0], dtype=self._dtype)
        self.__r = Recursive(self._nop, self._nop.shape[0], self._x,
                             self._mem, self._g, self._cases,
                             nlags=self._nlags)

    def eval_ind(self, ind, pos=0, constants=None):
        self._mem.fill(0)
        constants = constants if constants is not None else self._constants
        self.__r.eval_ind_inner_iter(ind,
                                     constants,
                                     pos)
        return self._g

RGP = RecursiveGP










