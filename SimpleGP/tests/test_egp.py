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
from SimpleGP import EGPS, SparseArray


class TestSGPS(object):
    def __init__(self):
        x = np.linspace(-10, 10, 100)
        pol = np.array([0.2, -0.3, 0.2])
        self._pol = pol
        X = np.vstack((x**2, x, np.ones(x.shape[0])))
        self._y = (X.T * pol).sum(axis=1)
        self._x = x[:, np.newaxis]

    def test_EGPS(self):
        gp = EGPS(generations=3).fit(self._x,
                                     self._y, test=self._x,
                                     test_y=self._y)
        assert isinstance(gp.early_stopping[-1], SparseArray)
