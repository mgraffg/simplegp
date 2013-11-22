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
from .forest import GPForest


class Classification(GPForest):
    def train(self, x, f):
        y = np.zeros((f.shape[0], np.unique(f).shape[0]),
                     dtype=self._dtype)
        y[np.arange(y.shape[0]), f] = 1
        super(Classification, self).train(x, y)
