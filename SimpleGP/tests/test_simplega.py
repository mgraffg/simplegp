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
from SimpleGP import SimpleGA
import numpy as np


def test_SimpleGA():
    np.random.RandomState(0)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA(generations=10000,
                 popsize=3, pm=0.1,
                 pxo=0.9,
                 fname_best=None,
                 verbose=False).train(X, f)
    s.run()
    assert np.fabs(pol - s._p[s.get_best()]).mean() < 0.1
