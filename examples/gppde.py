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

from SimpleGP import GPPDE
import numpy as np


x = np.linspace(-10, 10, 100)
pol = np.array([0.2, -0.3, 0.2])
X = np.vstack((x**2, x, np.ones(x.shape[0])))
y = (X.T * pol).sum(axis=1)
x = x[:, np.newaxis]
gp = GPPDE(generations=30, verbose=True,
           seed=2,
           max_length=1000).fit(x, y, test=x, test_y=y)
assert gp.fitness(gp.best) >= -2.2399825722547702e-06
print gp.fitness(gp.best)
print gp.print_infix()
