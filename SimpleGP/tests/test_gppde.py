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
from SimpleGP import GPPDE, GPMAE, GPwRestart
import numpy as np


def test_gppde():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPPDE(generations=30, seed=0,
               max_length=1000).train(x, y)
    gp.run()
    assert gp.fitness(gp.get_best()) >= -3.272897322e-05


def test_gppde_mae():
    class G(GPMAE, GPPDE):
        pass
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = G(generations=30, seed=0,
           max_length=1000).train(x, y)
    gp.run()
    print gp.fitness(gp.get_best())
    assert gp.fitness(gp.get_best()) >= -0.000165552947597


def test_gppde_rprop():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPPDE(generations=50, update_best_w_rprop=True, seed=0,
               max_length=1000).train(x, y)
    gp.run()
    assert gp.fitness(gp.get_best()) >= -6.24303635529e-08


def test_gppde_test_set():
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    x1 = np.linspace(-20, 20, 1000)[:, np.newaxis]
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = GPPDE.run_cl(x, y, test=x1, max_length=1024,
                      generations=30,
                      seed=0, verbose=True)
    assert gp is not None


def test_gppde_predict():
    class G(GPwRestart, GPMAE, GPPDE):
        pass
    x = np.linspace(-10, 10, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    x1 = np.linspace(-20, 20, 1000)[:, np.newaxis]
    y = (X.T * pol).sum(axis=1)
    x = x[:, np.newaxis]
    gp = G.init_cl(max_length=1024,
                   generations=3, ntimes=1,
                   seed=0, verbose=True).train(x, y)
    gp.run(exit_call=False)
    gp.predict(x1)
    print "*"*10
    gp._gens = 6
    gp.run()
    assert gp is not None





