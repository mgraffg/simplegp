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
from SimpleGP import TimeSeries
import time

ts = np.array([3600.0, 7700.0, 12300.0, 30500.0, 47390.0, 57006.0,
               98563.0, 117759.0, 115097.0, 133759.0, 142485.0,
               169611.0, 216229.0, 253227.0, 313096.0, 322681.0,
               296245.0, 370333.0, 443826.0, 426751.0, 453627.0,
               553400.0])
vs = np.array([588568.0, 646758.0, 849998.0, 1106740.0, 1184550.0, 1425090.0])


class TestTimeSeries(object):
    def test_walltime(self):
        t = time.time()
        TimeSeries.run_cl(ts, walltime=1,
                          nsteps=vs.shape[0])
        assert time.time() - t < 1.1

    def test_compute_nlags(self):
        nlags = np.floor(np.log2(15))
        assert nlags == TimeSeries.compute_nlags(15)
        nlags = np.ceil(np.log2(16))
        assert nlags == TimeSeries.compute_nlags(16)

    def test_nlags(self):
        gp = TimeSeries.run_cl(ts, generations=2,
                               nsteps=vs.shape[0])
        assert gp.nlags == gp._x.shape[1]
        TS = TimeSeries
        x, y = TimeSeries.create_W(ts,
                                   window=TS.compute_nlags(ts.shape[0]))
        gp = TimeSeries.run_cl(x, y, generations=2, nlags=x.shape[1],
                               nsteps=vs.shape[0])
        assert gp.nlags == gp._x.shape[1]

    def test_gp(self):
        generations = 2
        gp = TimeSeries.run_cl(ts, generations=generations,
                               nsteps=vs.shape[0])
        pr1 = gp.predict_best()
        assert not np.any(np.isinf(pr1)) and not np.any(np.isnan(pr1))
        TS = TimeSeries
        x, y = TimeSeries.create_W(ts,
                                   window=TS.compute_nlags(ts.shape[0]))
        gp1 = TimeSeries.run_cl(x, y, generations=generations,
                                nlags=x.shape[1], nsteps=vs.shape[0])
        pr = gp1.predict_best()
        assert np.all(pr1 == pr)
        nlags = x.shape[1]
        x = np.concatenate((x, np.arange(x.shape[0])[:, np.newaxis]), axis=1)
        gp = TimeSeries.run_cl(x, y, generations=5, nlags=nlags,
                               nsteps=vs.shape[0])
        pr = gp.predict_best()
        assert np.all(pr1 != pr)

    def test_test_set(self):
        gp = TimeSeries.run_cl(ts, generations=2,
                               nsteps=6)
        nfunc = gp._nop.shape[0]
        gp.population[0] = np.array([0, nfunc, nfunc+1])
        x = np.ones(gp.nlags)
        pr = gp.predict(np.atleast_2d(x), ind=0)
        output = np.array([2, 3, 5, 8, 13, 21])
        assert np.all(pr == output)
        TS = TimeSeries
        x, y = TS.create_W(ts, TS.compute_nlags(ts.shape[0]))
        gp = TS.run_cl(x, y, nlags=x.shape[1] - 1, generations=2,
                       nsteps=6)
        gp.population[0] = np.array([0, nfunc, nfunc+gp.nlags])
        x = np.ones((6, x.shape[1]))
        x[-1, -1] = 10
        pr = gp.predict(np.atleast_2d(x), ind=0)
        output = np.array([2, 3, 4, 5, 6, 16])
        assert np.all(pr == output)
        assert gp._test_set is not None

    def test_max_length(self):
        gp = TimeSeries.run_cl(ts, generations=2,
                               nsteps=6)
        max_length = gp._max_length
        TS = TimeSeries
        x, y = TS.create_W(ts, TS.compute_nlags(ts.shape[0]))
        gp = TS.run_cl(x, y, generations=2, nlags=x.shape[1],
                       nsteps=6)
        assert gp._max_length == max_length

    def test_test_f(self):
        gp = TimeSeries.run_cl(ts, generations=2,
                               nsteps=6)
        output = np.empty(6)
        output.fill(np.inf)
        assert not gp.test_f(output)
        output.fill(1)
        assert gp.test_f(output)

    def test_gppde(self):
        from SimpleGP import GPPDE

        class TGP(TimeSeries, GPPDE):
            pass
        gp = TimeSeries.run_cl(ts, generations=2, nsteps=6)
        base = gp.predict_best()
        gp = TGP.run_cl(ts, generations=2, nsteps=6)
        pr = gp.predict_best()
        assert np.all(pr != base)
