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
from SimpleGP.recursiveGP import RecursiveGP


class TimeSeries(GP):
    def __init__(self, nsteps=2, nlags=1, **kwargs):
        super(TimeSeries, self).__init__(**kwargs)
        self._nsteps = nsteps
        self._nlags = nlags

    def predict_best(self, xreg=None):
        x, f = self._x, self._f
        pr = np.zeros(self._nsteps, dtype=self._dtype)
        xp = np.zeros((self._nsteps, x.shape[1]), dtype=self._dtype)
        xp[0, :self._nlags] = f[-self._nlags:][::-1].copy()
        if xreg is not None:
            xp[:, self._nlags:] = xreg[:, :]
        for i in range(self._nsteps):
            self.train(xp, np.zeros(1, dtype=self._dtype))
            pr[i] = self.eval(self.get_best())[i]
            if i+1 < self._nsteps:
                xp[i+1, 1:self._nlags] = xp[i, :self._nlags-1].copy()
                xp[i+1, 0] = pr[i].copy()
        self.train(x, f)
        return pr

    @staticmethod
    def create_W(serie, window=10):
        assert serie.shape[0] > window
        w = np.zeros((serie.shape[0] - window, window), dtype=int)
        w[:, :] = np.arange(window)
        w = w + np.arange(w.shape[0])[:, np.newaxis]
        return serie[w][:, ::-1], serie[window:]


class RTimeSeries(RecursiveGP, TimeSeries):
    def train(self, x, f):
        index = np.arange(0, f.shape[0], self._nsteps)
        self._cases = np.zeros(x.shape[0], dtype=np.int)
        self._cases[index] = 1
        super(RTimeSeries, self).train(x, f)

    def predict_best(self, xreg=None):
        x, f = self._x, self._f
        xp = np.zeros((self._nsteps, x.shape[1]), dtype=self._dtype)
        xp[0, :self._nlags] = f[-self._nlags:][::-1].copy()
        if xreg is not None:
            xp[:, self._nlags:] = xreg[:, :]
        self.train(xp, np.zeros(self._nsteps, dtype=self._dtype))
        pr = self.eval(self.get_best())
        self.train(x, f)
        self._xp = xp
        return pr

