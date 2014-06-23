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
# from SimpleGP.recursiveGP import RecursiveGP
from SimpleGP.utils import VerifyOutput
import types


class TimeSeries(GP):
    def __init__(self, nsteps=2, positive=False,
                 nlags=1, **kwargs):
        super(TimeSeries, self).__init__(**kwargs)
        self._nsteps = nsteps
        self._nlags = nlags
        self._positive = positive
        self._verify_output = VerifyOutput()

    @property
    def nsteps(self):
        """Number of steps, i.e., points ahead"""
        return self._nsteps

    @property
    def nlags(self):
        """Number of Lags"""
        return self._nlags

    def test_f(self, r):
        flag = super(TimeSeries, self).test_f(r)
        if not flag:
            return flag
        if self._positive and np.any(r < 0):
            return False
        return self._verify_output.verify(self._f, r)

    def predict(self, X, ind=None):
        end = self.nsteps
        if X.shape[1] > self.nlags and X.shape[0] < end:
            end = X.shape[0]
        if X.shape[0] < end:
            x = np.repeat(np.atleast_2d(X[-1]), end,
                          axis=0)
        else:
            x = X.copy()
        xorigin = self._x.copy()
        nlags = self.nlags
        pr = np.zeros(end, dtype=self._dtype)
        for i in range(end):
            self._x[0] = x[i]
            pr[i] = self.eval(ind)[0].copy()
            if i+1 < end:
                x[i+1, 1:nlags] = x[i, :nlags-1]
                x[i+1, 0] = pr[i]
        self._x[:] = xorigin[:]
        return pr

    def predict_best(self, X=None):
        if X is None:
            X = np.atleast_2d(self._f[-self.nlags:][::-1].copy())
        return self.predict(X, ind=self.best)

    @classmethod
    def run_cl(cls, serie, y=None, test=None, nlags=None,
               max_length=None, **kwargs):
        if serie.ndim == 1:
            assert y is None
            if nlags is None:
                nlags = cls.compute_nlags(serie.shape[0])
            serie, y = cls.create_W(serie, window=nlags)
        assert y is not None and nlags is not None
        if max_length is None:
            max_length = serie.shape[0] // 2
        if isinstance(max_length, types.FunctionType):
            max_length = max_length(serie.shape[0])
        if max_length < 8:
            max_length = 8
        if test is None and nlags == serie.shape[1]:
            test = np.atleast_2d(y[-nlags:][::-1].copy())
        return super(TimeSeries, cls).run_cl(serie, y, nlags=nlags,
                                             test=test,
                                             max_length=max_length, **kwargs)

    @staticmethod
    def compute_nlags(size):
        if size < 16:
            return int(np.floor(np.log2(size)))
        return int(np.ceil(np.log2(size)))

    @staticmethod
    def create_W(serie, window=10):
        assert serie.shape[0] > window
        w = np.zeros((serie.shape[0] - window, window), dtype=int)
        w[:, :] = np.arange(window)
        w = w + np.arange(w.shape[0])[:, np.newaxis]
        return serie[w][:, ::-1], serie[window:]


# class RTimeSeries(RecursiveGP, TimeSeries):
#     def train(self, x, f):
#         index = np.arange(0, f.shape[0], self._nsteps)
#         self._cases = np.zeros(x.shape[0], dtype=np.int)
#         self._cases[index] = 1
#         super(RTimeSeries, self).train(x, f)

#     def predict_best(self, xreg=None):
#         x, f = self._x, self._f
#         xp = np.zeros((self._nsteps, x.shape[1]), dtype=self._dtype)
#         xp[0, :self._nlags] = f[-self._nlags:][::-1].copy()
#         if xreg is not None:
#             xp[:, self._nlags:] = xreg[:, :]
#         self.train(xp, np.zeros(self._nsteps, dtype=self._dtype))
#         pr = self.eval(self.best)
#         self.train(x, f)
#         self._xp = xp
#         return pr
