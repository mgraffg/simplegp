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
from multiprocessing import Process, cpu_count
import time


class Pool(object):
    def __init__(self, cpu=cpu_count()/2):
        self._cpu = cpu

    def run(self, func, lst_args):
        lst_args.reverse()
        process_lst = []
        while len(lst_args):
            process_lst = filter(lambda x: x.is_alive(), process_lst)
            for i in range(len(process_lst), self._cpu):
                if len(lst_args) == 0:
                    continue
                p = Process(target=func, args=lst_args.pop())
                p.start()
                process_lst.append(p)
            time.sleep(0.1)


class VerifyOutput(object):
    def __init__(self, max_error=16, forecast=True):
        self._max_error = max_error
        self.value = None
        self._forecast = forecast

    def get_value(self, ts, vs):
        self.compute(ts, vs)
        return self.value

    def get_value_ts(self, ts, vs):
        self.compute(ts, vs)
        return self.value_ts

    def ratio(self, ts, vs):
        self.compute(ts, vs)
        return (self.value + 1) / (self.value_ts + 1)

    def compute(self, ts, vs):
        from scipy import optimize
        if self._forecast:
            x = np.arange(ts.shape[0] + vs.shape[0])
        else:
            x = np.arange(ts.shape[0])
            x = np.concatenate((x, np.arange(vs.shape[0])))
        lst = []
        for func in [self.line, self.power, self.exp]:
            try:
                popt, pcov = optimize.curve_fit(func, x[:ts.shape[0]], ts)
            except RuntimeError:
                continue
            lst.append((self.n_mae(ts, func(x[:ts.shape[0]],
                                            *popt)),
                        self.n_mae(np.concatenate((ts,
                                                   func(x[ts.shape[0]:],
                                                        *popt))),
                                   np.concatenate((ts, vs)))))
        if len(lst) == 0:
            self.value = np.inf
            self.value_ts = 0
        else:
            r = lst[np.argmin(map(lambda x: x[0], lst))]
            self.value = r[1]
            self.value_ts = r[0]

    def verify(self, ts, vs):
        if ts.shape[0] + vs.shape[0] > 200:
            if vs.shape[0] > 190:
                raise Exception("At this moment the forecast cannot be\
                greater than 190")
            cnt = 200 - vs.shape[0]
            ts = ts[-cnt:]
        r = self.ratio(ts, vs)
        if np.isnan(r):
            r = np.inf
        if r > self._max_error:
            return False
        return True

    @staticmethod
    def line(x, a, b):
        return a*x + b

    @staticmethod
    def exp(x, a):
        return np.exp(a*x)

    @staticmethod
    def power(x, a, b, c):
        return a*x**b + c

    @staticmethod
    def n_mae(a, b):
        try:
            m, c = np.linalg.solve([[a.min(), 1], [a.max(), 1]],
                                   [0, 1])
        except np.linalg.LinAlgError:
            m = 1
            c = 0
        ap = a * m + c
        bp = b * m + c
        return np.fabs(ap - bp).mean()
