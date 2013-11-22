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
import types
import os
import signal


class SimpleGA(object):
    def __init__(self, popsize=1000, ppm=0.1, chromosome_length=3,
                 tournament_size=2, generations=50, seed=0, verbose=False,
                 pxo=0.9, pm=0.2, stats=False, fname_best=None,
                 walltime=None,
                 dtype=np.float):
        self._popsize = popsize
        self._ppm = 1 - ppm
        self._tsize = tournament_size
        self._gens = generations
        self._pxo = pxo
        self._pm = pm
        self._verbose = verbose
        self._chromosome_length = chromosome_length
        self.gens_ind = popsize
        self._dtype = dtype
        self._stats = stats
        if stats:
            self.fit_per_gen = np.zeros(self._gens)
        self.set_seed(seed)
        self._best_fit = None
        self._fname_best = fname_best
        self._run = True
        self._last_call_to_stats = 0
        signal.signal(signal.SIGTERM, self.on_exit)
        if walltime is not None:
            signal.alarm(walltime)
            signal.signal(signal.SIGALRM, self.walltime)

    def new_best(self, k):
        pass

    def init(self):
        self.gens_ind = 0
        self._run = True
        self._last_call_to_stats = 0
        self._best_fit = None

    def walltime(self, *args, **kwargs):
        self.on_exit(*args, **kwargs)

    def on_exit(self, *args, **kwargs):
        self.save()
        self._run = False

    def set_seed(self, seed):
        np.random.seed(seed)

    def train(self, x, f):
        self._x = x.astype(self._dtype, copy=False, order='C')
        self._f = f.astype(self._dtype, copy=False, order='C')
        return self

    def crossover(self, father1, father2):
        mask = np.random.binomial(1, 0.5, self._p.shape[1]).astype(np.bool)
        return father1 * mask + father2 * ~mask

    def random_ind(self, size=None):
        size = size if size is not None else self._p.shape[1]
        return np.random.uniform(-1, 1, size)

    def mutation(self, father1):
        father2 = self.random_ind()
        mask = np.random.binomial(1, self._ppm,
                                  self._p.shape[1]).astype(np.bool)
        return father1 * mask + father2 * ~mask

    def selection(self, *args, **kwargs):
        return self.tournament(*args)

    def tournament(self, neg=False):
        if not neg:
            func_cmp = lambda x, y: x < y
        else:
            func_cmp = lambda x, y: x > y
        best = np.random.randint(self._popsize) if self._popsize > 2 else 0
        for i in range(self._tsize-1):
            comp = np.random.randint(self._popsize) if self._popsize > 2 else 1
            while comp == best:
                comp = np.random.randint(self._popsize)
            if func_cmp(self.fitness(best), self.fitness(comp)):
                best = comp
        return best

    def load_prev_run(self):
        try:
            fpt = open(self._fname_best)
            self._p = np.load(fpt)
            self._fitness = np.load(fpt)
            self.gens_ind = np.load(fpt)
            fpt.close()
            if self._p.ndim == 2 and self._p.shape[0] == self._popsize \
               and self._p.shape[1] == self._chromosome_length:
                return True
        except IOError:
            pass
        return False

    def create_population(self):
        if self._fname_best is not None \
           and os.path.isfile(self._fname_best) \
           and self.load_prev_run():
            return
        self._p = self.random_ind(size=(self._popsize,
                                        self._chromosome_length))
        self._fitness = np.zeros(self._popsize)
        self._fitness[:] = -np.inf

    def eval(self, ind):
        return (self._x * ind).sum(axis=1)

    def distance(self, y, hy):
        return ((y - hy)**2).mean()

    def fitness(self, ind):
        k = ind
        if isinstance(ind, types.IntType):
            if self._fitness[k] > -np.inf:
                return self._fitness[k]
            ind = self._p[ind]
        f = self.eval(ind)
        f = - self.distance(self._f, f)
        if np.isnan(f):
            f = -np.inf
        if isinstance(k, types.IntType):
            self._fitness[k] = f
            if self._best_fit is None or self._best_fit < f:
                self._best_fit = f
                self.new_best(k)
        return f

    def get_best(self):
        return int(self._fitness.argmax())

    def genetic_operators(self):
        son = None
        if np.random.rand() < self._pxo:
            father1 = self.tournament()
            father2 = self.tournament()
            while father1 == father2:
                father2 = self.tournament()
            son = self.crossover(self._p[father1], self._p[father2])
        if np.random.rand() < self._pm:
            son = son if son is not None else self._p[self.tournament()]
            son = self.mutation(son)
        son = son if son is not None else self.random_ind()
        return son

    def kill_ind(self, kill, son):
        self._p[kill] = son
        self._fitness[kill] = -np.inf

    def stats(self):
        i = self.gens_ind
        if i - self._last_call_to_stats < self._popsize:
            return
        self._last_call_to_stats = i
        if self._stats:
            self.fit_per_gen[i/self._popsize] = self._fitness[self.get_best()]
        if self._verbose:
            print "Gen: " + str(i) + "/" + str(self._gens * self._popsize) + \
                " " + "%0.4f" % self._fitness[self.get_best()]

    def run(self):
        self.create_population()
        while self.gens_ind < self._gens*self._popsize and self._run:
            try:
                son = self.genetic_operators()
                kill = self.tournament(neg=True)
                self.kill_ind(kill, son)
                self.stats()
                self.gens_ind += 1
            except KeyboardInterrupt:
                self.on_exit()
                return False
        flag = True
        if not self._run:
            flag = False
        self.on_exit()
        return flag

    def save(self, fname=None):
        fname = fname if fname is not None else self._fname_best
        if fname is None:
            return
        fpt = open(fname, 'w')
        np.save(fpt, self._p)
        np.save(fpt, self._fitness)
        np.save(fpt, self.gens_ind)
        if self._stats:
            np.save(fpt, self.fit_per_gen)
        fpt.close()


if __name__ == '__main__':
    np.random.seed(1)
    x = np.linspace(0, 1, 100)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    f = (X * pol).sum(axis=1)
    s = SimpleGA(generations=10000, popsize=3, pm=1.0, pxo=0.0,
                 fname_best=None,
                 verbose=True)
    s.train(X, f)
    s.create_population()
    s.run()
