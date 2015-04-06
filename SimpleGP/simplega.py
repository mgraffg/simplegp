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


class BestNotFound(Exception):
    """
    This exception is raised when the run exit with an non-viable individual.
    """
    pass


class SimpleGA(object):
    """
    SimpleGA is a steady state genetic algorithm with tournament selection,
    uniform crossover and mutation.

    >>> import numpy as np
    >>> from SimpleGP import SimpleGA

    First let us create a simple regression problem
    >>> _ = np.random.RandomState(0)
    >>> x = np.linspace(0, 1, 100)
    >>> pol = np.array([0.2, -0.3, 0.2])
    >>> X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    >>> f = (X * pol).sum(axis=1)

    The objective is to find the coefficients 0.2, -0.3, and 0.2
    >>> s = SimpleGA.init_cl().train(X, f)
    >>> s.run()
    True

    The coefficients are:

    >>> print s._p[s.best]
    [ 0.10430681 -0.18460194  0.17084382]
    """
    def __init__(self, popsize=1000, ppm=0.1, chromosome_length=3,
                 tournament_size=2, generations=50, seed=None, verbose=False,
                 pxo=0.9, pm=0.2, stats=False, fname_best=None,
                 save_only_best=False,
                 walltime=None,
                 dtype=np.float,
                 ind_dtype=np.int):
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
        self._ind_dtype = ind_dtype
        self._timeout = False
        self._stats = stats
        if stats:
            self.fit_per_gen = np.zeros(self._gens)
        self.set_seed(seed)
        self._best_fit = None
        self._best = None
        self._fname_best = fname_best
        self._run = True
        self._last_call_to_stats = 0
        self._p = None
        self._test_set = None
        self._save_only_best = save_only_best
        signal.signal(signal.SIGTERM, self.on_exit)
        if walltime is not None:
            signal.signal(signal.SIGALRM, self.walltime)
            signal.alarm(walltime)

    @property
    def population(self):
        return self._p

    @property
    def popsize(self):
        """Population size"""
        return self._popsize

    @popsize.setter
    def popsize(self, popsize):
        """Set the population size, it handles the case where the new
population size is smaller or larger than the current one

        """
        if self._popsize == popsize:
            return
        if self._popsize > popsize:
            index = self._fitness.argsort()[::-1][:popsize]
            self._p = self._p[index]
            self._fitness = self._fitness[index]
        else:
            d = popsize - self._popsize
            cl = self._chromosome_length
            self._p.resize((popsize, cl))
            self._p[self._popsize:] = self.random_ind(size=((d, cl)))
        self._popsize = popsize

    @property
    def generations(self):
        """Number of generations"""
        return self._gens

    @generations.setter
    def generations(self, v):
        self._gens = v

    def init(self):
        """
        Setting some variables to the defaults values
        """
        self.gens_ind = 0
        self._run = True
        self._last_call_to_stats = 0
        self._best_fit = None
        self._best = None

    def walltime(self, *args, **kwargs):
        """
        This method is called when the maximum number of seconds is reached.
        """
        self.on_exit(*args, **kwargs)
        self._timeout = True

    def on_exit(self, *args, **kwargs):
        """
        Method called at the end of the evolutionary process or when a
        signal is received
        """
        self.save()
        self._run = False

    def set_seed(self, seed):
        if seed is not None:
            np.random.seed(seed)

    def set_test(self, x):
        """
        x is the set test, this is used to test, during the evolution, that
        the best individual does not produce nan or inf
        """
        self._test_set = x.astype(self._dtype, copy=False, order='C')

    def train(self, x, f):
        """
        This is to set the training set.
        x and f are copy only if their types are not dtype
        """
        self._x = x.astype(self._dtype, copy=False, order='C')
        self._f = f.astype(self._dtype, copy=False, order='C')
        return self

    def crossover(self, father1, father2):
        """
        crossover performs an uniform crossover
        """
        mask = np.random.binomial(1, 0.5, self._p.shape[1]).astype(np.bool)
        return father1 * mask + father2 * ~mask

    def random_ind(self, size=None):
        """
        Create a random individual
        """
        size = size if size is not None else self._p.shape[1]
        return np.random.uniform(-1, 1, size)

    def mutation(self, father1):
        """
        Mutation performs an uniform mutation with point mutation probability
        set by ppm
        """
        father2 = self.random_ind()
        mask = np.random.binomial(1, self._ppm,
                                  self._p.shape[1]).astype(np.bool)
        return father1 * mask + father2 * ~mask

    def selection(self, *args, **kwargs):
        """
        Select a individual from the population.
        """
        return self.tournament(*args)

    def tournament(self, neg=False):
        """
        Tournament selection, it also performs negative tournament selection if
        neg=True
        """
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

    def load_extras(self, fpt):
        pass
        
    def load_prev_run(self):
        """
        Method used to load a previous run. It returns False if fails
        """
        import gzip
        
        def load(ftp):
            self._p = np.load(fpt)
            self._fitness = np.load(fpt)
            self.gens_ind = np.load(fpt)
            a = self._p
            m = np.all(np.isfinite(a), axis=1)
            if (~m).sum():
                tmp = self.random_ind(size=((~m).sum(),
                                            self._chromosome_length))
                a[~m] = tmp
                self._fitness[~m] = -np.inf
            self.best = self._fitness.argmax()
            if self._stats:
                self.fit_per_gen = np.load(fpt)
            self.load_extras(fpt)
            
        try:
            if self._fname_best.count('.gz'):
                with gzip.open(self._fname_best, 'rb') as fpt:
                    load(fpt)
            else:
                with open(self._fname_best, 'rb') as fpt:
                    load(fpt)
            if self._p.ndim == 2 and self._p.shape[0] == self._popsize \
               and self._p.shape[1] == self._chromosome_length:
                return True
        except IOError:
            pass
        return False

    def create_population(self):
        """
        Create the initial population. It first called load_prev_run if
        this method returns False then it creates the population.
        create_population returns True if the population was created and
        False if it was loaded from a previous run
        """
        if self._fname_best is not None \
           and os.path.isfile(self._fname_best) \
           and self.load_prev_run():
            return False
        if self._p is not None:
            return False
        self._p = self.random_ind(size=(self._popsize,
                                        self._chromosome_length))
        self._fitness = np.zeros(self._popsize)
        self._fitness[:] = -np.inf
        return True

    def eval(self, ind):
        """
        Evaluate a individual it receives the actual individual, i.e., the
        chromosomes
        """
        return (self._x * ind).sum(axis=1)

    def predict_test_set(self, ind):
        """Predicting the test set"""
        return self.predict(self._test_set, ind)

    def predict(self, X, ind=None):
        """
        Outputs the evaluation of the (ind)-th individual when the
        features are X
        """
        if ind is None:
            ind = self.best
        return (X * self._p[ind]).sum(axis=1)

    def distance(self, y, hy):
        """
        Sum of squares errors
        """
        return ((y - hy)**2).mean()

    def fitness(self, ind):
        """
        Computes the fitness of ind.  If ind is an integer, then it
        computes the fitness of the (ind)-th individual only if it has
        not been previously computed.
        """
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
            self.new_best(k)
            f = self._fitness[k]
        return f

    @property
    def best(self):
        """
        Get the position of the best individual
        """
        if self._best is None:
            raise BestNotFound()
        return self._best

    def get_best(self):
        return self.best

    def test_f(self, x):
        """This method test whether the prediction is valid. It is called from
        new_best. Returns True when x is a valid prediction

        """
        return ((not np.any(np.isnan(x))) and
                (not np.any(np.isinf(x))))

    @best.setter
    def best(self, k):
        """Returns the best so far (the position in the population). It raises
        an exception if the best has not been set

        """
        return self.new_best(k)

    def new_best_comparison(self, k):
        """
        This function is called from new_best
        """
        f = self._fitness[k]
        return self._best_fit is None or self._best_fit < f

    def new_best(self, k):
        """
        This method is called to test whether the best so far is beaten by k.
        Here is verified that the best individual is capable of
        predicting the test set, in the case it is given.
        """
        if self.new_best_comparison(k):
            if self._test_set is not None:
                # x = self._test_set
                r = self.predict_test_set(k)
                if not self.test_f(r):
                    self._fitness[k] = -np.inf
                    return False
            self._best_fit = self._fitness[k]
            self._best = k
            return True
        return False

    def pre_crossover(self, father1, father2):
        """
        This function is call before calling crossover, the idea
        is to test that the fathers are different.
        It returns True when the fathers are different
        """
        return not (father1 == father2)

    def genetic_operators(self):
        """
        Perform the genetic operations.
        """
        son = None
        if np.random.rand() < self._pxo:
            father1 = self.tournament()
            father2 = self.tournament()
            while not self.pre_crossover(father1, father2):
                father2 = self.tournament()
            son = self.crossover(self._p[father1], self._p[father2])
        if np.random.rand() < self._pm:
            son = son if son is not None else self._p[self.tournament()]
            son = self.mutation(son)
        son = son if son is not None else self.random_ind()
        return son

    def kill_ind(self, kill, son):
        """
        Replace the (kill)-th individual with son
        """
        if self._best == kill:
            raise BestNotFound("Killing the best so far!")
        self._p[kill] = son
        self._fitness[kill] = -np.inf

    def stats(self):
        """This function is call every time an offspring is created. The
        original idea is to print only statistics of the evolutionary process;
        however, at this stage is also used to verify the memory in GPPDE.
        This function is executed at the end of each generation and it returns
        False if this is not the case, otherwise returns True.
        """
        i = self.gens_ind
        if i - self._last_call_to_stats < self._popsize:
            return False
        self._last_call_to_stats = i
        if self._stats:
            self.fit_per_gen[i/self._popsize] = self._fitness[self.best]
        if self._verbose:
            print "Gen: " + str(i) + "/" + str(self._gens * self._popsize) + \
                " " + "%0.4f" % self._fitness[self.best]
        return True

    def run(self, exit_call=True):
        """
        Steady state genetic algorithm. Returns True if the evolution
        ended because the number of evaluations is reached. It returns False
        if it receives a signal or finds a perfect solution.
        The flag self._run is used to stop the evolution.
        """
        self.create_population()
        while (not self._timeout and
               self.gens_ind < self._gens*self._popsize and self._run):
            try:
                son = self.genetic_operators()
                kill = self.tournament(neg=True)
                while kill == self._best:
                    kill = self.tournament(neg=True)
                self._kill_ind = kill
                self.kill_ind(kill, son)
                self.stats()
                self.gens_ind += 1
            except KeyboardInterrupt:
                if exit_call:
                    self.on_exit()
                return False
        flag = True
        if not self._run:
            flag = False
        if exit_call:
            self.on_exit()
        return flag

    def clear_population_except_best(self):
        bs = self.best
        mask = np.ones(self.popsize, dtype=np.bool)
        mask[bs] = False
        self.population[mask] = None
        self._fitness[mask] = -np.inf
        return mask

    def save_extras(self, fpt):
        pass

    def save(self, fname=None):
        """
        Save the population to fname if fname is None the save in
        self._fname_best. If both are None then it does nothing.
        """
        import gzip
        
        def save_inner(fpt):
            np.save(fpt, self._p)
            np.save(fpt, self._fitness)
            np.save(fpt, self.gens_ind)
            if self._stats:
                np.save(fpt, self.fit_per_gen)
            self.save_extras(fpt)
            
        fname = fname if fname is not None else self._fname_best
        if fname is None:
            return False
        if self._save_only_best:
            self.clear_population_except_best()
        if fname.count('.gz'):
            raise NotImplementedError('There is a bug here')
            with gzip.open(fname, 'wb') as fpt:
                save_inner(fpt)
        else:
            with open(fname, 'wb') as fpt:
                save_inner(fpt)
        return True
            
    @classmethod
    def init_cl(cls, generations=10000,
                popsize=3, pm=0.1, pxo=0.9, seed=0,
                **kwargs):
        """
        Create a new instance of the class.
        """
        ins = cls(generations=generations,
                  popsize=popsize,
                  seed=seed,
                  pxo=pxo,
                  pm=pm,
                  **kwargs)
        return ins

    @classmethod
    def run_cl(cls, x, f, test=None,
               **kwargs):
        """
        Returns a trained system that does not output nan or inf neither
        in the training set (i.e., x) or test set (i.e., test).
        """
        ins = cls.init_cl(**kwargs).train(x, f)
        if test is not None:
            ins.set_test(test)
        ins.run()
        if ins._best is None:
            raise BestNotFound()
        return ins


if __name__ == "__main__":
    import doctest
    doctest.testmod()









