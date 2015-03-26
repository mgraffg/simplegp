from SimpleGP import GP, BestNotFound
import numpy as np
from numpy.linalg import lstsq


class lstsqGP(GP):
    def __init__(self, **kwargs):
        super(lstsqGP, self).__init__(**kwargs)
        self._pop_eval_mut = None
        self._pop_eval = None
        self._pop_hist = None
        self._nparents = 2

    def kill_ind(self, kill, son):
        """
        Replace the (kill)-th individual with son
        """
        if self._best == kill:
            raise BestNotFound("Killing the best so far!")
        self._pop_eval[kill] = son
        f1 = self._xo_father1
        if self._nparents == 1:
            self._pop_hist[kill] = [self._pop_hist[f1],
                                    self._xo_father2[0],
                                    self._xo_father2[1],
                                    self._ind_generated_c]
        else:
            f2 = self._xo_father2
            self._pop_hist[kill] = [self._pop_hist[f1],
                                    self._pop_hist[f2],
                                    self._ind_generated_c]
        self._fitness[kill] = - self.distance(self._f, son)
        self.new_best(kill)

    def create_population(self):
        def test_fitness():
            m = np.array(map(lambda x: self.fitness(x),
                             range(self.popsize)))
            m = np.isfinite(m)
            return m
        super(lstsqGP, self).create_population()
        m = test_fitness()
        while (~m).sum():
            index = np.where(~m)[0]
            generator = self.create_population_generator
            for i, ind, cons in generator(popsize=index.shape[0]):
                self._p[index[i]] = ind
                self._p_constants[index[i]] = cons
            m = test_fitness()
        self._pop_eval = np.empty((self.popsize, self._f.shape[0]))
        self._pop_hist = np.empty(self.popsize, dtype=np.object)
        self._pop_hist[:] = np.arange(self.popsize)
        for i in range(self.popsize):
            self._pop_eval[i] = self.eval(i)
        self._pop_eval_mut = self._pop_eval.copy()

    def compute_alpha_beta(self, X):
        return lstsq(X, self._f)[0]

    def crossover(self, father1, father2, *args):
        self._nparents = 2
        f1 = self._xo_father1
        f2 = self._xo_father2
        X = self._pop_eval[np.array([f1, f2])].T
        alpha_beta = self.compute_alpha_beta(X)
        y = X.dot(alpha_beta)
        self._ind_generated_c = alpha_beta
        return y

    def mutation(self, father1):
        self._nparents = 1
        f1 = self._xo_father1
        p1 = np.random.randint(self.popsize)
        p2 = np.random.randint(self.popsize)
        while p1 == p2:
            p2 = np.random.randint(self.popsize)
        self._xo_father2 = [p1, p2]
        X = np.vstack((self._pop_eval[f1],
                       self._pop_eval_mut[p1] - self._pop_eval_mut[p2])).T
        alpha_beta = lstsq(X, self._f)[0]
        y = X.dot(alpha_beta)
        self._ind_generated_c = alpha_beta
        return y
