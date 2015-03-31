from SimpleGP import GP, BestNotFound
import types
import numpy as np
from numpy.linalg import lstsq


class lstsqGP(GP):
    def __init__(self, save_only_best=False, **kwargs):
        if save_only_best:
            raise NotImplementedError("This option is not implemented")
        super(lstsqGP, self).__init__(save_only_best=save_only_best,
                                      **kwargs)
        self._pop_eval_mut = None
        self._pop_eval = None
        self._pop_hist = None
        self._nparents = 2

    def save_extras(self, fpt):
        best = self.best
        np.save(fpt, best)
        np.save(fpt, self._pop_hist)
        np.save(fpt, self._pop_eval)
        np.save(fpt, self._history_coef)
        np.save(fpt, self._history_ind)
        np.save(fpt, self._history_index)

    def load_extras(self, fpt):
        best = np.load(fpt)
        hist = np.load(fpt)
        eval = np.load(fpt)
        coef = np.load(fpt)
        ind = np.load(fpt)
        index = np.load(fpt)
        self._load_tmp = [best, hist, eval, coef,
                          ind, index]

    def save_hist(self, kill):
        index = self._history_index
        self._history_coef[index] = self._ind_generated_c
        f1 = self._pop_hist[self._xo_father1]
        if self._nparents == 1:
            self._history_ind[index] = np.array([f1,
                                                 self._xo_father2[0],
                                                 self._xo_father2[1]])
        else:
            f2 = self._pop_hist[self._xo_father2]
            self._history_ind[index] = np.array([f1, f2, -1])
        self._pop_hist[kill] = index
        self._history_index += 1

    def improve_fit(self, son):
        fit = - self.distance(self._f, son)
        if fit <= self.fitness(self._xo_father1):
            return fit, False
        if self._nparents == 2 and fit <= self.fitness(self._xo_father2):
            return fit, False
        return fit, True

    def kill_ind(self, kill, son):
        """
        Replace the (kill)-th individual with son
        """
        if self._best == kill:
            raise BestNotFound("Killing the best so far!")
        fit, imp = self.improve_fit(son)
        if not imp:
            return
        self._pop_eval[kill] = son
        self.save_hist(kill)
        self._fitness[kill] = fit
        self.new_best(kill)

    def create_population(self):
        def test_fitness():
            m = np.array(map(lambda x: self.fitness(x),
                             range(self.popsize)))
            m = np.isfinite(m)
            return m
        flag = super(lstsqGP, self).create_population()
        if not flag:
            self._fitness.fill(-np.inf)
        m = test_fitness()
        while (~m).sum():
            index = np.where(~m)[0]
            generator = self.create_population_generator
            for i, ind, cons in generator(popsize=index.shape[0]):
                self._p[index[i]] = ind
                self._p_constants[index[i]] = cons
            m = test_fitness()
        self._pop_eval = np.empty((self.popsize, self._f.shape[0]))
        self._pop_hist = np.arange(self.popsize)
        for i in range(self.popsize):
            self._pop_eval[i] = self.eval(i).copy()
        self._pop_eval_mut = self._pop_eval.copy()
        if not flag:
            best, hist, eval, coef, ind, index = self._load_tmp
            self._pop_hist = hist
            self._pop_eval = eval
            for i in range(self.popsize):
                self._fitness[i] = - self.distance(self._f, eval[i])
            self._history_coef = coef
            self._history_ind = ind
            self._history_index = index
            self.new_best(int(best))
        else:
            self._history_coef = np.zeros((self.popsize *
                                           self.generations,
                                           2), dtype=self._dtype)
            self._history_ind = np.empty((self.popsize *
                                          self.generations,
                                          3), dtype=np.int)
            self._history_ind.fill(-1)
            self._history_index = self.popsize

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

    def predict(self, X, ind=None):
        init = np.array(map(lambda x: super(lstsqGP, self).predict(X, ind=x),
                            range(self.popsize)))
        if ind is None:
            ind = self.best
        eval = lstsqEval(init, self._history_ind, self._history_coef)
        pr = eval.eval(self._pop_hist[ind])
        return pr


class lstsqEval(object):
    def __init__(self, init, hist_ind, hist_coef):
        self._init = init
        self._hist_ind = hist_ind
        self._hist_coef = hist_coef
        self._pos = 0

    def inds_to_eval(self, ind):
        h = {ind: 1}
        lst = self._hist_ind[ind].tolist()
        while len(lst):
            k = lst.pop()
            if k in h:
                continue
            lst += self._hist_ind[k].tolist()
            h[k] = 1
        h = h.keys()
        h.sort()
        if h[0] == -1:
            return h[1:]
        return h

    def eval(self, ind):
        def get_ev(_ind):
            if _ind < popsize:
                return init[_ind]
            return st[_m[_ind]]

        inds = self.inds_to_eval(ind)
        popsize = self._init.shape[0]
        coef = self._hist_coef
        init = self._init
        hist_ind = self._hist_ind
        st = []
        _m = {}
        c = 0
        for index in inds:
            if index < popsize:
                continue
            hist = hist_ind[index]
            c1, c2 = coef[index]
            f1 = get_ev(hist[0])
            if hist[-1] == -1:
                f2 = get_ev(hist[1])
            else:
                f2 = init[hist[1]] - init[hist[2]]
            st.append(c1 * f1 + c2 * f2)
            _m[index] = c
            c += 1
        return get_ev(ind)


class GSGP(lstsqGP):
    def __init__(self, ms=0.001, **kwargs):
        super(GSGP, self).__init__(**kwargs)
        self._ms = ms

    def compute_alpha_beta(self, X):
        if self._nparents == 2:
            beta = np.random.rand()
            return np.array([beta, 1-beta])
        return np.array([1, self._ms])
