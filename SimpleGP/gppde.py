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
from SimpleGP.tree import PDEXO
from SimpleGP.Rprop_mod import RpropSign
from SimpleGP.simplegp import GP


class GPPDE(GP):
    def __init__(self, compute_derivatives=True,
                 max_mem=300.0,
                 update_best_w_rprop=False,
                 **kwargs):
        assert compute_derivatives
        super(GPPDE, self).__init__(compute_derivatives=compute_derivatives,
                                    **kwargs)
        self._max_mem = max_mem
        self._update_best_w_rprop = update_best_w_rprop
        self._p_st = np.empty(self._popsize, dtype=np.object)
        self._p_error_st = np.empty(self._popsize, dtype=np.object)
        self._used_mem = 0

    def new_best(self, k):
        super(GPPDE, self).new_best(k)
        fit = self._best_fit
        if not self._update_best_w_rprop or fit is None:
            return None
        self.rprop(k)
        if self._fitness[k] > fit:
            self._best_fit = self._fitness[k]
            return super(GPPDE, self).new_best(k)

    def stats(self):
        flag = super(GPPDE, self).stats()
        self.free_mem()
        return flag

    def update_mem(self, d, sign=1):
        if d is not None:
            d = d.nbytes / 1024. / 1024.
            self._used_mem += (d * sign)

    def max_mem_per_individual(self, xs=None):
        if xs is None:
            xs = self._x.shape[0]
        p_st = np.empty((self._max_length, xs),
                        dtype=self._dtype, order='C').nbytes
        p_error_st = np.ones((self._max_length,
                              xs),
                             dtype=np.int8, order='C').nbytes
        p_der_st = np.ones((self._max_length,
                            xs*self._max_nargs),
                           dtype=self._dtype,
                           order='C').nbytes
        return (p_der_st / 1024. / 1024.,
                (p_st + p_error_st) / 1024. / 1024.)

    def tree_params(self):
        self._tree_length = np.empty(self._max_length,
                                     dtype=np.int)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=np.int)
        self._tree = PDEXO(self._nop,
                           self._tree_length,
                           self._tree_mask,
                           self._min_length,
                           self._max_length)

    def train(self, x, f):
        super(GPPDE, self).train(x, f)
        self.free_mem()
        return self

    def load_prev_run(self):
        r = super(GPPDE, self).load_prev_run()
        if r:
            self._fitness.fill(-np.inf)
            gens_ind = self.gens_ind
            for i in range(self._popsize):
                self.fitness(i)
            self.gens_ind = gens_ind
        return r

    def mem(self):
        """
        Memory used
        """
        return self._used_mem

    def free_mem(self):
        """
        This method free the memory when the memory used is more than
        self._max_mem
        """
        if self.mem() < self._max_mem:
            return None
        for i in range(self._popsize):
            self.update_mem(self._p_st[i], -1)
            self._p_st[i] = None
            self.update_mem(self._p_error_st[i], -1)
            self._p_error_st[i] = None
            if hasattr(self, '_fitness'):
                self._fitness[i] = -np.inf

    def mutation(self, father1):
        kill = self.tournament(neg=True)
        while kill == self._xo_father1:
            kill = self.tournament(neg=True)
        d = np.random.randint(self._min_depth,
                              self._mutation_depth)
        son = self.create_random_ind(depth=d,
                                     first_call=True)
        self.kill_ind(kill, son)
        self._xo_father2 = kill
        self.fitness(kill)
        ind = self.crossover(father1, son)
        return ind

    def crossover(self, father1, father2, p1=-1, p2=-1,
                  force_xo=False):
        error = self._p_error_st[self._xo_father1]
        x = self._p_st[self._xo_father1]
        s = self._p_st[self._xo_father2]
        self._tree.father2_xp_extras(error,
                                     x, s)
        if not force_xo and (
                self.fitness(self._xo_father1) == -np.inf or
                self.fitness(self._xo_father2) == -np.inf):
            if p1 == -1:
                if self._tree.get_select_root():
                    p1 = np.random.randint(father1.shape[0])
                else:
                    p1 = np.random.randint(father1.shape[0]-1) + 1
            if p2 == -1:
                p2 = self._tree.father2_xo_point_super(father1,
                                                       father2, p1)
        return super(GPPDE, self).crossover(father1, father2,
                                            p1, p2)

    def get_p_der_st(self, ind):
        if self._p_der_st is None:
            self._p_der_st = np.ones((ind.shape[0],
                                      self._x.shape[0]*self._max_nargs),
                                     dtype=self._dtype,
                                     order='C')
            self.update_mem(self._p_der_st)
        elif self._p_der_st.shape[0] < ind.shape[0]:
            self.update_mem(self._p_der_st, -1)
            self._p_der_st.resize((ind.shape[0],
                                   self._x.shape[0]*self._max_nargs))
            self.update_mem(self._p_der_st)
            self._p_der_st.fill(1)
        return None, self._p_der_st

    def get_st(self, ind):
        if self._computing_fitness is None:
            if self._st is None:
                self._st = np.empty((ind.shape[0], self._x.shape[0]),
                                    dtype=self._dtype, order='C')
            elif self._st.shape[0] < ind.shape[0]:
                self._st.resize((ind.shape[0], self._x.shape[0]))
            return self._st
        else:
            k = self._computing_fitness
            l = ind.shape[0]
            if self._p_st[k] is None:
                self._p_st[k] = np.empty((ind.shape[0], self._x.shape[0]),
                                         dtype=self._dtype, order='C')
                self.update_mem(self._p_st[k])
            elif self._p_st[k].shape[0] < l:
                self.update_mem(self._p_st[k], -1)
                self._p_st[k].resize(l, self._x.shape[0])
                self.update_mem(self._p_st[k])
            return self._p_st[k]

    def fitness(self, ind):
        fit = super(GPPDE, self).fitness(ind)
        if not self._use_cache and isinstance(ind,
                                              types.IntType):
            self.compute_derivatives(ind)
            self.set_extras_to_ind(ind, self._p[ind])
        if isinstance(ind, types.IntType):
            return self._fitness[ind]
        return fit

    def get_error_st(self):
        assert self._computing_fitness is not None
        ind = self._computing_fitness
        l = self._p[ind].shape[0]
        if self._p_error_st[ind] is None:
            self._p_error_st[ind] = np.ones((l,
                                             self._x.shape[0]),
                                            dtype=np.int8, order='C')
            self.update_mem(self._p_error_st[ind])
        if self._p_error_st[ind].shape[0] < l:
            self.update_mem(self._p_error_st[ind], -1)
            self._p_error_st[ind].resize(l, self._x.shape[0])
            self.update_mem(self._p_error_st[ind])
            self._p_error_st[ind].fill(1)
        return self._p_error_st[ind]

    def get_rprop(self, update_constants=0):
        assert self._computing_fitness is not None
        k = self._computing_fitness
        ind = self._p[k]
        constants = self._p_constants[k]
        self.__prev_step = np.zeros(ind.shape[0], dtype=self._dtype)
        self.__prev_slope = np.zeros(ind.shape[0], dtype=self._dtype)
        rprop = RpropSign(ind, constants, self._nop,
                          self._x.shape[1], self._p_der_st,
                          self._x.shape[0],
                          self.__prev_step,
                          self.__prev_slope,
                          update_constants=update_constants,
                          max_nargs=self._max_nargs)
        return rprop

    def compute_derivatives(self, ind=None, pos=0, constants=None):
        """
        Compute the partial derivative of the error w.r.t. every node
        of the tree
        """
        rprop = self.get_rprop()
        k = self._computing_fitness
        ind = self._p[k]
        constants = self._p_constants[k]
        error_st = self.get_error_st()
        rprop.set_error_st_sign(error_st)
        e, g = self.compute_error_pr(ind, pos=pos, constants=constants,
                                     epoch=0)
        rprop.set_zero_pos()
        error_st[self._output] = np.sign(e.T)
        rprop.update_constants_rprop()

    def rprop(self, ind=None, pos=0, constants=None,
              epochs=10000):
        """Update the constants of the tree using RPROP"""
        fit_best = -np.inf
        epoch_best = 0
        rprop = self.get_rprop(update_constants=1)
        error_st = self.get_error_st()
        rprop.set_error_st_sign(error_st)
        k = self._computing_fitness
        ind = self._p[k]
        constants = self._p_constants[k]
        best_cons = constants.copy()
        if not self.any_constant(ind):
            return None
        # print "empezando", k
        for i in range(epochs):
            if i > 0:
                self.gens_ind += 1
            # print "compute_error_pr", k
            e, g = self.compute_error_pr(ind, pos=pos, constants=constants,
                                         epoch=i)
            # print "end compute_error_pr", k
            fit = - self.distance(self._f, g)
            if fit > fit_best and not np.isnan(fit):
                fit_best = fit
                best_cons = constants.copy()
                epoch_best = i
            if i < epochs - 1:
                # print "update_constants", k
                rprop.set_zero_pos()
                error_st[self._output] = np.sign(e.T)
                rprop.update_constants_rprop()
                # print "end update_constants", k
            if i - epoch_best >= self._max_n_worst_epochs:
                break
        constants[:] = best_cons[:]
        self._fitness[k] = fit_best
        # print "terminando", k

    @classmethod
    def init_cl(cls, training_size=None,
                update_best_w_rprop=True,
                max_length=1024,
                max_mem=500, **kwargs):
        ins = cls(max_mem=max_mem,
                  max_length=max_length,
                  update_best_w_rprop=update_best_w_rprop,
                  **kwargs)
        if training_size is None:
            return ins
        base, pr = ins.max_mem_per_individual(training_size)
        if (pr * ins._popsize) + base > ins._max_mem:
            mm = ins._max_mem - base
            assert mm > 0
            popsize = np.floor(mm / np.float(pr)).astype(np.int)
            nind = ins._gens * ins._popsize
            popsize = filter(lambda x: (nind % x) == 0,
                             range(2, popsize+1))[-1]
            ins._gens = np.floor(nind / popsize).astype(np.int)
            ins._popsize = popsize
        return ins

    @classmethod
    def run_cl(cls, x, f, test=None, ntries=10, pgrow=0.0,
               **kwargs):
        """
        Returns a trained system that does not output nan or inf neither
        in the training set (i.e., x) or test set (i.e., test).
        """
        if 'seed' in kwargs:
            seed = kwargs['seed']
            if seed is not None:
                seed = int(seed)
        else:
            seed = 0
        kwargs['seed'] = seed
        kwargs['training_size'] = x.shape[0]
        ins = cls.init_cl(**kwargs).train(x, f)
        if test is not None:
            ins.set_test(test)
        ins.run()
        test_f = lambda x: ((not np.any(np.isnan(x))) and
                            (not np.any(np.isinf(x))))
        a = test_f(ins.predict(x))
        if test is not None:
            b = test_f(ins.predict(test))
        else:
            b = True
        if a and b:
            return ins
        return None
