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
from SimpleGP.pde import PDE
from SimpleGP.tree import PDEXO
from SimpleGP.Rprop_mod import RPROP2


class GPPDE(GP):
    def __init__(self, max_mem=500.0,
                 update_best_w_rprop=False,
                 ppm2=0.1,
                 pm_only_functions=0,
                 **kwargs):
        super(GPPDE, self).__init__(**kwargs)
        self._max_mem = max_mem
        self._update_best_w_rprop = update_best_w_rprop
        self._p_st = np.empty(self._popsize, dtype=np.object)
        self._used_mem = 0
        self._ppm2 = ppm2
        self._pm_only_functions = pm_only_functions

    def new_best(self, k):
        flag = super(GPPDE, self).new_best(k)
        if not self._update_best_w_rprop or not flag:
            return flag
        cons = self._p_constants[k].copy()
        fit = self._fitness[k]
        self.rprop(k)
        flag = super(GPPDE, self).new_best(k)
        if flag:
            return flag
        self._fitness[k] = fit
        self._p_constants[k] = cons
        return True

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
        p_der_st = np.ones((self._max_length,
                            xs),
                           dtype=self._dtype,
                           order='C').nbytes
        return (p_der_st / 1024. / 1024.,
                p_st / 1024. / 1024.)

    def train(self, x, f):
        super(GPPDE, self).train(x, f)
        self.free_mem()
        self._p_der = np.empty((self._max_length, self._x.shape[0]),
                               dtype=self._dtype)
        self._pde = PDE(self._tree, self._p_der)
        if self._x.shape[1] < 10:
            self._tree.set_number_var_pm(self._x.shape[1])
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
            if hasattr(self, '_fitness'):
                self._fitness[i] = -np.inf

    def point_mutation(self, father1):
        if self._ppm2 == 0:
            self._npmutation = 1
            return self.one_point_mutation(father1)
        ind = father1.copy()
        index = np.zeros_like(ind)
        self.set_error_p_der()
        c = self._pde.compute_pdepm(ind,
                                    self._p_st[self._xo_father1],
                                    index, self._ppm2,
                                    self._pm_only_functions)
        # print c, index
        self._npmutation = c
        if c == 0:
            self._npmutation = 1
            return self.one_point_mutation(father1)
        constants = np.concatenate((self._p_constants[self._xo_father1],
                                    np.empty(c, dtype=self._dtype)))
        ncons = self._p_constants[self._xo_father1].shape[0]
        st = self._p_st[self._xo_father1]
        for i in index[:c]:
            e = np.sign(self._p_der[i])
            if ind[i] < self.nfunc:
                func = self._tree.pmutation_func_change(ind, i, st,
                                                        e, self._eval)
                ind[i] = func
            else:
                ncons += self._tree.pmutation_terminal_change(ind,
                                                              i, st,
                                                              e,
                                                              self._x,
                                                              constants,
                                                              ncons,
                                                              self._eval)
        # print ind, "*", index[:c]
        ind = self.simplify(ind, constants)
        # print ind, "-", index[:c]
        return ind

    def one_point_mutation(self, father1):
        sel_type = self._tree.get_type_xpoint_selection()
        self._tree.set_type_xpoint_selection(1)
        p1 = self._tree.father1_crossing_point(father1)
        if self._pm_only_functions:
            while father1[p1] >= self.nfunc:
                p1 = self._tree.father1_crossing_point(father1)
        self._tree.set_type_xpoint_selection(sel_type)
        ind = father1.copy()
        st = self._p_st[self._xo_father1]
        e = self.get_error(p1)
        if self.isfunc(ind[p1]):
            func = self._tree.pmutation_func_change(father1,
                                                    p1, st, e, self._eval)
            ind[p1] = func
            constants = self._p_constants[self._xo_father1].copy()
        else:
            constants = np.concatenate((self._p_constants[self._xo_father1],
                                        np.empty(1, dtype=self._dtype)))
            ncons = self._p_constants[self._xo_father1].shape[0]
            ncons += self._tree.pmutation_terminal_change(ind,
                                                          p1, st,
                                                          e,
                                                          self._x,
                                                          constants,
                                                          ncons,
                                                          self._eval)
        # print self._func[father1[p1]], self._func[func]
        ind = self.simplify(ind,
                            constants)
        return ind

    def mutation(self, father1):
        if father1.shape[0] > 1 and np.random.rand() < self._ppm:
            return self.point_mutation(father1)
        kill = self.tournament(neg=True)
        while kill == self._xo_father1 or kill == self._best:
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

    def tree_params(self, type_xpoint_selection=0):
        self._tree_length = np.empty(self._max_length,
                                     dtype=np.int)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=np.int)
        self._tree = PDEXO(self._nop,
                           self._tree_length,
                           self._tree_mask,
                           self._min_length,
                           self._max_length,
                           type_xpoint_selection=type_xpoint_selection)

    def set_error_p_der(self):
        self._computing_fitness = self._xo_father1
        e, g = self.compute_error_pr(None)
        self._p_der[self._output] = e.T

    def get_error(self, p1):
        self.set_error_p_der()
        self._pde.compute(self._p[self._xo_father1], p1,
                          self._p_st[self._xo_father1])
        e = np.sign(self._p_der[p1])
        return e

    def crossover(self, father1, father2, p1=-1, p2=-1,
                  force_xo=False):
        if p1 == -1:
            p1 = self._tree.father1_crossing_point(father1)
        if p2 == -1:
            e = self.get_error(p1)
            s = self._p_st[self._xo_father2]
            p = self._p_st[self._xo_father1][p1]
            self._tree.father2_xp_extras(e, p, s)
            p2 = self._tree.father2_crossing_point(father1, father2, p1)
        return super(GPPDE, self).crossover(father1, father2,
                                            p1, p2)

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

    def compute_error_pr(self, ind, pos=0, constants=None, epoch=0):
        if epoch == 0:
            g = self._p_st[self._computing_fitness][self._output].T
        else:
            if ind is None:
                g = self.eval(self._computing_fitness)
            else:
                g = self.eval_ind(ind, pos=pos, constants=constants)
        # e = - 2 * ( self._f - g)
        e = 2 * (g - self._f)
        return e, g

    def rprop(self, k, epochs=10000):
        """Update the constants of the tree using RPROP"""
        self._computing_fitness = k
        ind = self._p[k]
        constants = self._p_constants[k]
        self._computing_fitness = k
        if not self.any_constant(ind):
            return None
        best_cons = constants.copy()
        fit_best = self._fitness[k]
        epoch_best = 0
        rprop = RPROP2(ind, constants,
                       self._p_der, self._tree)
        e, g = self.compute_error_pr(None)
        self._p_der[self._output] = e.T
        for i in range(epochs):
            if i > 0:
                self.gens_ind += 1
            self._pde.compute_constants(ind, self._p_st[k])
            rprop.update_constants_rprop()
            e, g = self.compute_error_pr(None, epoch=i)
            fit = - self.distance(self._f, g)
            if fit > fit_best and not np.isnan(fit):
                fit_best = fit
                best_cons = constants.copy()
                epoch_best = i
            if i < epochs - 1:
                self._p_der[self._output] = e.T
            if i - epoch_best >= self._max_n_worst_epochs:
                break
        constants[:] = best_cons[:]
        self._fitness[k] = fit_best
        e, g = self.compute_error_pr(None, epoch=i)

    @classmethod
    def init_cl(cls, training_size=None,
                max_length=1024, max_mem=500, argmax_nargs=2,
                func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt', 'sin',
                      'cos', 'sigmoid', 'if', 'max', 'min', 'ln', 'sq',
                      'argmax'], seed=0, **kwargs):
        ins = cls(max_mem=max_mem, max_length=max_length,
                  argmax_nargs=argmax_nargs, func=func, seed=seed,
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
    def run_cl(cls, x, f, training_size=None, **kwargs):
        """
        Returns a trained system that does not output nan or inf neither
        in the training set (i.e., x) or test set (i.e., test).
        """
        if training_size is None:
            training_size = x.shape[0]
        return super(GPPDE, cls).run_cl(x, f,
                                        training_size=training_size, **kwargs)
