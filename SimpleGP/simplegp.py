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
import os
import types

import numpy as np
from SimpleGP.Simplify import Simplify
from SimpleGP.eval import Eval
from SimpleGP.tree import Tree

from SimpleGP.simplega import SimpleGA


class GP(SimpleGA):
    """
    Steady state Genetic Programming system with tournament selection,
    subtree crossover and mutation.

    - It simplifies each new individual if do_simplify is set to True.

    >>> import numpy as np
    >>> from SimpleGP import GP

    First let us create a simple regression problem
    >>> _ = np.random.RandomState(0)
    >>> x = np.linspace(0, 1, 100)
    >>> pol = np.array([0.2, -0.3, 0.2])
    >>> X = np.vstack((x**2, x, np.ones(x.shape[0]))).T
    >>> f = (X * pol).sum(axis=1)

    The objective is to find an expresion that approximate f
    >>> s = GP.init_cl(max_length=100).train(x[:, np.newaxis], f)
    >>> s.run()
    True

    The expression found is:
    >>> ex = s.print_infix()

    Eval the best expression found with a new inputs

    >>> x1 = np.linspace(-1, 1, 100)
    >>> pr = s.predict(x1[:, np.newaxis])
    """
    def __init__(self,
                 func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
                       'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                       'ln', 'sq', 'argmax'],
                 mutation_depth=5, min_length=2,
                 nrandom=100, max_length=262143, verbose=False,
                 max_depth=7, max_length_subtree=np.inf,
                 min_depth=1, pgrow=0.5, pleaf=None,
                 verbose_nind=None, argmax_nargs=2,
                 do_simplify=True, max_n_worst_epochs=3, ppm=0.0,
                 type_xpoint_selection=0,
                 ppm2=0.1,
                 pm_only_functions=0,
                 **kwargs):
        super(GP, self).__init__(**kwargs)
        self.individuals_params(do_simplify, min_depth,
                                max_depth, max_length, min_length,
                                max_length_subtree)
        self.constants_params(nrandom)
        self.genetic_operators_params(pgrow, pleaf, ppm,
                                      mutation_depth)
        self.st_params()
        self.max_length_checks()
        self.function_set(func, argmax_nargs)
        self.format_params(verbose, verbose_nind)
        self.eval_params(max_n_worst_epochs)
        self.min_max_length_params()
        self.tree_params(type_xpoint_selection)
        self._ppm2 = ppm2
        self._pm_only_functions = pm_only_functions

    def individuals_params(self, do_simplify, min_depth,
                           max_depth, max_length, min_length,
                           max_length_subtree):
        self._ind_generated_c = None
        self._do_simplify = do_simplify
        self._simplify = None
        self.nodes_evaluated = 0
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._max_length = max_length
        self._min_length = min_length
        self._max_length_subtree = max_length_subtree
        self._doing_tree = 0
        self._computing_fitness = 0

    def max_length_checks(self):
        if 2**self._min_depth < self._min_length:
            self._min_depth = int(np.ceil(np.log2(self._min_length)))
        try:
            depth = int(np.ceil(np.log2(self._max_length)))
            if depth < self._max_depth:
                self._max_depth = depth
            if depth < self._mutation_depth:
                self._mutation_depth = depth
            if self._mutation_depth <= self._min_depth:
                self._mutation_depth = self._min_depth + 1
        except OverflowError:
            pass

    def genetic_operators_params(self, pgrow, pleaf, ppm,
                                 mutation_depth):
        self._pgrow = pgrow
        self._pleaf = pleaf
        self._ppm = ppm
        self._mutation_depth = mutation_depth

    def constants_params(self, nrandom):
        self._nrandom = nrandom
        self.create_random_constants()
        t = self._nrandom if self._nrandom >= 100 else 100
        self._constants2 = np.zeros(t).astype(self._dtype)

    def create_random_constants(self):
        self._constants = np.random.uniform(-10,
                                            10,
                                            self._nrandom).astype(self._dtype)

    def function_set(self, _func, argmax_nargs=None):
        """
        This function set from all the functions available which ones
        form the function set
        """
        self._output = 0
        self._output_pos = 15
        func = ['+', '-', '*', '/', 'abs', 'exp', 'sqrt',
                'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                'ln', 'sq', 'output', 'argmax']
        self._nop = np.array([2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                              3, 2, 2, 1, 1, -1, -1], dtype=self._ind_dtype)
        self._func = np.asarray(func)
        self.__set_narg_to_argmax(argmax_nargs)
        key = {}
        [key.setdefault(v, k) for k, v in enumerate(self._func)]
        _func = filter(lambda x: x in key, _func)
        _func = filter(lambda x: x != 'output', _func)
        self._func_allow = np.array(map(lambda x: key[x], _func),
                                    dtype=self._ind_dtype)
        m = np.ones(self._nop.shape[0], dtype=np.bool)
        m[self._func_allow] = False
        self._nop[m] = -1
        self._max_nargs = self._nop[self._func_allow].max()
        if (self._func_allow == 16).sum() and self._nop[16] == -1:
            raise Exception("It is not specified the number\
            of argmax arguments")

    def available_functions(self):
        return self._func

    def __set_narg_to_argmax(self, nargs):
        """Setting the number of arguments to argmax"""
        self._argmax_nargs = nargs
        if nargs is None:
            return
        self._nop[16] = nargs

    def eval_params(self, max_n_worst_epochs):
        self._eval = None
        self._use_cache = False
        self._max_n_worst_epochs = max_n_worst_epochs

    def format_params(self, verbose, verbose_nind):
        self._verbose = verbose
        if self._stats:
            self.length_per_gen = np.zeros(self._generations)
        self._left_p = "("
        self._right_p = ")"
        if verbose_nind is None:
            self._verbose_nind = self._popsize
        else:
            self._verbose_nind = verbose_nind

    def st_params(self):
        self._st = None
        self._p_der_st = None
        self._error_st = None

    def min_max_length_params(self, minimum=None, maximum=None):
        if minimum is not None:
            self._min_length = minimum
        if maximum is not None:
            self._max_length = maximum

    def tree_params(self, type_xpoint_selection=0):
        self._type_xpoint_selection = type_xpoint_selection
        self._tree_length = np.empty(self._max_length,
                                     dtype=self._ind_dtype)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=self._ind_dtype)
        self._tree = Tree(self._nop,
                          self._tree_length,
                          self._tree_mask,
                          self._min_length,
                          self._max_length,
                          type_xpoint_selection=type_xpoint_selection)

    def early_stopping_save(self, k, fit_k=None):
        """
        Storing the best so far on the validation set.
        This funtion is called from early_stopping
        """
        assert fit_k
        self._early_stopping = [fit_k,
                                self.population[k].copy(),
                                self._p_constants[k].copy(),
                                self._pr_test_set.copy()]

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
            self._p_constants = self._p_constants[index]
        else:
            d = popsize - self._popsize
            self._p.resize(popsize)
            self._p_constants.resize(popsize)
            for i, ind, cons in self.create_population_generator(d):
                pos = i + self._popsize
                self._p[pos] = ind
                self._p_constants[pos] = cons
        self._popsize = popsize
        self.set_best()

    @property
    def nfunc(self):
        """Number of function in the function set"""
        return self._nop.shape[0]

    @property
    def nvar(self):
        """Number of independent variables"""
        return self._x.shape[1]

    def new_best_comparison(self, k):
        if self._best_fit is None:
            return True
        f = self._fitness[k]
        if self._best_fit > f:
            return False
        if self._best_fit == f:
            lbf = self.population[self.best].shape[0]
            lk = self.population[k].shape[0]
            if lbf < lk:
                return False
        return True

    def simplify(self, ind, constants=None):
        k = ind
        if isinstance(ind, types.IntType):
            ind = self._p[k]
            constants = self._p_constants[k]
        if constants is None:
            constants = self._constants
        if not self._do_simplify:
            c = np.where(ind >= (self._nop.shape[0] + self.nvar))[0]
            cons = np.zeros(c.shape[0], dtype=self._dtype)
            ncons = (self._nop.shape[0] + self.nvar)
            for _k, v in enumerate(c):
                cons[_k] = constants[ind[v] - ncons]
                ind[v] = _k + ncons
            self._ind_generated_c = cons
            if isinstance(k, types.IntType):
                self._p[k] = ind
                self._p_constants[k] = self._ind_generated_c
            return ind
        if ind.shape[0] >= self._constants2.shape[0]:
            self._constants2 = np.zeros(ind.shape[0]).astype(self._dtype)
            self._simplify.set_constants(self._constants2)
        ind = self._simplify.simplify(ind, constants)
        ncons = self._simplify.get_nconstants()
        self._ind_generated_c = self._constants2[:ncons].copy()
        if isinstance(k, types.IntType):
            self._p[k] = ind
            self._p_constants[k] = self._ind_generated_c
        return ind

    def train(self, x, f):
        super(GP, self).train(x, f)
        self._eval = Eval(0, self.nvar,
                          self._nop, self._max_nargs)
        self._st = None
        self._p_der_st = None
        self._error_st = None
        self._simplify = Simplify(x.shape[1], self._nop)
        self._simplify.set_constants(self._constants2)
        self._tree.set_nvar(self.nvar)
        return self

    def predict(self, X, ind=None):
        if ind is None:
            ind = self.best
        x = self._x.copy()
        init, end = 0, None
        Xs = X.shape[0]
        xs = x.shape[0]
        pr = None
        npcon = np.concatenate
        while end is None or init < Xs:
            end = xs if (init + xs) < Xs else Xs - init
            self._x[:end] = X[init:(end+init)]
            init += xs
            _pr = self.eval(ind).copy()
            pr = _pr[:end] if pr is None else npcon((pr, _pr[:end]))
        self._x[:] = x[:]
        self.eval(ind)
        return pr

    def create_population_generator(self, popsize=None):
        if popsize is None:
            popsize = self._popsize
        depth = self._max_depth
        for i in range(popsize):
            ind = self.create_random_ind(depth=depth)
            cons = self._ind_generated_c
            depth -= 1
            if depth < self._min_depth:
                depth = self._max_depth
            yield (i, ind, cons)

    def create_population(self):
        if self._fname_best is not None and\
           os.path.isfile(self._fname_best)\
           and self.load_prev_run():
            return False
        if self._p is not None:
            return False
        self._p_constants = np.empty(self._popsize, dtype=np.object)
        self._p = np.empty(self._popsize, dtype=np.object)
        self._fitness = np.zeros(self._popsize)
        self._fitness[:] = -np.inf
        for i, ind, cons in self.create_population_generator():
            self._p[i] = ind
            self._p_constants[i] = cons
        return True

    def isfunc(self, a):
        return a < self.nfunc

    def isvar(self, a):
        nfunc = self.nfunc
        nvar = self.nvar
        return (a >= nfunc) and (a < nfunc+nvar)

    def isconstant(self, a):
        nfunc = self.nfunc
        nvar = self.nvar
        return a >= nfunc+nvar

    def any_constant(self, ind):
        return self._tree.any_constant(ind)

    def random_func(self, first_call=False):
        k = np.random.randint(self._func_allow.shape[0])
        return self._func_allow[k]

    def random_leaf(self):
        if np.random.rand() < 0.5 or self._nrandom == 0:
            l = np.random.randint(self.nvar)
            return l + self.nfunc
        else:
            l = np.random.randint(self._constants.shape[0])
            return l + self.nfunc + self.nvar

    def create_random_ind_full(self, depth=3, first_call=True,
                               **kwargs):
        res = self.create_random_ind_full_inner(depth=depth,
                                                first_call=first_call)
        if isinstance(res, types.ListType):
            res = np.asarray(res, dtype=self._ind_dtype)
        else:
            res = np.asarray([res], dtype=self._ind_dtype)
        ind = res
        ind = self.simplify(ind)
        return ind

    def create_random_ind_full_inner(self, depth=3, first_call=False):
        if depth == 0:
            return self.random_leaf()
        else:
            op = self.random_func(first_call)
            if isinstance(op, types.ListType):
                return op
            nargs = self._nop[op]
            res = [op]
            for i in range(nargs):
                if first_call:
                    self._doing_tree = i
                tmp = self.create_random_ind_full_inner(depth-1)
                if isinstance(tmp, types.ListType):
                    res += tmp
                else:
                    res.append(tmp)
            return res

    def create_random_ind_grow(self, depth=3,
                               first_call=True,
                               **kwargs):
        res = self.create_random_ind_grow_inner(depth=depth,
                                                first_call=first_call)
        if isinstance(res, types.ListType):
            res = np.asarray(res, dtype=self._ind_dtype)
        else:
            res = np.asarray([res], dtype=self._ind_dtype)
        ind = res
        ind = self.simplify(ind)
        return ind

    def create_random_ind_grow_inner(self, depth=3, first_call=False):
        if depth == 0:
            return self.random_leaf()
        elif first_call or np.random.rand() < 0.5:
            op = self.random_func(first_call)
            if isinstance(op, types.ListType):
                return op
            nargs = self._nop[op]
            res = [op]
            for i in range(nargs):
                if first_call:
                    self._doing_tree = i
                tmp = self.create_random_ind_grow_inner(depth-1)
                if isinstance(tmp, types.ListType):
                    res += tmp
                else:
                    res.append(tmp)
            return res
        else:
            return self.random_leaf()

    def create_random_ind(self, depth=4, first_call=True):
        ind = None
        while (ind is None or ind.shape[0] > self._max_length
               or ind.shape[0] < self._min_length):
            if np.random.rand() < self._pgrow:
                ind = self.create_random_ind_grow(depth=depth,
                                                  first_call=first_call)
            else:
                ind = self.create_random_ind_full(depth=depth,
                                                  first_call=first_call)
            if depth > self._min_depth:
                depth -= 1
        return ind

    def traverse(self, ind, pos=0):
        return self._tree.traverse(ind, pos=pos)

    def subtree_selection(self, father):
        if father.shape[0] == 1:
            return 0
        if self._pleaf is None:
            return np.random.randint(father.shape[0])
        if np.random.rand() < 1 - self._pleaf:
            points = np.where(father < self._func.shape[0])[0]
            if points.shape[0] > 0:
                point = np.random.randint(points.shape[0])
                return points[point]
        points = np.where(father >= self._func.shape[0])[0]
        point = np.random.randint(points.shape[0])
        return points[point]

    def length(self, ind):
        # l_p2 = np.zeros_like(ind)
        p = self._tree.length(ind)
        return self._tree_length[:p]

    def crossover(self, father1, father2, p1=-1,
                  p2=-1):
        """
        Performs subtree crossover. p1 and p2 are the crossing points
        where -1 indicates that these points are computed in self._tree
        """
        ncons = self._p_constants[self._xo_father1].shape[0]
        ind = self._tree.crossover(father1,
                                   father2,
                                   ncons=ncons,
                                   p1=p1, p2=p2)
        if self._xo_father2 is not None:
            c2 = self._p_constants[self._xo_father2]
        else:
            c2 = self._constants2
        constants = np.concatenate((self._p_constants[self._xo_father1], c2))
        ind = self.simplify(ind, constants)
        if ind.shape[0] > self._max_length or ind.shape[0] < self._min_length:
            return self.create_random_ind()
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
        constants = self._p_constants[self._xo_father1].copy()
        if self.isfunc(ind[p1]):
            self._tree.pmutation_func_change(ind, p1)
        else:
            self._tree.pmutation_terminal_change(ind, p1,
                                                 constants,
                                                 self._constants)
        ind = self.simplify(ind, constants)
        if ind.shape[0] > self._max_length or ind.shape[0] < self._min_length:
            return self.create_random_ind()
        return ind

    def point_mutation(self, father1):
        if self._ppm2 == 0:
            self._npmutation = 1
            return self.one_point_mutation(father1)
        index = np.zeros_like(father1)
        c = self._tree.select_pm(father1, index, self._pm_only_functions,
                                 self._ppm2)
        if c == 0:
            self._npmutation = 1
            return self.one_point_mutation(father1)
        ind = father1.copy()
        constants = self._p_constants[self._xo_father1].copy()
        for i in range(c):
            p1 = index[i]
            if ind[p1] < self.nfunc:
                self._tree.pmutation_func_change(ind, p1)
            else:
                self._tree.pmutation_terminal_change(ind, p1,
                                                     constants,
                                                     self._constants)
        ind = self.simplify(ind, constants)
        if ind.shape[0] > self._max_length or ind.shape[0] < self._min_length:
            return self.create_random_ind()
        return ind

    def mutation(self, father1):
        if father1.shape[0] > 1 and np.random.rand() < self._ppm:
            return self.point_mutation(father1)
        d = np.random.randint(self._min_depth,
                              self._mutation_depth)
        father2 = self.create_random_ind(depth=d,
                                         first_call=True)
        self._xo_father2 = None
        ind = self.crossover(father1, father2)
        return ind

    def set_extras_to_ind(self, k, *args, **kwargs):
        pass

    def eval(self, ind=None, **kwargs):
        if ind is None:
            ind = self.best
        self._computing_fitness = None
        if isinstance(ind, types.IntType):
            self._computing_fitness = ind
            return self.eval_ind(self._p[ind],
                                 pos=0,
                                 constants=self._p_constants[ind],
                                 **kwargs)
        return self.eval_ind(ind, **kwargs)

    def get_p_der_st(self, ind):
        if self._p_der_st is None:
            self._error_st = np.ones_like(self._st)
            self._p_der_st = np.ones((self._st.shape[0],
                                      self._st.shape[1]*self._max_nargs),
                                     dtype=self._dtype,
                                     order='C')
        elif self._p_der_st.shape[0] < ind.shape[0]:
            self._error_st.resize(self._st.shape)
            self._p_der_st.resize((self._st.shape[0],
                                   self._st.shape[1]*self._max_nargs))
            self._error_st.fill(1)
            self._p_der_st.fill(1)
        return self._error_st, self._p_der_st

    def get_st(self, ind):
        if self._st is None:
            self._st = np.empty((ind.shape[0], self._x.shape[0]),
                                dtype=self._dtype, order='C')
        elif self._st.shape[0] < ind.shape[0]:
            self._st.resize((ind.shape[0], self._x.shape[0]))
        return self._st

    def eval_ind(self, ind, pos=0, constants=None):
        c = constants if constants is not None else self._constants
        self.nodes_evaluated += ind.shape[0]
        st = self.get_st(ind)
        e = self._eval
        e.set_pos(0)
        e.eval_ind(ind,
                   self._x,
                   st,
                   c)
        g = st[self._output].T
        return g

    def fitness(self, ind):
        self._use_cache = False
        self._computing_fitness = None
        k = ind
        if isinstance(ind, types.IntType):
            self._computing_fitness = ind
            if self._fitness[k] > -np.inf:
                self._use_cache = True
                return self._fitness[k]
            constants = self._p_constants[ind]
            ind = self._p[ind]
        else:
            constants = self._ind_generated_c
        f = self.eval_ind(ind, pos=0, constants=constants)
        f = - self.distance(self._f, f)
        if np.isnan(f):
            f = -np.inf
        if isinstance(k, types.IntType):
            self.set_extras_to_ind(k, ind=ind,
                                   constants=constants)
            self._fitness[k] = f
            self.new_best(k)
            f = self._fitness[k]
        return f

    def pre_crossover(self, father1=None, father2=None):
        """
        This function is called before calling crossover, here
        it is set _xo_father1 and _xo_father2 which contains
        the position where the parents are.
        """
        father1 = self.tournament() if father1 is None else father1
        father2 = self.tournament() if father2 is None else father2
        while not super(GP,
                        self).pre_crossover(father1,
                                            father2):
            father2 = self.tournament()
        self._xo_father1 = father1
        self._xo_father2 = father2
        if self._tree.equal_non_const(self._p[father1],
                                      self._p[father2]):
            return False
        return True

    def genetic_operators(self):
        if np.random.rand() < self._pxo and self.pre_crossover():
            son = self.crossover(self._p[self._xo_father1],
                                 self._p[self._xo_father2])
        else:
            self._xo_father1 = self.tournament()
            son = self._p[self._xo_father1]
            son = self.mutation(son)
        return son

    def stats(self):
        i = self.gens_ind
        if i - self._last_call_to_stats < self._verbose_nind:
            return False
        self._last_call_to_stats = i
        if self._stats:
            self.fit_per_gen[i/self._popsize] = self._fitness[self.best]
            i_pop = i / self._popsize
            self.length_per_gen[i_pop] = np.asarray(map(lambda x: x.shape[0],
                                                        self._p)).mean()
        if self._verbose:
            bf = self._best_fit
            if bf is None:
                bf = -1.
            print "Gen: " + str(i) + "/" + str(self._generations * self._popsize)\
                + " " + "%0.4f" % bf
        return True

    def kill_ind(self, kill, son):
        super(GP, self).kill_ind(kill, son)
        self._p_constants[kill] = self._ind_generated_c
        self.set_extras_to_ind(kill, son, delete=True)

    def run(self, exit_call=True):
        self.length_per_gen = np.zeros(self._generations)
        self.nodes_evaluated = 0
        return super(GP, self).run(exit_call=exit_call)

    def print_infix(self, ind=None, pos=0, constants=None):
        if ind is None or isinstance(ind, types.IntType):
            k = self.best if ind is None else ind
            ind = self._p[k]
            if hasattr(self, '_p_constants'):
                constants = self._p_constants[k]
        if constants is None:
            constants = self._constants
        cdn, pos = self._print_infix(ind, pos=pos, constants=constants)
        return cdn

    def _print_infix(self, ind, pos=0, constants=None):
        if ind[pos] < self._func.shape[0]:
            func = self._func[ind[pos]]
            nargs = self._nop[ind[pos]]
            cdn = " "+func+" "
            pos += 1
            args = []
            for i in range(nargs):
                r, pos = self._print_infix(ind, pos, constants=constants)
                args.append(r)
            if nargs == 1:
                cdn = cdn[1:-1]
                cdn += self._left_p+args[0]+self._right_p
            elif (nargs == 2 and func != 'max' and
                  func != 'min'):
                cdn = cdn.join(args)
            else:
                cdn += self._left_p+",".join(args)+self._right_p
            if nargs == 2:
                return "("+cdn+")", pos
            else:
                return cdn, pos
        elif ind[pos] < self._func.shape[0] + self.nvar:
            return "X%s" % (ind[pos] - self._func.shape[0]), pos + 1
        else:
            c = ind[pos] - self._func.shape[0] - self.nvar
            return str(constants[c]), pos + 1

    def graphviz(self, ind=None, constants=None, fname=None,
                 var_names=None):
        import StringIO
        if ind is None or isinstance(ind, types.IntType):
            k = self.best if ind is None else ind
            ind = self._p[k]
            if hasattr(self, '_p_constants'):
                constants = self._p_constants[k]
        if constants is None:
            constants = self._constants
        self._g_pos = 0
        if var_names is None:
            self._var_names = map(lambda x: "X%s" % x, range(self.nvar))
        else:
            self._var_names = var_names
        if isinstance(fname, types.FileType):
            s = fname
        elif isinstance(fname, StringIO.StringIO):
            s = fname
        elif fname is None:
            s = StringIO.StringIO()
        else:
            s = open(fname, 'w')
        s.write("digraph SimpleGP {\n")
        s.write("""edge [dir="none"];\n""")
        self._graphviz(ind, constants, s)
        s.write("}\n")
        if isinstance(s, StringIO.StringIO):
            s.seek(0)
            print s.read()

    def _graphviz(self, ind, constants, fpt):
        pos = self._g_pos
        self._g_pos += 1
        if ind[pos] < self._nop.shape[0]:
            func = self._func[ind[pos]]
            nargs = self._nop[ind[pos]]
            fpt.write("""n%s [label="%s"];\n""" % (pos, func))
            for i in range(nargs):
                posd = self._graphviz(ind, constants, fpt)
                fpt.write("""n%s -> n%s;\n""" % (pos, posd))
            return pos
        elif ind[pos] < self._func.shape[0] + self.nvar:
            vn = self._var_names[ind[pos] - self._func.shape[0]]
            fpt.write("""n%s [label="%s"];\n""" % (pos, vn))
            return pos
        else:
            c = ind[pos] - self._func.shape[0] - self.nvar
            c = constants[c]
            fpt.write("""n%s [label="%0.4f"];\n""" % (pos, c))
            return pos

    def save_extras(self, fpt):
        pass

    def save(self, fname=None):
        import gzip
        import tempfile
        import os

        def save_inner(fpt):
            np.save(fpt, self._p)
            np.save(fpt, self._p_constants)
            np.save(fpt, self._fitness)
            np.save(fpt, self.gens_ind)
            np.save(fpt, self.nodes_evaluated)
            if self._stats:
                np.save(fpt, self.fit_per_gen)
                np.save(fpt, self.length_per_gen)
            self.save_extras(fpt)

        fname = fname if fname is not None else self._fname_best
        fname_is_gzip = False
        if fname is None:
            return False
        if self._save_only_best:
            self.clear_population_except_best()
        if fname.count('.gz'):
            fname_is_gzip = True
            fname_o = fname
            fname = tempfile.mktemp()
        with open(fname, 'wb') as fpt:
            save_inner(fpt)
        if fname_is_gzip:
            with open(fname, 'rb') as original:
                with gzip.open(fname_o, 'wb') as fpt:
                    fpt.writelines(original)
            os.unlink(fname)
        return True

    def clear_population_except_best(self):
        mask = super(GP, self).clear_population_except_best()
        self._p_constants[mask] = None
        return mask

    def save_best(self, fname=None):
        """
        Save only best individual
        """
        self.clear_population_except_best()
        return self.save(fname=fname)

    def set_best(self):
        bsf = self._fitness.max()
        m = np.where(bsf == self._fitness)[0]
        for i in m:
            self._fitness[i] = -np.inf
            self.fitness(i)

    def load_extras(self, fpt):
        pass

    def load_prev_run(self):
        import gzip
        import tempfile
        import os

        def load(fpt):
            self._p = np.load(fpt)
            self._p_constants = np.load(fpt)
            arr = filter(lambda x: self._p[x] is None,
                         range(self._p.shape[0]))
            if len(arr):
                cp_gen = self.create_population_generator
                for i, ind, c in cp_gen(len(arr)):
                    _i = arr[i]
                    self.population[_i] = ind
                    self._p_constants[_i] = c
            self._fitness = np.load(fpt)
            self.gens_ind = int(np.load(fpt))
            self.nodes_evaluated = np.load(fpt)
            if self._stats:
                self.fit_per_gen = np.load(fpt)
                self.length_per_gen = np.load(fpt)
            self.load_extras(fpt)

        try:
            fname = self._fname_best
            if self._fname_best.count('.gz'):
                fname = tempfile.mktemp()
                with gzip.open(self._fname_best, 'rb') as original:
                    with open(fname, 'wb') as fpt:
                        fpt.writelines(original)
            with open(fname, 'rb') as fpt:
                load(fpt)
            if self._fname_best.count('.gz'):
                os.unlink(fname)
            self.set_best()
            if self._p.dtype == np.object\
               and self._p.shape[0] == self._popsize:
                return True
        except IOError:
            pass
        return False

    @classmethod
    def init_cl(cls, argmax_nargs=2,
                func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
                      'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                      'ln', 'sq', 'argmax'],
                seed=0, **kwargs):
        ins = cls(argmax_nargs=argmax_nargs,
                  func=func, seed=seed, **kwargs)
        return ins

    @classmethod
    def max_time_per_eval(cls, x, y,
                          popsize=1000,
                          max_length=1024,
                          seed=0,
                          **kwargs):
        import time

        class G(cls):
            def __init__(self, **kwargs):
                super(G, self).__init__(**kwargs)
                self.wstime = 0

            def fitness(self, ind):
                init = time.time()
                fit = super(G, self).fitness(ind)
                t = time.time() - init
                if self.wstime < t:
                    self.wstime = t
                return fit
        kwargs['func'] = ['+', '*', 'argmax']
        kwargs['nrandom'] = 0
        kwargs['argmax_nargs'] = 2
        kwargs['generations'] = 10
        max_depth = int(np.ceil(np.log2(max_length)))
        g = G.run_cl(x, y, max_length=max_length,
                     pgrow=0, verbose=True,
                     max_depth=max_depth,
                     min_depth=max_depth-1,
                     seed=0, popsize=popsize,
                     **kwargs)
        return g.wstime


class GPMAE(GP):
    """
    This class exemplifies the change of the distance function.
    In the example, the distance is MAE then the derivative of this
    function is computed in the method compute_error_pr
    """
    def distance(self, y, yh):
        return np.fabs(y - yh).mean()

    def compute_error_pr(self, ind, pos=0, constants=None, epoch=0):
        if epoch == 0:
            k = self._computing_fitness
            if k is None or (not hasattr(self, '_p_st')
                             or self._p_st[k] is None):
                g = self._st[self._output].T
            else:
                g = self._p_st[self._computing_fitness][self._output].T
        else:
            if ind is None:
                g = self.eval(self._computing_fitness)
            else:
                g = self.eval_ind(ind, pos=pos, constants=constants)
        e = self._f - g
        s = np.sign(e)
        e = -1 * s
        return e, g


class GPwRestart(GP):
    def __init__(self, ntimes=2, **kwargs):
        super(GPwRestart, self).__init__(**kwargs)
        self._ind_eval_per_gen = self.generations * self.popsize
        self.generations = self.generations * ntimes
        self._ntimes = 0

    def stats(self):
        tot = self.generations * self.popsize
        gens_ind = self.gens_ind
        if 0 == (gens_ind % self._ind_eval_per_gen) and gens_ind < tot:
            # print "*"*10, self.gens_ind
            self._ntimes += 1
            for i, ind, cons in self.create_population_generator(self.popsize):
                if i == self.best:
                    continue
                self._p[i] = ind
                self._p_constants[i] = cons
                self._fitness[i] = -np.inf
        return super(GPwRestart, self).stats()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
