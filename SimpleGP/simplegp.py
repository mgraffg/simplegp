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
import types
import numpy as np
import os
from .simplega import SimpleGA
from .Simplify_mod import Simplify
from .eval import Eval
from .tree import Tree, PDEXO
from .Rprop_mod import RPROP, RpropSign


class GP(SimpleGA):
    def __init__(self,
                 func=['+', '-', '*', '/'],
                 mutation_depth=5, min_length=2,
                 nrandom=100, max_length=262143, verbose=False,
                 max_depth=7, max_length_subtree=np.inf,
                 min_depth=1, pgrow=0.5, pleaf=None,
                 compute_derivatives=False,
                 verbose_nind=None, argmax_nargs=None,
                 do_simplify=True, max_n_worst_epochs=5, ppm=0.0,
                 **kwargs):
        super(GP, self).__init__(**kwargs)
        self.individuals_params(do_simplify, min_depth,
                                max_depth, max_length, min_length,
                                max_length_subtree)
        self.constants_params(nrandom)
        self.genetic_operators_params(pgrow, pleaf, ppm,
                                      mutation_depth)
        self.derivative_params(compute_derivatives)
        self.max_length_checks()
        self.function_set(func, argmax_nargs)
        self.format_params(verbose, verbose_nind)
        self.eval_params(max_n_worst_epochs)
        self.min_max_length_params()
        self.tree_params()

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
        """This function set from all the functions available which ones
        form the function set"""
        self._output = 0
        self._output_pos = 15
        func = ['+', '-', '*', '/', 'abs', 'exp', 'sqrt',
                'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                'ln', 'sq', 'output', 'argmax']
        self._nop = np.array([2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                              3, 2, 2, 1, 1, -1, -1])
        self._func = np.asarray(func)
        self.__set_narg_to_argmax(argmax_nargs)
        key = {}
        [key.setdefault(v, k) for k, v in enumerate(self._func)]
        _func = filter(lambda x: x in key, _func)
        _func = filter(lambda x: x != 'output', _func)
        self._func_allow = np.array(map(lambda x: key[x], _func),
                                    dtype=np.int)
        self._max_nargs = self._nop[self._func_allow].max()
        if (self._func_allow == 16).sum() and self._nop[16] == -1:
            raise Exception("It is not specified the number\
            of argmax arguments")

    def available_functions(self):
        return self._func

    def __set_narg_to_argmax(self, nargs):
        """Setting the number of arguments to argmax"""
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
            self.length_per_gen = np.zeros(self._gens)
        self._left_p = "("
        self._right_p = ")"
        if verbose_nind is None:
            self._verbose_nind = self._popsize
        else:
            self._verbose_nind = verbose_nind

    def derivative_params(self, compute_derivatives):
        self._st = None
        self._p_der_st = None
        self._error_st = None
        self._compute_derivatives = compute_derivatives

    def min_max_length_params(self, minimum=None, maximum=None):
        if minimum is not None:
            self._min_length = minimum
        if maximum is not None:
            self._max_length = maximum

    def tree_params(self):
        self._tree_length = np.empty(self._max_length,
                                     dtype=np.int)
        self._tree_mask = np.empty(self._max_length,
                                   dtype=np.int)
        self._tree = Tree(self._nop,
                          self._tree_length,
                          self._tree_mask,
                          self._min_length,
                          self._max_length)

    def simplify(self, ind, constants=None):
        k = ind
        if isinstance(ind, types.IntType):
            ind = self._p[k]
            constants = self._p_constants[k]
        if constants is None:
            constants = self._constants
        if not self._do_simplify:
            c = np.where(ind >= (self._nop.shape[0] + self._x.shape[1]))[0]
            cons = np.zeros(c.shape[0], dtype=self._dtype)
            ncons = (self._nop.shape[0] + self._x.shape[1])
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
        self._eval = Eval(0, self._x.shape[1],
                          self._nop, self._max_nargs)
        self._st = None
        self._p_der_st = None
        self._error_st = None
        self._simplify = Simplify(x.shape[1], self._nop)
        self._simplify.set_constants(self._constants2)
        self._tree.set_nvar(self._x.shape[1])
        return self

    def predict(self, X, ind=None):
        if ind is None:
            ind = self.get_best()
        x = self._x
        f = self._f
        self.train(X, np.zeros(X.shape[0]))
        pr = self.eval(ind)
        self.train(x, f)
        return pr

    def new_best(self, k):
        pass

    def create_population(self):
        if self._fname_best is not None and\
           os.path.isfile(self._fname_best)\
           and self.load_prev_run():
            return
        self._p_constants = np.empty(self._popsize, dtype=np.object)
        self._p = np.empty(self._popsize, dtype=np.object)
        self._fitness = np.zeros(self._popsize)
        self._fitness[:] = -np.inf
        depth = self._max_depth
        for i in range(self._popsize):
            self._p[i] = self.create_random_ind(depth=depth)
            self._p_constants[i] = self._ind_generated_c
            depth -= 1
            if depth < self._min_depth:
                depth = self._max_depth

    def isfunc(self, a):
        return a < self._nop.shape[0]

    def isvar(self, a):
        nfunc = self._nop.shape[0]
        nvar = self._x.shape[1]
        return (a >= nfunc) and (a < nfunc+nvar)

    def isconstant(self, a):
        nfunc = self._nop.shape[0]
        nvar = self._x.shape[1]
        return a >= nfunc+nvar

    def any_constant(self, ind):
        return self._tree.any_constant(ind)

    def random_func(self, first_call=False):
        k = np.random.randint(self._func_allow.shape[0])
        return self._func_allow[k]

    def random_leaf(self):
        if np.random.rand() < 0.5 or self._nrandom == 0:
            l = np.random.randint(self._x.shape[1])
            return l + self._func.shape[0]
        else:
            l = np.random.randint(self._constants.shape[0])
            return l + self._func.shape[0] + self._x.shape[1]

    def create_random_ind_full(self, depth=3, first_call=True,
                               **kwargs):
        res = self.create_random_ind_full_inner(depth=depth,
                                                first_call=first_call)
        if isinstance(res, types.ListType):
            res = np.asarray(res)
        else:
            res = np.asarray([res])
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
            res = np.asarray(res)
        else:
            res = np.asarray([res])
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

    def crossover(self, father1, father2):
        ncons = self._p_constants[self._xo_father1].shape[0]
        ind = self._tree.crossover(father1,
                                   father2,
                                   ncons=ncons,
                                   p1=-1, p2=-1)
        if self._xo_father2 is not None:
            c2 = self._p_constants[self._xo_father2]
        else:
            c2 = self._constants2
        constants = np.concatenate((self._p_constants[self._xo_father1], c2))
        ind = self.simplify(ind, constants)
        if ind.shape[0] > self._max_length or ind.shape[0] < self._min_length:
            return self.create_random_ind()
        return ind

    def point_mutation(self, father1):
        try:
            cl_nop = self._cl_nop
        except AttributeError:
            cl_nop = {}
            for id in np.unique(self._nop):
                cl_nop[id] = np.where(self._nop == id)[0]
            self._cl_nop = cl_nop
        ind = father1.copy()
        ele = int(np.ceil(ind.shape[0] / 10.))
        index = np.arange(ind.shape[0])
        np.random.shuffle(index)
        index = index[:ele]
        for i in index:
            op = ind[i]
            if op < self._nop.shape[0]:
                a = cl_nop[self._nop[op]]
                np.random.shuffle(a)
                ind[i] = a[0]
            else:
                ind[i] = self.random_leaf()
        return ind

    def mutation(self, father1):
        if np.random.rand() < self._ppm:
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
            ind = self.get_best()
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
        if self._compute_derivatives:
            error_st, p_der_st = self.get_p_der_st(ind)
            e.set_p_der_st(p_der_st)
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
            if self._best_fit is None or self._best_fit < f:
                self._best_fit = f
                self.new_best(k)
        return f

    def compute_error_pr(self, ind, pos=0, constants=None, epoch=0):
        if epoch == 0:
            k = self._computing_fitness
            if k is None or (not hasattr(self, '_p_st')
                             or self._p_st[k] is None):
                g = self._st[self._output].T
            else:
                g = self._p_st[self._computing_fitness][self._output].T
        else:
            g = self.eval_ind(ind, pos=pos, constants=constants)
        e = g - self._f
        return e, g

    def rprop(self, ind=None, pos=0, constants=None,
              epochs=10000):
        """Update the constants of the tree using RPROP"""
        assert self._compute_derivatives
        self._computing_fitness = None
        k = ind
        if ind is None:
            ind = self._p[self.get_best()]
            constants = self._p_constants[self.get_best()]
            self._computing_fitness = self.get_best()
        if isinstance(ind, types.IntType):
            ind = self._p[k]
            constants = self._p_constants[k]
            self._computing_fitness = k
        if not self.any_constant(ind):
            return None
        constants if constants is not None else self._constants
        prev = np.zeros(ind.shape[0], dtype=self._dtype)
        slope = np.zeros(ind.shape[0], dtype=self._dtype)
        fit_best = -np.inf
        epoch_best = 0
        best_cons = constants.copy()
        rprop = RPROP(ind, constants, self._nop,
                      self._x.shape[1], self._p_der_st,
                      self._x.shape[0],
                      prev, slope,
                      max_nargs=self._max_nargs)
        error_st = self._error_st
        rprop.set_error_st(error_st)
        for i in range(epochs):
            if i > 0:
                self.gens_ind += 1
            e, g = self.compute_error_pr(ind, pos=pos, constants=constants,
                                         epoch=i)
            fit = - self.distance(self._f, g)
            if fit > fit_best and not np.isnan(fit):
                fit_best = fit
                best_cons = constants.copy()
                epoch_best = i
            if i < epochs - 1:
                rprop.set_zero_pos()
                error_st[self._output] = e.T
                rprop.update_constants_rprop()
            if i - epoch_best >= self._max_n_worst_epochs:
                break
        constants[:] = best_cons[:]
        if isinstance(k, types.IntType):
            self._fitness[k] = fit_best
            self.set_extras_to_ind(k, ind)

    def genetic_operators(self):
        if np.random.rand() < self._pxo:
            father1 = self.tournament()
            father2 = self.tournament()
            while father1 == father2:
                father2 = self.tournament()
            self._xo_father1 = father1
            self._xo_father2 = father2
            son = self.crossover(self._p[father1], self._p[father2])
        else:
            self._xo_father1 = self.tournament()
            son = self._p[self._xo_father1]
            son = self.mutation(son)
        return son

    def stats(self):
        i = self.gens_ind
        if i - self._last_call_to_stats < self._verbose_nind:
            return
        self._last_call_to_stats = i
        if self._stats:
            self.fit_per_gen[i/self._popsize] = self._fitness[self.get_best()]
            i_pop = i / self._popsize
            self.length_per_gen[i_pop] = np.asarray(map(lambda x: x.shape[0],
                                                        self._p)).mean()
        if self._verbose:
            print "Gen: " + str(i) + "/" + str(self._gens * self._popsize)\
                + " " + "%0.4f" % self._fitness[self.get_best()]

    def kill_ind(self, kill, son):
        super(GP, self).kill_ind(kill, son)
        self._p_constants[kill] = self._ind_generated_c
        self.set_extras_to_ind(kill, son, delete=True)

    def run(self):
        self.length_per_gen = np.zeros(self._gens)
        self.nodes_evaluated = 0
        return super(GP, self).run()

    def print_infix(self, ind=None, pos=0, constants=None):
        if ind is None or isinstance(ind, types.IntType):
            k = self.get_best() if ind is None else ind
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
        elif ind[pos] < self._func.shape[0] + self._x.shape[1]:
            return "X%s" % (ind[pos] - self._func.shape[0]), pos + 1
        else:
            c = ind[pos] - self._func.shape[0] - self._x.shape[1]
            return str(constants[c]), pos + 1

    def graphviz(self, ind=None, constants=None, fname=None,
                 var_names=None):
        import StringIO
        if ind is None or isinstance(ind, types.IntType):
            k = self.get_best() if ind is None else ind
            ind = self._p[k]
            if hasattr(self, '_p_constants'):
                constants = self._p_constants[k]
        if constants is None:
            constants = self._constants
        self._g_pos = 0
        if var_names is None:
            self._var_names = map(lambda x: "X%s" % x, range(self._x.shape[1]))
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
        elif ind[pos] < self._func.shape[0] + self._x.shape[1]:
            vn = self._var_names[ind[pos] - self._func.shape[0]]
            fpt.write("""n%s [label="%s"];\n""" % (pos, vn))
            return pos
        else:
            c = ind[pos] - self._func.shape[0] - self._x.shape[1]
            c = constants[c]
            fpt.write("""n%s [label="%0.4f"];\n""" % (pos, c))
            return pos

    def save(self, fname=None):
        fname = fname if fname is not None else self._fname_best
        if fname is None:
            return
        fpt = open(fname, 'w')
        np.save(fpt, self._p)
        np.save(fpt, self._p_constants)
        np.save(fpt, self._fitness)
        np.save(fpt, self.gens_ind)
        np.save(fpt, self.nodes_evaluated)
        if self._stats:
            np.save(fpt, self.fit_per_gen)
            np.save(fpt, self.length_per_gen)
        fpt.close()

    def load_prev_run(self):
        try:
            fpt = open(self._fname_best)
            self._p = np.load(fpt)
            self._p_constants = np.load(fpt)
            self._fitness = np.load(fpt)
            self.gens_ind = int(np.load(fpt))
            self.nodes_evaluated = np.load(fpt)
            if self._stats:
                self.fit_per_gen = np.load(fpt)
                self.length_per_gen = np.load(fpt)
            fpt.close()
            if self._p.dtype == np.object\
               and self._p.shape[0] == self._popsize:
                return True
        except IOError:
            pass
        return False

    @classmethod
    def init_cl(cls, popsize=1000, generations=50, verbose=False,
                verbose_nind=1000,
                argmax_nargs=2,
                func=["+", "-", "*", "/", 'abs', 'exp', 'sqrt',
                      'sin', 'cos', 'sigmoid', 'if', 'max', 'min',
                      'ln', 'sq', 'argmax'],
                fname_best=None,
                seed=0,
                pxo=0.9, pgrow=0.5, walltime=None, **kwargs):
        ins = cls(popsize=popsize, generations=generations,
                  argmax_nargs=argmax_nargs,
                  verbose=verbose, verbose_nind=verbose_nind,
                  func=func, fname_best=fname_best, seed=seed,
                  nrandom=0, pxo=pxo, pgrow=pgrow, walltime=walltime,
                  **kwargs)
        return ins


class GPMAE(GP):
    """This class exemplifies the change of the distance function.
    In the example, the distance is MAE then the derivative of this
    function is computed in the method compute_error_pr"""
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
            g = self.eval_ind(ind, pos=pos, constants=constants)
        e = self._f - g
        s = np.sign(e)
        e = -1 * s
        return e, g


class GPwRestart(GP):
    def create_population(self, flag=False):
        if flag or not hasattr(self, '_p'):
            super(GPwRestart, self).create_population()
        self._fitness.fill(-np.inf)

    def run(self, ntimes=2):
        cnt_ntimes = 0
        while not self._timeout and (cnt_ntimes < ntimes or ntimes <= 0):
            cnt_ntimes += 1
            self.init()
            flag = super(GPwRestart, self).run()
            if not flag:
                return False
            ind = self._p[self.get_best()].copy()
            cons = self._p_constants[self.get_best()].copy()
            fit = self.fitness(self.get_best())
            self.create_population(flag=True)
            self._p[0] = ind
            self._p_constants[0] = cons
            self._fitness[0] = fit
        return True

    def on_exit(self):
        super(GPwRestart, self).on_exit()
        self._run = True


class GPRPropU(GP):
    """This GP updates always the constants using RPROP."""
    def __init__(self, compute_derivatives=True, **kwargs):
        super(GPRPropU, self).__init__(compute_derivatives=True,
                                       **kwargs)

    def fitness(self, ind):
        fit = super(GPRPropU, self).fitness(ind)
        if not self._use_cache and isinstance(ind,
                                              types.IntType):
            if fit == -np.inf:
                return fit
            self.rprop(ind)
            fit2 = self._fitness[ind]
            if (fit2 > fit and (self._best_fit is None
                                or self._best_fit < fit2)):
                self._best_fit = fit2
                self.new_best(ind)
            return fit2
        return fit


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
        for i in range(self._popsize):
            self._p_st[i] = None
            self._p_error_st[i] = None
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

    def new_best(self, k):
        if self._update_best_w_rprop:
            fit2 = self._fitness[k]
            self.rprop(k)
            fit = self._fitness[k]
            if fit > fit2:
                self._best_fit = fit2

    def mem(self):
        l = filter(lambda x: self._p_st[x] is not None, range(self._popsize))
        st = np.array(map(lambda x: self._p_st[x].nbytes,
                          l)).sum() / 1024. / 1024.
        l = filter(lambda x: self._p_error_st[x] is not None,
                   range(self._popsize))
        est = np.array(map(lambda x: self._p_error_st[x].nbytes,
                           l)).sum() / 1024. / 1024.
        return st + est

    def free_mem(self):
        for i in range(self._popsize):
            self._p_st[i] = None
            self._p_error_st[i] = None
            self._fitness[i] = -np.inf

    def stats(self):
        super(GPPDE, self).stats()
        if self._last_call_to_stats == self.gens_ind:
            if self.mem() >= self._max_mem:
                self.free_mem()

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

    def crossover(self, father1, father2):
        error = self._p_error_st[self._xo_father1]
        x = self._p_st[self._xo_father1]
        s = self._p_st[self._xo_father2]
        self._tree.father2_xp_extras(error,
                                     x, s)
        return super(GPPDE, self).crossover(father1, father2)

    def get_p_der_st(self, ind):
        if self._p_der_st is None:
            self._p_der_st = np.ones((ind.shape[0],
                                      self._x.shape[0]*self._max_nargs),
                                     dtype=self._dtype,
                                     order='C')
        elif self._p_der_st.shape[0] < ind.shape[0]:
            self._p_der_st.resize((ind.shape[0],
                                   self._x.shape[0]*self._max_nargs))
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
            elif self._p_st[k].shape[0] < l:
                self._p_st[k].resize(l, self._x.shape[0])
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
        if self._p_error_st[ind].shape[0] < l:
            self._p_error_st[ind].resize(l, self._x.shape[0])
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
        """Compute the partial derivative of the error w.r.t. every node
        of the tree"""
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
        # print "terminando", k

    @classmethod
    def init_cl(cls, max_mem=300, verbose_nind=50, **kwargs):
        ins = cls(max_mem=max_mem, verbose_nind=verbose_nind,
                  **kwargs)
        return ins











