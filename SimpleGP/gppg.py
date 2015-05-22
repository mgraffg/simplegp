# Copyright 2015 Mario Graff Guerrero

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
from SimpleGP.forest import SubTreeXO
from SimpleGP.sparse_array import SparseEval, SparseArray
from SimpleGP.Simplify import Simplify
import tempfile
import shutil


class SparseGPPG(SubTreeXO):
    def __init__(self, prototypes=[], tree_cl=None,
                 **kwargs):
        super(SparseGPPG, self).__init__(**kwargs)
        self._prototypes = prototypes
        if tree_cl is not None:
            self._tree_cl = np.array(tree_cl)
        else:
            self._tree_cl = tree_cl
        self._pg_cl = None
        self._pg_d = None
        self._prototypes_argsort = None

    @property
    def prototypes(self):
        ps = self._prototypes_argsort
        p = self._prototypes + [(self.population[self.best],
                                 self._p_constants[self.best],
                                 self._tree_cl)]
        if ps is None:
            _, perf = self.prototypes_performance(p)
            perf = perf.tolist()
            cnt = []
            for cl in map(lambda x: x[2], p):
                if len(cl) > 1:
                    s = sum(perf[:len(cl)])
                    del perf[:len(cl)]
                else:
                    s = perf[0]
                    del perf[0]
                cnt.append(s)
            ps = np.argsort(cnt)[::-1]
            self._prototypes_argsort = ps
        return map(lambda x: p[x], ps)

    def save_extras(self, fpt):
        super(SparseGPPG, self).save_extras(fpt)
        np.save(fpt, self._prototypes)
        np.save(fpt, self._tree_cl)

    def load_extras(self, fpt):
        super(SparseGPPG, self).load_extras(fpt)
        self._prototypes += np.load(fpt).tolist()
        self._tree_cl = np.load(fpt)
        self._ntrees = self._tree_cl.shape[0]
        self.init_ntrees()
        self.eval_prev_prototypes()

    def associate_terminals_cl(self):
        self._type_terminals = np.zeros(self._ntrees,
                                        dtype=np.object)
        for k, v in enumerate(self._tree_cl):
            self._type_terminals[k] = np.where(self._f == v)[0]

    def init_ntrees(self):
        cdn = "ntress must be greater or equal to the number of classes"
        if self._ntrees is None and self._tree_cl is None:
            cl = np.unique(self._f)
            self._ntrees = cl.shape[0]
            self._tree_cl = np.array(map(lambda x: x % cl.shape[0],
                                         range(self._ntrees)))
        elif self._tree_cl is not None:
            self._ntrees = len(self._tree_cl)
        elif self._ntrees is not None:
            cl = np.unique(self._f)
            if cl.shape[0] > self._ntrees:
                raise NotImplementedError(cdn)
            self._tree_cl = np.array(map(lambda x: x % cl.shape[0],
                                         range(self._ntrees)))
        if len(self._tree_cl) != self._ntrees:
            raise NotImplementedError(cdn)
        self._nop[self._output_pos] = self._ntrees
        self._ntrees_per_class_mod = self._ntrees
        self.associate_terminals_cl()
        self._dist_matrix = np.zeros((self._ntrees, self._nvar))

    def train(self, x, f):
        self._eval = SparseEval(self._nop)
        self._eval.X(x)
        self._nvar = len(x)
        self._x = x
        self._f = f
        self._st = None
        self._p_der_st = None
        self._error_st = None
        self._simplify = Simplify(self.nvar, self._nop)
        self._simplify.set_constants(self._constants2)
        self._tree.set_nvar(self.nvar)
        self.init_ntrees()
        self.eval_prev_prototypes()
        return self

    def random_leaf(self):
        """Choose a leaf depending on the i-th argument of root, i.e.,
        Output function  """
        if np.random.rand() < 0.5 or self._nrandom == 0:
            i = self._doing_tree
            cnt = self._type_terminals[i].shape[0]
            l = np.random.randint(cnt)
            l = self._type_terminals[i][l]
            return l + self.nfunc
        else:
            l = np.random.randint(self._constants.shape[0])
            return l + self.nfunc + self.nvar

    @property
    def nvar(self):
        """Number of independent variables"""
        return self._nvar

    def create_population(self):
        flag = super(SparseGPPG, self).create_population()
        self.eval_prev_prototypes()
        return flag

    def eval_prev_prototypes(self):
        l, cl = self.eval_prototypes(self._prototypes)
        if len(l):
            tree_cl = cl
            D = np.zeros((len(l), len(self._x)))
            SparseArray.distance(self._x, l, D)
            s = D.argmin(axis=0)
            self._pg_cl = tree_cl[s]
            self._pg_d = D[s, np.arange(s.shape[0])]
        else:
            self._pg_cl = None
            self._pg_d = None

    def eval_prototypes(self, prototypes):
        if len(prototypes) == 0:
            return [], []
        l = []
        cl = []
        for ind, c, tree_cl in prototypes:
            self._nop[self._output_pos] = tree_cl.shape[0]
            self._eval.eval(ind, c, to_np_array=False)
            l += self._eval.get_output()
            cl.append(tree_cl)
        self._nop[self._output_pos] = self._ntrees
        return l, np.concatenate(cl)

    def prototypes_performance(self, prototypes=None):
        if prototypes is None:
            prototypes = self.prototypes
        l, cl = self.eval_prototypes(prototypes)
        D = np.zeros((len(l), len(self._x)))
        SparseArray.distance(self._x, l, D)
        s = D.argmin(axis=0)
        perf = np.zeros(len(l))
        for k, v in zip(s, cl[s] == self._f):
            perf[k] += v
        s = np.array(map(lambda x: (s == x).sum(), range(len(l))))
        return s, perf

    def eval_ind(self, ind, pos=0, constants=None):
        c = constants if constants is not None else self._constants
        self.nodes_evaluated += ind.shape[0]
        self._eval.eval(ind, c, to_np_array=False)
        yh = self._eval.get_output()
        SparseArray.distance(self._x, yh, self._dist_matrix)
        mn_finite = ~np.isfinite(self._dist_matrix)
        self._dist_matrix[mn_finite] = np.inf
        s = self._dist_matrix.argmin(axis=0)
        yh = self._tree_cl[s]
        if self._pg_d is not None:
            d = self._dist_matrix[s, np.arange(s.shape[0])]
            W = np.vstack((self._pg_d, d))
            s = W.argmin(axis=0)
            yh = np.vstack((self._pg_cl, yh))[s, np.arange(s.shape[0])]
            self._dist_matrix_W = W
        return yh

    def predict(self, X, ind=None, nprototypes=None, prototypes=None):
        if ind is None:
            ind = self.best
        p = (self.population[ind],
             self._p_constants[ind],
             self._tree_cl)
        if prototypes is None:
            prototypes = self._prototypes + [p]
        npro = nprototypes if nprototypes is not None else len(prototypes)
        l, cl = self.eval_prototypes(prototypes[:npro])
        D = np.zeros((len(l), len(X)))
        SparseArray.distance(X, l, D)
        s = D.argmin(axis=0)
        return cl[s]

    def distance(self, _y, _yh):
        return -self.recall(_y, _yh).mean()

    def verbose(self, i):
        if self._verbose:
            p = self.precision(self._f, self.eval())
            r = self.recall(self._f, self.eval())
            print "Iter:", i, "R:", map(lambda x: "%0.4f" % x, r),\
                "P:", map(lambda x: "%0.4f" % x, p)

    @classmethod
    def run_cl(cls, X, y, nprototypes=10, fname_best=None,
               func=['+', '-', 'abs', 'sin', 'sq', 'sqrt', 'sigmoid', 'if'],
               tol=0.0, func_select=None,
               nrandom=0, verbose=False, max_length=512, tree_cl=None,
               seed=0, prototypes=None, **kwargs):
        prototypes = [] if prototypes is None else prototypes
        func_select = cls.recall if func_select is None else func_select
        nprot = 0
        fbest = -np.inf
        fname = None
        while nprot < nprototypes:
            if fname_best is not None:
                if fname is None:
                    fname = fname_best
                else:
                    fname = tempfile.mktemp() + ".npy"
                    if fname_best.count('.gz'):
                        fname += '.gz'
            gp = cls(fname_best=fname, prototypes=prototypes,
                     tree_cl=tree_cl, nrandom=nrandom, verbose=verbose,
                     func=func, max_length=max_length,
                     seed=seed, **kwargs).train(X, y)
            gp.run()
            nprot = len(gp.prototypes)
            if fname_best is not None and fname != fname_best:
                shutil.move(fname, fname_best)
            gp.verbose(nprot)
            if fbest > -np.inf and fbest > gp.fitness(gp.best):
                return gp
            else:
                fbest = gp.fitness(gp.best)
                prototypes = gp.prototypes
                r = func_select(gp._f, gp.eval())
                tree_cl = np.where((r - r.min()) <= tol)[0].tolist()
        return gp

    @classmethod
    def f1(cls, y, yh):
        p = cls.precision(y, yh)
        r = cls.recall(y, yh)
        f1 = 2 * p * r / (p + r)
        f1[~np.isfinite(f1)] = -np.inf
        return f1

    @staticmethod
    def recall(y, yh):
        l = []
        for cl in np.unique(y):
            m = y == cl
            r = (yh[m] == cl).sum() / float(m.sum())
            l.append(r)
        return np.array(l)

    @staticmethod
    def precision(y, yh):
        l = []
        for cl in np.unique(y):
            m = yh == cl
            p = (y[m] == cl).sum() / float(m.sum())
            l.append(p)
        return np.array(l)


class SparseGPPGD(SparseGPPG):
    def distance(self, y, yh):
        W = self._dist_matrix
        if self._pg_d is not None:
            W = self._dist_matrix_W
        W = W.min(axis=0)
        max = W.max()
        lst = []
        for i in np.unique(y):
            m = y == i
            y1 = y[m]
            yh1 = yh[m]
            W1 = W[m]
            m1 = y1 == yh1
            avg = W1[m1].mean() if m1.sum() else 0
            wrs = max * (~m1).sum() / float(m.shape[0])
            lst.append(avg + wrs)
        return sum(lst)
