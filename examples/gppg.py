import numpy as np
from SimpleGP import SubTreeXO
from sklearn.metrics.pairwise import euclidean_distances


class GPPG(SubTreeXO):
    def train(self, x, f):
        super(GPPG, self).train(x.T, f)
        return self

    def eval_ind_X(self, ind, **kwargs):
        trees = super(GPPG, self).eval_ind(ind, **kwargs)
        if np.any(np.isnan(trees)) or np.any(np.isinf(trees)):
            return None
        return trees

    def eval_ind(self, ind, x=None, **kwargs):
        """Eval a tree when the tree is a set of prototypes,
        one prototype per class"""
        trees = self.eval_ind_X(ind, **kwargs)
        if trees is None:
            r = np.empty_like(self._f)
            r.fill(np.inf)
            return r
        x = self._x.T if x is None else x
        d = euclidean_distances(x, trees.T)
        return d.argmin(axis=1)

    def distance(self, t, f):
        cl = np.where(t == f, 1, 0).sum()
        return 1 - cl / float(t.shape[0])


class GPPG2(GPPG):
    def __init__(self, **kwargs):
        super(GPPG2, self).__init__(**kwargs)
        self._type_terminals = None

    def train(self, x, f):
        """Each terminal is associated to the class, consequently,
        the prototype can only be build using the values that belong
        to the corresponding class"""
        super(GPPG2, self).train(x, f)
        self._ntrees_per_class_mod = np.unique(f).shape[0]
        self._type_terminals = np.zeros(self._ntrees_per_class_mod,
                                        dtype=np.object)
        for i in range(self._ntrees_per_class_mod):
            self._type_terminals[i] = np.where(self._f == i)[0]
        return self

    def random_leaf(self):
        """Choose a leaf depending on the i-th argument of root, i.e.,
        Output function  """
        if np.random.rand() < 0.5 or self._nrandom == 0:
            i = self._doing_tree
            cnt = self._type_terminals[i].shape[0]
            l = np.random.randint(cnt)
            l = self._type_terminals[i][l]
            return l + self._func.shape[0]
        else:
            l = np.random.randint(self._constants.shape[0])
            return l + self._func.shape[0] + self._x.shape[1]


def run():
    fpt = open('data/iris.npy', 'rb')
    X = np.load(fpt)
    cl = np.load(fpt)
    fpt.close()
    gp = GPPG2(popsize=1000, generations=50,
               verbose=True).train(X, cl)
    gp.run()
    return gp

if __name__ == '__main__':
    run()
