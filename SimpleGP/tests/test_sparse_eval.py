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

from SimpleGP import GP, GPS
from SimpleGP import SparseArray, SparseEval
from nose.tools import assert_almost_equals
import numpy as np
np.set_printoptions(precision=3)


def create_numpy_array(size=100, nvalues=10):
    var = np.zeros(size)
    index = np.arange(size)
    np.random.shuffle(index)
    var[index[:nvalues]] = np.random.uniform(size=nvalues)
    return var


def test_sparse_array():
    np.random.seed(0)
    var = create_numpy_array()
    array = SparseArray.fromlist(var)
    print array.size()
    assert array.size() == 100
    print array.print_data()
    print var[:50]
    print np.array(array.tolist())[:50]
    assert np.fabs(var - np.array(array.tolist())).sum() == 0
    sp = SparseArray().empty(100, 1000)
    assert sp.size() == 1000
    assert sp.nele() == 100


def test_get_data_index():
    np.random.seed(0)
    var = create_numpy_array()
    array = SparseArray.fromlist(var)
    n = SparseArray()
    n.init(array.nele())
    n.set_size(array.size())
    n.set_data_index(array.get_data(), array.get_index())
    assert (array - n).fabs().sum() == 0


def test_nunion():
    size = 1000
    uno = create_numpy_array(size)
    suno = SparseArray.fromlist(uno)
    dos = create_numpy_array(size)
    sdos = SparseArray.fromlist(dos)
    mask = np.zeros(uno.shape[0], dtype=np.bool)
    mask[~(uno == 0)] = True
    mask[~(dos == 0)] = True
    r = suno.nunion(sdos)
    print mask.sum(), "-", r
    assert mask.sum() == suno.nunion(sdos)


def test_nintersection():
    np.random.seed(0)
    size = 15
    uno = create_numpy_array(size)
    suno = SparseArray.fromlist(uno)
    dos = create_numpy_array(size)
    sdos = SparseArray.fromlist(dos)
    mask = np.zeros(uno.shape[0], dtype=np.bool)
    uno_m = ~(uno == 0)
    dos_m = ~(dos == 0)
    mask[uno_m & dos_m] = True
    r = suno.nintersection(sdos)
    print mask.sum(), "-", r
    assert mask.sum() == r


def test_sparse_array_sum():
    np.random.seed(0)
    uno = create_numpy_array()
    dos = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    sdos = SparseArray.fromlist(dos)
    zero = suno.constant(0, suno.size())
    r = suno + zero
    assert ((suno - r).fabs()).sum() == 0
    sr = (suno + sdos).tonparray()
    r = uno + dos
    print r[:10]
    print sr[:10]
    assert np.all(r == sr)


def test_sparse_array_sum3():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    a = (uno + uno.mean())
    b = (suno + suno.mean()).tonparray()
    assert np.all(a == b)


def test_sparse_array_sub():
    np.random.seed(0)
    uno = create_numpy_array()
    dos = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    sdos = SparseArray.fromlist(dos)
    sr = (suno - sdos).tonparray()
    r = uno - dos
    m = r != sr
    print uno[m]
    print dos[m]
    print r[m]
    print sr[m]
    assert np.all(r == sr)
    tmp = sdos.constant(0, sdos.size()) - sdos
    tmp.print_data()
    assert np.all(((tmp).tonparray() == -dos))


def test_sparse_array_sub2():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    a = (uno - uno.mean())
    b = (suno - suno.mean()).tonparray()
    assert np.all(a == b)


def test_sparse_array_mul():
    np.random.seed(0)
    uno = create_numpy_array(10, 5)
    dos = create_numpy_array(10, 5)
    suno = SparseArray.fromlist(uno)
    sdos = SparseArray.fromlist(dos)
    sr = suno * sdos
    sr = np.array((sr).tolist())
    r = uno * dos
    print r[:10]
    print sr[:10]
    assert np.all(r == sr)


def test_sparse_array_div():
    np.random.seed(0)
    uno = create_numpy_array(10)
    uno[0] = 0
    dos = create_numpy_array(10)
    dos[1] = 0
    suno = SparseArray.fromlist(uno)
    sdos = SparseArray.fromlist(dos)
    sr = (suno / sdos).tonparray()
    r = uno / dos
    r[1] = 0
    print r[:10]
    print sr[:10]
    assert np.all(r == sr)


def test_sparse_array_sum2():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    assert suno.sum() == uno.sum()


def test_sparse_array_mean():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    assert suno.mean() == uno.mean()


def test_sparse_array_std():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    print suno.std(), uno.std()
    assert_almost_equals(suno.std(), uno.std())


def test_sparse_array_fabs():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno * -1)
    assert suno.fabs().sum() == uno.sum()


def test_sparse_array_exp():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno).exp().tonparray()
    print np.exp(uno), suno
    assert np.all(np.exp(uno) == suno)


def test_sparse_array_sin():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno).sin().tonparray()
    print uno[:10]
    print suno[:10]
    assert np.all(np.sin(uno) == suno)


def test_sparse_array_cos():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno).cos().tonparray()
    print uno[:10]
    print suno[:10]
    assert np.all(np.cos(uno) == suno)


def test_sparse_array_ln():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno).ln().tonparray()
    uno = np.log(np.fabs(uno))
    uno[np.isinf(uno)] = 0
    print uno[:10]
    print suno[:10]
    assert np.all(uno == suno)


def test_sparse_array_sq():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno).sq().tonparray()
    print uno[:10]
    print suno[:10]
    assert np.all(uno**2 == suno)

    
def test_sparse_array_sqrt():
    np.random.seed(0)
    uno = create_numpy_array()
    uno[0] = -1
    suno = SparseArray.fromlist(uno).sqrt().tonparray()
    uno = np.sqrt(uno)
    print uno[:10]
    print suno[:10]
    assert np.all(uno[1:] == suno[1:])
    assert np.isnan(uno[0]) and np.isnan(suno[0])


def test_sparse_array_sigmoid():
    np.random.seed(0)
    uno = create_numpy_array() * 100
    uno[0] = -1000
    uno[1] = 1000
    suno = SparseArray.fromlist(uno).sigmoid().tonparray()
    uno = 1 / (1 + np.exp((-uno + 1) * 30))
    print uno[:10]
    print suno[:10]
    map(lambda x: assert_almost_equals(x[0], x[1]), zip(uno, suno))


def test_sparse_array_if():
    def sigmoid(x):
        return 1 / (1 + np.exp((-x + 1) * 30))

    sa = SparseArray.fromlist
    np.random.seed(0)
    x = create_numpy_array() * 10
    x[0] = -1
    x[1] = 1
    y = np.ones_like(x)
    z = np.ones_like(x) + 1
    s = sigmoid(x)
    r = s * y - s * z + z
    # print r
    map(lambda x: assert_almost_equals(x[0], x[1]),
        zip(r, (sa(x).if_func(sa(y),
                              sa(z))).tonparray()))


def test_sparse_array_SAE():
    np.random.seed(0)
    uno = create_numpy_array()
    dos = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    sdos = SparseArray.fromlist(dos)
    print suno.SAE(sdos), np.fabs(uno-dos).sum()
    assert_almost_equals(suno.SAE(sdos), np.fabs(uno-dos).sum())
    uno[0] = np.nan
    suno = SparseArray.fromlist(uno)
    assert suno.SAE(sdos) == np.inf
    

def test_sparse_constant():
    s = SparseArray().constant(12, 10)
    assert len(filter(lambda x: x == 12, s.tolist())) == 10


def test_tonparray():
    uno = create_numpy_array(10)
    suno = SparseArray.fromlist(uno)
    assert np.all(uno == suno.tonparray())


def create_problem(size=10):
    x = np.linspace(-10, 10, size)
    pol = np.array([0.2, -0.3, 0.2])
    X = np.vstack((x**2, x, np.ones(x.shape[0])))
    y = (X.T * pol).sum(axis=1)
    x = np.vstack((x, np.sqrt(np.fabs(x)))).T
    return x, y


def test_seval():
    x, y = create_problem()
    gp = GP.run_cl(x, y, generations=2,
                   func=['+', '-', '*', 'abs', 'sin', 'sq', 'sqrt'])
    sparse = SparseEval(gp._nop)
    sparse.X(x)
    for i in range(gp.popsize):
        y = gp.predict(gp._x, ind=i)
        if not np.all(np.isfinite(y)):
            continue
        print gp.print_infix(gp.population[i],
                             constants=gp._p_constants[i])
        hy = sparse.eval(gp.population[i], gp._p_constants[i])
        # print hy[:10]
        # print y[:10]
        m = np.fabs(y - hy) > 0
        print hy[m]
        print y[m]
        print gp._x[m]
        assert_almost_equals(np.fabs(y - hy).sum(), 0)


def test_seval_simple_tree():
    x, y = create_problem()
    gp = GP().train(x, y)
    sparse = SparseEval(gp._nop)
    sparse.X(x)
    y = x[:, 0] + x[:, 1] * 12.1 + 12.1
    var = gp._nop.shape[0]
    cons = var + x.shape[1]
    ind = np.array([0, var, 0, 2, var+1, cons, cons], dtype=np.int)
    print ind
    cons = np.array([12.10])
    print gp.print_infix(ind, constants=cons)
    hy = sparse.eval(ind, cons)
    print hy[:10]
    print y[:10]
    print np.fabs(y - hy).sum()
    map(lambda x: assert_almost_equals(y[x], hy[x]), range(y.shape[0]))
    # assert_almost_equals(np.fabs(y - hy).sum(), 0)


def test_seval_output():
    x, y = create_problem()
    gp = GP().train(x, y)
    gp._nop[gp._output_pos] = 2
    sparse = SparseEval(gp._nop)
    sparse.X(x)
    y = SparseArray.fromlist(x[:, 0] + x[:, 1] * 12.1 + 12.1)
    var = gp._nop.shape[0]
    cons = var + x.shape[1]
    ind = np.array([15, 0, var, 0, 2, var+1, cons, cons, 0, var, 0, 2,
                    var+1, cons, cons], dtype=np.int)
    print ind
    cons = np.array([12.10])
    print gp.print_infix(ind, constants=cons)
    sparse.eval(ind, cons)
    hy2 = sparse.get_output()
    for hy in hy2:
        assert_almost_equals((y - hy).fabs().sum(), 0)


def test_finite():
    a = SparseArray.fromlist([1, np.inf])
    assert not a.isfinite()
    a = SparseArray.fromlist([1, np.nan])
    assert not a.isfinite()
    a = SparseArray.fromlist([1, 0, 1])
    assert a.isfinite()


def test_fromlist_force():
    a = SparseArray.fromlist([1, np.inf], force_finite=True)
    assert a.isfinite() and a.nele() == 1
    a = SparseArray.fromlist([1, np.nan], force_finite=True)
    assert a.isfinite() and a.nele() == 1
    a = SparseArray.fromlist([1, np.nan, np.inf, 1])
    assert a.nele() == 4 and not a.isfinite()


def test_slice():
    uno = create_numpy_array(10)
    suno = SparseArray.fromlist(uno)
    assert np.all(suno[:50].tonparray() == uno[:50])


def test_slice2():
    np.random.seed(3)
    uno = create_numpy_array(100, 50)
    suno = SparseArray.fromlist(uno)
    index = np.arange(100)
    np.random.shuffle(index)
    index = index[:10]
    try:
        suno[index]
        assert False
    except NotImplementedError:
        pass
    index.sort()
    # print suno[index].tonparray(), uno[index]
    assert np.all(suno[index].tonparray() == uno[index])
    # print suno[index].size(), index.shape
    assert suno[index].size() == index.shape[0]
    index = np.where(uno == 0)[0]
    assert suno[index].nele() == 0
    print index, suno.get_index()
    print suno[index].size(), index.shape[0]
    assert suno[index].size() == index.shape[0]


def test_slice3():
    np.random.seed(3)
    uno = create_numpy_array(100, 50)
    suno = SparseArray.fromlist(uno)
    index = np.array([np.where(uno > 0)[0][0]] * 5)
    index = np.concatenate((index, np.where(uno > 0)[0][1:]), axis=0)
    print suno[index].tonparray()
    print uno[index]
    assert np.all(suno[index].tonparray() == uno[index])


def test_copy():
    uno = SparseArray.fromlist(create_numpy_array(10))
    assert uno.copy().SSE(uno) == 0


def test_pickle():
    import StringIO
    import pickle
    uno = SparseArray.fromlist(create_numpy_array(10))
    fpt = StringIO.StringIO()
    pickle.dump(uno, fpt)
    dos = pickle.load(fpt)
    assert uno.SSE(dos) == 0


def test_mul_vec_cons():
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    print (suno * 12.3).tonparray(), (uno * 12.3)
    assert np.all((uno * 12.3) == (suno * 12.3).tonparray())


def test_div_vec_cons():
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno)
    print (suno / 12.3).tonparray(), (uno / 12.3)
    assert np.all((uno / 12.3) == (suno / 12.3).tonparray())
    
def test_seval2():
    import time

    def pred(ins):
        t1 = time.time()
        pr = ins(gp.population[i],
                 gp._p_constants[i], 0)
        t1 = time.time() - t1
        return pr.tonparray(), t1

    x, y = create_problem(10)
    gp = GPS(generations=3, use_st=0, seed=0, nrandom=0).fit(x, y)
    var = gp._nop.shape[0]
    cons = var + x.shape[1]
    ind = np.array([0, var, 0, 2, var+1, cons, cons], dtype=np.int)
    cons = np.array([12.10])
    gp.population[0] = ind
    gp._p_constants[0] = cons
    gp1 = 0
    gp2 = 0
    for i in range(gp.popsize):
        pr, t = pred(gp._eval.eval)
        gp1 += t
        pr2, t = pred(gp._eval.eval2)
        gp2 += t
        m = np.isfinite(pr)
        assert np.all(pr[m] == pr2[m])
    print gp1, gp2


def test_seval_time():
    import time

    def pred(ins):
        t1 = time.time()
        pr = ins(gp.population[i],
                 gp._p_constants[i], 0, st=st)
        t1 = time.time() - t1
        return pr.tonparray(), t1

    def pred2(ins):
        p1 = np.random.randint(gp.population[i].shape[0])
        parent = np.zeros_like(gp.population[i])
        path = np.zeros_like(gp.population[i])
        t1 = time.time()
        gp._tree.compute_parents(gp.population[i],
                                 parent)
        c = gp._tree.path_to_root(parent, path, p1)
        for j in range(c):
            st[path[j]] = None
        pr = ins(gp.population[i],
                 gp._p_constants[i], 0, st=st)
        t1 = time.time() - t1
        return pr.tonparray(), t1

    x, y = create_problem(1000)
    gp = GPS(generations=3, popsize=10000,
             seed=0, nrandom=0).train(x, y)
    gp.create_population()
    gp1 = 0
    gp2 = 0
    for i in range(gp.popsize):
        st = map(lambda x: None, gp.population[i])
        # st = None
        pr, t = pred(gp._eval.eval)
        gp1 += t
        _, t = pred2(gp._eval.eval)
        gp2 += t
    print gp1, gp2
    print st, 0.50
    assert gp1 < 0.55
    # assert False
