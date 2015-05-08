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

from SimpleGP import GP
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


def test_sparse_array_fabs():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno * -1)
    assert suno.fabs().sum() == uno.sum()


def test_sparse_array_sin():
    np.random.seed(0)
    uno = create_numpy_array()
    suno = SparseArray.fromlist(uno).sin().tonparray()
    print uno[:10]
    print suno[:10]
    assert np.all(np.sin(uno) == suno)


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

    
def test_sparse_constant():
    s = SparseArray().constant(12, 10)
    assert len(filter(lambda x: x == 12, s.tolist())) == 10


def test_tonparray():
    uno = create_numpy_array(10)
    suno = SparseArray.fromlist(uno)
    assert np.all(uno == suno.tonparray())


def create_problem():
    x = np.linspace(-10, 10, 10)
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
