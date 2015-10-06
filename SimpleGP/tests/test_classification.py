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
from SimpleGP import Classification, ClassificationPDE
from SimpleGP import SparseArray
from nose.tools import assert_almost_equals


cl = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
X = np.array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2],
       [ 5. ,  3.6,  1.4,  0.2],
       [ 5.4,  3.9,  1.7,  0.4],
       [ 4.6,  3.4,  1.4,  0.3],
       [ 5. ,  3.4,  1.5,  0.2],
       [ 4.4,  2.9,  1.4,  0.2],
       [ 4.9,  3.1,  1.5,  0.1],
       [ 5.4,  3.7,  1.5,  0.2],
       [ 4.8,  3.4,  1.6,  0.2],
       [ 4.8,  3. ,  1.4,  0.1],
       [ 4.3,  3. ,  1.1,  0.1],
       [ 5.8,  4. ,  1.2,  0.2],
       [ 5.7,  4.4,  1.5,  0.4],
       [ 5.4,  3.9,  1.3,  0.4],
       [ 5.1,  3.5,  1.4,  0.3],
       [ 5.7,  3.8,  1.7,  0.3],
       [ 5.1,  3.8,  1.5,  0.3],
       [ 5.4,  3.4,  1.7,  0.2],
       [ 5.1,  3.7,  1.5,  0.4],
       [ 4.6,  3.6,  1. ,  0.2],
       [ 5.1,  3.3,  1.7,  0.5],
       [ 4.8,  3.4,  1.9,  0.2],
       [ 5. ,  3. ,  1.6,  0.2],
       [ 5. ,  3.4,  1.6,  0.4],
       [ 5.2,  3.5,  1.5,  0.2],
       [ 5.2,  3.4,  1.4,  0.2],
       [ 4.7,  3.2,  1.6,  0.2],
       [ 4.8,  3.1,  1.6,  0.2],
       [ 5.4,  3.4,  1.5,  0.4],
       [ 5.2,  4.1,  1.5,  0.1],
       [ 5.5,  4.2,  1.4,  0.2],
       [ 4.9,  3.1,  1.5,  0.1],
       [ 5. ,  3.2,  1.2,  0.2],
       [ 5.5,  3.5,  1.3,  0.2],
       [ 4.9,  3.1,  1.5,  0.1],
       [ 4.4,  3. ,  1.3,  0.2],
       [ 5.1,  3.4,  1.5,  0.2],
       [ 5. ,  3.5,  1.3,  0.3],
       [ 4.5,  2.3,  1.3,  0.3],
       [ 4.4,  3.2,  1.3,  0.2],
       [ 5. ,  3.5,  1.6,  0.6],
       [ 5.1,  3.8,  1.9,  0.4],
       [ 4.8,  3. ,  1.4,  0.3],
       [ 5.1,  3.8,  1.6,  0.2],
       [ 4.6,  3.2,  1.4,  0.2],
       [ 5.3,  3.7,  1.5,  0.2],
       [ 5. ,  3.3,  1.4,  0.2],
       [ 7. ,  3.2,  4.7,  1.4],
       [ 6.4,  3.2,  4.5,  1.5],
       [ 6.9,  3.1,  4.9,  1.5],
       [ 5.5,  2.3,  4. ,  1.3],
       [ 6.5,  2.8,  4.6,  1.5],
       [ 5.7,  2.8,  4.5,  1.3],
       [ 6.3,  3.3,  4.7,  1.6],
       [ 4.9,  2.4,  3.3,  1. ],
       [ 6.6,  2.9,  4.6,  1.3],
       [ 5.2,  2.7,  3.9,  1.4],
       [ 5. ,  2. ,  3.5,  1. ],
       [ 5.9,  3. ,  4.2,  1.5],
       [ 6. ,  2.2,  4. ,  1. ],
       [ 6.1,  2.9,  4.7,  1.4],
       [ 5.6,  2.9,  3.6,  1.3],
       [ 6.7,  3.1,  4.4,  1.4],
       [ 5.6,  3. ,  4.5,  1.5],
       [ 5.8,  2.7,  4.1,  1. ],
       [ 6.2,  2.2,  4.5,  1.5],
       [ 5.6,  2.5,  3.9,  1.1],
       [ 5.9,  3.2,  4.8,  1.8],
       [ 6.1,  2.8,  4. ,  1.3],
       [ 6.3,  2.5,  4.9,  1.5],
       [ 6.1,  2.8,  4.7,  1.2],
       [ 6.4,  2.9,  4.3,  1.3],
       [ 6.6,  3. ,  4.4,  1.4],
       [ 6.8,  2.8,  4.8,  1.4],
       [ 6.7,  3. ,  5. ,  1.7],
       [ 6. ,  2.9,  4.5,  1.5],
       [ 5.7,  2.6,  3.5,  1. ],
       [ 5.5,  2.4,  3.8,  1.1],
       [ 5.5,  2.4,  3.7,  1. ],
       [ 5.8,  2.7,  3.9,  1.2],
       [ 6. ,  2.7,  5.1,  1.6],
       [ 5.4,  3. ,  4.5,  1.5],
       [ 6. ,  3.4,  4.5,  1.6],
       [ 6.7,  3.1,  4.7,  1.5],
       [ 6.3,  2.3,  4.4,  1.3],
       [ 5.6,  3. ,  4.1,  1.3],
       [ 5.5,  2.5,  4. ,  1.3],
       [ 5.5,  2.6,  4.4,  1.2],
       [ 6.1,  3. ,  4.6,  1.4],
       [ 5.8,  2.6,  4. ,  1.2],
       [ 5. ,  2.3,  3.3,  1. ],
       [ 5.6,  2.7,  4.2,  1.3],
       [ 5.7,  3. ,  4.2,  1.2],
       [ 5.7,  2.9,  4.2,  1.3],
       [ 6.2,  2.9,  4.3,  1.3],
       [ 5.1,  2.5,  3. ,  1.1],
       [ 5.7,  2.8,  4.1,  1.3],
       [ 6.3,  3.3,  6. ,  2.5],
       [ 5.8,  2.7,  5.1,  1.9],
       [ 7.1,  3. ,  5.9,  2.1],
       [ 6.3,  2.9,  5.6,  1.8],
       [ 6.5,  3. ,  5.8,  2.2],
       [ 7.6,  3. ,  6.6,  2.1],
       [ 4.9,  2.5,  4.5,  1.7],
       [ 7.3,  2.9,  6.3,  1.8],
       [ 6.7,  2.5,  5.8,  1.8],
       [ 7.2,  3.6,  6.1,  2.5],
       [ 6.5,  3.2,  5.1,  2. ],
       [ 6.4,  2.7,  5.3,  1.9],
       [ 6.8,  3. ,  5.5,  2.1],
       [ 5.7,  2.5,  5. ,  2. ],
       [ 5.8,  2.8,  5.1,  2.4],
       [ 6.4,  3.2,  5.3,  2.3],
       [ 6.5,  3. ,  5.5,  1.8],
       [ 7.7,  3.8,  6.7,  2.2],
       [ 7.7,  2.6,  6.9,  2.3],
       [ 6. ,  2.2,  5. ,  1.5],
       [ 6.9,  3.2,  5.7,  2.3],
       [ 5.6,  2.8,  4.9,  2. ],
       [ 7.7,  2.8,  6.7,  2. ],
       [ 6.3,  2.7,  4.9,  1.8],
       [ 6.7,  3.3,  5.7,  2.1],
       [ 7.2,  3.2,  6. ,  1.8],
       [ 6.2,  2.8,  4.8,  1.8],
       [ 6.1,  3. ,  4.9,  1.8],
       [ 6.4,  2.8,  5.6,  2.1],
       [ 7.2,  3. ,  5.8,  1.6],
       [ 7.4,  2.8,  6.1,  1.9],
       [ 7.9,  3.8,  6.4,  2. ],
       [ 6.4,  2.8,  5.6,  2.2],
       [ 6.3,  2.8,  5.1,  1.5],
       [ 6.1,  2.6,  5.6,  1.4],
       [ 7.7,  3. ,  6.1,  2.3],
       [ 6.3,  3.4,  5.6,  2.4],
       [ 6.4,  3.1,  5.5,  1.8],
       [ 6. ,  3. ,  4.8,  1.8],
       [ 6.9,  3.1,  5.4,  2.1],
       [ 6.7,  3.1,  5.6,  2.4],
       [ 6.9,  3.1,  5.1,  2.3],
       [ 5.8,  2.7,  5.1,  1.9],
       [ 6.8,  3.2,  5.9,  2.3],
       [ 6.7,  3.3,  5.7,  2.5],
       [ 6.7,  3. ,  5.2,  2.3],
       [ 6.3,  2.5,  5. ,  1.9],
       [ 6.5,  3. ,  5.2,  2. ],
       [ 6.2,  3.4,  5.4,  2.3],
       [ 5.9,  3. ,  5.1,  1.8]])


def test_sparse_array_class_mean():
    np.set_printoptions(precision=3)
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    a = y.mean_per_cl(Xs, y.class_freq(np.unique(cl).shape[0]))
    for i, v in zip(np.unique(cl), np.array(a).T):
        print X[cl == i].mean(axis=0), v
        assert np.all(X[cl == i].mean(axis=0) == np.array(v))


def test_sparse_array_class_var():
    np.set_printoptions(precision=3)
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    kfreq = y.class_freq(np.unique(cl).shape[0])
    a = y.mean_per_cl(Xs, kfreq)
    var = y.var_per_cl(Xs, a, kfreq)
    for i, v in zip(np.unique(cl), np.array(var).T):
        print np.var(X[cl == i], axis=0), v
        assert np.all(np.var(X[cl == i], axis=0) == np.array(v))


def test_sparse_class_freq():
    y = SparseArray.fromlist(cl)
    f = y.class_freq(np.unique(cl).shape[0])
    for i in np.unique(cl):
        assert (cl == i).sum() == f[i]


def test_sparse_joint_log_likelihood():
    np.set_printoptions(precision=3)
    import array
    import math
    from sklearn.naive_bayes import GaussianNB
    m = GaussianNB().fit(X, cl)
    llh = m._joint_log_likelihood(X)
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    kfreq = y.class_freq(np.unique(cl).shape[0])
    mu = y.mean_per_cl(Xs, kfreq)
    var = y.var_per_cl(Xs, mu, kfreq)
    tot = sum(kfreq)
    cl_prior = array.array('d', map(lambda x: math.log(x / tot), kfreq))
    llh2 = SparseArray.joint_log_likelihood(Xs, mu, var, cl_prior)
    map(lambda (x, y):
        map(lambda (x1, y1):
            assert_almost_equals(x1, y1),
            zip(x, y)),
        zip(llh, llh2))


def test_classification():
    np.random.RandomState(0)
    gp = Classification.run_cl(X, cl, generations=5)
    pr = gp.predict(X)
    assert (not np.any(np.isnan(pr))) and (not np.any(np.isinf(pr)))


def test_classification_run_cl():
    np.random.RandomState(0)
    X1 = X + np.random.normal(loc=0.0, scale=0.01,
                              size=X.shape)
    gp = Classification.run_cl(X, cl, generations=5,
                               test=X1)
    assert gp is not None


def test_classification_gppde():
    np.random.RandomState(0)
    gp = ClassificationPDE.run_cl(X, cl, generations=5)
    pr = gp.predict(X)
    assert (not np.any(np.isnan(pr))) and (not np.any(np.isinf(pr)))


def test_BER():
    a = np.zeros(10)
    a[-1] = 1
    b = np.zeros(10)
    assert Classification.BER(a, b) == 50


def test_success():
    a = np.zeros(10)
    a[-1] = 1
    b = np.zeros(10)
    assert Classification.success(a, b) == 0.9


def test_balance():
    cl2 = cl[:-3]
    index = Classification.balance(cl2)
    l = np.array(map(lambda x: (cl2[index] == x).sum(), np.unique(cl2)))
    assert np.all(l == l.min())
    index = Classification.balance(cl2, nele=20)
    l = np.array(map(lambda x: (cl2[index] == x).sum(), np.unique(cl2)))
    assert np.all(l == 20)


def test_bayes_train():
    from SimpleGP import Bayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = Bayes().train(Xs, y)
    assert bayes._class_freq
    b2 = Bayes(class_freq=bayes._class_freq,
               ncl=3).train(Xs, y)
    map(lambda (x, y): assert_almost_equals(x, y),
        zip(bayes._class_freq, b2._class_freq))


def test_bayes_llh():
    from SimpleGP import Bayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = Bayes().train(Xs, y)
    bayes.create_population()
    a = bayes.joint_log_likelihood(1)
    assert np.all(a)


def test_bayes_eval_ind():
    from SimpleGP import Bayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = Bayes().train(Xs, y)
    bayes.create_population()
    a = bayes.eval(0)
    assert not a.isfinite()
    assert bayes.eval(1).isfinite()


def test_bayes_BER():
    from SimpleGP import Bayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = Bayes().train(Xs, y)
    bayes.create_population()
    yh = bayes.eval(1)
    b = bayes.distance(bayes._f, yh)
    b2 = Classification.BER(bayes._f.tonparray(),
                            yh.tonparray())
    print b, b2
    assert b == b2


def test_bayes_predict():
    from SimpleGP import Bayes
    np.random.seed(0)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    Xtr = map(SparseArray.fromlist, X[index[:120]].T)
    ytr = SparseArray.fromlist(cl[index[:120]])
    Xvs = map(SparseArray.fromlist, X[index[120:]].T)
    yvs = SparseArray.fromlist(cl[index[120:]])
    bayes = Bayes().train(Xtr, ytr)
    bayes.set_test(Xvs, y=yvs)
    bayes.create_population()
    print bayes.fitness(1)
    pr = bayes.predict(Xvs)
    assert bayes._early_stopping[-1].SSE(pr) == 0
    b2 = Bayes().train(Xvs, yvs)
    b2.create_population()
    b2.set_early_stopping_ind(bayes._early_stopping)
    pr2 = b2.predict(Xvs, ind=0)
    assert bayes._early_stopping[-1].SSE(pr2) == 0
