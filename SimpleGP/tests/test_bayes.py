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
from SimpleGP import Classification
from SimpleGP import SparseArray
from nose.tools import assert_almost_equals
from test_classification import *


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
        map(lambda (x, y):
            assert_almost_equals(x, y),
            zip(np.var(X[cl == i], axis=0), np.array(v)))


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
    a = bayes.joint_log_likelihood(0)
    print bayes._elm_constants[0]
    b = bayes.joint_log_likelihood(0)
    map(lambda (x, y):
        map(lambda (x1, y1): assert_almost_equals(x1, y1),
            zip(x, y)),
        zip(a, b))


def test_bayes_eval_ind():
    from SimpleGP import Bayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = Bayes().train(Xs, y)
    bayes.create_population()
    a = bayes.eval(0)
    assert a.isfinite()
    bayes._elm_constants[0][-1] < bayes._ntrees
    assert bayes.eval(1).isfinite()


def test_bayes_predict_proba():
    from SimpleGP import Bayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = Bayes().train(Xs, y)
    bayes.create_population()
    bayes.fitness(0)
    pr = bayes.predict(bayes._x, ind=0).tonparray()
    a = bayes.predict_proba(bayes._x, ind=0)
    assert np.all(pr == a.argmax(axis=1))


def test_bayes_no_use_st():
    from SimpleGP import Bayes
    try:
        Bayes(use_st=1)
        assert False
    except NotImplementedError:
        pass


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
    b2 = Bayes().train(Xvs, yvs)
    b2.create_population()
    b2.set_early_stopping_ind(bayes._early_stopping)
    map(lambda (x, y):
        map(lambda (x1, y1): assert_almost_equals(x1, y1), zip(x, y)),
        zip(bayes._elm_constants[1][0], b2._elm_constants[0][0]))
    pr2 = b2.predict(Xvs, ind=0)
    assert pr.SSE(pr2) == 0
    map(lambda (x, y):
        map(lambda (x1, y1): assert_almost_equals(x1, y1), zip(x, y)),
        zip(bayes._elm_constants[1][0], b2._elm_constants[0][0]))
    b2.predict_proba(Xvs, ind=0)
    map(lambda (x, y):
        map(lambda (x1, y1): assert_almost_equals(x1, y1), zip(x, y)),
        zip(bayes._elm_constants[1][0], b2._elm_constants[0][0]))
    assert b2._fitness[0] == -np.inf


def test_ibayes_predict():
    from SimpleGP import IBayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = IBayes().train(Xs, y)
    bayes.create_population()
    print bayes.fitness(0)
    print bayes.fitness(1)
    a = bayes.predict_proba(bayes._x, ind=0)
    b = bayes.predict_proba(bayes._x, ind=1)
    r = np.concatenate((a, b), axis=1)
    bayes._inds.append([None,
                        bayes.population[1].copy(),
                        bayes._p_constants[1].copy(),
                        bayes._elm_constants[1],
                        bayes._class_freq])
    r2 = bayes.predict_proba(bayes._x, ind=0)
    assert np.fabs(r - r2).mean() == 0
    bayes._inds = []
    b1 = bayes.predict_proba(bayes._x, ind=1)
    assert np.fabs(b - b1).mean() == 0


def test_ibayes():
    from SimpleGP import IBayes
    Xs = map(lambda x: SparseArray.fromlist(x), X.T)
    y = SparseArray.fromlist(cl)
    bayes = IBayes().train(Xs, y)
    bayes.create_population()
    bayes.fitness(0)
    bayes.fitness(2)
    a = bayes.predict_llh(bayes._x, ind=0)
    b = bayes.predict_llh(bayes._x, ind=2)
    c1 = np.concatenate((a, b), axis=1)
    bayes.prev_llh(a)
    c2 = bayes.eval(2)
    assert np.all(c1.argmax(axis=1) % bayes._ncl == c2)
    bayes._inds.append([None,
                        bayes.population[0].copy(),
                        bayes._p_constants[0].copy(),
                        bayes._elm_constants[0],
                        bayes._class_freq])
    c3 = bayes.predict_llh(bayes._x, 2).argmax(axis=1) % bayes._ncl
    assert np.all(c2 == c3)


def test_ibayes_fit():
    from SimpleGP import IBayes

    def callback(self):
        if hasattr(self, 'calls'):
            self.calls += 1
        else:
            self.calls = 1
        print "*"*10

    np.random.seed(0)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    Xtr = map(SparseArray.fromlist, X[index[:120]].T)
    ytr = SparseArray.fromlist(cl[index[:120]])
    Xvs = map(SparseArray.fromlist, X[index[120:]].T)
    yvs = SparseArray.fromlist(cl[index[120:]])
    bayes = IBayes(generations=2, popsize=10,
                   ntimes=2, verbose=True).fit(Xtr, ytr,
                                               test=Xvs, test_y=yvs,
                                               callback=callback)
    print map(lambda x: x[0], bayes._inds)
    print len(bayes._inds), bayes.calls
    assert len(bayes._inds) == bayes.calls

