from SimpleGP import lstsqGP
from sklearn import cross_validation
import numpy as np


class Cl(lstsqGP):
    def eval(self, *args, **kwargs):
        return super(Cl, self).eval(*args, **kwargs)


fpt = open('data/iris.npy', 'rb')
X = np.load(fpt)
cl = np.load(fpt)
fpt.close()

pr = np.empty_like(cl)
for ts, vs in cross_validation.KFold(cl.shape[0], n_folds=5, shuffle=True):
    gp = Cl.run_cl(X[ts], cl[ts], verbose=True, test=X[vs],
                   generations=50)
    pr[vs] = gp.predict_test_set(gp.best).round().astype(np.int)
print (pr == cl).sum() / float(cl.shape[0])
