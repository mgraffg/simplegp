import numpy as np
from SimpleGP import lstsqGP

x = np.arange(-1, 1.1, 0.1)[:, np.newaxis]
l = np.array(map(lambda x: map(float, x.split()),
                 open('data/rational-problems.txt', 'r').readlines()))

gp = lstsqGP(seed=0, func=['+', '-', '*', '/'], verbose=True,
             fname_best='tmp.npy.gz',
             pxo=0.5, generations=50).train(x, l[0])
gp.run()
