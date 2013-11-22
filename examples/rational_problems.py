import numpy as np
from SimpleGP import GPPDE
from multiprocessing import Process, cpu_count
import time

x = np.arange(-1, 1.1, 0.1)[:, np.newaxis]
l = np.array(map(lambda x: map(float, x.split()),
                 open('data/rational-problems.txt').readlines()))


def run(pr, seed):
    print 'Haciendo r-%s-%s.npy' % (pr, seed)
    f = l[pr]
    gp = GPPDE(popsize=1000, generations=50, pxo=1.0,
               fname_best=None, max_mem=500,
               seed=seed)
    gp.train(x, f)
    gp.run()


lst = []
for i in range(l.shape[0]):
    for j in range(30):
        lst.append((i, j))
lst.reverse()

process_lst = []
while len(lst):
    process_lst = filter(lambda x: x.is_alive(), process_lst)
    for i in range(len(process_lst), cpu_count()):
        if len(lst) == 0:
            continue
        p = Process(target=run, args=lst.pop())
        p.start()
        process_lst.append(p)
    time.sleep(5)

