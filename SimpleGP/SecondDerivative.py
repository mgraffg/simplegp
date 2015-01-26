import numpy as np
from SimpleGP.gppde import GPPDE
import math
# from SimpleGP.pde import PDE


class GPPDE2(GPPDE):
    def __init__(self, **kwargs):
        self._path2 = None
        self._end2 = None
        super(GPPDE2, self).__init__(**kwargs)

    def train(self, x, f):
        r = super(GPPDE2, self).train(x, f)
        # self.free_mem()
        self._p_der2 = np.empty((self._max_length, self._x.shape[0]),
                                dtype=self._dtype)
        # self._pde = PDE(self._tree, self._p_der2)
        return r

    def compute_Path(self, p1):
        "** only works with the first derivative already computed **"
        self._ind = self._p[self._xo_father1]
        # self._st = self._p_st[self._xo_father1].copy()
        # self._parent = np.empty_like(self._ind)
        self._path2 = np.ones_like(self._ind)
        # self._tree.compute_parents(self._ind, self._parent)
        self._end2 = self._pde.get_path(self._path2)
        # print "path: ", self._path2
        # print "path to root: ", self._path2[:self._end2]

    def crossover(self, father1, father2, p1=-1, p2=-1):
        if p1 == -1:
            p1 = self._tree.father1_crossing_point(father1)
        if p2 == -1:
            e = self.get_error(p1)
            self.compute_Path(p1)

        self.whichFunction()
        primera = self._p_der[p1]
        if len(list(self._path2[:self._end2])) == 1:
            segunda = self._p_der2[0]
            # print "Caicaicaicaicaicaica"
        else:
            segunda = self._p_der2[self._path2[:self._end2][-2]]
            # print "error de segunda: ", segunda
        # print "error de segunda: ", segunda
        paso = primera / segunda.sum()
        # paso = primera / segunda
        newton = self._p_st[self._xo_father1][p1]-paso
        n = self._p[self._xo_father2].shape[0]
        dist = []
        # print "Padre antes: ", self._xo_father2
        for i in range(n):
            # print "n: ", n
            dist.append(self.dist(newton, self._p_st[self._xo_father2][i]))
        p2 = np.argsort(dist)[0]
        # print "Padre despues: ", self._xo_father2
        # print "puntos: ", p1, " ", p2
        # print "longitud2: ", self._p[self._xo_father2].shape
        return super(GPPDE2, self).crossover(father1,
                                             father2, p1, p2)

    def firstPlaceDerivative(self):
        g1 = self._p_der[0]
        g2 = np.ones(self._x.shape[1])*2
        f1 = 0
        f2 = 0
        node = self._p[self._xo_father1][self._path2[0]]
        ii = self._path2[0]
        if node == 0:
            f1 = np.ones(self._x.shape[0])
            f2 = np.zeros(self._x.shape[0])

        elif node == 1:
            if self.is_first_var(0):
                f1 = np.ones(self._x.shape[0])
            else:
                f1 = -np.ones(self._x.shape[0])
            f2 = np.zeros(self._x.shape[0])

        elif node == 2:
            p = self._p_st[self._xo_father1]
            if self.is_first_var(0):
                pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
                f1 = p[pos2]
            else:
                pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
                f1 = p[pos1]
            f2 = np.zeros(self._x.shape[0])

        elif node == 3:
            p = self._p_st[self._xo_father1]
            if self.is_first_var(0):
                pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
                f1 = 1/p[pos2]
                f2 = np.zeros(self._x.shape[0])
            else:
                # print "si soy segunda variable"
                pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
                pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
                f1 = -p[pos1]/(p[pos2]*p[pos2])
                f2 = 2*p[pos1]/(p[pos2]*p[pos2]*p[pos2])

        elif node == 4:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = np.sign(p[oi])
            f2 = np.zeros(self._x.shape[0])

        elif node == 5:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = np.exp(p[oi])
            f2 = f1

        elif node == 6:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = 1/(2*np.sqrt(p[oi]))
            f2 = -1/(4*np.power(p[oi], (3/2)))

        elif node == 7:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = np.cos(p[oi])
            f2 = -np.sin(p[oi])

        elif node == 8:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = -np.sin(p[oi])
            f2 = -np.cos(p[oi])

        elif node == 9:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = np.exp(-p[oi])/(np.power(1+np.exp(-p[oi]), 2))
            f2 = 2*np.exp(-2*p[oi])/(np.power(1+np.exp(-p[oi]), 3)) - f1

        elif node == 10:
            p = self._p_st[self._xo_father1]
            pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
            pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
            pos3 = self._tree.get_pos_arg(self._ind, ii, 2)
            x = p[pos1]
            y = p[pos2]
            z = p[pos3]
            exp = np.exp(-100*x)
            exp2 = np.exp(-200*x)
            # pvar = self.which_var(ii, oi)
            f1 = 0
            f2 = 0
            if self._path2[1] == pos1:
                # print "Soy Left: "
                f1 = -1 - (100*exp*(y-z))/np.power(1+exp, 2)
                f2 = (20000*exp2*(y-z))/np.power(1+exp, 3) - (10000*exp*(y-z))/np.power(1+exp, 2)
            elif self._path2[1] == pos2:
                # print "Soy Center: "
                f1 = 1/(1+exp)
                f2 = np.zeros(self._x.shape[0])
            elif self._path2[1] == pos3:
                # print "Soy Right: "
                f1 = -1/(1+exp)
                f2 = np.zeros(self._x.shape[0])

        elif node == 11:
            p = self._p_st[self._xo_father1]
            pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
            pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
            x = p[pos1]
            y = p[pos2]
            exp = np.exp(-100*(x-y))
            exp2 = np.exp(-200*(x-y))
            if self.is_first_var(0):
                f1 = 1/(1+exp) + (100*exp*(x-y))/np.power((1+exp), 2)
            else:
                f1 = 1 - 1/(1+exp) - (100*exp*(x-y))/np.power((1+exp), 2)
            f2 = 200*exp/np.power((1+exp), 2) + (20000*exp2*(x-y))/np.power((1+exp), 3) - (10000*exp*(x-y))/np.power((1+exp), 2)

        elif node == 12:
            p = self._p_st[self._xo_father1]
            pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
            pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
            x = p[pos1]
            y = p[pos2]
            exp = np.exp(-100*(x-y))
            exp2 = np.exp(-200*(x-y))
            if self.is_first_var(0):
                f1 = 1 - 1/(1+exp) - (100*exp*(x-y))/np.power((1+exp), 2)
            else:
                f1 = 1/(1+exp) + (100*exp*(x-y))/np.power((1+exp), 2)
            f2 = 200*exp/np.power((1+exp), 2) + (20000*exp2*(x-y))/np.power((1+exp), 3) - (10000*exp*(x-y))/np.power((1+exp), 2)

        elif node == 13:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = 1/p[oi]
            f2 = 1/(-p[oi]*p[oi])

        elif node == 14:
            oi = self._path2[1]
            p = self._p_st[self._xo_father1]
            f1 = 2*p[oi]
            f2 = np.ones(self._x.shape[0])*2

        elif node == 15:
            f1 = np.ones(self._x.shape[1])
            f2 = np.ones(self._x.shape[1])

        self._p_der2[0] = g2*(f1*f1) + g1*f2
        # print "puse: ", self._p_der2[0]
        return

    def whichFunction(self):
        path = self._path2
        # ind = self._ind
        ind = self._p[self._xo_father1]
        # segunda derivada de la funcion de Error (y - Y)^2
        self.firstPlaceDerivative()
        # self._p_der2[self._output] = np.ones(self._x.shape[1])*2
        # si el path tiene un solo elemento
        # print "path: ", path[:self._end2]
        # print "largo: ", len(list(path[:self._end2]))
        if len(list(path[:self._end2])) == 1:
            # print "Path chiquito!!"
            return
        for i in range(1, self._end2-1):
            node = ind[path[i]]
            # Revisar nodos
            # print "ESTO TIENE I: ", node
            if node == 0:
                self.add(i)
            elif node == 1:
                self.substract(i)
            elif node == 2:
                self.multiply(i)
            elif node == 3:
                self.divide(i)
            elif node == 4:
                self.fabs(i)
            elif node == 5:
                self.exp(i)
            elif node == 6:
                self.sqrt(i)
            elif node == 7:
                self.sin(i)
            elif node == 8:
                self.cos(i)
            elif node == 9:
                self.sigmoid(i)
            elif node == 10:
                self.if_func(i)
            elif node == 11:
                self.max(i)
            elif node == 12:
                self.min(i)
            elif node == 13:
                self.ln(i)
            elif node == 14:
                self.sq(i)
            elif node == 15:
                self.output(i)

    def is_first_var(self, pos):
        if self._path2[pos] + 1 == self._path2[pos + 1]:
            return True
        return False

    def add(self, i):
        path = self._path2
        ii = path[i]
        ai = path[i-1]
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        # Derivadas de F
        f1 = np.ones(self._x.shape[0])
        f2 = np.zeros(self._x.shape[0])
        # Derivadas de G
        # self.checkFunc(self._ind[path[i-1]])
        # self._p_der2[ii] = self._g2*f1*f1 + self._g1*f2
        self._p_der2[ii] = g2*f1*f1 + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse: ", self._p_der2[ii]
        return

    def substract(self, i):
        path = self._path2
        ii = path[i]
        ai = path[i-1]
        # Derivadas de F
        if self.is_first_var(i):
            f1 = np.ones(self._x.shape[0])
        else:
            f1 = -np.ones(self._x.shape[0])
        f2 = np.zeros(self._x.shape[0])
        # Derivadas de G
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        self._p_der2[ii] = g2*f1*f1 + g1*f2
        return

    def multiply(self, i):
        path = self._path2
        ii = path[i]
        ai = path[i-1]

        p = self._p_st[self._xo_father1]
        if self.is_first_var(i):
            # print "first var"
            pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
            f1 = p[pos2]
        else:
            # print "second var"
            pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
            f1 = p[pos1]

        # print "pos2: ", pos2
        # print "pos1: ", pos1
        f2 = np.zeros(self._x.shape[0])
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "i: ", i
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse aqui aqui: ", self._p_der2[ii]
        return

    def sin(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]

        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = np.cos(p[oi])
        f2 = -np.sin(p[oi])
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en Sin: ", self._p_der2[ii]
        # return

    def cos(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        # Derivadas de F
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = -np.sin(p[oi])
        f2 = -np.cos(p[oi])
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en Cos: ", self._p_der2[ii]
        return

    def sq(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = 2*p[oi]
        f2 = np.ones(self._x.shape[0])*2
        # self._p_der2[ii] = self._g2*(f1*f1) + self._g1*f2
        self._p_der2[ii] = g2*np.power(f1, 2) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en SQ: ", self._p_der2[ii]
        return

    def exp(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = np.exp(p[oi])
        f2 = f1
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en SQ: ", self._p_der2[ii]
        return

    def sqrt(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = 1/(2*np.sqrt(p[oi]))
        f2 = -1/(4*np.power(p[oi], (3/2)))
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en sqrt: ", self._p_der2[ii]
        return

    def sigmoid(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = np.exp(-p[oi])/(np.power(1+np.exp(-p[oi]), 2))
        f2 = 2*np.exp(-2*p[oi])/(np.power(1+np.exp(-p[oi]), 3)) - f1
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en sigmoid: ", self._p_der2[ii]
        return

    def ln(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = 1/p[oi]
        f2 = 1/(-p[oi]*p[oi])
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en ln: ", self._p_der2[ii]
        return

    def fabs(self, i):
        path = self._path2
        ii = path[i]
        oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        f1 = np.sign(p[oi])
        f2 = np.zeros(self._x.shape[0])
        self._p_der2[ii] = g2*(f1*f1) + g1*f2

    def divide(self, i):
        path = self._path2
        ii = path[i]
        # oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]

        if self.is_first_var(i):
            # print "si soy primera variable"
            pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
            f1 = 1/p[pos2]
            f2 = np.zeros(self._x.shape[0])
        else:
            # print "si soy segunda variable"
            pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
            pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
            f1 = -p[pos1]/(p[pos2]*p[pos2])
            f2 = 2*p[pos1]/(p[pos2]*p[pos2]*p[pos2])

        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        # print "Soy nodo: ", ii
        # print "f1: ", f1
        # print "f2: ", f2
        # print "g1: ", g1
        # print "g2: ", g2
        # print "Esto puse estando en divide: ", self._p_der2[ii]
        return

    def max(self, i):
        path = self._path2
        ii = path[i]
        # oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
        pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
        x = p[pos1]
        y = p[pos2]
        exp = np.exp(-100*(x-y))
        exp2 = np.exp(-200*(x-y))
        if self.is_first_var(i):
            f1 = 1/(1+exp) + (100*exp*(x-y))/np.power((1+exp), 2)
        else:
            f1 = 1 - 1/(1+exp) - (100*exp*(x-y))/np.power((1+exp), 2)
        f2 = 200*exp/np.power((1+exp), 2) + (20000*exp2*(x-y))/np.power((1+exp), 3) - (10000*exp*(x-y))/np.power((1+exp), 2)

        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        self._p_der2[ii] = g2*(f1*f1) + g1*f2

    def min(self, i):
        path = self._path2
        ii = path[i]
        # oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
        pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
        x = p[pos1]
        y = p[pos2]
        exp = np.exp(-100*(x-y))
        exp2 = np.exp(-200*(x-y))
        if self.is_first_var(i):
            f1 = f1 = 1 - 1/(1+exp) + (100*exp*(y-x))/np.power((1+exp), 2)

        else:
            f1 = f1 = 1/(1+exp) - (100*exp*(y-x))/np.power((1+exp), 2)
        f2 = -200*exp/np.power((1+exp), 2) + (20000*exp2*(y-x))/np.power((1+exp), 3) - (10000*exp*(y-x))/np.power((1+exp), 2)

        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        return

    def if_func(self, i):
        path = self._path2
        ii = path[i]
        # oi = path[i+1]
        ai = path[i-1]
        p = self._p_st[self._xo_father1]
        pos1 = self._tree.get_pos_arg(self._ind, ii, 0)
        pos2 = self._tree.get_pos_arg(self._ind, ii, 1)
        pos3 = self._tree.get_pos_arg(self._ind, ii, 2)
        x = p[pos1]
        y = p[pos2]
        z = p[pos3]
        exp = np.exp(-100*x)
        exp2 = np.exp(-200*x)
        # pvar = self.which_var(ii, oi)
        f1 = 0
        f2 = 0
        # print "pvar: ", pvar
        # print "path[oi]: ", 
        if path[i+1] == pos1:
            # print "Soy Left: "
            f1 = -1 - (100*exp*(y-z))/np.power(1+exp, 2)
            f2 = (20000*exp2*(y-z))/np.power(1+exp, 3) - (10000*exp*(y-z))/np.power(1+exp, 2)
        elif path[i+1] == pos2:
            # print "Soy Center: "
            f1 = 1/(1+exp)
            f2 = np.zeros(self._x.shape[0])
        elif path[i+1] == pos3:
            # print "Soy Right: "
            f1 = -1/(1+exp)
            f2 = np.zeros(self._x.shape[0])
        g1 = self._p_der[ai]
        g2 = self._p_der2[ai]
        self._p_der2[ii] = g2*(f1*f1) + g1*f2
        return

    def output(self, i):
        path = self._path2
        ii = path[i]
        ai = path[i-1]
        self._p_der2[ii] = self._p_der2[ai]
        return

    def which_var(self, parent, pos):
        var = 0
        while pos > parent:
            print "entre"
            if self._ind[pos] == parent:
                var += 1
            pos -= 1
        return var

    # def crossPoint(self, p1, father2, output):
    #     hess = 1./self._p_der2[p1]
    #     grad = self._p_der[p1]
    #     newton = output - hess*grad
    #     # Distancia entre vectores
    #     n = self.population[self._xo_father2].shape[0]
    #     l = []
    #     for i in range(n):
    #         l.append(self.dist(newton, father2[i, :]))
    #     point = np.array(l).argsort()[0]
    #     # print ">>", len(l), point
    #     return point

    # def get_error(self, p1):
    #     super(GPPDE2, self).get_error(p1)

    def dist(self, a, b):
        sum = 0
        for i in range(a.shape[0]):
            sum += (a[i]-b[i])**2
        return math.sqrt(sum)

gp = GPPDE2(generations=3, seed=0,
            popsize=10,
            # compute_derivatives=True,
            func=['+', '*', '-'],
            verbose=False, max_length=10)
assert gp is not None
