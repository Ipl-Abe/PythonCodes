import copy
import math
import numpy as np
import matplotlib.pyplot as plt
 
class module():
    def __init__(self, x=0, y=0, w=1, h=1, r=False):
        self.x, self.y, self.w, self.h, self.rotated = x, y, w, h, r
 
    def draw(self):
        if self.rotated:
             plt.gca().add_patch(plt.Rectangle((self.x, self.y),
                     self.h, self.w, alpha=0.25, color='b'))
        else:
            plt.gca().add_patch(plt.Rectangle((self.x, self.y),
                    self.w, self.h, alpha=0.25, color='g'))
    def get_coords(self):
        print("test")
        tr = (self.h, self.w) if self.rotated else (self.w, self.h)
        return [(self.x, self.y), (self.x + tr[0], self.y),
            (self.x + tr[0], self.y + tr[1]),
            (self.x, self.y + tr[1]), (self.x, self.y)]
 
class sequence_pair():
    def __init__(self, mods=None, X=None, Y=None):
        self.__modules = mods
        self.__n = len(mods) or 0
        self.__X = X or np.random.permutation(self.__n).tolist() 
        self.__Y = Y or np.random.permutation(self.__n).tolist() 
        self.__Xr = self.__X[::-1]
        self.__match = self.__make_match()
        self.__placement()
 
    def __make_match(self):
        m = [0 for i in range(self.__n)]
        for i in range(self.__n):
            m[self.__Y[i]] = i
        return m 
 
    def __placement(self):
        self.max_w = self.__x()
        self.max_h = self.__y()
 
    def __x(self):
        L = [0] * (self.__n + 1)
        for i in range(self.__n):
            self.__modules[self.__X[i]].x = L[self.__match[self.__X[i]]]
            t = self.__modules[self.__X[i]].x + self.__modules[self.__X[i]].h
            if self.__modules[self.__X[i]].rotated == False:
                t = self.__modules[self.__X[i]].x + self.__modules[self.__X[i]].w
                for j in range(self.__match[self.__X[i]], len(L)):
                    if t > L[j]:
                        L[j] = t
                    else:
                        break
        return  L[-1] 
 
    def __y(self):
        L = [0] * (self.__n + 1)
        for i in range(self.__n):
            self.__modules[self.__Xr[i]].y = L[self.__match[self.__Xr[i]]]
            t = self.__modules[self.__Xr[i]].y + self.__modules[self.__Xr[i]].w
            if self.__modules[self.__Xr[i]].rotated == False:
                t = self.__modules[self.__Xr[i]].y + self.__modules[self.__Xr[i]].h
                for j in range(self.__match[self.__Xr[i]], len(L)):
                    if t > L[j]:
                        L[j] = t
                    else:
                        break
        return L[-1]
 
    def __rotate(self):
        target = np.random.randint(self.__n)
        if self.__modules[target].rotated:
            self.__modules[target].rotated = False
        else:
            self.__modules[target].rotated = True
        return sequence_pair(self.__modules, self.__X, self.__Y)
 
    def __xswap(self):
        x = copy.deepcopy(self.__X)
        u, v = 0, 0
        while u == v:
            u, v = np.random.randint(0, self.__n, 2)
        x[u], x[v] = x[v], x[u]
        return sequence_pair(self.__modules, x, self.__Y)
 
    def __yswap(self):
        y = copy.deepcopy(self.__Y)
        u, v = 0, 0
        while u == v:
            u, v = np.random.randint(0, self.__n, 2)
        y[u], y[v] = y[v], y[u]
        return sequence_pair(self.__modules, self.__X, y)
 
    def __xshift(self):
        x = copy.deepcopy(self.__X)
        u, v = 0, 0
        while u == v:
            u, v = np.random.randint(0, self.__n, 2)
        x.insert(u, x.pop(v))
        return sequence_pair(self.__modules, x, self.__Y)
 
    def __yshift(self):
        y = copy.deepcopy(self.__Y)
        u, v = 0, 0
        while u == v:
            u, v = np.random.randint(0, self.__n, 2)
        y.insert(u, y.pop(v))
        return sequence_pair(self.__modules, self.__X, y)
 
    def get_neighbor(self):
        mode = np.random.randint(6)
        if mode == 0: return self.__xswap()
        elif mode == 1: return self.__yswap()
        elif mode == 2: return self.__xshift()
        elif mode == 3: return self.__yshift()
        elif mode == 4: return self.__xswap().__yswap()
        elif mode == 5: return self.__xshift().__yshift()
        else: return self.__rotate()
     



    def get_modules(self):
        return [m.get_coords() for m in self.__modules]


    def area(self):
        return self.max_w * self.max_h
 
    def draw(self):
        plt.figure(facecolor='white')
        plt.axis('equal')
        plt.xlim([0, self.max_w])
        plt.ylim([0, self.max_h])
        for m in self.__modules:
            m.draw()
 
class simulated_annealing():
    def __init__(self, mods):
        self.__csp = sequence_pair(mods)
        self.__best = self.__csp
        self.__T = 1
        self.__k = 1
        self.__log_k = []
        self.__log_a = []
 
    def __move(self):
        n = self.__csp.get_neighbor()
        delta = n.area() - self.__csp.area()
        if delta < 0:
            self.__csp = n
            self.__log_k.append(self.__k)
            self.__log_a.append(n.area())
        else:
            if np.random.random() < math.exp(-delta * 0.001 / self.__T):
                self.__csp = n 
                self.__log_k.append(self.__k)
                self.__log_a.append(n.area())
       
    def sim(self):
        while self.__T > 1e-5:
            self.__move()
            if self.__csp.area() < self.__best.area():
                self.__best = copy.deepcopy(self.__csp)
            self.__T *= 0.999
            self.__k += 1
        print(self.__k, self.__best.area())

    def result(self):
        return self.__best.get_modules()
 
    def draw(self):
        self.__best.draw()
 
    def draw_log(self):
        plt.figure(facecolor='white')
        plt.xlim([0, self.__k])
        plt.ylim([0, max(self.__log_a) + 10])
        plt.plot(self.__log_k, self.__log_a)
 
def instance_generator(n):
    m = []
    for i in range(n):
        m.append(module(w=np.random.randint(10) + 1,
                                      h=np.random.randint(10) + 1))
    return m
 
if __name__ == '__main__':
    sa = simulated_annealing(instance_generator(30))
    sa.sim()
    sa.draw()
    sa.draw_log()
    plt.show()