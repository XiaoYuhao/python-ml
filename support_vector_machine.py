

import numpy as np
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()


X = np.array(data["data"])
Y = np.array(data['target'])
classes = data["target_names"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

class SupportVectorMachine:
    def __init__(self, itera=50, C=0.8):
        self.itera = itera
        self.C = C
        pass

    def _clip(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        return alpha

    def _f(self, x):
        x = np.matrix(x).T
        x_train = np.matrix(self.x_train)
        ks = x_train * x
        wx = np.matrix(self.alphas * self.y_train) * ks
        fx = wx + self.b
        return fx[0, 0]

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        feature_dim = x_train.shape[0]
        self.alphas = np.zeros(feature_dim)
        self.b = 0
        it = 0

        while it < self.itera:
            changed = 0
            for i in range(feature_dim):
                a_i, x_i, y_i = self.alphas[i], x_train[i], y_train[i]
                j = random.choice([k for k in range(0,i)] + [k for k in range(i+1,feature_dim)])
                a_j, x_j, y_j = self.alphas[j], x_train[j], y_train[j]
                fx_i = self._f(x_i)
                fx_j = self._f(x_j)
                E_i = fx_i - y_i
                E_j = fx_j - y_j
                K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
                eta = K_ii + K_jj - 2 * K_ij
                if eta <= 0:
                    continue
                a_i_old, a_j_old = a_i, a_j
                a_j_new = a_j_old + y_j * (E_i - E_j) / eta
                if y_i != y_j:
                    L = max(0, a_j_old - a_i_old)
                    H = min(self.C, self.C + a_j_old - a_i_old)
                else:
                    L = max(0, a_i_old + a_j_old - self.C)
                    H = min(self.C, a_j_old + a_i_old)
                a_j_new = self._clip(a_j_new, L, H)
                a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)
                if abs(a_j_new - a_j_old) < 0.0000001:
                    continue
                self.alphas[i] = a_i_new
                self.alphas[j] = a_j_new

                b_i = -E_i - y_i * K_ii * (a_i_new - a_i_old) - y_j * K_ij * (a_j_new - a_j_old) + self.b
                b_j = -E_j - y_i * K_ij * (a_i_new - a_i_old) - y_j * K_jj * (a_j_new - a_j_old) + self.b
                if 0 < a_i_new < self.C:
                    self.b = b_i
                elif 0 < a_j_new < self.C:
                    self.b = b_j
                else:
                    self.b = (b_i + b_j) / 2
                changed += 1
                print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, changed))
            if changed == 0:
                it += 1
            else:
                it = 0

    def pred(self, point):
        p = self._f(point)
        if p >= 0:
            return 0
        else:
            return 1
    
    def eval(self, x_test, y_test):
        num = 0
        acc = 0
        for _x, _y in zip(x_test, y_test):
            if self.pred(_x) == _y:
                acc += 1
            num += 1
        acc /= num
        print(f"Accuary: {acc} \t")



if __name__ == '__main__':
    svm = SupportVectorMachine()
    svm.fit(X_train, Y_train)
    svm.eval(X_test, Y_test)
    
                

