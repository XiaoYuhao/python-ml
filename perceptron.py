

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()

X = np.array(data['data'])
Y = np.array(data['target'])
classes = data['target_names']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

class Perceptron:
    def __init__(self, lr=0.1, itera=1000):
        self.lr = lr
        self.itera = itera
        pass

    def _sgd(self):
        flag = False
        for _x, _y in zip(self.x_train, self.y_train):
            _y = 1 if _y == 1 else -1
            y_p = np.dot(_x, self.W) + self.b
            if y_p * _y <= 0:
                self.W = self.W + self.lr * _y * _x
                self.b = self.b + self.lr * _y
                flag = True
        return flag


    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        feature_dim = x_train.shape[1]
        self.W = np.zeros(feature_dim)
        self.b = 0

        it = 0
        while True:
            if self._sgd() is False:
                break
            it += 1
            if it >= self.itera:
                print("Warn: Iterations maybe too small.")
                break

        print(self.W)
        print(self.b)
        
    def pred(self, point):
        p = np.dot(self.W, point) + self.b
        if p >= 0:
            return 1
        else:
            return 0

    def eval(self, x_test, y_test):
        num = 0
        acc = 0
        for _x, _y in zip(x_test, y_test):
            p = self.pred(_x)
            if p == _y:
                acc += 1
            num += 1
        acc /= num
        print(f'Accuary: {acc}')




if __name__ == '__main__':
    _x = np.array([(3,3),(4,3),(1,1)])
    _y = np.array([1,1,-1])
    classifier = Perceptron(lr=0.001, itera=10000)
    classifier.fit(X_train, Y_train)
    classifier.eval(X_test, Y_test)
