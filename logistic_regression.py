
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
#data = datasets.load_iris()


X = np.array(data["data"])
Y = np.array(data['target'])
classes = data["target_names"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


class LogisticRegression:
    def __init__(self, lr=1, itera=1000):
        self.lr = lr
        self.itera = itera
        pass

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, x ,y):
        h = self._sigmoid(np.dot(x, self.W))
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


    def fit(self, x_train, y_train):
        classes = np.unique(y_train)
        assert classes.shape[0] == 2
        X = np.column_stack((x_train, np.ones(x_train.shape[0])))
        self.W = np.zeros(X.shape[1])

        # 使用批梯度下降法求解w
        for i in range(self.itera):
            gradient = np.dot(X.T, self._sigmoid(np.dot(X, self.W)) - y_train) / y_train.shape[0]
            self.W = self.W - self.lr * gradient
            if i % 1000 == 0:
                J = self._cost_function(X, y_train)
                print(f"loss : {J} \t")
    
    def pred(self, point):
        p = self._sigmoid(np.dot(point, self.W))
        if p >= 0.5:
            return 1
        else:
            return 0

    def eval(self, x_test, y_test):
        acc = 0
        num = 0
        x_test = np.column_stack((x_test, np.ones(x_test.shape[0])))
        for _x, _y in zip(x_test, y_test):
            if self.pred(_x) == _y:
                acc += 1
            num += 1
        acc /= num
        print(f"Accuary: {acc} \t")


class MultiLogisticRegression:
    def __init__(self, lr=0.00001, itera=10000):
        self.lr = lr
        self.itera = itera
        pass

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x_train, y_train):
        self.classes = np.unique(y_train)
        self.K = classes.shape[0]
        self.W = list()

        for i in range(self.K):
            cur_c = self.classes[i]
            _y = np.zeros_like(y_train)
            _y[np.where(y_train==cur_c)[0]] = 1
            lr = LogisticRegression(lr=self.lr, itera=self.itera)
            lr.fit(x_train, _y)
            self.W.append(lr.W)
        
    def pred(self, point):
        max_p = 0
        for i in range(self.K):
            p = self._sigmoid(np.dot(np.concatenate((point, np.ones(1))), self.W[i]))
            if p > max_p:
                max_p = p
                res = self.classes[i]
        return res
    
    def eval(self, x_test, y_test):
        num = 0
        acc = 0
        for _x, _y in zip(x_test, y_test):
            if self.pred(_x) == _y:
                acc += 1
            num += 1
        acc /= num
        print(f"Accuray: {acc} \t")


if __name__ == '__main__':
    
    lr = LogisticRegression(lr=0.00003, itera=400000)
    lr.fit(X_train, Y_train)
    lr.eval(X_test, Y_test)
    
    
    '''
    lr = MultiLogisticRegression(lr=0.0005, itera=100000)
    lr.fit(X_train, Y_train)
    lr.eval(X_train, Y_train)
    lr.eval(X_test, Y_test)
    '''
    

    


