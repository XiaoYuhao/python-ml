
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_boston()

X = np.array(data['data'])
Y = np.array(data['target'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        X = np.column_stack((x_train, np.ones(x_train.shape[0])))
        self.W = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, y_train))

    def pred(self, point):
        return np.dot(self.W, np.concatenate((point, np.ones(1))))


if __name__ == '__main__':
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    for _x, _y in zip(X_test, Y_test):
        print(lr.pred(_x), _y)



