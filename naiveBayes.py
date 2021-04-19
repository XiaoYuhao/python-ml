
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_iris()

X = np.array(data["data"])
Y = np.array(data['target'])
classes = data["target_names"]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

ratio = 0.8
train_len = int(X.shape[0] * ratio)
X_train = X[:train_len]
Y_train = Y[:train_len]
X_test = X[train_len:]
Y_test = Y[train_len:]

#X_train = np.array([(1,'S'), (1,'M'), (1,'M'), (1,'S'), (1,'S'), (2,'S'), (2,'M'), (2,'M'), (2,'L'), (2,'L'), (3,'L'), (3,'M'), (3,'M'), (3,'L'), (3,'L')])
#Y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])



class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, x_train, y_train, smooth=0):
        classes = np.unique(y_train)
        feature_dim = x_train.shape[1]
        feature_i = list()
        for i in range(feature_dim):
            feature_i.append(np.unique(x_train[:,i]))
        feature_i = np.array(feature_i)
        
        '''
        P_y[k] ==> P(Y_k)
        '''
        P_y = [0 for i in range(classes.shape[0])]
    
        sum_y = 0
        for t in y_train:
            P_y[np.where(classes==t)[0][0]] += 1
            sum_y += 1
        P_y = [(p + smooth) / (sum_y + smooth*classes.shape[0]) for p in P_y]
    
        '''
        P_xy[i][j][k] ==> P(X_ij | Y_k)
        '''
        P_xy = [[[0 for k in range(classes.shape[0])] for j in range(feature_i[i].shape[0])] for i in range(feature_dim)]

        sum_yk = [0 for i in range(classes.shape[0])]
        for _x, _y in zip(x_train, y_train):
            k = np.where(classes==_y)[0][0]
            for i in range(feature_dim):
                j = np.where(feature_i[i]==_x[i])[0][0]
                P_xy[i][j][k] += 1
            sum_yk[k] += 1

        for i in range(feature_dim):
            for j in range(feature_i[i].shape[0]):
                for k in range(classes.shape[0]):
                    P_xy[i][j][k] = (P_xy[i][j][k] + smooth) / (sum_yk[k] + smooth*feature_i[i].shape[0])

        self.feature_dim = feature_dim
        self.feature_i = feature_i
        self.classes = classes
        self.P_y = P_y
        self.P_xy = P_xy
    
    def pred(self, point):
        p_max = 0
        for k in range(self.classes.shape[0]):
            p = self.P_y[k]
            for i in range(self.feature_dim):
                j = np.where(self.feature_i[i]==point[i])[0][0]
                p *= self.P_xy[i][j][k]
            if p > p_max:
                p_max = p
                prediction = self.classes[k]
        print(p_max)
        return prediction

class GaussianNB:
    def __init__(self):
        pass

    def _gaussian(self, x, k):
        return (1 / np.sqrt(2 * np.pi * self.vars[:,k]) * np.exp(-(x - self.avgs[:,k])**2 / (2 * self.vars[:,k]))).prod(axis=0)

    def fit(self, x_train, y_train, smooth=0):
        classes = np.unique(y_train)
        feature_dim = x_train.shape[1]
        
        '''
        P_y[k] ==> P(Y_k)
        '''
        P_y = [0 for i in range(classes.shape[0])]
    
        sum_y = 0
        for t in y_train:
            P_y[np.where(classes==t)[0][0]] += 1
            sum_y += 1
        self.P_y = [(p + smooth) / (sum_y + smooth*classes.shape[0]) for p in P_y]

        self.avgs = np.array([[x_train[np.where(y_train==t)][:,i].mean() for t in classes] for i in range(feature_dim)])
        self.vars = np.array([[x_train[np.where(y_train==t)][:,i].var() for t in classes] for i in range(feature_dim)])
        self.classes = classes
        self.feature_dim = feature_dim

    def pred(self, point):
        p_max = 0
        for k in range(self.classes.shape[0]):
            p = self.P_y[k]
            p *= self._gaussian(point, k)
            if p > p_max:
                p_max = p
                prediction = self.classes[k]
        return prediction

    def eval(self, x_test, y_test):
        num = 0
        acc = 0
        for _x, _y in zip(x_test, y_test):
            p = self.pred(_x)
            if p == _y:
                acc += 1
            num += 1
            print(_x, _y, p)
        acc /= num
        print(f"Accuary: {acc}")

       

if __name__ == '__main__':
    test_point = np.array([2,'S'])
    '''
    nb = NaiveBayes()
    nb.fit(X_train, Y_train, 1)
    print(nb.pred(test_point))
    '''
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    #test_point = np.array([4.4, 3.1, 1.3, 1.4])
    nb.eval(X_test, Y_test)


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        