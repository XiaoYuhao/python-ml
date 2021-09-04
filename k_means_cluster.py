
import numpy as np
import random
import matplotlib.pyplot as plt

data = np.load("iris_data.npy", allow_pickle=True)
data = data[()]

X = np.array(data["data"])
Y = np.array(data['target'])
classes = data["target_names"]

COLORS = ['r', 'g', 'b', 'k', 'y']

def train_test_split(X, Y, ratio=0.8):
    select = [i for i in range(X.shape[0])]
    random.shuffle(select)
    train_len = int(X.shape[0] * ratio)
    X_train = X[select[:train_len]]
    X_test = X[select[train_len:]]
    Y_train = Y[select[:train_len]]
    Y_test = Y[select[train_len:]]
    return X_train, Y_train, X_test, Y_test

def euclidean_distance(a, b):
    '''
    求两个样本点之间的欧式距离
    np.linalg.norm函数用于求范数，默认求二次范数
    '''
    return np.linalg.norm(np.array(a) - np.array(b), ord=2)
    #return np.linalg.norm(np.array(a) - np.array(b), ord=1)


class KNearestCluster:
    def __init__(self, k):
        self.k = k
        pass

    def fit(self, x):
        select = []
        while True:
            if len(select) == self.k:
                break
            t = random.randint(0, x.shape[0] - 1)                   #初始化，随机选择k个点作为类簇
            if t not in select:
                select.append(t)
        
        center = x[select]
        y = np.zeros(x.shape[0], dtype='int')
        colors = [COLORS[0] for i in range(x.shape[0])]
        iter = 1
        while True:
            plt.cla()                                               #数据可视化
            plt.title(f"Iter = {iter}")
            plt.scatter(x[:,0], x[:,1], marker=".", color=colors)
            plt.scatter(center[:,0], center[:,1], marker="x")
            plt.savefig("image/datamap{:0>2d}_k3".format(iter))
            points = [[] for i in range(self.k)]
            flag = True
            for i, _x in enumerate(x):
                min_d = 0x3f3f3f3f
                min_c = 0
                for j, _c in enumerate(center):                     #计算每个点与每个类簇中心点的距离
                    distance = euclidean_distance(_x, _c)
                    if distance < min_d:                            #选择最近的类簇加入
                        min_d = distance
                        min_c = j
                if y[i] != min_c:
                    y[i] = min_c
                    flag = False
                points[min_c].append(_x)

            if flag:                                                #结束迭代的条件：没有样本的类别发生变化
                break

            for i in range(self.k):                                 #重新计算每一个类的中心点
                center[i] = np.mean(np.array(points[i]), axis=0)
            for i in range(x.shape[0]):
                colors[i] = COLORS[y[i]]

            iter += 1

        print(y)

if __name__ == "__main__":
    knc = KNearestCluster(k=3)
    knc.fit(X[:,2:4])
    print(Y)