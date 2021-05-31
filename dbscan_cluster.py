
import numpy as np
import random
import matplotlib.pyplot as plt

data = np.load("iris_data.npy", allow_pickle=True)
data = data[()]

X = np.array(data["data"])
Y = np.array(data['target'])
classes = data["target_names"]

COLORS = ['r', 'g', 'b', 'k', 'y']

def euclidean_distance(a, b):
    '''
    求两个样本点之间的欧式距离
    np.linalg.norm函数用于求范数，默认求二次范数
    '''
    return np.linalg.norm(np.array(a) - np.array(b), ord=2)
    #return np.linalg.norm(np.array(a) - np.array(b), ord=1)

class DbscanCluster:
    def __init__(self, minpts, eps):
        self.minpts = minpts
        self.eps = eps
        pass

    def fit(self, x):
        n = x.shape[0]
        neighbor = [[] for i in range(n)]
        for i in range(n):
            for j in range(n):
                if self.eps > euclidean_distance(x[i], x[j]):
                    neighbor[i].append(j)

        def _findkernel(p):
            visited.append(p)
            _cluster = [p]
            if len(neighbor[p]) >= self.minpts:             #点p为核心点
                for q in neighbor[p]:                       #与点p密度相连的点q
                    if q not in visited:
                        _cluster.extend(_findkernel(q))     #搜索点q
            return _cluster


        clusters = []
        visited = []
        for i in range(n):
            if i in visited:                            #点i已经被访问过
                continue
            if len(neighbor[i]) >= self.minpts:         #核心点
                clusters.append(_findkernel(i))
        
        print(clusters)
        colors = [0 for i in range(n)]
        for i, cluster in enumerate(clusters):
            for p in cluster:
                colors[p] = i
        plt.cla()
        plt.scatter(x[:,0], x[:,1], marker=".", c=colors)
        plt.savefig("image/dbscan.png")
    
    def p2pdistance(self, x, p):
        d = []
        for i in range(x.shape[0]):
            d.append(euclidean_distance(x[p], x[i]))
        d.sort()
        print(d)

if __name__ == '__main__':
    dbscan = DbscanCluster(3, 1.0)
    #dbscan.p2pdistance(X, 0)
    dbscan.fit(X[:, 2:4])

