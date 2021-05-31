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

def classes_distance(a, b):
    '''
    求两个类的类间距离
    '''
    '''
    min_d = 0x3f3f3f3f
    for _xa in a:
        for _xb in b:
            min_d = min(min_d, euclidean_distance(_xa, _xb))
    return min_d
    '''
    max_d = 0
    for _xa in a:
        for _xb in b:
            max_d = max(max_d, euclidean_distance(_xa, _xb))
    return max_d
    

class AgglomerativeCluster:
    def __init__(self):
        pass

    def fit(self, x, k):
        n = x.shape[0]
        clusters = [[x[i]] for i in range(n)]
        indexs = [[i] for i in range(n)]

        iter = 0
        while True:
            #print(clusters)
            print(f"Iter: {iter}")
            if len(clusters) == k:
                break
            min_d = 0x3f3f3f3f
            merge_i = 0
            merge_j = 0
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    d = classes_distance(clusters[i], clusters[j])
                    if min_d > d:
                        min_d = d
                        merge_i = i
                        merge_j = j
        
            #print(merge_i, merge_j)
            new_clusters = []
            new_indexs = []
            for i in range(len(clusters)):
                if i != merge_i and i != merge_j:
                    new_clusters.append(clusters[i])
                    new_indexs.append(indexs[i])
            clusters[merge_i].extend(clusters[merge_j])
            indexs[merge_i].extend(indexs[merge_j])
            new_clusters.append(clusters[merge_i])
            new_indexs.append(indexs[merge_i])
            clusters = new_clusters
            indexs = new_indexs
            iter += 1
        
        y = [0 for i in range(n)]
        for c, _indexs in enumerate(indexs):
            for _i in _indexs:
                y[_i] = c
    
        print(np.array(y))



if __name__ == "__main__":
    acluster = AgglomerativeCluster()
    acluster.fit(X, k=3)
    print(Y)
