
import numpy as np
import random

data = np.load("iris_data.npy", allow_pickle=True)
data = data[()]

X = np.array(data["data"])
Y = np.array(data['target'])
classes = data["target_names"]

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
    #return np.linalg.norm(np.array(a) - np.array(b), ord=2)
    return np.linalg.norm(np.array(a) - np.array(b), ord=1)


#kd_tree

class TreeNode(object):
    def __init__(self, val=-1, tag=-1, lchild=None, rchild=None):
        self.val = val
        self.tag = tag
        self.lchild = lchild
        self.rchild = rchild

class KNearestNeighbours:
    def __init__(self, ctype='base', k=5):
        assert ctype == 'base' or ctype == 'kd_tree'
        self.ctype = ctype
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        if self.ctype == 'kd_tree':
            self.dim = x_train.shape[1]
            _data = np.hstack((x_train, np.expand_dims(y_train,axis=1)))
            self.kd_root = self._build_kd_tree(_data, self.dim, 0)

    def pred(self, point):
        if self.ctype == 'base':
            return self._classifier_base(self.x_train, self.y_train, point, self.k)
        elif self.ctype == 'kd_tree':
            return self._classifier_kd_tree(point, self.k)
    
    def eval(self, x_test, y_test):
        num = 0
        acc = 0
        for _x, _y in zip(x_test, y_test):
            p = self.pred(_x)
            if p == _y:
                acc += 1
            num += 1
            #print(_x, _y, p)
        acc /= num
        #print(f"Accuary: {acc}")
        return acc

    def _classifier_base(self, train_data, train_target, point, k=5):
        '''
        K近邻的核心思想就是：根据给定的距离度量，在训练集T中找出与x最邻近的k个点
        再根据分类决策规则决定x的类别y
        '''
        data = zip(train_data, train_target)
        distance = []
        for sample, t in data:
            distance.append((euclidean_distance(sample, point), t))
        distance = sorted(distance)
        #print(distance[:k])
        votes = [i[1] for i in distance[:k]]
        counts = np.bincount(votes)
        return np.argmax(counts)

    def _split_data(self, data, axis):
        data = data[np.argsort(data[:,axis])]
        index = data.shape[0] // 2 
        med_val = data[index]
        data_left = data[:index]
        data_right = data[index+1:]
        return med_val, data_left, data_right

    def _build_kd_tree(self, data, k, dep):
        dim = dep % k 
        med_val, data_left, data_right = self._split_data(data, dim)

        node = TreeNode(med_val[:k], med_val[-1])
        if data_left.shape[0] > 0:
            node.lchild = self._build_kd_tree(data_left, k, dep+1)
        if data_right.shape[0] > 0:
            node.rchild = self._build_kd_tree(data_right, k, dep+1)
        return node

    def _search(self, node, d, dep, point, cur_min=None):
        if node is None:
            return cur_min

        dim = dep % d

        if point[dim] < node.val[dim]:
            flag = 0
            _cur_min = self._search(node.lchild, d, dep+1, point, cur_min)
        else:
            flag = 1
            _cur_min = self._search(node.rchild, d, dep+1, point, cur_min)

        if _cur_min is None:
            _cur_min = (node.val, node.tag)
        elif euclidean_distance(node.val, point) < euclidean_distance(_cur_min[0], point):
            _cur_min = (node.val, node.tag)

        if flag == 0:
            next_node = node.rchild
        else:
            next_node = node.lchild
        radiu = euclidean_distance(_cur_min[0], point)                 #计算超球体半径
        if radiu > abs(point[dim] - node.val[dim]):                 #若超球面与另一子树区域相交，则搜索该子树
            cur_min = self._search(next_node, d, dep+1, point, _cur_min)
            if euclidean_distance(cur_min[0], point) < euclidean_distance(_cur_min[0], point):
                _cur_min = cur_min

        return _cur_min

    def _search_k(self, node, d, dep, point, k, closepoinst):
        if node is None:
            return closepoinst

        dim = dep % d

        if point[dim] < node.val[dim]:
            flag = 0
            _closepoinst = self._search_k(node.lchild, d ,dep+1, point, k, closepoinst)
        else:
            flag = 1
            _closepoinst = self._search_k(node.rchild, d, dep+1, point, k, closepoinst)

        if _closepoinst is None:
            _closepoinst = []
            _closepoinst.append((euclidean_distance(node.val, point), node.val, node.tag))
        elif len(_closepoinst) < k :
            _closepoinst.append((euclidean_distance(node.val, point), node.val, node.tag))
            _closepoinst = sorted(_closepoinst, key=lambda x:x[0])
        elif euclidean_distance(node.val, point) < _closepoinst[-1][0]:
            _closepoinst.append((euclidean_distance(node.val, point), node.val, node.tag))
            _closepoinst = sorted(_closepoinst, key=lambda x:x[0])[:k]

        search_next = 0
        _dis = abs(point[dim] - node.val[dim])
        for p in _closepoinst:
            radiu = euclidean_distance(p[1], point)
            if radiu > _dis:
                search_next = 1
                break
            
        if search_next == 1:
            if flag == 0:
                _closepoinst = self._search_k(node.rchild, d, dep+1, point, k, _closepoinst)
            else:
                _closepoinst = self._search_k(node.lchild, d, dep+1, point, k, _closepoinst)

        return _closepoinst

    def _classifier_kd_tree(self, point, k=10):
        closepoint = self._search_k(self.kd_root, self.dim, 0, point, self.k, None)
        #print(closepoint)
        votes = [int(p[2]) for p in closepoint]
        counts = np.bincount(votes)
        return np.argmax(counts)



if __name__ == '__main__':
    '''
    test_point = np.array([4.4, 3.1, 1.3, 1.4])
    #print(classifier(X_train, Y_train, classes, test_point))
    distance = []
    for index, sample in enumerate(X_train):
        distance.append((euclidean_distance(sample, test_point), index))
    distance = sorted(distance)
    print(X_train[distance[0][1]])
    
    X = np.array([[5,4],[2,3],[9,6],[4,8],[8,1],[7,2]])
    node = build_kd_tree(X, 2, 0)
    #val = search(node, 2, 0, np.array([2,5]))
    val = search_k(node, 2, 0, np.array([2,5]), 3, None)
    print(val)
    
    knn = KNearestNeighbours(ctype='kd_tree', k=5)
    knn.fit(X_train, Y_train)
    #test_point = np.array([4.4, 3.1, 1.3, 1.4])
    #print(classes[knn.pred(test_point)])
    knn.eval(X_test, Y_test)
    '''
    test_k = [1,3,5,10]
    acc_k = [[] for i in range(len(test_k))]
    for i in range(5):
        X_train, Y_train, X_test, Y_test = train_test_split(X, Y)
        for j, k in enumerate(test_k):
            knn = KNearestNeighbours(ctype='kd_tree', k=k)
            knn.fit(X_train, Y_train)
            acc = knn.eval(X_test, Y_test)
            acc_k[j].append(acc)
    
    for j, k in enumerate(test_k):
        print(f"K = {k}")
        print(acc_k[j])
        acc_avg = sum(acc_k[j]) / len(acc_k[j])
        print(f"avg acc = {acc_avg}")

    
    