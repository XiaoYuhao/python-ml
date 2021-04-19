
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_iris()


X = np.array(data["data"])
Y = np.array(data['target'])
classes = data["target_names"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

def euclidean_distance(a, b):
    '''
    求两个样本点之间的欧式距离
    np.linalg.norm函数用于求范数，默认求二次范数
    '''
    return np.linalg.norm(np.array(a) - np.array(b))

def _classifier_base(train_data, train_target, classes, point, k=5):
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
    return classes[np.argmax(counts)]

#kd_tree

class TreeNode(object):
    def __init__(self, val=-1, tag=-1, lchild=None, rchild=None):
        self.val = val
        self.tag = tag
        self.lchild = lchild
        self.rchild = rchild

def split_data(data, axis):
    data = data[np.argsort(data[:,axis])]
    index = data.shape[0] // 2 
    med_val = data[index]
    data_left = data[:index]
    data_right = data[index+1:]
    return med_val, data_left, data_right

def build_kd_tree(data, k, dep):
    dim = dep % k 
    med_val, data_left, data_right = split_data(data, dim)
    
    node = TreeNode(med_val[:k], med_val[-1])
    if data_left.shape[0] > 0:
        node.lchild = build_kd_tree(data_left, k, dep+1)
    if data_right.shape[0] > 0:
        node.rchild = build_kd_tree(data_right, k, dep+1)
    return node

def search(node, d, dep, point, cur_min=None):
    if node is None:
        return cur_min

    dim = dep % d

    if point[dim] < node.val[dim]:
        flag = 0
        _cur_min = search(node.lchild, d, dep+1, point, cur_min)
    else:
        flag = 1
        _cur_min = search(node.rchild, d, dep+1, point, cur_min)
    
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
        cur_min = search(next_node, d, dep+1, point, _cur_min)
        if euclidean_distance(cur_min[0], point) < euclidean_distance(_cur_min[0], point):
            _cur_min = cur_min
    
    return _cur_min

def search_k(node, d, dep, point, k, closepoinst):
    if node is None:
        return closepoinst
    
    dim = dep % d

    if point[dim] < node.val[dim]:
        flag = 0
        _closepoinst = search_k(node.lchild, d ,dep+1, point, k, closepoinst)
    else:
        flag = 1
        _closepoinst = search_k(node.rchild, d, dep+1, point, k, closepoinst)

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
            _closepoinst = search_k(node.rchild, d, dep+1, point, k, _closepoinst)
        else:
            _closepoinst = search_k(node.lchild, d, dep+1, point, k, _closepoinst)
    
    return _closepoinst


def _classifier_kd_tree(train_data, train_target, classes, point, k=5):
    dim = train_data.shape[1]
    #print(f"Feature dimension : {dim}")
    _data = np.hstack((train_data, np.expand_dims(train_target,axis=1)))
    root = build_kd_tree(_data, dim, 0)
    closepoint = search_k(root, dim, 0, point, k, None)
    #print(closepoint)
    votes = [int(p[2]) for p in closepoint]
    counts = np.bincount(votes)
    return classes[np.argmax(counts)]


def classifier(train_data, train_target, classes, point, k=5, type='base'):
    if type == 'base':
        return _classifier_base(train_data, train_target, classes, point, k)
    elif type == 'kd_tree':
        return _classifier_kd_tree(train_data, train_target, classes, point, k)
    else:
        print("Error type!")
        return None



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

    '''
    test_point = np.array([4.4, 3.1, 1.3, 1.4])
    print(classifier(X_train, Y_train, classes, test_point, 5, 'kd_tree'))
    
    