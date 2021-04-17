
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

def classifier(train_data, train_target, classes, point, k=5):
    '''
    K近邻的核心思想就是：根据给定的距离度量，在训练集T中找出与x最邻近的k个点
    再根据分类决策规则决定x的类别y
    '''
    data = zip(train_data, train_target)
    distance = []
    for sample, t in data:
        distance.append((euclidean_distance(sample, point), t))
    distance = sorted(distance)
    votes = [i[1] for i in distance[:k]]
    counts = np.bincount(votes)
    return classes[np.argmax(counts)]

#todo kd_tree

class TreeNode(object):
    def __init__(self, val=-1, lchild=None, rchild=None):
        self.val = val
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
    
    node = TreeNode(med_val)
    if data_left.shape[0] > 0:
        node.lchild = build_kd_tree(data_left, k, dep+1)
    if data_right.shape[0] > 0:
        node.rchild = build_kd_tree(data_right, k, dep+1)
    return node

def search(node, k, dep, point, cur_min=None):
    if node is None:
        return cur_min

    dim = dep % k

    if point[dim] < node.val[dim]:
        flag = 0
        _cur_min = search(node.lchild, k, dep+1, point, cur_min)
    else:
        flag = 1
        _cur_min = search(node.rchild, k, dep+1, point, cur_min)
    
    search_next = 0
    if _cur_min is None:
        _cur_min = node.val
        search_next = 1
    elif euclidean_distance(node.val, point) < euclidean_distance(_cur_min, point):
        _cur_min = node.val
        search_next = 1

    if search_next == 0:
        if flag == 0:
            next_node = node.rchild
        else:
            next_node = node.lchild
        radiu = euclidean_distance(_cur_min, point)
        if radiu > abs(point[dim] - node.val[dim]):
            cur_min = search(next_node, k, dep+1, point, _cur_min)
            if euclidean_distance(cur_min, point) < euclidean_distance(_cur_min, point):
                _cur_min = cur_min
    
    return _cur_min

def _classifier_kd_tree(train_data, train_target, classes, point, k=5):
    

if __name__ == '__main__':
    test_point = np.array([4.4, 3.1, 1.3, 1.4])
    #print(classifier(X_train, Y_train, classes, test_point))
    distance = []
    for index, sample in enumerate(X_train):
        distance.append((euclidean_distance(sample, test_point), index))
    distance = sorted(distance)
    print(X_train[distance[0][1]])

    node = build_kd_tree(X_train, 4, 0)
    val = search(node, 4, 0, test_point)
    print(val)



