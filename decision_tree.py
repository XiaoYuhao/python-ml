
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math
#data = datasets.load_wine()
#data = datasets.load_iris()

from datasets import watermelon

data = watermelon


X = np.array(data["data"])
Y = np.array(data['target'])
#classes = data["target_names"]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

class _cart_tree_node:
    def __init__(self, c=-1):
        self.c = c                      #叶子节点代表的类别
        self.lchild = None            #左子树节点
        self.rchild = None            #右子树节点
        self.select_feature = None      #选择的特征
        self.feature_type = None        #特征的类型（离散型或连续型）
        self.decsion_val = None         #作为判别依据的特征值

class _id45_tree_node:
    def __init__(self, c=-1):
        self.c = c                      #叶子节点代表的类别
        self.childs = list()            #多叉树的子树节点
        self.select_feature = None      #选择的特征
        self.feature_type = None        #特征的类型（离散型或连续型）
        self.decsion_val = None         #作为判别依据的特征值（对应于每颗子树）


class DecisionTree:
    def __init__(self, sample_thresh=5, thresh=0, clip_alpha=2.0):
        self.sample_thresh = sample_thresh
        self.thresh = thresh
        self.clip_alpha = clip_alpha
        pass

    def _build_tree_id45(self, x_train, y_train, x_type, x_select):
        if np.unique(y_train).shape[0] == 1 or x_select.shape[0] == 0:
            print("Leaf node")
            node = _id45_tree_node(c=np.argmax(np.bincount(y_train)))
            return node

        feature_dim = x_train.shape[1]
        classes = np.unique(y_train)
        assert feature_dim == x_type.shape[0]

        h_d = 0
        for i in range(classes.shape[0]):
            ratio = np.where(y_train==classes[i])[0].shape[0] / y_train.shape[0]
            h_d += ratio * math.log2(ratio)
        h_d = - h_d
        gain_max = 0
        for i in x_select:
            a = x_train[:, i]
            if x_type[i] == 'D':
                a = np.unique(a)
                ds = list()
                for v in a:
                    ds.append(np.where(x_train[:,i]==v)[0])
                gain = 0
                for j in range(len(ds)):
                    h_di = 0
                    for k in range(classes.shape[0]):
                        p = np.where(y_train[ds[j]]==classes[k])[0].shape[0] / y_train[ds[j]].shape[0]
                        if p > 0.0:
                            h_di += p * np.log2(p)
                    gain += (ds[j].shape[0] / x_train.shape[0]) * h_di
                gain = h_d + gain
            elif x_type[i] == 'C':
                a = np.sort(a)
                _a = list()
                for j in range(a.shape[0] - 1):
                    _a.append((a[j] + a[j+1]) / 2.0)
                a = np.array(_a)
                _gain_max = 0
                for v in a:
                    d1 = np.where(x_train[:,i] <= v)[0]
                    d2 = np.where(x_train[:,i] > v)[0]
                    gain = 0
                    h_d1 = 0
                    h_d2 = 0
                    for k in range(classes.shape[0]):
                        p1 = np.where(y_train[d1]==classes[k])[0].shape[0] / y_train[d1].shape[0]
                        p2 = np.where(y_train[d2]==classes[k])[0].shape[0] / y_train[d2].shape[0]
                        if p1 > 0.0:
                            h_d1 += p1 * np.log2(p1)
                        if p2 > 0.0:
                            h_d2 += p2 * np.log2(p2)
                    gain += (d1.shape[0] / x_train.shape[0]) * h_d1 + (d2.shape[0] / x_train.shape[0]) * h_d2
                    gain = h_d + gain
                    if gain >= _gain_max:
                        _gain_max = gain
                        _v = v
                gain = _gain_max
            print(i, h_d, gain)
            if gain > gain_max:
                gain_max = gain
                select_i = i
                if x_type[i] == 'D':
                    select_type = 'D'
                else:
                    select_type = 'C'
                    select_v = _v

        if gain_max <= self.thresh:
            node = _id45_tree_node(c=np.argmax(np.bincount(y_train)))
            return node

        node = _id45_tree_node()
        if select_type == 'D':
            a = np.unique(x_train[:,select_i])
            ds = list()
            node.feature_type = 'D'
            node.select_feature = select_i
            node.decsion_val = list()
            for v in a:
                ds.append(np.where(x_train[:,select_i]==v)[0])
                node.decsion_val.append(v)
            for j in range(len(ds)):
                x_train_d = x_train[ds[j]]
                y_train_d = y_train[ds[j]]
                x_select_d = np.delete(x_select, np.where(x_select==select_i)[0])
                node_d = self._build_tree_id45(x_train_d, y_train_d, x_type, x_select_d)
                node.childs.append(node_d)
        else:
            node.feature_type = 'C'
            node.select_feature = select_i
            node.decsion_val = select_v
            d1 = np.where(x_train[:,select_i] <= select_v)[0]
            x_train_d1 = x_train[d1]
            y_train_d1 = y_train[d1]
            node_d1 = self._build_tree_id45(x_train_d1, y_train_d1, x_type, x_select)
            d2 = np.where(x_train[:,select_i] > select_v)[0]
            x_train_d2 = x_train[d2]
            y_train_d2 = y_train[d2]
            node_d2 = self._build_tree_id45(x_train_d2, y_train_d2, x_type, x_select)
            node.childs = [node_d1, node_d2]
        
        return node

    def _clip_tree_id45(self, node, x_train, y_train):
        # 计算经验熵
        h = 0
        classes = np.unique(y_train)
        for i in range(classes.shape[0]):
            ratio = np.where(y_train==classes[i])[0].shape[0] / y_train.shape[0]
            h += ratio * math.log2(ratio)
        h = -h
        if node.c >= 0:
            return h
        
        if node.feature_type == 'D':
            select_f = node.select_feature
            C_T = 0
            for val, child in zip(node.decsion_val, node.childs):
                d = np.where(x_train[:, select_f]==val)[0]
                x_train_d = x_train[d]
                y_train_d = y_train[d]
                h_t = self._clip_tree_id45(child, x_train_d, y_train_d)
                C_T += d.shape[0] * h_t
        else:
            select_f = node.select_feature
            C_T = 0
            d1 = np.where(x_train[:, select_f] <= node.decsion_val)[0]
            x_train_d1 = x_train[d1]
            y_train_d1 = y_train[d1]
            h_t1 = self._clip_tree_id45(node.childs[0], x_train_d1, y_train_d1)
            d2 = np.where(x_train[:, select_f] > node.decsion_val)[0]
            x_train_d2 = x_train[d2]
            y_train_d2 = y_train[d2]
            h_t2 = self._clip_tree_id45(node.childs[1], x_train_d2, y_train_d2)
            C_T = d1.shape[0] * h_t1 + d2.shape[0] * h_t2
        
        C_T_alpha_before = C_T + self.clip_alpha * (len(node.childs))   #未剪枝之前的损失函数
        C_T_alpha_after = y_train.shape[0] * h + self.clip_alpha        #剪枝之后，节点变为叶节点，此时的损失函数

        if C_T_alpha_before >= C_T_alpha_after:                    #进行剪枝
            node.c = np.argmax(np.bincount(y_train))
            node.childs = list()
            node.select_feature = None
            node.select_type = None
            node.decsion_val = None

        return h

    def _build_tree_cart(self, x_train, y_train, x_type):
        if x_train.shape[0] < self.sample_thresh:
            Node = _cart_tree_node(c=np.argmax(np.bincount(y_train)))
            return Node
            

        feature_dim = x_train.shape[1]
        classes = np.unique(y_train)
        assert feature_dim == x_type.shape[0]

        gini_min = 2
        for i in range(feature_dim):
            a = x_train[:, i]
            if x_type[i] == 'D':        #离散值
                a = np.unique(a)
            elif x_type[i] == 'C':      #连续值      
                a = np.sort(a)
                _a = list()
                for j in range(a.shape[0] - 1):
                    _a.append((a[j] + a[j+1]) / 2.0)
                a = np.array(_a)
            for v in a:
                if x_type[i] == 'D':
                    d1 = np.where(x_train[:,i] == v)[0]
                    d2 = np.where(x_train[:,i] != v)[0]
                if x_type[i] == 'C':
                    d1 = np.where(x_train[:,i] <= v)[0]
                    d2 = np.where(x_train[:,i] > v)[0]
                gini_d1 = 0
                gini_d2 = 0
                for j in range(classes.shape[0]):
                    if d1.shape[0] != 0:
                        gini_d1 += (np.where(y_train[d1]==classes[j])[0].shape[0] / d1.shape[0]) ** 2
                    if d2.shape[0] != 0:
                        gini_d2 += (np.where(y_train[d2]==classes[j])[0].shape[0] / d2.shape[0]) ** 2
                gini_d1 = 1 - gini_d1
                gini_d2 = 1 - gini_d2
                gini = gini_d1 * (d1.shape[0] / (d1.shape[0] + d2.shape[0])) + gini_d2 * (d2.shape[0] / (d1.shape[0] + d2.shape[0]))
                if gini_min > gini:
                    gini_min = gini
                    select_i = i
                    select_a = v
        
        if gini_min < self.thresh:
            Node = _cart_tree_node(c=np.argmax(np.bincount(y_train)))
            return Node

        if x_type[select_i] == 'D':
            d1 = np.where(x_train[:, select_i] == select_a)[0]
            d2 = np.where(x_train[:, select_i] != select_a)[0]
        elif x_type[select_i] == 'C':
            d1 = np.where(x_train[:, select_i] <= select_a)[0]
            d2 = np.where(x_train[:, select_i] > select_a)[0]

        if d1.shape[0] == 0 or d2.shape[0] == 0:
            Node = _cart_tree_node(c=np.argmax(np.bincount(y_train)))
            return Node

        x_train_d1 = x_train[d1]
        y_train_d1 = y_train[d1]
        x_train_d2 = x_train[d2]
        y_train_d2 = y_train[d2]

        #print(select_i, select_a)

        Node = _cart_tree_node()
        Node.lchild = self._build_tree(x_train_d1, y_train_d1, x_type)
        Node.rchild = self._build_tree(x_train_d2, y_train_d2, x_type)
        Node.select_feature = select_i
        Node.feature_type = x_type[select_i]
        Node.decsion_val = select_a
        return Node

    def _dfs(self, node):
        if node.c >= 0:
            print("Left Node : ", node.c)
        for child in node.childs:
            self._dfs(child)

    def fit(self, x_train, y_train, x_type):
        self.x_type = x_type
        #self.root = self._build_tree_cart(x_train, y_train, x_type)
        x_select = np.array([i for i in range(x_type.shape[0])])
        self.root = self._build_tree_id45(x_train, y_train, x_type, x_select)
        print("Before clip")
        self._dfs(self.root)
        self._clip_tree_id45(self.root, x_train, y_train)
        print("After clip")
        self._dfs(self.root)

    def _search_id45(self, node, x):
        if node.c >= 0:
            return node.c
        if node.feature_type == 'D':
            index = node.decsion_val.index(x[node.select_feature])
            return self._search_id45(node.childs[index], x)
        if node.feature_type == 'C':
            if x[node.select_feature] <= node.decsion_val:
                return self._search_id45(node.childs[0], x)
            else:
                return self._search_id45(node.childs[1], x)

    def pred(self, point):
        return self._search_id45(self.root, point)
        
 
    def eval(self, x_test, y_test, x_type):
        acc = 0
        num = 0
        for _x, _y in zip(x_test, y_test):
            print(_x, self.pred(_x))
            if _y == self.pred(_x):
                acc += 1
            num += 1
        acc /= num
        print(f"Accuary: {acc}")


if __name__ == '__main__':
    dt = DecisionTree()
    x_type = data['xtype']
    train_set = data['train']
    val_set = data['val']
    #print(X[train_set,:6])
    #print(Y)
    #print(x_type)
    #dt.fit(X[train_set,:6], Y[train_set], x_type[:6])
    #dt.eval(X[val_set,:6], Y[val_set], x_type[:6])
    dt.fit(X[:,:6], Y, x_type[:6])
