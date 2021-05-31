### DBSCAN聚类


#### 相关概念

**点的eps邻域**：在某点处，给定其半径eps后，所得到的覆盖区域。
核心点：对于给定的最少样本量minpts，如果某点的p的eps邻域内至少包含minpts个样本点，则点p就为核心对象。

**直接密度可达**：假设点p为核心点，且在点p的eps邻域内存在点q，则从点p出发到点q是直接密度可达的。

**密度可达**：假设存在一系列的对象链，$p_1,p_2,...,p_k$，如果每一个$p_i$直接密度可达$p_{i+1}$，则$p_1$密度可达$p_k$。

**密度相连**：假设点o为核心点，从点o出发可以得到两个密度可达点p和q，则称点p和点q是密度相连的。

**聚类的簇**：簇包含了最大的密度相连所构成的样本点。

**边界点**：假设点p为核心点，在其邻域内包含了点b，若b不是核心点，则称其为点p的边界点。

**异常点**：不属于任何簇的样本点。

#### 核心算法

1) 随机选择一个没有访问过的点p，若在其eps邻域内有大于等于minpts个点，则p为核心点。

2) 若p不是核心点，则跳过。

3) 若p是核心点，则找出从点p出发的密度可达的所有点，形成新簇，同时标记这些点已被访问。

4) 若还有没有访问的点，返回第一步。

#### 核心代码：

```python
    def fit(self, x):
        n = x.shape[0]
        neighbor = [[] for i in range(n)]
        for i in range(n):
            for j in range(n):
                if self.eps > euclidean_distance(x[i], x[j]):
                    neighbor[i].append(j)                   #寻找点i的eps邻域

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
```