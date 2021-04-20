### 线性回归

线性模型形式简单、易于建模，但却蕴涵着机器学习中一些重要的基本思想。许多功能更为强大的非线性模型可在线性模型的基础上通过引入层级结构或高维映射而得。此外，由于w直观表达了各属性在预测中的重要性，因此线性模型由很好的可解释性。

向量形式：

$$f(x) = \pmb{\omega}^Tx+b $$

为了便于讨论，我们把$\omega$和$b$合并成为$\hat{\omega}=(\omega;b)$，相应地，把数据集D表示为一个$m \times (d+1)$大小的矩阵$X$，其中每行对应于一个示例，该行前d个元素对应于示例的$d$个属性值，最后一个元素横置为1，即：

$$
\pmb{X} = 
\left\{
\begin{matrix}
x_{11} & x_{12} & ... & x_{1d} & 1 \\
x_{21} & x_{22} & ... & x_{2d} & 1 \\
x_{31} & x_{32} & ... & x_{3d} & 1 \\
... & ... & ... & ... & ... \\ 
x_{m1} & x_{m2} & ... & x_{md} & 1 
\end{matrix}
\right\}
$$

将标记也写成向量的形式：

$$
\pmb{y} = \{y_1;y_2;y_3;...;y_m\}
$$

均方误差函数：

$$
E_{\hat{\pmb{\omega}}} = {(\pmb{y} - \pmb{X}{\hat{\pmb{\omega}}})}^T(\pmb{y} - \pmb{X}{\hat{\pmb{\omega}}})
$$

对$\hat{\pmb{\omega}}$求导可得：

$$
\frac{\delta{E_{\hat{\pmb{\omega}}}}}{\delta{\pmb{\hat{\omega}}}} = 2 \pmb{X}^T (\pmb{X\hat{\omega}-y})
$$

令上式为零，可得$\hat{\pmb{\omega}}$最优解

即 $\pmb{X^TX\hat{\omega}} = \pmb{X^Ty}$，解此线性方程组可求得$\hat{\pmb{\omega}}^*$

Python代码如下：

```python
class LinearRegression:
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        X = np.column_stack((x_train, np.ones(x_train.shape[0])))
        self.W = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, y_train))

    def pred(self, point):
        return np.dot(self.W, np.concatenate((point, np.ones(1))))

```

