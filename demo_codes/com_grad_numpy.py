#coding:utf-8
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# 以下两行代码解决jupyter notebook显示图片模糊问题
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
##  测试数据
x1 = np.arange(0, 10, 0.2) # len=50
x2 = np.arange(-10, 0, 0.2)
y = 10 * x1 + 3 * x2 + 5 + np.random.random((len(x1))) * 20

# 计算X*W^T
def f(X, W):
    return X.dot(W.T)

## 初始化参数
W = np.mat(np.random.random([3])*2-1)
X = np.mat(np.column_stack((np.ones(len(x1)), x1, x2)))  # 按列组合:1,x1,x2;第1列是bias,对应的W0
Y = np.mat(y).T
print(X.shape)
print(W.shape)
print(y.shape)

# 梯度下降
lr = 0.00005
for i in range(60):
    # 计算与更新梯度
    w = -2 * (Y - X.dot(W.T)).T.dot(X)  # L=(y-y_)^2
    W = W - lr * w

# 绘图
Y_Test = f(X, W)
x = np.arange(len(Y))+1
plt.scatter(x, list(Y), s=10)
plt.plot(x, Y, 'b')
plt.plot(x, Y_Test, 'r')
plt.show()

