import numpy as np
import matplotlib.pyplot as plt
import copy,math
# 这是修复后的一个测试
'''
这里是线性回归算法的原理复现
'''
def compute_cost(x, y, w, b):
    '''
    多维线性回归的损失函数：
    x表示为多维特征数据 np.array((m,n))
    y表示为预测值，np.array(m,)
    w表示多个权重 np.array(n,)
    b表示偏置项，标量
    返回线性回归所需要的损失函数值
    '''
    m=x.shape[0]
    err=0.0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b #利用索引到第i行的特征数据，和权重进行点积运算，再加上偏置项（相当于而喂喂喂数组）
        err+=(f_wb-y[i])**2
    cost=err/(2*m)
    return cost

def compute_gradient(x, y, w, b):
    '''
    计算多元线性回归的梯度
    x表示为多维特征数据 np.array((m,n))
    y表示为预测值，np.array(m,)
    w表示多个权重 np.array(n,)
    b表示偏置项，标量
 Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    '''
    m,n=x.shape
    dj_dw=np.zeros((n,))
    dj_db=0.0
    for i in range(m):
        f_wb=np.dot(x[i], w) + b
        err=(f_wb-y[i])
        # for j in range(n):
        #     dj_dw[j]+=err*x[i][j]
        # dj_db+=err
        # 利用广播机制
        dj_dw+=err*x[i]
        dj_db+=err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    '''
    执行批量梯度下降算法来学习参数w和b
    x表示为多维特征数据 np.array((m,n))
    y表示为预测值，np.array(m,)
    w_in表示多个权重 np.array(n,)
    b_in表示偏置项，标量
    alpha表示学习率，标量
    num_iters表示迭代次数epochs，标量
    '''
    j_history=[] # 用于记录每次迭代的损失函数值
    w = copy.deepcopy(w_in)  # 避免在函数中修改全局W
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            j_history.append(compute_cost(x, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"迭代{i:4d}: 损失函数值 {j_history[-1]:8.2f}")
    return w, b, j_history
# 实例化
# 带入数据即可