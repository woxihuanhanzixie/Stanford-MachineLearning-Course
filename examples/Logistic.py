import numpy as np
import matplotlib.pyplot as plt
import copy, math


"""
逻辑回归算法的原理复现 (Logistic Regression Implementation)
"""

def sigmoid(z):
    """
    计算 Sigmoid 函数
    Args:
        z (scalar or ndarray): 输入值
    Returns:
        g (scalar or ndarray): sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, lambda_=0):
    """
    计算逻辑回归的损失函数 (带正则化)
    Args:
      X (ndarray (m,n)): 特征数据
      y (ndarray (m,)):  目标值
      w (ndarray (n,)):  权重
      b (scalar):        偏置
      lambda_ (scalar):  正则化参数
    Returns:
      total_cost (scalar): 损失函数值
    """
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    
    cost = cost / m
    
    reg_cost = 0 # 正则化项的成本
    if lambda_ != 0:
        for j in range(n):
            reg_cost += (w[j]**2)
        reg_cost = (lambda_ / (2 * m)) * reg_cost
        
    total_cost = cost + reg_cost
    return total_cost

def compute_gradient(X, y, w, b, lambda_=0):
    """
    计算逻辑回归的梯度 (带正则化)
    Args:
      X (ndarray (m,n)): 特征数据
      y (ndarray (m,)):  目标值
      w (ndarray (n,)):  权重
      b (scalar):        偏置
      lambda_ (scalar):  正则化参数
    Returns:
      dj_dw (ndarray (n,)): 权重梯度
      dj_db (scalar):       偏置梯度
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
        dj_db = dj_db + err_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    #这里多增加了一个lambda_ != 0的条件
    if lambda_ != 0:
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]
            
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0):
    """
    执行梯度下降算法
    """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i < 100000:
            J_history.append(compute_cost(X, y, w, b, lambda_))
            
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
            
    return w, b, J_history

# 简单的测试用例
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.
alph = 0.1
iters = 10000

w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nFinal parameters: w:{w_final}, b:{b_final}")

# 绘制二维分类散点图
# 这里绘制二维的散点图，x_train表示使用布尔索引取获得对应的label是0或者1的特征数据，取0或者1表示第一个还是第二个1特征值
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='o', label='y=0')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', marker='x', label='y=1')

# 绘制决策边界: w0*x0 + w1*x1 + b = 0  =>  x1 = -(w0*x0 + b) / w1
x0_min, x0_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
x0_boundary = np.array([x0_min, x0_max])
x1_boundary = -(w_final[0] * x0_boundary + b_final) / w_final[1]
#这里取了两个值作为画图的边界（两点连成线），x0_boundary是x轴的边界，x1_boundary是根据决策边界方程计算出的对应的y轴边界（预测值）
plt.plot(x0_boundary, x1_boundary, c='green', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression 2D Boundary')
plt.legend()
plt.show()