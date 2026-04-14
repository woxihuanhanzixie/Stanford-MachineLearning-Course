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


if __name__ == "__main__":
    # 简单一元线性回归示例
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

    w_init = np.zeros(X_train.shape[1])
    b_init = 0.0
    alpha = 0.01
    iters = 3000

    w_final, b_final, j_history = gradient_descent(X_train, y_train, w_init, b_init, alpha, iters)
    print(f"\n训练完成: w={w_final}, b={b_final:.4f}, final_cost={j_history[-1]:.6f}")

    # 绘制拟合结果与损失曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(X_train[:, 0], y_train, c="steelblue", label="Train Data")
    x_line = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    y_line = w_final[0] * x_line + b_final
    axes[0].plot(x_line, y_line, c="tomato", label="Fitted Line")
    axes[0].set_title("Linear Regression Fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(j_history, c="purple")
    axes[1].set_title("Cost History")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cost")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()