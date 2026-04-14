import numpy as np
import matplotlib.pyplot as plt

#先定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    '''
    计算正则化的梯度下降
    :param
     X:表示数据值（特征和训练数据）,ndarray(m,n)
     y:表示为标签,ndarray(m,)
     w:ndarray(n,)参数
     b:scalar
     这里计算我们的梯度下降，要对其求导
    '''
    m,n = X.shape
    dj_dw=np.zeros(n) #用于储存数据的梯度值，n个特征对应n个权重参数，先初始化
    dj_db= 0
    for i in range(m):
        #X[i]表示第i行的特征数据，w表示权重参数，b表示偏置项（所以向量相乘的结果是一个值）
        err = sigmoid(np.dot(X[i], w) + b) -y[i] #    相当于输入参数z
        dj_dw = dj_dw + err* X[i]  #err相当于一个常数，把x[i]这个n个的数组加起来的的结果赋值给对应的参数的数组
        dj_db = dj_db + err
    dj_db = dj_db/m
    dj_dw = dj_dw/m + (lambda_*w)/m #正则化项的梯度，lambda_是正则化强度，w是权重参数，m是样本数量
    return dj_dw,dj_db

def compute_cost_logistic_reg(X, y, w, b, lambda_): #用于显示当前的损失值
    m, _ = X.shape
    z = X @ w + b
    h = sigmoid(z)
 
    eps = 1e-12
    h = np.clip(h, eps, 1 - eps)
    loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return loss + reg

def gradient_descent_logistic_reg(X, y, w_tmp, b_tmp, lambda_tmp,alpha,iterations, print_every=0, track_loss=False):
    w = w_tmp.copy()
    b = b_tmp
    loss_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient_logistic_reg(X, y, w, b, lambda_tmp)
        w = w-alpha*dj_dw
        b= b- alpha*dj_db
        if print_every and (i % print_every == 0 or i == iterations - 1):
            cost = compute_cost_logistic_reg(X, y, w, b, lambda_tmp)
            print(f"iter={i+1:5d}/{iterations}, loss={cost:.6f}")
            if track_loss:
                loss_history.append((i + 1, cost))
    if track_loss:
        return w, b, loss_history
    return w,b

#函数调用
np.random.seed(1)  # 固定随机种子，保证每次结果一致

# 构造更适合学习展示的样本：x1/x2 两簇，x3 为小噪声特征
n_per_class = 120
class0_x12 = np.random.normal(loc=[0.30, 0.35], scale=[0.12, 0.10], size=(n_per_class, 2))
class1_x12 = np.random.normal(loc=[0.72, 0.68], scale=[0.12, 0.11], size=(n_per_class, 2))

class0_x3 = np.random.normal(loc=0.45, scale=0.10, size=(n_per_class, 1))
class1_x3 = np.random.normal(loc=0.55, scale=0.10, size=(n_per_class, 1))

X_class0 = np.hstack([class0_x12, class0_x3])
X_class1 = np.hstack([class1_x12, class1_x3])
X_tmp = np.vstack([X_class0, X_class1])
y_tmp = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

# 限制到可视范围，避免少量异常点影响观感
X_tmp = np.clip(X_tmp, 0.0, 1.0)

w_tmp = np.random.randn(X_tmp.shape[1]) * 0.1
b_tmp = 0.5
lambda_tmp = 0.2
iterations = 10000
alpha = 0.1
print_every = 300
w, b, loss_history = gradient_descent_logistic_reg(
    X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp, alpha,
    iterations=iterations, print_every=print_every, track_loss=True
)

train_prob = sigmoid(X_tmp @ w + b)
train_pred = (train_prob >= 0.5).astype(int)
train_acc = np.mean(train_pred == y_tmp)
print(f"samples={len(y_tmp)}, class0={(y_tmp == 0).sum()}, class1={(y_tmp == 1).sum()}, train_acc={train_acc:.3f}")

# --- 3. 二维分类图（固定 Feature 3） ---
fig, ax = plt.subplots(figsize=(8, 6))

# A. 绘制二维散点（x1, x2）
ax.scatter(X_tmp[y_tmp == 0, 0], X_tmp[y_tmp == 0, 1],
           c='blue', marker='o', s=100, label='Class 0 (y=0)')
ax.scatter(X_tmp[y_tmp == 1, 0], X_tmp[y_tmp == 1, 1],
           c='red', marker='x', s=100, label='Class 1 (y=1)')

# B. 固定 x3 为均值，在 x1-x2 平面上绘制分类区域和决策边界
x3_fix = np.mean(X_tmp[:, 2])
x1_min, x1_max = X_tmp[:, 0].min() - 0.1, X_tmp[:, 0].max() + 0.1
x2_min, x2_max = X_tmp[:, 1].min() - 0.1, X_tmp[:, 1].max() + 0.1
xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

grid = np.column_stack([
    xx.ravel(),
    yy.ravel(),
    np.full(xx.size, x3_fix)
])
proba = sigmoid(grid @ w + b).reshape(xx.shape)
pred = (proba >= 0.5).astype(int)

ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], alpha=0.15, colors=['blue', 'red'])
ax.contour(xx, yy, proba, levels=[0.5], colors='green', linewidths=2)

# 设置标签和图例
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title(
    f'2D Classification Map (N={len(y_tmp)}, Feature3 fixed={x3_fix:.2f})\n'
    f'w={w.round(2)}, b={round(b, 2)}, train_acc={train_acc:.3f}'
)
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.legend()
ax.grid(alpha=0.2)

plt.show()
