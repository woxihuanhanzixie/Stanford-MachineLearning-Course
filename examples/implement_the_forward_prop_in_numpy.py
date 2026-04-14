import numpy as np
import tensorflow as tf

def sigmoid(x):
    return 1/(1+np.exp(-x))

g = sigmoid

#但是这里是对单个样本输出，所以输出的维度是units
#应该满足多行的输入！
def my_dense(a_in , W, b):
    '''
    a_in: ndarray (m,n) 输入数据,m是样本个数,n表示的是特征的数量
    W: ndarray (n, j) 权重矩阵,i表示输入层的神经元所具有的参数个数j,表示该层神经元的个数
    b: ndarray (j,) 偏置向量,j是输入神经元个数
    Returns
       a_out (ndarray (m,j))  : m个样本,j个单位
    ''' 
    units = W.shape[1]
    m = a_in.shape[0]
    a_out = np.zeros((m,units))  #对于两个参数的数据输入构建矩阵需要使用tuple的形式
   
    #但是实际上可以使用numpy的广播形式，也就是.matmul函数来计算矩阵相乘
    z_out = np.matmul(a_in, W) + b
    a_out = g(z_out)
    return a_out

def my_sequential(X, W1, b1, W2, b2):
    '''
    X: ndarray (m,i) 这个是输入数据,m是样本个数,i表示的是特征的数量
    W1: ndarray (i,j) 第一层权重矩阵,i表示输入层的神经元所具有的参数个数j,表示该层神经元的个数
    b1: ndarray (j,) 第一层偏置向量,j是输入神经元个数
    W2: ndarray (j,k) 第二层权重矩阵,j表示第一层神经元个数,k表示第二层神经元个数
    b2: ndarray (k,) 第二层偏置向量,k是第二层神经元个数
    Returns
       f_x (ndarray (m,k))
    ''' 
    layer1 = my_dense(X, W1, b1)  #第一层的输出结果
    layer2 = my_dense(layer1, W2, b2)  #第二层的输出结果
    f_x = layer2
    return f_x

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def my_predict(X, W1, b1, W2, b2):
    # 直接批量前向，避免逐样本循环带来的维度问题
    p = my_sequential(X, W1, b1, W2, b2)
    return p

X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X_tst)
X_tstn = norm_l(X_tst).numpy()  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")