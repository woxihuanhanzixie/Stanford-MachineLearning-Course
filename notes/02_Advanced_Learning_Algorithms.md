#先决背景
对于神经网络，是模仿生物的神经突触的连接，所以可以预测到为什么神经网络需要激活函数，就是相当于传递信息给下一个神经节，也就是0,1信号，并且对于这个深度学习的算法是简化实际模型创建的，但是拟合效果却更好说明要分析好实际的模型创建的关键是问题简化和构建关系

## 神经网络先决直觉
### 激活函数sigmoid
使用激活函数接受参数作为输入层（就是多个函数放在一个层来接受参数，再将这个层的结果输出给下一层达到构成线性的关系（本质就是大量的线性组合）![[image.png|424x286]]
但是你需要考虑，什么时候对应的神经元接受哪几个特征的参数作为输入？怎么样输出？需不需要特征工程？同时还有神经网络的架构是什么？要几层？怎么分配隐藏层个数？字母分配神经元的个数？![[image-1.png|371x281]]

对于神经网路的training，其实拆分出很多层，将这些层分开计算处理得到的其实为不同的原数据的类型，example：对于图像数据集，其实多层感知机就是在处理图像的不同部分以及不同大小，有边缘检测和整体的对比等。

##  神经网络model
神经网络的每一层多个激活函数sigmoid函数，将该层的值输出到下一层的激活函数，（也就是上一个函数层的函数输出变为下一个函数层的输入值使得最终有一个结果。qia（输出的值是一个向量，一个神经元对应一个输出的值（若只有一个神经元，如在最后一层输出层，输出的是一个标量））
仔细看这个神经网络的图像，其实每一层的传播都是相对应的，这里和矩阵的运算比较多关系是因为前一层的输出作为下一个函数的神经元的输入，输出值可以是向量或者标量（所有这里写为点积的形式）
但是具体的细节是，对于任何神经层的输入都是应该向量的形式而不能是矩阵，若为多维地矩阵n x n类型的矩阵需要将其展平flatten，变为`[1,n^2]`或 `n^2,1]`,所以对于每一个神经元的参数w的个数是`a[l-1]` （由上一个层的输出，也就是这一层参数的输入决定）
![[image-2.png]]
$$
aj[l]​=g(wj[l]​⋅a[l−1]+bj[l]​)
$$

### forward：向前传播算法
[基本代码实现](D:\CS\CS_CodeSpace\ml-study\lab_jupyter_notebook\神经网络\lab1\Files\home\jovyan\work\C2_W1_Lab02_CoffeeRoasting_TF.ipynb)
使用过tensorflow实现神经层的构建时候，要对每一层进行处理
```python
layer_1 = dense(units = 25, activation = 'sigmoid')
#这里输入的是神经元个数，激活函数的类型
a1 = layer_1(X_train) #但是输入必须是二维的array形式，也就是[[]]的形式
#输出一个向量a1给下一个神经层
```
注意使用tensor构建神经网络的时候要注意数入参数的类型是二维，且再tensor内部的张量是tensor.而不是numpy（相当于有两个构成类型），
使用`.sequential`方法来构建模型（这样的神经层就是链接好的）
```python
model = sequential(
[
Dense(units = 23, activation = 'sigmoid', name = 'layer1')
Dense(21,activation = 'sigmoid', name = 'layer2')
]
)
```
使用数据标准化，可以使得数据和模型的权重拟合的更好，所以在数入数据前先对模型数据进行标准化处理

==注意区分神经元个数和神经元内部权重的参数个数==
![[image-4.png|523x234]]
### python实现神经网络的一般原理
其实就是把所有的向量相乘的过程使用循环和numpy实现但是这里暂未学习具体的梯度下降的过程和对应算法[python实现](D:\CS\CS_CodeSpace\ml-study\project_demo\implement_the_forward_prop_in_numpy.py)，注意这里的代码实现中参数w是放为列向量的形式，所以列数代表为神经元的个数。==脑海里有对应的可以实现的图像很重要！是什么样的==![[image-3.png|619x352]]
但是本质上的数学公式就是$Z=W\cdot X$
但是！很重要的是数据的输入！有多少个数据！我们是输入n个数据的，需要的是n行的样本空间，也就是输入的如果是$m\times n$的矩阵，该层有j个神经元，那么输出的时候应该是$m\times j$矩阵，因为特征值是n个,有j个神经元输出，而且每一个神经元的参数个数是n，这里是对每一个样本都需要训练的！也就是有m个样本![[image-5.png|636x190]]
也就是可以实现
```python
def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
    z = np.matmul(A_in,W) + b
    A_out = g(z)
    
    return(A_out)
```