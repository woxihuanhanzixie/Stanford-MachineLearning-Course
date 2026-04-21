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
==要注意是使用向量的矢量化实现，参考矩阵的基本运算：[[Numpy使用]],记住矩阵就是特殊的，其是满足向量化计算的！==
很重要的是数据的输入！有多少个数据！我们是输入n个数据的，需要的是n行的样本空间，也就是输入的如果是$m\times n$的矩阵，该层有j个神经元，那么输出的时候应该是$m\times j$矩阵，因为特征值是n个,有j个神经元输出，而且每一个神经元的参数个数是n，这里是对每一个样本都需要训练的！也就是有m个样本![[image-5.png|636x190]]
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

## 神经网络的代价（COST）函数
注意loss function是对应的单个样本的误差，成本函数（cost function)是对应的所有样本的平均总和（对于一个训练集上的所有误差）![[image-6.png|563]]
#lossFunction：
$L=−[ylogσ(z)+(1−y)log(1−σ(z))]$
对于二元交叉熵的损失函数（主要使用二元分类问题），使用log可以对模型的分类有更大的惩罚和奖励机制。
使用反向传播的方式来实现梯度下降的获得模型最优解

## 激活函数
### ReLu function
$ReLu_function = max(0,z）$，又称为修正线性单元（因为不是单纯的线性函数但是又有其特性）
在ReLu函数内部，还是先使用权重W进行拟合数据传给参数`z`，*使用每个 ReLU 单元，本质就是一个「在 x=x0​ 处打开的、斜率为 w 的线性开关」ReLu函数取理解创建节点。*，然后输出的神经元为
$a=[a0​, a1​, a2​]T$ 形式，给下一个神经元，在做加法（也就是拼接拟合非线性的函数）

---

*补充：*线性激活函数：$g(z)=z$，此时数值可以是正数或者负数（常见于回归问题：也就是对于连续可以无限细分的值的预测或者计算）
**在隐藏层里 **：使用默认的Relu函数作为激活函数，可以加速优化模型和学习，但是在输出层可以使用特定的激活函数：
- 二元分类问题可以使用sigmoid函数
- 回归预测（且y可以小于0）使用线性激活函数
**对于神经元模型，不能使用单独的线性回归作为隐藏层因为这样构建出的结果就一定是线性回归模型！**

## 多分类问题
（后面在无监督学习中也会用到）
这里我们是使用归一化的方式来解决多分类的问题，这里在训练的时候隐藏层还是ReLu函数作为激活函数，但是最后的输出层才使用softmax算法
#softmax函数
![[image-8.png|612]]
#SOFTMAX分类下的cost函数：
==注意其和逻辑回归的g(z)以及损失函数不一样！！！==
![[image-9.png|523]]
这里和我们所知道的逻辑回归[机器学习Machine learning入门](机器学习Machine%20learning入门.md)内容中不一样的点在于其对于概率的预测变为了占比--也就是越接近哪一个训练值的占比（也就是最后一层做这个结果处理），然后对于具体是哪一个内容，是通过下标索引判断 表示其对应的标签是什么！
关于输出：最后一层的神经元层是由n个神经元对应由n个分类问题，然后把这个对应的aj再换算为比例 （也就是这个是输出还是有n个！）
![[image-10.png|491]]
怎么实现这个效果的？对于模型在最后一层softmax输出的时候，都是同一个上一个神经元的输入到这个层，但是通过梯度下降获得的最优解使得对应的权重获得的model效果最好，使得再最后一层输出的时候a1占比最高（这里假设是分类的结果是a1对应的第一个），这个是使用模型训练的结果（但是这样也导致模型的训练回更加困难）

## softmax工程实现
对于储存空间是有限的，所以在实际的案例里我们不可以直接再最后一层进行输出softmax结果，因为这样计算损失函数的时候相当于把这个概率的中间值`aj`引入，如上面的[多分类问题](#多分类问题)中的aj的计算是先使用 $aj  = g(zj)$ 来表示，但是这样会导致在储存的时候有精度误差！！！
所以在这里我们进行多分类问题里在训练中直接不引入aj，但是其实逻辑上是存在的
```python
model = Sequrntial([
	Dense(units = 25,activation = 'relu'),
	Dense(units = 15, activation = 'relu'),
	Dense(units = 1, activation = 'linear')
])
model.compile(loss = SparseCategorialCrossentropy(from_logits = True))
model.fit(X_train,y_train,epochs = 100)
```
*`from_logits=True`表示为在最后计算的时候会使用g(z)的算法，但是这里是直接输入zj进入而不是aj*（所以其实这里的处理是可以同样的用在逻辑分类问题上的）
**但是这里不是输出概率而是输出的是本身的值**，所以要再处理
```python
model_pridect = model.pridect(X_test)
sm_model_pridect = tf.nn.softmax(model_pridect).numpy()
pint("largest value", np.max(sm_model_pridect), "smallest value", np.min(sm_model_pridect))

#使用np.argmax()找到最大索引也就是概率
print(f"category:{np.argmax(am_model_pridect)}")
```


#最后的输出层解读
1. **逻辑上：换元思想**（因为计算机的精度不够对极小的数值会越界或者因为精度的原因储存为0/1导致结果不准确）
    
    损失函数内部依然在进行“计算概率”的操作。
    
    - **普通法：** $Loss = -y \cdot \log(a)$，其中 $a = \text{sigmoid}(z)$。
        
    - **改进法：** $Loss = -y \cdot \log(\text{sigmoid}(z))$。
        
        从数学上看，这两者是完全一样的。
        
2. **计算上：公式合并（收缩）**
    
    当你告诉 TensorFlow `from_logits=True` 时，它不会笨笨地先算出 $a = 0.0000000001$，然后再去算 $\log(a)$（这会导致误差）。
    
    相反，它利用对数和指数的数学特性（如 $\log(\frac{1}{1+e^{-z}})$ 可以化简为 $- \log(1+e^{-z})$），将公式合并成一个**数值更稳定**的形式。
***所以此时的结果不再是概率而是原本的值直接输出，需要使用softmax再获取一次数据！***
在逻辑上 其实现的损失函数没有改变[神经网络的代价（COST）函数](#神经网络的代价（COST）函数)只是最后的输出值不再是aj形式
![[image-11.png|644x501]]

#多标签分类问题：
对于在同一个训练输入的时候同时识别多个物体做多个分类。e.g:同一个照片识别有人有猫有狗等，而不是单独判断某一个物体是什么。
### 成本函数
$J(\mathbf{w},b) = -\frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{N}  1\left\{y^{(i)} == j\right\} \log \frac{e^{z^{(i)}_j}}{\sum_{k=1}^N e^{z^{(i)}_k} }\right] \tag{4}$
只有那些对应的是这个标签的函数值才会纳入计算！比如说是标签为第二个，那么只有第二个`exp(z2)`才会计算其余的是不参与这个损失函数的计算的
![[image-12.png|697|697x193]]
（这个其实和逻辑函数[机器学习Machine learning入门](机器学习Machine%20learning入门.md)中的逻辑函数有关，也是在处理标签是0还是1的时候有类似的思路
#补充说明
Tensorflow has two potential formats ：
==对于SparseCategorialCrossentropy和CategoricalCrossEntropy区别==
就是第一个是直接使用整数代表对象是啥，但是要求本身是10个分类（0-9），直接输出对应的位置就是类别。
对于：CategoricalCrossEntropy
即目标索引位置的取值为 1，其余 N-1 个位置的取值均为 0。以存在 10 个潜在目标值的样本为例，若目标值为 2，则其编码形式为 `[0,0,1,0,0,0,0,0,0,0]`。表示使用位置进行输出的结果
这两个方式的比较，在于最后输出的pridect的不同：(一个是看索引，一个是直接表示对应的类)

| **损失函数**        | **模型输出 (y_pred)** | **真实标签 (y_true)** | **计算时的逻辑**                       |
| --------------- | ----------------- | ----------------- | -------------------------------- |
| **Categorical** | `[0.1, 0.1, 0.8]` | `[0, 0, 1]`       | 两个向量**点对点**做乘法求和。                |
| **Sparse**      | `[0.1, 0.1, 0.8]` | `2`               | 像查字典一样，直接去 y_pred 找**索引为 2** 的值。 |

### 解释神经函数内部的逻辑
[多元分类的神经网络实现](D:\CS\CS_CodeSpace\ml-study\lab_jupyter_notebook\Advanced_Learning_Algorithm\lab2\Files\home\jovyan\work\C2_W2_Multiclass_TF.ipynb)
![[image-13.png]]
使用relu函数，就是把输入的值进行分类实现种类的区分，也就是判断其在某一个神经元内部的是输出为0还是大于0的值，从而画出某在一个内部的边界
然后在最后一个神经元位置，使用相对值的方式显示为改层神经元的分类结果是什么
![[image-14.png]]

# 模型训练相关的算法
## 使用adam学习率
使用自动调节学习率的方式，使得学习率动态变化，加快训练的速度，在学习率过大导致成本函数震荡的时候时候使用较小的学习率 使得其获得最好的model。
```python
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate = 1.*e-3),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
```
## 卷积层
将输入的内容使用做"特征工程类似的行为" -- 在神经元输入的时候，每一个unit不是接受所有的X而是接受一部分作为输入，实现减小计算量

## 反向传播back prop
![[image-15.png]]
以上是应该简单的例子，我们使用倒过来的方式去找导数，也就是利用链式法则从右向左的形式来进行求解$\frac{\partial J}{\partial w} = \frac{\partial c}{\partial w} \cdot \frac{\partial a}{\partial c} \cdot \frac{\partial d}{\partial a} \cdot……\cdot \frac{\partial J}{\partial d}$ ，注意导数求解[[sympy导数的计算]]这样的好处是减少了计算的时间和跨度，也就是先从结果出发。因为只需要一次前向传播一次反向传播可以知道所有的参数，但是使用使用正向传播需要每一次都重新计算至某一个梯度（获得权重或其偏导）
[反向传播算法的求导理论实现](D:\CS\CS_CodeSpace\ml-study\lab_jupyter_notebook\Advanced_Learning_Algorithm\optional_lab2\Files\home\jovyan\work\C2_W2_Backprop.ipynb)
![[image-16.png]]