[导数的计算](D:\CS\CS_CodeSpace\ml-study\lab_jupyter_notebook\Advanced_Learning_Algorithm\optional_lab2\Files\home\jovyan\work\C2_W2_Derivatives.ipynb)
首先将求导的变量转换为特殊的符号变量，而不是python内置的普通变量，也就是可以先使用后赋值
```python
from sympy import symbols, diff

J,w = symbol('J,w')
```
先对$J = f(w) 定义表达式
然后使用求导使用`diff()`函数用于求表达式
使用`.subs()`方法带入 具体的变量得到导数的结果
```python
J = w**3

dj_dw = diff(J,w)
dj_dw
#输出：$3w^2$
dj_dw.subs([(w,2)])  #其实就是将表达式里的数值带入
#输出：12
```
较为复杂的案例实现：[反向传播](D:\CS\CS_CodeSpace\ml-study\lab_jupyter_notebook\Advanced_Learning_Algorithm\optional_lab2\Files\home\jovyan\work\C2_W2_Backprop.ipynb)