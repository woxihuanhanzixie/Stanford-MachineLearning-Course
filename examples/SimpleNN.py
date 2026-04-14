import numpy as np
import matplotlib.pyplot as plt

"""
简单的神经网络实现 (Simple Neural Network Implementation)
包含前向传播、反向传播和简单的训练逻辑
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    """
    一个包含一个隐藏层的简单神经网络
    """
    def __init__(self, input_size, hidden_size, output_size):
        # 随机初始化权重
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        # 反向传播 (简单的均方误差)
        m = X.shape[0]
        
        # 输出层梯度
        error = output - y
        d_output = error * sigmoid_derivative(self.z2)
        
        # 隐藏层梯度
        d_hidden = np.dot(d_output, self.W2.T) * sigmoid_derivative(self.z1)
        
        # 更新权重和偏置
        self.W2 -= learning_rate * np.dot(self.a1.T, d_output) / m
        self.b2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True) / m
        self.W1 -= learning_rate * np.dot(X.T, d_hidden) / m
        self.b1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True) / m

    def train(self, X, y, epochs, learning_rate):
        history = []
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            loss = np.mean(np.square(y - output))
            history.append(loss)
            if i % (epochs // 10) == 0:
                print(f"Epoch {i:5d}: Loss {loss:.4f}")
        return history

# 测试数据 (XOR 问题)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(2, 4, 1)
print("Training Simple Neural Network...")
losses = nn.train(X, y, epochs=10000, learning_rate=0.5)

# 预测
predictions = nn.forward(X)
print("\nPredictions for XOR:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Pred: {predictions[i][0]:.4f}")
