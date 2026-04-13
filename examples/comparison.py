import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from Linear import compute_gradient, gradient_descent as linear_gd
from Logistic import gradient_descent as logistic_gd

"""
对比 Scikit-learn 和手动实现的算法 (Comparison: Scikit-learn vs. Manual)
"""

def compare_linear():
    print("=== Linear Regression Comparison ===")
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    # Scikit-learn
    model = LinearRegression()
    model.fit(X, y)
    print(f"Sklearn: w={model.coef_[0]:.4f}, b={model.intercept_:.4f}")
    
    # Manual
    w_init = np.array([0.])
    b_init = 0.
    w_final, b_final, _ = linear_gd(X, y, w_init, b_init, 0.01, 1000)
    print(f"Manual:  w={w_final[0]:.4f}, b={b_final:.4f}")

def compare_logistic():
    print("\n=== Logistic Regression Comparison ===")
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Scikit-learn
    model = LogisticRegression()
    model.fit(X, y)
    print(f"Sklearn: w={model.coef_[0]}, b={model.intercept_[0]:.4f}")
    
    # Manual
    w_init = np.zeros(2)
    b_init = 0.
    w_final, b_final, _ = logistic_gd(X, y, w_init, b_init, 0.1, 10000)
    print(f"Manual:  w={w_final}, b={b_final:.4f}")

if __name__ == "__main__":
    compare_linear()
    compare_logistic()
