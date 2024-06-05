import numpy as np


# Implement Nesterov's method for solving LASSO regression
def nesterov_accelerated_gradient(y, X, lambda_, lr=0.01, max_iter=1000, tol=1e-6):
    def soft_thresholding(x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)

    n, p = X.shape
    beta = np.zeros(p)
    beta_prev = np.zeros(p)
    t = 1
    t_prev = 1
    loss_history = []

    for iteration in range(max_iter):
        y_tilde = beta + ((t_prev - 1) / t) * (beta - beta_prev)
        gradient = -X.T.dot(y - X.dot(y_tilde)) / n
        beta_prev = beta.copy()
        beta = soft_thresholding(y_tilde - lr * gradient, lr * lambda_)
        t_prev = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        loss = np.linalg.norm(y - X.dot(beta)) ** 2 / (
            2 * n
        ) + lambda_ * np.linalg.norm(beta, ord=1)
        loss_history.append(loss)
        if np.linalg.norm(beta - beta_prev, ord=2) < tol:
            break

    return beta, loss_history
