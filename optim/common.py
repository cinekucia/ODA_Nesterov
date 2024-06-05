import numpy as np


def lasso_objective(X, y, beta, lambda_):
    """
    Compute the LASSO objective function.
    """
    n = len(y)
    residual = y - np.dot(X, beta)
    loss = 0.5 * np.dot(residual, residual) / n
    penalty = lambda_ * np.linalg.norm(beta, ord=1)
    return loss + penalty


def lasso_gradient(X, y, beta, lambda_):
    """
    Compute the gradient of the LASSO objective function.
    """
    n = len(y)
    residual = y - np.dot(X, beta)
    gradient_loss = -np.dot(X.T, residual) / n
    gradient_penalty = lambda_ * np.sign(beta)
    return gradient_loss + gradient_penalty
