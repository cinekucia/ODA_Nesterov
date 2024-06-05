import numpy as np


def lasso_objective(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray, lambda_: float
) -> float:
    """Computes the objective function for LASSO regression.

    Args:
        X (array_like): The feature matrix of shape (n_samples, n_features).
        y (array_like): The target values of shape (n_samples,).
        beta (array_like): The coefficient vector of shape (n_features,).
        lambda_ (float): The regularization parameter.

    Returns:
        float: The value of the objective function for LASSO regression.
    """
    n = len(y)
    residual = y - np.dot(X, beta)
    loss = 0.5 * np.dot(residual, residual) / n
    penalty = lambda_ * np.linalg.norm(beta, ord=1)
    return loss + penalty


def lasso_gradient(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Compute the gradient of the LASSO objective function.

    Args:
        X (ndarray): The feature matrix of shape (n_samples, n_features).
        y (ndarray): The target values of shape (n_samples,).
        beta (ndarray): The coefficient vector of shape (n_features,).
        lambda_ (float): The regularization parameter.

    Returns:
        ndarray: The gradient of the LASSO objective function with respect to the coefficient vector.
    """
    n = len(y)
    residual = y - np.dot(X, beta)
    gradient_loss = -np.dot(X.T, residual) / n
    gradient_penalty = lambda_ * np.sign(beta)
    return gradient_loss + gradient_penalty
