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


def primal_gradient_descent(
    X, y, lambda_, learning_rate=0.01, max_iterations=1000, tol=1e-6
):
    """
    Primal gradient descent for LASSO linear regression.
    """
    n, p = X.shape
    beta = np.zeros(p)  # Initialize coefficients
    loss_history = []

    for _ in range(max_iterations):
        gradient = lasso_gradient(X, y, beta, lambda_)
        beta -= learning_rate * gradient

        # Compute current loss
        loss = lasso_objective(X, y, beta, lambda_)
        loss_history.append(loss)

        # Check convergence
        if np.linalg.norm(gradient, ord=np.inf) < tol:
            break

    return beta, loss_history
