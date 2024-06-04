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
