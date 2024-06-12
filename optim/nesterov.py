import numpy as np
# from .common import lasso_objective, lasso_gradient
import time


class NesterovAcceleratedGradientMethod:
    """Nesterov Accelerated Gradient Method for LASSO regression.

    Args:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        L0 (float): Initial value for the Lipschitz constant.
        v0 (np.ndarray): Initial guess for the solution.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criterion based on relative gap.

    Returns:
        Tuple containing:
        - np.ndarray: The solution vector x.
        - int: Total number of iterations performed.
        - float: Final relative gap.
        - List[float]: History of relative gaps.
        - List[float]: History of loss values.
        - List[float]: History of CPU times.
    """
    def __init__(self, A, b, L0, v0, max_iter=10000, tol=1e-6):
        self.A = A
        self.b = b
        self.L = L0
        self.v = v0
        self.max_iter = max_iter
        self.tol = tol
        self.gap_history = []
        self.loss_history = []
        self.cpu_time_history = []

    def gradient_f(self, x):
        """Compute the gradient of the objective function at x."""
        return self.A.T @ (self.A @ x - self.b)

    def f(self, x):
        """Compute the objective function at x."""
        return np.linalg.norm(self.A @ x - self.b)**2 / 2

    def nag_step(self, y, L):
        """Compute the Nesterov accelerated gradient step at y."""
        grad = self.gradient_f(y)
        return y - grad / L

    def compute_steps(self):
        """Run the Nesterov accelerated gradient method."""
        v_prev = self.v
        initial_loss = self.f(self.v)

        for k in range(self.max_iter):
            start_time = time.process_time()

            y = self.v + k / (k + 3) * (self.v - v_prev)
            T_L_y = self.nag_step(y, self.L)

            while np.linalg.norm(self.gradient_f(T_L_y) - self.gradient_f(y)) > self.L * np.linalg.norm(T_L_y - y):
                self.L *= 1.1  # Increase L and recompute
                T_L_y = self.nag_step(y, self.L)

            v_prev = self.v
            self.v = T_L_y

            current_loss = self.f(self.v)
            current_gap = current_loss / initial_loss

            end_time = time.process_time()

            self.gap_history.append(current_gap)
            self.loss_history.append(current_loss)
            self.cpu_time_history.append(end_time - start_time)

            if current_gap < self.tol:
                break

        return self.v, k, current_gap, self.gap_history, self.loss_history, self.cpu_time_history

# HISTORY:
# def nesterov_accelerated_gradient(
#     X: np.ndarray,
#     y: np.ndarray,
#     lambda_: float = 0,
#     lr: float = 0.01,
#     max_iter: int = 1000,
#     tol: float = 1e-6,
# ) -> tuple[np.ndarray, list[float], list[float], list[float]]:
#     """Nesterov accelerated gradient method for LASSO regression.

#     Args:
#         X (ndarray): The feature matrix of shape (n_samples, n_features).
#         y (ndarray): The target values of shape (n_samples,).
#         lambda_ (float): The regularization parameter.
#         lr (float, optional): The learning rate. Defaults to 0.01.
#         max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
#         tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

#     Returns:
#         Tuple[ndarray, List[float], List[float], List[float]]: A tuple containing the optimized coefficient vector,
#         the history of loss values, the history of gaps during optimization, and the history of CPU times.
#     """

#     def soft_thresholding(x: np.ndarray, lambda_: float) -> np.ndarray:
#         """Compute the soft thresholding function.

#         Args:
#             x (ndarray): The input array.
#             lambda_ (float): The thresholding parameter.

#         Returns:
#             ndarray: The result of applying soft thresholding to the input array.
#         """
#         return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)

#     n, p = X.shape
#     beta = np.zeros(p)
#     beta_prev = np.zeros(p)
#     t = 1
#     t_prev = 1
#     loss_history = []
#     gap_history = []
#     cpu_time_history = []

#     # Compute initial residual
#     initial_residual = np.linalg.norm(np.dot(X, beta) - y)**2
#     # initial_residual = np.linalg.norm(y - X.dot(beta))**2

#     for i in range(max_iter):
#         start_time = time.process_time()
#         y_tilde = beta + ((t_prev - 1) / t) * (beta - beta_prev)
#         gradient = -X.T.dot(y - X.dot(y_tilde)) / n
#         beta_prev = beta.copy()
#         beta = soft_thresholding(y_tilde - lr * gradient, lr * lambda_)
#         t_prev = t
#         t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
#         loss = np.linalg.norm(y - X.dot(beta)) ** 2 / (2 * n) + lambda_ * np.linalg.norm(beta, ord=1)
#         loss_history.append(loss)
#         # update cpu time
#         end_time = time.process_time()
#         cpu_time_history.append(end_time - start_time)
#         # Compute residual
#         # residual = np.linalg.norm(y - X.dot(beta))**2
#         residual = np.linalg.norm(np.dot(X, beta) - y)**2
#         # Compute the gap as the relative decrease of the initial residual
#         gap = residual / initial_residual
#         gap_history.append(gap)
#         if gap < tol:
#             break
#         # END ADDED

#     return beta, loss_history, gap_history, cpu_time_history
