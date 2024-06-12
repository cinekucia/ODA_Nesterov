import time
import numpy as np
# from .common import lasso_objective, lasso_gradient


def three_cases(changes, penalty, denominator):
    # Vectorized handling of three cases for the proximal gradient update
    return (np.less_equal(changes, -penalty) * (-changes - penalty) + np.greater_equal(changes, penalty) * (-changes + penalty)) / denominator


class DualGradientMethod:
    """Dual Gradient Method with Nesterov acceleration for solving Ax = b using least squares.

    Args:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        penalty (float): Regularization parameter.
        gamma_u (float): Rate at which L is adjusted upwards.
        gamma_d (float): Rate at which L is adjusted downwards.
        L_0 (float): Initial value for the Lipschitz constant.
        v_0 (np.ndarray): Initial guess for the solution.
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
    def __init__(self, A, b, penalty, gamma_u, gamma_d, L_0, v_0, max_iter=10000, tol=1e-6) -> None:
        self.A = A
        self.b = b
        self.penalty = penalty
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.L = L_0
        self.v = v_0
        self.max_iter = max_iter
        self.tol = tol
        self.gap_history = []
        self.loss_history = []
        self.cpu_time_history = []

    def gradient_f(self, x):
        """Gradient of the least squares loss function."""
        return 2 * np.dot(self.A.T, (np.dot(self.A, x) - self.b))

    def f(self, x):
        """Least squares loss function."""
        return np.linalg.norm(np.dot(self.A, x) - self.b)**2 / 2

    def psi(self, x):
        """Placeholder for potential regularizer function."""
        return 0

    def compute_steps(self):
        """Compute the Nesterov accelerated gradient steps."""
        v_prev = self.v
        initial_residual = np.dot(self.A, self.v) - self.b
        initial_loss = np.linalg.norm(initial_residual)**2 / 2

        for k in range(self.max_iter):
            start_time = time.process_time()

            # Nesterov acceleration
            if k > 0:
                y = self.v + (k - 1) / (k + 2) * (self.v - v_prev)
            else:
                y = self.v

            grad_f = self.gradient_f(y)
            Mk = np.linalg.norm(grad_f, 2)
            if Mk == 0:
                Mk = 1e-16  # Prevent division by zero

            L_k = max(self.L, Mk / self.gamma_d)
            T, L_new = self.gradient_iteration(self.penalty, self.gamma_u, y, L_k)
            self.L = L_new

            v_prev = self.v
            self.v = T

            current_residual = np.dot(self.A, self.v) - self.b
            current_loss = np.linalg.norm(current_residual)**2 / 2
            current_gap = current_loss / initial_loss

            end_time = time.process_time()

            self.gap_history.append(current_gap)
            self.loss_history.append(current_loss)
            self.cpu_time_history.append(end_time - start_time)

            if current_gap < self.tol:
                break

        return self.v, k, current_gap, self.gap_history, self.loss_history, self.cpu_time_history

    def gradient_iteration(self, penalty, gamma_u, x, M):
        """Gradient iteration with backtracking line search."""
        L = M
        while True:
            changes = self.gradient_f(x) - L * x
            T = three_cases(changes, penalty, L)
            if not self.psi(T) > np.dot(self.gradient_f(T), (x - T)) + np.linalg.norm(x - T)**2 * L / 2 + self.psi(x):
                break
            L *= gamma_u
        return T, L
