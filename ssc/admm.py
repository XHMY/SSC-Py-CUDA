import numpy as np

from tqdm import tqdm


# from joblib import Parallel, delayed
# def compute_lambda(Y, alpha_e, alpha_z, n_jobs=-1):
#     def compute_mu_e(i):
#         return np.max([np.linalg.norm(Y[:, j], ord=1) for j in range(Y.shape[1]) if i != j])
#
#     def compute_mu_z(i):
#         return np.max([np.abs(Y[:, i].T @ Y[:, j]) for j in range(Y.shape[1]) if i != j])
#
#     mu_e_values = Parallel(n_jobs=n_jobs)(delayed(compute_mu_e)(i) for i in range(Y.shape[1]))
#     mu_z_values = Parallel(n_jobs=n_jobs)(delayed(compute_mu_z)(i) for i in range(Y.shape[1]))
#
#     mu_e = np.min(mu_e_values)
#     mu_z = np.min(mu_z_values)
#
#     lambda_e = alpha_e / mu_e
#     lambda_z = alpha_z / mu_z
#
#     return lambda_e, lambda_z


def compute_lambda(Y, alpha_e, alpha_z):
    # Pre-compute norms
    norms = np.linalg.norm(Y, ord=1, axis=0)

    # Pre-compute dot products
    dot_products = np.abs(Y.T @ Y)
    np.fill_diagonal(dot_products, 0)

    # Compute mu_e by iterating over each column and taking the maximum of norms, excluding the current column
    mu_e = np.min([np.max([norms[j] for j in range(Y.shape[1]) if i != j]) for i in range(Y.shape[1])])

    # Compute mu_z using vectorized operations
    mu_z = np.min(np.max(dot_products, axis=1))

    lambda_e = alpha_e / mu_e
    lambda_z = alpha_z / mu_z

    print("lambda_e:", lambda_e, "lambda_z:", lambda_z)

    return lambda_e, lambda_z


def solve(Y, alpha_e=20, alpha_z=1, affine=True, rho=1, max_iter=1e+4,
          eps=1e-4, enable_tqdm=True, lambda_e=None, lambda_z=None):

    if lambda_e is None or lambda_z is None:
        lambda_e, lambda_z = compute_lambda(Y, alpha_e, alpha_z)

    d, n = Y.shape
    one = np.ones((n, 1))
    C = np.zeros((n, n))
    A = np.zeros((n, n))
    E = np.zeros((d, n))
    if affine:
        Delta1 = np.zeros((n, 1))  # Lagrange multiplier for affine constraint
    Delta2 = np.zeros((n, n))

    tao = lambda v, eta: np.sign(v) * np.maximum(np.abs(v) - eta, 0)

    if enable_tqdm:
        pbar = tqdm(range(int(max_iter)), desc="ADMM")
    else:
        pbar = range(int(max_iter))
    for i in pbar:
        # update A
        if affine:
            A = np.linalg.solve(lambda_z * Y.T @ Y + rho * np.eye(n) + rho * one @ one.T,
                                lambda_z * Y.T @ (Y - E) + rho * (one @ one.T + C) - one @ Delta1.T - Delta2)
        else:
            A = np.linalg.solve(lambda_z * Y.T @ Y + rho * np.eye(n),
                                lambda_z * Y.T @ (Y - E) + rho * C - Delta2)

        # update C
        J = tao(A + Delta2 / rho, 1 / rho)
        C = J - np.diag(np.diag(J))

        # update E
        E = tao(Y - Y @ A, lambda_e / lambda_z)

        if affine:
            # update Delta1
            Delta1 = Delta1 + rho * (A.T @ one - one)

        # update Delta2
        Delta2 = Delta2 + rho * (A - C)

        # check convergence
        convergence = np.linalg.norm(A - C, ord=np.inf)
        if convergence < eps:
            break

        if enable_tqdm:
            pbar.set_postfix_str(f"'A - C' = {convergence}")

    return C, i


if __name__ == '__main__':
    Y = np.random.randn(100, 100)
    C, converge_i = solve(Y, 1, 1, affine=False)
    print(C)
