import cupy as cp
import numpy as np
import sys
sys.path.append("..")
from ssc.admm import compute_lambda
from tqdm import tqdm


def solve(Y, alpha_e=20, alpha_z=1, affine=True, rho=1, max_iter=1e+4,
          eps=1e-4, enable_tqdm=True, lambda_e=None, lambda_z=None):
    dtype = cp.float64

    if lambda_e is None or lambda_z is None:
        lambda_e, lambda_z = compute_lambda(Y, alpha_e, alpha_z)

    Y = cp.array(Y)
    lambda_e, lambda_z = cp.array(lambda_e, dtype=dtype), cp.array(lambda_z, dtype=dtype)

    d, n = Y.shape
    one = cp.ones((n, 1), dtype=dtype)
    C = cp.zeros((n, n), dtype=dtype)
    A = cp.zeros((n, n), dtype=dtype)
    E = cp.zeros((d, n), dtype=dtype)
    if affine:
        Delta1 = cp.zeros((n, 1), dtype=dtype)
    Delta2 = cp.zeros((n, n), dtype=dtype)

    tao = lambda v, eta: cp.sign(v) * cp.maximum(cp.abs(v) - eta, 0)

    if enable_tqdm:
        pbar = tqdm(range(int(max_iter)), desc="ADMM")
    else:
        pbar = range(int(max_iter))

    for i in pbar:
        # update A
        if affine:
            A = cp.linalg.solve(lambda_z * Y.T @ Y + rho * cp.eye(n, dtype=dtype) + rho * one @ one.T,
                                lambda_z * Y.T @ (Y - E) + rho * (one @ one.T + C) - one @ Delta1.T - Delta2)
        else:
            A = cp.linalg.solve(lambda_z * Y.T @ Y + rho * cp.eye(n, dtype=dtype),
                                lambda_z * Y.T @ (Y - E) + rho * C - Delta2)

        # update C
        J = tao(A + Delta2 / rho, 1 / rho)
        C = J - cp.diag(cp.diag(J))

        # update E
        E = tao(Y - Y @ A, lambda_e / lambda_z)

        if affine:
            # update Delta1
            Delta1 = Delta1 + rho * (A.T @ one - one)

        # update Delta2
        Delta2 = Delta2 + rho * (A - C)

        # check convergence
        convergence = cp.linalg.norm(A - C, ord=cp.inf)
        if convergence < eps:
            break
        
        if enable_tqdm:
            pbar.set_postfix_str(f"A - C = {convergence:.5f}")

    return cp.asnumpy(C), i


if __name__ == '__main__':
    Y = np.random.randn(100, 100)
    C, converge_i = solve(Y, 1, 1)
    print(C)
