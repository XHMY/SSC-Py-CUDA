from argparse import ArgumentParser
import numpy as np
from timeit import default_timer as timer
import pandas as pd
try:
    from ssc.admm_gpu import solve as solve_gpu
except ImportError:
    print("Warning: CuPy is not installed. Falling back to NumPy.")
from ssc.admm import solve as solve_cpu


def bench(solve_func, max_iter=10, enable_tqdm=True):
    result_list = []
    for n in [1000, 10000, 30000]:
        Y_mat = np.random.rand(1024, n).astype(np.float64)
        timer_start = timer()
        _, iter_cnt = solve_func(Y_mat, lambda_e=0.1, lambda_z=0.1, affine=True,
                                 eps=1e-10, max_iter=max_iter, enable_tqdm=enable_tqdm)
        timer_end = timer()
        result_list.append({"n": n, "time / iter": (timer_end - timer_start) / iter_cnt})

    df = pd.DataFrame(result_list)
    return df


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--cuda', action='store_true')
    args.add_argument('--max_iter', type=int, default=10)
    args.add_argument('--enable_tqdm', action='store_true')

    args = args.parse_args()

    solve_func = solve_gpu if args.cuda else solve_cpu

    result = bench(solve_func, max_iter=args.max_iter, enable_tqdm=args.enable_tqdm)
    print(result)
