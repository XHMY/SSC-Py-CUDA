import os
import time
from argparse import ArgumentParser
import numpy as np
import scipy.io
from tqdm import tqdm
import json
from statistics import mean, median
from timeit import default_timer as timer

from ssc.ssc import sparse_subspace_clustering
from ssc.utility import clustering_accuracy


def run_ssc_faces(mat_file_path, config, enable_tqdm=True, max_iter=10000):
    """
    Load the YaleB face dataset from a .mat file and run the SSC algorithm.

    Parameters:
    mat_file_path (str): Path to the .mat file containing the YaleB face dataset.
    r (int): Dimension of the PCA projection. If r = 0, no projection is performed.
    affine (bool): Use the affine constraint if True.
    alpha (float): The parameter alpha for the ADMM algorithm.

    Returns:
    numpy.ndarray: An N-dimensional vector containing the cluster memberships.
    """
    # Load data
    data = scipy.io.loadmat(mat_file_path)
    Y = data['Y']
    Ind = data['Ind']
    s = data['s']

    cluster_err = dict()
    summary = []
    nSet = [2, 3, 5, 8, 10]

    for n in nSet:
        idx = Ind[0][n - 1]  # Adjusted for Python indexing
        cluster_err[n] = []
        converge_iters = []
        pbar = tqdm(range(idx.shape[0])[:10], desc=f"{n} Set Evaluation")
        start = timer()
        for j in pbar:
            X = np.hstack([Y[:, :, int(idx[j, p] - 1)] for p in range(n)])  # Adjust for Python indexing
            pred, C, converge_i = sparse_subspace_clustering(X, config["r"], n_clusters=s[0][n - 1].max(),
                                                          affine=config["affine"], rho=config["rho"],
                                                          alpha_e=config["alpha_e"], alpha_z=config["alpha_z"],
                                                          enable_tqdm=enable_tqdm, max_iter=max_iter)
            acc = clustering_accuracy(s[0][n - 1], pred)
            converge_iters.append(converge_i)
            cluster_err[n].append(1 - acc)
            pbar.set_postfix_str(f"acc = {acc * 100:.4f} %, iter = {converge_i}")

        end = timer()
        summary.append({"nSet:": n, "time (s):": (end - start) / idx.shape[0],
                        "mean_iterations:": mean(converge_iters), "median_iterations:": median(converge_iters),
                        "min:": min(cluster_err[n]), "max:": max(cluster_err[n]),
                        "mean:": mean(cluster_err[n]), "median:": median(cluster_err[n])})
        print(summary[-1])

    return summary


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--rho', type=int, default=300)
    args.add_argument('--alpha_e', type=int, default=20)
    args.add_argument('--alpha_z', type=int, default=1)
    args.add_argument('--affine', action='store_true')
    args.add_argument('--r', type=int, default=0)
    args.add_argument('--tqdm', action='store_true')
    args.add_argument('--max_iter', type=int, default=10000)
    args = args.parse_args()

    config = {
        "rho": args.rho,
        "alpha_e": args.alpha_e,
        "alpha_z": args.alpha_z,
        "affine": args.affine,
        "r": args.r
    }
    print(args)
    summary = run_ssc_faces('data/YaleBCrop025.mat',
                            config=config, enable_tqdm=args.tqdm, max_iter=args.max_iter)
    os.makedirs('logs/face', exist_ok=True)
    with open(f'logs/face/summary-{int(time.time()) }.json', 'w') as fd:
        json.dump({"config": config, "data": summary}, fd, indent=4)
