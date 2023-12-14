import json
import os
import time
from argparse import ArgumentParser
import numpy as np
import scipy.io
from timeit import default_timer as timer
from statistics import mean, median

from tqdm import tqdm

from ssc.ssc import sparse_subspace_clustering
from ssc.utility import clustering_accuracy
from glob import glob


def run_ssc_motion_segmentation(folder_path, config, enable_tqdm=True):
    """
    Run SSC algorithm on the Hopkins 155 motion segmentation dataset.

    Parameters:
    folder_path (str): Path to the folder containing the Hopkins 155 sequences.
    alpha (float): The parameter alpha for the SSC algorithm.

    Returns:
    dict: A dictionary containing average and median miss rates.
    """

    summary = []

    for dir_path in tqdm(glob(os.path.join(folder_path, "*")), desc="Hopkins155"):
        if not os.path.isdir(dir_path):
            continue

        err_1, err_4 = [], []
        converge_iters_1, converge_iters_4 = [], []

        start = timer()

        data = scipy.io.loadmat(glob(os.path.join(dir_path, "*truth.mat"))[0])
        s = data['s'].squeeze()
        x = data['x']

        n = s.max()
        N, F = x.shape[1], x.shape[2]
        D = 2 * F
        X = np.transpose(x[0:2, :, :], (0, 2, 1)).reshape(D, N, order='F')

        # Call to a Python SSC function
        pred_1, C, converge_1 = sparse_subspace_clustering(X, r=0, n_clusters=n,
                                                        affine=config["affine"], rho=config["rho"],
                                                        alpha_e=config["alpha_e"], alpha_z=config["alpha_z"],
                                                        enable_tqdm=enable_tqdm, threshold_c=0.7)
        err_1.append(1 - clustering_accuracy(s, pred_1))
        converge_iters_1.append(converge_1)
        pred_4, C, converge_4 = sparse_subspace_clustering(X, r=4 * n, n_clusters=n,
                                                        affine=config["affine"], rho=config["rho"],
                                                        alpha_e=config["alpha_e"], alpha_z=config["alpha_z"],
                                                        enable_tqdm=enable_tqdm, threshold_c=0.7)
        err_4.append(1 - clustering_accuracy(s, pred_4))
        converge_iters_4.append(converge_4)

        end = timer()
        summary.append({"file:": dir_path, "time (s):": (end - start), "x shape": str(x.shape),
                        "mean_iterations_1:": mean(converge_iters_1), "mean_iterations_4:": mean(converge_iters_4),
                        "mean_err_1:": mean(err_1), "mean_err_4:": mean(err_4)})
        print(summary[-1])



    return summary


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--rho', type=int, default=100)
    args.add_argument('--alpha_e', type=int, default=0)
    args.add_argument('--alpha_z', type=int, default=800)
    args.add_argument('--affine', action='store_true')
    args.add_argument('--tqdm', action='store_true')
    args = args.parse_args()

    config = {
        "rho": args.rho,
        "alpha_e": args.alpha_e,
        "alpha_z": args.alpha_z,
        "affine": args.affine,
    }
    print(args)
    summary = run_ssc_motion_segmentation('data/Hopkins155',
                                          config=config, enable_tqdm=args.tqdm)
    os.makedirs('logs/ms', exist_ok=True)
    with open(f'logs/ms/summary-{int(time.time())}.json', 'w') as fd:
        json.dump({"config": config, "data": summary}, fd, indent=4)
