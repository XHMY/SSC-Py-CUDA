from ssc.utility import data_projection, build_adjacency, spectral_clustering, thrC

try:
    import cupy as cp
    from ssc.admm_gpu import solve
except ImportError:
    print("Warning: CuPy is not installed. Falling back to NumPy.")
    from ssc.admm import solve


def sparse_subspace_clustering(Y, r, n_clusters, affine=True, rho=300, alpha_e=800, alpha_z=800,
                               threshold_c=1, enable_tqdm=True, max_iter=10000, lambda_e=None, lambda_z=None):
    """
    Sparse Subspace Clustering (SSC) algorithm.

    Parameters:
    Y (numpy.ndarray): A D x N data matrix.
    r (int): Dimension of the PCA projection. If r = 0, no projection is performed.
    affine (bool): Use the affine constraint if True.
    alpha_e alpha_z (float): The parameter alpha for the ADMM algorithm to compute lambda_e and lambda_z.
    n_clusters (int): The number of clusters for spectral clustering. If None, it will be determined automatically.

    Returns:
    numpy.ndarray: An N-dimensional vector containing the cluster memberships.
    """
    # Step 1: Data Projection
    Yp = data_projection(Y, r)

    # Step 2: ADMM Lasso
    C, converge_i = solve(Yp, affine=affine, alpha_e=alpha_e, alpha_z=alpha_z,
                          max_iter=max_iter, rho=rho, enable_tqdm=enable_tqdm, lambda_e=lambda_e, lambda_z=lambda_z)

    # Step 3: Build Adjacency Matrix
    CKSym = build_adjacency(thrC(C, threshold_c))

    # Step 4: Spectral Clustering
    labels = spectral_clustering(CKSym, n_clusters)

    return labels, C, converge_i
