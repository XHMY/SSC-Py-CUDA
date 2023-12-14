import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


def data_projection(X, r):
    """
    Project the data into a lower-dimensional space using PCA.
    
    Parameters:
    X (numpy.ndarray): A D x N data matrix, where each column is a data point.
    r (int): The dimension of the PCA projection. If r = 0, no projection is performed.
    
    Returns:
    numpy.ndarray: An r x N matrix of the projected data points.
    """
    if r == 0:
        # No projection
        return X
    else:
        # Perform PCA to reduce dimensions
        pca = PCA(n_components=r)
        Xp = pca.fit_transform(X.T).T  # Transpose to align with Matlab's convention
        return Xp


def build_adjacency(CMat, K=0):
    """
    Build an adjacency matrix from a coefficient matrix.

    Parameters:
    CMat (numpy.ndarray): An N x N coefficient matrix.
    K (int): Number of strongest edges to keep. If K=0, use all existing edges.

    Returns:
    numpy.ndarray: An N x N symmetric adjacency matrix.
    """
    N = CMat.shape[0]
    CKSym = np.abs(CMat) + np.abs(CMat).T

    # If K is not 0, keep only K strongest connections
    if K > 0:
        for i in range(N):
            sort_idx = np.argsort(CKSym[i, :])[::-1]
            CKSym[i, sort_idx[K:]] = 0

    return CKSym


# Note: This function creates a symmetric adjacency matrix by summing the absolute values of CMat and its transpose. If K is specified, it retains only the K strongest connections for each node.


def spectral_clustering(CMat, n_clusters):
    """
    Perform spectral clustering.

    Parameters:
    CMat (numpy.ndarray): An N x N adjacency matrix.
    n_clusters (int): The number of clusters to form.

    Returns:
    numpy.ndarray: An N-dimensional vector containing the cluster memberships.
    """
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize')
    labels = clustering.fit_predict(CMat)

    return labels


# Note: This function uses scikit-learn's SpectralClustering class. The 'affinity' parameter is set to 'precomputed' because we are providing an adjacency matrix directly. The 'assign_labels' parameter is set to 'discretize' for a discrete clustering assignment.

def clustering_accuracy(s, pred):
    # Creating a confusion matrix
    K = max(s.max(), pred.max()) + 1  # number of clusters in true labels and predictions
    confusion_matrix = np.zeros((K, K), dtype=int)

    for i in range(s.size):
        confusion_matrix[s[i], pred[i]] += 1

    # Applying Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Calculating accuracy
    accuracy = confusion_matrix[row_ind, col_ind].sum() / s.size
    return accuracy


def thrC(C, ro=1):
    """
    Thresholds the matrix C.

    Parameters:
    C (numpy.ndarray): A matrix to be thresholded.
    ro (float): A threshold ratio. Default is 1.

    Returns:
    numpy.ndarray: The thresholded matrix.
    """
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros_like(C)
        for i in range(N):
            # Sort the absolute values in descending order
            S = np.sort(np.abs(C[:, i]))[::-1]
            Ind = np.argsort(np.abs(C[:, i]))[::-1]

            cL1 = np.sum(S)
            cSum = 0
            t = 0
            stop = False

            while not stop:
                cSum += S[t]
                if cSum >= ro * cL1:
                    stop = True
                    selected_indices = Ind[:t + 1]
                    Cp[selected_indices, i] = C[selected_indices, i]
                t += 1
        return Cp
    else:
        return C
