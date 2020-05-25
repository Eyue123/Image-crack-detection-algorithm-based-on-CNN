import numpy as np


def compute_pca(data):
    m = np.mean(data, axis=0)
    datac = np.array([obs - m for obs in data])
    T = np.dot(datac, datac.T)
    [u, s, v] = np.linalg.svd(T, full_matrices = False)

    # here iteration is over rows but the columns are the eigenvectors of T
    pcs = [np.dot(datac.T, item) for item in u.T ]

    # note that the eigenvectors are not normed after multiplication by T^T
    pcs = np.array([d / np.linalg.norm(d) for d in pcs])

    return pcs, s
