import warnings
import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform


class SSDBSSCN(DBSCAN):
    def fit(self, X, L=None):
        X = self._validate_data(X, accept_sparse="csr")

        if self.metric == "precomputed" and sparse.issparse(X):
            X = X.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())

        distance_matrix = squareform(pdist(X))
        cDist = np.sort(distance_matrix, axis=1)[:, 3]
        cDist_expanded_i = np.expand_dims(cDist, axis=1)
        cDist_expanded_j = np.expand_dims(cDist, axis=0)
        rDist = np.maximum(np.maximum(cDist_expanded_i, cDist_expanded_j), distance_matrix)

        ssdbscan_inner(rDist, L)
        self.labels_ = ssdbscan_inner(rDist, L)


def ssdbscan_inner(rDist, L):
    visited = [False] * len(L)
    for i in range(L.shape[0]):
        if L[i] == -1 or visited[i]:
            continue
        cluster = [i]
        distances = []
        
        while True:
            row_mask, col_mask = np.zeros(L.shape[0], dtype=bool), np.ones(L.shape[0], dtype=bool)
            row_mask[cluster], col_mask[cluster] = True, False
            row_indices, col_indices = np.where(np.outer(row_mask, col_mask))
            l = [(r, c, rDist[r][c]) for r, c in zip(row_indices, col_indices) if not visited[c]]
            l = sorted(l, key=lambda x: x[2])

            if not l:
                for j in cluster:
                    L[j] = L[i]
                    visited[j] = True
                break

            v, d = l[0][1], l[0][2]
            cluster.append(v)
            distances.append(d) 
            if L[v] != -1 and L[v] != L[i]:
                pos = np.argmax(distances)
                cluster = cluster[:pos+1]
                for j in cluster:
                    L[j] = L[i]
                    visited[j] = True
                break
    return L
