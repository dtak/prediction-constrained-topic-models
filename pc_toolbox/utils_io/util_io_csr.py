import numpy as np
import scipy.sparse

def load_csr_matrix(filename):
    Q = np.load(filename)
    return scipy.sparse.csr_matrix(
        (Q['data'], Q['indices'], Q['indptr']),
        shape=Q['shape'])

def save_csr_matrix(filename, array):
    np.savez(
        filename,
        data=array.data,
        indices=array.indices,
        indptr=array.indptr,
        shape=array.shape)
