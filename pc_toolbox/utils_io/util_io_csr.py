import numpy as np
import scipy.sparse
import os

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

def load_csr_matrix_from_ldac_txtfile(
        filepath=None,
        shape=None,
        n_vocabs=None,
        index_dtype=np.int32,
        data_dtype=np.float64,
        ):
    ''' Creates csr_matrix from a .ldac formatted plain-text file.

    Returns
    -------
    x_DV : scipy.sparse.csr_matrix
    '''
    assert n_vocabs is not None or shape is not None
    # Estimate num tokens in the file
    fileSize_bytes = os.path.getsize(filepath)
    nTokensPerByte = 1.0 / 5
    estimate_nUniqueTokens = int(nTokensPerByte * fileSize_bytes)

    # Preallocate space
    word_id_U = np.zeros(estimate_nUniqueTokens, dtype=index_dtype)
    word_ct_U = np.zeros(estimate_nUniqueTokens, dtype=data_dtype)
    nSeen = 0
    doc_sizes = []
    with open(filepath, 'r') as f:
        # Simple case: read the whole file
        for line in f.readlines():
            nUnique_d = -1
            while nUnique_d < 0:
                try:
                    nUnique_d = process_ldac_line_into_preallocated_arrays(
                        line, word_id, word_ct, nSeen)
                    assert nUnique_d >= 0
                except IndexError as e:
                    # Preallocated arrays not large enough
                    # Double our preallocation, then try again
                    extra_word_id = np.zeros(word_id.size, dtype=word_id.dtype)
                    extra_word_ct = np.zeros(word_ct.size, dtype=word_ct.dtype)
                    word_id = np.hstack([word_id, extra_word_id])
                    word_ct = np.hstack([word_ct, extra_word_ct])

            doc_sizes.append(nUnique_d)
            nSeen += nUnique_d
    word_id = word_id[:nSeen]
    word_ct = word_ct[:nSeen]
    n_docs = len(doc_sizes)
    doc_range = np.hstack([0, np.cumsum(doc_sizes)], dtype=index_dtype)

    if shape is None:
        assert n_vocabs is not None
        shape = (n_docs, n_vocabs)
    x_csr_DV = scipy.csr_matrix(
        (word_ct, word_id, doc_range),
        shape=shape)
    return x_csr_DV

def process_ldac_line_into_preallocated_arrays(line, word_id, word_ct, start):
    """

    Returns
    -------

    Examples
    --------
    >>> word_id = np.zeros(5, dtype=np.int32)
    >>> word_ct = np.zeros(5, dtype=np.float64)
    >>> a = process_ldac_line_into_preallocated_arrays(
    ...    '5 66:6 77:7 88:8',
    ...    word_id, word_ct, 0)
    >>> a
    5
    >>> word_id.tolist()
    [66, 77, 88, 0, 0]
    >>> word_ct.tolist()
    [ 6.,7., 8., 0., 0.]
    """
    line = line.replace(':', ' ')
    data = np.fromstring(line, sep=' ', dtype=np.int32)
    stop = start + (len(data) - 1) // 2
    if stop >= word_id.size:
        raise IndexError("Provided array not large enough")    
    word_id[start:stop] = data[1::2]
    word_ct[start:stop] = data[2::2]
    return data[0]
