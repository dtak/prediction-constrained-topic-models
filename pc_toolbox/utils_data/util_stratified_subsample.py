import numpy as np

from pc_toolbox.utils_io import pprint

def get_stratified_subsample_ids(
        y_DC=None,
        n_subsamples=1000,
        min_per_label=5,
        seed=42,
        verbose=False):
    ''' Get row ids of examples to keep in subsample for initializing weights

    Returns
    -------
    doc_ids : 1D array of ids

    Examples
    --------
    >>> y_DC = np.zeros((1000, 3))
    >>> y_DC[200:205, 0] = 1
    >>> y_DC[400:405, 1] = 1
    >>> y_DC[:995, 2] = 1
    >>> mask = get_stratified_subsample_ids(y_DC, 10, min_per_label=5)
    >>> mask.tolist()
    [200, 201, 202, 203, 204, 400, 401, 402, 403, 404, 995, 996, 997, 998, 999]
    >>> np.sum(y_DC[mask] == 0, axis=0).tolist()
    [10, 10, 10]
    >>> np.sum(y_DC[mask] == 1, axis=0).tolist()
    [5, 5, 5]
    '''
    n_labels = y_DC.shape[1]
    n_examples = y_DC.shape[0]
    if n_subsamples >= n_examples:
        return np.arange(n_examples)
    # If here, we actually need to subsample

    # Make version of y_DC where 1 is the minority class in EVERY column
    sums_total = np.sum(y_DC, axis=0)
    need_flip = sums_total / n_examples > 0.5
    y_DC[:, need_flip] = 1.0 - y_DC[:, need_flip]
    sums_total[need_flip] = n_examples - sums_total[need_flip]


    keep_mask = np.zeros(y_DC.shape[0], dtype=np.bool)
    sums_subsample = np.sum(y_DC[keep_mask], axis=0)
    for c in xrange(n_labels):
        if sums_subsample[c] < min_per_label \
                and sums_subsample[c] < sums_total[c]:
            n_more = np.minimum(min_per_label, sums_total[c])
            on_ids = np.flatnonzero(y_DC[:, c])[:min_per_label]
            keep_mask[on_ids] = True
    size = np.sum(keep_mask)
    if size < n_subsamples:
        prng = np.random.RandomState(seed)
        eligible_ids = np.flatnonzero(keep_mask == 0)
        chosen_ids = prng.choice(
            eligible_ids, n_subsamples - size, replace=False)
        keep_mask[chosen_ids] = 1
        size = np.sum(keep_mask)
    assert size >= n_subsamples
    sums_subsample = np.sum(y_DC[keep_mask], axis=0)
    if verbose:
        pprint('Minority examples per label in dataset of size %d' % n_examples)
        pprint(' '.join(['%4d' % val for val in sums_total]))
        pprint('Minority examples per label in subsample of size %d:' % size)
        pprint(' '.join(['%4d' % val for val in sums_subsample]))
    return np.flatnonzero(keep_mask)
