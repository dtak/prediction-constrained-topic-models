'''
Differentiable transform for 2D array whose rows sum to one.

Need to_common_arr to have no edge cases ...
    always deliver something that sums to 1.0
    with minimum value min_eps

Need to_diffable_arr to be robust
    always cast its input to the proper domain
    and then take logs, etc.

'''

import autograd.numpy as np
from autograd.scipy.misc import logsumexp

MIN_EPS = 1e-11

def to_common_arr(
        log_topics_KVm1,
        min_eps=MIN_EPS,
        **kwargs):
    ''' Convert unconstrained topic weights to proper normalized topics

    Should handle any non-nan, non-inf input without numerical problems.

    Args
    ----
    log_topics_KVm1 : 2D array, size K x V-1

    Returns
    -------
    topics_KV : 2D array, size K x V
        minimum value of any entry will be min_eps
        each row will sum to 1.0 (+/- min_eps)
    '''
    K, Vm1 = log_topics_KVm1.shape
    V = Vm1 + 1
    log_topics_KV = np.hstack([
        log_topics_KVm1,
        np.zeros((K, 1))])
    log_topics_KV -= logsumexp(log_topics_KV, axis=1, keepdims=1)
    log_topics_KV += np.log1p(-V * min_eps)
    topics_KV = np.exp(log_topics_KV)

    return min_eps + topics_KV

def to_diffable_arr(topics_KV, min_eps=MIN_EPS, do_force_safe=False):
    ''' Transform normalized topics to unconstrained space.

    Args
    ----
    topics_KV : 2D array, size K x V
        minimum value of any entry must be min_eps
        each row should sum to 1.0

    Returns
    -------
    log_topics_vec : 2D array, size K x (V-1)
        unconstrained real values

    Examples
    --------
    >>> topics_KV = np.eye(3) + np.ones((3,3))
    >>> topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]
    >>> log_topics_vec = to_diffable_arr(topics_KV)
    >>> out_KV = to_common_arr(log_topics_vec)
    >>> np.allclose(out_KV, topics_KV)
    True
    '''
    if do_force_safe:
        topics_KV = to_safe_common_arr(topics_KV, min_eps)
    K, V = topics_KV.shape
    log_topics_KV = np.log(topics_KV)
    log_topics_KVm1 = log_topics_KV[:, :-1]
    log_topics_KVm1 = log_topics_KVm1 - log_topics_KV[:, -1][:,np.newaxis]
    return log_topics_KVm1 + np.log1p(-V * min_eps)

def to_safe_common_arr(topics_KV, min_eps=MIN_EPS):
    ''' Force provided topics_KV array to be numerically safe.

    Returns
    -------
    topics_KV : 2D array, size K x V
        minimum value of each row is min_eps
        each row will sum to 1.0 (+/- min_eps)
    '''
    K, V = topics_KV.shape
    topics_KV = topics_KV.copy()
    for rep in range(2):
        topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]
        np.maximum(topics_KV, min_eps, out=topics_KV)
    return topics_KV


if __name__ == '__main__':
    topics_KV = np.eye(3) + np.ones((3,3))
    topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]

    print('------ before')
    print(topics_KV)
    print('------ after')
    print(to_common_arr(to_diffable_arr(topics_KV)))