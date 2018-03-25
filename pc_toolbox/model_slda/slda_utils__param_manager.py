"""
Parameter management functions for sLDA

Key functions:
* flatten_to_differentiable_param_vec
* unflatten_to_common_param_dict
"""

import autograd.numpy as np
from pc_toolbox.utils_diffable_transforms import (
    tfm__2D_rows_sum_to_one,
    )

def flatten_to_differentiable_param_vec(
        param_dict=None,
        topics_KV=None,
        w_CK=None,
        **unused_kwargs):
    """ Convert common parameters of sLDA into flat vector of reals.

    Examples
    --------
    >>> K = 2; V = 3; C = 2;
    >>> topics_KV = np.asarray([[0.6, 0.3, 0.1], [0.2, 0.1, 0.7]])
    >>> w_CK = np.asarray([[4.0, -4.0], [-1.0, 1.0]])
    >>> param_vec = flatten_to_differentiable_param_vec(
    ...     topics_KV=topics_KV,
    ...     w_CK=w_CK)
    >>> param_dict = unflatten_to_common_param_dict(
    ...     param_vec=param_vec, n_states=K, n_vocabs=V, n_labels=C)
    >>> print param_dict['w_CK']
    [[ 4. -4.]
     [-1.  1.]]

    >>> print param_dict['topics_KV']
    [[0.6 0.3 0.1]
     [0.2 0.1 0.7]]
    >>> np.allclose(param_dict['topics_KV'], topics_KV)
    True
    """
    if isinstance(param_dict, dict):
        topics_KV = param_dict['topics_KV']
        w_CK = param_dict['w_CK']
    return np.hstack([
        tfm__2D_rows_sum_to_one.to_diffable_arr(topics_KV).flatten(),
        w_CK.flatten()])

def unflatten_to_common_param_dict(
        param_vec=None,
        n_states=0,
        n_vocabs=0,
        n_labels=0,
        **unused_kwargs):
    K = int(n_states)
    V = int(n_vocabs)
    C = int(n_labels)
    F_topics = K * (V-1)
    topics_KV = tfm__2D_rows_sum_to_one.to_common_arr(
        param_vec[:F_topics].reshape(K, V-1))
    w_CK = np.reshape(param_vec[F_topics:], (C, K))
    return dict(topics_KV=topics_KV, w_CK=w_CK)