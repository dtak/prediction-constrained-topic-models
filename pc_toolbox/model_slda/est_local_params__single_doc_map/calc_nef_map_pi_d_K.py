"""
calc_nef_map_pi_d_K.py : User-friend tool for MAP estimation of pi_d_K

API
---
Makes useful functions available:
* calc_nef_map_pi_d_K(...)

Validation
----------
$ python calc_nef_map_pi_d_K.py

Runs some diagnostic tests comparing different pi_d_K estimation methods.

"""
import argparse
import numpy as np
import time
import sys
import os

from calc_nef_map_pi_d_K__defaults import DefaultDocTopicOptKwargs

## Load other modules
from calc_nef_map_pi_d_K__numpy import (
    calc_nef_map_pi_d_K__numpy)

from calc_nef_map_pi_d_K__numpy_linesearch import (
    calc_nef_map_pi_d_K__numpy_linesearch)

# Try to load cython code
# Fall back on python code
try:
    from calc_nef_map_pi_d_K__cython import calc_nef_map_pi_d_K__cython
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

try:
    from calc_nef_map_pi_d_K__cython_linesearch import (
        calc_nef_map_pi_d_K__cython_linesearch)
    HAS_CYTHON_LINESEARCH = True
except ImportError:
    HAS_CYTHON_LINESEARCH = False


def calc_nef_map_pi_d_K(
        word_id_d_Ud=None,
        word_ct_d_Ud=None,
        topics_KUd=None,
        topics_KV=None,
        alpha=None,
        nef_alpha=None,
        init_pi_d_K=None,
        method='numpy',
        max_iters=DefaultDocTopicOptKwargs['max_iters'],
        converge_thr=DefaultDocTopicOptKwargs['converge_thr'],
        pi_step_size=DefaultDocTopicOptKwargs['pi_step_size'],
        min_pi_step_size=DefaultDocTopicOptKwargs['min_pi_step_size'],
        pi_step_decay_rate=DefaultDocTopicOptKwargs['pi_step_decay_rate'],
        pi_min_mass_preserved_to_trust_step=\
            DefaultDocTopicOptKwargs['pi_min_mass_preserved_to_trust_step'],
        **kwargs):
    # Common preprocessing
    if topics_KUd is None:
        topics_KUd = topics_KV[:, word_id_d_Ud]

    # Precompute some useful things
    ct_topics_KUd = topics_KUd * word_ct_d_Ud[np.newaxis, :]
    K = topics_KUd.shape[0]

    # Parse alpha into natural EF alpha (so estimation is always convex)
    if nef_alpha is not None:
        nef_alpha = float(nef_alpha)
    elif alpha is not None:
        nef_alpha = float(alpha)
    else:
        raise ValueError("Need to define alpha or nef_alpha")
    assert isinstance(nef_alpha, float)

    # Now translate into convex_alpha_minus_1
    if nef_alpha > 1.0:
        convex_alpha_minus_1 = nef_alpha - 1.0
    else:
        convex_alpha_minus_1 = nef_alpha
    # These are unused below. Lets be sure of that.
    alpha = None     # unused below
    nef_alpha = None # unused below
    assert convex_alpha_minus_1 < 1.0
    assert convex_alpha_minus_1 >= 0.0

    # Initialize as uniform vector over K simplex
    if init_pi_d_K is None:
        init_pi_d_K = np.ones(K) / float(K)
    else:
        init_pi_d_K = np.asarray(init_pi_d_K)
    assert init_pi_d_K.size == K

    if method.count("cython_linesearch"):
        if not HAS_CYTHON_LINESEARCH:
            raise ImportError("No compiled cython function: " + method)
        calc_pi_d_K = calc_nef_map_pi_d_K__cython_linesearch
    elif method.count("cython"):
        if not HAS_CYTHON:
            raise ImportError("No compiled cython function: " + method)
        calc_pi_d_K = calc_nef_map_pi_d_K__cython
    elif method.count("numpy_linesearch"):
        calc_pi_d_K = calc_nef_map_pi_d_K__numpy_linesearch
    elif method.count("numpy"):
        calc_pi_d_K = calc_nef_map_pi_d_K__numpy
    else:
        raise ValueError("Unrecognized pi_d_K estimation method:" + method)
    pi_d_K, info = calc_pi_d_K(
        init_pi_d_K=init_pi_d_K,
        topics_KUd=topics_KUd,
        word_ct_d_Ud=np.asarray(word_ct_d_Ud, dtype=np.float64),
        ct_topics_KUd=ct_topics_KUd,
        convex_alpha_minus_1=convex_alpha_minus_1,
        max_iters=int(max_iters),
        pi_step_size=float(pi_step_size),
        min_pi_step_size=float(min_pi_step_size),
        pi_step_decay_rate=float(pi_step_decay_rate),
        pi_min_mass_preserved_to_trust_step=\
            float(pi_min_mass_preserved_to_trust_step),
        converge_thr=float(converge_thr),
        **kwargs)
    return pi_d_K, info



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--Ud', type=int, default=100)
    parser.add_argument('--nef_alpha', type=float, default=1.1)
    parser.add_argument(
        '--max_iters',
        type=int,
        default=DefaultDocTopicOptKwargs['max_iters'])
    parser.add_argument('--pi_step_size',
        type=float,
        default=DefaultDocTopicOptKwargs['pi_step_size'])
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--param_npz_path', type=str, default=None)
    args = parser.parse_args()

    lstep_kwargs = dict(**DefaultDocTopicOptKwargs)
    lstep_kwargs['max_iters'] = args.max_iters
    lstep_kwargs['pi_step_size'] = args.pi_step_size
    if args.verbose:
        lstep_kwargs['verbose'] = True
        lstep_kwargs['very_verbose'] = True

    if args.param_npz_path is not None and os.path.exists(args.param_npz_path):
        Params = dict(np.load(args.param_npz_path).items())
        nef_alpha = float(Params.get('nef_alpha', args.nef_alpha))
        word_ct_d_Ud = Params.get('word_ct_d_Ud')
        topics_KUd = Params.get('topics_KV')[:, Params.get('word_id_d_Ud')]

        K, Ud = topics_KUd.shape
    else:
        K = args.K
        Ud = args.Ud
        nef_alpha = args.nef_alpha

        prng = np.random.RandomState(12342)
        topics_KUd = prng.rand(K, Ud)
        topics_KUd /= np.sum(topics_KUd, axis=1)[:,np.newaxis]
        word_ct_d_Ud = prng.randint(low=1, high=3, size=Ud)
        word_ct_d_Ud = np.asarray(word_ct_d_Ud, dtype=np.float64)
    print "Applying K=%d topics to doc with Ud=%d uniq terms" % (K, Ud)
    print "nef_alpha = ", nef_alpha

    print "default kwargs"
    for key in sorted(lstep_kwargs):
        print "%-50s %s" % (key, lstep_kwargs[key])

    for method in [
            'cython',
            'numpy',
            #'linesearch_numpy',
            #'linesearch_cython',
            ]:
        start_time = time.time()
        pi_d_K, info_dict = calc_nef_map_pi_d_K(
            word_ct_d_Ud=word_ct_d_Ud,
            topics_KUd=topics_KUd,
            nef_alpha=nef_alpha,
            method=method,
            **lstep_kwargs)
        elapsed_time_sec = time.time() - start_time

        if pi_d_K.size > 8:
            top_ids = np.argsort(-1 * pi_d_K)[:8]
        else:
            top_ids = np.arange(K)
        print "RESULT %-20s : after %8.3f sec" % (method, elapsed_time_sec)
        print "    ", ' '.join(['%.4f' % x for x in pi_d_K[top_ids]])
        print "        n_iters      = %5d" % info_dict['n_iters']
        print "        did_converge = %d" % info_dict['did_converge']
        print "        cur_L1_diff  = %.5f" % info_dict['cur_L1_diff']
        print "        pi_step_size = %.5f" % info_dict['pi_step_size']
        print "        n_restarts   = %d" % info_dict['n_restarts']
        print "        nef_alpha    = %.3f" % (
            info_dict['convex_alpha_minus_1'] + 1.0)
