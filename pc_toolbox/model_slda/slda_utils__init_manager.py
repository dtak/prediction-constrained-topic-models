import numpy as np
import sys
import os
import time
from sklearn.externals import joblib

from slda_estimator__w_given_pi import (
    estimate_w_CK__given_pi_DK)
from slda_utils__dataset_manager import (
    load_dataset,
    )
from pc_toolbox.utils_io import (
    pprint,
    )
from pc_toolbox.model_slda.est_local_params__many_doc_map import (
    calc_nef_map_pi_DK)

def init_param_dict(
        dataset=None,
        topics_KV=None,
        w_CK=None,
        n_states=None,
        init_name=None,
        init_name_topics='rand_docs',
        init_name_w='regress_given_topics',
        init_model_path=None,
        max_n_docs=100000,
        min_n_docs_per_label=10,
        seed=0,
        alpha=1.1,
        tau=1.1,
        lambda_w=.001,
        verbose=True,
        **kwargs):
    ''' Create initial param dict for slda optimization problem.

    Returns
    -------
    init_params_dict : dict, with fields
        topics_KV : 2D array, K x V
        w_CK : 2D array, C x K
    '''
    if n_states is not None:
        n_states = int(n_states)
    n_states = int(n_states)
    lambda_w = float(lambda_w)
    tau = float(tau)
    alpha = float(alpha)

    # Parse init_name
    # For backwards compat: init_name means same thing as init_name_topics
    if init_name is not None:
        init_name_topics = init_name
    del init_name

    if str(init_model_path).lower() != 'none':
        pprint('[init_params] Loading from init_model_path ...')

        if init_model_path.count('snapshot'):
           initfromdisk_param_dict = load_topic_model_param_dict(
               snapshot_path=init_model_path)
        else:
            if init_model_path.endswith(os.path.sep):
                init_model_path = os.path.join(
                    init_model_path, 'param_dict.dump')
            initfromdisk_param_dict = joblib.load(init_model_path)
        topics_KV = initfromdisk_param_dict['topics_KV']
        if 'w_CK' in initfromdisk_param_dict:
            w_CK = initfromdisk_param_dict['w_CK']

    if topics_KV is None or topics_KV.shape[0] < n_states:
        pprint('[init_params] Running init_topics_KV %s ...' % (
            init_name_topics))
        topics_KV = init_topics_KV(
            dataset=dataset,
            topics_KV=topics_KV,
            n_states=n_states,
            seed=seed,
            init_name=init_name_topics,
            alpha=alpha,
            tau=tau,
            )

    if w_CK is None or w_CK.shape[1] < n_states:
        pprint('[init_params] Running init_w_CK %s ...' % (
            init_name_w))
        if init_name_w.count('regress'):
            assert dataset['n_docs'] < 1e6 # don't want this too big

            pprint('[init_params] Regress Step 1/2: Extract pi_DK...')
            pi_DK = calc_nef_map_pi_DK(
                dataset,
                topics_KV=topics_KV,
                alpha=alpha,
                n_seconds_between_print=600)

            prefix = '[init_params] Regress Step 2/2:'
            w_CK = estimate_w_CK__given_pi_DK(
                dataset=dataset,
                pi_DK=pi_DK,
                lambda_w=lambda_w,
                prefix=prefix,
                verbose=verbose,
                )
        else:
            raise ValueError("Unsupported init_name_w: " + init_name_w)

    assert topics_KV is not None
    assert w_CK is not None
    assert topics_KV.shape[0] == n_states
    assert w_CK.shape[1] == n_states
    pprint('[init_params] Done. Created init_param_dict.')
    return dict(
        w_CK=w_CK,
        topics_KV=topics_KV)



def init_topics_KV(
        dataset=None,
        topics_KV=None,
        n_states=None,
        init_name='rand_docs',
        init_model_path=None,
        n_missing_words_per_topic=0,
        seed=0,
        **kwargs):
    ''' Create topic-word parameters for specific dataset.

    Returns
    -------
    topics_KV : 2D array, K x V
        Each row sums to one
    '''

    if init_model_path is not None:
        assert init_model_path.endswith('.dump')
        param_dict = joblib.load(init_model_path)
        topics_KV = param_dict['topics_KV']
        return topics_KV

    if n_states is not None:
        K = int(n_states)
    V = int(dataset['n_vocabs'])
    if topics_KV is None:
        topics_KV = np.zeros((0, V))
    Ktmp = topics_KV.shape[0]
    if Ktmp < K:
        Knew = K - Ktmp
        topics_xKV = np.zeros((K, V))
        topics_xKV[:Ktmp] = topics_KV
        if init_name == 'rand_active_words':
            topics_xKV[-Knew:] = init_topics_KV__rand_active_words(
                n_states=Knew,
                n_vocabs=V,
                seed=seed)
        elif init_name == 'rand_docs':
            topics_xKV[-Knew:] = init_topics_KV__rand_docs(
                dataset=dataset,
                n_states=Knew,
                n_vocabs=V,
                seed=seed)
        elif init_name == 'rand_smooth':
            topics_xKV[-Knew:] = init_topics_KV__rand_smooth(
                dataset=dataset,
                n_states=Knew,
                n_vocabs=V,
                seed=seed)  
        else:
            raise ValueError("Unrecognized init_name %s" % init_name)
        topics_KV = topics_xKV
    topics_KV = topics_KV[:n_states]
    assert topics_KV.shape[0] == n_states

    # Post-process if desired
    if n_missing_words_per_topic > 0:
        prng = np.random.RandomState(int(seed))
        for k in xrange(n_states):
            high_prob_word_ids = np.flatnonzero(
                topics_KV[k] > 1.0 / n_vocabs)
            prng.shuffle(high_prob_word_ids)
            chosen_word_ids = high_prob_word_ids[:n_missing_words_per_topic]
            topics_KV[k, chosen_word_ids] = 1e-8
        topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]

    return topics_KV

def init_topics_KV__rand_active_words(
        n_states=10,
        frac_words_active=0.5,
        blend_frac_active=0.5,
        n_vocabs=144,
        seed=0):
    prng = np.random.RandomState(int(seed))
    unif_topics_KV = np.ones((n_states, n_vocabs)) / float(n_vocabs)
    active_topics_KV = np.zeros((n_states, n_vocabs))
    for k in xrange(n_states):
        active_words_U = prng.choice(
            np.arange(n_vocabs, dtype=np.int32),
            int(frac_words_active * n_vocabs),
            replace=False)
        active_topics_KV[k, active_words_U] = 1.0 / active_words_U.size
    topics_KV = (1 - blend_frac_active) * unif_topics_KV \
        + blend_frac_active * active_topics_KV
    topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]
    return topics_KV

def init_topics_KV__rand_smooth(
        dataset=None,
        n_states=10,
        n_vocabs=144,
        base_score=5.0,
        rand_score_type='uniform',
        rand_score_scale=1.0,
        seed=0):
    K = int(n_states)
    V = int(n_vocabs)
    prng = np.random.RandomState(int(seed))
    if rand_score_type == 'uniform':
        bg_topics_KV = base_score + rand_score_scale * prng.rand(K, V)
    else:
        raise ValueError("Unrecognized score type: %s" % rand_score_type)
    bg_topics_KV /= bg_topics_KV.sum(axis=1)[:,np.newaxis]
    return bg_topics_KV

def init_topics_KV__rand_docs(
        dataset=None,
        n_states=10,
        n_vocabs=144,
        blend_frac_doc=0.5,
        seed=0):
    prng = np.random.RandomState(int(seed))
    unif_topics_KV = np.ones((n_states, n_vocabs)) / float(n_vocabs)
    doc_KV = np.zeros((n_states, n_vocabs))
    chosen_doc_ids = prng.choice(
        np.arange(dataset['n_docs'], dtype=np.int32),
        n_states,
        replace=False)
    for k in xrange(n_states):
        start_d = dataset['doc_indptr_Dp1'][chosen_doc_ids[k]]
        stop_d = dataset['doc_indptr_Dp1'][chosen_doc_ids[k] + 1]
        active_words_U = dataset['word_id_U'][start_d:stop_d]
        doc_KV[k, active_words_U] = dataset['word_ct_U'][start_d:stop_d]
    doc_KV /= doc_KV.sum(axis=1)[:,np.newaxis]
    topics_KV = (1 - blend_frac_doc) * unif_topics_KV \
        + blend_frac_doc * doc_KV
    topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]
    return topics_KV



if __name__ == '__main__':
    import os
    from sklearn.externals import joblib

    # Simplest possible test
    # Load the toy bars dataset
    # Try several random initializations
    dataset_path = os.path.expandvars("$PC_REPO_DIR/datasets/toy_bars_3x3/")
    dataset = load_dataset(dataset_path, split_name='train')

    K = 5
    for init_name in ['rand_smooth', 'rand_docs']:
        for seed in range(3):
            GP_by_rep = dict()
            for rep in [0, 1]:
                GP_by_rep[rep] = init_param_dict(
                    dataset=dataset,
                    init_name=init_name,
                    n_states=int(K),
                    alpha=1.1,
                    tau=1.1,
                    lambda_w=0.001,
                    seed=seed,
                    )
            GP0 = GP_by_rep[0]
            GP1 = GP_by_rep[1]
            for key in GP0:
                assert np.allclose(GP0[key], GP1[key])

