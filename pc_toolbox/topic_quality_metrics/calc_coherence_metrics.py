import numpy as np
import scipy.sparse

def calc_npmi_and_pmi_coherence_for_top_ranked_terms_in_topic(
        top_vocab_ids=None,
        ndocs_csc_VV=None,
        dataset=None,
        pair_smooth_eps=0.1,
        marg_smooth_eps=None,
        ):
    """ Compute Coherence metric for given topic's top-ranked terms.

    Returns
    -------
    coherence_score : float
        Larger values indicate more coherent topics.

    Examples
    --------
    >>> x_DV = np.arange(6)[:,np.newaxis] * np.hstack([np.eye(6), np.zeros((6, 3))])
    >>> x_DV[:3, :3] += 1
    >>> x_DV[4, 5] += 17
    >>> _, ndocs_csc_VV = calc_pairwise_cooccurance_counts(x_csr_DV=x_DV)

    # Compute coherence for a very related pair
    >>> calc_npmi_and_pmi_coherence_for_top_ranked_terms_in_topic([2, 0], ndocs_csc_VV)[0]
    0.86755478351365201

    # Compute coherence for a very unrelated pair
    >>> calc_npmi_and_pmi_coherence_for_top_ranked_terms_in_topic([2, 5], ndocs_csc_VV)[0]
    -0.16789018869324493

    # Compute coherence for a pair where one word doesnt appear much
    >>> calc_npmi_and_pmi_coherence_for_top_ranked_terms_in_topic([0, 8], ndocs_csc_VV)[0]
    -0.0093324001175008262

    # Try coherence for first 3 (should be large)
    >>> calc_npmi_and_pmi_coherence_for_top_ranked_terms_in_topic([0,1,2], ndocs_csc_VV)
    (0.86755478351365201, 1.2954904783676406)

    # Try coherence for a bad set of 3 (should be small)
    >>> calc_npmi_and_pmi_coherence_for_top_ranked_terms_in_topic([0,3,6], ndocs_csc_VV)
    (0.13222810917463279, 0.65152143821207875)
    """
    top_vocab_ids = np.asarray(top_vocab_ids, dtype=np.int32)
    M = top_vocab_ids.size
    V = ndocs_csc_VV.shape[0]
    diag_ids = np.diag_indices(V)
    triu_ids = np.triu_indices(V, 1)
    P = len(triu_ids[0]) + len(diag_ids)

    ndocs_V = np.squeeze(np.asarray(ndocs_csc_VV.sum(axis=0)))
    ndocs_V -= np.squeeze(np.asarray(ndocs_csc_VV[diag_ids]))
    ndocs_V /= 2.0
    n_utoken_pairs = float(np.sum(ndocs_csc_VV[triu_ids]))
    assert np.allclose(n_utoken_pairs, ndocs_V.sum())

    if marg_smooth_eps is None:
        marg_smooth_eps = float(pair_smooth_eps * P) / V
    assert np.allclose(
        pair_smooth_eps * P,
        marg_smooth_eps * V)

    n_top_pairs = 0.0
    npmi_coherence_score = 0.0
    pmi_coherence_score = 0.0
    for mm, v in enumerate(top_vocab_ids[:-1]):
        Mrem = M - mm - 1
        counts_v_Mrem = ndocs_csc_VV[v, top_vocab_ids[mm+1:]]
        try:
            counts_v_Mrem = counts_v_Mrem.toarray()
        except AttributeError:
            pass
        assert counts_v_Mrem.size == Mrem
        jointprob_v_Mrem = (counts_v_Mrem + pair_smooth_eps) / (n_utoken_pairs + pair_smooth_eps * P)
        margprob_v_Mrem = (ndocs_V[top_vocab_ids[mm+1:]] + marg_smooth_eps) / (ndocs_V.sum() + marg_smooth_eps * V)
        margprob_v = (ndocs_V[v] + marg_smooth_eps) / (ndocs_V.sum() + marg_smooth_eps * V)

        denom_Mrem = np.log(jointprob_v_Mrem)
        numer_Mrem = denom_Mrem - np.log(margprob_v_Mrem) - np.log(margprob_v)

        npmi_coherence_score_Mrem = numer_Mrem / (-1.0 * denom_Mrem)
        assert np.all(npmi_coherence_score_Mrem >= -(1.00001))
        assert np.all(npmi_coherence_score_Mrem <=  (1.00001))

        pmi_coherence_score += np.sum(numer_Mrem)
        npmi_coherence_score += np.sum(npmi_coherence_score_Mrem)
        n_top_pairs += Mrem
    return (
        npmi_coherence_score / n_top_pairs,
        pmi_coherence_score / n_top_pairs,
        )


def calc_umass_coherence_for_top_ranked_terms_in_topic(
        top_vocab_ids=None,
        ndocs_V=None,
        ndocs_csc_VV=None,
        topics_KV=None,
        k=None,
        dataset=None,
        pair_smooth_eps=0.1,
        marg_smooth_eps=1e-9,
        ):
    """ Compute Coherence metric for given topic's top-ranked terms.

    Returns
    -------
    coherence_score : float
        Larger values indicate more coherent topics.

    Examples
    --------
    >>> x_DV = np.arange(6)[:,np.newaxis] * np.hstack([np.eye(6), np.zeros((6, 3))])
    >>> x_DV[:3, :3] += 1
    >>> x_DV[4, 5] += 17
    >>> ndocs_V, ndocs_csc_VV = calc_pairwise_cooccurance_counts(x_csr_DV=x_DV)
    >>> coh = calc_umass_coherence_for_top_ranked_terms_in_topic([0, 8], ndocs_V, ndocs_csc_VV)
    >>> coh2 = np.log(0.1 / 3.0)
    >>> np.allclose(coh, coh2)
    True
    >>> coh_good = calc_umass_coherence_for_top_ranked_terms_in_topic([0, 1, 2], ndocs_V, ndocs_csc_VV)
    >>> coh_bad = calc_umass_coherence_for_top_ranked_terms_in_topic([0, 4, 5], ndocs_V, ndocs_csc_VV)
    >>> coh_worst = calc_umass_coherence_for_top_ranked_terms_in_topic([0, 3, 7], ndocs_V, ndocs_csc_VV)
    >>> coh_good > coh_bad
    True
    >>> coh_bad > coh_worst
    True
    """
    V = ndocs_V.size
    top_vocab_ids = np.asarray(top_vocab_ids, dtype=np.int32)
    M = top_vocab_ids.size
    coherence_score = 0.0
    for mm, v in enumerate(top_vocab_ids[:-1]):
        Mrem = M - mm - 1
        counts_Mrem = ndocs_csc_VV[v, top_vocab_ids[mm+1:]]
        try:
            counts_Mrem = counts_Mrem.toarray()
        except AttributeError:
            pass
        assert counts_Mrem.size == Mrem
        coherence_score += (
            np.sum(np.log(counts_Mrem + pair_smooth_eps))
            - Mrem * np.log(ndocs_V[v] + marg_smooth_eps)
            )
    return coherence_score

def calc_pairwise_cooccurance_counts(
        x_csr_DV=None,
        dataset=None,
        ):
    """ Calculate word cooccurances across a corpus of D documents

    Returns
    -------
    ndocs_V : 1D array, size V
        entry v counts the number of documents that contain v at least once
    ndocs_csc_VV : 2D csc sparse matrix, V x V
        entry v,w counts the number of documents which contain
        the word pair (v, w) at least once

    Examples
    --------
    >>> x_DV = np.arange(6)[:,np.newaxis] * np.hstack([np.eye(6), np.zeros((6, 3))])
    >>> x_DV[:3, :3] += 1
    >>> x_DV[4, 5] += 17
    >>> ndocs_V, ndocs_csc_VV = calc_pairwise_cooccurance_counts(x_csr_DV=x_DV)
    >>> ndocs_V.astype(np.int32).tolist()
    [3, 3, 3, 1, 1, 2, 0, 0, 0]
    >>> ndocs_csc_VV.toarray()[:3, :3]
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.],
           [ 3.,  3.,  3.]])
    """
    if x_csr_DV is None:
        x_csr_DV = dataset['x_csr_DV']
    x_csr_DV = scipy.sparse.csr_matrix(x_csr_DV, dtype=np.float64)

    binx_csr_DV = x_csr_DV.copy()
    binx_csr_DV.data[:] = 1.0

    ndocs_V = np.squeeze(np.asarray(binx_csr_DV.sum(axis=0)))

    ndocs_csc_VV = (binx_csr_DV.T * binx_csr_DV).tocsc()
    return ndocs_V, ndocs_csc_VV


