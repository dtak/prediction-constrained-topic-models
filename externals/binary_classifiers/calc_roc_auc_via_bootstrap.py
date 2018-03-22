import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

def verify_min_examples_per_label(y_NC, min_examples_per_label):
    '''
    
    Examples
    --------
    >>> y_all_0 = np.zeros(10)
    >>> y_all_1 = np.ones(30)
    >>> verify_min_examples_per_label(y_all_0, 3)
    False
    >>> verify_min_examples_per_label(y_all_1, 2)
    False
    >>> verify_min_examples_per_label(np.hstack([y_all_0, y_all_1]), 10)
    True
    >>> verify_min_examples_per_label(np.eye(3), 2)
    False
    '''
    if y_NC.ndim < 2:
        y_NC = np.atleast_2d(y_NC).T
    n_C = np.sum(np.isfinite(y_NC), axis=0)
    n_pos_C = n_C * np.nanmean(y_NC, axis=0)
    min_neg = np.max(n_C - n_pos_C)
    min_pos = np.min(n_pos_C)
    if min_pos < min_examples_per_label:
        return False
    elif min_neg < min_examples_per_label:
        return False
    return True

def calc_binary_clf_metric_with_ci_via_bootstrap(
        y_pred=None,
        y_true=None,
        metric_func=roc_auc_score,
        seed=42,
        verbose=False,
        n_bootstraps=1000,
        stratify_pos_and_neg=True,
        min_examples_per_label=10,
        return_dict=False,
        ci_tuples=[(10,90)]):
    if not isinstance(ci_tuples, list):
        ci_tuples = [ci_tuples]
    for ci_tuple in ci_tuples:
        assert len(ci_tuple) == 2

    roc_auc_value = metric_func(y_true, y_pred)
    if verbose:
        print(
            "Original score: {:0.3f}".format(roc_auc_value))

    n_samples = y_true.shape[0]
    prng = np.random.RandomState(seed)

    bootstrapped_scores = np.zeros(n_bootstraps, dtype=np.float64)

    if stratify_pos_and_neg:
        assert y_true.ndim == 1
        pos_ids = np.flatnonzero(y_true == 1)
        neg_ids = np.flatnonzero(y_true == 0)
        min_ex = np.minimum(pos_ids.size, neg_ids.size)
        assert min_ex >= min_examples_per_label
    i = 0
    while i < n_bootstraps:

        # Sample from original population with replacement
        if stratify_pos_and_neg:
            # Preserving the original number of pos and neg examples
            sampled_pos_inds = prng.random_integers(
                0, len(pos_ids) - 1, len(pos_ids))
            sampled_neg_inds = prng.random_integers(
                0, len(neg_ids) - 1, len(neg_ids))
            sampled_ids = np.hstack([
                neg_ids[sampled_neg_inds],
                pos_ids[sampled_pos_inds]])
        else:
            # Don't care about pos and neg ratio at all
            sampled_ids = prng.choice(
                n_samples, size=n_samples, replace=True)

        sampled_y_true = y_true[sampled_ids]
        sampled_y_pred = y_pred[sampled_ids]
        is_good = verify_min_examples_per_label(sampled_y_true, min_examples_per_label)
        if not is_good:
            continue

        bootstrapped_scores[i] = metric_func(sampled_y_true, sampled_y_pred)
        i += 1

    if verbose:
        for perc in [05, 10, 25, 50, 75, 90, 95]:
            print "%02d percentile: %.3f" % (
                perc, np.percentile(bootstrapped_scores, perc))

    intervals = list()
    for ci_tuple in ci_tuples:
        ci_bound_low = int(ci_tuple[0])
        ci_bound_high = int(ci_tuple[1])
        interval = (
                np.percentile(bootstrapped_scores, ci_bound_low),
                np.percentile(bootstrapped_scores, ci_bound_high),
                )
        intervals.append(interval)
        if verbose:
            print "CI %2d-%2d:  %.3f - %.3f" % (
                ci_bound_low, ci_bound_high,
                interval[0], interval[1],
                )

    if return_dict:
        info_dict = dict(
            ci_tuples=ci_tuples,
            bootstrapped_scores=bootstrapped_scores)
        return roc_auc_value, intervals, info_dict
    else:
        return roc_auc_value, intervals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--stratify_pos_and_neg', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    arg_dict = vars(parser.parse_args())

    y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
    y_true = np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])

    val, ci = calc_binary_clf_metric_with_ci_via_bootstrap(
        y_pred=y_pred,
        y_true=y_true,
        return_dict=False,
        **arg_dict)
