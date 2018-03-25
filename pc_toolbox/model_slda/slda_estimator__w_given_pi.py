import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge as RidgeRegression

from pc_toolbox.utils_io import (
    pprint,
    )

def estimate_w_CK__given_pi_DK(
        dataset=None,
        pi_DK=None,
        lambda_w=0.001,
        seed=42,
        prefix='',
        verbose=False,
        **kwargs):
    """ Estimate regression weights from provided probability features.

    Uses sklearn's regularized regressors under the hood.

    Returns
    -------
    w_CK : 2D array, size C x K
        Regression weights
    """

    K = pi_DK.shape[1]
    C = int(dataset['n_labels'])
    if verbose:
        pprint('%s Fitting %d regressions...' % (
            prefix, C))

    w_CK = np.zeros((C, K))

    u_y_vals = np.unique(dataset['y_DC'].flatten())
    if u_y_vals.size <= 2 and np.union1d([0.0, 1.0], u_y_vals).size == 2:
        output_data_type = 'binary'
    else:
        output_data_type = 'real'

    if 'y_rowmask' in dataset:
        y_DC = dataset['y_DC'][1 == dataset['y_rowmask']]
        pi_DK = pi_DK[1 == dataset['y_rowmask']]
        u_y_vals = np.unique(y_DC.sum(axis=1))
        assert u_y_vals.size > 1
    else:
        y_DC = dataset['y_DC']

    for c in xrange(C):
        # Do a quick regression to get initial weights!
        if output_data_type.count('binary') > 0:
            clf = LogisticRegression(
                fit_intercept=False,
                C=0.5/lambda_w,
                random_state=seed,
                )
        else:
            clf = RidgeRegression(
                fit_intercept=False,
                alpha=lambda_w,
                random_state=seed,
                )

        clf.fit(pi_DK, y_DC[:, c])
        w_CK[c] = clf.coef_
        if verbose:
            pprint('  w_CK[%d, :5]=' % c + ' '.join(['% .2f' % w for w in w_CK[c, :5]]))
            pprint('  label id %d / %d done with lambda_w = %.5f' % (
                c+1, C, lambda_w))
    return w_CK