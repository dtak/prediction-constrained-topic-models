'''
train_and_eval_sklearn_binary_classifier.py

Usage
-----
$ python train_and_eval_sklearn_binary_classifier.py \
    --dataset_path [path] \
    --output_path [path] \
    [optional args]

Optional arguments
------------------
--dataset_path DATASET_PATH
                        Path to folder containing:
                            *.npy files: X_train, y_train, P_train
                            *.txt files: X_colnames.txt, y_colnames.txt
--output_path OUTPUT_PATH
                        Path to folder holding output from this evaluator.
                        Includes:
                            * clf_<id>_.dump : loadable clf object
                            * clf_<id>_callback_train.csv : perf metrics
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import datetime
import sys
import os
import argparse
import itertools
import time
import scipy.sparse
import copy
import dill

from collections import OrderedDict
from distutils.dir_util import mkpath

from sklearn.externals import joblib
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, MetaEstimatorMixin
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, Binarizer, FunctionTransformer
from sklearn.preprocessing import normalize, binarize, minmax_scale
from sklearn.pipeline import Pipeline
from pc_toolbox.utils_io import (
    load_csr_matrix, pprint, config_pprint_logging,
    load_list_of_strings_from_txt,
    load_list_of_unicode_from_txt,
    )

from calc_roc_auc_via_bootstrap import calc_binary_clf_metric_with_ci_via_bootstrap

def read_args_from_stdin_and_run():
    ''' Main executable function to train and evaluate classifier.

    Post Condition
    --------------
    AUC and other eval info printed to stdout.
    Trained classifier saved ???.
    '''
    if not sys.stdin.isatty():
        for line in sys.stdin.readlines():
            line = line.strip()
            sys.argv.extend(line.split(' '))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        default='/tmp/',
        type=str,
        help="Path to folder containing:" +
        " *.npy files: X_train, y_train, P_train"
        " *.txt files: X_colnames.txt and y_colnames.txt")
    parser.add_argument(
        '--output_path',
        default='/tmp/',
        type=str,
        help="Path to folder to hold output from classifier. Includes:" +
        " perf_metric*.txt files: auc_train.txt & auc_test.txt" +
        " settings.txt: description of all settings to reproduce.")
    parser.add_argument(
        '--feature_arr_names',
        type=str,
        default='X',
        help='Name of feature files to use for training')
    parser.add_argument(
        '--features_path',
        default='/tmp/',
        type=str,
        help="Path to folder with extra feature files")
    parser.add_argument(
        '--target_arr_name',
        default='Y',
        type=str,
        )
    parser.add_argument(
        '--target_names',
        default='all',
        type=str,
        help='Name of response/intervention to test.' +
        ' To try specific interventions, write names separated by commas.' +
        ' To try all interventions, use special name "all"')
    parser.add_argument(
        '--n_folds',
        default=1,
        type=int,
        help='Number of folds for cross validation during classification.')
    parser.add_argument(
        '--classifier_name',
        default='logistic_regression',
        choices=[
            'k_nearest_neighbors',
            'mlp',
            'logistic_regression',
            'extra_trees',
            'svm_with_linear_kernel',
            'svm_with_rbf_kernel'],
        help='Name of classifier')
    parser.add_argument(
        '--class_weight_opts',
        choices=['none', 'balanced'],
        default='none',
        )
    parser.add_argument(
        '--max_grid_search_steps', 
        default=None,
        type=int,
        help='max number of steps for grid search')
    parser.add_argument(
        '--frac_labels_train', 
        default=1.0,
        type=float,
        help='Fraction of the training data to use')
    parser.add_argument(
        '--c_logspace_arg_str',
        default="-6,4,7",
        type=unicode,
        help='Comma-sep list of args to np.logspace')
    parser.add_argument(
        '--seed',
        default=8675309,
        type=int,
        help='Seed for random number generation')
    parser.add_argument(
        '--seed_bootstrap',
        default=42,
        type=int,
        help='Seed for bootstrap')
    parser.add_argument(
        '--n_bootstraps',
        default=5000,
        type=int,
        help='Number of samples for bootstrap conf. intervals')
    parser.add_argument(
        '--bootstrap_stratify_pos_and_neg',
        default=True,
        type=int,
        help='Whether to stratify examples or not')
    args, unk_list = parser.parse_known_args()
    arg_dict = vars(args)

    dataset_path = arg_dict['dataset_path']
    for key, val in arg_dict.items():
        if arg_dict['output_path'].count('$' + key):
            arg_dict['output_path'] = \
                arg_dict['output_path'].replace('$' + key, str(val))
    if not os.path.exists(arg_dict['output_path']):
        mkpath(arg_dict['output_path'])

    config_pprint_logging(
        arg_dict['output_path'],
        txtfile='stdout_%s.txt' % arg_dict['target_names'])
    pprint('[run_classifier says:] Parsing args ...')

    # Parse possible preprocessors
    feat_preproc_grid_dict = dict()
    for key, val in zip(unk_list[::2], unk_list[1::2]):
        if key.startswith('--preproc_'):
            feat_preproc_grid_dict[key[2:]] = str(val).split(',')
            pprint(key + " : " + val)
            arg_dict[key[2:]] = val

    for key in feat_preproc_grid_dict.keys():
        ii = unk_list.index('--' + key)
        del unk_list[ii+1]
        del unk_list[ii]
    if len(unk_list) > 0:
        pprint("UNKNOWN ARGS (ignored)")
        for key in unk_list:
            pprint(key)

    # Set default seed for numpy
    np.random.seed(arg_dict['seed'])

    # Write parsed args to plain-text file
    # so we can exactly reproduce later
    with open(os.path.join(arg_dict['output_path'], 'settings.txt'), 'w') as f:
        for key, val in arg_dict.items():
            f.write(key + ' = ' + str(val) + '\n')
            pprint(key + ' = ' + str(val))
    with open(os.path.join(arg_dict['output_path'], 'args.txt'), 'w') as f:
        for key, val in arg_dict.items():
            f.write('--' + key + ' ' + str(val) + '\n')
    pprint('')


    feat_path_list = [
        arg_dict['dataset_path'],
        arg_dict['features_path']]

    pprint('[run_classifier says:] Loading dataset ...')
    start_time = time.time()
    feature_arr_names = arg_dict['feature_arr_names'].split(',')
    pprint('feature_arr_names:')
    feat_colnames_by_arr = OrderedDict()
    for feat_arr_name in feature_arr_names:
        pprint(feat_arr_name)
        cur_feat_colnames = None
        for feat_path in feat_path_list:
            colname_fpath = os.path.join(
                feat_path,
                feat_arr_name + '_colnames.txt')
            if os.path.exists(colname_fpath):
                cur_feat_colnames = \
                    [unicode(feat_arr_name + ":") + s
                        for s in load_list_of_unicode_from_txt(colname_fpath)]
                break
        feat_colnames_by_arr[feat_arr_name] = cur_feat_colnames

    target_arr_name = arg_dict['target_arr_name']
    all_target_names = load_list_of_strings_from_txt(os.path.join(
        arg_dict['dataset_path'],
        target_arr_name + '_colnames.txt'))

    target_names = arg_dict['target_names']
    if target_names == 'all':
        target_names = all_target_names
        target_cols = np.arange(len(all_target_names)).tolist()
    else:
        target_names = target_names.split(',')
        target_cols = list()
        for name in target_names:
            assert name in all_target_names
            target_cols.append(all_target_names.index(name))


    datasets_by_split = dict()
    for split_name in ['train', 'valid', 'test']:
        datasets_by_split[split_name] = dict()
        split_dataset = datasets_by_split[split_name]

        # Load Y
        dense_fpath = os.path.join(
            dataset_path,
            target_arr_name + "_%s.npy" % split_name)
        y = np.asarray(np.load(dense_fpath), order='C', dtype=np.float32) # 0/1/nan
        if y.ndim < 2:
            y = y[:,np.newaxis]
        assert y.ndim == 2
        assert y.shape[1] == len(all_target_names)
        split_dataset['y'] = y[:, target_cols]
        assert split_dataset['y'].shape[1] == len(target_cols)

        # Load X
        x_list = list()      
        for feat_arr_name in feature_arr_names:
            for ii, feat_path in enumerate(feat_path_list):
                dense_fpath = os.path.join(
                    feat_path,
                    feat_arr_name + "_%s.npy" % split_name)
                sparse_fpath = os.path.join(
                    feat_path,
                    feat_arr_name + "_csr_%s.npz" % split_name)
                x_cur = None
                try:
                    if os.path.exists(sparse_fpath):
                        x_cur = load_csr_matrix(sparse_fpath)
                        assert np.all(np.isfinite(x_cur.data))
                        break
                    else:
                        x_cur = np.asarray(
                            np.load(dense_fpath),
                            order='C', dtype=np.float64)
                        if x_cur.ndim < 2:
                            x_cur = np.atleast_2d(x_cur).T
                        assert np.all(np.isfinite(x_cur))
                        break
                except IOError as e:
                    if ii == len(feat_path_list) - 1:
                        # Couldn't find desired file in any feat_path
                        raise e
                    else:
                        # Try the next feat_path in the list
                        pass
            if x_cur is not None:
                if feat_colnames_by_arr[feat_arr_name] is not None:
                    feat_dim = len(feat_colnames_by_arr[feat_arr_name])
                    assert x_cur.shape[1] == feat_dim
                else:
                    # Add dummy colnames
                    feat_dim = x_cur.shape[1]
                    n_sig_digits = np.maximum(
                        3, int(np.ceil(np.log10(feat_dim))))
                    fmt_str = "%s_%0" + str(n_sig_digits) + "d"
                    feat_colnames_by_arr[feat_arr_name] = [
                        fmt_str % (feat_arr_name, fid)
                            for fid in range(feat_dim)]
                x_list.append(x_cur)

        if isinstance(x_list[0], np.ndarray):
            split_dataset['x'] = np.hstack(x_list)
        else:
            split_dataset['x'] = scipy.sparse.hstack(x_list, format='csr')

        #Use only a fraction of the training dataset if specified
        frac_labels_train = arg_dict['frac_labels_train']
        if split_name == 'train' and frac_labels_train < 1.0:
            # Same random seed taken from bow_dataset.py
            data_prng = np.random.RandomState(int(42))
            n_rows = y.shape[0]

            #Note: does not handle truly missing labels
            indexed_rows = np.arange(n_rows)
            shuffled_rows = data_prng.permutation(indexed_rows)
            n_visible = int(np.ceil(frac_labels_train*n_rows))
            visible_rows = shuffled_rows[:n_visible]
            split_dataset['x'] = split_dataset['x'][visible_rows, :]
            split_dataset['y'] = split_dataset['y'][visible_rows, :]

        assert split_dataset['x'].ndim == 2
        assert split_dataset['x'].shape[0] == split_dataset['y'].shape[0]
        assert (
            isinstance(split_dataset['x'], np.ndarray)
            or isinstance(split_dataset['x'], scipy.sparse.csr_matrix)
            )

        if split_name == 'train':
            # Flatten feat colnames into single list
            feat_colnames = sum(feat_colnames_by_arr.values(), [])
            assert isinstance(feat_colnames, list)
            assert len(feat_colnames) == split_dataset['x'].shape[1]
            if len(feat_colnames) > 10:
                pprint(
                    'x colnames: %s ... %s' % (
                        ' '.join(feat_colnames[:5]),
                        ' '.join(feat_colnames[-5:])))
            else:
                pprint('x colnames: %s' % ' '.join(feat_colnames))
            pprint('y colnames: %s' % ' '.join(target_names))

        pprint('---- %5s dataset summary' % split_name)
        pprint('%9d total examples' % y.shape[0])
        pprint('y : %d x %d targets' % split_dataset['y'].shape)
        pprint('x : %d x %d features' % split_dataset['x'].shape)

        for c in xrange(len(target_names)):
            y_c = split_dataset['y'][:,c]
            nan_bmask = np.isnan(y_c)
            pos_bmask = y_c == 1
            neg_bmask = y_c == 0
            pprint('target %s :' % target_names[c])
            pprint('    %6d pos examples | %.3f' % (np.sum(pos_bmask), calcfrac(pos_bmask)))
            pprint('    %6d neg examples | %.3f' % (np.sum(neg_bmask), calcfrac(neg_bmask)))
            pprint('    %6d NaN examples | %.3f' % (np.sum(nan_bmask), calcfrac(nan_bmask)))
            assert nan_bmask.sum() + pos_bmask.sum() + neg_bmask.sum() == neg_bmask.size

    elapsed_time = time.time() - start_time
    pprint('[run_classifier says:] dataset loaded after %.2f sec.' % elapsed_time)

    n_cols = len(target_names)
    for c in xrange(n_cols):
        pprint('[run_classifier says:] train for target %s' % target_names[c])
        train_and_eval_clf_with_best_params_via_grid_search(
            arg_dict['classifier_name'],
            datasets_by_split=datasets_by_split,
            y_col_id=c,
            y_orig_col_id=all_target_names.index(target_names[c]),
            y_col_name=target_names[c],
            feat_colnames=feat_colnames,
            feat_preproc_grid_dict=feat_preproc_grid_dict,
            output_path=arg_dict['output_path'],
            max_grid_search_steps=arg_dict['max_grid_search_steps'],
            class_weight_opts=arg_dict['class_weight_opts'],
            c_logspace_arg_str=arg_dict['c_logspace_arg_str'],
            random_state=arg_dict['seed'],
            seed_bootstrap=arg_dict['seed_bootstrap'],
            n_bootstraps=arg_dict['n_bootstraps'],
            bootstrap_stratify_pos_and_neg=arg_dict['bootstrap_stratify_pos_and_neg'],
            )
        elapsed_time = time.time() - start_time
        pprint('[run_classifier says:] target %s completed after %.2f sec' % (target_names[c], elapsed_time))

def calcfrac(bmask):
    return np.sum(bmask) / float(bmask.size)

def default_param_grid(classifier_name, c_logspace_arg_str='-6,4,6', **kwargs):
    C_range = np.logspace(*map(float, c_logspace_arg_str.split(','))) 
    if classifier_name == 'logistic_regression':
        return OrderedDict([
            ('penalty', ['l2', 'l1']),
            ('class_weight', [None]),
            ('C', C_range),
            ('thr_', [0.5]),
            ])
    elif classifier_name == 'extra_trees':
        return OrderedDict([
            ('class_weight', [None]),
            ('n_estimators', np.asarray([16, 64, 256])),
            ('max_features', np.asarray([0.04, 0.16, 0.64])),
            ('min_samples_leaf', np.asarray([4, 16, 64, 256])), # bigger seems to be better
            ('thr_', [0.5]),
            ])
    elif classifier_name == 'svm_with_linear_kernel':
        return OrderedDict([
            ('kernel', ['linear']),
            ('C', C_range),
            ('class_weight', [None]),
            ('probability', [False]),
            ])
    elif classifier_name == 'svm_with_rbf_kernel':
        return OrderedDict([
            ('kernel', ['rbf']),
            ('C', C_range),
            ('gamma', np.logspace(-6, 6, 5)),
            ('class_weight', [None]),
            ('probability', [False]),
            ])
    elif classifier_name == 'k_nearest_neighbors':
        return OrderedDict([
            ('n_neighbors', [4, 16, 32, 64]),
            ('metric', ['euclidean', 'manhattan']),
            ('weight', ['uniform', 'distance']),
            ('algorithm', ['brute']),
            ])
    elif classifier_name == 'mlp':
        return OrderedDict([
            #('norm', ['l1', 'none']),
            ('hidden_layer_sizes', [(16), (64), (256), (1024)]),
            ('solver', ['adam']),
            ('alpha', np.logspace(-6, 0, 3)),
            ('learning_rate_init', np.asarray([0.01, 0.1])),
            ('activation', ['relu']),
            ('batch_size', [200]),
            ('early_stopping', [True]),
            ])
    else:
        raise ValueError("Unrecognized: " + classifier_name)

def make_constructor_and_evaluator_funcs(
        classifier_name,
        n_bootstraps=5000,
        seed_bootstrap=None,
        bootstrap_stratify_pos_and_neg=True,
        ):

    def calc_auc_score(clf, x, y):
        try:
            yscore = clf.decision_function(x)
        except AttributeError as e:
            yscore = clf.predict_proba(x)
        if yscore.ndim > 1:
            assert yscore.shape[1] == 2
            yscore = yscore[:, 1]
        assert y.ndim == 1
        assert yscore.ndim == 1
        return roc_auc_score(y, yscore)

    def calc_f1_conf_intervals(
            clf, x, y, ci_tuples=[(2.5,97.5), (10,90)], pos_label=1):
        yhat = clf.predict(x)
        assert y.ndim == 1
        assert yhat.ndim == 1

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        if n_pos > 1.2 / 2.0 * (n_pos + n_neg):
            raise Warning("Positive examples are much more common")
        def my_f1_score(ya, yb):
            return f1_score(ya, yb, pos_label=pos_label)
        return calc_binary_clf_metric_with_ci_via_bootstrap(
            y_pred=yhat,
            y_true=y,
            metric_func=my_f1_score,
            ci_tuples=ci_tuples,
            n_bootstraps=n_bootstraps,
            seed=seed_bootstrap,
            stratify_pos_and_neg=bootstrap_stratify_pos_and_neg)

    def calc_auc_conf_intervals(clf, x, y, ci_tuples=[(2.5,97.5), (10,90)]):
        try:
            yscore = clf.decision_function(x)
        except AttributeError as e:
            yscore = clf.predict_proba(x)
        if yscore.ndim > 1:
            assert yscore.shape[1] == 2
            yscore = yscore[:, 1]
        assert y.ndim == 1
        assert yscore.ndim == 1
        return calc_binary_clf_metric_with_ci_via_bootstrap(
            y_pred=yscore,
            y_true=y,
            metric_func=roc_auc_score,
            ci_tuples=ci_tuples,
            n_bootstraps=n_bootstraps,
            seed=seed_bootstrap,
            stratify_pos_and_neg=bootstrap_stratify_pos_and_neg)

    def calc_accuracy_score(clf, x, y):
        yhat = clf.predict(x)
        assert y.ndim == 1
        assert yhat.ndim == 1
        return np.sum(y == yhat) / float(y.size)

    def calc_f1_score(clf, x, y):
        yhat = clf.predict(x)
        assert y.ndim == 1
        assert yhat.ndim == 1
        return f1_score(y, yhat, pos_label=clf.classes_[1])

    def make_confusion_matrix_report(clf, x, y):
        assert len(clf.classes_) == 2
        assert clf.classes_[0] == 0
        assert clf.classes_[1] == 1
        
        y_pred = clf.predict(x)
        cm = sk_confusion_matrix(y, y_pred)
        cm = pd.DataFrame(data=cm, columns=[0, 1], index=[0, 1])
        cm.columns.name = 'Predicted label'
        cm.index.name = 'True label'
        return "\n%s\n" % unicode(cm)

    def make_clf_report(clf, x, y, header=''):
        r_str = header
        r_str += make_confusion_matrix_report(clf, x, y)
        r_str += u"acc %.4f\n" % calc_accuracy_score(clf, x, y)
        r_str += u" f1 %.4f\n" % calc_f1_score(clf, x, y)
        r_str += u"auc %.4f\n" % calc_auc_score(clf, x, y)
        return r_str

    def make_csv_row_dict(clf, x, y, y_col_name, split_name, classifier_name):
        keepers = np.isfinite(y)
        x = x[keepers]
        y = y[keepers]

        ci_tuples = [(2.5,97.5), (10,90)]
        auc_val, auc_intervals = calc_auc_conf_intervals(
            clf, x, y, ci_tuples)
        f1_val, f1_intervals = calc_f1_conf_intervals(
            clf, x, y, ci_tuples)
        row_dict = OrderedDict([
            ('Y_COL_NAME', y_col_name),
            ('SPLIT_NAME', split_name.upper()),
            ('Y_ROC_AUC', auc_val),
            ('Y_ERROR_RATE', 1.0 - calc_accuracy_score(clf, x, y)),
            ('Y_F1_SCORE', f1_val),
            ('LEGEND_NAME', classifier_name),
            ('IS_BEST_SNAPSHOT', 1),
            ('N_BOOTSTRAPS', n_bootstraps),
            ('BOOTSTRAP_STRATIFY_POS_AND_NEG', bootstrap_stratify_pos_and_neg),
            ])
        for ci_tuple, auc_tuple in zip(ci_tuples, auc_intervals):
            for perc, val in zip(ci_tuple, auc_tuple):
                row_dict['Y_ROC_AUC_CI_%05.2f%%' % perc] = val

        for ci_tuple, f1_tuple in zip(ci_tuples, f1_intervals):
            for perc, val in zip(ci_tuple, f1_tuple):
                row_dict['Y_F1_SCORE_CI_%05.2f%%' % perc] = val
        return row_dict

    def make_interpretability_report_for_clf(
            clf, feat_colnames, y_colname, n_top_feats=20):
        if isinstance(clf, Pipeline):
            clf = clf.named_steps['clf']
        
        if hasattr(clf, 'clf'):
            clf = clf.clf # Unwrap Thr classifier

        if isinstance(clf, LogisticRegression):
            w = clf.coef_[0]
            F = w.size
            assert F == len(feat_colnames)
            sorted_feat_ids = np.argsort(w)
            topk = np.minimum(n_top_feats, int(np.ceil(F / 2.0)))
            neg_ids = sorted_feat_ids[:topk].tolist()
            pos_ids = sorted_feat_ids[-topk:][::-1].tolist()
            if len(neg_ids) + len(pos_ids) == F + 1:
                pos_ids.pop() # double counting
            assert len(neg_ids) + len(pos_ids) <= F
            while len(pos_ids) > 0 and w[pos_ids[-1]] < 0:
                neg_ids.append(pos_ids.pop())
            while len(neg_ids) > 0 and w[neg_ids[-1]] > 0:
                pos_ids.append(neg_ids.pop())

            list_of_u = list()
            for name, feat_ids in [('neg', neg_ids), ('pos', pos_ids)]:
                list_of_u.append(u'--- top %s features for task %s' % (name, y_colname))
                for ii, f in enumerate(feat_ids):
                    u = u'rank %3d/%d  feat_uid %6d  coef % 13.6f %s' % (1+ii, F, f, w[f], feat_colnames[f])
                    list_of_u.append(u)
                list_of_u.append(u'')
            return u'\n'.join(list_of_u)
        elif isinstance(clf, ExtraTreesClassifier):
            score_per_tree = [tree.feature_importances_ for tree in clf.estimators_]
            mean_score_F = np.mean(score_per_tree, axis=0)
            stdv_score_F = np.std(score_per_tree, axis=0)
            F = mean_score_F.size
            assert F == len(feat_colnames)
            sorted_feat_ids = np.argsort(-1 * mean_score_F)

            list_of_u = [u'--- top features by importance score for task %s' % y_colname]
            for ii, f in enumerate(sorted_feat_ids[:int(n_top_feats)]):
                u = u"rank %3d/%d  feat_uid %6d   mean %13.6f  stddev %13.4f %s" % (
                    ii+1, F, f, mean_score_F[f], stdv_score_F[f], feat_colnames[f])
                list_of_u.append(u)
            list_of_u.append('')
            return u'\n'.join(list_of_u)

        # Base case: classifier not easily interpreted 
        return u''

    def make_classifier(feat_colnames=None, **params):
        steps = list()

        def make_t_func_for_specific_cols(t_func, t_kws, col_names, prefix):
             def tmp_t_func(X, _state_vars=dict()):
                 col_ids = [i for i in range(len(col_names)) if col_names[i].startswith(prefix)]
                 X = X.copy()
                 if t_func == 'rescale01':
                     if isinstance(X, np.ndarray):
                         x_col_arr = X[:, col_ids]
                     else:
                         x_col_arr = X[:, col_ids].toarray()
                     try:
                         min_val = _state_vars['min_val']
                         max_val = _state_vars['max_val']
                     except KeyError:
                         max_val = np.max(x_col_arr, axis=0)
                         min_val = np.min(x_col_arr, axis=0)
                         _state_vars['min_val'] = min_val
                         _state_vars['max_val'] = max_val
                     X[:, col_ids] = (x_col_arr - min_val) / (max_val - min_val)
                 else:
                     X[:, col_ids] = t_func(X[:, col_ids], **t_kws)
                 return X
             return tmp_t_func

        # STAGE ONE: preprocessing / normalization
        thr = None
        for key in params.keys():
            if key.startswith('thr_'):
                thr = params.pop(key)
            if key.startswith('preproc_'):
                t_func_name = str(params.pop(key))
                feat_prefix = key.split('preproc_')[1]
                if t_func_name == 'none':
                    continue
                elif t_func_name == 'normalize_l1':
                    t_func = make_t_func_for_specific_cols(normalize, dict(norm='l1'), feat_colnames, feat_prefix)
                elif t_func_name == 'binarize':
                    t_func = make_t_func_for_specific_cols(binarize, {}, feat_colnames, feat_prefix)
                elif t_func_name == 'rescale01':
                    t_func = make_t_func_for_specific_cols('rescale01', {}, feat_colnames, feat_prefix)
                else:
                    raise ValueError("Unrecognized t_func_name: %s" % t_func_name)
                steps.append((key, FunctionTransformer(t_func, accept_sparse=True)))
        if 'norm' in params:
            norm = params.pop('norm')
            norm_str = str(norm).lower()
            if norm_str == 'l1':
                steps.append(
                    ('preproc', Normalizer(norm=norm)))
            elif norm_str == 'bin':
                steps.append(
                    ('preproc', Binarizer(threshold=0.0)))
        # STAGE TWO: classifier
        if classifier_name.lower().count('logistic_regression'):
            steps.append(
                ('clf', LogisticRegression(**params)))
        elif classifier_name.lower() == 'extra_trees':
            steps.append(
                ('clf', ExtraTreesClassifier(**params)))
        elif classifier_name.lower().startswith('svm'):
            steps.append(
                ('clf', SVC(**params)))
        elif classifier_name.lower().startswith('k_nearest_neighbors'):
            steps.append(
                ('clf', KNeighborsClassifier(**params)))
        elif classifier_name.lower().startswith('mlp'):
            steps.append(
                ('clf', MLPClassifier(**params)))
        else:
            raise ValueError(
                "Unrecognized classifier_name: " + classifier_name)

        # STAGE THREE: thresholding
        if thr is not None:
            # Overwrite the previous step
            clf = steps[-1][1]
            steps[-1] = (
                'clf', ThresholdClassifier(clf, thr))
        return Pipeline(steps)

    return (
        make_classifier, calc_auc_score, np.argmax,
        make_clf_report, make_csv_row_dict,
        make_interpretability_report_for_clf)

def make_param_dict_generator(param_grid_dict):
    ''' Make iterable that will loop thru each combo of params

    Example
    -------
    >>> pgD = OrderedDict()
    >>> pgD['C'] = np.asarray([1,2,3])
    >>> pgD['alpha'] = np.asarray([0.5, 2.5])
    >>> gen = make_param_dict_generator(pgD)
    >>> gen.next()
    OrderedDict([('C', 1), ('alpha', 0.5)])
    >>> gen.next()
    OrderedDict([('C', 1), ('alpha', 2.5)])
    >>> gen.next()
    OrderedDict([('C', 2), ('alpha', 0.5)])
    '''
    list_of_keys = param_grid_dict.keys()
    list_of_grids = param_grid_dict.values()
    for list_of_vals in itertools.product(*list_of_grids):
        yield OrderedDict(zip(list_of_keys, list_of_vals))

def make_nonnan_xy_for_target(
        dataset_dict,
        y_col_id=0):
    x = dataset_dict['x']
    y = dataset_dict['y'][:, y_col_id]
    assert y.ndim == 1
    assert y.shape[0] == x.shape[0]
    keepers = np.isfinite(y)
    return x[keepers], y[keepers]

def train_and_eval_clf_with_best_params_via_grid_search(
        classifier_name='logreg',
        param_grid_dict=None,
        datasets_by_split=None,
        verbose=True,
        feat_colnames=None,
        feat_preproc_grid_dict=None,
        y_col_id=0,
        y_orig_col_id=0,
        y_col_name='',
        output_path='/tmp/',
        max_grid_search_steps=None,
        class_weight_opts='',
        c_logspace_arg_str='',
        random_state=8675309,
        n_bootstraps=5000,
        seed_bootstrap=42,
        bootstrap_stratify_pos_and_neg=True,
        ):
    (make_classifier, score_classifier, calc_best_idx,
        make_clf_report, make_csv_row_dict, make_interp_report) = \
            make_constructor_and_evaluator_funcs(
                classifier_name,
                n_bootstraps=n_bootstraps,
                seed_bootstrap=seed_bootstrap,
                bootstrap_stratify_pos_and_neg=bootstrap_stratify_pos_and_neg)
    if param_grid_dict is None:
        param_grid_dict = default_param_grid(classifier_name, c_logspace_arg_str=c_logspace_arg_str)
        if class_weight_opts == 'balanced':
            if 'class_weight' in param_grid_dict:
                 param_grid_dict['class_weight'].insert(0, 'balanced')
    if isinstance(feat_preproc_grid_dict, dict):
        param_grid_dict.update(feat_preproc_grid_dict)

    n_grid = 1
    for key, val_list in param_grid_dict.items():
        n_grid *= len(val_list)
    if verbose:
        if max_grid_search_steps:
            pprint('Max   configs in grid search: %d' % max_grid_search_steps)
        pprint('Total configs in grid search: %d' % n_grid)

    param_generator = make_param_dict_generator(param_grid_dict)

    clf_list = list()
    param_dict_list = list()
    score_list = list()
    start_time = time.time()

    x_tr, y_tr = make_nonnan_xy_for_target(
        datasets_by_split['train'], y_col_id)
    x_va, y_va = make_nonnan_xy_for_target(
        datasets_by_split['valid'], y_col_id)
    x_te, y_te = make_nonnan_xy_for_target(
        datasets_by_split['test'], y_col_id)
    for ii, param_dict in enumerate(param_generator):
        np.random.seed(random_state)
        clf = make_classifier(
            feat_colnames=feat_colnames,
            random_state=random_state,
            **param_dict)
        
        clf.fit(x_tr, y_tr)
        score = score_classifier(
            clf, x_va, y_va)
        clf_list.append(clf)
        score_list.append(score)
        param_dict_list.append(param_dict)

        if verbose:
            tr_score = score_classifier(
                clf, x_tr, y_tr)

            elapsed_time = time.time() - start_time
            param_str = str(param_dict)
            param_str = param_str.replace('),', '  ')
            for badstr in ['OrderedDict', '[', ']', '(', ')', ',']:
                param_str = param_str.replace(badstr, '')
            pprint("%4d/%d %10.2f sec va_auc %.4f   tr_auc %.4f  %s" % (
                1+ii, n_grid, elapsed_time, score, tr_score, param_str))

        if max_grid_search_steps and ((ii+1) >= max_grid_search_steps):
            if verbose:
                pprint("Exceed max_grid_search_steps. Break!")
            break

    best_id = calc_best_idx(score_list)
    best_score = score_list[best_id]
    best_param_dict = param_dict_list[best_id]
    best_clf = clf_list[best_id]

    if verbose:
        pprint("------")
        pprint(" best param dict, using function " + calc_best_idx.__name__)
        pprint("------")
        pprint("va_auc = %.4f %s" % (best_score, str(best_param_dict)))

    ## Now tuning threshold, if applicable
    if isinstance(best_clf.named_steps['clf'], ThresholdClassifier):
        yproba_class1 = best_clf.predict_proba(x_va)[:,1]
        if verbose:
            pprint("Percentiles of Pr(y=1) on validation...")
            for perc in [0, 1, 10, 50, 90, 99, 100]:
                perc_str = "  %4d%% %.4f" % (
                    perc,
                    np.percentile(yproba_class1, perc))
                pprint(perc_str)
        thr_min = np.maximum(0.001, np.min(yproba_class1))
        thr_max = np.minimum(0.999, np.max(yproba_class1))
        thr_grid = np.linspace(thr_min, thr_max, num=101)
        if verbose:
            pprint("Searching thresholds...")
            pprint("thr_grid = %.4f, %.4f, %.4f ... %.4f, %.4f" % (
                thr_grid[0], thr_grid[1], thr_grid[2], thr_grid[-2], thr_grid[-1]))
        score_grid = np.zeros_like(thr_grid, dtype=np.float64)
        tmp_clf = copy.deepcopy(best_clf)
        for gg, thr in enumerate(thr_grid):
            tmp_clf.named_steps['clf'].set_threshold(thr)
            yhat = tmp_clf.predict(x_va)
            score_grid[gg] = f1_score(y_va, yhat, pos_label=1)
        gg_best = np.argmax(score_grid)
        best_clf.named_steps['clf'].set_threshold(thr_grid[gg_best])
        if verbose:
            pprint("------")
            pprint(" best threshold by f1 score on validation")
            pprint("------")
            pprint("thr = %.4f f1_score %.4f" % (
                thr_grid[gg_best],
                score_grid[gg_best],
                ))

    if verbose:
        pprint()
        pprint(make_clf_report(
            best_clf, x_va, y_va,
            y_col_name + '_valid'))
        pprint(make_clf_report(
            best_clf, x_te, y_te,
            y_col_name + '_test'))
    ireport = make_interp_report(best_clf, feat_colnames, y_col_name)
    if len(ireport) > 0:
        clf_ireport_path = os.path.join(
            output_path,
            'clf_%d_interpretation.txt' % (y_orig_col_id))
        with open(clf_ireport_path, 'w') as f:
            f.write(ireport)
        if verbose:
            pprint(ireport)

    # Write the classifier obj to disk
    if classifier_name != 'k_nearest_neighbors':
        clf_path = os.path.join(
            output_path,
            'clf_%d_object.dump' % (y_orig_col_id))
        joblib.dump(best_clf, clf_path, compress=1)
        pprint("wrote clf object to file via joblib:")
        pprint(clf_path)

    clf_repr_path = os.path.join(
        output_path,
        'clf_%d_repr.txt' % (y_orig_col_id))
    with open(clf_repr_path, 'w') as f:
        f.write(repr(best_clf) + "\n")
    clf_repr_path = os.path.join(
        output_path,
        'clf_%d_best_param_dict_repr.txt' % (y_orig_col_id))
    with open(clf_repr_path, 'w') as f:
        f.write(repr(best_param_dict) + "\n")

    if verbose:
        pprint("completed clf saving after %11.2f sec" % (time.time() - start_time))

    if os.path.exists(output_path):
        for ss, split in enumerate(['valid', 'test', 'train']):
            csv_fpath = os.path.join(
                output_path,
                'clf_%d_callback_%s.csv' % (y_orig_col_id, split))
            row_dict = make_csv_row_dict(
                best_clf,
                datasets_by_split[split]['x'],
                datasets_by_split[split]['y'][:, y_col_id],
                y_col_name,
                split,
                classifier_name)
            csv_df = pd.DataFrame([row_dict], columns=row_dict.keys())
            csv_df.to_csv(
                csv_fpath,
                index=False)
            if verbose:
                elapsed_time = time.time() - start_time
                pprint("eval %d/%d on %5s split done after %11.2f sec" % (
                    ss + 1, 3, split, elapsed_time))
                pprint("wrote csv file: " + csv_fpath)
    return best_clf, best_param_dict


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf=None, proba_thr_for_class1=0.5, classes=[0,1]):
        """ Make hard predictions at custom-tuned thresholds

        Args
        ----
        clf : sklearn ClassifierMixin
        threshold : float within (0.0, 1.0)
            Provides value at which we call labels of second category
        classes : list
            Provides numeric/string values of class labels

        Examples
        --------
        # Create toy dataset with 80% label=0, 20% label=1
        >>> prng = np.random.RandomState(0)
        >>> x_N = prng.randn(100, 1)
        >>> y_N = np.asarray(prng.rand(100) > 0.8, dtype=np.int32)

        # 'Train' neighbor classifier
        >>> clf = KNeighborsClassifier(n_neighbors=100);
        >>> clf = clf.fit(x_N, y_N)
        >>> clf.classes_
        array([0, 1], dtype=int32)

        # A classifier with 0.5 threshold calls all 0
        >>> thr050 = ThresholdClassifier(clf, 0.5)
        >>> thr050.predict(x_N).min()
        0

        # A classifier with 0.15 threshold calls all 1 
        >>> thr015 = ThresholdClassifier(clf, 0.15)
        >>> thr015.predict(x_N).min()
        1

        # A classifier with 0.95 threshold calls all 0 
        >>> thr015 = ThresholdClassifier(clf, 0.95)
        >>> thr015.predict(x_N).min()
        0
        """
        self.clf = clf
        self.proba_thr_for_class1 = proba_thr_for_class1
        try:
            self.classes_ = clf.classes_
        except AttributeError:
            self.classes_ = classes
        assert len(self.classes_) == 2


    def fit(self, x, y):
        return self.clf.fit(x, y)

    def decision_function(self, x):
        return self.clf.decision_function(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def predict(self, x):
        ''' Make thresholded predictions

        Returns
        -------
        yhat_N : 1D array of class labels
        '''
        yproba_class1_N = self.clf.predict_proba(x)[:,1]
        # Recall that np.where assigns as follows:
        # first value in self.classes when True
        # second value when False
        yhat_N = np.where(yproba_class1_N <= self.proba_thr_for_class1, *self.classes_)
        return yhat_N

    def set_threshold(self, thr):
        self.proba_thr_for_class1 = thr

if __name__ == '__main__':
    read_args_from_stdin_and_run()
