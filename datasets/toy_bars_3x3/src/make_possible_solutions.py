import argparse
import numpy as np
import sklearn.linear_model
import sys
import os

from sklearn.externals import joblib

from pc_toolbox.model_slda import (
    slda_loss__autograd,
    slda_utils__dataset_manager)

V = 9
b1 = np.asarray([
    [.00, .00, .00],
    [.33, .33, .33],
    [.00, .00, .00],
    ])
b2 = np.asarray([
    [.00, .00, .00],
    [.00, .00, .00],
    [.33, .33, .33],
    ])
b3 = np.asarray([
    [.00, .33, .00],
    [.00, .33, .00],
    [.00, .33, .00],
    ])
b4 = np.asarray([
    [.00, .00, .33],
    [.00, .00, .33],
    [.00, .00, .33],
    ])
bY = np.asarray([
    [.99, .00, .00],
    [.00, .00, .00],
    [.00, .00, .00],
    ])

def make_one_hot_topic(hot_word_id):
    bY = np.zeros((1,9))
    bY[0, hot_word_id] = 0.99
    return bY

# Reshape to 1 x V
for arr_name in ['b1', 'b2', 'b3', 'b4', 'bY']:
    arr = locals()[arr_name]
    assert np.allclose(0.99, np.sum(arr))
    locals()[arr_name] = np.reshape(arr, (1,9))

topics_KV_by_name = {
    'good_loss_x_K4':
        np.vstack([b3, b4, b1, b2]),
    'good_loss_pc_K4':
        np.vstack([bY, b3 + b4, b1, b2]),
    'good_loss_label_rep_K4':
        np.vstack([b3 + b4, b3 + b4, b1 + b2, b1 + b2]),
    'good_loss_y_K4':
        np.vstack([
            make_one_hot_topic(0),
            make_one_hot_topic(1),
            make_one_hot_topic(3),
            make_one_hot_topic(8)]),
    }

for arr in topics_KV_by_name.values():
    # Start each topic with mass ~1.0
    arr /= arr.sum(axis=1)[:,np.newaxis]
    # Add small extra mass to each vocab term
    arr += .001
    # Normalize so sums to one
    arr /= arr.sum(axis=1)[:,np.newaxis]

np.set_printoptions(linewidth=120, precision=4, suppress=1)
for key in topics_KV_by_name:
    print key
    print topics_KV_by_name[key]


w_CK_by_name = {
    'good_loss_x_K4':
        np.asarray([[-01.0, -01.0, -01.0, -01.0]]),
    'good_loss_pc_K4':
        np.asarray([[+40.0, -02.0, -02.0, -02.0]]),
    'good_loss_label_rep_K4':
        np.asarray([[+10.0, -10.0, +10.0, -10.0]]),
    'good_loss_y_K4':
        np.asarray([[+30.0, -3.0, -3.0, -3.0]]),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.path.abspath('.'), type=str)
    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset_path)


    dataset = slda_utils__dataset_manager.load_dataset(dataset_path, 'train')
    keys = w_CK_by_name.keys()
    Cs_grid = np.logspace(-5, 5, 11)
    best_pos = np.flatnonzero(1.0/Cs_grid == 0.001)[0]
    prior_Cs_grid = 0.01 * np.exp(-1.0 * (best_pos - np.arange(11))**2 / 50.0)

    print ""
    print "==== FINE TUNING WEIGHT VECTORS"
    pi_estimation_weight_y = 0.0
    pi_estimation_mode = 'missing_y'
    nef_alpha = 1.1
    tau = 1.1
    lambda_w = 0.001

    for key in keys:
        topics_KV = topics_KV_by_name[key]
        w_CK = w_CK_by_name[key]

        # Perform loss calculation (also delivers pi_DK)
        loss_dict = slda_loss__autograd.calc_loss__slda(
            dataset=dataset,
            topics_KV=topics_KV,
            w_CK=w_CK,
            weight_x=1.0,
            weight_y=1.0,
            pi_estimation_mode=pi_estimation_mode,
            pi_estimation_weight_y=pi_estimation_weight_y,
            nef_alpha=nef_alpha,
            tau=tau,
            lambda_w=lambda_w,
            return_dict=True)

        print ""
        print "======", key
        print loss_dict['summary_msg']

        # Fit logistic regression model via cross-validation
        feat_DK = loss_dict['pi_DK']
        y_D = dataset['y_DC'][:,0]
        cv_clf = sklearn.linear_model.LogisticRegressionCV(
            fit_intercept=False,
            Cs=Cs_grid,
            cv=3, # num folds
            random_state=np.random.RandomState(42))
        cv_clf.fit(feat_DK, y_D)

        acc_per_Cval = (
            np.median(cv_clf.scores_.values()[0], axis=0) 
            + prior_Cs_grid)
        best_p = np.argmax(acc_per_Cval)
        best_C = Cs_grid[best_p]
        print "## best lambda_w"
        print 0.5 / best_C

        clf_with_best_C = sklearn.linear_model.LogisticRegression(
            fit_intercept=False, C=best_C)
        clf_with_best_C.fit(feat_DK, y_D)

        print "## best w_CK:"
        print clf_with_best_C.coef_
        w_CK_by_name[key] = clf_with_best_C.coef_


    print ""
    print "==== REPORTING RESULTS WITH FIXED TOPICS AND FINE-TUNED WEIGHTS"
    for pi_estimation_mode in ['missing_y']: #, 'observe_y']:
        print ""
        print '---- pi_estimation_mode =', pi_estimation_mode
        for key in keys:
            topics_KV = topics_KV_by_name[key]
            w_CK = w_CK_by_name[key]
            loss_dict = slda_loss__autograd.calc_loss__slda(
                dataset=dataset,
                topics_KV=topics_KV,
                w_CK=w_CK,
                weight_x=1.0,
                weight_y=1.0,
                pi_estimation_mode=pi_estimation_mode,
                pi_estimation_weight_y=pi_estimation_weight_y,
                nef_alpha=nef_alpha,
                tau=tau,
                lambda_w=lambda_w,
                return_dict=True)
            print "%-25s uloss_x__pertok %.4f\n%-25s uloss_y__perdoc %.4f\n" % (
                key, loss_dict['uloss_x__pertok'],
                '', loss_dict['uloss_y__perdoc'])



    print ""
    print "==== SAVING PARAMS PACKAGED UP AS _param_dict.dump"
    for key in keys:
        fpath = os.path.join(dataset_path, '%s_param_dict.dump' % (key))
        GP = dict(
            topics_KV=topics_KV_by_name[key],
            w_CK=w_CK_by_name[key],
            n_labels=1,
            n_states=4,
            n_vocabs=9)
        joblib.dump(
            GP,
            fpath,
            compress=1)
