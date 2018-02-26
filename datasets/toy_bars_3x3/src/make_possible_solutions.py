import numpy as np
import sklearn.linear_model
import sys
import os

from sklearn.externals import joblib

sys.path.append(
    os.path.expandvars("$SSCAPEROOT/sscape/models_lda"))
import slda__fastloss
import bow_dataset

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
    bY = np.zeros(9)
    bY[hot_word_id] = 0.99
    return bY

# Reshape to 1 x V
for arr_name in ['b1', 'b2', 'b3', 'b4', 'bY']:
    arr = locals()[arr_name]
    locals()[arr_name] = np.reshape(arr, (1,9))

topics_KV_by_name = {
    'good_loss_x':
        np.vstack([b3, b4, b1, b2]),
    'good_loss_pc':
        np.vstack([bY, b3 + b4, b1, b2]),
    'good_loss_label_rep':
        np.vstack([b3 + b4, b3 + b4, b1 + b2, b1 + b2]),
    'good_loss_yy':
        np.vstack([
            make_one_hot_topic(0),
            make_one_hot_topic(1),
            make_one_hot_topic(3),
            make_one_hot_topic(8)]),
    }

# Add smoothing
for arr in topics_KV_by_name.values():
    # TODO: might be better to add *same* smoothing each one
    # but this is pretty minor
    arr += .01 / float(V) * np.sum(arr)
    arr /= arr.sum(axis=1)[:,np.newaxis]

w_CK_by_name = {
    'good_loss_x':
        np.asarray([[-01.0, -01.0, -01.0, -01.0]]),
    'good_loss_pc':
        np.asarray([[+40.0, -02.0, -02.0, -02.0]]),
    'good_loss_label_rep':
        np.asarray([[+10.0, -10.0, +10.0, -10.0]]),
    'good_loss_yy':
        np.asarray([[+30.0, -3.0, -3.0, -3.0]]),
    }

dataset = bow_dataset.load_dataset('.', 'train')['dataset']

keys = w_CK_by_name.keys()
Cs_grid = np.logspace(-5, 5, 11)

best_pos = np.flatnonzero(1.0/Cs_grid == 0.001)[0]
prior_Cs_grid = 0.01 * np.exp(-1.0 * (best_pos - np.arange(11))**2 / 50.0)

print ""
print "==== FINE TUNING WEIGHT VECTORS FOR SUPERVISION"
pi_estimation_weight_y = 1.0
for key in keys:
    if key.count('label_rep'):
        pi_estimation_mode = 'observe_y'
    else:
        pi_estimation_mode = 'missing_y'
    topics_KV = topics_KV_by_name[key]
    w_CK = w_CK_by_name[key]
    loss_dict = slda__fastloss.calc_neg_log_proba__slda(
        dataset=dataset,
        topics_KV=topics_KV,
        w_CK=w_CK,
        weight_x=1.0,
        weight_y=1.0,
        pi_estimation_mode=pi_estimation_mode,
        pi_estimation_weight_y=pi_estimation_weight_y,
        nef_alpha=1.1,
        tau=1.1,
        lambda_w=0.001,
        return_dict=True)

    print ""
    print "======", key, pi_estimation_mode
    print loss_dict['summary_msg']

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
    print 1.0 / best_C

    clf_with_best_C = sklearn.linear_model.LogisticRegression(
        fit_intercept=False, C=best_C)
    clf_with_best_C.fit(feat_DK, y_D)

    print "## best w_CK:"
    print clf_with_best_C.coef_
    w_CK_by_name[key] = clf_with_best_C.coef_


print ""
print "==== REPORTING RESULTS WITH FIXED TOPICS AND FINE-TUNED WEIGHTS"
for pi_estimation_mode in ['missing_y', 'observe_y']:
    print ""
    print '----', pi_estimation_mode
    for key in keys:
        topics_KV = topics_KV_by_name[key]
        w_CK = w_CK_by_name[key]
        loss_dict = slda__fastloss.calc_neg_log_proba__slda(
            dataset=dataset,
            topics_KV=topics_KV,
            w_CK=w_CK,
            weight_x=1.0,
            weight_y=1.0,
            pi_estimation_mode=pi_estimation_mode,
            pi_estimation_weight_y=pi_estimation_weight_y,
            nef_alpha=1.1,
            tau=1.1,
            lambda_w=0.001,
            return_dict=True)
        print "loss_x %.4f loss_y  %.4f    %-20s" % (
            loss_dict['loss_x'], loss_dict['loss_y'], key)


print ""
print "==== SAVING PARAMS PACKAGED UP AS _param_dict.dump"
for key in keys:
    fpath = '%s_param_dict.dump' % (key)
    GP = dict(
        topics_KV=topics_KV_by_name[key], w_CK=w_CK_by_name[key],
        n_states=4,
        n_vocabs=9)
    joblib.dump(
        GP,
        fpath,
        compress=1)
