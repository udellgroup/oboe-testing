from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold, StratifiedKFold
from subprocess import check_output

import sys
import os
automl_path = '../../../oboe/automl'
sys.path.append(automl_path)
import util
import glob
import re
from datetime import datetime

f = sys.argv[1]
dirname = sys.argv[2]
whole_filename = sys.argv[3]

num_outer_folds = 5
RANDOM_STATE = 0
runtime_limit_range = np.array([1, 2, 4, 8, 16, 32, 64])
runtime_indices = pd.read_csv(str(automl_path)+'/defaults/runtime_matrix.csv', index_col=0).index.tolist()
log_file = 'finished.txt'
colname = ['set_runtime_limit_per_fold', 'average_training_error', 'average_test_error', 'actual_runtime_per_fold'] + ['training_error_fold_{}'.format(i+1) for i in range(num_outer_folds)] + ['test_error_fold_{}'.format(i+1) for i in range(num_outer_folds)]

def AutoSklearn(total_runtime, train_features, train_labels):
    clf = AutoSklearnClassifier(
            time_left_for_this_task=total_runtime,
            include_preprocessors=["no_preprocessing"],
            include_estimators = ["adaboost","gaussian_nb", "extra_trees", "gradient_boosting", "liblinear_svc", "libsvm_svc","random_forest",
                 "k_nearest_neighbors","decision_tree"],
    )
        
    clf.fit(train_features, train_labels, metric = balanced_accuracy)    
    return clf

def kfolderror(x, y, runtime, num_outer_folds=num_outer_folds):
    training_error = []
    test_error = []
    time_elapsed = []
    kf = StratifiedKFold(num_outer_folds, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in kf.split(x, y):
        x_tr = x[train_idx, :]
        y_tr = y[train_idx]
        x_te = x[test_idx, :]
        y_te = y[test_idx]
        
        start = time.time()
        clf = AutoSklearn(runtime, x_tr, y_tr)
        clf.refit(x_tr, y_tr)
        time_elapsed.append(time.time() - start)
        y_tr_pred = clf.predict(x_tr)
        y_pred = clf.predict(x_te)
        training_error.append(util.error(y_tr, y_tr_pred, 'classification'))
        test_error.append(util.error(y_te, y_pred, 'classification'))
    
    time_elapsed = np.array(time_elapsed).mean()
    training_error = np.array(training_error)
    test_error = np.array(test_error)
    return training_error, test_error, time_elapsed

if int(whole_filename):
    i = os.path.splitext(os.path.basename(filename))[0]
else:
    i = re.findall(r'\d+', os.path.basename(f))[0]
    
dataset = pd.read_csv(f, header=None).values
print("Run Autosklearn on dataset {}".format(i))
x = dataset[:, :-1]
y = dataset[:, -1]

results = np.full((len(runtime_limit_range), len(colname)), np.nan)

for idx, runtime_limit in enumerate(runtime_limit_range):
    print("Dataset No.{} with pre-set runtime limit {}".format(i, runtime_limit))
    try:
        training_error, test_error, time_elapsed = kfolderror(x, y, int(runtime_limit))
        average_training_error = training_error.mean()
        average_test_error = test_error.mean()
    except:
        training_error = np.full(num_outer_folds, np.nan)
        test_error = np.full(num_outer_folds, np.nan)
        time_elapsed = np.nan
        average_training_error = np.nan
        average_test_error = np.nan
        pass
    results[idx, :] = np.concatenate((np.array([runtime_limit, average_training_error, average_test_error, time_elapsed]), training_error, test_error))
    pd.DataFrame(results, columns=colname).to_csv('{}/dataset_{}_autosklearn.csv'.format(dirname, i), index=False)

with open(os.path.join(dirname, log_file), 'a') as log:
    log.write('dataset No.{} finished \n \n'.format(i))

