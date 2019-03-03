"""
Test doubling algorithm.
"""

### configurations ###
automl_path = '../../../oboe/automl'
# experimental settings
OUTER_FOLDS = 5
RANDOM_STATE = 0
VERBOSE = False
N_CORES = 1
DATASET_INDICES = runtime_indices

configs_file = 'README.md'
doubling = True
parallel = True
inner_stratified = True
outer_stratified = True

autolearner_kwargs = {
    'verbose': VERBOSE,
    'selection_method': 'min_variance',
    'stacking_alg': 'greedy',
    'n_cores': N_CORES,
}
### end of configurations ###

# import oboe modules
import sys

sys.path.append(automl_path)
from auto_learner import AutoLearner
import util
import mkl
import re
from subprocess import check_output

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

runtime_indices = pd.read_csv(automl_path + '/defaults/runtime_matrix.csv', index_col=0).index.tolist()
mkl.set_num_threads(N_CORES)
kf = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

filename = sys.argv[1]
whole_filename = sys.argv[2]

if int(whole_filename):
    i = os.path.splitext(os.path.basename(filename))[0] #filename of the preprocessed dataset
else:
    i = int(re.findall(r'\d+', filename.split('/')[-1])[0]) #number in filename of the preprocessed dataset
print('Dataset ID: {}'.format(i))

folder = filename.split('/')[-2].split('_')[0]
experiment_start = str(datetime.now())[:10] + '-' + folder

if not os.path.exists(experiment_start):
    try:
        os.makedirs(experiment_start)
    except FileExistsError:
        pass

if not os.path.exists(os.path.join(experiment_start, 'log.txt')):
    with open(os.path.join(experiment_start, 'log.txt'), 'w') as f:
        print('LOG FILE\n', file=f)

s = ['AB', 'GNB', 'ExtraTrees', 'GBT', 'lSVM', 'kSVM', 'RF', 'KNN', 'DT']
default_error_matrix = pd.read_csv(automl_path + '/defaults/error_matrix.csv', index_col=0)
default_runtime_matrix = pd.read_csv(automl_path + '/defaults/runtime_matrix.csv', index_col=0)
columns = [i for i in range(default_error_matrix.shape[1]) if eval(default_error_matrix.columns[i])['algorithm'] in s]
default_error_matrix = default_error_matrix.iloc[:, columns]
default_runtime_matrix = default_runtime_matrix.iloc[:, columns]

if not os.path.exists(os.path.join(experiment_start, configs_file)):
    with open(os.path.join(experiment_start, configs_file), 'a') as log:
        log.write('doubling: {}\n \n'.format(doubling))
        for key in autolearner_kwargs:
            log.write('{}: '.format(key))
            log.write('{}\n \n'.format(autolearner_kwargs[key]))
        log.write('parallel: {}\n\n'.format(parallel))
        log.write('inner stratified: {}\n\n'.format(inner_stratified))
        log.write('outer stratified: {}\n\n'.format(outer_stratified))
        log.write('algorithms = {}'.format(s))

try:
    dataset = pd.read_csv(filename, header=None).values
    if dataset.shape[0] < 150:
        with open(os.path.join(experiment_start, 'log.txt'), 'a') as log:
            print('Dataset {} skipped.'.format(i), file=log)
    else:
        x = dataset[:, :-1]
        y = dataset[:, -1]

        try:
            error_matrix = default_error_matrix.copy().drop(i)
            runtime_matrix = default_runtime_matrix.copy().drop(i)
        except ValueError:
            error_matrix = default_error_matrix.copy()
            runtime_matrix = default_runtime_matrix.copy()

        errors_per_fold = []
        summaries = []
        for train_idx, test_idx in kf.split(x, y):
            x_tr = x[train_idx, :]
            y_tr = y[train_idx]
            x_te = x[test_idx, :]
            y_te = y[test_idx]

            m = AutoLearner('classification', runtime_limit=128,
                            algorithms=s,
                            error_matrix=error_matrix, runtime_matrix=runtime_matrix, **autolearner_kwargs)

            summary = m.fit_doubling(x_tr, y_tr)
            summaries.append(summary)

            errors_per_runtime_limit = []
            for ensemble in summary['models']:
                errors_per_runtime_limit.append(util.error(y_te, ensemble.predict(x_te), 'classification'))

            errors_per_fold.append(errors_per_runtime_limit)
        results = {'cv_error': np.array(errors_per_fold), 'summaries': summaries}

        with open(os.path.join(experiment_start, str(i) + '.pkl'), 'wb') as f:
            pickle.dump(results, f)

        with open(os.path.join(experiment_start, 'log.txt'), 'a') as log:
            print('Dataset {} complete.'.format(i), file=log)

except Exception as e:
    with open(os.path.join(experiment_start, 'errors.txt'), 'a') as err:
        err.write('Dataset {}: {}\n'.format(str(i), str(e)))

    with open(os.path.join(experiment_start, 'log.txt'), 'a') as log:
        print('Dataset {} incomplete.'.format(i), file=log)
