import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import pickle
import pylab
import os
import re
import sys
from tqdm import tqdm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'weight': 'bold'})
rc('text', usetex=True)

sys.path.append('../../oboe/automl')
from util import error

### configurations ###
#whether we plot the results of OpenML or UCI datasets
OPENML = False
#whether we plot the results of best possible base learners within time budget
BEST = True
#whether we plot the results of random selection in oboe
RANDOM = True
#settings of plots
percentiles = [75, 50, 25]
fontsize_axes = 40
legend_marker_size = 40
linewidth = 4
automl_path = '../../oboe/automl'
uci_error_matrix_path = <UCI_ERROR_MATRIX> #to be specified
uci_runtime_matrix_path = <UCI_RUNTIME_MATRIX> #to be specified
### end of configurations ###

if OPENML==False:
    filename='UCI'
else:
    filename='OpenML'

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'weight': 'bold'})
rc('text', usetex=True)

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('oboe', type=str, help='Path to oboe results')
parser.add_argument('autosk', type=str, help='Path to autosklearn results')
parser.add_argument('random', type=str, help='Path to random results')
args = parser.parse_args(sys.argv[1:])

sys.path.insert(1, automl_path)
from model import Model, Ensemble

def find_id(filename):
    return int(re.findall(r'\d+', filename.split('/')[-1])[0])

def find_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def find_name_from_autosklearn_output(filename):
    return os.path.splitext(os.path.basename(filename))[0][8:-12]

def impute_missing(values):
    for i, val in enumerate(values):
        if np.isnan(val):
            if i > 0:
                assert not np.isnan(values[i-1])
                values[i] = values[i-1]
            else:
                values[i] = 0.5           
    return values

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def avg_nested_lists(nested_vals):
    """
    Averages a 2D array and returns a 1D array of all of the columns averaged together, regardless of their dimensions.
    """
    output = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum):    # Go through each index of longest list
        temp = []
        for lst in nested_vals:     # Go through each list
            if index < len(lst):    # If not an index error
                temp.append(lst[index])
        output.append(np.nanmean(temp))
    return np.array(output)

if OPENML:
    lowrank_ids = [find_id(file) for file in os.listdir(args.oboe) if file.endswith('.pkl')]
    autosk_ids = [find_id(file) for file in os.listdir(args.autosk) if file.endswith('.csv')]
    if RANDOM:
        random_ids = [find_id(file) for file in os.listdir(args.random) if file.endswith('.pkl')]
else:
    lowrank_ids = [find_name(file) for file in os.listdir(args.oboe) if file.endswith('.pkl')]
    autosk_ids = [find_name_from_autosklearn_output(file) for file in os.listdir(args.autosk) if file.endswith('.csv')]
    if RANDOM:
        random_ids = [find_name(file) for file in os.listdir(args.random) if file.endswith('.pkl')]

if RANDOM:
    dataset_ids = list(set(lowrank_ids).intersection(set(autosk_ids)).intersection(random_ids))
else:
    dataset_ids = list(set(lowrank_ids).intersection(set(autosk_ids)))
times = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

print('Collecting autosklearn results...')
autosk_results = ()
for id in tqdm(dataset_ids):
    autosk = pd.read_csv(os.path.join(args.autosk, 'dataset_{}_autosklearn.csv'.format(id)))
    autosk_test = impute_missing(autosk[autosk['set_runtime_limit_per_fold'].isin(times)]['average_test_error'].values)
    # some results do not include an entry for large runtimes
    if len(autosk_test) < len(times):
        autosk_test = np.concatenate((autosk_test, [autosk_test[-1]] * (len(times) - len(autosk_test))))
    assert len(autosk_test) == len(times)
    autosk_results += (autosk_test, )
autosk_results = np.vstack(autosk_results)

print('Computing percentiles...')
autosk_percentiles = {}
for p in percentiles:
    results = []
    for i, time in enumerate(times):
        results.append(find_nearest(autosk_results[:, i], np.percentile(autosk_results[:, i], p)))
    autosk_percentiles[p] = results

print('Collecting oboe results...')
oboe_results = ()
for id in tqdm(dataset_ids):
    with open(os.path.join(args.oboe, '{}.pkl'.format(id)), 'rb') as f:
        oboe = pickle.load(f)
    runtime_limits = avg_nested_lists([s['runtime_limits'] for s in oboe['summaries']])
    cv_errors = avg_nested_lists(oboe['cv_error'])
    indices = np.in1d(runtime_limits, times)
    oboe_test = cv_errors[indices]
    if len(oboe_test) == 0:
        print(id)
    # some results do not include an entry for large runtimes
    if len(oboe_test) < len(times):
        oboe_test = np.concatenate((oboe_test, [oboe_test[-1]] * (len(times) - len(oboe_test))))
    assert len(oboe_test) == len(times)
    oboe_results += (oboe_test,)
oboe_results = np.vstack(oboe_results)

print('Computing percentiles...')
oboe_percentiles = {}
for p in percentiles:
    results = []
    for i, time in enumerate(times):
        results.append(find_nearest(oboe_results[:, i], np.percentile(oboe_results[:, i], p)))
    oboe_percentiles[p] = results

if RANDOM:
    try:
        print('Collecting random results...')
        random_results = ()
        for id in tqdm(dataset_ids):
            with open(os.path.join(args.random, '{}.pkl'.format(id)), 'rb') as f:
                random = pickle.load(f)
            runtime_limits = avg_nested_lists([s['runtime_limits'] for s in random['summaries']])
            cv_errors = avg_nested_lists(random['cv_error'])
            indices = np.in1d(runtime_limits, times)
            random_test = cv_errors[indices]
            if len(random_test) == 0:
                print(id)
            # some results do not include an entry for large runtimes
            elif len(random_test) < len(times):
                random_test = np.concatenate((random_test, [random_test[-1]] * (len(times) - len(random_test))))
            assert len(random_test) == len(times)
            random_results += (random_test,)
        random_results = np.vstack(random_results)

        print('Computing percentiles...')
        random_percentiles = {}
        for p in percentiles:
            results = []
            for i, time in enumerate(times):
                results.append(find_nearest(random_results[:, i], np.percentile(random_results[:, i], p)))
            random_percentiles[p] = results
    except:
        pass
    
if BEST:
    if OPENML:
        try:
            print('Collecting best results in error matrix...')
            default_error_matrix = pd.read_csv(os.path.join(automl_path, 'defaults/error_matrix.csv'), index_col=0, header=0)
            default_runtime_matrix = pd.read_csv(os.path.join(automl_path, 'defaults/runtime_matrix.csv'), index_col=0, header=0)
            best_results = ()
            for id in tqdm(dataset_ids):
                try:
                    best = []
                    for time in times:
                        whether_able_to_finish = np.array(default_runtime_matrix.loc[int(id)]) <= time
                        best.append(min(np.array(default_error_matrix.loc[int(id)])[whether_able_to_finish]))
                    best = np.array(best)
                    best_results += (best,)
                except:
                    pass
            best_results = np.vstack(best_results)


            print('Computing percentiles...')
            best_percentiles = {}
            for p in percentiles:
                results = []
                for i, time in enumerate(times):
                    results.append(find_nearest(best_results[:, i], np.percentile(best_results[:, i], p)))
                best_percentiles[p] = results
        except:
            print('No dataset in error matrix.')
    else:
        try:
            print('Collecting best results in error matrix...')
            default_error_matrix = pd.read_csv(uci_error_matrix_path, index_col=0, header=0)
            default_runtime_matrix = pd.read_csv(uci_runtime_matrix_path, index_col=0, header=0)
            best_results = ()
            for id in tqdm(range(len(default_error_matrix.index))):
                try:
                    best = []
                    for time in times:
                        whether_able_to_finish = np.array(default_runtime_matrix.iloc[id]) <= time
                        best.append(min(np.array(default_error_matrix.iloc[id])[whether_able_to_finish]))
                    best = np.array(best)
                    best_results += (best,)
                except:
                    pass
            best_results = np.vstack(best_results)


            print('Computing percentiles...')
            best_percentiles = {}
            for p in percentiles:
                results = []
                for i, time in enumerate(times):
                    results.append(find_nearest(best_results[:, i], np.percentile(best_results[:, i], p)))
                best_percentiles[p] = results
        except:
            print('No dataset in error matrix.')        

print('Generating image...')
fig, ax = plt.subplots(figsize=(18, 9))
colors = ['red', 'blue', 'green']
assert len(colors) == len(percentiles), 'Not enough colors.'
for i, p in enumerate(percentiles):
    ax.step(np.log2(times), oboe_percentiles[p], color=colors[i], linewidth=linewidth, where='post', label='oboe {}th percentile'.format(p))
    ax.step(np.log2(times), autosk_percentiles[p], color=colors[i], linewidth=linewidth, where='post', linestyle='dashdot', label='autosklearn {}th percentile'.format(p))
    if BEST:
        try:
            ax.step(np.log2(times), best_percentiles[p], color=colors[i], linewidth=linewidth, where='post', linestyle='dotted', label='best entry {}th percentile'.format(p))
        except:
            pass
        
    if RANDOM:
        try:
            ax.step(np.log2(times), random_percentiles[p], color=colors[i], linewidth=linewidth, where='post', linestyle='dashed', label='random entry {}th percentile'.format(p))
        except:
            pass
        
errors = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_xticklabels(times, fontsize=fontsize_axes)
ax.set_xlabel('Runtime budget (s)', fontsize=fontsize_axes)
ax.set_ylabel('Balanced error rate', fontsize=fontsize_axes)
ax.legend(loc='lower right', fontsize=legend_marker_size, bbox_to_anchor=(1.7, -0.05))
plt.yticks(fontsize=fontsize_axes)

plt.savefig(os.path.join(args.oboe, '{}.pdf'.format(filename)), pad_inches=0.1, bbox_inches='tight')
