import numpy as np
import scipy as sp
import scipy.sparse as sps
import openml
import random
import pandas as pd
import sys
import os
sys.path.append('../../oboe/automl')
from preprocessing import pre_process

output_directory = 'selected_OpenML_classification_datasets'
selected_datasets = pd.read_csv("selected_OpenML_classification_dataset_indices.csv", index_col=None, header=None).as_matrix().T[0]

i = sys.argv[1]
dataset_id = int(selected_datasets[int(i)])

target_files_path = '{}'.format(output_directory)
if not os.path.exists(target_files_path):
    os.makedirs(target_files_path)

try:
    dataset = openml.datasets.get_dataset(dataset_id)
    data_numeric,data_labels,categorical = dataset.get_data(target=dataset.default_target_attribute, return_categorical_indicator=True)
except:
    directory = '{}/error_datasets_one_hot_encoding'.format(target_files_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.system('touch '+str(directory)+'/'+str(dataset_id)+'.txt')

if sps.issparse(data_numeric):
    data_numeric=data_numeric.todense()
    
    #doing imputation and standardization and not doing one-hot-encoding achieves optimal empirical performances (smallest classification error) on a bunch of OpenML datasets
data_numeric, categorical = pre_process(raw_data=data_numeric, categorical=categorical, impute=False, standardize=True, one_hot_encode=True)
    
    #the output is a preprocessed dataset with all the columns except the last one being preprocessed features, and the last column being labels
data = np.append(data_numeric, np.array(data_labels, ndmin=2).T, axis=1)

pd.DataFrame(data, index=None, columns=None).to_csv(str(target_files_path)+'/dataset_'+str(dataset_id)+'_features_and_labels.csv', header=False, index=False)

print("dataset "+str(dataset_id)+" finished")
