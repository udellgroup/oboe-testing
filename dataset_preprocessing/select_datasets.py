"""
Select OpenML datasets with conditions.
"""

import numpy as np
import scipy as sp
import scipy.sparse as sps
import openml
import random
import pandas as pd
import sys
import os

#The condition for selecting OpenML datasets
condition = "(openml_datasets.NumberOfInstances < 10000) & (openml_datasets.NumberOfInstances >= 150) & (openml_datasets.NumberOfMissingValues == 0) & (openml_datasets.NumberOfClasses < 0)"
#This should be the filename without extension
filename = 'selected_OpenML_classification_dataset_indices'

openml_datasets = openml.datasets.list_datasets()
openml_datasets = pd.DataFrame.from_dict(openml_datasets, orient='index')
openml_datasets_selected = openml_datasets[eval(condition)]

pd.DataFrame(openml_datasets_selected.index, columns=None, index=None).to_csv('{}.csv'.format(filename), header=False, index=False)

with open('{}.txt'.format(filename), 'a') as log:
    log.write('condition for dataset selection: {}'.format(condition))

